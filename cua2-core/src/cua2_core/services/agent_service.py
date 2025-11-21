import asyncio
import base64
import fcntl
import json
import logging
import os
import time
from io import BytesIO
from typing import IO, Callable, Literal
from uuid import uuid4

from cua2_core.models.models import (
    ActiveTask,
    AgentAction,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
)
from cua2_core.services.agent_utils.desktop_agent import E2BVisionAgent
from cua2_core.services.agent_utils.function_parser import parse_function_call
from cua2_core.services.agent_utils.get_model import get_model
from cua2_core.services.archival_service import ArchivalService
from cua2_core.services.sandbox_service import SandboxService
from cua2_core.services.utils import compress_image_to_max_size
from cua2_core.websocket.websocket_manager import WebSocketException, WebSocketManager
from e2b_desktop import Sandbox, TimeoutException
from fastapi import WebSocket
from PIL import Image
from smolagents import ActionStep, AgentMaxStepsError, TaskStep
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class AgentStopException(Exception):
    """Exception for agent stop"""

    pass


class AgentService:
    """Service for handling agent tasks and processing"""

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        sandbox_service: SandboxService,
        max_sandboxes: int,
    ):
        self.active_tasks: dict[str, ActiveTask] = {}
        self.websocket_manager: WebSocketManager = websocket_manager
        self.task_websockets: dict[str, WebSocket] = {}
        self.sandbox_service: SandboxService = sandbox_service
        self.last_screenshot: dict[str, tuple[Image.Image, str] | None] = {}
        self._lock = asyncio.Lock()
        self.max_sandboxes = max_sandboxes
        self._archival_lock_file: IO[str] | None = None

        # Initialize archival service in dedicated process
        self.archival_service = ArchivalService(
            hf_token=os.getenv("HF_TOKEN"),
            hf_dataset_repo="smolagents/cua_traces",
            data_dir="data",
            archive_interval_minutes=30,
            folder_age_threshold_minutes=30,
        )
        # Start the archival service process only on one worker
        if self._should_start_archival_service():
            self.archival_service.start()
            logger.info(f"Started archival service in worker PID {os.getpid()}")
        else:
            logger.info(
                f"Skipping archival service start in worker PID {os.getpid()} (already running in another worker)"
            )

    def _should_start_archival_service(self) -> bool:
        """
        Determine if this worker should start the archival service.
        Uses file-based locking to ensure only one worker across all processes
        starts the archival service.

        Returns:
            True if this worker should start the archival service, False otherwise
        """
        lock_file_path = "/tmp/cua2_archival_service.lock"

        try:
            self._archival_lock_file = open(lock_file_path, "w")
            fcntl.flock(
                self._archival_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )

            self._archival_lock_file.write(str(os.getpid()))
            self._archival_lock_file.flush()
            return True

        except (IOError, OSError):
            if self._archival_lock_file:
                self._archival_lock_file.close()
                self._archival_lock_file = None
            return False

    def _update_archival_active_tasks(self):
        """
        Update the archival service with current active task IDs.
        Should be called whenever tasks are added or removed.
        """
        if self.archival_service.is_alive():
            self.archival_service.update_active_tasks(set(self.active_tasks.keys()))

    async def create_id_and_sandbox(self, websocket: WebSocket) -> str:
        """Create a new ID and sandbox"""
        async with self._lock:
            uuid = str(uuid4())
            while uuid in self.active_tasks:
                uuid = str(uuid4())
            self.task_websockets[uuid] = websocket
        await self.sandbox_service.acquire_sandbox(uuid)
        return uuid

    async def process_user_task(
        self, trace: AgentTrace, websocket: WebSocket
    ) -> str | None:
        """Process a user task and return the trace ID"""

        trace_id = trace.id
        trace.steps = []
        trace.traceMetadata = AgentTraceMetadata(traceId=trace_id)

        async with self._lock:
            if self.task_websockets[trace_id] != websocket:
                raise WebSocketException("WebSocket mismatch")

            active_task = ActiveTask(
                message_id=trace_id,
                instruction=trace.instruction,
                model_id=trace.modelId,
                timestamp=trace.timestamp,
                steps=trace.steps,
                traceMetadata=trace.traceMetadata,
            )

            if len(self.active_tasks) >= self.max_sandboxes:
                await self.websocket_manager.send_agent_start(
                    active_task=active_task,
                    status="max_sandboxes_reached",
                    websocket=websocket,
                )
                return trace_id

            # Store the task and websocket for this task
            self.active_tasks[trace_id] = active_task
            self.last_screenshot[trace_id] = None

        # Update archival service with new active task
        self._update_archival_active_tasks()

        asyncio.create_task(self._agent_processing(trace_id))

        return trace_id

    async def _agent_runner(
        self,
        message_id: str,
        step_callback: Callable[[ActionStep, E2BVisionAgent], None],
    ):
        """Run the task with the appropriate agent"""

        sandbox: Sandbox | None = None
        agent = None
        novnc_active = False
        websocket_exception = False
        final_state = "success"

        try:
            # Get the websocket for this task
            websocket = self.task_websockets.get(message_id)

            await self.websocket_manager.send_agent_start(
                active_task=self.active_tasks[message_id],
                websocket=websocket,
                status="success",
            )

            model = get_model(self.active_tasks[message_id].model_id)

            max_attempts = 20
            for _ in range(max_attempts):
                response = await self.sandbox_service.acquire_sandbox(message_id)
                if response.sandbox is not None and response.state == "ready":
                    sandbox = response.sandbox
                    break
                elif response.state == "max_sandboxes_reached":
                    raise Exception("No sandbox available: pool limit reached")
                await asyncio.sleep(2)
            if sandbox is None:
                raise Exception("No sandbox available: pool limit reached")

            data_dir = self.active_tasks[message_id].trace_path
            user_content = self.active_tasks[message_id].instruction

            agent = E2BVisionAgent(
                model=model,
                data_dir=data_dir,
                desktop=sandbox,
                step_callbacks=[step_callback],
            )

            self.active_tasks[message_id].traceMetadata.maxSteps = agent.max_steps

            await self.websocket_manager.send_vnc_url_set(
                vnc_url=sandbox.stream.get_url(
                    auto_connect=True,
                    view_only=True,
                    resize="scale",
                    auth_key=sandbox.stream.get_auth_key(),
                )
                or "",
                websocket=websocket,
            )
            novnc_active = True

            step_filename = f"{message_id}-1"
            screenshot_bytes = agent.desktop.screenshot()
            image = Image.open(BytesIO(screenshot_bytes))
            self.last_screenshot[message_id] = (image, step_filename)

            await asyncio.to_thread(
                agent.run,
                user_content,
            )

            self.active_tasks[message_id].traceMetadata.completed = True

        except AgentStopException as e:
            if str(e) == "Max steps reached":
                final_state = "max_steps_reached"
            elif str(e) == "Task not completed":
                final_state = "stopped"

        except WebSocketException:
            websocket_exception = True

        except TimeoutException:
            final_state = "sandbox_timeout"

        except (Exception, KeyboardInterrupt):
            import traceback

            logger.error(
                f"Error processing task: {traceback.format_exc()}", exc_info=True
            )
            final_state = "error"
            await self.websocket_manager.send_agent_error(
                error="Error processing task", websocket=websocket
            )

        finally:
            # Send completion event
            # Check if websocket is still connected before sending
            if (
                not websocket_exception
                and websocket
                and websocket.client_state == WebSocketState.CONNECTED
            ):
                await self.websocket_manager.send_agent_complete(
                    metadata=self.active_tasks[message_id].traceMetadata,
                    websocket=websocket,
                    final_state=final_state,
                )

                if novnc_active:
                    await self.websocket_manager.send_vnc_url_unset(websocket=websocket)

            novnc_active = False

            self.active_tasks[message_id].update_trace_metadata(
                final_state=final_state,
                completed=True,
            )

            if message_id in self.active_tasks:
                self.active_tasks[message_id].store_model()

            # Clean up
            async with self._lock:
                if message_id in self.active_tasks:
                    del self.active_tasks[message_id]

                if message_id in self.task_websockets:
                    del self.task_websockets[message_id]

                if message_id in self.last_screenshot:
                    del self.last_screenshot[message_id]

            # Update archival service after task removal
            self._update_archival_active_tasks()

            # Always release sandbox back to the pool, even if it's still in "creating" state
            # This handles cases where acquire_sandbox was called but sandbox never became ready
            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as e:
                logger.error(
                    f"Error releasing sandbox for {message_id}: {e}", exc_info=True
                )

    async def _agent_processing(
        self,
        message_id: str,
    ):
        """Process the user task with the appropriate agent"""
        try:
            # Set up log file for this task
            active_task = self.active_tasks[message_id]

            # Ensure the directory exists
            os.makedirs(active_task.trace_path, exist_ok=True)

            # Capture the event loop reference in the async context
            # This will be used in the callback to safely schedule coroutines from the worker thread
            loop = asyncio.get_running_loop()

            def step_callback(memory_step: ActionStep, agent: E2BVisionAgent):
                assert memory_step.step_number is not None

                if memory_step.step_number > agent.max_steps:
                    raise AgentStopException("Max steps reached")

                if self.active_tasks[message_id].traceMetadata.completed:
                    raise AgentStopException("Task not completed")

                model_output = (
                    memory_step.model_output_message.content
                    if memory_step.model_output_message
                    else None
                )
                if isinstance(memory_step.error, AgentMaxStepsError):
                    model_output = memory_step.action_output

                thought = (
                    model_output.split("```")[0].replace("\nAction:\n", "")
                    if model_output
                    and (
                        memory_step.error is None
                        or isinstance(memory_step.error, AgentMaxStepsError)
                    )
                    else None
                )

                if model_output is not None:
                    action_sequence = model_output.split("```")[1]
                else:
                    action_sequence = """The task failed due to an error"""  # TODO: To Handle in front

                agent_actions = (
                    AgentAction.from_function_calls(
                        parse_function_call(action_sequence)
                    )
                    if action_sequence
                    else None
                )

                # if not (
                #     agent_actions is not None
                #     and not any(action.function_name == "wait" for action in agent_actions)
                # ):
                time.sleep(3)

                image, step_filename = self.last_screenshot[message_id]  # type: ignore
                assert image is not None and step_filename is not None
                screenshot_path = os.path.join(agent.data_dir, f"{step_filename}.png")
                image.save(screenshot_path)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                del buffered
                del image

                if memory_step.token_usage is not None:
                    step = AgentStep(
                        traceId=message_id,
                        stepId=str(memory_step.step_number),
                        image=image_base64,
                        thought=thought,
                        actions=agent_actions,
                        error=memory_step.error.message if memory_step.error else None,
                        duration=memory_step.timing.duration,
                        inputTokensUsed=memory_step.token_usage.input_tokens,
                        outputTokensUsed=memory_step.token_usage.output_tokens,
                        step_evaluation="neutral",
                    )

                    self.active_tasks[message_id].update_trace_metadata(
                        step_input_tokens_used=memory_step.token_usage.input_tokens,
                        step_output_tokens_used=memory_step.token_usage.output_tokens,
                        step_duration=memory_step.timing.duration,
                        step_numberOfSteps=1,
                    )

                    self.active_tasks[message_id].update_step(step)

                    websocket = self.task_websockets.get(message_id)
                    if websocket and websocket.client_state == WebSocketState.CONNECTED:
                        future = asyncio.run_coroutine_threadsafe(
                            self.websocket_manager.send_agent_progress(
                                step=step,
                                metadata=self.active_tasks[message_id].traceMetadata,
                                websocket=websocket,
                            ),
                            loop,
                        )
                        future.result()

                if self.active_tasks[message_id].traceMetadata.completed:
                    raise AgentStopException("Task not completed")

                step_filename = f"{message_id}-{memory_step.step_number + 1}"
                screenshot_bytes = agent.desktop.screenshot()
                original_image = Image.open(BytesIO(screenshot_bytes))
                image = compress_image_to_max_size(original_image, max_size_kb=500)
                del original_image

                for previous_memory_step in (
                    agent.memory.steps
                ):  # Remove previous screenshots from logs for lean processing
                    if isinstance(previous_memory_step, ActionStep):
                        previous_memory_step.observations_images = None
                    elif isinstance(previous_memory_step, TaskStep):
                        previous_memory_step.task_images = None

                memory_step.observations_images = [image.copy()]

                del self.last_screenshot[message_id]
                self.last_screenshot[message_id] = (image, step_filename)

            await self._agent_runner(message_id, step_callback)
        except Exception as e:
            # If _agent_processing fails before _agent_runner is called,
            # we still need to release the sandbox that was acquired in create_id_and_sandbox
            logger.error(
                f"Error in _agent_processing for {message_id}: {e}", exc_info=True
            )
            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as release_error:
                logger.error(
                    f"Error releasing sandbox in _agent_processing cleanup for {message_id}: {release_error}",
                    exc_info=True,
                )
            # Re-raise to ensure error is logged
            raise

    def update_trace_step(
        self,
        trace_id: str,
        step_id: str,
        step_evaluation: Literal["like", "dislike", "neutral"],
    ):
        """
        Update a specific step in a trace (e.g., update step evaluation)

        Args:
            trace_id: The trace ID
            step_id: The step ID (1-indexed)
            step_evaluation: The evaluation value to set

        Returns:
            The updated AgentStep

        Raises:
            ValueError: If step_id is invalid or step not found
            FileNotFoundError: If trace not found
        """
        # Try to find in active tasks first
        active_task = self.active_tasks.get(trace_id)

        if active_task:
            # Task is still active
            try:
                step_index = int(step_id) - 1
                if 0 <= step_index < len(active_task.steps):
                    active_task.steps[step_index].step_evaluation = step_evaluation
                    active_task.update_step(active_task.steps[step_index])
                else:
                    raise ValueError(f"Step {step_id} not found in trace")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid step_id format: {e}")
        else:
            # Task is not active, try to load from file
            data_dir = "data"
            trace_dirs = [
                d for d in os.listdir(data_dir) if d.startswith(f"trace-{trace_id}")
            ]

            if not trace_dirs:
                raise FileNotFoundError("Trace not found")

            trace_path = os.path.join(data_dir, trace_dirs[0])
            tasks_file = os.path.join(trace_path, "tasks.json")

            if not os.path.exists(tasks_file):
                raise FileNotFoundError("Trace data not found")

            try:
                # Load the trace data
                with open(tasks_file, "r") as f:
                    task_data = json.load(f)

                # Find and update the step
                step_index = int(step_id) - 1
                if 0 <= step_index < len(task_data["steps"]):
                    task_data["steps"][step_index]["step_evaluation"] = step_evaluation

                    # Save the updated data
                    with open(tasks_file, "w") as f:
                        json.dump(task_data, f, indent=2)

                    # Convert to AgentStep for response
                    updated_step = AgentStep(**task_data["steps"][step_index])
                    return updated_step
                else:
                    raise ValueError(f"Step {step_id} not found in trace")
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"Error processing step update: {e}")

    def update_trace_evaluation(
        self,
        trace_id: str,
        user_evaluation: Literal["success", "failed", "not_evaluated"],
    ):
        """
        Update the user evaluation for a trace

        Args:
            trace_id: The trace ID
            user_evaluation: The evaluation value to set

        Raises:
            FileNotFoundError: If trace not found
        """
        # Try to find in active tasks first
        active_task = self.active_tasks.get(trace_id)

        if active_task:
            # Task is still active
            active_task.update_trace_metadata(user_evaluation=user_evaluation)
        else:
            # Task is not active, try to load from file
            data_dir = "data"
            trace_dirs = [
                d for d in os.listdir(data_dir) if d.startswith(f"trace-{trace_id}")
            ]

            if not trace_dirs:
                raise FileNotFoundError("Trace not found")

            trace_path = os.path.join(data_dir, trace_dirs[0])
            tasks_file = os.path.join(trace_path, "tasks.json")

            if not os.path.exists(tasks_file):
                raise FileNotFoundError("Trace data not found")

            try:
                # Load the trace data
                with open(tasks_file, "r") as f:
                    task_data = json.load(f)

                # Update the user_evaluation
                task_data["traceMetadata"]["user_evaluation"] = user_evaluation

                # Save the updated data
                with open(tasks_file, "w") as f:
                    json.dump(task_data, f, indent=2)

            except (KeyError, TypeError) as e:
                raise ValueError(f"Error processing trace evaluation update: {e}")

    async def stop_task(self, trace_id: str):
        """Stop a task"""
        if trace_id in self.active_tasks:
            self.active_tasks[trace_id].update_trace_metadata(
                completed=True,
            )

    async def cleanup_tasks_for_websocket(self, websocket: WebSocket):
        """
        Clean up all tasks associated with a disconnected websocket.
        This will stop the tasks and release their sandboxes.

        Note: This also cleans up sandboxes that were acquired in create_id_and_sandbox
        but never had a task created (e.g., if websocket disconnects before process_user_task).
        """
        tasks_to_cleanup = []

        # Find all message_ids associated with this websocket
        async with self._lock:
            for message_id, ws in list(self.task_websockets.items()):
                if ws == websocket:
                    tasks_to_cleanup.append(message_id)
                    logger.info(
                        f"Marking task {message_id} for cleanup due to websocket disconnect"
                    )
                    # Remove from task_websockets immediately to prevent double cleanup
                    del self.task_websockets[message_id]

        # Cleanup each task
        for message_id in tasks_to_cleanup:
            try:
                # Mark task as completed to stop the agent (if task exists)
                if message_id in self.active_tasks:
                    self.active_tasks[message_id].update_trace_metadata(
                        completed=True,
                    )
                    logger.info(
                        f"Stopped task {message_id} due to websocket disconnect"
                    )

                # Always release the sandbox, even if no task was created
                # This handles the case where create_id_and_sandbox succeeded but
                # process_user_task was never called
                await self.sandbox_service.release_sandbox(message_id)
                logger.info(
                    f"Released sandbox for task {message_id} due to websocket disconnect"
                )

            except Exception as e:
                logger.error(f"Error cleaning up task {message_id}: {e}", exc_info=True)

    async def cleanup(self):
        """
        Cleanup method called during service shutdown.
        Stops the archival service and releases the lock file.
        """
        try:
            # Stop the archival service if it's running
            if self.archival_service.is_alive():
                logger.info("Stopping archival service...")
                self.archival_service.stop()
                logger.info("Archival service stopped")

            # Release the lock file if we hold it
            if self._archival_lock_file:
                try:
                    fcntl.flock(self._archival_lock_file.fileno(), fcntl.LOCK_UN)
                    self._archival_lock_file.close()
                    logger.info("Released archival service lock")
                except Exception as e:
                    logger.warning(f"Error releasing archival lock: {e}")
                finally:
                    self._archival_lock_file = None

        except Exception as e:
            logger.error(f"Error during AgentService cleanup: {e}", exc_info=True)
