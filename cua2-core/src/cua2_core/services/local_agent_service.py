"""
로컬 에이전트 서비스 - E2B 없이 로컬 데스크톱 사용
"""

import asyncio
import base64
import logging
import os
import time
from io import BytesIO
from typing import Callable
from uuid import uuid4

from cua2_core.models.models import (
    ActiveTask,
    AgentAction,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
)
from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent
from cua2_core.services.agent_utils.function_parser import parse_function_call
from cua2_core.services.agent_utils.get_model import get_model
from cua2_core.services.local_sandbox_service import LocalSandboxService
from cua2_core.services.utils import compress_image_to_max_size
from cua2_core.websocket.websocket_manager import WebSocketException, WebSocketManager
from fastapi import WebSocket
from PIL import Image
from smolagents import ActionStep, AgentMaxStepsError, TaskStep
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

AGENT_RUN_TIMEOUT = 1000


class AgentStopException(Exception):
    """에이전트 중지 예외"""
    pass


class LocalAgentService:
    """로컬 데스크톱 에이전트 서비스"""

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        max_sandboxes: int = 1,
    ):
        self.active_tasks: dict[str, ActiveTask] = {}
        self.websocket_manager = websocket_manager
        self.task_websockets: dict[str, WebSocket] = {}
        self.sandbox_service = LocalSandboxService(max_sandboxes=max_sandboxes)
        self.last_screenshot: dict[str, tuple[Image.Image, str] | None] = {}
        self._lock = asyncio.Lock()
        self.max_sandboxes = max_sandboxes

    async def create_id_and_sandbox(self, websocket: WebSocket) -> str:
        """새 ID와 샌드박스 생성"""
        async with self._lock:
            uuid = str(uuid4())
            while uuid in self.active_tasks:
                uuid = str(uuid4())
            self.task_websockets[uuid] = websocket
        logger.info(f"UUID 생성: {uuid}")
        return uuid

    async def process_user_task(
        self, trace: AgentTrace, websocket: WebSocket
    ) -> str | None:
        """사용자 작업 처리"""
        trace_id = trace.id
        trace.steps = []
        trace.traceMetadata = AgentTraceMetadata(traceId=trace_id)

        async with self._lock:
            # 웹소켓을 trace_id로 등록 (기존 uuid 대신)
            self.task_websockets[trace_id] = websocket

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

            self.active_tasks[trace_id] = active_task
            self.last_screenshot[trace_id] = None

        asyncio.create_task(self._agent_processing(trace_id))
        return trace_id

    async def _agent_runner(
        self,
        message_id: str,
        step_callback: Callable[[ActionStep, LocalVisionAgent], None],
    ):
        """에이전트 실행"""
        sandbox = None
        agent = None
        final_state = "success"
        websocket_exception = False

        try:
            websocket = self.task_websockets.get(message_id)

            await self.websocket_manager.send_agent_start(
                active_task=self.active_tasks[message_id],
                websocket=websocket,
                status="success",
            )

            model = get_model(self.active_tasks[message_id].model_id)

            # 로컬 샌드박스 획득
            max_attempts = 30
            for attempt in range(max_attempts):
                response = await self.sandbox_service.acquire_sandbox(message_id)

                if response.error:
                    logger.error(f"샌드박스 생성 실패: {response.error}")
                    await asyncio.sleep(1)
                    continue

                if response.sandbox is not None and response.state == "ready":
                    sandbox = response.sandbox
                    break

                if response.state == "max_sandboxes_reached":
                    await asyncio.sleep(1)
                    continue

                await asyncio.sleep(1)

            if sandbox is None:
                raise Exception(f"샌드박스를 사용할 수 없음: {message_id}")

            data_dir = self.active_tasks[message_id].trace_path
            user_content = self.active_tasks[message_id].instruction

            agent = LocalVisionAgent(
                model=model,
                data_dir=data_dir,
                desktop=sandbox,
                step_callbacks=[step_callback],
            )

            self.active_tasks[message_id].traceMetadata.maxSteps = agent.max_steps

            # 로컬이므로 VNC URL은 빈 문자열
            await self.websocket_manager.send_vnc_url_set(
                vnc_url="",
                websocket=websocket,
            )

            # 초기 스크린샷
            step_filename = f"{message_id}-1"
            screenshot_image = agent.desktop.screenshot()
            self.last_screenshot[message_id] = (screenshot_image, step_filename)

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(agent.run, user_content),
                    timeout=AGENT_RUN_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(f"에이전트 타임아웃: {message_id}")
                raise Exception(f"에이전트 타임아웃")

            self.active_tasks[message_id].traceMetadata.completed = True

        except AgentStopException as e:
            if str(e) == "Max steps reached":
                final_state = "max_steps_reached"
            else:
                final_state = "stopped"

        except WebSocketException:
            websocket_exception = True

        except Exception as e:
            import traceback
            logger.error(f"작업 처리 오류: {traceback.format_exc()}")
            final_state = "error"
            if (
                not websocket_exception
                and websocket
                and websocket.client_state == WebSocketState.CONNECTED
            ):
                await self.websocket_manager.send_agent_error(
                    error=str(e), websocket=websocket
                )

        finally:
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
                await self.websocket_manager.send_vnc_url_unset(websocket=websocket)

            await self.active_tasks[message_id].update_trace_metadata(
                final_state=final_state,
                completed=True,
            )

            if message_id in self.active_tasks:
                await self.active_tasks[message_id].save_to_file()

            async with self._lock:
                self.active_tasks.pop(message_id, None)
                self.task_websockets.pop(message_id, None)
                self.last_screenshot.pop(message_id, None)

            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as e:
                logger.error(f"샌드박스 해제 오류: {e}")

    async def _agent_processing(self, message_id: str):
        """에이전트 처리"""
        try:
            active_task = self.active_tasks[message_id]
            os.makedirs(active_task.trace_path, exist_ok=True)
            loop = asyncio.get_running_loop()

            def step_callback(memory_step: ActionStep, agent: LocalVisionAgent):
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
                    action_sequence = """The task failed due to an error"""

                agent_actions = (
                    AgentAction.from_function_calls(
                        parse_function_call(action_sequence)
                    )
                    if action_sequence
                    else None
                )

                time.sleep(3)

                image, step_filename = self.last_screenshot[message_id]
                assert image is not None and step_filename is not None
                screenshot_path = os.path.join(agent.data_dir, f"{step_filename}.png")
                image.save(screenshot_path)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                del buffered

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
                        # 에이전트/모델 정보
                        agentType="VLMAgent",
                        modelId=self.active_tasks[message_id].model_id,
                    )

                    future1 = asyncio.run_coroutine_threadsafe(
                        self.active_tasks[message_id].update_trace_metadata(
                            step_input_tokens_used=memory_step.token_usage.input_tokens,
                            step_output_tokens_used=memory_step.token_usage.output_tokens,
                            step_duration=memory_step.timing.duration,
                            step_numberOfSteps=1,
                        ),
                        loop,
                    )
                    future2 = asyncio.run_coroutine_threadsafe(
                        self.active_tasks[message_id].update_step(step),
                        loop,
                    )
                    future1.result()
                    future2.result()

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
                screenshot_image = agent.desktop.screenshot()
                image = compress_image_to_max_size(screenshot_image, max_size_kb=500)

                for previous_memory_step in agent.memory.steps:
                    if isinstance(previous_memory_step, ActionStep):
                        previous_memory_step.observations_images = None
                    elif isinstance(previous_memory_step, TaskStep):
                        previous_memory_step.task_images = None

                memory_step.observations_images = [image.copy()]

                del self.last_screenshot[message_id]
                self.last_screenshot[message_id] = (image, step_filename)

            await self._agent_runner(message_id, step_callback)

        except Exception as e:
            logger.error(f"에이전트 처리 오류: {e}")
            try:
                await self.sandbox_service.release_sandbox(message_id)
            except Exception:
                pass
            raise

    async def stop_task(self, trace_id: str):
        """작업 중지"""
        if trace_id in self.active_tasks:
            await self.active_tasks[trace_id].update_trace_metadata(completed=True)

    async def cleanup_tasks_for_websocket(self, websocket: WebSocket):
        """웹소켓 연결 끊김 시 정리"""
        tasks_to_cleanup = []

        async with self._lock:
            for message_id, ws in list(self.task_websockets.items()):
                if ws == websocket:
                    tasks_to_cleanup.append(message_id)
                    del self.task_websockets[message_id]

        for message_id in tasks_to_cleanup:
            try:
                if message_id in self.active_tasks:
                    await self.active_tasks[message_id].update_trace_metadata(
                        completed=True
                    )
                await self.sandbox_service.release_sandbox(message_id)
            except Exception as e:
                logger.error(f"정리 오류 ({message_id}): {e}")

    async def cleanup(self):
        """서비스 종료 시 정리"""
        await self.sandbox_service.cleanup_sandboxes()
