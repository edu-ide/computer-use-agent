import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

from cua2_core.models.models import (
    ActiveTask,
    AgentAction,
    AgentCompleteEvent,
    AgentErrorEvent,
    AgentProgressEvent,
    AgentStartEvent,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
    VncUrlSetEvent,
    VncUrlUnsetEvent,
)
from cua2_core.websocket.websocket_manager import WebSocketManager


class AgentService:
    """Service for handling agent tasks and processing"""

    def __init__(self, websocket_manager):
        self.active_tasks: dict[str, ActiveTask] = {}
        self.websocket_manager: WebSocketManager = websocket_manager
        self.simulation_data_path = (
            Path(__file__).parent / "simulation_metadata" / "simulated_trace.json"
        )
        self.simulation_images_path = (
            Path(__file__).parent / "simulation_metadata" / "images"
        )

    async def process_user_task(self, trace: AgentTrace) -> str:
        """Process a user task and return the trace ID"""

        trace_id = trace.id
        trace.steps = []
        trace.traceMetadata = AgentTraceMetadata(traceId=trace_id)

        # Store the task
        self.active_tasks[trace_id] = ActiveTask(
            message_id=trace_id,
            instruction=trace.instruction,
            modelId=trace.modelId,
            timestamp=trace.timestamp,
            steps=trace.steps,
            traceMetadata=trace.traceMetadata,
        )

        # Start the agent processing in the background
        asyncio.create_task(self._simulate_agent_processing(trace))

        return trace_id

    async def _simulate_agent_processing(self, trace: AgentTrace):
        """Simulate agent processing using simulated_trace.json data"""
        trace_id = trace.id

        try:
            # Load simulation data
            with open(self.simulation_data_path, "r") as f:
                simulation_data = json.load(f)

            # Send agent start event with the initial trace
            start_event = AgentStartEvent(type="agent_start", agentTrace=trace)
            await self.websocket_manager.broadcast(start_event)

            # mock VNC URL
            vnc_url = "https://www.youtube.com/embed/VCutEsRSJ5A?si=PT0ETJ7zIJ9ywhGW"
            vnc_set_event = VncUrlSetEvent(type="vnc_url_set", vncUrl=vnc_url)
            await self.websocket_manager.broadcast(vnc_set_event)

            trace_metadata = AgentTraceMetadata(traceId=trace_id, maxSteps=20)

            # Process each step from the simulation data
            for step_data in simulation_data["steps"]:
                # Wait before sending the next step to simulate processing time
                await asyncio.sleep(step_data["duration"])

                # Load and encode the image
                image_path = (
                    self.simulation_images_path / step_data["image"].split("/")[-1]
                )
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                    image_base64 = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

                # Convert actions to AgentAction objects
                actions = [
                    AgentAction(
                        actionType=action["actionType"],
                        actionArguments=action["actionArguments"],
                    )
                    for action in step_data["actions"]
                ]

                # Create agent step
                agent_step = AgentStep(
                    traceId=trace_id,
                    stepId=step_data["stepId"],
                    image=image_base64,
                    thought=step_data["thought"],
                    actions=actions,
                    error="",
                    duration=step_data["duration"],
                    inputTokensUsed=step_data["inputTokensUsed"],
                    outputTokensUsed=step_data["outputTokensUsed"],
                    step_evaluation=step_data["step_evaluation"],
                )

                trace_metadata.numberOfSteps += 1
                trace_metadata.duration += step_data["duration"]
                trace_metadata.inputTokensUsed += step_data["inputTokensUsed"]
                trace_metadata.outputTokensUsed += step_data["outputTokensUsed"]

                # Send progress event
                progress_event = AgentProgressEvent(
                    type="agent_progress",
                    agentStep=agent_step,
                    traceMetadata=trace_metadata,
                )
                await self.websocket_manager.broadcast(progress_event)

                # Update active task
                self.active_tasks[trace_id].steps.append(agent_step)

            # Unset VNC URL before completion
            vnc_unset_event = VncUrlUnsetEvent(type="vnc_url_unset")
            await self.websocket_manager.broadcast(vnc_unset_event)

            # Send completion event
            complete_event = AgentCompleteEvent(
                type="agent_complete", traceMetadata=trace_metadata
            )
            await self.websocket_manager.broadcast(complete_event)

            # Update active task with final metadata
            self.active_tasks[trace_id].traceMetadata = trace_metadata

            # Clean up after a delay
            await asyncio.sleep(1)
            if trace_id in self.active_tasks:
                del self.active_tasks[trace_id]

        except Exception as e:
            print(f"Error in agent simulation: {str(e)}")
            # Send error event
            error_event = AgentErrorEvent(
                type="agent_error", error=f"Error processing task: {str(e)}"
            )
            await self.websocket_manager.broadcast(error_event)

            # Clean up
            if trace_id in self.active_tasks:
                del self.active_tasks[trace_id]

    def get_active_tasks(self) -> dict:
        """Get currently active tasks"""
        return self.active_tasks.copy()

    def get_task_status(self, message_id: str) -> Optional[dict]:
        """Get status of a specific task"""
        return self.active_tasks.get(message_id)
