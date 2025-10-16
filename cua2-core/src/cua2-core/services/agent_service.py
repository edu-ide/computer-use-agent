import asyncio
import uuid
from datetime import datetime
from typing import Optional

from smolagents import Model

from backend.models.models import ActiveTask, AgentMetadata
from backend.services.agents.get_agents import get_agent
from backend.services.models.get_model import get_model
from backend.websocket.websocket_manager import WebSocketManager
from computer_use_studio import Sandbox
from computer_use_studio.logger import get_logger

logger = get_logger(__name__)


class AgentService:
    """Service for handling agent tasks and processing"""

    def __init__(self, websocket_manager):
        self.active_tasks: dict[str, ActiveTask] = {}
        self.websocket_manager: WebSocketManager = websocket_manager

    async def process_user_task(self, content: str, model_id: str) -> str:
        """Process a user task and return the message ID"""

        message_id = str(uuid.uuid4())
        while message_id in self.active_tasks.keys():
            message_id = str(uuid.uuid4())

        # Store the task
        self.active_tasks[message_id] = ActiveTask(
            message_id=message_id,
            content=content,
            model_id=model_id,
            start_time=datetime.now(),
            status="processing",
        )

        # Determine the agent type based on the content of the task (TODO: implement agent type detection using LLM)
        prompt_type = "FORM_SYSTEM_PROMPT"

        # Start the agent processing in the background
        asyncio.create_task(
            self._simulate_agent_processing(content, model_id, message_id, prompt_type)
        )

        return message_id


    #     async def _simulate_agent_processing(self, message_id: str, content: str):
    #         """Simulate agent processing with progress updates"""
    #         try:
    #             # Send agent start event
    #             await self.websocket_manager.send_agent_start(
    #                 content=f"Starting task: {content}", message_id=message_id
    #             )
    #
    #             # Simulate processing steps
    #             steps = [
    #                 "Analyzing task requirements...",
    #                 "Planning execution steps...",
    #                 "Initializing computer interface...",
    #                 "Executing task commands...",
    #                 "Verifying results...",
    #                 "Finalizing task completion...",
    #             ]
    #
    #             for i, step in enumerate(steps):
    #                 await asyncio.sleep(2)  # Simulate processing time
    #
    #                 # Send progress update
    #                 await self.websocket_manager.send_agent_progress(
    #                     content=f"{step} ({i + 1}/{len(steps)})", message_id=message_id
    #                 )
    #
    #                 # Simulate VNC URL events during processing
    #                 if i == 2:  # After "Initializing computer interface..."
    #                     # Set VNC URL when computer interface is ready
    #                     vnc_url = "http://localhost:6080/vnc.html?host=localhost&port=5900&autoconnect=true"
    #                     await self.websocket_manager.send_vnc_url_set(
    #                         vnc_url=vnc_url,
    #                         content="Computer interface ready, VNC stream connected",
    #                     )
    #                 elif i == 4:  # After "Verifying results..."
    #                     # Unset VNC URL when task is almost complete
    #                     await self.websocket_manager.send_vnc_url_unset(
    #                         content="Task verification complete, disconnecting VNC stream"
    #                     )
    #
    #             # Calculate metadata
    #             end_time = datetime.now()
    #             start_time = self.active_tasks[message_id]["start_time"]
    #             time_taken = (end_time - start_time).total_seconds()
    #
    #             metadata = AgentMetadata(
    #                 tokensUsed=150 + len(content) * 2,  # Simulate token usage
    #                 timeTaken=time_taken,
    #                 numberOfSteps=len(steps),
    #             )
    #
    #             # Send completion event
    #             await self.websocket_manager.send_agent_complete(
    #                 content=f"Task completed successfully: {content}",
    #                 message_id=message_id,
    #                 metadata=metadata,
    #             )
    #
    #             # Clean up
    #             if message_id in self.active_tasks:
    #                 del self.active_tasks[message_id]
    #
    #         except Exception as e:
    #             # Send error event
    #             await self.websocket_manager.send_agent_error(
    #                 content=f"Error processing task: {str(e)}", message_id=message_id
    #             )
    #
    #             # Clean up
    #             if message_id in self.active_tasks:
    #                 del self.active_tasks[message_id]

    def get_active_tasks(self) -> dict:
        """Get currently active tasks"""
        return self.active_tasks.copy()

    def get_task_status(self, message_id: str) -> Optional[dict]:
        """Get status of a specific task"""
        return self.active_tasks.get(message_id)
