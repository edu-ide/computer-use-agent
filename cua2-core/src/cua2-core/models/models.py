import json
import os
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class AgentMetadata(BaseModel):
    """Metadata for agent execution"""

    inputTokensUsed: int
    outputTokensUsed: int
    timeTaken: float  # in seconds
    numberOfSteps: int


class AgentType(str, Enum):
    """Agent type"""

    PIXEL_COORDINATES = "pixel_coordinates"
    NORMALIZED_1000_COORDINATES = "normalized_1000_coordinates"
    NORMALIZED_COORDINATES = "normalized_coordinates"


class ActiveTask(BaseModel):
    """Active task"""

    message_id: str
    content: str
    model_id: str
    start_time: datetime
    status: str

    @property
    def trace_path(self):
        """Trace path"""
        return f"data/trace-{self.message_id}-{self.model_id}"

    @model_validator(mode="after")
    def validate_model_id(self):
        """Validate model ID"""
        os.makedirs(self.trace_path, exist_ok=True)
        with open(f"{self.trace_path}/user_tasks.json", "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

        return self


class WebSocketEvent(BaseModel):
    """WebSocket event structure"""

    type: Literal[
        "agent_start",
        "agent_progress",
        "agent_complete",
        "agent_error",
        "vnc_url_set",
        "vnc_url_unset",
        "heartbeat",
    ]
    content: Optional[str] = None
    metadata: Optional[AgentMetadata] = None
    messageId: Optional[str] = None
    vncUrl: Optional[str] = None


class UserTaskMessage(BaseModel):
    """Message sent from frontend to backend"""

    type: Literal["user_task"]
    content: str
    model_id: str
    timestamp: str


class AgentMessage(BaseModel):
    """Agent message structure"""

    id: str
    type: Literal["user", "agent"]
    content: str
    timestamp: datetime
    metadata: Optional[AgentMetadata] = None
    isLoading: Optional[bool] = None
    truncated: Optional[bool] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    websocket_connections: int
