import json
import os
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, field_serializer, model_validator

#################### Backend -> Frontend ########################

class AgentAction(BaseModel):
    """Agent action structure"""

    actionType: Literal["click", "write", "press", "scroll", "wait", "open", "launch_app", "refresh", "go_back"]
    actionArguments: dict

    def to_string(self) -> str:
        """Convert action to a human-readable string"""
        action_type = self.actionType
        args = self.actionArguments
        
        if action_type == "click":
            x = args.get("x", "?")
            y = args.get("y", "?")
            return f"Click at coordinates ({x}, {y})"
        
        elif action_type == "write":
            text = args.get("text", "")
            return f"Type text: '{text}'"
        
        elif action_type == "press":
            key = args.get("key", "")
            return f"Press key: {key}"
        
        elif action_type == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 2)
            return f"Scroll {direction} by {amount}"
        
        elif action_type == "wait":
            seconds = args.get("seconds", 0)
            return f"Wait for {seconds} seconds"
        
        elif action_type == "open":
            file_or_url = args.get("file_or_url", "")
            return f"Open: {file_or_url}"
        
        elif action_type == "launch_app":
            app_name = args.get("app_name", "")
            return f"Launch app: {app_name}"
        
        elif action_type == "refresh":
            return "Refresh the current page"
        
        elif action_type == "go_back":
            return "Go back one page"


class AgentStep(BaseModel):
    """Agent step structure"""

    traceId: str
    stepId: str
    image: str
    thought: str
    actions: list[AgentAction]
    timeTaken: float
    inputTokensUsed: int
    outputTokensUsed: int
    timestamp: datetime
    step_evaluation: Literal['like', 'dislike', 'neutral']
    
    @field_serializer('actions')
    def serialize_actions(self, actions: list[AgentAction], _info):
        """Convert actions to list of strings when dumping (controlled by context)"""

        if _info.context and _info.context.get('actions_as_json', False):
            return [action.model_dump(mode="json") for action in actions]

        return [action.to_string() for action in actions]


class AgentTraceMetadata(BaseModel):
    """Metadata for agent execution"""

    traceId: str = ""
    inputTokensUsed: int = 0
    outputTokensUsed: int = 0
    timeTaken: float = 0.0  # in seconds
    numberOfSteps: int = 0


class AgentTrace(BaseModel):
    """Agent message structure"""

    id: str
    timestamp: datetime
    instruction: str
    modelId: str
    isRunning: bool
    steps: list[AgentStep] = []
    traceMetadata: AgentTraceMetadata = AgentTraceMetadata()

    @model_validator(mode="after")
    def validate_trace(self):
        """Validate trace"""
        if not self.steps:
            self.steps = []
        if not self.traceMetadata:
            self.traceMetadata = AgentTraceMetadata()
        return self


#################### WebSocket Events ########################


class AgentStartEvent(BaseModel):
    """Agent start event"""

    type: Literal["agent_start"] = "agent_start"
    agentTrace: AgentTrace


class AgentProgressEvent(BaseModel):
    """Agent progress event"""

    type: Literal["agent_progress"] = "agent_progress"
    agentStep: AgentStep
    traceMetadata: AgentTraceMetadata


class AgentCompleteEvent(BaseModel):
    """Agent complete event"""

    type: Literal["agent_complete"] = "agent_complete"
    traceMetadata: AgentTraceMetadata


class AgentErrorEvent(BaseModel):
    """Agent error event"""

    type: Literal["agent_error"] = "agent_error"
    error: str


class VncUrlSetEvent(BaseModel):
    """Vnc url set event"""

    type: Literal["vnc_url_set"] = "vnc_url_set"
    vncUrl: str


class VncUrlUnsetEvent(BaseModel):
    """Vnc url unset event"""

    type: Literal["vnc_url_unset"] = "vnc_url_unset"


class HeartbeatEvent(BaseModel):
    """Heartbeat event"""

    type: Literal["heartbeat"] = "heartbeat"


WebSocketEvent: TypeAlias = Annotated[
    AgentStartEvent
    | AgentProgressEvent
    | AgentCompleteEvent
    | AgentErrorEvent
    | VncUrlSetEvent
    | VncUrlUnsetEvent
    | HeartbeatEvent,
    Field(discriminator="type"),
]


#################### Frontend -> Backend ########################


class UserTaskMessage(BaseModel):
    """Message sent from frontend to backend"""

    event_type: Literal["user_task"]
    agent_trace: AgentTrace | None = None


##################### Agent Service ########################


class ActiveTask(BaseModel):
    """Active task"""

    message_id: str
    instruction: str
    modelId: str
    timestamp: datetime = datetime.now()
    steps: list[AgentStep] = []
    traceMetadata: AgentTraceMetadata = AgentTraceMetadata()

    @property
    def trace_path(self):
        """Trace path"""
        return f"data/trace-{self.message_id}-{self.modelId}"

    @model_validator(mode="after")
    def store_model(self):
        """Validate model ID"""
        self.traceMetadata.traceId = self.message_id
        os.makedirs(self.trace_path, exist_ok=True)
        with open(f"{self.trace_path}/tasks.json", "w") as f:
            json.dump(self.model_dump(mode="json", context={"actions_as_json": True}), f, indent=2)

        return self


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    websocket_connections: int
