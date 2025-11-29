import asyncio
import json
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional
from uuid import uuid4

from cua2_core.services.agent_utils.function_parser import FunctionCall
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, model_validator
from typing_extensions import TypeAlias

#################### Backend -> Frontend ########################


class AgentAction(FunctionCall):
    """Agent action structure"""

    @classmethod
    def from_function_calls(
        cls, function_calls: list[FunctionCall]
    ) -> list["AgentAction"]:
        list_of_actions = [cls(**action.model_dump()) for action in function_calls]
        for action in list_of_actions:
            action.description = action.to_string()
        return list_of_actions

    def to_string(self) -> str:
        """Convert action to a human-readable string"""
        action_type = self.function_name
        args = self.parameters

        if action_type == "click":
            x = args.get("x") or args.get("arg_0")
            y = args.get("y") or args.get("arg_1")
            return f"Click at coordinates ({x}, {y})"

        if action_type == "right_click":
            x = args.get("x") or args.get("arg_0")
            y = args.get("y") or args.get("arg_1")
            return f"Right click at coordinates ({x}, {y})"

        if action_type == "double_click":
            x = args.get("x") or args.get("arg_0")
            y = args.get("y") or args.get("arg_1")
            return f"Double click at coordinates ({x}, {y})"

        if action_type == "move_mouse":
            x = args.get("x") or args.get("arg_0")
            y = args.get("y") or args.get("arg_1")
            return f"Move mouse to coordinates ({x}, {y})"

        elif action_type == "write":
            text = args.get("text") or args.get("arg_0")
            return f"Type text: '{text}'"

        elif action_type == "press":
            key = args.get("key") or args.get("arg_0")
            return f"Press key: {key}"

        elif action_type == "go_back":
            return "Go back one page"

        elif action_type == "drag":
            x1 = args.get("x1") or args.get("arg_0")
            y1 = args.get("y1") or args.get("arg_1")
            x2 = args.get("x2") or args.get("arg_2")
            y2 = args.get("y2") or args.get("arg_3")
            return f"Drag from ({x1}, {y1}) to ({x2}, {y2})"

        elif action_type == "scroll":
            x = args.get("x") or args.get("arg_0")
            y = args.get("y") or args.get("arg_1")
            direction = args.get("direction") or args.get("arg_2")
            amount = args.get("amount") or args.get("arg_3") or 2
            return f"Scroll {direction} by {amount}"

        elif action_type == "wait":
            seconds = args.get("seconds") or args.get("arg_0")
            return f"Wait for {seconds} seconds"

        elif action_type == "open_url":
            url = args.get("url") or args.get("arg_0")
            return f"Open: {url}"

        elif action_type == "launch":
            url = args.get("app") or args.get("arg_0")
            return f"Open: {url}"

        elif action_type == "final_answer":
            answer = args.get("answer") or args.get("arg_0")
            return f"Final answer: {answer}"

        return "Unknown action"


class AgentStep(BaseModel):
    """Agent step structure"""

    traceId: str
    stepId: str
    image: str
    duration: float
    inputTokensUsed: int
    outputTokensUsed: int
    step_evaluation: Literal["like", "dislike", "neutral"]
    error: Optional[str] = None
    thought: Optional[str] = None
    actions: list[AgentAction] = []
    # 에이전트/모델 정보
    agentType: Optional[str] = None  # "VLMAgent", "SearchAgent", "AnalysisAgent" 등
    modelId: Optional[str] = None  # "local-qwen3-vl", "gpt-4o" 등
    nodeName: Optional[str] = None  # 워크플로우 노드 이름

    @field_serializer("image")
    def serialize_image(self, image: str, _info):
        """Convert image to path when dumping to JSON"""

        if _info.context and _info.context.get("image_as_path", True):
            return f"{self.traceId}-{self.stepId}.png"

        return image

    @field_serializer("actions")
    def serialize_actions(self, actions: list[AgentAction], _info):
        """Convert actions to list of strings when dumping (controlled by context)"""

        if _info.context and _info.context.get("actions_as_json", True):
            return [action.model_dump(mode="json") for action in actions]

        return [action.description for action in actions]


class AgentTraceMetadata(BaseModel):
    """Metadata for agent execution"""

    traceId: str = ""
    inputTokensUsed: int = 0
    outputTokensUsed: int = 0
    duration: float = 0.0  # in seconds
    numberOfSteps: int = 0
    maxSteps: int = 0
    completed: bool = False
    final_state: (
        Literal["success", "stopped", "max_steps_reached", "error", "sandbox_timeout"]
        | None
    ) = None
    user_evaluation: Literal["success", "failed", "not_evaluated"] = "not_evaluated"


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
    status: Literal["max_sandboxes_reached", "success"] = "success"


class AgentProgressEvent(BaseModel):
    """Agent progress event"""

    type: Literal["agent_progress"] = "agent_progress"
    agentStep: AgentStep
    traceMetadata: AgentTraceMetadata


class AgentCompleteEvent(BaseModel):
    """Agent complete event"""

    type: Literal["agent_complete"] = "agent_complete"
    traceMetadata: AgentTraceMetadata
    final_state: Literal[
        "success", "stopped", "max_steps_reached", "error", "sandbox_timeout"
    ]


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
    uuid: str = Field(default_factory=lambda: str(uuid4()))


# =========================================
# Human-in-the-Loop 이벤트
# =========================================

class ConfirmationRequiredEvent(BaseModel):
    """사용자 확인 요청 이벤트"""

    type: Literal["confirmation_required"] = "confirmation_required"
    workflow_id: str
    node_name: str
    message: str
    is_dangerous: bool = False
    input_type: Optional[str] = None  # "text", "captcha", "2fa"


class ConfirmationReceivedEvent(BaseModel):
    """사용자 확인 완료 이벤트"""

    type: Literal["confirmation_received"] = "confirmation_received"
    workflow_id: str
    node_name: str
    confirmed: bool
    user_input: Optional[str] = None


class WorkflowStateUpdateEvent(BaseModel):
    """워크플로우 상태 업데이트 이벤트"""

    type: Literal["workflow_state_update"] = "workflow_state_update"
    workflow_id: str
    execution_id: str
    status: str
    current_node: Optional[str] = None
    completed_nodes: List[str] = []
    failed_nodes: List[str] = []
    progress_percent: int = 0
    error: Optional[str] = None


class BreakpointHitEvent(BaseModel):
    """브레이크포인트 도달 이벤트"""

    type: Literal["breakpoint_hit"] = "breakpoint_hit"
    workflow_id: str
    node_name: str
    state: Optional[Dict[str, Any]] = None


class BreakpointResumedEvent(BaseModel):
    """브레이크포인트 재개 이벤트"""

    type: Literal["breakpoint_resumed"] = "breakpoint_resumed"
    workflow_id: str
    node_name: str


WebSocketEvent: TypeAlias = Annotated[
    AgentStartEvent
    | AgentProgressEvent
    | AgentCompleteEvent
    | AgentErrorEvent
    | VncUrlSetEvent
    | VncUrlUnsetEvent
    | HeartbeatEvent
    | ConfirmationRequiredEvent
    | ConfirmationReceivedEvent
    | WorkflowStateUpdateEvent
    | BreakpointHitEvent
    | BreakpointResumedEvent,
    Field(discriminator="type"),
]


#################### Frontend -> Backend ########################


class UserTaskMessage(BaseModel):
    """Message sent from frontend to backend"""

    event_type: Literal["user_task"]
    agent_trace: AgentTrace | None = None


class StopTask(BaseModel):
    """Stop task message"""

    event_type: Literal["stop_task"]
    traceId: str


class TraceEvaluation(BaseModel):
    """Trace evaluation message"""

    event_type: Literal["trace_evaluation"]
    traceId: str
    user_evaluation: Literal["success", "failed", "not_evaluated"]


##################### Agent Service ########################


class ActiveTask(BaseModel):
    """Active task"""

    message_id: str
    instruction: str
    model_id: str
    timestamp: datetime = datetime.now()
    steps: list[AgentStep] = []
    traceMetadata: AgentTraceMetadata = AgentTraceMetadata()
    _file_lock: asyncio.Lock | None = PrivateAttr(default=None)

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock (lazy initialization)"""
        if self._file_lock is None:
            self._file_lock = asyncio.Lock()
        return self._file_lock

    @property
    def trace_path(self):
        """Trace path"""
        return f"data/trace-{self.message_id}-{self.model_id.replace('/', '-')}"

    def _write_to_file_sync(self):
        """Synchronous file write helper (used in async context via to_thread)"""
        self.traceMetadata.traceId = self.message_id
        os.makedirs(self.trace_path, exist_ok=True)
        with open(f"{self.trace_path}/tasks.json", "w") as f:
            json.dump(
                self.model_dump(
                    mode="json",
                    exclude={"_file_lock", "_lock_initialized"},
                    context={"actions_as_json": True, "image_as_path": True},
                ),
                f,
                indent=2,
            )

    @model_validator(mode="after")
    def store_model(self):
        """Validate model ID - creates directory, but file write is deferred to async method"""
        self.traceMetadata.traceId = self.message_id
        os.makedirs(self.trace_path, exist_ok=True)
        return self

    async def save_to_file(self):
        """Async method to save task data to file"""
        async with self._get_lock():
            await asyncio.to_thread(self._write_to_file_sync)

    async def update_step(self, step: AgentStep):
        """Update step"""
        async with self._get_lock():
            if int(step.stepId) <= len(self.steps):
                self.steps[int(step.stepId) - 1] = step
            else:
                self.steps.append(step)
                self.traceMetadata.numberOfSteps = len(self.steps)
            # Use to_thread for file I/O to avoid blocking
            await asyncio.to_thread(self._write_to_file_sync)

    async def update_trace_metadata(
        self,
        step_input_tokens_used: int | None = None,
        step_output_tokens_used: int | None = None,
        step_duration: float | None = None,
        step_numberOfSteps: int | None = None,
        completed: bool | None = None,
        final_state: Literal[
            "success", "stopped", "max_steps_reached", "error", "sandbox_timeout"
        ]
        | None = None,
        user_evaluation: Literal["success", "failed", "not_evaluated"] | None = None,
    ):
        """Update trace metadata"""
        async with self._get_lock():
            if step_input_tokens_used is not None:
                self.traceMetadata.inputTokensUsed += step_input_tokens_used
            if step_output_tokens_used is not None:
                self.traceMetadata.outputTokensUsed += step_output_tokens_used
            if step_duration is not None:
                self.traceMetadata.duration += step_duration
            if step_numberOfSteps is not None:
                self.traceMetadata.numberOfSteps += step_numberOfSteps
            if completed is not None:
                self.traceMetadata.completed = completed
            if final_state is not None:
                self.traceMetadata.final_state = final_state
            if user_evaluation is not None:
                self.traceMetadata.user_evaluation = user_evaluation


#################### API Routes Models ########################


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime
    websocket_connections: int


class TaskStatusResponse(BaseModel):
    """Response for a specific task status"""

    task_id: str
    status: ActiveTask


class ActiveTasksResponse(BaseModel):
    """Response for active tasks"""

    active_tasks: dict[str, ActiveTask]
    total_connections: int


class UpdateStepRequest(BaseModel):
    """Request model for updating a step"""

    step_evaluation: Literal["like", "dislike", "neutral"]


class UpdateStepResponse(BaseModel):
    """Response model for step update"""

    success: bool
    message: str


class UpdateTraceEvaluationRequest(BaseModel):
    """Request model for updating trace evaluation"""

    user_evaluation: Literal["success", "failed", "not_evaluated"]


class UpdateTraceEvaluationResponse(BaseModel):
    """Response model for trace evaluation update"""

    success: bool
    message: str


class AvailableModelsResponse(BaseModel):
    """Response for available models"""

    models: list[str]


class AgentTypeInfo(BaseModel):
    """에이전트 타입 정보"""

    name: str  # VLMAgent, SearchAgent 등
    description: str
    base_class: str  # smolagents.CodeAgent 등
    capabilities: List[str]  # 지원하는 도구/기능 목록
    default_model: Optional[str] = None  # 기본 사용 모델


class AvailableAgentTypesResponse(BaseModel):
    """Response for available agent types"""

    agent_types: List[AgentTypeInfo]


class GenerateInstructionResponse(BaseModel):
    """Response model for generated task instruction"""

    instruction: str
