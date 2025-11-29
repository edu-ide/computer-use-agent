from datetime import datetime
from typing import Any, Dict, List, Optional

# Get services from app state
from cua2_core.models.models import (
    AgentTypeInfo,
    AvailableAgentTypesResponse,
    AvailableModelsResponse,
    GenerateInstructionResponse,
    HealthResponse,
    UpdateStepRequest,
    UpdateStepResponse,
    UpdateTraceEvaluationRequest,
    UpdateTraceEvaluationResponse,
)
# AgentService는 app.state에서 가져옴 (로컬/클라우드 모드에 따라 다름)
from cua2_core.services.agent_utils.get_model import AVAILABLE_MODELS
from cua2_core.services.instruction_service import InstructionService
from cua2_core.services.orchestrator import (
    StrategySelector,
    WorkflowMonitor,
    StepEvaluator,
)
from cua2_core.websocket.websocket_manager import WebSocketManager
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

# Create router
router = APIRouter()


def get_websocket_manager(request: Request) -> WebSocketManager:
    """Dependency to get WebSocket manager from app state"""
    return request.app.state.websocket_manager


def get_agent_service(request: Request):
    """Dependency to get agent service from app state"""
    return request.app.state.agent_service


@router.get("/health", response_model=HealthResponse)
async def health_check(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        websocket_connections=websocket_manager.get_connection_count(),
    )


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get list of all available model IDs"""
    return AvailableModelsResponse(models=AVAILABLE_MODELS)


# 에이전트 타입 정의 (smolagents 기반)
AVAILABLE_AGENT_TYPES = [
    AgentTypeInfo(
        name="VLMAgent",
        description="Vision-Language Model 기반 GUI 자동화 에이전트. 화면을 보고 마우스/키보드 제어",
        base_class="smolagents.CodeAgent (LocalVisionAgent)",
        capabilities=["click", "scroll", "write", "press", "open_url", "screenshot"],
        default_model="local-qwen3-vl",
    ),
    AgentTypeInfo(
        name="SearchAgent",
        description="검색 및 페이지 탐색 전문 에이전트. 키워드 검색, 필터 적용, 페이지네이션",
        base_class="smolagents.CodeAgent (BaseSpecializedAgent)",
        capabilities=["search", "filter", "navigate", "paginate"],
        default_model="local-qwen3-vl",
    ),
    AgentTypeInfo(
        name="AnalysisAgent",
        description="데이터 추출 및 분석 전문 에이전트. 상품 정보 추출, 가격 분석",
        base_class="smolagents.CodeAgent (BaseSpecializedAgent)",
        capabilities=["extract", "analyze", "structure_data", "compare"],
        default_model="local-qwen3-vl",
    ),
    AgentTypeInfo(
        name="ValidationAgent",
        description="데이터 검증 전문 에이전트. 수집된 데이터의 유효성 검사",
        base_class="smolagents.CodeAgent (BaseSpecializedAgent)",
        capabilities=["validate", "verify", "check_format"],
        default_model="local-qwen3-vl",
    ),
    AgentTypeInfo(
        name="ManagerAgent",
        description="다중 에이전트 조정자. 다른 에이전트들을 오케스트레이션",
        base_class="smolagents.ManagedAgent",
        capabilities=["orchestrate", "delegate", "coordinate"],
        default_model="local-qwen3-vl",
    ),
]


@router.get("/agent-types", response_model=AvailableAgentTypesResponse)
async def get_available_agent_types():
    """Get list of all available agent types with their capabilities"""
    return AvailableAgentTypesResponse(agent_types=AVAILABLE_AGENT_TYPES)


@router.post("/generate-instruction", response_model=GenerateInstructionResponse)
async def generate_task_instruction():
    """Get a random task instruction from the pregenerated pool"""
    try:
        instruction = InstructionService.get_random_instruction()
        return GenerateInstructionResponse(instruction=instruction)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating instruction: {str(e)}",
        )


@router.patch("/traces/{trace_id}/steps/{step_id}", response_model=UpdateStepResponse)
async def update_trace_step(
    trace_id: str,
    step_id: str,
    request: UpdateStepRequest,
    agent_service = Depends(get_agent_service),
):
    """Update a specific step in a trace (e.g., update step evaluation)"""
    try:
        await agent_service.update_trace_step(
            trace_id=trace_id,
            step_id=step_id,
            step_evaluation=request.step_evaluation,
        )
        return UpdateStepResponse(
            success=True,
            message="Step updated successfully",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch(
    "/traces/{trace_id}/evaluation", response_model=UpdateTraceEvaluationResponse
)
async def update_trace_evaluation(
    trace_id: str,
    request: UpdateTraceEvaluationRequest,
    agent_service = Depends(get_agent_service),
):
    """Update the user evaluation for a trace (overall task feedback)"""
    try:
        await agent_service.update_trace_evaluation(
            trace_id=trace_id,
            user_evaluation=request.user_evaluation,
        )
        return UpdateTraceEvaluationResponse(
            success=True,
            message="Trace evaluation updated successfully",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================
# Orchestrator 엔드포인트
# =========================================

class StrategyRequest(BaseModel):
    """전략 결정 요청"""
    node_id: str
    instruction: str
    params: Optional[Dict[str, Any]] = None


class StrategyResponse(BaseModel):
    """전략 결정 응답"""
    strategy: str
    model_id: str
    reason: str
    complexity: Dict[str, Any]


class WorkflowStatusResponse(BaseModel):
    """워크플로우 상태 응답"""
    workflow_id: str
    execution_id: str
    status: str
    completed_nodes: int
    total_nodes: int
    failed_nodes: List[str]
    current_node: Optional[str]
    is_stuck: bool


class StepEvaluationRequest(BaseModel):
    """스텝 평가 요청"""
    workflow_id: str
    node_id: str
    step_number: int
    thought: str
    action: str
    observation: str


class StepEvaluationResponse(BaseModel):
    """스텝 평가 응답"""
    action: str  # continue, retry, skip, stop
    feedback: Optional[str]
    should_inject_hint: bool
    hint: Optional[str]


# Orchestrator 서비스 인스턴스 (싱글톤)
_strategy_selector: Optional[StrategySelector] = None
_workflow_monitor: Optional[WorkflowMonitor] = None
_step_evaluator: Optional[StepEvaluator] = None


def get_strategy_selector() -> StrategySelector:
    """전략 선택기 싱글톤"""
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = StrategySelector(prefer_local=True)
    return _strategy_selector


def get_workflow_monitor() -> WorkflowMonitor:
    """워크플로우 모니터 싱글톤"""
    global _workflow_monitor
    if _workflow_monitor is None:
        _workflow_monitor = WorkflowMonitor()
    return _workflow_monitor


def get_step_evaluator() -> StepEvaluator:
    """스텝 평가기 싱글톤"""
    global _step_evaluator
    if _step_evaluator is None:
        _step_evaluator = StepEvaluator()
    return _step_evaluator


@router.post("/orchestrator/strategy", response_model=StrategyResponse)
async def decide_strategy(request: StrategyRequest):
    """노드에 대한 실행 전략 결정"""
    selector = get_strategy_selector()

    decision = selector.decide(
        node_id=request.node_id,
        instruction=request.instruction,
        params=request.params or {},
    )

    return StrategyResponse(
        strategy=decision.strategy.value,
        model_id=decision.model_id,
        reason=decision.reason,
        complexity={
            "level": decision.complexity.level.value,
            "has_vision": decision.complexity.has_vision,
            "has_reasoning": decision.complexity.has_reasoning,
            "has_extraction": decision.complexity.has_extraction,
            "estimated_steps": decision.complexity.estimated_steps,
        },
    )


@router.post("/orchestrator/workflow/{workflow_id}/start")
async def start_workflow_tracking(
    workflow_id: str,
    execution_id: str,
    total_nodes: int,
):
    """워크플로우 추적 시작"""
    monitor = get_workflow_monitor()
    monitor.start_workflow_tracking(
        workflow_id=workflow_id,
        execution_id=execution_id,
        total_nodes=total_nodes,
    )
    return {"status": "tracking_started", "workflow_id": workflow_id}


@router.get("/orchestrator/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """워크플로우 상태 조회"""
    monitor = get_workflow_monitor()
    status = monitor.get_workflow_status(workflow_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        execution_id=status.get("execution_id", ""),
        status=status.get("status", "unknown"),
        completed_nodes=status.get("completed_nodes", 0),
        total_nodes=status.get("total_nodes", 0),
        failed_nodes=status.get("failed_nodes", []),
        current_node=status.get("current_node"),
        is_stuck=status.get("is_stuck", False),
    )


@router.post("/orchestrator/evaluate-step", response_model=StepEvaluationResponse)
async def evaluate_step(request: StepEvaluationRequest):
    """VLM 스텝 평가"""
    evaluator = get_step_evaluator()

    feedback = evaluator.evaluate_step(
        workflow_id=request.workflow_id,
        node_id=request.node_id,
        step_number=request.step_number,
        thought=request.thought,
        action=request.action,
        observation=request.observation,
    )

    return StepEvaluationResponse(
        action=feedback.action.value,
        feedback=feedback.feedback,
        should_inject_hint=feedback.should_inject_hint,
        hint=feedback.hint,
    )


# =========================================
# Human-in-the-Loop 엔드포인트
# =========================================

class ConfirmationStatusResponse(BaseModel):
    """확인 대기 상태 응답"""
    waiting: bool
    node: Optional[str] = None
    message: Optional[str] = None
    is_dangerous: bool = False


class ConfirmationRequest(BaseModel):
    """사용자 확인 요청"""
    confirmed: bool
    user_input: Optional[str] = None


class ConfirmationResponse(BaseModel):
    """확인 처리 응답"""
    success: bool
    message: str


@router.get("/workflow/{workflow_id}/confirmation", response_model=ConfirmationStatusResponse)
async def get_confirmation_status(
    workflow_id: str,
    request: Request,
):
    """
    워크플로우의 사용자 확인 대기 상태 조회

    Human-in-the-Loop: 워크플로우가 사용자 확인을 기다리고 있는지 확인
    """
    # workflow_registry에서 현재 실행 중인 워크플로우 조회
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    # 확인 대기 상태 조회
    if workflow.is_waiting_for_confirmation():
        pending = workflow.get_pending_confirmation()
        return ConfirmationStatusResponse(
            waiting=True,
            node=pending.get("node") if pending else None,
            message=pending.get("message") if pending else None,
            is_dangerous=pending.get("is_dangerous", False) if pending else False,
        )

    return ConfirmationStatusResponse(waiting=False)


@router.post("/workflow/{workflow_id}/confirm", response_model=ConfirmationResponse)
async def confirm_workflow_action(
    workflow_id: str,
    body: ConfirmationRequest,
    request: Request,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """
    워크플로우 작업 사용자 확인/취소

    Human-in-the-Loop: 대기 중인 작업을 승인하거나 취소
    """
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if not workflow.is_waiting_for_confirmation():
        raise HTTPException(status_code=400, detail="Workflow is not waiting for confirmation")

    # 대기 중인 노드 이름 가져오기
    pending = workflow.get_pending_confirmation()
    node_name = pending.get("node") if pending else None

    # 확인 처리
    workflow.confirm(confirmed=body.confirmed, user_input=body.user_input)

    # WebSocket으로 확인 결과 브로드캐스트
    await websocket_manager.send_confirmation_received(
        workflow_id=workflow_id,
        node_name=node_name or "",
        confirmed=body.confirmed,
        user_input=body.user_input,
    )

    action = "승인" if body.confirmed else "취소"
    return ConfirmationResponse(
        success=True,
        message=f"작업이 {action}되었습니다",
    )


# =========================================
# Breakpoint 엔드포인트
# =========================================

class BreakpointRequest(BaseModel):
    """브레이크포인트 요청"""
    node_name: str


class BreakpointListResponse(BaseModel):
    """브레이크포인트 목록 응답"""
    breakpoints: List[str]


class BreakpointActionResponse(BaseModel):
    """브레이크포인트 액션 응답"""
    success: bool
    message: str


@router.get("/workflow/{workflow_id}/breakpoints", response_model=BreakpointListResponse)
async def get_breakpoints(
    workflow_id: str,
    request: Request,
):
    """워크플로우의 브레이크포인트 목록 조회"""
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    return BreakpointListResponse(breakpoints=workflow.get_breakpoints())


@router.post("/workflow/{workflow_id}/breakpoints", response_model=BreakpointActionResponse)
async def add_breakpoint(
    workflow_id: str,
    body: BreakpointRequest,
    request: Request,
):
    """브레이크포인트 추가"""
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    workflow.add_breakpoint(body.node_name)
    return BreakpointActionResponse(
        success=True,
        message=f"브레이크포인트 추가됨: {body.node_name}",
    )


@router.delete("/workflow/{workflow_id}/breakpoints/{node_name}", response_model=BreakpointActionResponse)
async def remove_breakpoint(
    workflow_id: str,
    node_name: str,
    request: Request,
):
    """브레이크포인트 제거"""
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    workflow.remove_breakpoint(node_name)
    return BreakpointActionResponse(
        success=True,
        message=f"브레이크포인트 제거됨: {node_name}",
    )


@router.post("/workflow/{workflow_id}/resume", response_model=BreakpointActionResponse)
async def resume_from_breakpoint(
    workflow_id: str,
    request: Request,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """브레이크포인트에서 실행 재개"""
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if not workflow.is_paused_at_breakpoint():
        raise HTTPException(status_code=400, detail="Workflow is not paused at breakpoint")

    # 현재 브레이크포인트 노드
    breakpoint_info = workflow.get_breakpoint_info()
    node_name = breakpoint_info.get("node") if breakpoint_info else ""

    # 재개
    workflow.resume_from_breakpoint()

    # WebSocket으로 브로드캐스트
    await websocket_manager.send_breakpoint_resumed(
        workflow_id=workflow_id,
        node_name=node_name,
    )

    return BreakpointActionResponse(
        success=True,
        message="실행이 재개되었습니다",
    )


@router.post("/workflow/{workflow_id}/step-over", response_model=BreakpointActionResponse)
async def step_over(
    workflow_id: str,
    request: Request,
):
    """한 노드만 실행하고 다시 멈춤 (디버거 Step Over)"""
    workflow_registry = getattr(request.app.state, "workflow_registry", None)
    if not workflow_registry:
        raise HTTPException(status_code=404, detail="Workflow registry not found")

    workflow = workflow_registry.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if not workflow.is_paused_at_breakpoint():
        raise HTTPException(status_code=400, detail="Workflow is not paused at breakpoint")

    workflow.step_over()

    return BreakpointActionResponse(
        success=True,
        message="Step over 실행 중",
    )
