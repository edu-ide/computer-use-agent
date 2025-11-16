from datetime import datetime

# Get services from app state
from cua2_core.models.models import (
    AvailableModelsResponse,
    GenerateInstructionResponse,
    HealthResponse,
    UpdateStepRequest,
    UpdateStepResponse,
    UpdateTraceEvaluationRequest,
    UpdateTraceEvaluationResponse,
)
from cua2_core.services.agent_service import AgentService
from cua2_core.services.agent_utils.get_model import AVAILABLE_MODELS
from cua2_core.services.instruction_service import InstructionService
from cua2_core.websocket.websocket_manager import WebSocketManager
from fastapi import APIRouter, Depends, HTTPException, Request

# Create router
router = APIRouter()


def get_websocket_manager(request: Request) -> WebSocketManager:
    """Dependency to get WebSocket manager from app state"""
    return request.app.state.websocket_manager


def get_agent_service(request: Request) -> AgentService:
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
    agent_service: AgentService = Depends(get_agent_service),
):
    """Update a specific step in a trace (e.g., update step evaluation)"""
    try:
        agent_service.update_trace_step(
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
    agent_service: AgentService = Depends(get_agent_service),
):
    """Update the user evaluation for a trace (overall task feedback)"""
    try:
        agent_service.update_trace_evaluation(
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
