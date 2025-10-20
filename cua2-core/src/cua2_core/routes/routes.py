from datetime import datetime

# Get services from app state
from cua2_core.models.models import (
    ActiveTasksResponse,
    HealthResponse,
    TaskStatusResponse,
)
from cua2_core.services.agent_service import AgentService
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


@router.get("/tasks", response_model=ActiveTasksResponse)
async def get_active_tasks(
    agent_service: AgentService = Depends(get_agent_service),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """Get currently active tasks"""
    return ActiveTasksResponse(
        active_tasks=agent_service.get_active_tasks(),
        total_connections=websocket_manager.get_connection_count(),
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str, agent_service: AgentService = Depends(get_agent_service)
):
    """Get status of a specific task"""
    task_status = agent_service.get_task_status(task_id)
    if task_status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(task_id=task_id, status=task_status)
