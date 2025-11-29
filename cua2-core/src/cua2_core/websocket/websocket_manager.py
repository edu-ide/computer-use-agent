import asyncio
import json
from typing import Dict, Literal, Set

from cua2_core.models.models import (
    ActiveTask,
    AgentCompleteEvent,
    AgentErrorEvent,
    AgentProgressEvent,
    AgentStartEvent,
    AgentStep,
    AgentTrace,
    AgentTraceMetadata,
    VncUrlSetEvent,
    VncUrlUnsetEvent,
    WebSocketEvent,
    ConfirmationRequiredEvent,
    ConfirmationReceivedEvent,
    WorkflowStateUpdateEvent,
    BreakpointHitEvent,
    BreakpointResumedEvent,
)
from fastapi import WebSocket
from typing import Any


class WebSocketException(Exception):
    """Exception for WebSocket errors"""

    pass


class WebSocketManager:
    """Manages WebSocket connections and broadcasting"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        if websocket in self.connection_tasks:
            self.connection_tasks[websocket].cancel()
            del self.connection_tasks[websocket]
        print(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_message(self, message: WebSocketEvent, websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(
                json.dumps(
                    message.model_dump(
                        mode="json",
                        context={"actions_as_json": True, "image_as_path": False},
                    )
                )
            )
        except Exception as e:
            print(f"Error sending personal message: {e}")
            # Only disconnect if the connection is still in our set
            if websocket in self.active_connections:
                self.disconnect(websocket)
            raise WebSocketException()

    async def send_agent_start(
        self,
        active_task: ActiveTask,
        websocket: WebSocket,
        status: Literal["max_sandboxes_reached", "success"],
    ):
        """Send agent start event"""
        event = AgentStartEvent(
            agentTrace=AgentTrace(
                id=active_task.message_id,
                timestamp=active_task.timestamp,
                instruction=active_task.instruction,
                modelId=active_task.model_id,
                steps=active_task.steps,
                traceMetadata=active_task.traceMetadata,
                isRunning=True,
            ),
            status=status,
        )
        await self.send_message(event, websocket)

    async def send_agent_progress(
        self,
        step: AgentStep,
        metadata: AgentTraceMetadata,
        websocket: WebSocket,
    ):
        """Send agent progress event"""
        event = AgentProgressEvent(
            agentStep=step,
            traceMetadata=metadata,
        )
        await self.send_message(event, websocket)

    async def send_agent_complete(
        self,
        metadata: AgentTraceMetadata,
        websocket: WebSocket,
        final_state: Literal[
            "success", "stopped", "max_steps_reached", "error", "timeout"
        ],
    ):
        """Send agent complete event"""
        event = AgentCompleteEvent(traceMetadata=metadata, final_state=final_state)
        await self.send_message(event, websocket)

    async def send_agent_error(self, error: str, websocket: WebSocket):
        """Send agent error event"""
        event = AgentErrorEvent(error=error)
        await self.send_message(event, websocket)

    async def send_vnc_url_set(self, vnc_url: str, websocket: WebSocket):
        """Send VNC URL set event"""
        event = VncUrlSetEvent(
            vncUrl=vnc_url,
        )
        await self.send_message(event, websocket)

    async def send_vnc_url_unset(self, websocket: WebSocket):
        """Send VNC URL unset event (reset to default display)"""
        event = VncUrlUnsetEvent()
        await self.send_message(event, websocket)

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

    # =========================================
    # Human-in-the-Loop 이벤트 전송
    # =========================================

    async def send_confirmation_required(
        self,
        workflow_id: str,
        node_name: str,
        message: str,
        is_dangerous: bool = False,
        input_type: str = None,
        websocket: WebSocket = None,
    ):
        """
        사용자 확인 요청 이벤트 전송

        Args:
            workflow_id: 워크플로우 ID
            node_name: 확인이 필요한 노드 이름
            message: 확인 메시지
            is_dangerous: 위험한 작업 여부
            input_type: 입력 타입 ("text", "captcha", "2fa")
            websocket: 특정 WebSocket (None이면 브로드캐스트)
        """
        event = ConfirmationRequiredEvent(
            workflow_id=workflow_id,
            node_name=node_name,
            message=message,
            is_dangerous=is_dangerous,
            input_type=input_type,
        )

        if websocket:
            await self.send_message(event, websocket)
        else:
            await self.broadcast(event)

    async def send_confirmation_received(
        self,
        workflow_id: str,
        node_name: str,
        confirmed: bool,
        user_input: str = None,
        websocket: WebSocket = None,
    ):
        """
        사용자 확인 완료 이벤트 전송

        Args:
            workflow_id: 워크플로우 ID
            node_name: 확인된 노드 이름
            confirmed: 확인/취소 여부
            user_input: 사용자 입력 (있는 경우)
            websocket: 특정 WebSocket (None이면 브로드캐스트)
        """
        event = ConfirmationReceivedEvent(
            workflow_id=workflow_id,
            node_name=node_name,
            confirmed=confirmed,
            user_input=user_input,
        )

        if websocket:
            await self.send_message(event, websocket)
        else:
            await self.broadcast(event)

    async def send_workflow_state_update(
        self,
        workflow_id: str,
        execution_id: str,
        status: str,
        current_node: str = None,
        completed_nodes: list = None,
        failed_nodes: list = None,
        progress_percent: int = 0,
        error: str = None,
        websocket: WebSocket = None,
    ):
        """
        워크플로우 상태 업데이트 이벤트 전송

        Args:
            workflow_id: 워크플로우 ID
            execution_id: 실행 ID
            status: 상태 (running, completed, failed, stopped)
            current_node: 현재 실행 중인 노드
            completed_nodes: 완료된 노드 목록
            failed_nodes: 실패한 노드 목록
            progress_percent: 진행률 (0-100)
            error: 에러 메시지
            websocket: 특정 WebSocket (None이면 브로드캐스트)
        """
        event = WorkflowStateUpdateEvent(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=status,
            current_node=current_node,
            completed_nodes=completed_nodes or [],
            failed_nodes=failed_nodes or [],
            progress_percent=progress_percent,
            error=error,
        )

        if websocket:
            await self.send_message(event, websocket)
        else:
            await self.broadcast(event)

    async def send_breakpoint_hit(
        self,
        workflow_id: str,
        node_name: str,
        state: Dict[str, Any] = None,
        websocket: WebSocket = None,
    ):
        """
        브레이크포인트 도달 이벤트 전송

        Args:
            workflow_id: 워크플로우 ID
            node_name: 브레이크포인트 노드 이름
            state: 현재 상태 (디버깅용)
            websocket: 특정 WebSocket (None이면 브로드캐스트)
        """
        event = BreakpointHitEvent(
            workflow_id=workflow_id,
            node_name=node_name,
            state=state,
        )

        if websocket:
            await self.send_message(event, websocket)
        else:
            await self.broadcast(event)

    async def send_breakpoint_resumed(
        self,
        workflow_id: str,
        node_name: str,
        websocket: WebSocket = None,
    ):
        """
        브레이크포인트 재개 이벤트 전송

        Args:
            workflow_id: 워크플로우 ID
            node_name: 재개된 노드 이름
            websocket: 특정 WebSocket (None이면 브로드캐스트)
        """
        event = BreakpointResumedEvent(
            workflow_id=workflow_id,
            node_name=node_name,
        )

        if websocket:
            await self.send_message(event, websocket)
        else:
            await self.broadcast(event)

    async def broadcast(self, message: WebSocketEvent):
        """
        모든 연결된 클라이언트에 메시지 브로드캐스트

        Args:
            message: 전송할 WebSocket 이벤트
        """
        disconnected = set()

        for websocket in self.active_connections:
            try:
                await websocket.send_text(
                    json.dumps(
                        message.model_dump(
                            mode="json",
                            context={"actions_as_json": True, "image_as_path": False},
                        )
                    )
                )
            except Exception as e:
                print(f"Error broadcasting message: {e}")
                disconnected.add(websocket)

        # 끊어진 연결 정리
        for websocket in disconnected:
            self.disconnect(websocket)
