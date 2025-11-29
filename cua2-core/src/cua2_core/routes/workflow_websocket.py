"""
워크플로우 WebSocket 라우트 - 실시간 실행 상태 전송
"""

import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from cua2_core.workflows import get_workflow_registry


router = APIRouter()


# 연결된 클라이언트들 (execution_id -> WebSocket Set)
_workflow_connections: Dict[str, Set[WebSocket]] = {}


class WorkflowWebSocketManager:
    """워크플로우 WebSocket 연결 관리"""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, execution_id: str):
        """클라이언트 연결"""
        await websocket.accept()
        if execution_id not in self.connections:
            self.connections[execution_id] = set()
        self.connections[execution_id].add(websocket)
        print(f"워크플로우 WebSocket 연결: {execution_id}")

    def disconnect(self, websocket: WebSocket, execution_id: str):
        """클라이언트 연결 해제"""
        if execution_id in self.connections:
            self.connections[execution_id].discard(websocket)
            if not self.connections[execution_id]:
                del self.connections[execution_id]
        print(f"워크플로우 WebSocket 연결 해제: {execution_id}")

    async def broadcast(self, execution_id: str, message: dict):
        """특정 실행에 연결된 모든 클라이언트에게 메시지 전송"""
        if execution_id not in self.connections:
            return

        disconnected = set()
        for websocket in self.connections[execution_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"WebSocket 전송 오류: {e}")
                disconnected.add(websocket)

        # 끊어진 연결 정리
        for ws in disconnected:
            self.disconnect(ws, execution_id)


# 전역 매니저 인스턴스
workflow_ws_manager = WorkflowWebSocketManager()


def get_workflow_ws_manager() -> WorkflowWebSocketManager:
    """워크플로우 WebSocket 매니저 가져오기"""
    return workflow_ws_manager


@router.websocket("/ws/workflow/{execution_id}")
async def workflow_websocket_endpoint(websocket: WebSocket, execution_id: str):
    """
    워크플로우 실행 상태 실시간 WebSocket

    클라이언트가 연결하면 해당 execution_id의 상태를 주기적으로 전송
    """
    await workflow_ws_manager.connect(websocket, execution_id)

    registry = get_workflow_registry()
    last_step_count = 0
    wait_count = 0
    max_wait = 30  # 최대 30번 대기 (약 15초)

    try:
        while True:
            # 실행 상태 조회
            state = registry.get_execution_state(execution_id)

            if not state:
                # 워크플로우 시작 대기 (아직 인스턴스가 생성되지 않은 경우)
                wait_count += 1
                if wait_count <= max_wait:
                    await websocket.send_json({
                        "type": "waiting",
                        "message": f"워크플로우 시작 대기 중... ({wait_count}/{max_wait})",
                        "execution_id": execution_id,
                    })
                    await asyncio.sleep(0.5)
                    continue
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"실행 상태를 찾을 수 없음: {execution_id}",
                    })
                    break

            # 상태를 찾으면 대기 카운터 초기화
            wait_count = 0

            # 현재 노드의 VLM 스텝 로그
            current_node = state.get("current_node")
            all_steps = []
            node_logs = state.get("node_logs", {})

            for node_id in state.get("completed_nodes", []) + ([current_node] if current_node else []):
                if node_id and node_id in node_logs:
                    for step in node_logs[node_id]:
                        step_with_node = {**step, "node_id": node_id}
                        all_steps.append(step_with_node)

            # 스크린샷은 마지막 스텝에서만
            from cua2_core.routes.workflow_routes import _active_agent_runners
            last_screenshot = None
            if execution_id in _active_agent_runners:
                last_screenshot = _active_agent_runners[execution_id].get_last_screenshot()

            # 노드별 trace 다운로드 URL 생성
            node_traces = {}
            for node_id in state.get("completed_nodes", []):
                node_traces[node_id] = {
                    "download_url": f"/api/traces/node/{execution_id}/{node_id}/download",
                    "view_url": f"/api/traces/node/{execution_id}/{node_id}",
                }

            # 상태 메시지 전송
            status_message = {
                "type": "status",
                "execution_id": execution_id,
                "workflow_id": state.get("workflow_id"),
                "status": state.get("status"),
                "current_node": current_node,
                "completed_nodes": state.get("completed_nodes", []),
                "failed_nodes": state.get("failed_nodes", []),
                "data": state.get("data", {}),
                "error": state.get("error"),
                "start_time": state.get("start_time"),
                "end_time": state.get("end_time"),
                "last_screenshot": last_screenshot,
                "all_steps": all_steps,
                # 노드별 trace 다운로드 URL
                "node_traces": node_traces,
            }

            # 디버깅: 상태 로깅
            print(f"[WS] Sending status: current_node={current_node}, completed={state.get('completed_nodes', [])}, status={state.get('status')}")

            await websocket.send_json(status_message)

            # 새 스텝이 추가되면 별도 이벤트 전송
            if len(all_steps) > last_step_count:
                new_steps = all_steps[last_step_count:]
                for step in new_steps:
                    await websocket.send_json({
                        "type": "step",
                        "step": step,
                    })
                last_step_count = len(all_steps)

            # 완료/실패/중지 시 종료
            if state.get("status") in ["completed", "failed", "stopped"]:
                await websocket.send_json({
                    "type": "complete",
                    "execution_id": execution_id,
                    "status": state.get("status"),
                    "error": state.get("error"),
                })
                break

            # 500ms 간격으로 폴링 (서버에서 push 방식보다 간단)
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print(f"워크플로우 WebSocket 정상 연결 해제: {execution_id}")
    except Exception as e:
        print(f"워크플로우 WebSocket 오류: {e}")
    finally:
        workflow_ws_manager.disconnect(websocket, execution_id)


@router.websocket("/ws/workflows")
async def workflows_list_websocket_endpoint(websocket: WebSocket):
    """
    실행 중인 워크플로우 목록 실시간 WebSocket
    """
    await websocket.accept()

    registry = get_workflow_registry()

    try:
        while True:
            # 실행 중인 워크플로우 목록
            running = registry.list_running_workflows()

            await websocket.send_json({
                "type": "running_list",
                "executions": running,
                "count": len(running),
            })

            # 1초 간격으로 업데이트
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("워크플로우 목록 WebSocket 연결 해제")
    except Exception as e:
        print(f"워크플로우 목록 WebSocket 오류: {e}")
