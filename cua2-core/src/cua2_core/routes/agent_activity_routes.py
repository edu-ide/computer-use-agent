"""
에이전트 활동 로그 API 및 WebSocket 라우트
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from cua2_core.services.agent_activity_log import (
    get_agent_activity_log,
    AgentType,
    AgentActivity,
)


router = APIRouter(prefix="/api/agents", tags=["agents"])


# === REST API ===

@router.get("/activities")
async def get_activities(
    limit: int = Query(50, ge=1, le=100, description="최대 개수"),
    agent_type: Optional[str] = Query(None, description="에이전트 유형 필터"),
    execution_id: Optional[str] = Query(None, description="실행 ID 필터"),
):
    """
    에이전트 활동 로그 조회

    모든 에이전트의 활동 이력을 조회합니다.
    """
    log = get_agent_activity_log()

    # agent_type 변환
    agent_type_enum = None
    if agent_type:
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            pass

    activities = log.get_logs(
        limit=limit,
        agent_type=agent_type_enum,
        execution_id=execution_id,
    )

    return {
        "activities": activities,
        "count": len(activities),
    }


@router.get("/status")
async def get_agent_status():
    """
    에이전트 상태 조회

    각 에이전트의 현재 상태와 최신 활동을 조회합니다.
    """
    log = get_agent_activity_log()
    return {
        "agents": log.get_agent_status(),
    }


@router.get("/latest")
async def get_latest_activities():
    """
    에이전트별 최신 활동 조회

    각 에이전트의 가장 최근 활동만 조회합니다.
    """
    log = get_agent_activity_log()
    return {
        "latest": log.get_latest_by_agent(),
    }


@router.delete("/activities")
async def clear_activities():
    """활동 로그 초기화"""
    log = get_agent_activity_log()
    log.clear()
    return {"cleared": True}


# === WebSocket ===

@router.websocket("/ws/activities")
async def agent_activities_websocket(websocket: WebSocket):
    """
    에이전트 활동 실시간 WebSocket

    새로운 활동이 발생할 때마다 클라이언트에게 전송합니다.
    연결 시 현재 에이전트 상태도 함께 전송합니다.
    """
    await websocket.accept()
    log = get_agent_activity_log()

    # 연결된 클라이언트 추적
    connected = True

    async def on_activity(activity: AgentActivity):
        """새 활동 발생 시 전송"""
        if not connected:
            return
        try:
            await websocket.send_json({
                "type": "activity",
                "activity": activity.to_dict(),
            })
        except Exception:
            pass

    # 비동기 리스너 등록
    log.add_async_listener(on_activity)

    try:
        # 초기 상태 전송
        await websocket.send_json({
            "type": "init",
            "agents": log.get_agent_status(),
            "recent_activities": log.get_logs(limit=20),
        })

        # 주기적으로 에이전트 상태 업데이트 (time_ago 갱신)
        while True:
            await asyncio.sleep(5)  # 5초마다

            # 에이전트 상태 업데이트 전송
            await websocket.send_json({
                "type": "status_update",
                "agents": log.get_agent_status(),
            })

    except WebSocketDisconnect:
        print("[AgentActivityWS] 연결 해제")
    except Exception as e:
        print(f"[AgentActivityWS] 오류: {e}")
    finally:
        connected = False
        log.remove_listener(on_activity)


@router.websocket("/ws/activities/{execution_id}")
async def execution_activities_websocket(websocket: WebSocket, execution_id: str):
    """
    특정 실행의 에이전트 활동 실시간 WebSocket

    특정 워크플로우 실행에 대한 활동만 전송합니다.
    """
    await websocket.accept()
    log = get_agent_activity_log()

    connected = True

    async def on_activity(activity: AgentActivity):
        """해당 실행의 활동만 전송"""
        if not connected:
            return
        if activity.execution_id != execution_id:
            return
        try:
            await websocket.send_json({
                "type": "activity",
                "activity": activity.to_dict(),
            })
        except Exception:
            pass

    log.add_async_listener(on_activity)

    try:
        # 초기: 해당 실행의 활동만
        await websocket.send_json({
            "type": "init",
            "execution_id": execution_id,
            "agents": log.get_agent_status(),
            "recent_activities": log.get_logs(limit=50, execution_id=execution_id),
        })

        # 연결 유지
        while True:
            await asyncio.sleep(5)
            # 상태 업데이트
            await websocket.send_json({
                "type": "status_update",
                "agents": log.get_agent_status(),
            })

    except WebSocketDisconnect:
        print(f"[AgentActivityWS] {execution_id} 연결 해제")
    except Exception as e:
        print(f"[AgentActivityWS] 오류: {e}")
    finally:
        connected = False
        log.remove_listener(on_activity)
