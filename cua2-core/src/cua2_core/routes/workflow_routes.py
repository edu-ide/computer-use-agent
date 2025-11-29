"""
워크플로우 API 라우트 - LangGraph 기반
"""

from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cua2_core.workflows import get_workflow_registry
from cua2_core.services.vlm_agent_runner import VLMAgentRunner


router = APIRouter(prefix="/api/workflows", tags=["workflows"])


# 전역 VLM 에이전트 실행기 관리
_active_agent_runners: Dict[str, VLMAgentRunner] = {}


class StartWorkflowRequest(BaseModel):
    """워크플로우 시작 요청"""
    parameters: Dict[str, Any]
    execution_id: Optional[str] = None
    use_vlm_agent: bool = True  # VLM 에이전트 사용 여부
    model_id: str = "local-qwen3-vl"


# === 워크플로우 정의 조회 ===

@router.get("")
async def list_workflows():
    """등록된 모든 워크플로우 목록"""
    registry = get_workflow_registry()
    workflows = registry.list_workflows()

    return {
        "workflows": workflows,
        "count": len(workflows),
    }


@router.get("/{workflow_id}")
async def get_workflow_detail(workflow_id: str):
    """워크플로우 상세 정보 (노드, 엣지 포함)"""
    registry = get_workflow_registry()
    detail = registry.get_workflow_detail(workflow_id)

    if not detail:
        raise HTTPException(status_code=404, detail=f"워크플로우를 찾을 수 없음: {workflow_id}")

    return detail


# === 워크플로우 실행 ===

@router.post("/{workflow_id}/start")
async def start_workflow(workflow_id: str, request: StartWorkflowRequest):
    """워크플로우 실행 시작"""
    registry = get_workflow_registry()

    workflow_class = registry.get_workflow_class(workflow_id)
    if not workflow_class:
        raise HTTPException(status_code=404, detail=f"워크플로우를 찾을 수 없음: {workflow_id}")

    # 파라미터 검증 (시작 전에 미리 체크)
    temp_instance = workflow_class()
    validation_errors = temp_instance.validate_parameters(request.parameters)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "파라미터 검증 실패",
                "errors": validation_errors,
            }
        )

    try:
        agent_runner = None

        # VLM 에이전트 사용 시 초기화
        if request.use_vlm_agent:
            # 임시 execution_id 생성 (실제 ID는 registry에서 생성됨)
            from datetime import datetime
            temp_id = request.execution_id or f"{workflow_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            agent_runner = VLMAgentRunner(
                model_id=request.model_id,
                max_steps=15,
                data_dir=f"/tmp/workflow_agent/{temp_id}",
            )

            # 에이전트 초기화
            initialized = await agent_runner.initialize(temp_id)
            if not initialized:
                raise HTTPException(
                    status_code=500,
                    detail="VLM 에이전트 초기화 실패 - 샌드박스를 획득할 수 없습니다"
                )

            # 활성 에이전트 등록
            _active_agent_runners[temp_id] = agent_runner

        execution_id = await registry.start_workflow(
            workflow_id=workflow_id,
            parameters=request.parameters,
            execution_id=request.execution_id,
            agent_runner=agent_runner,
        )

        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "started",
            "parameters": request.parameters,
            "vlm_agent_enabled": request.use_vlm_agent,
        }
    except HTTPException:
        raise
    except Exception as e:
        # 에러 발생 시 에이전트 정리
        if request.execution_id and request.execution_id in _active_agent_runners:
            await _active_agent_runners[request.execution_id].cleanup()
            del _active_agent_runners[request.execution_id]
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/executions/{execution_id}/stop")
async def stop_workflow(execution_id: str):
    """워크플로우 실행 중지"""
    registry = get_workflow_registry()

    # VLM 에이전트 중지
    if execution_id in _active_agent_runners:
        _active_agent_runners[execution_id].stop()

    if not registry.stop_workflow(execution_id):
        raise HTTPException(status_code=404, detail=f"실행 중인 워크플로우를 찾을 수 없음: {execution_id}")

    # 에이전트 정리
    if execution_id in _active_agent_runners:
        await _active_agent_runners[execution_id].cleanup()
        del _active_agent_runners[execution_id]

    return {
        "execution_id": execution_id,
        "status": "stop_requested",
    }


@router.get("/executions/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """워크플로우 실행 상태 조회"""
    registry = get_workflow_registry()
    state = registry.get_execution_state(execution_id)

    if not state:
        raise HTTPException(status_code=404, detail=f"실행 상태를 찾을 수 없음: {execution_id}")

    # 마지막 스크린샷 추가
    last_screenshot = None
    if execution_id in _active_agent_runners:
        last_screenshot = _active_agent_runners[execution_id].get_last_screenshot()

    # 현재 노드의 VLM 스텝 로그
    current_node = state.get("current_node")
    current_steps = []
    if current_node:
        current_steps = state.get("node_logs", {}).get(current_node, [])

    # 모든 노드의 VLM 스텝 로그 (실시간 표시용)
    all_steps = []
    node_logs = state.get("node_logs", {})
    for node_id in state.get("completed_nodes", []) + ([current_node] if current_node else []):
        if node_id and node_id in node_logs:
            for step in node_logs[node_id]:
                step_with_node = {**step, "node_id": node_id}
                all_steps.append(step_with_node)

    return {
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
        "current_steps": current_steps,
        "all_steps": all_steps,
    }


@router.get("/executions")
async def list_running_workflows():
    """실행 중인 워크플로우 목록"""
    registry = get_workflow_registry()
    running = registry.list_running_workflows()

    return {
        "executions": running,
        "count": len(running),
    }


@router.get("/{workflow_id}/history")
async def get_workflow_history(
    workflow_id: str,
    limit: int = Query(10, ge=1, le=100, description="최대 결과 수"),
):
    """워크플로우 실행 히스토리"""
    registry = get_workflow_registry()
    history = registry.get_execution_history(workflow_id, limit=limit)

    return {
        "workflow_id": workflow_id,
        "history": history,
        "count": len(history),
    }


@router.get("/executions/{execution_id}/nodes/{node_id}/logs")
async def get_node_execution_logs(execution_id: str, node_id: str):
    """노드 실행 로그 조회 (VLM 스텝 정보)"""
    registry = get_workflow_registry()
    state = registry.get_execution_state(execution_id)

    if not state:
        raise HTTPException(status_code=404, detail=f"실행 상태를 찾을 수 없음: {execution_id}")

    node_logs = state.get("node_logs", {}).get(node_id, [])
    node_result = state.get("node_results", {}).get(node_id, {})

    # 워크플로우 전체 데이터에서 해당 노드 관련 데이터 추출
    workflow_data = state.get("data", {})

    # 노드별 결과 데이터 (예: 수집된 상품 정보)
    node_data = node_result.get("data", {})

    return {
        "execution_id": execution_id,
        "node_id": node_id,
        "status": "success" if node_result.get("success") else "failed",
        "error": node_result.get("error"),
        "logs": node_logs,
        "log_count": len(node_logs),
        "data": node_data,  # 노드 결과 데이터
        "workflow_data": workflow_data,  # 워크플로우 전체 데이터
    }


@router.get("/executions/{execution_id}/screenshot")
async def get_current_screenshot(execution_id: str):
    """현재 스크린샷 조회"""
    if execution_id not in _active_agent_runners:
        raise HTTPException(status_code=404, detail=f"활성 에이전트를 찾을 수 없음: {execution_id}")

    screenshot = _active_agent_runners[execution_id].get_last_screenshot()

    if not screenshot:
        raise HTTPException(status_code=404, detail="스크린샷을 찾을 수 없음")

    return {
        "execution_id": execution_id,
        "screenshot": screenshot,
    }


@router.get("/executions/{execution_id}/report")
async def get_execution_report(execution_id: str):
    """
    워크플로우 실행 리포트 조회

    완료된 워크플로우의 실행 결과 요약, 에러 목록, 권장사항을 반환합니다.
    """
    registry = get_workflow_registry()
    state = registry.get_execution_state(execution_id)

    if not state:
        raise HTTPException(status_code=404, detail=f"실행 상태를 찾을 수 없음: {execution_id}")

    # 상태에서 리포트 추출
    report = state.get("report")

    if not report:
        # 리포트가 없으면 기본 정보로 생성
        completed = state.get("completed_nodes", [])
        failed = state.get("failed_nodes", [])

        report = {
            "workflow_id": state.get("workflow_id"),
            "execution_id": execution_id,
            "status": state.get("status"),
            "start_time": state.get("start_time"),
            "end_time": state.get("end_time"),
            "total_nodes": len(completed) + len(failed),
            "completed_nodes": len(completed),
            "failed_nodes": len(failed),
            "skipped_nodes": 0,
            "total_cost": 0.0,
            "summary": f"워크플로우 {state.get('status')}: {len(completed)}개 노드 완료, {len(failed)}개 실패",
            "errors": [state.get("error")] if state.get("error") else [],
            "recommendations": [],
            "node_records": [],
        }

    return {
        "execution_id": execution_id,
        "report": report,
    }


# 앱 종료 시 정리 함수
async def cleanup_all_agents():
    """모든 활성 에이전트 정리"""
    for execution_id, runner in list(_active_agent_runners.items()):
        try:
            await runner.cleanup()
        except Exception as e:
            print(f"에이전트 정리 오류 ({execution_id}): {e}")
    _active_agent_runners.clear()
