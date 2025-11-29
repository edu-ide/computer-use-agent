"""
트레이스 API 라우트 - 워크플로우 실행 기록 및 피드백 관리
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cua2_core.services.trace_db_service import (
    get_trace_db,
    WorkflowTrace,
    TraceStep,
)
from cua2_core.services.trace_store import get_trace_store


router = APIRouter(prefix="/api/traces", tags=["traces"])


# === Request/Response Models ===

class SaveTraceRequest(BaseModel):
    """트레이스 저장 요청"""
    execution_id: str
    workflow_id: str
    instruction: Optional[str] = None
    model_id: str = "vlm-agent"
    status: str = "completed"
    final_state: Optional[str] = None
    error_message: Optional[str] = None
    error_cause: Optional[str] = None
    user_evaluation: str = "not_evaluated"
    evaluation_reason: Optional[str] = None
    steps_count: int = 0
    max_steps: int = 15
    duration_seconds: float = 0.0
    start_time: str = ""
    end_time: Optional[str] = None
    steps: List[Dict[str, Any]] = []


class UpdateEvaluationRequest(BaseModel):
    """평가 업데이트 요청"""
    user_evaluation: str  # success, failed, not_evaluated
    evaluation_reason: Optional[str] = None


class UpdateStepEvaluationRequest(BaseModel):
    """스텝 평가 업데이트 요청"""
    evaluation: str  # like, dislike, neutral


class CompleteTraceRequest(BaseModel):
    """트레이스 완료 요청"""
    status: str  # completed, failed, stopped
    final_state: Optional[str] = None
    error_message: Optional[str] = None
    error_cause: Optional[str] = None
    duration_seconds: float = 0.0


# === API Endpoints ===

@router.post("")
async def save_trace(request: SaveTraceRequest):
    """
    트레이스 저장

    워크플로우 실행 완료 후 전체 트레이스를 저장합니다.
    스텝 정보도 함께 저장됩니다.
    """
    db = get_trace_db()

    # trace_id 생성 (execution_id 기반)
    trace_id = f"trace-{request.execution_id}"

    # 트레이스 생성
    trace = WorkflowTrace(
        trace_id=trace_id,
        execution_id=request.execution_id,
        workflow_id=request.workflow_id,
        instruction=request.instruction,
        model_id=request.model_id,
        status=request.status,
        final_state=request.final_state,
        error_message=request.error_message,
        error_cause=request.error_cause,
        user_evaluation=request.user_evaluation,
        evaluation_reason=request.evaluation_reason,
        steps_count=request.steps_count or len(request.steps),
        max_steps=request.max_steps,
        duration_seconds=request.duration_seconds,
        start_time=request.start_time,
        end_time=request.end_time,
    )

    # 트레이스 저장
    if not db.save_trace(trace):
        raise HTTPException(status_code=500, detail="트레이스 저장 실패")

    # 스텝 저장
    for step_data in request.steps:
        step = TraceStep(
            step_id=step_data.get("step_id") or f"step-{step_data.get('step_number', 0)}",
            step_number=step_data.get("step_number", 0),
            screenshot=step_data.get("image") or step_data.get("screenshot"),
            thought=step_data.get("thought"),
            action=step_data.get("action"),
            observation=step_data.get("observation"),
            error=step_data.get("error"),
            tool_calls=step_data.get("tool_calls") or step_data.get("actions") or [],
            evaluation=step_data.get("step_evaluation") or step_data.get("evaluation") or "neutral",
            timestamp=step_data.get("timestamp") or "",
        )
        db.save_step(trace_id, step)

    return {
        "trace_id": trace_id,
        "execution_id": request.execution_id,
        "status": "saved",
        "steps_saved": len(request.steps),
    }


@router.get("")
async def list_traces(
    workflow_id: Optional[str] = Query(None, description="워크플로우 ID 필터"),
    status: Optional[str] = Query(None, description="상태 필터"),
    user_evaluation: Optional[str] = Query(None, description="평가 필터"),
    limit: int = Query(50, ge=1, le=200, description="최대 결과 수"),
    offset: int = Query(0, ge=0, description="오프셋"),
):
    """트레이스 목록 조회"""
    db = get_trace_db()
    traces = db.get_traces(
        workflow_id=workflow_id,
        status=status,
        user_evaluation=user_evaluation,
        limit=limit,
        offset=offset,
    )

    return {
        "traces": traces,
        "count": len(traces),
        "limit": limit,
        "offset": offset,
    }


@router.get("/stats")
async def get_stats():
    """트레이스 통계 조회"""
    db = get_trace_db()
    return db.get_stats()


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    """트레이스 상세 조회"""
    db = get_trace_db()
    trace = db.get_trace(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    steps = db.get_trace_steps(trace_id)

    return {
        "trace": trace,
        "steps": steps,
    }


@router.get("/{trace_id}/steps")
async def get_trace_steps(trace_id: str):
    """트레이스 스텝 조회"""
    db = get_trace_db()
    steps = db.get_trace_steps(trace_id)

    return {
        "trace_id": trace_id,
        "steps": steps,
        "count": len(steps),
    }


@router.put("/{trace_id}/evaluation")
async def update_evaluation(trace_id: str, request: UpdateEvaluationRequest):
    """
    트레이스 평가 업데이트

    사용자가 성공/실패로 평가하고 이유를 입력할 수 있습니다.
    """
    db = get_trace_db()

    # 트레이스 존재 확인
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    if not db.update_trace_evaluation(
        trace_id,
        request.user_evaluation,
        request.evaluation_reason
    ):
        raise HTTPException(status_code=500, detail="평가 업데이트 실패")

    return {
        "trace_id": trace_id,
        "user_evaluation": request.user_evaluation,
        "evaluation_reason": request.evaluation_reason,
        "status": "updated",
    }


@router.put("/{trace_id}/steps/{step_id}/evaluation")
async def update_step_evaluation(
    trace_id: str,
    step_id: str,
    request: UpdateStepEvaluationRequest
):
    """스텝 평가 업데이트"""
    db = get_trace_db()

    if not db.update_step_evaluation(trace_id, step_id, request.evaluation):
        raise HTTPException(status_code=500, detail="스텝 평가 업데이트 실패")

    return {
        "trace_id": trace_id,
        "step_id": step_id,
        "evaluation": request.evaluation,
        "status": "updated",
    }


@router.post("/{trace_id}/complete")
async def complete_trace(trace_id: str, request: CompleteTraceRequest):
    """트레이스 완료 처리"""
    db = get_trace_db()

    if not db.complete_trace(
        trace_id,
        request.status,
        request.final_state,
        request.error_message,
        request.error_cause,
        request.duration_seconds
    ):
        raise HTTPException(status_code=500, detail="트레이스 완료 처리 실패")

    return {
        "trace_id": trace_id,
        "status": request.status,
        "final_state": request.final_state,
    }


@router.get("/{trace_id}/export")
async def export_trace(trace_id: str):
    """트레이스 JSON 내보내기"""
    db = get_trace_db()
    json_data = db.export_trace_json(trace_id)

    if not json_data:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    return {
        "trace_id": trace_id,
        "json": json_data,
    }


@router.delete("/{trace_id}")
async def delete_trace(trace_id: str):
    """트레이스 삭제"""
    db = get_trace_db()

    if not db.delete_trace(trace_id):
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    return {
        "trace_id": trace_id,
        "status": "deleted",
    }


# === execution_id 기반 조회 ===

@router.get("/by-execution/{execution_id}")
async def get_trace_by_execution(execution_id: str):
    """실행 ID로 트레이스 조회"""
    db = get_trace_db()
    trace_id = f"trace-{execution_id}"
    trace = db.get_trace(trace_id)

    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    steps = db.get_trace_steps(trace_id)

    return {
        "trace": trace,
        "steps": steps,
    }


# === 노드별 Trace 캐시 API (재사용용) ===

class SaveNodeTraceRequest(BaseModel):
    """노드 Trace 저장 요청"""
    workflow_id: str
    node_id: str
    success: bool
    steps: List[Dict[str, Any]]
    data: Dict[str, Any] = {}
    params: Dict[str, Any] = {}  # 캐시 키 생성용 파라미터
    key_params: List[str] = []  # 캐시 키에 사용할 파라미터 목록


@router.post("/node-cache")
async def save_node_trace(request: SaveNodeTraceRequest):
    """
    노드 Trace 캐시 저장

    노드 실행 결과를 캐시에 저장하여 재사용 가능하게 합니다.
    성공한 trace만 재사용 가능합니다.
    """
    store = get_trace_store()

    # 캐시 키 생성
    cache_key = store.generate_cache_key(
        request.workflow_id,
        request.node_id,
        request.params,
        request.key_params if request.key_params else None,
    )

    # Trace 저장
    trace = store.save_trace(
        workflow_id=request.workflow_id,
        node_id=request.node_id,
        cache_key=cache_key,
        success=request.success,
        steps=request.steps,
        data=request.data,
    )

    return {
        "cache_key": cache_key,
        "saved": True,
        "reusable": trace.success,
    }


@router.get("/node-cache")
async def list_node_traces(
    workflow_id: str = Query(None, description="워크플로우 ID로 필터링"),
    node_id: str = Query(None, description="노드 ID로 필터링"),
    success_only: bool = Query(True, description="성공한 것만 조회"),
):
    """
    노드 Trace 캐시 목록 조회

    캐시된 노드 trace 목록을 조회합니다.
    """
    store = get_trace_store()
    traces = store.list_traces(
        workflow_id=workflow_id,
        node_id=node_id,
        success_only=success_only,
    )

    return {
        "traces": [
            {
                "cache_key": t.cache_key,
                "workflow_id": t.workflow_id,
                "node_id": t.node_id,
                "success": t.success,
                "steps_count": len(t.steps),
                "created_at": t.created_at,
                "used_count": t.used_count,
                "last_used_at": t.last_used_at,
            }
            for t in traces
        ],
        "count": len(traces),
    }


@router.get("/node-cache/lookup")
async def lookup_reusable_node_trace(
    workflow_id: str = Query(..., description="워크플로우 ID"),
    node_id: str = Query(..., description="노드 ID"),
    params: str = Query("{}", description="파라미터 (JSON 문자열)"),
    key_params: str = Query("", description="캐시 키 파라미터 (쉼표 구분)"),
):
    """
    재사용 가능한 노드 Trace 조회

    동일한 파라미터로 성공한 이전 trace를 찾습니다.
    이 API를 사용하여 노드 실행 전에 재사용 가능한 trace가 있는지 확인합니다.
    """
    import json

    store = get_trace_store()

    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="잘못된 params JSON 형식")

    key_params_list = [p.strip() for p in key_params.split(",") if p.strip()] if key_params else None

    trace = store.get_reusable_trace(
        workflow_id=workflow_id,
        node_id=node_id,
        params=params_dict,
        key_params=key_params_list,
    )

    if not trace:
        return {
            "found": False,
            "trace": None,
        }

    return {
        "found": True,
        "cache_key": trace.cache_key,
        "node_id": trace.node_id,
        "steps": trace.steps,  # 재사용 시 스텝 포함
        "data": trace.data,
        "used_count": trace.used_count,
    }


@router.delete("/node-cache/{cache_key}")
async def delete_node_trace(cache_key: str):
    """노드 Trace 캐시 삭제"""
    store = get_trace_store()

    if not store.delete_trace(cache_key):
        raise HTTPException(status_code=404, detail="캐시된 Trace를 찾을 수 없음")

    return {
        "deleted": True,
        "cache_key": cache_key,
    }


@router.delete("/node-cache")
async def clear_node_traces():
    """모든 노드 Trace 캐시 삭제"""
    store = get_trace_store()
    store.clear_all()

    return {
        "cleared": True,
    }


# === 노드별 Trace 다운로드 API ===

@router.get("/node/{execution_id}/{node_id}/download")
async def download_node_trace(execution_id: str, node_id: str):
    """
    노드 실행 Trace JSON 다운로드

    각 VLM 노드 실행 후 trace.json 파일을 다운로드할 수 있습니다.
    """
    from fastapi.responses import JSONResponse
    import json

    db = get_trace_db()
    trace_id = f"trace-{execution_id}"

    # 트레이스 존재 확인
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    # 해당 노드의 스텝만 필터링
    all_steps = db.get_trace_steps(trace_id)
    node_steps = [
        s for s in all_steps
        if s.get("node_id") == node_id or node_id in str(s.get("step_id", ""))
    ]

    # 노드 trace 데이터 구성
    node_trace = {
        "execution_id": execution_id,
        "trace_id": trace_id,
        "node_id": node_id,
        "workflow_id": trace.get("workflow_id"),
        "steps": node_steps,
        "steps_count": len(node_steps),
        "status": trace.get("status"),
        "created_at": trace.get("start_time"),
    }

    # JSON 응답 (다운로드용 헤더)
    return JSONResponse(
        content=node_trace,
        headers={
            "Content-Disposition": f'attachment; filename="{execution_id}_{node_id}_trace.json"',
            "Content-Type": "application/json",
        }
    )


@router.get("/node/{execution_id}/{node_id}")
async def get_node_trace(execution_id: str, node_id: str):
    """
    노드 실행 Trace 조회

    특정 노드의 실행 trace를 조회합니다.
    """
    db = get_trace_db()
    trace_id = f"trace-{execution_id}"

    # 트레이스 존재 확인
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    # 해당 노드의 스텝만 필터링
    all_steps = db.get_trace_steps(trace_id)
    node_steps = [
        s for s in all_steps
        if s.get("node_id") == node_id or node_id in str(s.get("step_id", ""))
    ]

    # 다운로드 URL
    download_url = f"/api/traces/node/{execution_id}/{node_id}/download"

    return {
        "execution_id": execution_id,
        "node_id": node_id,
        "steps": node_steps,
        "steps_count": len(node_steps),
        "download_url": download_url,
    }


@router.get("/execution/{execution_id}/nodes")
async def list_execution_nodes(execution_id: str):
    """
    실행의 모든 노드 Trace 목록

    워크플로우 실행에서 각 노드별 trace 정보를 조회합니다.
    각 노드의 다운로드 URL도 포함됩니다.
    """
    db = get_trace_db()
    trace_id = f"trace-{execution_id}"

    # 트레이스 존재 확인
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="트레이스를 찾을 수 없음")

    # 모든 스텝 조회
    all_steps = db.get_trace_steps(trace_id)

    # 노드별로 그룹화
    nodes: Dict[str, List[Dict[str, Any]]] = {}
    for step in all_steps:
        node_id = step.get("node_id", "unknown")
        if node_id not in nodes:
            nodes[node_id] = []
        nodes[node_id].append(step)

    # 노드별 요약
    node_summaries = []
    for node_id, steps in nodes.items():
        node_summaries.append({
            "node_id": node_id,
            "steps_count": len(steps),
            "has_error": any(s.get("error") for s in steps),
            "download_url": f"/api/traces/node/{execution_id}/{node_id}/download",
            "view_url": f"/api/traces/node/{execution_id}/{node_id}",
        })

    return {
        "execution_id": execution_id,
        "trace_id": trace_id,
        "workflow_id": trace.get("workflow_id"),
        "nodes": node_summaries,
        "total_nodes": len(node_summaries),
    }