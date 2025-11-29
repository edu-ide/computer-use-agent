"""
워크플로우 모니터링 모듈

워크플로우 실행 추적 및 리포트 생성:
- 노드 실행 시작/완료 기록
- Stuck 노드 감지
- 실행 리포트 생성
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..agent_activity_log import (
    log_orchestrator,
    ActivityType,
)

from .types import (
    NodeStatus,
    ExecutionStrategy,
    NodeExecutionRecord,
    WorkflowReport,
)

logger = logging.getLogger(__name__)


class WorkflowMonitor:
    """
    워크플로우 모니터

    워크플로우 실행 상태를 추적하고 리포트를 생성합니다.
    """

    # 노드별 타임아웃 설정 (초)
    NODE_TIMEOUTS = {
        "default": 120,  # 기본 2분
        "open_url": 30,
        "search": 60,
        "navigate": 60,
        "extract": 180,  # 데이터 추출은 더 오래 걸릴 수 있음
        "analyze": 180,
    }

    def __init__(self):
        # 워크플로우 실행 추적
        self._workflow_executions: Dict[str, Dict[str, Any]] = {}

    def start_workflow_tracking(
        self,
        workflow_id: str,
        execution_id: str,
        total_nodes: int,
    ):
        """워크플로우 실행 추적 시작"""
        self._workflow_executions[execution_id] = {
            "workflow_id": workflow_id,
            "start_time": time.time(),
            "total_nodes": total_nodes,
            "node_records": [],
            "total_cost": 0.0,
            "errors": [],
        }

        log_orchestrator(
            ActivityType.INFO,
            f"워크플로우 시작: {workflow_id}",
            details={"total_nodes": total_nodes},
            execution_id=execution_id,
        )

    def record_node_start(
        self,
        execution_id: str,
        node_id: str,
        strategy: ExecutionStrategy,
    ):
        """노드 실행 시작 기록"""
        if execution_id not in self._workflow_executions:
            return

        record = NodeExecutionRecord(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            strategy=strategy,
            start_time=time.time(),
        )

        # 기존 레코드 업데이트 또는 추가
        exec_data = self._workflow_executions[execution_id]
        existing = next(
            (r for r in exec_data["node_records"] if r.node_id == node_id),
            None
        )
        if existing:
            existing.status = NodeStatus.RUNNING
            existing.start_time = time.time()
        else:
            exec_data["node_records"].append(record)

    def record_node_complete(
        self,
        execution_id: str,
        node_id: str,
        success: bool,
        duration_ms: int,
        cost: float = 0.0,
        error: Optional[str] = None,
        result_summary: str = "",
    ):
        """노드 실행 완료 기록"""
        if execution_id not in self._workflow_executions:
            return

        exec_data = self._workflow_executions[execution_id]
        record = next(
            (r for r in exec_data["node_records"] if r.node_id == node_id),
            None
        )

        if record:
            record.status = NodeStatus.SUCCESS if success else NodeStatus.FAILED
            record.end_time = time.time()
            record.duration_ms = duration_ms
            record.error = error
            record.result_summary = result_summary

        exec_data["total_cost"] += cost
        if error:
            exec_data["errors"].append(f"{node_id}: {error}")

    def check_stuck_node(
        self,
        execution_id: str,
        node_id: str,
    ) -> bool:
        """노드가 stuck 상태인지 확인"""
        if execution_id not in self._workflow_executions:
            return False

        exec_data = self._workflow_executions[execution_id]
        record = next(
            (r for r in exec_data["node_records"] if r.node_id == node_id),
            None
        )

        if record and record.status == NodeStatus.RUNNING:
            elapsed = time.time() - record.start_time
            timeout = self.NODE_TIMEOUTS.get("default", 120)

            if elapsed > timeout:
                logger.warning(f"[WorkflowMonitor] {node_id} stuck 감지: {elapsed:.1f}초 경과")
                return True

        return False

    def get_node_timeout(self, node_id: str, instruction: str) -> int:
        """노드별 타임아웃 시간 반환 (초)"""
        instruction_lower = instruction.lower()

        # 노드 이름이나 instruction에서 타입 추론
        for key, timeout in self.NODE_TIMEOUTS.items():
            if key in node_id.lower() or key in instruction_lower:
                return timeout

        return self.NODE_TIMEOUTS["default"]

    async def generate_report(
        self,
        execution_id: str,
        final_status: str = "completed",
    ) -> WorkflowReport:
        """
        워크플로우 실행 리포트 생성

        Args:
            execution_id: 실행 ID
            final_status: 최종 상태

        Returns:
            WorkflowReport: 실행 리포트
        """
        if execution_id not in self._workflow_executions:
            raise ValueError(f"Unknown execution: {execution_id}")

        exec_data = self._workflow_executions[execution_id]
        end_time = time.time()

        # 통계 계산
        node_records = exec_data["node_records"]
        completed = len([r for r in node_records if r.status == NodeStatus.SUCCESS])
        failed = len([r for r in node_records if r.status == NodeStatus.FAILED])
        skipped = len([r for r in node_records if r.status == NodeStatus.SKIPPED])
        total_duration = int((end_time - exec_data["start_time"]) * 1000)

        # 상태 결정
        if failed > 0 and completed == 0:
            final_status = "failed"
        elif failed > 0:
            final_status = "partial"
        elif completed == exec_data["total_nodes"]:
            final_status = "completed"

        # 요약 생성
        summary = self._generate_summary(node_records, total_duration, exec_data["total_cost"])

        # 권장사항 생성
        recommendations = self._generate_recommendations(node_records, exec_data["errors"])

        report = WorkflowReport(
            workflow_id=exec_data["workflow_id"],
            execution_id=execution_id,
            status=final_status,
            start_time=exec_data["start_time"],
            end_time=end_time,
            total_duration_ms=total_duration,
            total_nodes=exec_data["total_nodes"],
            completed_nodes=completed,
            failed_nodes=failed,
            skipped_nodes=skipped,
            total_cost=exec_data["total_cost"],
            node_records=node_records,
            summary=summary,
            errors=exec_data["errors"],
            recommendations=recommendations,
        )

        # 활동 로그: 리포트 생성
        log_orchestrator(
            ActivityType.INFO,
            f"리포트 생성: {final_status} ({completed}/{exec_data['total_nodes']})",
            details={
                "status": final_status,
                "duration_ms": total_duration,
                "cost": exec_data["total_cost"],
            },
            execution_id=execution_id,
        )

        return report

    def _generate_summary(
        self,
        records: List[NodeExecutionRecord],
        total_duration_ms: int,
        total_cost: float,
    ) -> str:
        """실행 요약 생성"""
        success = len([r for r in records if r.status == NodeStatus.SUCCESS])
        total = len(records)

        duration_sec = total_duration_ms / 1000

        if success == total:
            return f"워크플로우 완료: {total}개 노드 모두 성공 ({duration_sec:.1f}초, ${total_cost:.4f})"
        elif success > 0:
            return f"부분 완료: {success}/{total} 노드 성공 ({duration_sec:.1f}초, ${total_cost:.4f})"
        else:
            return f"실패: 모든 노드 실패 ({duration_sec:.1f}초)"

    def _generate_recommendations(
        self,
        records: List[NodeExecutionRecord],
        errors: List[str],
    ) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # 자주 실패하는 노드 분석
        failed_nodes = [r for r in records if r.status == NodeStatus.FAILED]
        if failed_nodes:
            recommendations.append(
                f"실패한 노드 {len(failed_nodes)}개 검토 필요: "
                + ", ".join(r.node_id for r in failed_nodes[:3])
            )

        # 느린 노드 분석
        slow_nodes = [r for r in records if r.duration_ms > 30000]  # 30초 이상
        if slow_nodes:
            recommendations.append(
                f"느린 노드 {len(slow_nodes)}개 최적화 권장: "
                + ", ".join(f"{r.node_id}({r.duration_ms/1000:.1f}s)" for r in slow_nodes[:3])
            )

        # 타임아웃 분석
        timeout_errors = [e for e in errors if "timeout" in e.lower()]
        if timeout_errors:
            recommendations.append("타임아웃 발생 - 네트워크 상태 또는 페이지 로딩 확인 필요")

        # 재시도가 많은 노드
        retry_nodes = [r for r in records if r.retry_count >= 2]
        if retry_nodes:
            recommendations.append(
                f"재시도가 많은 노드: "
                + ", ".join(f"{r.node_id}({r.retry_count}회)" for r in retry_nodes)
            )

        if not recommendations:
            recommendations.append("모든 노드가 정상 실행되었습니다.")

        return recommendations

    def cleanup_execution(self, execution_id: str):
        """실행 데이터 정리"""
        if execution_id in self._workflow_executions:
            del self._workflow_executions[execution_id]

    def get_execution_data(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """실행 데이터 조회"""
        return self._workflow_executions.get(execution_id)
