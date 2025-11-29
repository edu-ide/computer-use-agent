"""
트레이스 DB 서비스 - 워크플로우 실행 기록 및 피드백 저장
"""

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    """트레이스 스텝"""
    step_id: str
    step_number: int
    screenshot: Optional[str] = None  # base64 이미지 (저장 시 파일로 분리 가능)
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: str = "neutral"  # like, dislike, neutral
    timestamp: str = ""


@dataclass
class WorkflowTrace:
    """워크플로우 트레이스"""
    trace_id: str
    execution_id: str
    workflow_id: str
    instruction: Optional[str] = None  # 워크플로우 설명 또는 명령
    model_id: str = "vlm-agent"

    # 실행 결과
    status: str = "running"  # running, completed, failed, stopped
    final_state: Optional[str] = None  # success, error, stopped, max_steps_reached
    error_message: Optional[str] = None
    error_cause: Optional[str] = None  # 실패 원인 상세

    # 평가
    user_evaluation: str = "not_evaluated"  # success, failed, not_evaluated
    evaluation_reason: Optional[str] = None  # 평가 이유

    # 메타데이터
    steps_count: int = 0
    max_steps: int = 15
    duration_seconds: float = 0.0

    # 시간
    start_time: str = ""
    end_time: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class TraceDBService:
    """트레이스 DB 관리 서비스"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_dir = os.path.expanduser("~/.cua-coupang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "traces.db")

        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """DB 초기화 및 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            # 트레이스 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    execution_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    instruction TEXT,
                    model_id TEXT DEFAULT 'vlm-agent',

                    status TEXT DEFAULT 'running',
                    final_state TEXT,
                    error_message TEXT,
                    error_cause TEXT,

                    user_evaluation TEXT DEFAULT 'not_evaluated',
                    evaluation_reason TEXT,

                    steps_count INTEGER DEFAULT 0,
                    max_steps INTEGER DEFAULT 15,
                    duration_seconds REAL DEFAULT 0.0,

                    start_time TEXT,
                    end_time TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 스텝 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trace_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    screenshot_path TEXT,
                    thought TEXT,
                    action TEXT,
                    observation TEXT,
                    error TEXT,
                    tool_calls TEXT,
                    evaluation TEXT DEFAULT 'neutral',
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES workflow_traces(trace_id),
                    UNIQUE(trace_id, step_id)
                )
            """)

            # 인덱스
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_execution ON workflow_traces(execution_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_workflow ON workflow_traces(workflow_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_status ON workflow_traces(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_evaluation ON workflow_traces(user_evaluation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_step_trace ON trace_steps(trace_id)")

            conn.commit()

    def save_trace(self, trace: WorkflowTrace) -> bool:
        """트레이스 저장"""
        now = datetime.utcnow().isoformat()
        trace.updated_at = now
        if not trace.created_at:
            trace.created_at = now

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO workflow_traces
                        (trace_id, execution_id, workflow_id, instruction, model_id,
                         status, final_state, error_message, error_cause,
                         user_evaluation, evaluation_reason,
                         steps_count, max_steps, duration_seconds,
                         start_time, end_time, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trace.trace_id,
                        trace.execution_id,
                        trace.workflow_id,
                        trace.instruction,
                        trace.model_id,
                        trace.status,
                        trace.final_state,
                        trace.error_message,
                        trace.error_cause,
                        trace.user_evaluation,
                        trace.evaluation_reason,
                        trace.steps_count,
                        trace.max_steps,
                        trace.duration_seconds,
                        trace.start_time,
                        trace.end_time,
                        trace.created_at,
                        trace.updated_at,
                    ))
                    conn.commit()
                    return True
            except Exception as e:
                print(f"트레이스 저장 오류: {e}")
                return False

    def save_step(self, trace_id: str, step: TraceStep, screenshot_dir: Optional[str] = None) -> bool:
        """스텝 저장 (스크린샷은 별도 파일로 저장 가능)"""
        screenshot_path = None

        # 스크린샷을 파일로 저장 (옵션)
        if step.screenshot and screenshot_dir:
            try:
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, f"{trace_id}_{step.step_id}.png")

                # base64 디코딩 및 저장
                import base64
                img_data = step.screenshot
                if img_data.startswith("data:"):
                    img_data = img_data.split(",", 1)[1]

                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(img_data))
            except Exception as e:
                print(f"스크린샷 저장 오류: {e}")
                screenshot_path = None

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO trace_steps
                        (trace_id, step_id, step_number, screenshot_path, thought, action,
                         observation, error, tool_calls, evaluation, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trace_id,
                        step.step_id,
                        step.step_number,
                        screenshot_path,
                        step.thought,
                        step.action,
                        step.observation,
                        step.error,
                        json.dumps(step.tool_calls) if step.tool_calls else None,
                        step.evaluation,
                        step.timestamp,
                    ))
                    conn.commit()
                    return True
            except Exception as e:
                print(f"스텝 저장 오류: {e}")
                return False

    def update_trace_evaluation(
        self,
        trace_id: str,
        user_evaluation: str,
        evaluation_reason: Optional[str] = None
    ) -> bool:
        """트레이스 평가 업데이트"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE workflow_traces
                        SET user_evaluation = ?, evaluation_reason = ?, updated_at = ?
                        WHERE trace_id = ?
                    """, (user_evaluation, evaluation_reason, datetime.utcnow().isoformat(), trace_id))
                    conn.commit()
                    return conn.total_changes > 0
            except Exception as e:
                print(f"평가 업데이트 오류: {e}")
                return False

    def update_step_evaluation(self, trace_id: str, step_id: str, evaluation: str) -> bool:
        """스텝 평가 업데이트"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE trace_steps
                        SET evaluation = ?
                        WHERE trace_id = ? AND step_id = ?
                    """, (evaluation, trace_id, step_id))
                    conn.commit()
                    return conn.total_changes > 0
            except Exception as e:
                print(f"스텝 평가 업데이트 오류: {e}")
                return False

    def complete_trace(
        self,
        trace_id: str,
        status: str,
        final_state: Optional[str] = None,
        error_message: Optional[str] = None,
        error_cause: Optional[str] = None,
        duration_seconds: float = 0.0
    ) -> bool:
        """트레이스 완료 처리"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 스텝 수 계산
                    steps_count = conn.execute(
                        "SELECT COUNT(*) FROM trace_steps WHERE trace_id = ?",
                        (trace_id,)
                    ).fetchone()[0]

                    conn.execute("""
                        UPDATE workflow_traces
                        SET status = ?, final_state = ?, error_message = ?, error_cause = ?,
                            steps_count = ?, duration_seconds = ?, end_time = ?, updated_at = ?
                        WHERE trace_id = ?
                    """, (
                        status, final_state, error_message, error_cause,
                        steps_count, duration_seconds,
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                        trace_id
                    ))
                    conn.commit()
                    return conn.total_changes > 0
            except Exception as e:
                print(f"트레이스 완료 오류: {e}")
                return False

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """트레이스 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM workflow_traces WHERE trace_id = ?",
                (trace_id,)
            ).fetchone()

            if not row:
                return None

            return dict(row)

    def get_trace_steps(self, trace_id: str) -> List[Dict[str, Any]]:
        """트레이스 스텝 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trace_steps WHERE trace_id = ? ORDER BY step_number",
                (trace_id,)
            ).fetchall()

            steps = []
            for row in rows:
                step = dict(row)
                if step.get("tool_calls"):
                    step["tool_calls"] = json.loads(step["tool_calls"])
                steps.append(step)

            return steps

    def get_traces(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        user_evaluation: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """트레이스 목록 조회"""
        query = "SELECT * FROM workflow_traces WHERE 1=1"
        params = []

        if workflow_id:
            query += " AND workflow_id = ?"
            params.append(workflow_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        if user_evaluation:
            query += " AND user_evaluation = ?"
            params.append(user_evaluation)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

            return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM workflow_traces").fetchone()[0]

            by_status = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM workflow_traces
                GROUP BY status
            """).fetchall()

            by_evaluation = conn.execute("""
                SELECT user_evaluation, COUNT(*) as count
                FROM workflow_traces
                GROUP BY user_evaluation
            """).fetchall()

            by_workflow = conn.execute("""
                SELECT workflow_id, COUNT(*) as count
                FROM workflow_traces
                GROUP BY workflow_id
            """).fetchall()

            # 성공률 계산
            success_count = conn.execute("""
                SELECT COUNT(*) FROM workflow_traces WHERE user_evaluation = 'success'
            """).fetchone()[0]

            failed_count = conn.execute("""
                SELECT COUNT(*) FROM workflow_traces WHERE user_evaluation = 'failed'
            """).fetchone()[0]

            evaluated_count = success_count + failed_count
            success_rate = (success_count / evaluated_count * 100) if evaluated_count > 0 else 0

        return {
            "total_traces": total,
            "by_status": {row[0]: row[1] for row in by_status},
            "by_evaluation": {row[0]: row[1] for row in by_evaluation},
            "by_workflow": {row[0]: row[1] for row in by_workflow},
            "success_rate": round(success_rate, 2),
            "evaluated_count": evaluated_count,
        }

    def delete_trace(self, trace_id: str) -> bool:
        """트레이스 삭제"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM trace_steps WHERE trace_id = ?", (trace_id,))
                    conn.execute("DELETE FROM workflow_traces WHERE trace_id = ?", (trace_id,))
                    conn.commit()
                    return conn.total_changes > 0
            except Exception as e:
                print(f"트레이스 삭제 오류: {e}")
                return False

    def export_trace_json(self, trace_id: str) -> Optional[str]:
        """트레이스를 JSON으로 내보내기"""
        trace = self.get_trace(trace_id)
        if not trace:
            return None

        steps = self.get_trace_steps(trace_id)

        export_data = {
            "trace": trace,
            "steps": steps,
            "exported_at": datetime.utcnow().isoformat(),
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)


# 싱글톤 인스턴스
_trace_db_service: Optional[TraceDBService] = None


def get_trace_db() -> TraceDBService:
    """트레이스 DB 서비스 싱글톤 인스턴스 반환"""
    global _trace_db_service
    if _trace_db_service is None:
        _trace_db_service = TraceDBService()
    return _trace_db_service
