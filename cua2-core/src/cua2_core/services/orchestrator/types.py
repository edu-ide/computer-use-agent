"""
Orchestrator 공통 타입 정의

모든 Orchestrator 관련 모듈에서 사용하는 공통 타입들
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStrategy(Enum):
    """실행 전략"""
    CACHE_HIT = "cache_hit"  # 캐시에서 즉시 반환 (0.1초)
    RULE_BASED = "rule_based"  # 규칙 기반 실행 (무료, 빠름)
    LOCAL_MODEL = "local_model"  # 로컬 모델 (빠름, 저렴)
    CLOUD_LIGHT = "cloud_light"  # 클라우드 경량 모델 (중간)
    CLOUD_HEAVY = "cloud_heavy"  # 클라우드 고성능 모델 (느림, 비쌈)


class ErrorAction(Enum):
    """에러 발생 시 액션 (레거시 - LangGraph 에러 핸들링으로 대체됨)"""
    RETRY = "retry"  # 재시도
    SKIP = "skip"  # 이 노드 건너뛰기
    ABORT = "abort"  # 워크플로우 중단
    FALLBACK = "fallback"  # 다른 전략으로 재시도


class NodeStatus(Enum):
    """노드 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class StepAction(Enum):
    """스텝별 Orchestrator 액션"""
    CONTINUE = "continue"  # 계속 진행
    STOP = "stop"  # 즉시 중단
    INJECT_PROMPT = "inject_prompt"  # 추가 프롬프트 주입
    RETRY_STEP = "retry_step"  # 현재 스텝 재시도
    CHANGE_STRATEGY = "change_strategy"  # 전략 변경


@dataclass
class ExecutionDecision:
    """실행 결정 결과"""
    strategy: ExecutionStrategy
    model_id: Optional[str] = None
    cached_result: Optional[Dict[str, Any]] = None
    reason: str = ""
    estimated_time_ms: int = 0
    estimated_cost: float = 0.0
    confidence: float = 1.0

    # Orchestrator가 판단한 재사용 설정
    reusable: bool = False  # 이 노드의 결과를 재사용 가능한지
    reuse_trace: bool = False  # trace를 캐시할지
    share_memory: bool = False  # 이전 노드 메모리 공유할지
    cache_key_params: List[str] = field(default_factory=list)  # 캐시 키 파라미터


@dataclass
class ErrorDecision:
    """에러 처리 결정 (레거시 - LangGraph 에러 핸들링으로 대체됨)"""
    action: ErrorAction
    reason: str
    retry_count: int = 0
    max_retries: int = 3
    fallback_strategy: Optional[ExecutionStrategy] = None
    should_notify_user: bool = False
    error_message: str = ""


@dataclass
class StepFeedback:
    """스텝 실행 후 Orchestrator 피드백"""
    action: StepAction
    reason: str = ""
    # 프롬프트 주입
    injected_prompt: Optional[str] = None  # VLM에 추가할 지시사항
    # 전략 변경
    new_strategy: Optional[ExecutionStrategy] = None
    # 학습된 패턴
    learned_pattern: Optional[str] = None
    # 다음 스텝 힌트
    next_step_hint: Optional[str] = None
    # 메모리 저장 요청 {pattern, reason}
    save_to_memory: Optional[Dict[str, str]] = None


@dataclass
class NodeExecutionRecord:
    """노드 실행 기록"""
    node_id: str
    status: NodeStatus
    strategy: ExecutionStrategy
    start_time: float
    end_time: Optional[float] = None
    duration_ms: int = 0
    error: Optional[str] = None
    retry_count: int = 0
    result_summary: str = ""


@dataclass
class NodeComplexity:
    """노드 복잡도 분석"""
    requires_vision: bool = False  # 화면 분석 필요
    requires_interaction: bool = False  # 브라우저 상호작용 필요
    requires_reasoning: bool = False  # 복잡한 추론 필요
    requires_data_extraction: bool = False  # 데이터 추출 필요
    has_dynamic_content: bool = False  # 동적 콘텐츠 (변하는 데이터)
    complexity_score: float = 0.0  # 0~1


@dataclass
class ModelConfig:
    """모델 설정"""
    model_id: str
    name: str
    cost_per_1k_tokens: float  # 입력 기준
    avg_latency_ms: int
    supports_vision: bool = True
    max_complexity: float = 1.0  # 처리 가능한 최대 복잡도


@dataclass
class WorkflowReport:
    """워크플로우 실행 리포트"""
    workflow_id: str
    execution_id: str
    status: str  # "completed", "failed", "timeout", "partial"
    start_time: float
    end_time: float
    total_duration_ms: int
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    skipped_nodes: int
    total_cost: float
    node_records: List[NodeExecutionRecord] = field(default_factory=list)
    summary: str = ""
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "skipped_nodes": self.skipped_nodes,
            "total_cost": self.total_cost,
            "summary": self.summary,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "node_records": [
                {
                    "node_id": r.node_id,
                    "status": r.status.value,
                    "strategy": r.strategy.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "retry_count": r.retry_count,
                    "result_summary": r.result_summary,
                }
                for r in self.node_records
            ],
        }
