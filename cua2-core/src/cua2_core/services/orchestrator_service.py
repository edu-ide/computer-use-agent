"""
Orchestrator Service - ToolOrchestra 패턴 구현

작은 모델(또는 규칙)이 어떤 전략으로 노드를 실행할지 결정합니다.
- 캐시 히트: 즉시 반환 (판단도 스킵)
- 단순 작업: 로컬/저렴한 모델
- 복잡한 판단: 고성능 모델
- 학습된 패턴: 규칙 기반 실행

Reference: NVIDIA ToolOrchestra (arXiv:2511.21689)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

# 버전 정보
ORCHESTRATOR_VERSION = "1.1.0"
ORCHESTRATOR_VERSION_DATE = "2025-11-29"

from .trace_store import get_trace_store, TraceStore
from .node_reuse_analyzer import get_node_reuse_analyzer, NodeReuseAnalyzer, ReuseDecision
from .letta_memory_service import get_letta_memory_service, LettaMemoryService
from .agent_activity_log import (
    log_orchestrator,
    log_trace,
    ActivityType,
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """실행 전략"""
    CACHE_HIT = "cache_hit"  # 캐시에서 즉시 반환 (0.1초)
    RULE_BASED = "rule_based"  # 규칙 기반 실행 (무료, 빠름)
    LOCAL_MODEL = "local_model"  # 로컬 모델 (빠름, 저렴)
    CLOUD_LIGHT = "cloud_light"  # 클라우드 경량 모델 (중간)
    CLOUD_HEAVY = "cloud_heavy"  # 클라우드 고성능 모델 (느림, 비쌈)


class ErrorAction(Enum):
    """에러 발생 시 액션"""
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

    # Orchestrator가 결정한 실행 설정
    agent_type: str = "VLMAgent"  # "VLMAgent" or "CodeAgent"
    max_tokens: int = 1024  # 최대 출력 토큰 수

    # Orchestrator가 판단한 재사용 설정
    reusable: bool = False  # 이 노드의 결과를 재사용 가능한지
    reuse_trace: bool = False  # trace를 캐시할지
    share_memory: bool = False  # 이전 노드 메모리 공유할지
    cache_key_params: List[str] = field(default_factory=list)  # 캐시 키 파라미터


@dataclass
class ErrorDecision:
    """에러 처리 결정"""
    action: ErrorAction
    reason: str
    retry_count: int = 0
    max_retries: int = 3
    fallback_strategy: Optional[ExecutionStrategy] = None
    should_notify_user: bool = False
    error_message: str = ""


class StepAction(Enum):
    """스텝별 Orchestrator 액션"""
    CONTINUE = "continue"  # 계속 진행
    STOP = "stop"  # 즉시 중단
    INJECT_PROMPT = "inject_prompt"  # 추가 프롬프트 주입
    RETRY_STEP = "retry_step"  # 현재 스텝 재시도
    CHANGE_STRATEGY = "change_strategy"  # 전략 변경


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


class OrchestratorService:
    """
    Orchestrator Service

    노드 실행 전 최적의 실행 전략을 결정합니다.
    ToolOrchestra 논문의 핵심 아이디어:
    - 작은 orchestrator가 비용/성능 균형을 맞춤
    - 캐시 활용으로 반복 작업 최소화
    - 복잡도에 따른 모델 라우팅
    """

    # 사용 가능한 모델 설정
    MODELS: Dict[str, ModelConfig] = {
        "local-qwen-vl": ModelConfig(
            model_id="local-qwen-vl",
            name="Qwen-VL (Local)",
            cost_per_1k_tokens=0.0,
            avg_latency_ms=500,
            supports_vision=True,
            max_complexity=0.5,
        ),
        "gpt-4o-mini": ModelConfig(
            model_id="gpt-4o-mini",
            name="GPT-4o Mini",
            cost_per_1k_tokens=0.00015,
            avg_latency_ms=1000,
            supports_vision=True,
            max_complexity=0.7,
        ),
        "gpt-4o": ModelConfig(
            model_id="gpt-4o",
            name="GPT-4o",
            cost_per_1k_tokens=0.0025,
            avg_latency_ms=2000,
            supports_vision=True,
            max_complexity=0.9,
        ),
        "claude-sonnet": ModelConfig(
            model_id="claude-3-5-sonnet",
            name="Claude 3.5 Sonnet",
            cost_per_1k_tokens=0.003,
            avg_latency_ms=2500,
            supports_vision=True,
            max_complexity=1.0,
        ),
    }

    # 복잡도 임계값
    COMPLEXITY_THRESHOLDS = {
        "simple": 0.3,  # 단순 작업 (클릭, 입력)
        "medium": 0.6,  # 중간 복잡도 (데이터 추출)
        "complex": 0.8,  # 복잡한 작업 (추론 필요)
    }

    # Orchestrator-8B 서버 설정
    ORCHESTRATOR_API_URL = "http://localhost:8081/v1/chat/completions"
    ORCHESTRATOR_TIMEOUT = 5.0  # 5초 타임아웃 (빠른 판단 필요)

    # Note: Rule-based BOT_DETECTION_PATTERNS / FAILURE_PATTERNS 제거됨
    # VLM이 스크린샷을 보고 직접 [ERROR:TYPE] 형식으로 보고함
    # LangGraph 조건부 엣지에서 에러 타입별 라우팅 처리

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        reuse_analyzer: Optional[NodeReuseAnalyzer] = None,
        prefer_local: bool = True,
        cost_weight: float = 0.5,
        use_orchestrator_model: bool = True,
        orchestrator_url: Optional[str] = None,
        letta_service: Optional[LettaMemoryService] = None,
    ):
        """
        Args:
            trace_store: Trace 캐시 저장소
            reuse_analyzer: 재사용 분석기
            prefer_local: 로컬 모델 우선 여부
            cost_weight: 비용 가중치 (높을수록 저렴한 옵션 선호)
            use_orchestrator_model: Orchestrator-8B 모델 사용 여부
            orchestrator_url: Orchestrator API URL (기본: localhost:8081)
            letta_service: Letta Memory 서비스
        """
        self._trace_store = trace_store or get_trace_store()
        self._reuse_analyzer = reuse_analyzer or get_node_reuse_analyzer()
        self._letta = letta_service or get_letta_memory_service()
        self._prefer_local = prefer_local
        self._cost_weight = cost_weight
        self._use_orchestrator_model = use_orchestrator_model
        self._orchestrator_url = orchestrator_url or self.ORCHESTRATOR_API_URL

        # 노드별 복잡도 캐시
        self._complexity_cache: Dict[str, NodeComplexity] = {}

        # 실행 통계 (최적화에 활용)
        self._execution_stats: Dict[str, Dict[str, Any]] = {}

        # HTTP 클라이언트
        self._http_client: Optional[httpx.AsyncClient] = None

        # 워크플로우 실행 추적
        self._workflow_executions: Dict[str, Dict[str, Any]] = {}

        # 버전 관리 및 개선 이력
        self._version = ORCHESTRATOR_VERSION
        self._version_date = ORCHESTRATOR_VERSION_DATE
        self._improvement_history: List[Dict[str, Any]] = []
        self._pattern_versions: Dict[str, int] = {}  # 패턴별 버전

        # 개선 이력 로드
        self._load_improvement_history()

        logger.info(
            f"[Orchestrator] 초기화 v{self._version} ({self._version_date}), "
            f"개선 이력 {len(self._improvement_history)}건"
        )

    def _get_data_dir(self) -> str:
        """데이터 디렉토리 반환"""
        data_dir = os.path.join(os.path.expanduser("~"), ".cua2", "orchestrator")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def _load_improvement_history(self):
        """개선 이력 로드"""
        history_file = os.path.join(self._get_data_dir(), "improvement_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._improvement_history = data.get("history", [])
                    self._pattern_versions = data.get("pattern_versions", {})
                    # 학습된 패턴도 로드
                    self._step_patterns = data.get("step_patterns", {})
                    logger.info(f"[Orchestrator] 개선 이력 로드: {len(self._improvement_history)}건")
            except Exception as e:
                logger.warning(f"[Orchestrator] 개선 이력 로드 실패: {e}")

    def _save_improvement_history(self):
        """개선 이력 저장"""
        history_file = os.path.join(self._get_data_dir(), "improvement_history.json")
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump({
                    "version": self._version,
                    "last_updated": datetime.now().isoformat(),
                    "history": self._improvement_history[-100:],  # 최근 100건만 유지
                    "pattern_versions": self._pattern_versions,
                    "step_patterns": self._step_patterns,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Orchestrator] 개선 이력 저장 실패: {e}")

    def record_improvement(
        self,
        improvement_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ):
        """
        개선 사항 기록

        Args:
            improvement_type: 개선 유형 (pattern_added, prompt_updated, strategy_changed 등)
            description: 개선 설명
            details: 추가 상세 정보
            node_id: 관련 노드 ID
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "version": self._version,
            "type": improvement_type,
            "description": description,
            "node_id": node_id,
            "details": details or {},
        }

        self._improvement_history.append(entry)

        # 패턴 버전 업데이트
        if node_id:
            self._pattern_versions[node_id] = self._pattern_versions.get(node_id, 0) + 1

        logger.info(f"[Orchestrator] 개선 기록: {improvement_type} - {description}")

        # 자동 저장
        self._save_improvement_history()

    def get_version_info(self) -> Dict[str, Any]:
        """버전 정보 반환"""
        return {
            "version": self._version,
            "version_date": self._version_date,
            "total_improvements": len(self._improvement_history),
            "patterns_learned": len(self._step_patterns) if hasattr(self, '_step_patterns') else 0,
            "recent_improvements": self._improvement_history[-5:] if self._improvement_history else [],
        }

    def get_improvement_summary(self) -> str:
        """개선 이력 요약 반환"""
        if not self._improvement_history:
            return "개선 이력 없음"

        summary_lines = [
            f"Orchestrator v{self._version} ({self._version_date})",
            f"총 개선: {len(self._improvement_history)}건",
            "",
            "최근 개선:",
        ]

        for entry in self._improvement_history[-5:]:
            summary_lines.append(
                f"  - [{entry['type']}] {entry['description']} ({entry['timestamp'][:10]})"
            )

        return "\n".join(summary_lines)

    def decide(
        self,
        workflow_id: str,
        node_id: str,
        instruction: str,
        params: Dict[str, Any],
        node_config: Optional[Any] = None,
    ) -> ExecutionDecision:
        """
        노드 실행 전략 결정

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            instruction: 노드 instruction
            params: 실행 파라미터
            node_config: 노드 설정 (WorkflowNode)

        Returns:
            ExecutionDecision: 실행 결정
        """
        start_time = time.time()

        # 1. 캐시 체크 (가장 먼저!) - 히트하면 즉시 반환
        cache_decision = self._check_cache(workflow_id, node_id, params, node_config)
        if cache_decision:
            logger.info(f"[Orchestrator] {node_id}: CACHE_HIT (판단 스킵)")
            return cache_decision

        # 2. 학습된 재사용 설정 확인
        learned_settings = self._reuse_analyzer.get_recommended_settings(node_id, workflow_id)

        # 3. 노드 복잡도 분석
        complexity = self._analyze_complexity(node_id, instruction, node_config)

        # 4. 최적 전략 결정
        decision = self._select_strategy(
            node_id=node_id,
            complexity=complexity,
            learned_settings=learned_settings,
            instruction=instruction,
            node_config=node_config,  # 노드 설정 전달
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[Orchestrator] {node_id}: {decision.strategy.value} "
            f"(agent={decision.agent_type}, model={decision.model_id}, "
            f"max_tokens={decision.max_tokens}, complexity={complexity.complexity_score:.2f}, "
            f"decision_time={elapsed_ms}ms)"
        )

        return decision

    def _check_cache(
        self,
        workflow_id: str,
        node_id: str,
        params: Dict[str, Any],
        node_config: Optional[Any] = None,
    ) -> Optional[ExecutionDecision]:
        """캐시 체크 - 히트하면 ExecutionDecision 반환"""

        # 노드 설정에서 캐시 키 파라미터 가져오기
        cache_key_params = []
        reuse_trace = False

        if node_config:
            cache_key_params = getattr(node_config, 'cache_key_params', [])
            reuse_trace = getattr(node_config, 'reuse_trace', False)

        # 학습된 설정 확인 (노드 설정보다 우선)
        learned_settings = self._reuse_analyzer.get_recommended_settings(node_id, workflow_id)
        if learned_settings.get("confidence", 0) > 0.7:
            # 학습된 설정이 신뢰도 높으면 사용
            reuse_trace = learned_settings.get("reuse_trace", reuse_trace)
            cache_key_params = learned_settings.get("cache_key_params", cache_key_params)
            logger.debug(f"[Orchestrator] {node_id}: 학습된 reuse_trace={reuse_trace} 사용")

        if not reuse_trace:
            return None

        # Trace Store에서 캐시 조회
        cached_trace = self._trace_store.get_reusable_trace(
            workflow_id=workflow_id,
            node_id=node_id,
            params=params,
            key_params=cache_key_params,
        )

        if cached_trace and cached_trace.success:
            return ExecutionDecision(
                strategy=ExecutionStrategy.CACHE_HIT,
                model_id=None,
                cached_result={
                    "success": True,
                    "data": cached_trace.data,
                    "steps": cached_trace.steps,
                    "reused_from_cache": True,
                    "cache_key": cached_trace.cache_key,
                    "used_count": cached_trace.used_count,
                },
                reason=f"캐시 히트 (사용 횟수: {cached_trace.used_count})",
                estimated_time_ms=100,  # 캐시는 거의 즉시
                estimated_cost=0.0,
                confidence=1.0,
            )

        return None

    def _analyze_complexity(
        self,
        node_id: str,
        instruction: str,
        node_config: Optional[Any] = None,
    ) -> NodeComplexity:
        """노드 복잡도 분석"""

        # 캐시 확인
        cache_key = f"{node_id}:{hash(instruction)}"
        if cache_key in self._complexity_cache:
            return self._complexity_cache[cache_key]

        instruction_lower = instruction.lower()

        # 특성 분석
        requires_vision = any(kw in instruction_lower for kw in [
            "look", "see", "find", "scroll", "screenshot", "화면", "보이는", "찾"
        ])

        requires_interaction = any(kw in instruction_lower for kw in [
            "click", "type", "press", "scroll", "navigate", "클릭", "입력", "스크롤"
        ])

        requires_reasoning = any(kw in instruction_lower for kw in [
            "analyze", "decide", "compare", "if", "check", "판단", "분석", "비교", "확인"
        ])

        requires_data_extraction = any(kw in instruction_lower for kw in [
            "extract", "collect", "list", "price", "name", "추출", "수집", "목록", "가격"
        ])

        has_dynamic_content = any(kw in instruction_lower for kw in [
            "search result", "product", "next page", "검색", "상품", "다음"
        ])

        # 노드 타입 기반 추가 분석
        node_type = getattr(node_config, 'node_type', None) if node_config else None
        if node_type == "process":
            # 프로세스 노드는 단순
            requires_vision = False
            requires_reasoning = False

        # 복잡도 점수 계산
        score = 0.0
        if requires_vision:
            score += 0.2
        if requires_interaction:
            score += 0.15
        if requires_reasoning:
            score += 0.3
        if requires_data_extraction:
            score += 0.2
        if has_dynamic_content:
            score += 0.15

        complexity = NodeComplexity(
            requires_vision=requires_vision,
            requires_interaction=requires_interaction,
            requires_reasoning=requires_reasoning,
            requires_data_extraction=requires_data_extraction,
            has_dynamic_content=has_dynamic_content,
            complexity_score=min(score, 1.0),
        )

        # 캐시 저장
        self._complexity_cache[cache_key] = complexity

        return complexity

    def _select_strategy(
        self,
        node_id: str,
        complexity: NodeComplexity,
        learned_settings: Dict[str, Any],
        instruction: str,
        node_config: Optional[Any] = None,
    ) -> ExecutionDecision:
        """최적 실행 전략 선택"""

        score = complexity.complexity_score

        # 학습된 재사용 설정 추출
        use_learned = learned_settings.get("confidence", 0) > 0.7
        learned_reusable = learned_settings.get("reusable", False) if use_learned else False
        learned_reuse_trace = learned_settings.get("reuse_trace", False) if use_learned else False
        learned_share_memory = learned_settings.get("share_memory", False) if use_learned else False
        learned_cache_key_params = learned_settings.get("cache_key_params", []) if use_learned else []

        # node_type 확인 (extract_data인 경우 Code Agent 사용)
        node_type = getattr(node_config, 'node_type', None) if node_config else None
        
        # Agent 타입 및 max_tokens 결정
        if node_type == "extract_data":
            # Code Agent 사용
            agent_type = "CodeAgent"
            model_id = "Orchestrator-8B"
            max_tokens = 2048  # JS 생성에는 더 많은 토큰 필요
            strategy = ExecutionStrategy.CLOUD_HEAVY  # Code Agent는 복잡한 분석
        else:
            # VLM Agent 사용
            agent_type = "VLMAgent"
            
            # 복잡도에 따른 max_tokens 설정
            if score >= self.COMPLEXITY_THRESHOLDS["complex"]:
                max_tokens = 4096  # 복잡한 작업
            elif score >= self.COMPLEXITY_THRESHOLDS["medium"]:
                max_tokens = 2048  # 중간 복잡도
            else:
                max_tokens = 1024  # 단순 작업
            
            # 기존 전략 선택 로직
            model_id = None
            strategy = ExecutionStrategy.CLOUD_HEAVY

        # 1. 규칙 기반으로 처리 가능한 경우 (가장 빠름)
        if self._can_use_rule_based(node_id, instruction):
            return ExecutionDecision(
                strategy=ExecutionStrategy.RULE_BASED,
                model_id=None,
                reason="규칙 기반으로 처리 가능",
                estimated_time_ms=50,
                estimated_cost=0.0,
                confidence=0.95,
                agent_type="VLMAgent",  # 규칙 기반도 VLM
                max_tokens=512,  # 규칙 기반은 짧은 응답
                reusable=learned_reusable,
                reuse_trace=learned_reuse_trace,
                share_memory=learned_share_memory,
                cache_key_params=learned_cache_key_params,
            )

        # 2. Code Agent 결정이 내려진 경우
        if agent_type == "CodeAgent":
            return ExecutionDecision(
                strategy=ExecutionStrategy.CLOUD_HEAVY,
                model_id=model_id,
                reason=f"데이터 추출 작업 (node_type=extract_data)",
                estimated_time_ms=3000,  # Code Agent는 시간이 좀 걸림
                estimated_cost=0.001,
                confidence=0.9,
                agent_type=agent_type,
                max_tokens=max_tokens,
                reusable=learned_reusable,
                reuse_trace=learned_reuse_trace,
                share_memory=learned_share_memory,
                cache_key_params=learned_cache_key_params,
            )

        # 3. VLM Agent - 복잡도에 따른 모델 선택
        if score <= self.COMPLEXITY_THRESHOLDS["simple"]:
            # 단순 작업 - 로컬 모델
            if self._prefer_local:
                return ExecutionDecision(
                    strategy=ExecutionStrategy.LOCAL_MODEL,
                    model_id="local-qwen-vl",
                    reason=f"단순 작업 (complexity={score:.2f})",
                    estimated_time_ms=self.MODELS["local-qwen-vl"].avg_latency_ms,
                    estimated_cost=0.0,
                    confidence=0.85,
                    agent_type=agent_type,
                    max_tokens=max_tokens,
                    reusable=learned_reusable,
                    reuse_trace=learned_reuse_trace,
                    share_memory=learned_share_memory,
                    cache_key_params=learned_cache_key_params,
                )
            else:
                return ExecutionDecision(
                    strategy=ExecutionStrategy.CLOUD_LIGHT,
                    model_id="gpt-4o-mini",
                    reason=f"단순 작업 (complexity={score:.2f})",
                    estimated_time_ms=self.MODELS["gpt-4o-mini"].avg_latency_ms,
                    estimated_cost=self.MODELS["gpt-4o-mini"].cost_per_1k_tokens * 2,
                    confidence=0.9,
                    agent_type=agent_type,
                    max_tokens=max_tokens,
                    reusable=learned_reusable,
                    reuse_trace=learned_reuse_trace,
                    share_memory=learned_share_memory,
                    cache_key_params=learned_cache_key_params,
                )

        elif score <= self.COMPLEXITY_THRESHOLDS["medium"]:
            # 중간 복잡도 - 경량 클라우드 모델
            return ExecutionDecision(
                strategy=ExecutionStrategy.CLOUD_LIGHT,
                model_id="gpt-4o-mini",
                reason=f"중간 복잡도 (complexity={score:.2f})",
                estimated_time_ms=self.MODELS["gpt-4o-mini"].avg_latency_ms,
                estimated_cost=self.MODELS["gpt-4o-mini"].cost_per_1k_tokens * 3,
                confidence=0.85,
                agent_type=agent_type,
                max_tokens=max_tokens,
                reusable=learned_reusable,
                reuse_trace=learned_reuse_trace,
                share_memory=learned_share_memory,
                cache_key_params=learned_cache_key_params,
            )

        elif score <= self.COMPLEXITY_THRESHOLDS["complex"]:
            # 복잡한 작업 - 고성능 모델
            return ExecutionDecision(
                strategy=ExecutionStrategy.CLOUD_HEAVY,
                model_id="gpt-4o",
                reason=f"복잡한 작업 (complexity={score:.2f})",
                estimated_time_ms=self.MODELS["gpt-4o"].avg_latency_ms,
                estimated_cost=self.MODELS["gpt-4o"].cost_per_1k_tokens * 5,
                confidence=0.9,
                agent_type=agent_type,
                max_tokens=max_tokens,
                reusable=learned_reusable,
                reuse_trace=learned_reuse_trace,
                share_memory=learned_share_memory,
                cache_key_params=learned_cache_key_params,
            )

        else:
            # 매우 복잡 - 최고 성능 모델
            return ExecutionDecision(
                strategy=ExecutionStrategy.CLOUD_HEAVY,
                model_id="claude-sonnet",
                reason=f"매우 복잡한 작업 (complexity={score:.2f})",
                estimated_time_ms=self.MODELS["claude-sonnet"].avg_latency_ms,
                estimated_cost=self.MODELS["claude-sonnet"].cost_per_1k_tokens * 5,
                confidence=0.95,
                agent_type=agent_type,
                max_tokens=max_tokens,
                reusable=learned_reusable,
                reuse_trace=learned_reuse_trace,
                share_memory=learned_share_memory,
                cache_key_params=learned_cache_key_params,
            )

    def _can_use_rule_based(self, node_id: str, instruction: str) -> bool:
        """규칙 기반으로 처리 가능한지 확인"""

        # 특정 노드는 규칙 기반으로 처리 가능
        rule_based_patterns = [
            # URL 열기는 단순 명령
            ("open_url", "open_url" in instruction.lower()),
            # 단순 대기
            ("wait", "wait(" in instruction.lower() and "wait for" not in instruction.lower()),
        ]

        for pattern_name, matches in rule_based_patterns:
            if matches:
                return True

        return False

    def record_execution_result(
        self,
        workflow_id: str,
        node_id: str,
        strategy: ExecutionStrategy,
        model_id: Optional[str],
        success: bool,
        actual_time_ms: int,
        actual_cost: float,
    ):
        """실행 결과 기록 (향후 최적화에 활용)"""
        key = f"{workflow_id}:{node_id}"

        if key not in self._execution_stats:
            self._execution_stats[key] = {
                "executions": [],
                "strategy_success_rate": {},
            }

        self._execution_stats[key]["executions"].append({
            "strategy": strategy.value,
            "model_id": model_id,
            "success": success,
            "time_ms": actual_time_ms,
            "cost": actual_cost,
        })

        # 전략별 성공률 업데이트
        strategy_key = strategy.value
        if strategy_key not in self._execution_stats[key]["strategy_success_rate"]:
            self._execution_stats[key]["strategy_success_rate"][strategy_key] = {
                "success": 0,
                "total": 0,
            }

        stats = self._execution_stats[key]["strategy_success_rate"][strategy_key]
        stats["total"] += 1
        if success:
            stats["success"] += 1

    def get_stats(self, workflow_id: str, node_id: str) -> Dict[str, Any]:
        """노드별 실행 통계 조회"""
        key = f"{workflow_id}:{node_id}"
        return self._execution_stats.get(key, {})

    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 반환 (lazy init)"""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.ORCHESTRATOR_TIMEOUT)
        return self._http_client

    async def decide_async(
        self,
        workflow_id: str,
        node_id: str,
        instruction: str,
        params: Dict[str, Any],
        node_config: Optional[Any] = None,
        execution_id: Optional[str] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionDecision:
        """
        비동기 노드 실행 전략 결정 (Orchestrator-8B 사용)

        Args:
            workflow_id: 워크플로우 ID (예: coupang-collect)
            node_id: 노드 ID
            instruction: 노드 instruction
            params: 실행 파라미터
            node_config: 노드 설정 (WorkflowNode)
            execution_id: 실행 ID (예: coupang-collect-20251129092555)
            execution_history: 최근 실행 이력 (컨텍스트 제공)

        Returns:
            ExecutionDecision: 실행 결정
        """
        # execution_id가 없으면 workflow_id 사용 (하위 호환)
        exec_id = execution_id or workflow_id
        start_time = time.time()

        # Load failure patterns from Letta
        if self._letta:
            try:
                if not hasattr(self, "_node_failure_patterns_cache"):
                    self._node_failure_patterns_cache = {}
                
                patterns = await self._letta.get_failure_patterns(workflow_id, node_id)
                self._node_failure_patterns_cache[node_id] = patterns
                if patterns:
                    logger.info(f"[Orchestrator] {node_id}: 메모리에서 실패 패턴 로드 ({len(patterns)}개)")
            except Exception as e:
                logger.warning(f"[Orchestrator] 실패 패턴 로드 실패: {e}")

        # 1. 캐시 체크 (가장 먼저!) - 히트하면 즉시 반환
        cache_decision = self._check_cache(workflow_id, node_id, params, node_config)
        if cache_decision:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[Orchestrator] {node_id}: CACHE_HIT (판단 스킵)")

            # 활동 로그: 캐시 히트
            log_trace(
                ActivityType.CACHE_HIT,
                f"캐시 히트: {node_id}",
                details={
                    "used_count": cache_decision.cached_result.get("used_count", 1) if cache_decision.cached_result else 1,
                },
                execution_id=exec_id,
                node_id=node_id,
                duration_ms=elapsed_ms,
            )
            return cache_decision

        # 2. Orchestrator-8B 모델 사용 여부
        if self._use_orchestrator_model:
            try:
                decision = await self._query_orchestrator_model(
                    node_id=node_id,
                    instruction=instruction,
                    execution_history=execution_history,
                )
                if decision:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"[Orchestrator] {node_id}: {decision.strategy.value} "
                        f"(model=orchestrator-8b, decision_time={elapsed_ms}ms)"
                    )

                    # 활동 로그: Orchestrator 결정
                    log_orchestrator(
                        ActivityType.DECISION,
                        f"{node_id} → {decision.strategy.value}",
                        details={
                            "strategy": decision.strategy.value,
                            "model": decision.model_id,
                            "reason": decision.reason,
                            "reuse_trace": decision.reuse_trace,
                            "share_memory": decision.share_memory,
                        },
                        execution_id=exec_id,
                        node_id=node_id,
                        duration_ms=elapsed_ms,
                    )
                    return decision
            except Exception as e:
                logger.warning(f"[Orchestrator] 모델 호출 실패, 규칙 기반으로 fallback: {e}")

                # 활동 로그: API 호출 실패
                log_orchestrator(
                    ActivityType.ERROR,
                    f"모델 호출 실패: {str(e)[:50]}",
                    details={"error": str(e)},
                    execution_id=exec_id,
                    node_id=node_id,
                )

        # 3. Fallback: 규칙 기반 판단
        decision = self.decide(workflow_id, node_id, instruction, params, node_config)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # 활동 로그: 규칙 기반 결정
        log_orchestrator(
            ActivityType.DECISION,
            f"{node_id} → {decision.strategy.value} (규칙)",
            details={
                "strategy": decision.strategy.value,
                "model": decision.model_id,
                "fallback": True,
            },
            execution_id=exec_id,
            node_id=node_id,
            duration_ms=elapsed_ms,
        )

        return decision

    async def _query_orchestrator_model(
        self,
        node_id: str,
        instruction: str,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ExecutionDecision]:
        """
        Orchestrator-8B 모델에 질의

        모델에게 노드 정보를 주고 최적의 실행 전략 + 재사용 설정을 결정하게 함
        """
        # 시스템 프롬프트: Orchestrator 역할 정의
        system_prompt = """You are an Orchestrator that decides the optimal execution strategy and reuse settings for workflow nodes.

## Execution Strategy
Choose based on task complexity:
- "local_model": Simple tasks (open URL, click button, basic navigation)
- "cloud_light": Medium complexity (search, form filling, data extraction)
- "cloud_heavy": Complex reasoning (analyze content, make decisions, extract structured data)

## Reuse Settings
Decide if this node's execution can be cached and reused:
- reuse_trace: true if the same instruction always produces same result (e.g., open homepage)
- reuse_trace: false if result depends on dynamic content (e.g., analyze current page)
- share_memory: true if this node needs context from previous nodes
- cache_key_params: list of parameter names that affect the result (e.g., ["keyword"] for search)

Respond with JSON only:
{
  "strategy": "local_model|cloud_light|cloud_heavy",
  "model": "model_id or null",
  "reason": "brief reason",
  "reuse_trace": true|false,
  "share_memory": true|false,
  "cache_key_params": ["param1", "param2"] or []
}"""

        # 사용자 프롬프트: 노드 정보
        user_content = f"""Node: {node_id}
Instruction: {instruction}"""

        if execution_history:
            recent = execution_history[-3:]  # 최근 3개만
            history_str = "\n".join([
                f"- {h.get('node_id')}: {h.get('strategy', 'unknown')} (success={h.get('success', '?')})"
                for h in recent
            ])
            user_content += f"\n\nRecent execution history:\n{history_str}"

        user_content += "\n\nDecide execution strategy and reuse settings (JSON only):"

        try:
            client = await self._get_http_client()
            response = await client.post(
                self._orchestrator_url,
                json={
                    "model": "orchestrator-8b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # JSON 파싱
            try:
                # JSON 부분 추출 (```json ... ``` 또는 순수 JSON)
                if "```" in content:
                    json_str = content.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    json_str = json_str.strip()
                else:
                    json_str = content.strip()

                decision_data = json.loads(json_str)
                strategy_str = decision_data.get("strategy", "cloud_light")
                model_id = decision_data.get("model")
                reason = decision_data.get("reason", "Orchestrator 결정")

                # 재사용 설정 추출
                reuse_trace = decision_data.get("reuse_trace", False)
                share_memory = decision_data.get("share_memory", False)
                cache_key_params = decision_data.get("cache_key_params", [])

                # 전략 매핑
                strategy_map = {
                    "local_model": ExecutionStrategy.LOCAL_MODEL,
                    "cloud_light": ExecutionStrategy.CLOUD_LIGHT,
                    "cloud_heavy": ExecutionStrategy.CLOUD_HEAVY,
                    "rule_based": ExecutionStrategy.RULE_BASED,
                }
                strategy = strategy_map.get(strategy_str, ExecutionStrategy.CLOUD_LIGHT)

                # 모델 ID 기본값
                if not model_id:
                    model_defaults = {
                        ExecutionStrategy.LOCAL_MODEL: "local-qwen-vl",
                        ExecutionStrategy.CLOUD_LIGHT: "gpt-4o-mini",
                        ExecutionStrategy.CLOUD_HEAVY: "gpt-4o",
                    }
                    model_id = model_defaults.get(strategy, "gpt-4o-mini")

                return ExecutionDecision(
                    strategy=strategy,
                    model_id=model_id,
                    reason=f"[Orchestrator-8B] {reason}",
                    estimated_time_ms=self.MODELS.get(model_id, self.MODELS["gpt-4o-mini"]).avg_latency_ms,
                    estimated_cost=self.MODELS.get(model_id, self.MODELS["gpt-4o-mini"]).cost_per_1k_tokens * 2,
                    confidence=0.9,
                    # Orchestrator가 판단한 재사용 설정
                    reusable=reuse_trace,  # reuse_trace면 reusable
                    reuse_trace=reuse_trace,
                    share_memory=share_memory,
                    cache_key_params=cache_key_params,
                )

            except json.JSONDecodeError as e:
                logger.warning(f"[Orchestrator] JSON 파싱 실패: {e}, content: {content}")
                return None

        except httpx.RequestError as e:
            logger.warning(f"[Orchestrator] API 요청 실패: {e}")
            return None

    # ===============================
    # Error Handling & Timeout
    # ===============================

    # 노드별 타임아웃 설정 (초)
    NODE_TIMEOUTS = {
        "default": 120,  # 기본 2분
        "open_url": 30,
        "search": 60,
        "navigate": 60,
        "extract": 180,  # 데이터 추출은 더 오래 걸릴 수 있음
        "analyze": 180,
    }

    # Note: 에러 타입별 액션은 LangGraph 조건부 엣지로 이동됨
    # workflow_base.py의 _create_router()에서 VLM 에러 타입에 따라 라우팅
    # - BOT_DETECTED → 즉시 중단
    # - PAGE_FAILED/TIMEOUT → 재시도 (max_retries까지)
    # - ELEMENT_NOT_FOUND → 스킵
    # - ACCESS_DENIED → 중단

    # Note: Rule-based 패턴 체크 완전 제거됨
    # VLM이 스크린샷을 보고 직접 [ERROR:TYPE] 형식으로 보고함

    def get_node_timeout(self, node_id: str, instruction: str) -> int:
        """노드별 타임아웃 시간 반환 (초)"""
        instruction_lower = instruction.lower()

        # 노드 이름이나 instruction에서 타입 추론
        for key, timeout in self.NODE_TIMEOUTS.items():
            if key in node_id.lower() or key in instruction_lower:
                return timeout

        return self.NODE_TIMEOUTS["default"]

    async def handle_error(
        self,
        workflow_id: str,
        node_id: str,
        error: Exception,
        current_retry: int = 0,
        strategy: Optional[ExecutionStrategy] = None,
    ) -> ErrorDecision:
        """
        [DEPRECATED] 에러 발생 시 처리 방법 결정

        Note: 이 메서드는 LangGraph 조건부 엣지로 대체되었습니다.
        workflow_base.py의 _create_router()에서 VLM 에러 타입에 따라 라우팅합니다.
        - VLM이 [ERROR:TYPE] 형식으로 에러 보고
        - LangGraph router가 에러 타입에 따라 retry/skip/abort 결정

        이 메서드는 레거시 호환성을 위해 유지되지만, 새로운 워크플로우에서는
        LangGraph 에러 핸들링을 사용해야 합니다.

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            error: 발생한 에러
            current_retry: 현재 재시도 횟수
            strategy: 현재 사용 중인 전략

        Returns:
            ErrorDecision: 에러 처리 결정
        """
        import warnings
        warnings.warn(
            "handle_error()는 deprecated입니다. LangGraph 조건부 엣지를 사용하세요.",
            DeprecationWarning,
            stacklevel=2,
        )

        error_str = str(error).lower()
        error_type = self._classify_error_legacy(error_str)

        # 레거시 기본 액션
        legacy_error_actions = {
            "timeout": ErrorAction.RETRY,
            "network": ErrorAction.RETRY,
            "element_not_found": ErrorAction.SKIP,
            "navigation_failed": ErrorAction.RETRY,
            "api_error": ErrorAction.FALLBACK,
            "bot_detected": ErrorAction.ABORT,
            "page_load_failed": ErrorAction.RETRY,
            "unknown": ErrorAction.ABORT,
        }
        default_action = legacy_error_actions.get(error_type, ErrorAction.ABORT)

        # Orchestrator-8B로 더 정교한 결정
        if self._use_orchestrator_model:
            try:
                decision = await self._query_error_handling(
                    node_id=node_id,
                    error_str=str(error),
                    error_type=error_type,
                    current_retry=current_retry,
                    strategy=strategy,
                )
                if decision:
                    # 활동 로그: 에러 핸들링 결정
                    log_orchestrator(
                        ActivityType.DECISION,
                        f"에러 처리: {node_id} → {decision.action.value}",
                        details={
                            "error_type": error_type,
                            "action": decision.action.value,
                            "retry_count": current_retry,
                        },
                        execution_id=workflow_id,
                        node_id=node_id,
                    )
                    return decision
            except Exception as e:
                logger.warning(f"[Orchestrator] 에러 핸들링 판단 실패: {e}")

        # 재시도 횟수 체크
        max_retries = 3
        if current_retry >= max_retries:
            if default_action == ErrorAction.RETRY:
                default_action = ErrorAction.SKIP  # 재시도 한도 초과 시 스킵

        # Fallback 전략 결정
        fallback_strategy = None
        if default_action == ErrorAction.FALLBACK and strategy:
            fallback_strategy = self._get_fallback_strategy(strategy)

        # 사용자 알림 필요 여부
        should_notify = default_action == ErrorAction.ABORT or current_retry >= 2

        decision = ErrorDecision(
            action=default_action,
            reason=f"{error_type} 에러 발생",
            retry_count=current_retry,
            max_retries=max_retries,
            fallback_strategy=fallback_strategy,
            should_notify_user=should_notify,
            error_message=str(error)[:200],
        )

        # 활동 로그
        log_orchestrator(
            ActivityType.ERROR,
            f"{node_id}: {default_action.value} ({error_type})",
            details={
                "error": str(error)[:100],
                "retry": current_retry,
            },
            execution_id=workflow_id,
            node_id=node_id,
        )

        return decision

    def _classify_error_legacy(self, error_str: str) -> str:
        """
        [DEPRECATED] 레거시 에러 분류

        Note: 새로운 워크플로우에서는 VLM이 [ERROR:TYPE] 형식으로 에러를 보고하므로
        이 메서드는 사용되지 않습니다.
        """
        error_lower = error_str.lower()

        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        elif "element" in error_lower or "not found" in error_lower or "selector" in error_lower:
            return "element_not_found"
        elif "navigation" in error_lower or "navigate" in error_lower:
            return "navigation_failed"
        elif "api" in error_lower or "rate limit" in error_lower or "quota" in error_lower:
            return "api_error"
        return "unknown"

    def _get_fallback_strategy(self, current: ExecutionStrategy) -> Optional[ExecutionStrategy]:
        """폴백 전략 반환"""
        fallbacks = {
            ExecutionStrategy.LOCAL_MODEL: ExecutionStrategy.CLOUD_LIGHT,
            ExecutionStrategy.CLOUD_LIGHT: ExecutionStrategy.CLOUD_HEAVY,
            ExecutionStrategy.CLOUD_HEAVY: None,  # 이미 최고 수준
        }
        return fallbacks.get(current)

    # ===============================
    # Step-level Intervention (스텝별 개입)
    # ===============================

    # 스텝별 학습 저장소
    _step_patterns: Dict[str, List[Dict[str, Any]]] = {}  # node_id -> 학습된 패턴들
    _prompt_injections: Dict[str, str] = {}  # 동적 프롬프트 주입

    # 상황별 프롬프트 템플릿
    SITUATION_PROMPTS = {
        "slow_loading": "페이지 로딩이 느립니다. 5초 정도 기다린 후 다시 시도하세요.",
        "element_not_visible": "요소가 보이지 않습니다. 스크롤하거나 페이지를 새로고침하세요.",
        "popup_detected": "팝업이 감지되었습니다. 닫기 버튼을 찾아 클릭하세요.",
        "login_required": "로그인이 필요합니다. 로그인 페이지로 이동하세요.",
        "captcha_warning": "캡차가 감지될 수 있습니다. 천천히 자연스럽게 행동하세요.",
        "page_changed": "페이지가 변경되었습니다. 현재 화면을 다시 분석하세요.",
        "error_recovery": "오류가 발생했습니다. 뒤로 가기 후 다시 시도하세요.",
    }

    def evaluate_step(
        self,
        workflow_id: str,
        node_id: str,
        step_number: int,
        thought: Optional[str],
        action: Optional[str],
        observation: Optional[str],
        screenshot_analysis: Optional[str] = None,
    ) -> StepFeedback:
        """
        각 VLM 스텝 실행 후 평가 및 피드백

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            step_number: 스텝 번호
            thought: VLM의 사고
            action: 실행한 액션
            observation: 관찰 결과
            screenshot_analysis: 스크린샷 분석 결과 (optional)

        Returns:
            StepFeedback: 다음 스텝에 대한 피드백
        """
        # 0. 메모리 기반 실패 패턴 체크 (최우선)
        if hasattr(self, "_node_failure_patterns_cache") and node_id in self._node_failure_patterns_cache:
            known_patterns = self._node_failure_patterns_cache[node_id]
            for text in [thought, observation, screenshot_analysis]:
                if text:
                    for pattern in known_patterns:
                        if pattern.lower() in text.lower():
                            log_orchestrator(
                                ActivityType.ERROR,
                                f"메모리 기반 얼리 스탑: {pattern}",
                                details={"pattern": pattern, "node_id": node_id, "source": "memory"},
                                execution_id=workflow_id,
                                node_id=node_id,
                            )
                            return StepFeedback(
                                action=StepAction.STOP,
                                reason=f"이전에 학습된 치명적 실패 패턴 감지: {pattern}",
                                learned_pattern=pattern,
                            )

        # Note: Rule-based 봇 감지/실패 패턴 체크 제거됨
        # VLM이 스크린샷을 보고 직접 [ERROR:TYPE] 형식으로 에러를 보고함
        # LangGraph 조건부 엣지에서 에러 타입별 라우팅 처리

        # 반복 상황 트래킹용
        tracking_key = f"{workflow_id}:{node_id}"
        if not hasattr(self, "_failure_tracking"):
            self._failure_tracking = {}

        # 1. 상황별 프롬프트 주입 체크 (반복 감지 포함)
        feedback = self._check_situation_and_inject(
            thought=thought,
            observation=observation,
            action=action,
        )
        if feedback:
            # Track situation repetition
            situation_key = f"{tracking_key}:situation"
            situation_tracking = self._failure_tracking.get(situation_key, {"count": 0, "last_reason": ""})
            
            if situation_tracking["last_reason"] == feedback.reason:
                situation_tracking["count"] += 1
            else:
                situation_tracking["count"] = 1
                situation_tracking["last_reason"] = feedback.reason
            
            self._failure_tracking[situation_key] = situation_tracking
            
            # 3회 이상 같은 상황 반복 시 중단 (Early Stop)
            if situation_tracking["count"] >= 3:
                 log_orchestrator(
                    ActivityType.ERROR,
                    f"반복적인 상황 감지 중단: {feedback.reason} ({situation_tracking['count']}회)",
                    details={"reason": feedback.reason, "node_id": node_id},
                    execution_id=workflow_id,
                    node_id=node_id,
                 )
                 return StepFeedback(
                     action=StepAction.STOP,
                     reason=f"반복적인 상황 해결 실패: {feedback.reason}",
                     learned_pattern=feedback.reason,
                     save_to_memory={"pattern": feedback.reason, "reason": f"반복적인 상황 ({situation_tracking['count']}회)"}
                 )

            log_orchestrator(
                ActivityType.WARNING,
                f"상황별 개입: {feedback.reason} (스텝 {step_number})",
                details={"reason": feedback.reason, "node_id": node_id},
                execution_id=workflow_id,
                node_id=node_id,
            )
            return feedback
        
        # 상황이 없으면 상황 트래킹 초기화
        if f"{tracking_key}:situation" in self._failure_tracking:
             self._failure_tracking[f"{tracking_key}:situation"] = {"count": 0, "last_reason": ""}

        # 4. 학습된 패턴 기반 조언
        hint = self._get_learned_hint(node_id, step_number, thought, action)

        # 5. 정상 진행
        log_orchestrator(
            ActivityType.INFO,
            f"스텝 {step_number} 정상",
            details={"action": action[:50] if action else None},
            execution_id=workflow_id,
            node_id=node_id,
        )

        return StepFeedback(
            action=StepAction.CONTINUE,
            reason="정상 진행",
            next_step_hint=hint,
        )

    def _check_situation_and_inject(
        self,
        thought: Optional[str],
        observation: Optional[str],
        action: Optional[str],
    ) -> Optional[StepFeedback]:
        """상황 감지 및 프롬프트 주입"""
        combined_text = " ".join(filter(None, [thought, observation])).lower()

        # 로딩 느림 감지
        if any(kw in combined_text for kw in ["loading", "로딩", "느림", "waiting", "기다"]):
            return StepFeedback(
                action=StepAction.INJECT_PROMPT,
                reason="느린 로딩 감지",
                injected_prompt=self.SITUATION_PROMPTS["slow_loading"],
            )

        # 요소 찾기 실패
        if any(kw in combined_text for kw in ["not found", "찾을 수 없", "보이지 않", "not visible"]):
            return StepFeedback(
                action=StepAction.INJECT_PROMPT,
                reason="요소 미발견",
                injected_prompt=self.SITUATION_PROMPTS["element_not_visible"],
            )

        # 팝업 감지
        if any(kw in combined_text for kw in ["popup", "팝업", "modal", "모달", "dialog", "알림"]):
            return StepFeedback(
                action=StepAction.INJECT_PROMPT,
                reason="팝업 감지",
                injected_prompt=self.SITUATION_PROMPTS["popup_detected"],
            )

        # 로그인 필요
        if any(kw in combined_text for kw in ["login", "로그인", "sign in", "인증"]):
            return StepFeedback(
                action=StepAction.INJECT_PROMPT,
                reason="로그인 필요",
                injected_prompt=self.SITUATION_PROMPTS["login_required"],
            )

        return None

    async def save_failure_to_memory(self, workflow_id: str, node_id: str, pattern: str, reason: str):
        """실패 패턴 메모리 저장"""
        if self._letta:
            await self._letta.add_failure_pattern(workflow_id, node_id, pattern, reason)

    def _get_learned_hint(
        self,
        node_id: str,
        step_number: int,
        thought: Optional[str],
        action: Optional[str],
    ) -> Optional[str]:
        """학습된 패턴 기반 힌트 제공"""
        if node_id not in self._step_patterns:
            return None

        patterns = self._step_patterns[node_id]
        for pattern in patterns:
            if pattern.get("step_number") == step_number:
                return pattern.get("hint")

        return None

    def learn_from_step(
        self,
        node_id: str,
        step_number: int,
        success: bool,
        thought: Optional[str],
        action: Optional[str],
        observation: Optional[str],
    ):
        """
        스텝 실행 결과로부터 학습

        성공/실패 패턴을 저장하여 향후 같은 상황에서 더 나은 결정
        """
        if node_id not in self._step_patterns:
            self._step_patterns[node_id] = []

        pattern_entry = {
            "step_number": step_number,
            "success": success,
            "action_type": self._extract_action_type(action),
            "context_keywords": self._extract_keywords(thought, observation),
            "timestamp": time.time(),
        }

        # 실패한 경우 힌트 추가 및 개선 기록
        if not success and action:
            pattern_entry["hint"] = f"이전에 '{action[:30]}' 액션이 실패했습니다. 다른 방법을 시도하세요."

            # 개선 이력에 실패 패턴 기록
            self.record_improvement(
                improvement_type="failure_pattern_learned",
                description=f"스텝 {step_number} 실패 학습: {action[:30]}",
                details={
                    "step_number": step_number,
                    "action": action[:50] if action else None,
                    "keywords": pattern_entry["context_keywords"][:5],
                },
                node_id=node_id,
            )

        self._step_patterns[node_id].append(pattern_entry)

        # 최대 20개 패턴만 유지
        if len(self._step_patterns[node_id]) > 20:
            self._step_patterns[node_id] = self._step_patterns[node_id][-20:]

        logger.debug(f"[Orchestrator] 학습 저장: {node_id} 스텝 {step_number} (success={success})")

        # 주기적으로 저장 (10개 패턴마다)
        total_patterns = sum(len(p) for p in self._step_patterns.values())
        if total_patterns % 10 == 0:
            self._save_improvement_history()

    def _extract_action_type(self, action: Optional[str]) -> Optional[str]:
        """액션에서 타입 추출"""
        if not action:
            return None
        action_lower = action.lower()
        if "click" in action_lower:
            return "click"
        elif "type" in action_lower or "input" in action_lower:
            return "type"
        elif "scroll" in action_lower:
            return "scroll"
        elif "wait" in action_lower:
            return "wait"
        return "other"

    def _extract_keywords(self, *texts: Optional[str]) -> List[str]:
        """텍스트에서 주요 키워드 추출"""
        keywords = []
        for text in texts:
            if not text:
                continue
            # 간단한 키워드 추출 (실제로는 더 정교한 NLP 사용 가능)
            words = text.lower().split()
            important = [w for w in words if len(w) > 3 and w.isalpha()]
            keywords.extend(important[:5])
        return list(set(keywords))[:10]

    def get_dynamic_system_prompt(
        self,
        node_id: str,
        base_instruction: str,
        step_number: int = 0,
    ) -> str:
        """
        동적 시스템 프롬프트 생성

        기본 instruction에 학습된 패턴, 상황별 조언을 추가
        """
        additions = []

        # 1. 노드별 학습된 조언
        if node_id in self._step_patterns:
            patterns = self._step_patterns[node_id]
            failed_patterns = [p for p in patterns if not p.get("success")]
            if failed_patterns:
                recent_failures = failed_patterns[-3:]  # 최근 3개
                failure_hints = [p.get("hint") for p in recent_failures if p.get("hint")]
                if failure_hints:
                    additions.append("주의사항:")
                    for hint in failure_hints:
                        additions.append(f"  - {hint}")

        # 2. 주입된 프롬프트
        injection_key = f"{node_id}:{step_number}"
        if injection_key in self._prompt_injections:
            additions.append(self._prompt_injections[injection_key])

        # 3. 일반적인 봇 회피 조언 (항상 추가)
        additions.append("\n[자연스러운 행동 가이드]")
        additions.append("- 각 액션 사이에 자연스러운 간격을 두세요")
        additions.append("- 한 번에 너무 많은 작업을 하지 마세요")
        additions.append("- 페이지가 완전히 로드될 때까지 기다리세요")

        if additions:
            return base_instruction + "\n\n" + "\n".join(additions)

        return base_instruction

    def inject_prompt_for_next_step(
        self,
        node_id: str,
        step_number: int,
        prompt: str,
    ):
        """다음 스텝에 프롬프트 주입"""
        injection_key = f"{node_id}:{step_number + 1}"
        self._prompt_injections[injection_key] = prompt
        logger.info(f"[Orchestrator] 프롬프트 주입 예약: {injection_key}")

    def clear_injections(self, node_id: str):
        """노드의 모든 주입된 프롬프트 제거"""
        keys_to_remove = [k for k in self._prompt_injections if k.startswith(f"{node_id}:")]
        for k in keys_to_remove:
            del self._prompt_injections[k]

    def validate_execution_result(
        self,
        workflow_id: str,
        node_id: str,
        result_data: Dict[str, Any],
        final_answer: Optional[str] = None,
        last_observation: Optional[str] = None,
        last_thought: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        VLM 실행 결과 검증 - VLM이 보고한 [ERROR:TYPE] 에러 확인

        Note: Rule-based 패턴 체크 제거됨.
              VLM이 스크린샷을 보고 직접 [ERROR:TYPE] 형식으로 보고함.

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            result_data: 실행 결과 데이터
            final_answer: final_answer 텍스트
            last_observation: 마지막 observation
            last_thought: 마지막 thought

        Returns:
            Tuple[is_valid, error_type, error_message]
            - is_valid: 결과가 유효한지
            - error_type: 에러 타입 (VLM이 보고한 타입)
            - error_message: 에러 메시지
        """
        import re

        # VLM이 보고한 에러 타입 체크 ([ERROR:TYPE] 형식)
        texts_to_check = [final_answer, last_observation, last_thought]

        for text in texts_to_check:
            if not text:
                continue

            # [ERROR:TYPE] 패턴 검색
            error_match = re.search(r'\[ERROR:(\w+)\]', text, re.IGNORECASE)
            if error_match:
                error_type = error_match.group(1).lower()
                logger.warning(f"[Orchestrator] VLM 에러 보고: node={node_id}, type={error_type}")
                log_orchestrator(
                    ActivityType.ERROR,
                    f"VLM 에러 보고: {node_id}",
                    details={
                        "error_type": error_type,
                        "text_preview": text[:100],
                    },
                    execution_id=workflow_id,
                    node_id=node_id,
                )
                return False, error_type, f"VLM 에러 보고: {error_type}"

        # 에러 없음 - 유효한 결과
        return True, None, None

    async def _query_error_handling(
        self,
        node_id: str,
        error_str: str,
        error_type: str,
        current_retry: int,
        strategy: Optional[ExecutionStrategy],
    ) -> Optional[ErrorDecision]:
        """Orchestrator-8B로 에러 핸들링 결정 질의"""
        system_prompt = """You are an error handler for workflow nodes. Decide how to handle errors.

Actions:
- "retry": Try again (max 3 times)
- "skip": Skip this node and continue
- "abort": Stop the workflow
- "fallback": Use a different/better model

Consider:
- Timeout errors: Usually retry works
- Element not found: Often means page changed, might skip
- API errors: Fallback to different model
- Critical errors: Abort workflow

Respond JSON only:
{
  "action": "retry|skip|abort|fallback",
  "reason": "brief reason",
  "notify_user": true|false
}"""

        user_content = f"""Node: {node_id}
Error: {error_str[:200]}
Error type: {error_type}
Current retry: {current_retry}/3
Current strategy: {strategy.value if strategy else 'unknown'}

Decide how to handle this error (JSON only):"""

        try:
            client = await self._get_http_client()
            response = await client.post(
                self._orchestrator_url,
                json={
                    "model": "orchestrator-8b",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 80,
                },
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # JSON 파싱
            if "```" in content:
                json_str = content.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)
            action_str = data.get("action", "retry")
            action_map = {
                "retry": ErrorAction.RETRY,
                "skip": ErrorAction.SKIP,
                "abort": ErrorAction.ABORT,
                "fallback": ErrorAction.FALLBACK,
            }
            action = action_map.get(action_str, ErrorAction.RETRY)

            return ErrorDecision(
                action=action,
                reason=data.get("reason", "Orchestrator 결정"),
                retry_count=current_retry,
                max_retries=3,
                fallback_strategy=self._get_fallback_strategy(strategy) if action == ErrorAction.FALLBACK else None,
                should_notify_user=data.get("notify_user", False),
                error_message=error_str[:200],
            )

        except Exception as e:
            logger.warning(f"[Orchestrator] 에러 핸들링 질의 실패: {e}")
            return None

    # ===============================
    # Workflow Monitoring & Reporting
    # ===============================

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
                logger.warning(f"[Orchestrator] {node_id} stuck 감지: {elapsed:.1f}초 경과")
                return True

        return False

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
            return f"✅ 워크플로우 완료: {total}개 노드 모두 성공 ({duration_sec:.1f}초, ${total_cost:.4f})"
        elif success > 0:
            return f"⚠️ 부분 완료: {success}/{total} 노드 성공 ({duration_sec:.1f}초, ${total_cost:.4f})"
        else:
            return f"❌ 실패: 모든 노드 실패 ({duration_sec:.1f}초)"

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

    async def close(self):
        """리소스 정리"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# 싱글톤 인스턴스
_orchestrator: Optional[OrchestratorService] = None


def get_orchestrator_service(
    prefer_local: bool = True,
    cost_weight: float = 0.5,
    use_orchestrator_model: bool = True,
) -> OrchestratorService:
    """Orchestrator 서비스 싱글톤 반환"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService(
            prefer_local=prefer_local,
            cost_weight=cost_weight,
            use_orchestrator_model=use_orchestrator_model,
        )
    return _orchestrator
