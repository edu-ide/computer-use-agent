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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .trace_store import get_trace_store, TraceStore
from .node_reuse_analyzer import get_node_reuse_analyzer, NodeReuseAnalyzer, ReuseDecision
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

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        reuse_analyzer: Optional[NodeReuseAnalyzer] = None,
        prefer_local: bool = True,
        cost_weight: float = 0.5,
        use_orchestrator_model: bool = True,
        orchestrator_url: Optional[str] = None,
    ):
        """
        Args:
            trace_store: Trace 캐시 저장소
            reuse_analyzer: 재사용 분석기
            prefer_local: 로컬 모델 우선 여부
            cost_weight: 비용 가중치 (높을수록 저렴한 옵션 선호)
            use_orchestrator_model: Orchestrator-8B 모델 사용 여부
            orchestrator_url: Orchestrator API URL (기본: localhost:8081)
        """
        self._trace_store = trace_store or get_trace_store()
        self._reuse_analyzer = reuse_analyzer or get_node_reuse_analyzer()
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
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[Orchestrator] {node_id}: {decision.strategy.value} "
            f"(model={decision.model_id}, complexity={complexity.complexity_score:.2f}, "
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
    ) -> ExecutionDecision:
        """최적 실행 전략 선택"""

        score = complexity.complexity_score

        # 1. 규칙 기반으로 처리 가능한 경우 (가장 빠름)
        if self._can_use_rule_based(node_id, instruction):
            return ExecutionDecision(
                strategy=ExecutionStrategy.RULE_BASED,
                model_id=None,
                reason="규칙 기반으로 처리 가능",
                estimated_time_ms=50,
                estimated_cost=0.0,
                confidence=0.95,
            )

        # 2. 복잡도에 따른 모델 선택
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
                )
            else:
                return ExecutionDecision(
                    strategy=ExecutionStrategy.CLOUD_LIGHT,
                    model_id="gpt-4o-mini",
                    reason=f"단순 작업 (complexity={score:.2f})",
                    estimated_time_ms=self.MODELS["gpt-4o-mini"].avg_latency_ms,
                    estimated_cost=self.MODELS["gpt-4o-mini"].cost_per_1k_tokens * 2,
                    confidence=0.9,
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
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionDecision:
        """
        비동기 노드 실행 전략 결정 (Orchestrator-8B 사용)

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            instruction: 노드 instruction
            params: 실행 파라미터
            node_config: 노드 설정 (WorkflowNode)
            execution_history: 최근 실행 이력 (컨텍스트 제공)

        Returns:
            ExecutionDecision: 실행 결정
        """
        start_time = time.time()

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
                execution_id=workflow_id,
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
                        execution_id=workflow_id,
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
                    execution_id=workflow_id,
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
            execution_id=workflow_id,
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

    # 에러 타입별 기본 액션
    ERROR_ACTIONS = {
        "timeout": ErrorAction.RETRY,
        "network": ErrorAction.RETRY,
        "element_not_found": ErrorAction.SKIP,
        "navigation_failed": ErrorAction.RETRY,
        "api_error": ErrorAction.FALLBACK,
        "unknown": ErrorAction.ABORT,
    }

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
        에러 발생 시 처리 방법 결정

        Args:
            workflow_id: 워크플로우 ID
            node_id: 노드 ID
            error: 발생한 에러
            current_retry: 현재 재시도 횟수
            strategy: 현재 사용 중인 전략

        Returns:
            ErrorDecision: 에러 처리 결정
        """
        error_str = str(error).lower()
        error_type = self._classify_error(error_str)

        # 기본 액션 결정
        default_action = self.ERROR_ACTIONS.get(error_type, ErrorAction.ABORT)

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

    def _classify_error(self, error_str: str) -> str:
        """에러 분류"""
        if "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "network" in error_str or "connection" in error_str:
            return "network"
        elif "element" in error_str or "not found" in error_str or "selector" in error_str:
            return "element_not_found"
        elif "navigation" in error_str or "navigate" in error_str:
            return "navigation_failed"
        elif "api" in error_str or "rate limit" in error_str or "quota" in error_str:
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
