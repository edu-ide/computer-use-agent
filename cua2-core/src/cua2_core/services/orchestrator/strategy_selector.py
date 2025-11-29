"""
전략 선택 모듈

노드 실행 전 최적의 실행 전략을 결정합니다:
- 캐시 히트 체크
- 복잡도 분석
- 모델 선택
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .types import (
    ExecutionStrategy,
    ExecutionDecision,
    NodeComplexity,
    ModelConfig,
)

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    전략 선택기

    노드의 복잡도를 분석하고 최적의 실행 전략을 결정합니다.
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

    def __init__(
        self,
        prefer_local: bool = True,
        cost_weight: float = 0.5,
    ):
        """
        Args:
            prefer_local: 로컬 모델 우선 여부
            cost_weight: 비용 가중치 (높을수록 저렴한 옵션 선호)
        """
        self._prefer_local = prefer_local
        self._cost_weight = cost_weight

        # 노드별 복잡도 캐시
        self._complexity_cache: Dict[str, NodeComplexity] = {}

    def analyze_complexity(
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

    def select_strategy(
        self,
        node_id: str,
        complexity: NodeComplexity,
        instruction: str,
        learned_settings: Optional[Dict[str, Any]] = None,
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

    def decide(
        self,
        node_id: str,
        instruction: str,
        params: Dict[str, Any],
        node_config: Optional[Any] = None,
        learned_settings: Optional[Dict[str, Any]] = None,
    ) -> ExecutionDecision:
        """
        노드 실행 전략 결정 (간단 버전)

        캐시 체크 없이 복잡도 분석 후 전략 선택
        """
        start_time = time.time()

        # 복잡도 분석
        complexity = self.analyze_complexity(node_id, instruction, node_config)

        # 전략 선택
        decision = self.select_strategy(
            node_id=node_id,
            complexity=complexity,
            instruction=instruction,
            learned_settings=learned_settings,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[StrategySelector] {node_id}: {decision.strategy.value} "
            f"(model={decision.model_id}, complexity={complexity.complexity_score:.2f}, "
            f"decision_time={elapsed_ms}ms)"
        )

        return decision

    def get_fallback_strategy(self, current: ExecutionStrategy) -> Optional[ExecutionStrategy]:
        """폴백 전략 반환"""
        fallbacks = {
            ExecutionStrategy.LOCAL_MODEL: ExecutionStrategy.CLOUD_LIGHT,
            ExecutionStrategy.CLOUD_LIGHT: ExecutionStrategy.CLOUD_HEAVY,
            ExecutionStrategy.CLOUD_HEAVY: None,  # 이미 최고 수준
        }
        return fallbacks.get(current)
