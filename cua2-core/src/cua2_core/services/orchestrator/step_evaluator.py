"""
스텝 평가 모듈

VLM 스텝 실행 결과 평가 및 학습:
- 스텝별 실시간 평가
- 상황별 프롬프트 주입
- 패턴 학습 및 힌트 제공
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
    StepAction,
    StepFeedback,
)

logger = logging.getLogger(__name__)


class StepEvaluator:
    """
    스텝 평가기

    VLM 스텝 실행 결과를 평가하고 피드백을 제공합니다.
    """

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

    def __init__(self, letta_service: Optional[Any] = None):
        """
        Args:
            letta_service: Letta Memory 서비스 (실패 패턴 조회용)
        """
        self._letta = letta_service

        # 스텝별 학습 저장소
        self._step_patterns: Dict[str, List[Dict[str, Any]]] = {}  # node_id -> 학습된 패턴들
        self._prompt_injections: Dict[str, str] = {}  # 동적 프롬프트 주입

        # 실패 트래킹
        self._failure_tracking: Dict[str, Dict[str, Any]] = {}

        # 메모리 캐시
        self._node_failure_patterns_cache: Dict[str, List[str]] = {}

    async def load_failure_patterns(self, workflow_id: str, node_id: str):
        """메모리에서 실패 패턴 로드"""
        if self._letta:
            try:
                patterns = await self._letta.get_failure_patterns(workflow_id, node_id)
                self._node_failure_patterns_cache[node_id] = patterns
                if patterns:
                    logger.info(f"[StepEvaluator] {node_id}: 메모리에서 실패 패턴 로드 ({len(patterns)}개)")
            except Exception as e:
                logger.warning(f"[StepEvaluator] 실패 패턴 로드 실패: {e}")

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
        if node_id in self._node_failure_patterns_cache:
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

        # 1. VLM 에러 패턴 체크 ([ERROR:TYPE] 형식)
        # Note: VLM이 스크린샷을 보고 직접 에러를 보고하므로
        # LangGraph router에서 처리됨 (workflow_base.py의 _create_router)

        # 2. 반복 실패 감지
        tracking_key = f"{workflow_id}:{node_id}"
        feedback = self._check_repetitive_failure(
            tracking_key=tracking_key,
            thought=thought,
            observation=observation,
            workflow_id=workflow_id,
            node_id=node_id,
        )
        if feedback:
            return feedback

        # 3. 상황별 프롬프트 주입 체크 (반복 감지 포함)
        feedback = self._check_situation_and_inject(
            tracking_key=tracking_key,
            thought=thought,
            observation=observation,
            action=action,
            workflow_id=workflow_id,
            node_id=node_id,
        )
        if feedback:
            return feedback

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

    def _check_repetitive_failure(
        self,
        tracking_key: str,
        thought: Optional[str],
        observation: Optional[str],
        workflow_id: str,
        node_id: str,
    ) -> Optional[StepFeedback]:
        """반복 실패 감지"""
        # 일반적인 실패 키워드
        failure_keywords = [
            "failed", "error", "not found", "cannot", "unable",
            "실패", "오류", "찾을 수 없", "불가능",
        ]

        for text in [thought, observation]:
            if text:
                text_lower = text.lower()
                for keyword in failure_keywords:
                    if keyword in text_lower:
                        # Track failure
                        tracking = self._failure_tracking.get(tracking_key, {"count": 0, "last_pattern": ""})
                        if tracking["last_pattern"] == keyword:
                            tracking["count"] += 1
                        else:
                            tracking["count"] = 1
                            tracking["last_pattern"] = keyword
                        self._failure_tracking[tracking_key] = tracking

                        # 2회 이상 반복 시 중단 (Early Stop)
                        if tracking["count"] >= 2:
                            log_orchestrator(
                                ActivityType.ERROR,
                                f"반복적인 실패 감지: {keyword} ({tracking['count']}회)",
                                details={"pattern": keyword, "node_id": node_id},
                                execution_id=workflow_id,
                                node_id=node_id,
                            )
                            return StepFeedback(
                                action=StepAction.STOP,
                                reason=f"반복적인 실패 패턴 감지: {keyword}",
                                learned_pattern=keyword,
                                save_to_memory={"pattern": keyword, "reason": f"반복적인 실패 ({tracking['count']}회)"}
                            )

                        # 첫 번째 실패 - 복구 프롬프트 주입
                        log_orchestrator(
                            ActivityType.WARNING,
                            f"실패 패턴 감지: {keyword}",
                            details={"pattern": keyword, "node_id": node_id},
                            execution_id=workflow_id,
                            node_id=node_id,
                        )
                        return StepFeedback(
                            action=StepAction.INJECT_PROMPT,
                            reason=f"실패 패턴 발견: {keyword}",
                            injected_prompt=self.SITUATION_PROMPTS.get("error_recovery", ""),
                            learned_pattern=keyword,
                        )

        # 실패 패턴이 없으면 트래킹 초기화
        if tracking_key in self._failure_tracking:
            self._failure_tracking[tracking_key] = {"count": 0, "last_pattern": ""}

        return None

    def _check_situation_and_inject(
        self,
        tracking_key: str,
        thought: Optional[str],
        observation: Optional[str],
        action: Optional[str],
        workflow_id: str,
        node_id: str,
    ) -> Optional[StepFeedback]:
        """상황 감지 및 프롬프트 주입"""
        combined_text = " ".join(filter(None, [thought, observation])).lower()

        # 상황 패턴 정의
        situation_patterns = [
            (["loading", "로딩", "느림", "waiting", "기다"], "slow_loading", "느린 로딩 감지"),
            (["not found", "찾을 수 없", "보이지 않", "not visible"], "element_not_visible", "요소 미발견"),
            (["popup", "팝업", "modal", "모달", "dialog", "알림"], "popup_detected", "팝업 감지"),
            (["login", "로그인", "sign in", "인증"], "login_required", "로그인 필요"),
        ]

        for keywords, situation_key, reason in situation_patterns:
            if any(kw in combined_text for kw in keywords):
                # Track situation repetition
                situation_tracking_key = f"{tracking_key}:situation"
                tracking = self._failure_tracking.get(situation_tracking_key, {"count": 0, "last_reason": ""})

                if tracking["last_reason"] == reason:
                    tracking["count"] += 1
                else:
                    tracking["count"] = 1
                    tracking["last_reason"] = reason

                self._failure_tracking[situation_tracking_key] = tracking

                # 3회 이상 같은 상황 반복 시 중단 (Early Stop)
                if tracking["count"] >= 3:
                    log_orchestrator(
                        ActivityType.ERROR,
                        f"반복적인 상황 감지 중단: {reason} ({tracking['count']}회)",
                        details={"reason": reason, "node_id": node_id},
                        execution_id=workflow_id,
                        node_id=node_id,
                    )
                    return StepFeedback(
                        action=StepAction.STOP,
                        reason=f"반복적인 상황 해결 실패: {reason}",
                        learned_pattern=reason,
                        save_to_memory={"pattern": reason, "reason": f"반복적인 상황 ({tracking['count']}회)"}
                    )

                log_orchestrator(
                    ActivityType.WARNING,
                    f"상황별 개입: {reason}",
                    details={"reason": reason, "node_id": node_id},
                    execution_id=workflow_id,
                    node_id=node_id,
                )

                return StepFeedback(
                    action=StepAction.INJECT_PROMPT,
                    reason=reason,
                    injected_prompt=self.SITUATION_PROMPTS.get(situation_key, ""),
                )

        # 상황이 없으면 상황 트래킹 초기화
        situation_tracking_key = f"{tracking_key}:situation"
        if situation_tracking_key in self._failure_tracking:
            self._failure_tracking[situation_tracking_key] = {"count": 0, "last_reason": ""}

        return None

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

        # 실패한 경우 힌트 추가
        if not success and action:
            pattern_entry["hint"] = f"이전에 '{action[:30]}' 액션이 실패했습니다. 다른 방법을 시도하세요."

        self._step_patterns[node_id].append(pattern_entry)

        # 최대 20개 패턴만 유지
        if len(self._step_patterns[node_id]) > 20:
            self._step_patterns[node_id] = self._step_patterns[node_id][-20:]

        logger.debug(f"[StepEvaluator] 학습 저장: {node_id} 스텝 {step_number} (success={success})")

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
            # 간단한 키워드 추출
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

        # 3. 일반적인 행동 가이드
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
        logger.info(f"[StepEvaluator] 프롬프트 주입 예약: {injection_key}")

    def clear_injections(self, node_id: str):
        """노드의 모든 주입된 프롬프트 제거"""
        keys_to_remove = [k for k in self._prompt_injections if k.startswith(f"{node_id}:")]
        for k in keys_to_remove:
            del self._prompt_injections[k]

    async def save_failure_to_memory(self, workflow_id: str, node_id: str, pattern: str, reason: str):
        """실패 패턴 메모리 저장"""
        if self._letta:
            await self._letta.add_failure_pattern(workflow_id, node_id, pattern, reason)

    def get_step_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """학습된 패턴 반환"""
        return self._step_patterns

    def set_step_patterns(self, patterns: Dict[str, List[Dict[str, Any]]]):
        """학습된 패턴 설정 (로드용)"""
        self._step_patterns = patterns
