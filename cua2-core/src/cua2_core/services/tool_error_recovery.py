"""
Tool Error Recovery

smolagents 도구 실패 시 자동 복구 전략:
- 재시도 (exponential backoff)
- 대체 도구 사용
- 상태 롤백
- 사용자 개입 요청
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RecoveryStrategy(str, Enum):
    """복구 전략"""
    RETRY = "retry"  # 재시도
    ALTERNATIVE = "alternative"  # 대체 도구 사용
    ROLLBACK = "rollback"  # 상태 롤백
    SKIP = "skip"  # 건너뛰기
    USER_INPUT = "user_input"  # 사용자 개입 요청
    ABORT = "abort"  # 중단


class ToolErrorType(str, Enum):
    """도구 에러 타입"""
    TIMEOUT = "timeout"  # 타임아웃
    ELEMENT_NOT_FOUND = "element_not_found"  # 요소 미발견
    NETWORK_ERROR = "network_error"  # 네트워크 에러
    PERMISSION_DENIED = "permission_denied"  # 권한 에러
    INVALID_INPUT = "invalid_input"  # 잘못된 입력
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # 리소스 없음
    RATE_LIMITED = "rate_limited"  # API 제한
    UNKNOWN = "unknown"  # 알 수 없는 에러


@dataclass
class ToolError:
    """도구 에러 정보"""
    tool_name: str
    error_type: ToolErrorType
    error_message: str
    timestamp: str
    attempt: int = 1
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """복구 액션"""
    strategy: RecoveryStrategy
    tool_name: Optional[str] = None  # 대체 도구 이름
    wait_seconds: float = 0  # 대기 시간
    modified_params: Optional[Dict[str, Any]] = None  # 수정된 파라미터
    message: Optional[str] = None  # 사용자 메시지
    rollback_to_step: Optional[int] = None  # 롤백할 스텝 번호


@dataclass
class RecoveryResult:
    """복구 결과"""
    success: bool
    action_taken: RecoveryAction
    new_result: Optional[Any] = None
    error: Optional[str] = None


class ToolErrorRecovery:
    """
    도구 에러 복구 서비스

    smolagents 도구 실행 중 발생하는 에러를 감지하고 자동 복구합니다.

    Example:
        ```python
        recovery = ToolErrorRecovery()

        # 도구별 대체 도구 설정
        recovery.register_alternative("click", "safe_click")
        recovery.register_alternative("type_text", "smart_type")

        # 복구 콜백 설정
        recovery.on_recovery(async_callback)

        # 에러 발생 시 복구
        action = recovery.decide_recovery(error)
        if action.strategy == RecoveryStrategy.RETRY:
            await asyncio.sleep(action.wait_seconds)
            result = await retry_tool(...)
        ```
    """

    # 기본 재시도 설정
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BACKOFF_BASE = 1.0  # 초
    DEFAULT_BACKOFF_MULTIPLIER = 2.0

    # 에러 타입별 기본 전략
    DEFAULT_STRATEGIES: Dict[ToolErrorType, RecoveryStrategy] = {
        ToolErrorType.TIMEOUT: RecoveryStrategy.RETRY,
        ToolErrorType.ELEMENT_NOT_FOUND: RecoveryStrategy.ALTERNATIVE,
        ToolErrorType.NETWORK_ERROR: RecoveryStrategy.RETRY,
        ToolErrorType.PERMISSION_DENIED: RecoveryStrategy.ABORT,
        ToolErrorType.INVALID_INPUT: RecoveryStrategy.SKIP,
        ToolErrorType.RESOURCE_UNAVAILABLE: RecoveryStrategy.RETRY,
        ToolErrorType.RATE_LIMITED: RecoveryStrategy.RETRY,
        ToolErrorType.UNKNOWN: RecoveryStrategy.ABORT,
    }

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_multiplier = backoff_multiplier

        # 도구별 대체 도구 매핑
        self._alternatives: Dict[str, List[str]] = {}

        # 에러 타입별 커스텀 전략
        self._custom_strategies: Dict[ToolErrorType, RecoveryStrategy] = {}

        # 에러 히스토리
        self._error_history: List[ToolError] = []

        # 콜백
        self._recovery_callback: Optional[Callable] = None
        self._user_input_callback: Optional[Callable] = None

    def register_alternative(self, tool_name: str, alternative: str):
        """
        대체 도구 등록

        Args:
            tool_name: 원본 도구 이름
            alternative: 대체 도구 이름

        Example:
            ```python
            recovery.register_alternative("click", "safe_click")
            recovery.register_alternative("click", "double_click")
            ```
        """
        if tool_name not in self._alternatives:
            self._alternatives[tool_name] = []
        if alternative not in self._alternatives[tool_name]:
            self._alternatives[tool_name].append(alternative)
        logger.info(f"[ToolErrorRecovery] 대체 도구 등록: {tool_name} -> {alternative}")

    def set_strategy(self, error_type: ToolErrorType, strategy: RecoveryStrategy):
        """
        에러 타입별 복구 전략 설정

        Args:
            error_type: 에러 타입
            strategy: 복구 전략
        """
        self._custom_strategies[error_type] = strategy
        logger.info(f"[ToolErrorRecovery] 전략 설정: {error_type} -> {strategy}")

    def on_recovery(self, callback: Callable):
        """복구 시도 콜백 설정"""
        self._recovery_callback = callback

    def on_user_input_needed(self, callback: Callable):
        """사용자 입력 필요 시 콜백 설정"""
        self._user_input_callback = callback

    def classify_error(self, error_message: str, tool_name: str) -> ToolErrorType:
        """
        에러 메시지를 분석하여 에러 타입 분류

        Args:
            error_message: 에러 메시지
            tool_name: 도구 이름

        Returns:
            에러 타입
        """
        error_lower = error_message.lower()

        # 타임아웃
        if any(kw in error_lower for kw in ["timeout", "timed out", "시간 초과"]):
            return ToolErrorType.TIMEOUT

        # 요소 미발견
        if any(kw in error_lower for kw in [
            "not found", "element not found", "no such element",
            "찾을 수 없", "없습니다", "selector"
        ]):
            return ToolErrorType.ELEMENT_NOT_FOUND

        # 네트워크 에러
        if any(kw in error_lower for kw in [
            "network", "connection", "dns", "socket",
            "네트워크", "연결"
        ]):
            return ToolErrorType.NETWORK_ERROR

        # 권한 에러
        if any(kw in error_lower for kw in [
            "permission", "access denied", "forbidden", "403",
            "권한", "거부"
        ]):
            return ToolErrorType.PERMISSION_DENIED

        # 잘못된 입력
        if any(kw in error_lower for kw in [
            "invalid", "argument", "parameter", "type error",
            "잘못된", "유효하지"
        ]):
            return ToolErrorType.INVALID_INPUT

        # 리소스 없음
        if any(kw in error_lower for kw in [
            "resource", "unavailable", "not available", "404",
            "리소스", "없음"
        ]):
            return ToolErrorType.RESOURCE_UNAVAILABLE

        # API 제한
        if any(kw in error_lower for kw in [
            "rate limit", "too many", "throttle", "429",
            "제한", "초과"
        ]):
            return ToolErrorType.RATE_LIMITED

        return ToolErrorType.UNKNOWN

    def decide_recovery(
        self,
        tool_name: str,
        error_message: str,
        attempt: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """
        에러에 대한 복구 전략 결정

        Args:
            tool_name: 실패한 도구 이름
            error_message: 에러 메시지
            attempt: 현재 시도 횟수
            context: 추가 컨텍스트

        Returns:
            RecoveryAction: 복구 액션
        """
        # 에러 분류
        error_type = self.classify_error(error_message, tool_name)

        # 에러 기록
        error = ToolError(
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            attempt=attempt,
            context=context or {},
        )
        self._error_history.append(error)

        logger.info(
            f"[ToolErrorRecovery] 에러 분류: {tool_name} - {error_type.value} "
            f"(attempt {attempt})"
        )

        # 전략 결정
        strategy = self._custom_strategies.get(
            error_type,
            self.DEFAULT_STRATEGIES.get(error_type, RecoveryStrategy.ABORT)
        )

        # 전략별 액션 생성
        if strategy == RecoveryStrategy.RETRY:
            if attempt >= self.max_retries:
                # 최대 재시도 초과 - 대체 도구 시도 또는 중단
                if tool_name in self._alternatives and self._alternatives[tool_name]:
                    return RecoveryAction(
                        strategy=RecoveryStrategy.ALTERNATIVE,
                        tool_name=self._alternatives[tool_name][0],
                        message=f"최대 재시도 초과. 대체 도구 사용: {self._alternatives[tool_name][0]}",
                    )
                return RecoveryAction(
                    strategy=RecoveryStrategy.ABORT,
                    message=f"최대 재시도 횟수({self.max_retries}) 초과",
                )

            # Exponential backoff
            wait_time = self.backoff_base * (self.backoff_multiplier ** (attempt - 1))
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                wait_seconds=wait_time,
                message=f"재시도 {attempt + 1}/{self.max_retries + 1} ({wait_time:.1f}초 후)",
            )

        elif strategy == RecoveryStrategy.ALTERNATIVE:
            alternatives = self._alternatives.get(tool_name, [])
            if alternatives:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ALTERNATIVE,
                    tool_name=alternatives[0],
                    message=f"대체 도구 사용: {alternatives[0]}",
                )
            # 대체 도구 없음 - 스킵
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="대체 도구 없음, 건너뛰기",
            )

        elif strategy == RecoveryStrategy.ROLLBACK:
            return RecoveryAction(
                strategy=RecoveryStrategy.ROLLBACK,
                rollback_to_step=max(0, (context or {}).get("current_step", 0) - 1),
                message="이전 상태로 롤백",
            )

        elif strategy == RecoveryStrategy.SKIP:
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="현재 작업 건너뛰기",
            )

        elif strategy == RecoveryStrategy.USER_INPUT:
            return RecoveryAction(
                strategy=RecoveryStrategy.USER_INPUT,
                message=f"사용자 입력 필요: {error_message}",
            )

        else:  # ABORT
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message=f"복구 불가: {error_message}",
            )

    async def execute_recovery(
        self,
        action: RecoveryAction,
        retry_func: Optional[Callable] = None,
        alternative_func: Optional[Callable] = None,
    ) -> RecoveryResult:
        """
        복구 액션 실행

        Args:
            action: 복구 액션
            retry_func: 재시도 함수
            alternative_func: 대체 도구 실행 함수

        Returns:
            RecoveryResult: 복구 결과
        """
        logger.info(f"[ToolErrorRecovery] 복구 실행: {action.strategy.value}")

        # 콜백 호출
        if self._recovery_callback:
            try:
                await self._recovery_callback(action)
            except Exception as e:
                logger.warning(f"[ToolErrorRecovery] 콜백 실패: {e}")

        if action.strategy == RecoveryStrategy.RETRY:
            if action.wait_seconds > 0:
                await asyncio.sleep(action.wait_seconds)

            if retry_func:
                try:
                    result = await retry_func()
                    return RecoveryResult(
                        success=True,
                        action_taken=action,
                        new_result=result,
                    )
                except Exception as e:
                    return RecoveryResult(
                        success=False,
                        action_taken=action,
                        error=str(e),
                    )

        elif action.strategy == RecoveryStrategy.ALTERNATIVE:
            if alternative_func and action.tool_name:
                try:
                    result = await alternative_func(action.tool_name)
                    return RecoveryResult(
                        success=True,
                        action_taken=action,
                        new_result=result,
                    )
                except Exception as e:
                    return RecoveryResult(
                        success=False,
                        action_taken=action,
                        error=str(e),
                    )

        elif action.strategy == RecoveryStrategy.USER_INPUT:
            if self._user_input_callback:
                try:
                    user_input = await self._user_input_callback(action.message)
                    return RecoveryResult(
                        success=True,
                        action_taken=action,
                        new_result=user_input,
                    )
                except Exception as e:
                    return RecoveryResult(
                        success=False,
                        action_taken=action,
                        error=str(e),
                    )

        elif action.strategy == RecoveryStrategy.SKIP:
            return RecoveryResult(
                success=True,
                action_taken=action,
                new_result=None,
            )

        elif action.strategy == RecoveryStrategy.ROLLBACK:
            # 롤백은 호출자가 처리해야 함
            return RecoveryResult(
                success=True,
                action_taken=action,
                new_result=None,
            )

        # ABORT 또는 기본
        return RecoveryResult(
            success=False,
            action_taken=action,
            error=action.message,
        )

    def get_error_history(self, limit: int = 50) -> List[ToolError]:
        """에러 히스토리 조회"""
        return self._error_history[-limit:]

    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계"""
        stats = {
            "total_errors": len(self._error_history),
            "by_type": {},
            "by_tool": {},
        }

        for error in self._error_history:
            # 타입별 집계
            error_type = error.error_type.value
            stats["by_type"][error_type] = stats["by_type"].get(error_type, 0) + 1

            # 도구별 집계
            stats["by_tool"][error.tool_name] = stats["by_tool"].get(error.tool_name, 0) + 1

        return stats

    def clear_history(self):
        """에러 히스토리 초기화"""
        self._error_history.clear()


# 싱글톤 인스턴스
_tool_error_recovery: Optional[ToolErrorRecovery] = None


def get_tool_error_recovery() -> ToolErrorRecovery:
    """Tool Error Recovery 싱글톤 반환"""
    global _tool_error_recovery
    if _tool_error_recovery is None:
        _tool_error_recovery = ToolErrorRecovery()

        # 기본 대체 도구 등록
        _tool_error_recovery.register_alternative("click", "safe_click")
        _tool_error_recovery.register_alternative("type_text", "smart_type")
        _tool_error_recovery.register_alternative("scroll", "smooth_scroll")

    return _tool_error_recovery
