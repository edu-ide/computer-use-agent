"""
LangGraph 기반 에러 핸들링

기존 Orchestrator의 규칙 기반 에러 분류를 LangGraph 조건부 엣지로 대체합니다.
VLM이 [ERROR:TYPE] 형식으로 보고한 에러를 기반으로 라우팅합니다.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from .workflow_base import VLMErrorType, WorkflowState


class ErrorAction(str, Enum):
    """에러 발생 시 액션 (LangGraph 라우팅용)"""
    RETRY = "retry"  # 재시도
    SKIP = "skip"  # 이 노드 건너뛰기
    ABORT = "abort"  # 워크플로우 중단
    FALLBACK = "fallback"  # 다른 전략으로 재시도


@dataclass
class ErrorHandlingConfig:
    """에러 핸들링 설정"""
    max_retries: int = 3
    retry_delay_sec: float = 2.0

    # VLM 에러 타입별 기본 액션
    error_actions: Dict[VLMErrorType, ErrorAction] = field(default_factory=lambda: {
        VLMErrorType.NONE: ErrorAction.RETRY,  # 일반 실패는 재시도
        VLMErrorType.BOT_DETECTED: ErrorAction.ABORT,  # 봇 감지 시 즉시 중단
        VLMErrorType.PAGE_FAILED: ErrorAction.RETRY,  # 페이지 로딩 실패는 재시도
        VLMErrorType.ACCESS_DENIED: ErrorAction.ABORT,  # 접근 거부 시 중단
        VLMErrorType.ELEMENT_NOT_FOUND: ErrorAction.SKIP,  # 요소 못찾으면 스킵
        VLMErrorType.TIMEOUT: ErrorAction.RETRY,  # 타임아웃은 재시도
        VLMErrorType.UNKNOWN: ErrorAction.ABORT,  # 알 수 없는 에러는 중단
    })


def create_error_router(config: Optional[ErrorHandlingConfig] = None) -> Callable:
    """
    LangGraph 조건부 엣지용 에러 라우터 생성

    Returns:
        에러 타입에 따라 다음 노드를 결정하는 함수
    """
    if config is None:
        config = ErrorHandlingConfig()

    def error_router(state: WorkflowState) -> str:
        """
        에러 타입에 따른 라우팅 결정

        LangGraph conditional_edge에서 사용됩니다.
        """
        vlm_error = state.get("vlm_error_type")
        retry_count = state.get("retry_count", 0)

        # VLM 에러 타입 파싱
        error_type = VLMErrorType.UNKNOWN
        if vlm_error:
            try:
                error_type = VLMErrorType(vlm_error.lower())
            except ValueError:
                # 문자열로 매핑 시도
                error_mapping = {
                    "BOT_DETECTED": VLMErrorType.BOT_DETECTED,
                    "PAGE_FAILED": VLMErrorType.PAGE_FAILED,
                    "ACCESS_DENIED": VLMErrorType.ACCESS_DENIED,
                    "ELEMENT_NOT_FOUND": VLMErrorType.ELEMENT_NOT_FOUND,
                    "TIMEOUT": VLMErrorType.TIMEOUT,
                }
                error_type = error_mapping.get(vlm_error, VLMErrorType.UNKNOWN)

        # 기본 액션 결정
        action = config.error_actions.get(error_type, ErrorAction.ABORT)

        # 재시도 횟수 체크
        if action == ErrorAction.RETRY and retry_count >= config.max_retries:
            action = ErrorAction.SKIP  # 재시도 한도 초과 시 스킵

        return action.value

    return error_router


def get_error_edge_mapping(
    retry_node: str = "retry",
    skip_node: str = "next",
    abort_node: str = "error_handler",
    fallback_node: str = "fallback",
) -> Dict[str, str]:
    """
    에러 액션별 노드 매핑 생성

    Args:
        retry_node: 재시도할 때 이동할 노드
        skip_node: 스킵할 때 이동할 노드
        abort_node: 중단할 때 이동할 노드
        fallback_node: 폴백할 때 이동할 노드

    Returns:
        LangGraph conditional_edges에 사용할 매핑
    """
    return {
        ErrorAction.RETRY.value: retry_node,
        ErrorAction.SKIP.value: skip_node,
        ErrorAction.ABORT.value: abort_node,
        ErrorAction.FALLBACK.value: fallback_node,
    }


# 에러 분류 유틸리티 함수 (문자열 기반 분류는 VLM 에러 감지 이전에만 사용)
def classify_error_from_string(error_str: str) -> VLMErrorType:
    """
    에러 문자열에서 에러 타입 추론 (레거시 호환용)

    Note: VLM이 [ERROR:TYPE] 형식으로 보고하지 않은 경우에만 사용
    """
    error_lower = error_str.lower()

    if "timeout" in error_lower or "timed out" in error_lower:
        return VLMErrorType.TIMEOUT
    elif "captcha" in error_lower or "robot" in error_lower or "blocked" in error_lower:
        return VLMErrorType.BOT_DETECTED
    elif "403" in error_lower or "forbidden" in error_lower or "denied" in error_lower:
        return VLMErrorType.ACCESS_DENIED
    elif "element" in error_lower or "not found" in error_lower or "selector" in error_lower:
        return VLMErrorType.ELEMENT_NOT_FOUND
    elif "network" in error_lower or "connection" in error_lower or "load" in error_lower:
        return VLMErrorType.PAGE_FAILED

    return VLMErrorType.UNKNOWN
