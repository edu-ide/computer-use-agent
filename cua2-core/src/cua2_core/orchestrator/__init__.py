"""
Magentic-One 스타일 Orchestrator 패키지

Task Ledger와 Progress Ledger를 사용한 Dual-Loop 시스템 구현.
로컬 LLM (Orchestrator-8B, Fara-7B)과 통합.

주요 컴포넌트:
- Orchestrator: 메인 오케스트레이터 클래스
- OrchestratorConfig: 설정 클래스
- LLMClient: 로컬 LLM 클라이언트
"""

from .config import OrchestratorConfig
from .orchestrator import AgentInfo, LLMClient, Orchestrator, OrchestratorState
from .prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_SYSTEM_MESSAGE_EXECUTION,
    get_orchestrator_plan_prompt_json,
    get_orchestrator_progress_ledger_prompt,
    get_orchestrator_system_message_planning,
    validate_ledger_json,
    validate_plan_json,
)
from .sentinel_prompts import (
    ORCHESTRATOR_SENTINEL_CONDITION_CHECK_PROMPT,
    validate_sentinel_condition_check_json,
)

__all__ = [
    # Main classes
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
    "LLMClient",
    "AgentInfo",
    # Prompts
    "ORCHESTRATOR_FINAL_ANSWER_PROMPT",
    "ORCHESTRATOR_SYSTEM_MESSAGE_EXECUTION",
    "ORCHESTRATOR_SENTINEL_CONDITION_CHECK_PROMPT",
    "get_orchestrator_plan_prompt_json",
    "get_orchestrator_progress_ledger_prompt",
    "get_orchestrator_system_message_planning",
    "validate_ledger_json",
    "validate_plan_json",
    "validate_sentinel_condition_check_json",
]
