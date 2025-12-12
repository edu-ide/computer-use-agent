"""
Magentic-UI 스타일 설정

SentinelPlanConfig, ModelClientConfigs, MagenticUIConfig 등
전역 설정 클래스들
"""

from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .types import Plan


class SentinelPlanConfig(BaseModel):
    """Sentinel plan 기능 설정.

    Attributes:
        enable_sentinel_steps (bool): Sentinel plan 스텝 활성화 여부. 기본값: True.
        dynamic_sentinel_sleep (bool): LLM 응답에 따라 sleep duration 동적 조정 여부. 기본값: False.
    """

    enable_sentinel_steps: bool = True
    dynamic_sentinel_sleep: bool = False


class ModelClientConfigs(BaseModel):
    """모델 클라이언트 설정.

    Attributes:
        orchestrator (Optional[Dict[str, Any]]): Orchestrator용 모델 설정.
        web_surfer (Optional[Dict[str, Any]]): WebSurfer용 모델 설정.
        coder (Optional[Dict[str, Any]]): Coder용 모델 설정.
    """

    orchestrator: Optional[Dict[str, Any]] = None
    web_surfer: Optional[Dict[str, Any]] = None
    coder: Optional[Dict[str, Any]] = None

    # 기본 클라이언트 설정 (로컬 SGLang 서버)
    default_client_config: ClassVar[Dict[str, Any]] = {
        "provider": "OpenAIChatCompletionClient",
        "config": {
            "base_url": "http://localhost:30000/v1",
            "model": "Orchestrator-8B",
            "api_key": "EMPTY",
        },
    }

    # Fara-7B용 기본 설정
    default_fara_config: ClassVar[Dict[str, Any]] = {
        "provider": "OpenAIChatCompletionClient",
        "config": {
            "base_url": "http://localhost:30001/v1",
            "model": "Fara-7B",
            "api_key": "EMPTY",
        },
    }

    @classmethod
    def get_default_client_config(cls) -> Dict[str, Any]:
        return cls.default_client_config

    @classmethod
    def get_default_fara_config(cls) -> Dict[str, Any]:
        return cls.default_fara_config


class MagenticUIConfig(BaseModel):
    """
    Magentic-UI 스타일 설정 (간소화 버전).

    Attributes:
        model_client_configs (ModelClientConfigs): 모델 클라이언트 설정.
        cooperative_planning (bool): 공동 계획 모드 활성화. 기본값: True.
        autonomous_execution (bool): 자율 실행 모드. 기본값: False.
        allowed_websites (List[str], optional): 허용된 웹사이트 목록.
        max_actions_per_step (int): 스텝당 최대 액션 수. 기본값: 5.
        max_turns (int): 최대 턴 수. 기본값: 20.
        plan (Plan, optional): 사전 정의된 플랜.
        allow_for_replans (bool): 재계획 허용 여부. 기본값: True.
        use_fara_agent (bool): Fara 에이전트 사용 여부. 기본값: True.
        retrieve_relevant_plans (Literal): 관련 플랜 검색 방식. 기본값: "never".
        memory_controller_key (str, optional): 메모리 컨트롤러 키.
        model_context_token_limit (int): 모델 컨텍스트 토큰 제한. 기본값: 32000.
        allow_follow_up_input (bool): 후속 입력 허용. 기본값: True.
        final_answer_prompt (str, optional): 최종 답변 프롬프트.
        browser_headless (bool): 헤드리스 브라우저 모드. 기본값: False.
        sentinel_plan (SentinelPlanConfig): Sentinel plan 설정.
    """

    model_client_configs: ModelClientConfigs = Field(default_factory=ModelClientConfigs)
    cooperative_planning: bool = True
    autonomous_execution: bool = False
    allowed_websites: Optional[List[str]] = None
    max_actions_per_step: int = 5
    max_turns: int = 20
    plan: Optional[Plan] = None
    allow_for_replans: bool = True
    use_fara_agent: bool = True  # 기본값으로 Fara 사용
    retrieve_relevant_plans: Literal["never", "hint", "reuse"] = "never"
    memory_controller_key: Optional[str] = None
    model_context_token_limit: int = 32000  # 로컬 모델에 맞게 조정
    allow_follow_up_input: bool = True
    final_answer_prompt: Optional[str] = None
    browser_headless: bool = False
    sentinel_plan: SentinelPlanConfig = Field(default_factory=SentinelPlanConfig)

    # 로컬 모델 관련 설정
    orchestrator_base_url: str = "http://localhost:30000/v1"
    fara_base_url: str = "http://localhost:30001/v1"
