"""
Fara-7B 전용 컴포넌트 (Microsoft Magentic-UI 포팅)

이 모듈은 Microsoft의 Magentic-UI에서 Fara-7B 모델용으로 설계된
핵심 컴포넌트들을 포팅한 것입니다.

주요 컴포넌트:
- PlaywrightController: Fara 전용 브라우저 제어 (에러 복구 기능 포함)
- FaraComputerUse: Fara 전용 tool 정의
- FaraWebSurfer: Fara 전용 웹 서퍼 에이전트

참조:
- https://github.com/microsoft/magentic-ui
- https://github.com/microsoft/fara
"""

from .playwright_controller import PlaywrightController
from .prompts import FaraComputerUse, get_computer_use_system_prompt
from .web_surfer import FaraWebSurfer, WebSurferConfig

__all__ = [
    "PlaywrightController",
    "FaraComputerUse",
    "get_computer_use_system_prompt",
    "FaraWebSurfer",
    "WebSurferConfig",
]
