"""
High-Level Tools

smolagents 고수준 도구 모음:
- 저수준 도구(click, type, scroll)를 조합한 복합 도구
- 웹 자동화에 특화된 도구
- 에러 처리가 내장된 안전한 도구

저수준 도구를 직접 사용하는 대신 이 도구들을 사용하면
더 안정적이고 자연스러운 자동화가 가능합니다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from smolagents import tool

logger = logging.getLogger(__name__)


def create_high_level_tools(desktop: Any):
    """
    고수준 도구 생성

    Args:
        desktop: LocalDesktop 인스턴스

    Returns:
        도구 함수들의 리스트

    Example:
        ```python
        tools = create_high_level_tools(desktop)
        agent = LocalVisionAgent(
            model=model,
            desktop=desktop,
            tools=tools,
        )
        ```
    """

    # =========================================
    # 스마트 클릭 도구
    # =========================================

    @tool
    def smart_click(element_description: str) -> str:
        """
        텍스트나 설명으로 요소를 찾아 클릭합니다.

        저수준 click(x, y) 대신 이 도구를 사용하면
        요소를 자동으로 찾아서 클릭합니다.

        Args:
            element_description: 클릭할 요소 설명 (예: "검색 버튼", "로그인 링크")

        Returns:
            클릭 결과 메시지
        """
        # 실제 구현에서는 OCR이나 요소 탐지 사용
        # 여기서는 VLM이 좌표를 제공하도록 안내
        return f"""
요소를 찾으려면:
1. 현재 스크린샷에서 '{element_description}'를 찾으세요
2. 해당 요소의 중앙 좌표(x, y)를 확인하세요
3. click(x, y) 도구를 사용하세요

'{element_description}'를 클릭하려고 합니다.
"""

    @tool
    def safe_click(x: int, y: int, wait_after_ms: int = 500) -> str:
        """
        안전한 클릭 - 클릭 후 대기 시간 포함

        봇 감지를 피하기 위해 클릭 후 잠시 대기합니다.

        Args:
            x: x 좌표
            y: y 좌표
            wait_after_ms: 클릭 후 대기 시간 (밀리초)

        Returns:
            클릭 결과
        """
        try:
            desktop.move_mouse(x, y)
            time.sleep(0.1)  # 마우스 이동 후 잠시 대기
            desktop.left_click()
            time.sleep(wait_after_ms / 1000)
            return f"클릭 완료: ({x}, {y}), {wait_after_ms}ms 대기"
        except Exception as e:
            return f"[ERROR:CLICK_FAILED] 클릭 실패: {e}"

    # =========================================
    # 스마트 입력 도구
    # =========================================

    @tool
    def smart_type(text: str, typing_speed: str = "natural") -> str:
        """
        자연스러운 속도로 텍스트 입력

        봇 감지를 피하기 위해 자연스러운 타이핑 속도를 사용합니다.

        Args:
            text: 입력할 텍스트
            typing_speed: 타이핑 속도 ("fast", "natural", "slow")

        Returns:
            입력 결과
        """
        # 속도별 딜레이 설정
        delays = {
            "fast": 0.02,
            "natural": 0.05,
            "slow": 0.1,
        }
        delay = delays.get(typing_speed, 0.05)

        try:
            for char in text:
                desktop.type_text(char)
                time.sleep(delay)
            return f"텍스트 입력 완료: '{text}' (speed: {typing_speed})"
        except Exception as e:
            return f"[ERROR:TYPE_FAILED] 텍스트 입력 실패: {e}"

    @tool
    def fill_input_field(
        field_x: int,
        field_y: int,
        text: str,
        clear_first: bool = True,
    ) -> str:
        """
        입력 필드에 텍스트 입력 (필드 클릭 + 입력)

        저수준 click + type을 조합한 고수준 도구입니다.

        Args:
            field_x: 입력 필드 x 좌표
            field_y: 입력 필드 y 좌표
            text: 입력할 텍스트
            clear_first: 기존 텍스트를 먼저 지울지 여부

        Returns:
            입력 결과
        """
        try:
            # 필드 클릭
            desktop.move_mouse(field_x, field_y)
            time.sleep(0.1)
            desktop.left_click()
            time.sleep(0.2)

            # 기존 텍스트 삭제
            if clear_first:
                desktop.key("ctrl+a")
                time.sleep(0.1)
                desktop.key("delete")
                time.sleep(0.1)

            # 텍스트 입력 (자연스러운 속도)
            for char in text:
                desktop.type_text(char)
                time.sleep(0.05)

            return f"입력 필드 채우기 완료: '{text}'"
        except Exception as e:
            return f"[ERROR:FILL_FAILED] 입력 필드 채우기 실패: {e}"

    # =========================================
    # 스크롤 도구
    # =========================================

    @tool
    def scroll_to_find(
        direction: str,
        target_description: str,
        max_scrolls: int = 5,
    ) -> str:
        """
        특정 요소가 보일 때까지 스크롤

        Args:
            direction: 스크롤 방향 ("up" 또는 "down")
            target_description: 찾을 요소 설명
            max_scrolls: 최대 스크롤 횟수

        Returns:
            스크롤 결과
        """
        scroll_amount = 300 if direction == "down" else -300

        for i in range(max_scrolls):
            try:
                desktop.scroll(scroll_amount)
                time.sleep(0.5)
            except Exception as e:
                return f"[ERROR:SCROLL_FAILED] 스크롤 실패: {e}"

        return f"""
{max_scrolls}번 {direction} 방향으로 스크롤했습니다.
현재 화면에서 '{target_description}'를 찾아보세요.
보이지 않으면 scroll_to_find를 다시 사용하세요.
"""

    @tool
    def smooth_scroll(direction: str, amount: str = "medium") -> str:
        """
        부드러운 스크롤

        Args:
            direction: 스크롤 방향 ("up" 또는 "down")
            amount: 스크롤 양 ("small", "medium", "large", "page")

        Returns:
            스크롤 결과
        """
        amounts = {
            "small": 100,
            "medium": 300,
            "large": 500,
            "page": 800,
        }
        scroll_pixels = amounts.get(amount, 300)
        if direction == "up":
            scroll_pixels = -scroll_pixels

        try:
            # 부드러운 스크롤을 위해 여러 번 나눠서
            steps = 3
            per_step = scroll_pixels // steps
            for _ in range(steps):
                desktop.scroll(per_step)
                time.sleep(0.1)

            return f"스크롤 완료: {direction}, {amount}"
        except Exception as e:
            return f"[ERROR:SCROLL_FAILED] 스크롤 실패: {e}"

    # =========================================
    # 페이지 네비게이션 도구
    # =========================================

    @tool
    def wait_for_page_load(max_wait_seconds: int = 10) -> str:
        """
        페이지 로딩 대기

        페이지가 완전히 로드될 때까지 대기합니다.

        Args:
            max_wait_seconds: 최대 대기 시간 (초)

        Returns:
            대기 결과
        """
        time.sleep(max_wait_seconds)
        return f"페이지 로딩 대기 완료 ({max_wait_seconds}초)"

    @tool
    def go_back() -> str:
        """
        이전 페이지로 돌아가기

        Returns:
            결과 메시지
        """
        try:
            desktop.key("alt+Left")
            time.sleep(1)
            return "이전 페이지로 이동"
        except Exception as e:
            return f"[ERROR:NAV_FAILED] 뒤로 가기 실패: {e}"

    @tool
    def refresh_page() -> str:
        """
        페이지 새로고침

        Returns:
            결과 메시지
        """
        try:
            desktop.key("F5")
            time.sleep(2)
            return "페이지 새로고침 완료"
        except Exception as e:
            return f"[ERROR:NAV_FAILED] 새로고침 실패: {e}"

    # =========================================
    # 검색 도구
    # =========================================

    @tool
    def search_on_page(keyword: str) -> str:
        """
        페이지 내 검색 (Ctrl+F)

        Args:
            keyword: 검색할 키워드

        Returns:
            검색 결과
        """
        try:
            desktop.key("ctrl+f")
            time.sleep(0.3)

            for char in keyword:
                desktop.type_text(char)
                time.sleep(0.05)

            time.sleep(0.5)
            return f"페이지 내 '{keyword}' 검색 중..."
        except Exception as e:
            return f"[ERROR:SEARCH_FAILED] 페이지 내 검색 실패: {e}"

    @tool
    def submit_search() -> str:
        """
        검색 제출 (Enter 키)

        Returns:
            결과 메시지
        """
        try:
            desktop.key("Return")
            time.sleep(1)
            return "검색 제출 완료"
        except Exception as e:
            return f"[ERROR:SUBMIT_FAILED] 검색 제출 실패: {e}"

    # =========================================
    # 데이터 추출 도구
    # =========================================

    @tool
    def copy_selected_text() -> str:
        """
        선택된 텍스트 복사

        Returns:
            복사된 텍스트 (또는 에러 메시지)
        """
        try:
            desktop.key("ctrl+c")
            time.sleep(0.2)
            # 클립보드에서 텍스트 가져오기
            import subprocess
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
            )
            return f"복사된 텍스트: {result.stdout}"
        except Exception as e:
            return f"[ERROR:COPY_FAILED] 텍스트 복사 실패: {e}"

    @tool
    def select_all_and_copy() -> str:
        """
        전체 선택 후 복사

        Returns:
            복사된 텍스트 (또는 에러 메시지)
        """
        try:
            desktop.key("ctrl+a")
            time.sleep(0.2)
            desktop.key("ctrl+c")
            time.sleep(0.2)
            return "전체 선택 및 복사 완료"
        except Exception as e:
            return f"[ERROR:COPY_FAILED] 전체 선택/복사 실패: {e}"

    # =========================================
    # 대기 및 확인 도구
    # =========================================

    @tool
    def wait_seconds(seconds: float) -> str:
        """
        지정된 시간 대기

        Args:
            seconds: 대기 시간 (초)

        Returns:
            대기 완료 메시지
        """
        time.sleep(seconds)
        return f"{seconds}초 대기 완료"

    @tool
    def take_screenshot_and_analyze() -> str:
        """
        스크린샷 촬영 및 분석 요청

        현재 화면을 캡처하고 분석을 요청합니다.

        Returns:
            분석 요청 메시지
        """
        return """
스크린샷을 촬영했습니다.
현재 화면을 분석하세요:
1. 어떤 페이지/화면인가요?
2. 주요 요소들은 무엇인가요?
3. 다음 액션은 무엇이어야 하나요?
"""

    # 모든 도구 반환
    return [
        smart_click,
        safe_click,
        smart_type,
        fill_input_field,
        scroll_to_find,
        smooth_scroll,
        wait_for_page_load,
        go_back,
        refresh_page,
        search_on_page,
        submit_search,
        copy_selected_text,
        select_all_and_copy,
        wait_seconds,
        take_screenshot_and_analyze,
    ]


# =========================================
# 도구 카테고리별 getter
# =========================================

def get_navigation_tools(desktop: Any) -> List:
    """네비게이션 관련 도구만 반환"""
    all_tools = create_high_level_tools(desktop)
    # 인덱스로 선택 (wait_for_page_load, go_back, refresh_page)
    return [all_tools[6], all_tools[7], all_tools[8]]


def get_input_tools(desktop: Any) -> List:
    """입력 관련 도구만 반환"""
    all_tools = create_high_level_tools(desktop)
    # smart_type, fill_input_field
    return [all_tools[2], all_tools[3]]


def get_scroll_tools(desktop: Any) -> List:
    """스크롤 관련 도구만 반환"""
    all_tools = create_high_level_tools(desktop)
    # scroll_to_find, smooth_scroll
    return [all_tools[4], all_tools[5]]
