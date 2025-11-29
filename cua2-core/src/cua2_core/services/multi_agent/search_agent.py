"""
Search Agent

검색 및 탐색 전문 에이전트:
- 웹 검색
- 페이지 탐색
- 키워드 검색
- 필터링
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from smolagents import Model

from .base_agent import BaseSpecializedAgent, AgentResult

logger = logging.getLogger(__name__)


class SearchAgent(BaseSpecializedAgent):
    """
    검색 전문 에이전트

    웹 페이지에서 검색 및 탐색 작업을 수행합니다.
    """

    AGENT_NAME = "SearchAgent"
    DEFAULT_MAX_STEPS = 10

    SYSTEM_PROMPT = """
You are a specialized Search Agent.

Your responsibilities:
1. Navigate to search pages
2. Enter search keywords
3. Apply filters (price, category, etc.)
4. Navigate through search results
5. Find specific elements on the page

Guidelines:
- Always wait for page loads after navigation
- Use natural typing speed to avoid detection
- Report any errors you encounter using [ERROR:TYPE] format
- When search fails, try alternative approaches

Available error types:
- [ERROR:BOT_DETECTED] - When captcha or bot detection appears
- [ERROR:PAGE_FAILED] - When page fails to load
- [ERROR:ELEMENT_NOT_FOUND] - When target element is not found
"""

    def __init__(
        self,
        model: Model,
        max_steps: Optional[int] = None,
        planning_interval: Optional[int] = None,
        tools: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            max_steps=max_steps or self.DEFAULT_MAX_STEPS,
            planning_interval=planning_interval,
            tools=tools,
        )

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    async def run(self, instruction: str, **kwargs) -> AgentResult:
        """
        검색 작업 실행

        Args:
            instruction: 검색 명령
            **kwargs: 추가 파라미터
                - keyword: 검색 키워드
                - filters: 적용할 필터

        Returns:
            AgentResult: 검색 결과
        """
        keyword = kwargs.get("keyword", "")
        filters = kwargs.get("filters", {})

        # 명령 보강
        enhanced_instruction = instruction
        if keyword:
            enhanced_instruction += f"\n\nSearch keyword: {keyword}"
        if filters:
            enhanced_instruction += f"\nFilters to apply: {filters}"

        return await self._execute(enhanced_instruction)

    async def search(
        self,
        keyword: str,
        filters: Optional[Dict[str, Any]] = None,
        pages: int = 1,
    ) -> AgentResult:
        """
        검색 실행

        Args:
            keyword: 검색 키워드
            filters: 필터 조건
            pages: 탐색할 페이지 수

        Returns:
            AgentResult: 검색 결과
        """
        instruction = f"""
Search for '{keyword}' and collect results from {pages} page(s).

Steps:
1. Find the search input field
2. Enter the keyword: {keyword}
3. Submit the search
4. Wait for results to load
{"5. Apply filters: " + str(filters) if filters else ""}
6. Collect visible results
7. Navigate to next page if needed (up to {pages} pages)
"""
        return await self._execute(instruction)

    async def navigate_to_page(self, page_number: int) -> AgentResult:
        """
        특정 페이지로 이동

        Args:
            page_number: 이동할 페이지 번호

        Returns:
            AgentResult: 결과
        """
        instruction = f"""
Navigate to page {page_number} of the search results.

Steps:
1. Find pagination controls
2. Click on page {page_number}
3. Wait for page to load
4. Confirm you are on the correct page
"""
        return await self._execute(instruction)
