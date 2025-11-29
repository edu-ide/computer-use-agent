"""
Analysis Agent

데이터 분석 및 추출 전문 에이전트:
- 상품 정보 추출
- 가격 분석
- 데이터 구조화
- 패턴 인식
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from smolagents import Model

from .base_agent import BaseSpecializedAgent, AgentResult

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseSpecializedAgent):
    """
    데이터 분석 전문 에이전트

    웹 페이지에서 데이터를 추출하고 분석합니다.
    """

    AGENT_NAME = "AnalysisAgent"
    DEFAULT_MAX_STEPS = 15

    SYSTEM_PROMPT = """
You are a specialized Analysis Agent.

Your responsibilities:
1. Extract structured data from web pages
2. Analyze patterns in the data
3. Identify key information (prices, names, descriptions)
4. Structure data in a consistent format
5. Detect anomalies or inconsistencies

Data extraction guidelines:
- Extract ALL visible items, not just a sample
- Maintain consistent field names across items
- Convert prices to numeric format when possible
- Note any missing or unclear data
- Group related information together

Output format:
Always structure your findings as:
{
    "items": [...],
    "total_count": N,
    "analysis": {...}
}

Report errors using [ERROR:TYPE] format when needed.
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
        분석 작업 실행

        Args:
            instruction: 분석 명령
            **kwargs: 추가 파라미터
                - fields: 추출할 필드 목록
                - format: 출력 형식

        Returns:
            AgentResult: 분석 결과
        """
        fields = kwargs.get("fields", [])
        output_format = kwargs.get("format", "json")

        enhanced_instruction = instruction
        if fields:
            enhanced_instruction += f"\n\nFields to extract: {', '.join(fields)}"
        enhanced_instruction += f"\nOutput format: {output_format}"

        return await self._execute(enhanced_instruction)

    async def extract_products(
        self,
        fields: Optional[List[str]] = None,
    ) -> AgentResult:
        """
        상품 정보 추출

        Args:
            fields: 추출할 필드 목록 (기본: name, price, seller, shipping)

        Returns:
            AgentResult: 추출된 상품 정보
        """
        default_fields = ["name", "price", "seller", "shipping", "rating", "reviews"]
        target_fields = fields or default_fields

        instruction = f"""
Extract product information from the current page.

Required fields for each product:
{chr(10).join(f'- {field}' for field in target_fields)}

Steps:
1. Look at the current page screenshot
2. Identify all product listings
3. For each product, extract the required fields
4. Structure the data as a list of dictionaries
5. Note any products with missing information

Return the data as JSON:
{{
    "products": [
        {{"name": "...", "price": ..., ...}},
        ...
    ],
    "total_extracted": N,
    "page_info": {{"current_page": N, "has_next": true/false}}
}}
"""
        return await self._execute(instruction)

    async def analyze_prices(self) -> AgentResult:
        """
        가격 분석

        Returns:
            AgentResult: 가격 분석 결과
        """
        instruction = """
Analyze the prices of all products on the current page.

Analysis to perform:
1. Extract all prices
2. Calculate min, max, average prices
3. Identify price outliers
4. Check for discounts or deals
5. Compare original vs discounted prices if available

Return analysis as:
{
    "price_stats": {
        "min": ...,
        "max": ...,
        "avg": ...,
        "median": ...
    },
    "outliers": [...],
    "discounts": [...]
}
"""
        return await self._execute(instruction)

    async def extract_specific_element(
        self,
        element_description: str,
    ) -> AgentResult:
        """
        특정 요소 추출

        Args:
            element_description: 추출할 요소 설명

        Returns:
            AgentResult: 추출된 요소 정보
        """
        instruction = f"""
Extract the following specific element from the page:
{element_description}

Steps:
1. Locate the described element
2. Extract all relevant information
3. Note the element's position and context
4. Return structured data

If element is not found, report [ERROR:ELEMENT_NOT_FOUND]
"""
        return await self._execute(instruction)
