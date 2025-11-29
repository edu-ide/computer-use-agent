"""
Validation Agent

데이터 검증 전문 에이전트:
- 추출된 데이터 검증
- 일관성 검사
- 이상치 탐지
- 품질 보장
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from smolagents import Model

from .base_agent import BaseSpecializedAgent, AgentResult

logger = logging.getLogger(__name__)


class ValidationAgent(BaseSpecializedAgent):
    """
    데이터 검증 전문 에이전트

    추출된 데이터의 품질을 검증합니다.
    """

    AGENT_NAME = "ValidationAgent"
    DEFAULT_MAX_STEPS = 8

    SYSTEM_PROMPT = """
You are a specialized Validation Agent.

Your responsibilities:
1. Verify extracted data accuracy
2. Check data consistency
3. Detect anomalies and outliers
4. Validate data formats
5. Ensure data completeness

Validation criteria:
- Prices should be positive numbers
- Product names should not be empty
- URLs should be valid format
- Required fields should be present
- Values should be within reasonable ranges

Report format:
{
    "valid": true/false,
    "issues": [...],
    "warnings": [...],
    "stats": {
        "total_items": N,
        "valid_items": N,
        "invalid_items": N
    }
}

Use [ERROR:TYPE] to report critical validation failures.
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
        검증 작업 실행

        Args:
            instruction: 검증 명령
            **kwargs: 추가 파라미터
                - data: 검증할 데이터
                - rules: 검증 규칙

        Returns:
            AgentResult: 검증 결과
        """
        data = kwargs.get("data", {})
        rules = kwargs.get("rules", [])

        enhanced_instruction = instruction
        if data:
            enhanced_instruction += f"\n\nData to validate:\n{data}"
        if rules:
            enhanced_instruction += f"\n\nValidation rules:\n{chr(10).join(f'- {r}' for r in rules)}"

        return await self._execute(enhanced_instruction)

    async def validate_products(
        self,
        products: List[Dict[str, Any]],
        required_fields: Optional[List[str]] = None,
    ) -> AgentResult:
        """
        상품 데이터 검증

        Args:
            products: 검증할 상품 목록
            required_fields: 필수 필드 목록

        Returns:
            AgentResult: 검증 결과
        """
        default_required = ["name", "price"]
        fields = required_fields or default_required

        instruction = f"""
Validate the following product data:

Products count: {len(products)}
Sample (first 3):
{products[:3]}

Required fields: {', '.join(fields)}

Validation checks:
1. All required fields are present
2. Prices are positive numbers
3. Names are non-empty strings
4. No duplicate entries
5. Values are within reasonable ranges

Return validation report:
{{
    "valid": true/false,
    "total_products": {len(products)},
    "valid_products": N,
    "invalid_products": N,
    "issues": [
        {{"product_index": N, "field": "...", "issue": "..."}}
    ],
    "warnings": [...]
}}
"""
        return await self._execute(instruction)

    async def check_consistency(
        self,
        data_sets: List[Dict[str, Any]],
    ) -> AgentResult:
        """
        데이터 일관성 검사

        Args:
            data_sets: 비교할 데이터셋 목록

        Returns:
            AgentResult: 일관성 검사 결과
        """
        instruction = f"""
Check consistency across multiple data sets:

Number of data sets: {len(data_sets)}

Consistency checks:
1. Same fields across all data sets
2. Consistent data types
3. Consistent value ranges
4. No conflicting information
5. Matching totals and counts

Return consistency report:
{{
    "consistent": true/false,
    "differences": [...],
    "recommendations": [...]
}}
"""
        return await self._execute(instruction)

    async def detect_anomalies(
        self,
        data: List[Dict[str, Any]],
        check_fields: Optional[List[str]] = None,
    ) -> AgentResult:
        """
        이상치 탐지

        Args:
            data: 분석할 데이터
            check_fields: 검사할 필드 목록

        Returns:
            AgentResult: 이상치 탐지 결과
        """
        instruction = f"""
Detect anomalies in the following data:

Data count: {len(data)}
{"Fields to check: " + ", ".join(check_fields) if check_fields else "Check all numeric fields"}

Anomaly detection:
1. Statistical outliers (beyond 2 standard deviations)
2. Impossible values (negative prices, etc.)
3. Suspicious patterns
4. Missing value patterns
5. Format inconsistencies

Return anomaly report:
{{
    "anomalies_found": N,
    "anomalies": [
        {{"index": N, "field": "...", "value": ..., "reason": "..."}}
    ],
    "severity": "low/medium/high"
}}
"""
        return await self._execute(instruction)
