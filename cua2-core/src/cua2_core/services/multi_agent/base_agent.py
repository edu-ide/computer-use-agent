"""
Base Specialized Agent

모든 전문화된 에이전트의 베이스 클래스
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smolagents import CodeAgent, Model

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """에이전트 실행 결과"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    steps_count: int = 0
    duration_ms: int = 0
    agent_name: str = ""


class BaseSpecializedAgent(ABC):
    """
    전문화된 에이전트 베이스 클래스

    각 에이전트는 특정 작업에 특화됩니다:
    - SearchAgent: 검색 및 탐색
    - AnalysisAgent: 데이터 분석 및 추출
    - ValidationAgent: 데이터 검증
    """

    # 에이전트 이름 (서브클래스에서 정의)
    AGENT_NAME: str = "base"

    # 기본 설정
    DEFAULT_MAX_STEPS: int = 10
    DEFAULT_PLANNING_INTERVAL: int = 1

    def __init__(
        self,
        model: Model,
        max_steps: Optional[int] = None,
        planning_interval: Optional[int] = None,
        tools: Optional[List] = None,
    ):
        """
        Args:
            model: smolagents Model 인스턴스
            max_steps: 최대 스텝 수
            planning_interval: 계획 재수립 간격 (1 = 매 스텝마다)
            tools: 사용할 도구 목록
        """
        self.model = model
        self.max_steps = max_steps or self.DEFAULT_MAX_STEPS
        self.planning_interval = planning_interval or self.DEFAULT_PLANNING_INTERVAL

        # 에이전트 생성
        self._agent = CodeAgent(
            model=model,
            max_steps=self.max_steps,
            planning_interval=self.planning_interval,
            tools=tools or [],
            stream_outputs=True,
        )

        logger.info(
            f"[{self.AGENT_NAME}] 에이전트 초기화 완료 "
            f"(max_steps={self.max_steps}, planning_interval={self.planning_interval})"
        )

    @property
    def agent(self) -> CodeAgent:
        """내부 smolagents 에이전트 반환 (Manager에서 사용)"""
        return self._agent

    @abstractmethod
    def get_system_prompt(self) -> str:
        """에이전트별 시스템 프롬프트 반환"""
        pass

    @abstractmethod
    async def run(self, instruction: str, **kwargs) -> AgentResult:
        """
        에이전트 실행

        Args:
            instruction: 실행할 명령
            **kwargs: 추가 파라미터

        Returns:
            AgentResult: 실행 결과
        """
        pass

    async def _execute(self, instruction: str) -> AgentResult:
        """
        공통 실행 로직

        Args:
            instruction: 실행할 명령

        Returns:
            AgentResult: 실행 결과
        """
        import time
        start_time = time.time()

        try:
            # 시스템 프롬프트 + 사용자 명령
            full_instruction = f"{self.get_system_prompt()}\n\n{instruction}"

            # 에이전트 실행
            result = self._agent.run(full_instruction)

            duration_ms = int((time.time() - start_time) * 1000)

            return AgentResult(
                success=True,
                data={"result": result},
                steps_count=len(self._agent.logs) if hasattr(self._agent, 'logs') else 0,
                duration_ms=duration_ms,
                agent_name=self.AGENT_NAME,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{self.AGENT_NAME}] 실행 실패: {e}")

            return AgentResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                agent_name=self.AGENT_NAME,
            )

    def add_tool(self, tool):
        """도구 추가"""
        self._agent.tools.append(tool)

    def get_logs(self) -> List[Any]:
        """실행 로그 반환"""
        return getattr(self._agent, 'logs', [])
