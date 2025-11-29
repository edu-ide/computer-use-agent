"""
Manager Agent

여러 전문 에이전트를 조정하는 관리자 에이전트:
- 작업 분배
- 에이전트 간 조정
- 결과 통합
- 에러 처리
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from smolagents import CodeAgent, Model

from .base_agent import BaseSpecializedAgent, AgentResult
from .search_agent import SearchAgent
from .analysis_agent import AnalysisAgent
from .validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """워크플로우 스텝 정의"""
    agent_type: str  # "search", "analysis", "validation"
    instruction: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[List[str]] = None  # 의존하는 스텝 이름들
    name: Optional[str] = None


@dataclass
class WorkflowResult:
    """워크플로우 실행 결과"""
    success: bool
    results: Dict[str, AgentResult] = field(default_factory=dict)
    final_data: Dict[str, Any] = field(default_factory=dict)
    total_duration_ms: int = 0
    error: Optional[str] = None


class ManagerAgent:
    """
    Multi-Agent Manager

    여러 전문화된 에이전트를 조정하여 복잡한 작업을 수행합니다.

    기능:
    - 순차/병렬 작업 실행
    - 에이전트 간 데이터 전달
    - 에러 복구 및 재시도
    - 결과 통합

    Example:
        ```python
        manager = ManagerAgent(
            manager_model=get_model("gpt-4o"),
            agent_model=get_model("local-qwen3-vl"),
        )

        result = await manager.run_workflow([
            WorkflowStep("search", "Search for iPhone", {"keyword": "iPhone"}),
            WorkflowStep("analysis", "Extract products", depends_on=["search"]),
            WorkflowStep("validation", "Validate data", depends_on=["analysis"]),
        ])
        ```
    """

    def __init__(
        self,
        manager_model: Model,
        agent_model: Optional[Model] = None,
        max_retries: int = 2,
    ):
        """
        Args:
            manager_model: Manager Agent용 모델 (더 강력한 모델 권장)
            agent_model: 전문 에이전트용 모델 (기본: manager_model과 동일)
            max_retries: 실패 시 최대 재시도 횟수
        """
        self.manager_model = manager_model
        self.agent_model = agent_model or manager_model
        self.max_retries = max_retries

        # 전문 에이전트들
        self._agents: Dict[str, BaseSpecializedAgent] = {}

        # 실행 결과 캐시
        self._step_results: Dict[str, AgentResult] = {}

        logger.info("[ManagerAgent] 초기화 완료")

    def _get_or_create_agent(self, agent_type: str) -> BaseSpecializedAgent:
        """에이전트 획득 또는 생성"""
        if agent_type not in self._agents:
            agent_class = self._get_agent_class(agent_type)
            self._agents[agent_type] = agent_class(model=self.agent_model)
            logger.info(f"[ManagerAgent] {agent_type} 에이전트 생성됨")

        return self._agents[agent_type]

    def _get_agent_class(self, agent_type: str) -> Type[BaseSpecializedAgent]:
        """에이전트 타입에 해당하는 클래스 반환"""
        agent_classes = {
            "search": SearchAgent,
            "analysis": AnalysisAgent,
            "validation": ValidationAgent,
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_classes[agent_type]

    async def run_workflow(
        self,
        steps: List[WorkflowStep],
        parallel: bool = False,
    ) -> WorkflowResult:
        """
        워크플로우 실행

        Args:
            steps: 실행할 스텝 목록
            parallel: 독립적인 스텝들을 병렬로 실행할지 여부

        Returns:
            WorkflowResult: 워크플로우 실행 결과
        """
        start_time = time.time()
        self._step_results = {}

        try:
            if parallel:
                await self._run_parallel(steps)
            else:
                await self._run_sequential(steps)

            # 결과 통합
            final_data = self._merge_results()

            total_duration = int((time.time() - start_time) * 1000)

            return WorkflowResult(
                success=True,
                results=self._step_results,
                final_data=final_data,
                total_duration_ms=total_duration,
            )

        except Exception as e:
            total_duration = int((time.time() - start_time) * 1000)
            logger.error(f"[ManagerAgent] 워크플로우 실패: {e}")

            return WorkflowResult(
                success=False,
                results=self._step_results,
                total_duration_ms=total_duration,
                error=str(e),
            )

    async def _run_sequential(self, steps: List[WorkflowStep]):
        """순차 실행"""
        for i, step in enumerate(steps):
            step_name = step.name or f"step_{i}"
            logger.info(f"[ManagerAgent] 스텝 실행: {step_name} ({step.agent_type})")

            # 의존성 체크
            if step.depends_on:
                for dep in step.depends_on:
                    if dep not in self._step_results:
                        raise RuntimeError(f"Dependency not met: {dep}")
                    if not self._step_results[dep].success:
                        raise RuntimeError(f"Dependency failed: {dep}")

            # 에이전트 실행
            result = await self._run_step_with_retry(step)
            self._step_results[step_name] = result

            if not result.success:
                logger.warning(f"[ManagerAgent] 스텝 실패: {step_name}")
                # 계속 진행할지 중단할지 결정
                # 여기서는 계속 진행 (실패한 스텝 결과도 저장)

    async def _run_parallel(self, steps: List[WorkflowStep]):
        """병렬 실행 (의존성 고려)"""
        # 의존성 그래프 구축
        pending = {(step.name or f"step_{i}"): step for i, step in enumerate(steps)}
        completed = set()

        while pending:
            # 현재 실행 가능한 스텝들 찾기
            runnable = []
            for name, step in pending.items():
                if step.depends_on is None:
                    runnable.append((name, step))
                elif all(dep in completed for dep in step.depends_on):
                    # 의존성이 모두 성공했는지 확인
                    if all(
                        self._step_results.get(dep, AgentResult(success=False)).success
                        for dep in step.depends_on
                    ):
                        runnable.append((name, step))

            if not runnable:
                if pending:
                    raise RuntimeError(
                        f"Cannot resolve dependencies for: {list(pending.keys())}"
                    )
                break

            # 병렬 실행
            logger.info(f"[ManagerAgent] 병렬 실행: {[n for n, _ in runnable]}")
            tasks = [
                self._run_step_with_retry(step)
                for _, step in runnable
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 저장
            for (name, _), result in zip(runnable, results):
                if isinstance(result, Exception):
                    self._step_results[name] = AgentResult(
                        success=False,
                        error=str(result),
                    )
                else:
                    self._step_results[name] = result

                completed.add(name)
                del pending[name]

    async def _run_step_with_retry(self, step: WorkflowStep) -> AgentResult:
        """재시도 로직이 포함된 스텝 실행"""
        agent = self._get_or_create_agent(step.agent_type)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # 이전 스텝 결과를 파라미터에 추가
                params = dict(step.params)
                if step.depends_on:
                    for dep in step.depends_on:
                        if dep in self._step_results:
                            params[f"prev_{dep}"] = self._step_results[dep].data

                result = await agent.run(step.instruction, **params)

                if result.success:
                    return result

                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[ManagerAgent] {step.agent_type} 실패 "
                    f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

            if attempt < self.max_retries:
                await asyncio.sleep(1)  # 재시도 전 대기

        return AgentResult(
            success=False,
            error=last_error,
            agent_name=step.agent_type,
        )

    def _merge_results(self) -> Dict[str, Any]:
        """모든 스텝 결과 통합"""
        merged = {}

        for step_name, result in self._step_results.items():
            if result.success and result.data:
                merged[step_name] = result.data

        return merged

    async def search_and_analyze(
        self,
        keyword: str,
        pages: int = 1,
        validate: bool = True,
    ) -> WorkflowResult:
        """
        검색 및 분석 워크플로우 (편의 메서드)

        Args:
            keyword: 검색 키워드
            pages: 수집할 페이지 수
            validate: 결과 검증 여부

        Returns:
            WorkflowResult: 워크플로우 결과
        """
        steps = [
            WorkflowStep(
                agent_type="search",
                instruction=f"Search for '{keyword}' and collect results from {pages} page(s)",
                params={"keyword": keyword, "pages": pages},
                name="search",
            ),
            WorkflowStep(
                agent_type="analysis",
                instruction="Extract all product information from the search results",
                depends_on=["search"],
                name="analysis",
            ),
        ]

        if validate:
            steps.append(
                WorkflowStep(
                    agent_type="validation",
                    instruction="Validate the extracted product data",
                    depends_on=["analysis"],
                    name="validation",
                )
            )

        return await self.run_workflow(steps, parallel=False)

    def get_agent(self, agent_type: str) -> Optional[BaseSpecializedAgent]:
        """특정 에이전트 인스턴스 반환"""
        return self._agents.get(agent_type)

    def clear_cache(self):
        """결과 캐시 초기화"""
        self._step_results = {}
