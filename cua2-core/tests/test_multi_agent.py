"""
Multi-Agent System 테스트

테스트 대상:
1. AgentResult - 에이전트 결과 데이터 구조
2. WorkflowStep & WorkflowResult - 워크플로우 데이터 구조
3. ManagerAgent - 에이전트 조정자
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any, Dict, List

import pytest

from cua2_core.services.multi_agent import (
    BaseSpecializedAgent,
    AgentResult,
    SearchAgent,
    AnalysisAgent,
    ValidationAgent,
    ManagerAgent,
    WorkflowStep,
    WorkflowResult,
)


class TestAgentResult:
    """AgentResult 테스트"""

    def test_agent_result_success(self):
        """성공 결과"""
        result = AgentResult(
            success=True,
            data={"products": [{"name": "iPhone", "price": 1000000}]},
            agent_name="search",
            duration_ms=1500,
        )

        assert result.success
        assert result.data["products"][0]["name"] == "iPhone"
        assert result.agent_name == "search"
        assert result.duration_ms == 1500
        assert result.error is None

    def test_agent_result_failure(self):
        """실패 결과"""
        result = AgentResult(
            success=False,
            data={},
            agent_name="analysis",
            error="Failed to extract data",
            duration_ms=500,
        )

        assert not result.success
        assert result.error == "Failed to extract data"
        assert result.agent_name == "analysis"

    def test_agent_result_defaults(self):
        """기본값 테스트"""
        result = AgentResult(success=True)

        assert result.success
        assert result.data == {}
        assert result.error is None
        assert result.steps_count == 0
        assert result.duration_ms == 0
        assert result.agent_name == ""


class TestWorkflowStep:
    """WorkflowStep 테스트"""

    def test_workflow_step_basic(self):
        """기본 스텝 생성"""
        step = WorkflowStep(
            agent_type="search",
            instruction="Search for iPhone",
            params={"keyword": "iPhone"},
            name="search_step",
        )

        assert step.agent_type == "search"
        assert step.instruction == "Search for iPhone"
        assert step.params["keyword"] == "iPhone"
        assert step.name == "search_step"
        assert step.depends_on is None

    def test_workflow_step_with_dependencies(self):
        """의존성 있는 스텝"""
        step = WorkflowStep(
            agent_type="analysis",
            instruction="Analyze search results",
            params={},
            depends_on=["search"],
            name="analysis_step",
        )

        assert step.depends_on == ["search"]
        assert step.name == "analysis_step"

    def test_workflow_step_defaults(self):
        """기본값 테스트"""
        step = WorkflowStep(
            agent_type="validation",
            instruction="Validate data",
        )

        assert step.agent_type == "validation"
        assert step.params == {}
        assert step.depends_on is None
        assert step.name is None


class TestWorkflowResult:
    """WorkflowResult 테스트"""

    def test_workflow_result_success(self):
        """성공적인 워크플로우 결과"""
        results = {
            "search": AgentResult(success=True, data={"products": []}, agent_name="search"),
            "analysis": AgentResult(success=True, data={"analyzed": True}, agent_name="analysis"),
        }

        result = WorkflowResult(
            success=True,
            results=results,
            total_duration_ms=3000,
        )

        assert result.success
        assert "search" in result.results
        assert "analysis" in result.results
        assert result.total_duration_ms == 3000
        assert result.error is None

    def test_workflow_result_partial_failure(self):
        """부분 실패 결과"""
        results = {
            "search": AgentResult(success=True, data={"products": []}, agent_name="search"),
            "analysis": AgentResult(success=False, data={}, agent_name="analysis", error="Failed"),
        }

        result = WorkflowResult(
            success=False,
            results=results,
            total_duration_ms=2000,
            error="Analysis step failed",
        )

        assert not result.success
        assert result.error == "Analysis step failed"

    def test_workflow_result_defaults(self):
        """기본값 테스트"""
        result = WorkflowResult(success=True)

        assert result.success
        assert result.results == {}
        assert result.final_data == {}
        assert result.total_duration_ms == 0
        assert result.error is None


class TestManagerAgent:
    """ManagerAgent 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock smolagents Model"""
        model = MagicMock()
        model.generate = MagicMock(return_value="Test response")
        return model

    @pytest.fixture
    def manager(self, mock_model):
        """ManagerAgent 인스턴스 생성"""
        with patch('cua2_core.services.multi_agent.base_agent.CodeAgent'):
            return ManagerAgent(
                manager_model=mock_model,
                agent_model=mock_model,
            )

    def test_manager_initialization(self, manager, mock_model):
        """매니저 초기화"""
        assert manager.manager_model == mock_model
        assert manager.agent_model == mock_model
        assert manager.max_retries == 2

    def test_get_agent_none_for_unknown(self, manager):
        """알 수 없는 타입 조회"""
        agent = manager.get_agent("unknown")
        assert agent is None

    def test_get_agent_class(self, manager):
        """에이전트 클래스 조회"""
        assert manager._get_agent_class("search") == SearchAgent
        assert manager._get_agent_class("analysis") == AnalysisAgent
        assert manager._get_agent_class("validation") == ValidationAgent

    def test_get_agent_class_unknown(self, manager):
        """알 수 없는 에이전트 타입"""
        with pytest.raises(ValueError, match="Unknown agent type"):
            manager._get_agent_class("unknown")

    @pytest.mark.asyncio
    async def test_run_workflow_sequential(self, manager):
        """순차 워크플로우 실행"""
        steps = [
            WorkflowStep(
                agent_type="search",
                instruction="Search",
                params={},
                name="search",
            ),
        ]

        # Mock _run_step_with_retry
        manager._run_step_with_retry = AsyncMock(return_value=AgentResult(
            success=True,
            data={"products": []},
            agent_name="search",
            duration_ms=1000,
        ))

        result = await manager.run_workflow(steps, parallel=False)

        assert result.success
        assert "search" in result.results

    @pytest.mark.asyncio
    async def test_run_workflow_with_failure(self, manager):
        """실패가 포함된 워크플로우"""
        steps = [
            WorkflowStep(
                agent_type="search",
                instruction="Search",
                params={},
                name="search",
            ),
        ]

        # Mock 실패 반환
        manager._run_step_with_retry = AsyncMock(return_value=AgentResult(
            success=False,
            data={},
            agent_name="search",
            error="Search failed",
            duration_ms=500,
        ))

        result = await manager.run_workflow(steps, parallel=False)

        # 스텝은 실패했지만 워크플로우 자체는 완료됨
        assert result.success  # 워크플로우는 계속 진행됨
        assert not result.results["search"].success

    def test_merge_results(self, manager):
        """결과 병합"""
        manager._step_results = {
            "search": AgentResult(success=True, data={"items": [1, 2, 3]}, agent_name="search"),
            "analysis": AgentResult(success=True, data={"count": 3}, agent_name="analysis"),
            "failed": AgentResult(success=False, data={}, agent_name="failed"),
        }

        merged = manager._merge_results()

        assert "search" in merged
        assert "analysis" in merged
        assert "failed" not in merged  # 실패한 건 제외
        assert merged["search"]["items"] == [1, 2, 3]

    def test_clear_cache(self, manager):
        """캐시 초기화"""
        manager._step_results = {"test": AgentResult(success=True)}
        manager.clear_cache()

        assert manager._step_results == {}


class TestParallelExecution:
    """병렬 실행 테스트"""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        return model

    @pytest.fixture
    def manager(self, mock_model):
        with patch('cua2_core.services.multi_agent.base_agent.CodeAgent'):
            return ManagerAgent(
                manager_model=mock_model,
                agent_model=mock_model,
            )

    @pytest.mark.asyncio
    async def test_parallel_execution_independent_steps(self, manager):
        """독립적인 스텝 병렬 실행"""
        steps = [
            WorkflowStep(agent_type="search", instruction="Search A", name="search1"),
            WorkflowStep(agent_type="search", instruction="Search B", name="search2"),
        ]

        # Mock
        call_count = 0
        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return AgentResult(success=True, data={"id": call_count}, agent_name="search")

        manager._run_step_with_retry = mock_run

        result = await manager.run_workflow(steps, parallel=True)

        assert result.success
        assert len(result.results) == 2
        assert "search1" in result.results
        assert "search2" in result.results

    @pytest.mark.asyncio
    async def test_parallel_execution_with_dependencies(self, manager):
        """의존성이 있는 스텝 병렬 실행"""
        steps = [
            WorkflowStep(agent_type="search", instruction="Search", name="search"),
            WorkflowStep(agent_type="analysis", instruction="Analyze", name="analysis", depends_on=["search"]),
        ]

        call_order = []
        async def mock_run(step):
            call_order.append(step.name)
            return AgentResult(success=True, data={}, agent_name=step.agent_type)

        manager._run_step_with_retry = mock_run

        result = await manager.run_workflow(steps, parallel=True)

        assert result.success
        # search가 먼저 실행되어야 함
        assert call_order.index("search") < call_order.index("analysis")


class TestSearchAndAnalyze:
    """search_and_analyze 편의 메서드 테스트"""

    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_model):
        with patch('cua2_core.services.multi_agent.base_agent.CodeAgent'):
            return ManagerAgent(
                manager_model=mock_model,
                agent_model=mock_model,
            )

    @pytest.mark.asyncio
    async def test_search_and_analyze_with_validation(self, manager):
        """검증 포함 search_and_analyze"""
        executed_steps = []

        async def mock_run(step):
            executed_steps.append(step.agent_type)
            return AgentResult(success=True, data={"step": step.agent_type}, agent_name=step.agent_type)

        manager._run_step_with_retry = mock_run

        result = await manager.search_and_analyze(keyword="test", pages=1, validate=True)

        assert result.success
        assert "search" in executed_steps
        assert "analysis" in executed_steps
        assert "validation" in executed_steps

    @pytest.mark.asyncio
    async def test_search_and_analyze_without_validation(self, manager):
        """검증 제외 search_and_analyze"""
        executed_steps = []

        async def mock_run(step):
            executed_steps.append(step.agent_type)
            return AgentResult(success=True, data={"step": step.agent_type}, agent_name=step.agent_type)

        manager._run_step_with_retry = mock_run

        result = await manager.search_and_analyze(keyword="test", pages=1, validate=False)

        assert result.success
        assert "search" in executed_steps
        assert "analysis" in executed_steps
        assert "validation" not in executed_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
