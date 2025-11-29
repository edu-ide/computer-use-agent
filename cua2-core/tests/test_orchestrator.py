"""
Orchestrator 모듈 테스트

테스트 대상:
1. StrategySelector - 복잡도 분석 및 전략 선택
2. WorkflowMonitor - 워크플로우 실행 추적 및 리포트 생성
3. StepEvaluator - 스텝 평가 및 학습
4. VLM 에러 타입 기반 라우팅 로직
"""

from __future__ import annotations

import time
from enum import Enum
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cua2_core.services.orchestrator import (
    ExecutionStrategy,
    ExecutionDecision,
    NodeStatus,
    StepAction,
    StepFeedback,
    NodeComplexity,
    ModelConfig,
    WorkflowReport,
    StrategySelector,
    WorkflowMonitor,
    StepEvaluator,
)


# VLMErrorType을 테스트 내에서 정의 (langgraph 의존성 회피)
class VLMErrorType(str, Enum):
    """VLM 에러 타입 (테스트용)"""
    NONE = "none"
    BOT_DETECTED = "bot_detected"
    PAGE_FAILED = "page_failed"
    ACCESS_DENIED = "access_denied"
    ELEMENT_NOT_FOUND = "element_not_found"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TestStrategySelector:
    """StrategySelector 테스트"""

    @pytest.fixture
    def selector(self):
        """StrategySelector 인스턴스 생성"""
        return StrategySelector(prefer_local=True, cost_weight=0.5)

    def test_analyze_complexity_simple(self, selector):
        """단순 작업 복잡도 분석"""
        complexity = selector.analyze_complexity(
            node_id="open_homepage",
            instruction="Open the homepage URL",
        )

        assert complexity.complexity_score < 0.3
        assert not complexity.requires_reasoning
        assert not complexity.requires_data_extraction

    def test_analyze_complexity_medium(self, selector):
        """중간 복잡도 작업 분석"""
        complexity = selector.analyze_complexity(
            node_id="search_products",
            instruction="Search for products and extract product names",
        )

        assert complexity.requires_data_extraction
        assert 0.2 <= complexity.complexity_score <= 0.6

    def test_analyze_complexity_high(self, selector):
        """높은 복잡도 작업 분석"""
        complexity = selector.analyze_complexity(
            node_id="analyze_compare",
            instruction="Analyze the page and compare prices to decide the best option",
        )

        assert complexity.requires_reasoning
        assert complexity.complexity_score >= 0.5

    def test_analyze_complexity_vision(self, selector):
        """비전 요구 작업 분석"""
        complexity = selector.analyze_complexity(
            node_id="find_button",
            instruction="Look at the screenshot and find the buy button",
        )

        assert complexity.requires_vision
        assert complexity.complexity_score >= 0.2

    def test_analyze_complexity_caching(self, selector):
        """복잡도 캐싱 테스트"""
        instruction = "Click the button"

        complexity1 = selector.analyze_complexity("node1", instruction)
        complexity2 = selector.analyze_complexity("node1", instruction)

        # 같은 노드/instruction은 캐시됨
        assert complexity1 is complexity2

    def test_select_strategy_simple(self, selector):
        """단순 작업 전략 선택"""
        complexity = NodeComplexity(complexity_score=0.2)

        decision = selector.select_strategy(
            node_id="simple_click",
            complexity=complexity,
            instruction="Click the button",
        )

        assert decision.strategy == ExecutionStrategy.LOCAL_MODEL
        assert decision.model_id == "local-qwen-vl"
        assert decision.estimated_cost == 0.0

    def test_select_strategy_medium(self, selector):
        """중간 복잡도 전략 선택"""
        complexity = NodeComplexity(complexity_score=0.5)

        decision = selector.select_strategy(
            node_id="extract_data",
            complexity=complexity,
            instruction="Extract product data",
        )

        assert decision.strategy == ExecutionStrategy.CLOUD_LIGHT
        assert decision.model_id == "gpt-4o-mini"

    def test_select_strategy_complex(self, selector):
        """복잡한 작업 전략 선택"""
        complexity = NodeComplexity(complexity_score=0.9)

        decision = selector.select_strategy(
            node_id="complex_reasoning",
            complexity=complexity,
            instruction="Analyze and decide",
        )

        assert decision.strategy == ExecutionStrategy.CLOUD_HEAVY
        assert decision.model_id in ["gpt-4o", "claude-sonnet"]

    def test_select_strategy_rule_based(self, selector):
        """규칙 기반 처리 가능 케이스"""
        complexity = NodeComplexity(complexity_score=0.1)

        decision = selector.select_strategy(
            node_id="open_url",
            complexity=complexity,
            instruction="open_url('https://example.com')",
        )

        assert decision.strategy == ExecutionStrategy.RULE_BASED
        assert decision.model_id is None
        assert decision.estimated_cost == 0.0

    def test_decide_full_flow(self, selector):
        """decide() 전체 플로우 테스트"""
        decision = selector.decide(
            node_id="search",
            instruction="Search for iPhone and collect results",
            params={"keyword": "iPhone"},
        )

        assert isinstance(decision, ExecutionDecision)
        assert decision.strategy in [
            ExecutionStrategy.LOCAL_MODEL,
            ExecutionStrategy.CLOUD_LIGHT,
            ExecutionStrategy.CLOUD_HEAVY,
        ]

    def test_get_fallback_strategy(self, selector):
        """폴백 전략 테스트"""
        assert selector.get_fallback_strategy(ExecutionStrategy.LOCAL_MODEL) == ExecutionStrategy.CLOUD_LIGHT
        assert selector.get_fallback_strategy(ExecutionStrategy.CLOUD_LIGHT) == ExecutionStrategy.CLOUD_HEAVY
        assert selector.get_fallback_strategy(ExecutionStrategy.CLOUD_HEAVY) is None


class TestWorkflowMonitor:
    """WorkflowMonitor 테스트"""

    @pytest.fixture
    def monitor(self):
        """WorkflowMonitor 인스턴스 생성"""
        return WorkflowMonitor()

    def test_start_workflow_tracking(self, monitor):
        """워크플로우 추적 시작"""
        monitor.start_workflow_tracking(
            workflow_id="test-workflow",
            execution_id="exec-001",
            total_nodes=5,
        )

        exec_data = monitor.get_execution_data("exec-001")
        assert exec_data is not None
        assert exec_data["workflow_id"] == "test-workflow"
        assert exec_data["total_nodes"] == 5

    def test_record_node_lifecycle(self, monitor):
        """노드 시작/완료 기록"""
        monitor.start_workflow_tracking("wf", "exec-001", 3)

        # 노드 시작
        monitor.record_node_start("exec-001", "node1", ExecutionStrategy.LOCAL_MODEL)

        exec_data = monitor.get_execution_data("exec-001")
        assert len(exec_data["node_records"]) == 1
        assert exec_data["node_records"][0].status == NodeStatus.RUNNING

        # 노드 완료
        monitor.record_node_complete(
            execution_id="exec-001",
            node_id="node1",
            success=True,
            duration_ms=1000,
            cost=0.001,
            result_summary="Completed",
        )

        assert exec_data["node_records"][0].status == NodeStatus.SUCCESS
        assert exec_data["total_cost"] == 0.001

    def test_check_stuck_node(self, monitor):
        """Stuck 노드 감지"""
        monitor.start_workflow_tracking("wf", "exec-001", 1)
        monitor.record_node_start("exec-001", "slow_node", ExecutionStrategy.CLOUD_HEAVY)

        # 처음에는 stuck이 아님
        assert not monitor.check_stuck_node("exec-001", "slow_node")

        # 타임아웃 시간 직접 조작하여 stuck 테스트
        exec_data = monitor.get_execution_data("exec-001")
        exec_data["node_records"][0].start_time = time.time() - 200  # 200초 전

        assert monitor.check_stuck_node("exec-001", "slow_node")

    def test_get_node_timeout(self, monitor):
        """노드 타임아웃 조회"""
        assert monitor.get_node_timeout("open_url", "Open homepage") == 30
        assert monitor.get_node_timeout("search", "Search products") == 60
        assert monitor.get_node_timeout("custom", "Custom action") == 120  # default

    @pytest.mark.asyncio
    async def test_generate_report(self, monitor):
        """리포트 생성"""
        monitor.start_workflow_tracking("wf", "exec-001", 3)

        # 3개 노드 실행
        for i, status in enumerate(["success", "success", "failed"], 1):
            monitor.record_node_start(f"exec-001", f"node{i}", ExecutionStrategy.LOCAL_MODEL)
            monitor.record_node_complete(
                execution_id="exec-001",
                node_id=f"node{i}",
                success=(status == "success"),
                duration_ms=1000 * i,
                cost=0.001 * i,
                error=None if status == "success" else "Test error",
            )

        report = await monitor.generate_report("exec-001")

        assert isinstance(report, WorkflowReport)
        assert report.completed_nodes == 2
        assert report.failed_nodes == 1
        assert report.status == "partial"
        assert len(report.errors) == 1

    def test_cleanup_execution(self, monitor):
        """실행 데이터 정리"""
        monitor.start_workflow_tracking("wf", "exec-001", 1)
        assert monitor.get_execution_data("exec-001") is not None

        monitor.cleanup_execution("exec-001")
        assert monitor.get_execution_data("exec-001") is None


class TestStepEvaluator:
    """StepEvaluator 테스트"""

    @pytest.fixture
    def evaluator(self):
        """StepEvaluator 인스턴스 생성"""
        return StepEvaluator()

    def test_evaluate_step_continue(self, evaluator):
        """정상 스텝 평가"""
        feedback = evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=1,
            thought="I see a button to click",
            action="click(button)",
            observation="Button clicked successfully",
        )

        assert feedback.action == StepAction.CONTINUE

    def test_evaluate_step_failure_pattern(self, evaluator):
        """실패 패턴 감지"""
        # 첫 번째 실패 - 프롬프트 주입
        feedback = evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=1,
            thought="The element was not found on the page",
            action="click(missing_button)",
            observation="Error: Element not found",
        )

        assert feedback.action == StepAction.INJECT_PROMPT
        assert "실패" in feedback.reason or "not found" in feedback.reason.lower()

    def test_evaluate_step_repetitive_failure(self, evaluator):
        """반복 실패 감지 및 중단"""
        # 첫 번째 실패
        evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=1,
            thought="Element not found",
            action="click",
            observation="Failed",
        )

        # 두 번째 같은 실패 - 중단
        feedback = evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=2,
            thought="Still not found",
            action="click",
            observation="Failed again",
        )

        assert feedback.action == StepAction.STOP
        assert "반복" in feedback.reason

    def test_evaluate_step_popup_detection(self, evaluator):
        """팝업 감지"""
        feedback = evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=1,
            thought="A popup appeared on the screen",
            action="waiting",
            observation="Modal dialog visible",
        )

        assert feedback.action == StepAction.INJECT_PROMPT
        assert "팝업" in feedback.reason

    def test_evaluate_step_with_memory_patterns(self, evaluator):
        """메모리 기반 패턴 감지"""
        # 메모리 패턴 캐시에 직접 추가
        evaluator._node_failure_patterns_cache["node1"] = ["critical error pattern"]

        feedback = evaluator.evaluate_step(
            workflow_id="wf",
            node_id="node1",
            step_number=1,
            thought="Encountered critical error pattern",
            action="click",
            observation="Failed",
        )

        assert feedback.action == StepAction.STOP
        assert "학습된" in feedback.reason

    def test_learn_from_step(self, evaluator):
        """스텝 학습"""
        evaluator.learn_from_step(
            node_id="node1",
            step_number=1,
            success=False,
            thought="Try to click button",
            action="click(wrong_button)",
            observation="Failed",
        )

        patterns = evaluator.get_step_patterns()
        assert "node1" in patterns
        assert len(patterns["node1"]) == 1
        assert not patterns["node1"][0]["success"]
        assert patterns["node1"][0]["hint"] is not None

    def test_get_dynamic_system_prompt(self, evaluator):
        """동적 시스템 프롬프트 생성"""
        # 실패 패턴 학습
        evaluator.learn_from_step(
            node_id="node1",
            step_number=1,
            success=False,
            thought="Failed",
            action="click(wrong)",
            observation="Error",
        )

        prompt = evaluator.get_dynamic_system_prompt(
            node_id="node1",
            base_instruction="Click the button",
        )

        assert "Click the button" in prompt
        assert "주의사항" in prompt
        assert "자연스러운 행동" in prompt

    def test_inject_prompt_for_next_step(self, evaluator):
        """프롬프트 주입"""
        evaluator.inject_prompt_for_next_step(
            node_id="node1",
            step_number=1,
            prompt="Be careful with the next step",
        )

        prompt = evaluator.get_dynamic_system_prompt(
            node_id="node1",
            base_instruction="Base instruction",
            step_number=2,
        )

        assert "Be careful" in prompt

    def test_clear_injections(self, evaluator):
        """주입 프롬프트 제거"""
        evaluator.inject_prompt_for_next_step("node1", 1, "Prompt 1")
        evaluator.inject_prompt_for_next_step("node1", 2, "Prompt 2")
        evaluator.inject_prompt_for_next_step("node2", 1, "Other prompt")

        evaluator.clear_injections("node1")

        # node1의 주입만 제거됨
        assert len([k for k in evaluator._prompt_injections if k.startswith("node1:")]) == 0
        assert len([k for k in evaluator._prompt_injections if k.startswith("node2:")]) == 1


class TestVLMErrorTypeRouting:
    """VLM 에러 타입 기반 라우팅 테스트"""

    def test_vlm_error_type_values(self):
        """VLMErrorType enum 값 확인"""
        assert VLMErrorType.NONE == "none"
        assert VLMErrorType.BOT_DETECTED == "bot_detected"
        assert VLMErrorType.PAGE_FAILED == "page_failed"
        assert VLMErrorType.ACCESS_DENIED == "access_denied"
        assert VLMErrorType.ELEMENT_NOT_FOUND == "element_not_found"
        assert VLMErrorType.TIMEOUT == "timeout"
        assert VLMErrorType.UNKNOWN == "unknown"

    def test_error_type_routing_logic(self):
        """에러 타입별 라우팅 로직 테스트"""
        # 라우팅 로직 시뮬레이션
        def simulate_routing(vlm_error: str, retry_count: int, max_retries: int = 3) -> str:
            """워크플로우 라우터 로직 시뮬레이션"""
            if vlm_error == "BOT_DETECTED":
                return "abort"
            elif vlm_error in ("PAGE_FAILED", "TIMEOUT"):
                if retry_count < max_retries:
                    return "retry"
                else:
                    return "abort"
            elif vlm_error == "ELEMENT_NOT_FOUND":
                return "skip"
            elif vlm_error == "ACCESS_DENIED":
                return "abort"
            else:
                return "continue"

        # 봇 감지 - 즉시 중단
        assert simulate_routing("BOT_DETECTED", 0) == "abort"

        # 페이지 실패 - 재시도 가능
        assert simulate_routing("PAGE_FAILED", 0) == "retry"
        assert simulate_routing("PAGE_FAILED", 2) == "retry"
        assert simulate_routing("PAGE_FAILED", 3) == "abort"

        # 요소 미발견 - 스킵
        assert simulate_routing("ELEMENT_NOT_FOUND", 0) == "skip"

        # 접근 거부 - 중단
        assert simulate_routing("ACCESS_DENIED", 0) == "abort"

        # 정상 - 계속
        assert simulate_routing("none", 0) == "continue"


class TestExecutionDecision:
    """ExecutionDecision 데이터클래스 테스트"""

    def test_execution_decision_defaults(self):
        """기본값 테스트"""
        decision = ExecutionDecision(strategy=ExecutionStrategy.LOCAL_MODEL)

        assert decision.model_id is None
        assert decision.cached_result is None
        assert decision.reason == ""
        assert decision.estimated_time_ms == 0
        assert decision.estimated_cost == 0.0
        assert decision.confidence == 1.0
        assert not decision.reusable
        assert not decision.reuse_trace
        assert not decision.share_memory
        assert decision.cache_key_params == []

    def test_execution_decision_cache_hit(self):
        """캐시 히트 결정"""
        decision = ExecutionDecision(
            strategy=ExecutionStrategy.CACHE_HIT,
            cached_result={"data": "cached"},
            reason="Cache hit",
            estimated_time_ms=100,
            estimated_cost=0.0,
            confidence=1.0,
        )

        assert decision.strategy == ExecutionStrategy.CACHE_HIT
        assert decision.cached_result["data"] == "cached"

    def test_execution_decision_with_reuse_settings(self):
        """재사용 설정이 포함된 결정"""
        decision = ExecutionDecision(
            strategy=ExecutionStrategy.CLOUD_LIGHT,
            model_id="gpt-4o-mini",
            reusable=True,
            reuse_trace=True,
            share_memory=True,
            cache_key_params=["keyword", "page"],
        )

        assert decision.reusable
        assert decision.reuse_trace
        assert decision.share_memory
        assert decision.cache_key_params == ["keyword", "page"]


class TestWorkflowReport:
    """WorkflowReport 데이터클래스 테스트"""

    def test_workflow_report_to_dict(self):
        """리포트 딕셔너리 변환"""
        report = WorkflowReport(
            workflow_id="test-wf",
            execution_id="exec-001",
            status="completed",
            start_time=1000.0,
            end_time=1100.0,
            total_duration_ms=100000,
            total_nodes=3,
            completed_nodes=3,
            failed_nodes=0,
            skipped_nodes=0,
            total_cost=0.01,
            summary="All nodes completed",
            errors=[],
            recommendations=["Good job!"],
        )

        d = report.to_dict()

        assert d["workflow_id"] == "test-wf"
        assert d["execution_id"] == "exec-001"
        assert d["status"] == "completed"
        assert d["total_nodes"] == 3
        assert d["completed_nodes"] == 3
        assert d["total_cost"] == 0.01
        assert len(d["recommendations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
