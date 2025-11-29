"""
Common Subgraphs

LangGraph 재사용 가능한 서브그래프들:
- 에러 처리 서브그래프
- 메모리 업데이트 서브그래프
- VLM 분석 서브그래프

모든 워크플로우에서 공통으로 사용할 수 있습니다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph, END

from .workflow_base import WorkflowState, VLMErrorType
from ..services.code_agent_service import get_code_agent

logger = logging.getLogger(__name__)


class CommonSubgraphs:
    """
    재사용 가능한 LangGraph 서브그래프 컬렉션

    Example:
        ```python
        from cua2_core.workflows.common_subgraphs import CommonSubgraphs

        class MyWorkflow(WorkflowBase):
            def build_graph(self):
                graph = StateGraph(WorkflowState)

                # 메인 노드
                graph.add_node("main_task", self._main_task)

                # 에러 처리 서브그래프 추가
                error_handler = CommonSubgraphs.create_error_handler()
                graph.add_node("error_handler", error_handler)

                # 조건부 엣지
                graph.add_conditional_edges(
                    "main_task",
                    self._route_on_error,
                    {"error": "error_handler", "success": END}
                )
        ```
    """

    # =========================================
    # 에러 처리 서브그래프
    # =========================================

    @staticmethod
    def create_error_handler(
        max_retries: int = 3,
        on_bot_detected: Optional[Callable] = None,
        on_max_retries: Optional[Callable] = None,
    ) -> Callable:
        """
        에러 처리 서브그래프 생성

        VLM 에러 타입에 따라 적절한 처리를 수행합니다:
        - BOT_DETECTED: 즉시 중단
        - PAGE_FAILED/TIMEOUT: 재시도 (max_retries까지)
        - ELEMENT_NOT_FOUND: 스킵
        - ACCESS_DENIED: 중단

        Args:
            max_retries: 최대 재시도 횟수
            on_bot_detected: 봇 감지 시 호출할 커스텀 콜백
            on_max_retries: 재시도 한도 초과 시 호출할 콜백

        Returns:
            서브그래프 함수
        """
        graph = StateGraph(WorkflowState)

        def classify_error(state: WorkflowState) -> Dict[str, Any]:
            """에러 분류"""
            vlm_error = state.get("vlm_error_type", "")
            error_msg = state.get("error", "")
            retry_count = state.get("retry_count", 0)

            logger.info(
                f"[ErrorHandler] 에러 분류: type={vlm_error}, "
                f"retry={retry_count}/{max_retries}"
            )

            return {
                "error_classification": {
                    "type": vlm_error,
                    "message": error_msg,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                }
            }

        def decide_action(state: WorkflowState) -> str:
            """에러 타입에 따른 액션 결정"""
            vlm_error = state.get("vlm_error_type", "")
            retry_count = state.get("retry_count", 0)

            if vlm_error == "BOT_DETECTED":
                return "abort"
            elif vlm_error == "ACCESS_DENIED":
                return "abort"
            elif vlm_error in ("PAGE_FAILED", "TIMEOUT"):
                if retry_count < max_retries:
                    return "retry"
                else:
                    return "max_retries"
            elif vlm_error == "ELEMENT_NOT_FOUND":
                return "skip"
            else:
                return "abort"

        def retry_node(state: WorkflowState) -> Dict[str, Any]:
            """재시도 처리"""
            current_retry = state.get("retry_count", 0) + 1
            logger.info(f"[ErrorHandler] 재시도: {current_retry}/{max_retries}")

            return {
                "retry_count": current_retry,
                "error": None,  # 에러 클리어
                "vlm_error_type": None,
                "_action": "retry",
            }

        def skip_node(state: WorkflowState) -> Dict[str, Any]:
            """스킵 처리"""
            logger.info("[ErrorHandler] 현재 노드 스킵")
            return {
                "error": None,
                "vlm_error_type": None,
                "_action": "skip",
            }

        def abort_node(state: WorkflowState) -> Dict[str, Any]:
            """중단 처리"""
            vlm_error = state.get("vlm_error_type", "")
            logger.warning(f"[ErrorHandler] 워크플로우 중단: {vlm_error}")

            # 봇 감지 콜백 호출
            if vlm_error == "BOT_DETECTED" and on_bot_detected:
                try:
                    on_bot_detected(state)
                except Exception as e:
                    logger.error(f"[ErrorHandler] 봇 감지 콜백 실패: {e}")

            return {
                "should_stop": True,
                "status": "failed",
                "_action": "abort",
            }

        def max_retries_node(state: WorkflowState) -> Dict[str, Any]:
            """재시도 한도 초과 처리"""
            logger.warning(
                f"[ErrorHandler] 재시도 한도 초과: "
                f"{state.get('retry_count', 0)}/{max_retries}"
            )

            # 콜백 호출
            if on_max_retries:
                try:
                    on_max_retries(state)
                except Exception as e:
                    logger.error(f"[ErrorHandler] 재시도 한도 콜백 실패: {e}")

            return {
                "should_stop": True,
                "status": "failed",
                "error": f"Max retries ({max_retries}) exceeded",
                "_action": "max_retries",
            }

        # 노드 추가
        graph.add_node("classify", classify_error)
        graph.add_node("retry", retry_node)
        graph.add_node("skip", skip_node)
        graph.add_node("abort", abort_node)
        graph.add_node("max_retries", max_retries_node)

        # 엔트리 포인트
        graph.set_entry_point("classify")

        # 조건부 엣지
        graph.add_conditional_edges(
            "classify",
            decide_action,
            {
                "retry": "retry",
                "skip": "skip",
                "abort": "abort",
                "max_retries": "max_retries",
            }
        )

        # 종료 엣지
        graph.add_edge("retry", END)
        graph.add_edge("skip", END)
        graph.add_edge("abort", END)
        graph.add_edge("max_retries", END)

        return graph.compile()

    # =========================================
    # 메모리 업데이트 서브그래프
    # =========================================

    @staticmethod
    def create_memory_updater(
        letta_service: Optional[Any] = None,
    ) -> Callable:
        """
        메모리 업데이트 서브그래프 생성

        워크플로우 상태를 Letta 메모리에 저장합니다.

        Args:
            letta_service: Letta Memory 서비스 인스턴스

        Returns:
            서브그래프 함수
        """
        graph = StateGraph(WorkflowState)

        async def update_memory(state: WorkflowState) -> Dict[str, Any]:
            """메모리 업데이트"""
            if not letta_service:
                logger.warning("[MemoryUpdater] Letta 서비스 없음, 스킵")
                return {}

            workflow_id = state.get("workflow_id", "")
            current_node = state.get("current_node", "")

            try:
                # 진행 상황 업데이트
                completed = state.get("completed_nodes", [])
                failed = state.get("failed_nodes", [])

                await letta_service.update_progress(
                    workflow_id=workflow_id,
                    completed_nodes=completed,
                    failed_nodes=failed,
                    current_node=current_node,
                )

                # 에러 패턴 저장
                error = state.get("error")
                vlm_error = state.get("vlm_error_type")
                if error or vlm_error:
                    await letta_service.add_failure_pattern(
                        workflow_id=workflow_id,
                        node_id=current_node,
                        pattern=vlm_error or "unknown",
                        reason=error or "VLM error detected",
                    )

                logger.debug(f"[MemoryUpdater] 메모리 업데이트 완료: {current_node}")
                return {"_memory_updated": True}

            except Exception as e:
                logger.error(f"[MemoryUpdater] 메모리 업데이트 실패: {e}")
                return {"_memory_updated": False}

        graph.add_node("update", update_memory)
        graph.set_entry_point("update")
        graph.add_edge("update", END)

        return graph.compile()

    # =========================================
    # VLM 분석 서브그래프
    # =========================================

    @staticmethod
    def create_vlm_analyzer(
        vlm_runner: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
    ) -> Callable:
        """
        VLM 분석 서브그래프 생성

        스크린샷 기반 VLM 분석을 수행합니다.

        Args:
            vlm_runner: VLM Agent Runner 인스턴스
            orchestrator: Orchestrator 서비스 인스턴스

        Returns:
            서브그래프 함수
        """
        graph = StateGraph(WorkflowState)

        async def pre_analyze(state: WorkflowState) -> Dict[str, Any]:
            """VLM 분석 전처리"""
            node_id = state.get("current_node", "")
            
            # node_config 추가 (라우팅용)
            # workflow_base나 coupang_workflow에서 전달된 node_config를 data에 포함
            # 만약 없으면 현재 상태에서 찾기
            data = state.get("data", {})
            if "node_config" not in data:
                # state의 다른 곳에서 node_config 찾기 (예: metadata)
                # 또는 워크플로우에서 직접 전달해야 함
                pass

            # Orchestrator로 전략 결정
            if orchestrator:
                instruction = state.get("data", {}).get("instruction", "")
                decision = orchestrator.decide(
                    node_id=node_id,
                    instruction=instruction,
                    params=state.get("parameters", {}),
                )
                logger.info(
                    f"[VLMAnalyzer] 전략 결정: {decision.strategy.value}, "
                    f"model={decision.model_id}"
                )
                return {"_vlm_strategy": decision.strategy.value}

            return {}

        async def run_vlm(state: WorkflowState) -> Dict[str, Any]:
            """VLM 실행 또는 Code Agent 라우팅"""
            if not vlm_runner:
                logger.warning("[VLMAnalyzer] VLM Runner 없음")
                return {"error": "VLM Runner not configured"}

            instruction = state.get("data", {}).get("instruction", "")
            node_id = state.get("current_node", "")
            node_config = state.get("data", {}).get("node_config")  # WorkflowNode 설정
            
            # 노드 타입 확인
            node_type = getattr(node_config, 'node_type', None) if node_config else None
            
            # Code Agent 라우팅 (extract_data 노드는 Code Agent가 처리)
            if node_type == "extract_data":
                logger.info(f"[VLMAnalyzer] Code Agent로 라우팅: {node_id}")
                try:
                    code_agent = get_code_agent()
                    desktop = vlm_runner._current_agent.desktop if hasattr(vlm_runner, '_current_agent') else None
                    
                    if not desktop:
                        logger.error("[VLMAnalyzer] Desktop 인스턴스 없음")
                        return {"error": "Desktop not available"}
                    
                    result = await code_agent.extract_data(
                        desktop=desktop,
                        task_description=instruction,
                    )
                    
                    if result.get("success"):
                        logger.info(f"[VLMAnalyzer] Code Agent 성공: {len(result.get('data', []))} 항목")
                        return {
                            "data": {
                                **state.get("data", {}),
                                "extracted_data": result.get("data"),
                            }
                        }
                    else:
                        logger.error(f"[VLMAnalyzer] Code Agent 실패: {result.get('error')}")
                        return {
                            "error": result.get("error", "Code Agent failed"),
                            "details": result.get("details"),
                        }
                
                except Exception as e:
                    logger.error(f"[VLMAnalyzer] Code Agent 실행 오류: {e}")
                    return {"error": str(e)}
            
            # VLM Agent 실행 (기존 로직)
            try:
                result = await vlm_runner.run_instruction(
                    instruction=instruction,
                    workflow_id=state.get("workflow_id"),
                    node_id=node_id,
                    params=state.get("parameters", {}),
                )

                if result.success:
                    return {
                        "data": {
                            **state.get("data", {}),
                            "vlm_result": result.data,
                        }
                    }
                else:
                    return {
                        "error": result.error,
                        "vlm_error_type": result.early_stop_reason,
                    }

            except Exception as e:
                logger.error(f"[VLMAnalyzer] VLM 실행 실패: {e}")
                return {"error": str(e)}

        async def post_analyze(state: WorkflowState) -> Dict[str, Any]:
            """VLM 분석 후처리"""
            # Orchestrator 피드백
            if orchestrator:
                node_id = state.get("current_node", "")
                success = not state.get("error")

                orchestrator.learn_from_execution(
                    node_id=node_id,
                    success=success,
                    duration_ms=0,  # 실제 duration은 호출자에서 계산
                )

            return {"_vlm_analyzed": True}

        # 노드 추가
        graph.add_node("pre_analyze", pre_analyze)
        graph.add_node("run_vlm", run_vlm)
        graph.add_node("post_analyze", post_analyze)

        # 엣지
        graph.set_entry_point("pre_analyze")
        graph.add_edge("pre_analyze", "run_vlm")
        graph.add_edge("run_vlm", "post_analyze")
        graph.add_edge("post_analyze", END)

        return graph.compile()

    # =========================================
    # 유틸리티 함수
    # =========================================

    @staticmethod
    def route_on_error(state: WorkflowState) -> str:
        """
        에러 발생 시 라우팅 (유틸리티)

        워크플로우에서 조건부 엣지 함수로 사용합니다.

        Returns:
            "error" 또는 "success"
        """
        if state.get("error") or state.get("vlm_error_type"):
            return "error"
        return "success"

    @staticmethod
    def should_retry(state: WorkflowState, max_retries: int = 3) -> bool:
        """
        재시도 여부 결정 (유틸리티)

        Args:
            state: 워크플로우 상태
            max_retries: 최대 재시도 횟수

        Returns:
            재시도 여부
        """
        vlm_error = state.get("vlm_error_type", "")
        retry_count = state.get("retry_count", 0)

        # 재시도 가능한 에러 타입
        retryable = {"PAGE_FAILED", "TIMEOUT", "ELEMENT_NOT_FOUND"}

        return vlm_error in retryable and retry_count < max_retries
