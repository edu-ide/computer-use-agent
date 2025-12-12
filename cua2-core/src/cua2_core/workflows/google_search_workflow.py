"""
구글 검색 자동화 워크플로우 - FaraWebSurfer 기반 (Fara-7B)

워크플로우 단계:
1. search_google - 구글에서 검색어 검색 및 결과 수집
2. complete - 완료
"""

import asyncio
import base64
import io
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    VLMErrorType,
)
from ..services.agent_activity_log import (
    log_fara,
    ActivityType,
)
from ..services.vlm_agent_runner import VLMStepLog
from ..fara import FaraWebSurfer, WebSurferConfig

logger = logging.getLogger(__name__)


class GoogleSearchWorkflow(WorkflowBase):
    """
    구글 검색 자동화 워크플로우 (FaraWebSurfer 기반)

    Fara-7B VLM을 사용하여 스크린샷 기반 GUI 자동화 수행.
    FaraWebSurfer의 전용 프롬프트 형식과 액션 체계를 사용.

    Flow:
    1. search_google - 구글 검색 실행 및 결과 수집
    2. complete - 완료
    """

    def __init__(self, agent_runner=None):
        """
        Args:
            agent_runner: VLMAgentRunner (미사용, FaraWebSurfer 대신 사용)
        """
        super().__init__()
        self._agent_runner = agent_runner  # 호환성 유지
        self._web_surfer: Optional[FaraWebSurfer] = None
        self._on_step_callback: Optional[Callable] = None
        self._step_counter = 0

    def set_step_callback(self, callback: Callable):
        """스텝 콜백 설정 - 실시간 UI 업데이트용"""
        self._on_step_callback = callback

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="google-search",
            name="구글 검색 자동화",
            description="구글에서 키워드를 검색하고 결과를 수집합니다. Fara-7B VLM을 사용한 GUI 자동화로 실시간 검색을 수행합니다.",
            icon="Search",
            color="#4285f4",
            category="search",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "label": "검색어",
                    "placeholder": "검색할 키워드를 입력하세요",
                    "required": True,
                },
                {
                    "name": "num_results",
                    "type": "number",
                    "label": "결과 수",
                    "default": 10,
                    "min": 1,
                    "max": 50,
                },
                {
                    "name": "language",
                    "type": "string",
                    "label": "검색 언어",
                    "default": "ko",
                    "options": [
                        {"value": "ko", "label": "한국어"},
                        {"value": "en", "label": "English"},
                        {"value": "ja", "label": "日本語"},
                    ],
                },
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="search_google",
                display_name="구글 검색",
                description="구글에서 검색어를 검색하고 결과 수집",
                on_success="complete",
                on_failure="error_handler",
                agent_type="FaraAgent",
                model_id="/mnt/sda1/models/llm/GELab-Zero-4B-preview",
                timeout_sec=180,
                avg_duration_sec=60,
            ),
            WorkflowNode(
                name="complete",
                display_name="완료",
                description="워크플로우 완료",
                node_type="end",
                timeout_sec=10,
                avg_duration_sec=1,
            ),
            WorkflowNode(
                name="error_handler",
                display_name="오류 처리",
                description="오류 발생 시 처리",
                on_success="complete",
                node_type="error",
                timeout_sec=30,
                avg_duration_sec=5,
            ),
        ]

    @property
    def start_node(self) -> str:
        return "search_google"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """노드별 실행 로직"""
        handlers = {
            "search_google": self._search_google,
            "complete": self._complete,
            "error_handler": self._error_handler,
        }

        handler = handlers.get(node_name)
        if handler:
            print(f"[GoogleSearchWorkflow] Executing node: {node_name}")
            result = await handler(state)
            print(f"[GoogleSearchWorkflow] Node completed: {node_name} (success={result.success})")
            return result

        return NodeResult(success=False, error=f"Unknown node: {node_name}")

    def _add_step_to_node_logs(
        self,
        node_name: str,
        action: Dict[str, Any],
        screenshot_base64: Optional[str] = None,
        thought: Optional[str] = None,
    ):
        """스텝 로그를 노드 로그에 추가 (실시간 패널 표시용)"""
        if not hasattr(self, '_current_state') or self._current_state is None:
            return

        self._step_counter += 1

        node_logs = self._current_state.get("node_logs", {})
        if node_name not in node_logs:
            node_logs[node_name] = []

        step_data = {
            "step_number": self._step_counter,
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_base64,
            "thought": thought or f"Fara-7B 액션 실행: {action.get('name', 'unknown')}",
            "action": str(action),
            "observation": action.get("result", {}).get("message", ""),
            "error": None,
            "tool_calls": [],
            "orchestrator_feedback": None,
        }

        # 디버깅 로그
        print(f"[StepLog] {node_name} step {self._step_counter}: "
              f"action={action.get('name')}, screenshot={bool(screenshot_base64)}")

        node_logs[node_name].append(step_data)
        self._current_state["node_logs"] = node_logs

        # 활동 로그
        log_fara(
            ActivityType.EXECUTION,
            f"스텝 {self._step_counter}: {action.get('name', 'unknown')}",
            details={
                "action": action.get("name"),
                "arguments": action.get("arguments"),
            },
            execution_id=self._current_state.get("execution_id", ""),
            node_id=node_name,
        )

    async def _capture_screenshot_base64(self) -> Optional[str]:
        """스크린샷 캡처 후 base64 반환"""
        if not self._web_surfer or not self._web_surfer._controller:
            return None
        try:
            screenshot = await self._web_surfer._controller.screenshot()
            if screenshot:
                buffered = io.BytesIO()
                screenshot.save(buffered, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.error(f"스크린샷 캡처 실패: {e}")
        return None

    # ========== 노드 구현 ==========

    async def _search_google(self, state: WorkflowState) -> NodeResult:
        """VLMAgentRunner를 사용한 구글 검색"""
        try:
            parameters = state.get("parameters", {})
            query = parameters.get("query", "")
            language = parameters.get("language", "ko")
            num_results = parameters.get("num_results", 10)

            if not query:
                return NodeResult(success=False, error="검색어가 없습니다")

            # 활동 로그 시작
            log_fara(
                ActivityType.EXECUTION,
                f"구글 검색 시작 (VLMAgent): {query}",
                details={"query": query, "language": language},
                execution_id=state.get("execution_id", ""),
                node_id="search_google",
            )

            # VLMAgentRunner 사용 (기존 FaraWebSurfer 대체)
            from ..services.vlm_agent_runner import VLMAgentRunner

            # Runner 초기화
            runner = VLMAgentRunner(
                model_id="/mnt/sda1/models/llm/GELab-Zero-4B-preview",
                max_steps=20,
                agent_type="headless_browser",
            )
            
            # 세션 초기화
            session_id = state.get("execution_id", f"google-search-{int(datetime.now().timestamp())}")
            if not await runner.initialize(session_id):
                return NodeResult(success=False, error="에이전트 초기화 실패")

            # 구글 검색 작업 프롬프트 구성
            instruction = f"""
            Task: Search on Google
            1. Go to 'https://www.google.co.kr' (or appropriate domain for {language})
            2. Type '{query}' into the search bar
            3. Press Enter to search
            4. Wait for results to appear
            5. Report the titles and URLs of the top {num_results} results
            6. Summarize the findings
            """

            # 스텝 콜백 (UI 로그 연동)
            def on_step(step_log):
                self._add_step_to_node_logs(
                    "search_google",
                    {"name": step_log.action or "thinking", "arguments": {}},
                    step_log.screenshot, # Before screenshot
                    step_log.thought or "Thinking...",
                )
                # After screenshot handling if needed (VLMStepLog has screenshot_after)
                if step_log.screenshot_after:
                     self._add_step_to_node_logs(
                        "search_google",
                        {"name": "observation", "arguments": {}},
                        step_log.screenshot_after,
                        f"관찰: {step_log.observation}",
                    )

            # 실행
            result = await runner.run_instruction(
                instruction=instruction,
                on_step=on_step,
                workflow_id=state.get("workflow_id"),
                node_id="search_google",
            )

            # 리소스 정리 (샌드박스 등은 Runner가 관리하거나 자동 정리됨)
            # runner.stop() # 필요한 경우 명시적 종료

            # 활동 로그 완료
            log_fara(
                ActivityType.EXECUTION,
                f"구글 검색 완료",
                details={
                    "success": result.success,
                    "steps": len(result.steps),
                },
                execution_id=state.get("execution_id", ""),
                node_id="search_google",
            )

            if result.success:
                return NodeResult(
                    success=True,
                    data={
                        "query": query,
                        "status": "success",
                        "message": result.data.get("final_answer", "완료"),
                        "rounds": len(result.steps),
                    }
                )
            else:
                return NodeResult(
                    success=False,
                    error=result.error or "검색 실패",
                    data={"status": "failed"},
                )

        except Exception as e:
            logger.error(f"구글 검색 오류: {e}")
            return NodeResult(success=False, error=str(e))

    async def _complete(self, state: WorkflowState) -> NodeResult:
        """워크플로우 완료"""
        data = state.get("data", {})
        query = state.get("parameters", {}).get("query", "")

        return NodeResult(
            success=True,
            data={
                "query": query,
                "completed": True,
                "message": data.get("message", "검색 완료"),
            }
        )

    async def _error_handler(self, state: WorkflowState) -> NodeResult:
        """오류 처리"""
        error = state.get("error", "Unknown error")

        print(f"[GoogleSearchWorkflow] Error: {error}")

        log_fara(
            ActivityType.ERROR,
            f"오류 발생: {error[:50] if error else 'Unknown'}",
            details={"error": error},
            execution_id=state.get("execution_id", ""),
            node_id="error_handler",
        )

        return NodeResult(
            success=True,
            data={"error_handled": True, "original_error": error},
            next_node="complete"
        )