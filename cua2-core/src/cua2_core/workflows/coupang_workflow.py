"""
쿠팡 상품 수집 워크플로우 - LangGraph 기반

Letta 메모리 통합:
- 워크플로우 시작 시 Letta 에이전트 생성
- 각 노드 실행 시 메모리 블록 업데이트
- VLM 에이전트에 메모리 컨텍스트 전달
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowNode,
    WorkflowState,
    NodeResult,
    NodeStatus,
)
from ..services.coupang_db_service import get_coupang_db
from ..services.letta_memory_service import (
    get_letta_memory_service,
    WorkflowMemoryConfig,
    LettaMemoryService,
)
from ..services.node_reuse_analyzer import (
    get_node_reuse_analyzer,
    NodeReuseAnalyzer,
    ReuseDecision,
)
from ..services.orchestrator_service import (
    get_orchestrator_service,
    OrchestratorService,
    ExecutionStrategy,
    ExecutionDecision,
    ErrorAction,
    WorkflowReport,
)
from ..services.agent_activity_log import (
    log_vlm,
    log_memory,
    ActivityType,
)
from ..services.vlm_agent_runner import VLMStepLog
from ..models.coupang_models import CoupangProduct


class CoupangCollectWorkflow(WorkflowBase):
    """
    쿠팡 비로켓배송 상품 수집 워크플로우

    Flow:
    1. open_coupang - 쿠팡 열기
    2. search_keyword - 키워드 검색
    3. analyze_page - 페이지 분석 및 상품 수집
    4. next_page - 다음 페이지로 이동 (loop back to analyze_page)
    5. find_related - 연관 키워드 탐색
    6. complete - 완료
    """

    def __init__(self, agent_runner=None):
        """
        Args:
            agent_runner: VLMAgentRunner 인스턴스
        """
        super().__init__()
        self._agent_runner = agent_runner
        self._db = get_coupang_db()
        self._on_step_callback: Optional[Callable] = None

        # Letta 메모리 서비스
        self._letta: LettaMemoryService = get_letta_memory_service()
        self._letta_initialized: bool = False

        # 노드 재사용 분석기 (자동 학습)
        self._reuse_analyzer: NodeReuseAnalyzer = get_node_reuse_analyzer()
        self._auto_learn_reuse: bool = True  # 재사용 설정 자동 학습 활성화

        # Orchestrator 서비스 (실행 전략 결정)
        self._orchestrator: OrchestratorService = get_orchestrator_service()
        self._use_orchestrator: bool = True  # Orchestrator 사용 여부 (8081 서버 필요)

        # VLMAgentRunner에 Orchestrator 주입 (Step-level intervention을 위해)
        if self._agent_runner and hasattr(self._agent_runner, "_orchestrator") and self._agent_runner._orchestrator is None:
            self._agent_runner._orchestrator = self._orchestrator

    def set_step_callback(self, callback: Callable):
        """스텝 콜백 설정 - 실시간 UI 업데이트용"""
        self._on_step_callback = callback

    async def _init_letta_memory(self, keyword: str):
        """Letta 메모리 에이전트 초기화"""
        if self._letta_initialized:
            return

        try:
            # 쿠팡 워크플로우용 메모리 설정 생성
            config = WorkflowMemoryConfig.for_coupang(keyword=keyword)

            # Letta 에이전트 생성
            await self._letta.create_workflow_agent(config)
            self._letta_initialized = True
            print(f"[CoupangWorkflow] Letta 메모리 에이전트 초기화 완료: {keyword}")

        except Exception as e:
            print(f"[CoupangWorkflow] Letta 메모리 초기화 실패 (폴백 모드 사용): {e}")
            self._letta_initialized = True  # 폴백 모드에서도 초기화된 것으로 처리

    async def _update_memory_on_node_start(self, node_name: str, state: WorkflowState):
        """노드 시작 시 메모리 업데이트"""
        if not self._letta_initialized:
            return

        workflow_id = self.config.id
        data = state.get("data", {})
        completed_nodes = state.get("completed_nodes", [])

        # 진행률 계산
        total_nodes = len(self.nodes) - 2  # error_handler, complete 제외
        progress = int((len(completed_nodes) / total_nodes) * 100) if total_nodes > 0 else 0

        # 워크플로우 상태 업데이트
        await self._letta.update_workflow_state(
            workflow_id=workflow_id,
            current_node=node_name,
            completed_nodes=completed_nodes,
            progress=progress,
            additional_info=f"- 키워드: {data.get('current_keyword', '-')}",
        )

        # 태스크 컨텍스트 업데이트
        node_config = self._get_node_config(node_name)
        if node_config:
            await self._letta.update_task_context(
                workflow_id=workflow_id,
                status="실행 중",
                current_goal=node_config.description,
                next_action=node_config.display_name,
            )

    async def _update_memory_on_node_complete(
        self,
        node_name: str,
        state: WorkflowState,
        result: NodeResult,
    ):
        """노드 완료 시 메모리 업데이트"""
        if not self._letta_initialized:
            return

        workflow_id = self.config.id
        data = state.get("data", {})

        # 수집 데이터 업데이트 (analyze_page, save_products 노드)
        if node_name in ["analyze_page", "save_products"]:
            collected_count = data.get("collected_count", 0)
            pages_analyzed = data.get("pages_analyzed", 0)

            await self._letta.update_collected_data(
                workflow_id=workflow_id,
                total_items=collected_count,
                pages_analyzed=pages_analyzed,
            )

        # 에러 발생 시 에러 히스토리 업데이트
        if not result.success and result.error:
            await self._letta.add_error(
                workflow_id=workflow_id,
                error=f"[{node_name}] {result.error}",
            )

        # 재사용 학습: 실행 결과 기록
        if self._auto_learn_reuse:
            await self._record_and_learn_reuse(node_name, state, result)

    def _get_recent_execution_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """최근 실행 이력 반환 (Orchestrator 컨텍스트용)"""
        history = []

        # node_logs에서 최근 실행 정보 추출
        node_logs = self._current_state.get("node_logs", {})
        completed_nodes = self._current_state.get("completed_nodes", [])

        for node_id in completed_nodes[-limit:]:
            logs = node_logs.get(node_id, [])
            success = len(logs) > 0 and not any(
                "error" in str(log).lower() for log in logs
            )
            history.append({
                "node_id": node_id,
                "success": success,
                "steps_count": len(logs),
            })

        return history

    async def _record_and_learn_reuse(
        self,
        node_name: str,
        state: WorkflowState,
        result: NodeResult,
    ):
        """실행 결과 기록 및 재사용 설정 학습"""
        workflow_id = self.config.id
        execution_id = state.get("execution_id", "")
        parameters = state.get("parameters", {})
        node_logs = self._current_state.get("node_logs", {})
        steps_count = len(node_logs.get(node_name, []))

        # 실행 결과 기록
        self._reuse_analyzer.record_execution(
            node_id=node_name,
            workflow_id=workflow_id,
            execution_id=execution_id,
            success=result.success,
            steps_count=steps_count,
            parameters=parameters,
            data_produced=result.data if result.data else {},
            error=result.error,
        )

        # 분석 및 학습 (충분한 데이터가 쌓이면)
        analysis = self._reuse_analyzer.analyze_node(node_name, workflow_id)

        if analysis.decision != ReuseDecision.UNCERTAIN:
            # Letta 메모리에 학습된 설정 저장
            settings = {
                "decision": analysis.decision.value,
                "confidence": analysis.confidence,
                "reason": analysis.reason,
                "reusable": analysis.recommended_reusable,
                "reuse_trace": analysis.recommended_reuse_trace,
                "share_memory": analysis.recommended_share_memory,
                "cache_key_params": analysis.recommended_cache_key_params,
            }

            await self._letta.update_node_reuse_settings(
                workflow_id=workflow_id,
                node_id=node_name,
                settings=settings,
            )

            print(f"[CoupangWorkflow] {node_name}: 재사용 설정 학습됨 - {analysis.decision.value} ({analysis.confidence:.1%})")

    async def _get_memory_context(self) -> str:
        """VLM 에이전트에 전달할 메모리 컨텍스트 조회"""
        if not self._letta_initialized:
            return ""

        return await self._letta.get_context_for_agent(self.config.id)

    async def _cleanup_letta_memory(self):
        """워크플로우 종료 시 Letta 메모리 정리 (안전 모드)"""
        if self._letta_initialized:
            try:
                # 타임아웃 2초로 제한
                await asyncio.wait_for(self._letta.cleanup_agent(self.config.id), timeout=2.0)
            except Exception as e:
                print(f"[CoupangWorkflow] 메모리 정리 실패 (무시됨): {e}")
            finally:
                self._letta_initialized = False

    @property
    def config(self) -> WorkflowConfig:
        return WorkflowConfig(
            id="coupang-collect",
            name="쿠팡 상품 수집",
            description="키워드로 쿠팡에서 로켓배송이 아닌 상품을 자동으로 수집합니다.",
            icon="ShoppingCart",
            color="#e31937",
            category="e-commerce",
            parameters=[
                {
                    "name": "keyword",
                    "type": "string",
                    "label": "검색 키워드",
                    "placeholder": "예: 무선이어폰",
                    "required": True,
                },
                {
                    "name": "max_pages",
                    "type": "number",
                    "label": "최대 페이지 수",
                    "default": 5,
                    "min": 1,
                    "max": 20,
                },
                {
                    "name": "follow_related",
                    "type": "boolean",
                    "label": "연관 키워드 탐색",
                    "default": True,
                },
            ],
        )

    @property
    def nodes(self) -> List[WorkflowNode]:
        return [
            WorkflowNode(
                name="open_coupang",
                display_name="쿠팡 열기",
                description="쿠팡 웹사이트 열기",
                on_success="search_keyword",
                on_failure="error_handler",
                node_type="vlm",
                timeout_sec=60,  # 1분
                avg_duration_sec=15,  # 평균 15초
                instruction="""Open Chrome browser and navigate to https://www.coupang.com

Steps:
1. Use open_url("https://www.coupang.com") to navigate to Coupang
2. Wait for the page to fully load
3. Confirm you see the Coupang homepage""",
            ),
            WorkflowNode(
                name="search_keyword",
                display_name="키워드 검색",
                description="키워드로 상품 검색",
                on_success="analyze_page",
                on_failure="error_handler",
                node_type="vlm",
                timeout_sec=60,  # 1분
                avg_duration_sec=20,  # 평균 20초
                instruction="""Search for "{keyword}" on Coupang:

Steps:
1. Find the search input box at the top of the page
2. Click on the search box
3. Type the keyword using write()
4. Press Enter to search using press(["enter"])
5. Wait for search results to load using wait(3)
6. Confirm search results are displayed""",
            ),
            WorkflowNode(
                name="analyze_page",
                display_name="페이지 분석",
                description="페이지 분석 및 비로켓 상품 수집",
                on_success="save_products",
                on_failure="error_handler",
                node_type="vlm",
                timeout_sec=180,  # 3분 (스크롤 + 분석)
                avg_duration_sec=90,  # 평균 1분 30초
                instruction="""Analyze the current Coupang search results page:

Look at the product listings and identify products that do NOT have:
- 로켓배송 badge (blue rocket icon)
- 로켓직구 badge
- 로켓와우 badge

These are "일반배송" (regular delivery) products.

Steps:
1. Scroll down slowly to see all products using scroll(500, 500, "down", 2)
2. Look for products without rocket badges
3. For each non-rocket product, extract:
   - Product name (상품명)
   - Price (가격)
   - Product URL or ID if visible
   - Seller name if visible (판매자)
4. Continue scrolling to see more products
5. Return a list of found products in JSON format""",
            ),
            WorkflowNode(
                name="save_products",
                display_name="상품 저장",
                description="수집한 상품을 DB에 저장",
                on_success="check_next_page",
                on_failure="error_handler",
                node_type="process",
                timeout_sec=30,  # 30초
                avg_duration_sec=2,  # 평균 2초 (빠른 DB 작업)
            ),
            WorkflowNode(
                name="check_next_page",
                display_name="다음 페이지",
                description="다음 페이지 확인",
                on_success="analyze_page",  # 다음 페이지 있으면 다시 분석
                on_failure="find_related",   # 없으면 연관 키워드로
                node_type="vlm",
                timeout_sec=60,  # 1분
                avg_duration_sec=25,  # 평균 25초
                instruction="""Check if there is a next page and navigate to it:

Steps:
1. Scroll to the bottom of the page to find pagination
2. Look for pagination controls (page numbers or "다음" button)
3. If next page exists, click on the next page number or "다음" button
4. Wait for the new page to load using wait(3)
5. If no next page available, report "no_next_page" """,
            ),
            WorkflowNode(
                name="find_related",
                display_name="연관 키워드",
                description="연관 키워드 탐색",
                on_success="search_keyword",  # 연관 키워드로 다시 검색
                on_failure="complete",  # 없으면 완료
                node_type="vlm",
                timeout_sec=60,  # 1분
                avg_duration_sec=30,  # 평균 30초
                instruction="""Find related search keywords on Coupang:

Steps:
1. Look for "연관 검색어" or "관련 검색어" section on the page
2. Or look for suggested keywords at the top or bottom of results
3. Find a new keyword that is not already searched
4. If you find a good related keyword, click on it
5. If no new related keywords exist, report "no_related_found" """,
            ),
            WorkflowNode(
                name="complete",
                display_name="완료",
                description="수집 완료",
                node_type="end",
                timeout_sec=10,
                avg_duration_sec=1,
            ),
            WorkflowNode(
                name="error_handler",
                display_name="에러 처리",
                description="에러 처리",
                node_type="error",
                timeout_sec=30,
                avg_duration_sec=5,
            ),
        ]

    @property
    def start_node(self) -> str:
        return "open_coupang"

    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """노드별 실행 로직 (Letta 메모리 통합)"""

        # 첫 번째 노드에서 Letta 메모리 초기화
        if node_name == self.start_node:
            keyword = state.get("parameters", {}).get("keyword", "")
            await self._init_letta_memory(keyword)

        # 노드 시작 시 메모리 업데이트
        await self._update_memory_on_node_start(node_name, state)

        handlers = {
            "open_coupang": self._open_coupang,
            "search_keyword": self._search_keyword,
            "analyze_page": self._analyze_page,
            "save_products": self._save_products,
            "check_next_page": self._check_next_page,
            "find_related": self._find_related,
            "complete": self._complete,
            "error_handler": self._error_handler,
        }

        handler = handlers.get(node_name)
        if handler:
            print(f"[Workflow] Executing node: {node_name}")
            result = await handler(state)
            print(f"[Workflow] Node completed: {node_name} (success={result.success}, next={result.next_node})")

            # 노드 완료 시 메모리 업데이트
            await self._update_memory_on_node_complete(node_name, state, result)

            # 완료 또는 에러 노드에서 메모리 정리
            if node_name in ["complete", "error_handler"]:
                await self._cleanup_letta_memory()

            return result

        return NodeResult(success=False, error=f"Unknown node: {node_name}")

    def _add_step_to_node_logs(self, state: WorkflowState, node_name: str, step_log):
        """스텝 로그를 노드 로그에 추가

        Note: LangGraph state는 불변이므로 직접 수정 대신 _current_state를 업데이트
        """
        # 현재 워크플로우 인스턴스의 _current_state에서 node_logs 가져오기
        if not hasattr(self, '_current_state') or self._current_state is None:
            return {}

        node_logs = self._current_state.get("node_logs", {})
        if node_name not in node_logs:
            node_logs[node_name] = []

        # tool_calls를 JSON 직렬화 가능한 형태로 변환
        tool_calls_serializable = []
        if step_log.tool_calls:
            for tc in step_log.tool_calls:
                if hasattr(tc, 'model_dump'):
                    # Pydantic 모델인 경우
                    tool_calls_serializable.append(tc.model_dump())
                elif hasattr(tc, '__dict__'):
                    # 일반 객체인 경우
                    tool_calls_serializable.append(str(tc))
                else:
                    tool_calls_serializable.append(str(tc))

        step_data = {
            "step_number": step_log.step_number,
            "timestamp": step_log.timestamp,
            "screenshot": step_log.screenshot,
            "thought": step_log.thought,
            "action": step_log.action,
            "observation": step_log.observation,
            "error": step_log.error,
            "tool_calls": tool_calls_serializable,
            "orchestrator_feedback": getattr(step_log, "orchestrator_feedback", None),
        }

        # 디버깅: 스텝 데이터 로깅
        print(f"[StepLog] {node_name} step {step_log.step_number}: "
              f"thought={bool(step_log.thought)}, "
              f"action={bool(step_log.action)}, "
              f"observation={bool(step_log.observation)}, "
              f"screenshot={bool(step_log.screenshot)}")
        if step_log.thought:
            print(f"  -> thought: {step_log.thought[:100]}...")

        node_logs[node_name].append(step_data)

        # _current_state에 직접 업데이트 (WebSocket 조회 시 즉시 반영)
        self._current_state["node_logs"] = node_logs

        if self._on_step_callback:
            self._on_step_callback(node_name, step_log)

        return node_logs

    def _get_node_config(self, node_name: str) -> Optional[WorkflowNode]:
        """노드 설정 가져오기"""
        for node in self.nodes:
            if node.name == node_name:
                return node
        return None

    async def _run_vlm_instruction(
        self,
        instruction: str,
        state: WorkflowState,
        node_name: str,
    ) -> tuple[bool, Dict[str, Any], Optional[str]]:
        """
        VLM 명령 실행 (Orchestrator 패턴)

        실행 흐름:
        1. Orchestrator가 실행 전략 결정
        2. CACHE_HIT → 즉시 반환 (VLM 호출 없음!)
        3. RULE_BASED → 규칙 기반 실행
        4. LOCAL_MODEL/CLOUD → VLM 실행
        """
        import time
        start_time = time.time()

        node_config = self._get_node_config(node_name)
        params = state.get("parameters", {})
        decision = None  # Orchestrator decision

        # 현재 실행 ID (Orchestrator 로그용)
        current_execution_id = state.get("execution_id", self.config.id)

        # === 1. Orchestrator 결정 (캐시 체크 포함) ===
        if self._use_orchestrator:
            # 비동기 Orchestrator-8B 모델 사용 (서버 실행 중일 때)
            # 서버 미실행시 자동으로 규칙 기반으로 fallback
            decision = await self._orchestrator.decide_async(
                workflow_id=self.config.id,
                node_id=node_name,
                instruction=instruction,
                params=params,
                node_config=node_config,
                execution_id=current_execution_id,  # 실행 ID 전달
                execution_history=self._get_recent_execution_history(),
            )
            
            # Calculate decision time
            decision_time = int((time.time() - start_time) * 1000)

            # Orchestrator 결정을 로그에 기록 (Step 0)
            if decision:
                self._add_step_to_node_logs(state, node_name, VLMStepLog(
                    step_number=0,
                    timestamp=datetime.now().isoformat(),
                    thought=f"Orchestrator Decision: {decision.strategy.value}\nReason: {decision.reason}\nTime: {decision_time}ms",
                    action=f"[Strategy] {decision.strategy.value}",
                    observation=f"Model: {decision.model_id}\nReuse: {decision.reuse_trace}\nTime: {decision_time}ms",
                    orchestrator_feedback={
                        "action": "strategy_selected",
                        "reason": f"{decision.reason} ({decision_time}ms)",
                        "learned_pattern": decision.strategy.value
                    }
                ))

            # CACHE_HIT: 즉시 반환! (VLM 호출 없음, 판단도 스킵)
            if decision.strategy == ExecutionStrategy.CACHE_HIT:
                elapsed_ms = int((time.time() - start_time) * 1000)
                print(f"[Orchestrator] {node_name}: ⚡ CACHE_HIT - {elapsed_ms}ms (VLM 스킵)")

                cached = decision.cached_result
                return True, cached.get("data", {}), None

            # RULE_BASED: 규칙 기반 실행 (VLM 없이)
            if decision.strategy == ExecutionStrategy.RULE_BASED:
                elapsed_ms = int((time.time() - start_time) * 1000)
                print(f"[Orchestrator] {node_name}: 📋 RULE_BASED - {elapsed_ms}ms")
                # TODO: 규칙 기반 실행 구현
                # 현재는 VLM으로 폴백
                pass

            # 모델 선택 로깅
            print(f"[Orchestrator] {node_name}: 🤖 {decision.strategy.value} "
                  f"(model={decision.model_id}, reason={decision.reason})")

        # === 2. VLM 실행이 필요한 경우 ===
        if not self._agent_runner:
            return True, {}, None

        def on_step(step_log):
            self._add_step_to_node_logs(state, node_name, step_log)

            # Activity Log에 스텝 완료 기록
            if step_log.step_number > 0:
                log_vlm(
                    ActivityType.EXECUTION,
                    f"스텝 {step_log.step_number} 실행",
                    details={
                        "action": step_log.action,
                        "thought": step_log.thought[:50] + "..." if step_log.thought else None,
                    },
                    execution_id=current_execution_id,
                    node_id=node_name,
                    duration_ms=getattr(step_log, "duration_ms", None),
                )

            # 동기 콜백에서 비동기 작업 안전하게 실행
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._update_observation_on_step(node_name, step_log))
            except RuntimeError:
                # 이벤트 루프가 없으면 동기적으로 처리하거나 스킵
                pass

        # 재사용 설정: Orchestrator 판단 > 학습된 설정 > 기본값
        # Orchestrator가 이미 decision에 재사용 설정을 포함함
        if self._use_orchestrator and decision:
            # Orchestrator-8B가 판단한 재사용 설정 사용
            reusable = decision.reusable
            reuse_trace = decision.reuse_trace
            share_memory = decision.share_memory
            cache_key_params = decision.cache_key_params
            print(f"[Orchestrator] {node_name}: 재사용설정 - "
                  f"reuse_trace={reuse_trace}, share_memory={share_memory}, "
                  f"cache_key_params={cache_key_params}")
        else:
            # Fallback: 학습된 설정 또는 기본값
            learned_settings = self._reuse_analyzer.get_recommended_settings(
                node_name, self.config.id
            )
            use_learned = learned_settings.get("confidence", 0) > 0.7

            if use_learned:
                reusable = learned_settings.get("reusable", False)
                reuse_trace = learned_settings.get("reuse_trace", False)
                share_memory = learned_settings.get("share_memory", False)
                cache_key_params = learned_settings.get("cache_key_params", [])
            else:
                reusable = node_config.reusable if node_config else False
                reuse_trace = node_config.reuse_trace if node_config else False
                share_memory = node_config.share_memory if node_config else False
                cache_key_params = node_config.cache_key_params if node_config else []

        # 메모리 컨텍스트
        memory_context = ""
        if share_memory:
            memory_context = await self._get_memory_context()

        enhanced_instruction = instruction
        if memory_context:
            enhanced_instruction = f"""## 이전 작업 컨텍스트 (메모리)
{memory_context}

## 현재 태스크
{instruction}"""

        # 활동 로그: VLM 실행 시작
        log_vlm(
            ActivityType.EXECUTION,
            f"{node_name} 실행 시작",
            details={
                "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
                "share_memory": share_memory,
                "reuse_trace": reuse_trace,
            },
            execution_id=current_execution_id,
            node_id=node_name,
        )

        # VLM 실행
        result = await self._agent_runner.run_instruction(
            enhanced_instruction,
            on_step=on_step,
            workflow_id=self.config.id,
            node_id=node_name,
            params=params,
            reuse_trace=reuse_trace,
            reusable=reusable,
            cache_key_params=cache_key_params,
            share_memory=share_memory,
        )

        # 실행 결과 기록 (Orchestrator 통계)
        elapsed_ms = int((time.time() - start_time) * 1000)
        if self._use_orchestrator:
            self._orchestrator.record_execution_result(
                workflow_id=self.config.id,
                node_id=node_name,
                strategy=decision.strategy if self._use_orchestrator else ExecutionStrategy.CLOUD_HEAVY,
                model_id=decision.model_id if self._use_orchestrator else None,
                success=result.success,
                actual_time_ms=elapsed_ms,
                actual_cost=0.0,  # TODO: 실제 비용 계산
            )

        # Orchestrator로 결과 검증 (봇 감지 / 실패 패턴 체크)
        if result.success and self._use_orchestrator:
            # 마지막 스텝에서 thought, observation 추출
            last_thought = None
            last_observation = None
            final_answer = result.data.get("final_answer") if result.data else None

            if result.steps and len(result.steps) > 0:
                last_step = result.steps[-1]
                last_thought = last_step.thought
                last_observation = last_step.observation

            is_valid, error_type, error_message = self._orchestrator.validate_execution_result(
                workflow_id=self.config.id,
                node_id=node_name,
                result_data=result.data or {},
                final_answer=final_answer,
                last_observation=last_observation,
                last_thought=last_thought,
            )

            if not is_valid:
                log_vlm(
                    ActivityType.ERROR,
                    f"{node_name} 검증 실패: {error_message}",
                    details={
                        "error_type": error_type,
                        "error_message": error_message,
                    },
                    execution_id=current_execution_id,
                    node_id=node_name,
                    duration_ms=elapsed_ms,
                )
                return False, result.data, error_message

        # 활동 로그: VLM 실행 완료
        if result.success:
            log_vlm(
                ActivityType.EXECUTION,
                f"{node_name} 완료",
                details={
                    "success": True,
                    "steps_count": len(result.data.get("steps", [])) if result.data else 0,
                    "reused": result.data.get("reused_from_cache", False) if result.data else False,
                },
                execution_id=current_execution_id,
                node_id=node_name,
                duration_ms=elapsed_ms,
            )
            if result.data.get("reused_from_cache"):
                print(f"[CoupangWorkflow] {node_name}: trace 캐시에서 재사용됨")
            return True, result.data, None
        else:
            log_vlm(
                ActivityType.ERROR,
                f"{node_name} 실패: {result.error[:50] if result.error else 'Unknown'}",
                details={
                    "success": False,
                    "error": result.error,
                },
                execution_id=current_execution_id,
                node_id=node_name,
                duration_ms=elapsed_ms,
            )
            return False, result.data, result.error

    async def _run_vlm_with_error_handling(
        self,
        instruction: str,
        state: WorkflowState,
        node_name: str,
        max_retries: int = 3,
    ) -> tuple[bool, Dict[str, Any], Optional[str]]:
        """
        에러 핸들링이 포함된 VLM 실행

        Orchestrator가 에러 발생 시 재시도/스킵/중단을 결정합니다.
        """
        retry_count = 0
        last_error = None
        current_strategy = None

        while retry_count <= max_retries:
            try:
                success, data, error = await self._run_vlm_instruction(
                    instruction, state, node_name
                )

                if success:
                    return True, data, None

                # 실패 시 Orchestrator에게 에러 핸들링 결정 요청
                if self._use_orchestrator and error:
                    error_decision = await self._orchestrator.handle_error(
                        workflow_id=self.config.id,
                        node_id=node_name,
                        error=Exception(error),
                        current_retry=retry_count,
                        strategy=current_strategy,
                    )

                    print(f"[Orchestrator] 에러 핸들링: {error_decision.action.value} "
                          f"(retry={retry_count}/{max_retries})")

                    if error_decision.action == ErrorAction.RETRY:
                        retry_count += 1
                        last_error = error
                        print(f"[Orchestrator] {node_name} 재시도 {retry_count}/{max_retries}")
                        await asyncio.sleep(2)  # 재시도 전 대기
                        continue

                    elif error_decision.action == ErrorAction.SKIP:
                        print(f"[Orchestrator] {node_name} 스킵 (workflow 계속)")
                        return True, {"skipped": True, "reason": error}, None

                    elif error_decision.action == ErrorAction.FALLBACK:
                        # 더 강력한 모델로 재시도
                        if error_decision.fallback_strategy:
                            print(f"[Orchestrator] Fallback: {error_decision.fallback_strategy.value}")
                            retry_count += 1
                            continue

                    elif error_decision.action == ErrorAction.ABORT:
                        print(f"[Orchestrator] {node_name} 중단 (workflow 중단)")
                        return False, data, error

                    # 사용자 알림 필요 시
                    if error_decision.should_notify_user:
                        print(f"[Orchestrator] ⚠️ 사용자 확인 필요: {error}")

                # Orchestrator 없으면 기본 동작: 실패 반환
                return False, data, error

            except asyncio.TimeoutError as e:
                retry_count += 1
                last_error = f"Timeout: {str(e)}"
                print(f"[Orchestrator] {node_name} 타임아웃, 재시도 {retry_count}/{max_retries}")
                if retry_count > max_retries:
                    return False, {}, last_error
                await asyncio.sleep(2)

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                print(f"[Orchestrator] {node_name} 예외 발생: {e}")
                if retry_count > max_retries:
                    return False, {}, last_error
                await asyncio.sleep(2)

        return False, {}, last_error or "Max retries exceeded"

    async def run(self, parameters: Dict[str, Any], thread_id: str = "default") -> WorkflowState:
        """
        워크플로우 실행 (Orchestrator 추적 및 리포트 생성 포함)

        Args:
            parameters: 워크플로우 파라미터
            thread_id: 실행 스레드 ID

        Returns:
            최종 워크플로우 상태 (리포트 포함)
        """
        execution_id = f"{self.config.id}-{thread_id}"

        # Orchestrator 추적 시작
        if self._use_orchestrator:
            total_nodes = len([n for n in self.nodes
                              if n.node_type == "vlm"])  # VLM 노드만 카운트
            self._orchestrator.start_workflow_tracking(
                workflow_id=self.config.id,
                execution_id=execution_id,
                total_nodes=total_nodes,
            )

        try:
            # 기본 워크플로우 실행
            final_state = await super().run(parameters, thread_id)

            # 리포트 생성
            if self._use_orchestrator:
                try:
                    report = await self._orchestrator.generate_report(
                        execution_id=execution_id,
                        final_status=final_state.get("status", "unknown"),
                    )
                    final_state["report"] = report.to_dict()
                    print(f"\n{report.summary}")
                    if report.recommendations:
                        print("권장사항:")
                        for rec in report.recommendations:
                            print(f"  - {rec}")
                except Exception as e:
                    print(f"[Orchestrator] 리포트 생성 실패: {e}")

            return final_state

        finally:
            # 정리
            if self._use_orchestrator:
                self._orchestrator.cleanup_execution(execution_id)

    async def _update_observation_on_step(self, node_name: str, step_log):
        """스텝 실행 시 관찰 기록 업데이트"""
        if not self._letta_initialized:
            return

        try:
            # thought나 observation이 있으면 기록
            observation = step_log.thought or step_log.observation
            if observation:
                await self._letta.add_observation(
                    workflow_id=self.config.id,
                    current_page=node_name,
                    observation=observation[:200],  # 200자로 제한
                )
        except Exception as e:
            # 관찰 기록 실패는 무시 (메인 로직에 영향 없음)
            pass

    async def _open_coupang(self, state: WorkflowState) -> NodeResult:
        """쿠팡 웹사이트 열기"""
        try:
            instruction = """
            Open Chrome browser and navigate to https://www.coupang.com

            Steps:
            1. Use open_url("https://www.coupang.com") to navigate to Coupang
            2. Wait for the page to fully load
            3. Confirm you see the Coupang homepage
            """

            success, data, error = await self._run_vlm_instruction(
                instruction, state, "open_coupang"
            )

            if not success:
                return NodeResult(success=False, error=error)

            return NodeResult(
                success=True,
                data={"coupang_opened": True}
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _search_keyword(self, state: WorkflowState) -> NodeResult:
        """키워드 검색"""
        try:
            parameters = state.get("parameters", {})
            data = state.get("data", {})

            # 현재 키워드 (연관 키워드 또는 원래 키워드)
            keyword = data.get("current_keyword") or parameters.get("keyword")

            if not keyword:
                return NodeResult(success=False, error="검색 키워드가 없습니다")

            instruction = f"""
            Search for "{keyword}" on Coupang:

            Steps:
            1. Find the search input box at the top of the page
            2. Click on the search box
            3. Type "{keyword}" using write("{keyword}")
            4. Press Enter to search using press(["enter"])
            5. Wait for search results to load using wait(3)
            6. Confirm search results are displayed
            """

            success, data_result, error = await self._run_vlm_instruction(
                instruction, state, "search_keyword"
            )

            if not success:
                return NodeResult(success=False, error=error)

            return NodeResult(
                success=True,
                data={
                    "current_keyword": keyword,
                    "current_page": 1,
                    "searched_keywords": state.get("data", {}).get("searched_keywords", []) + [keyword],
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _analyze_page(self, state: WorkflowState) -> NodeResult:
        """페이지 분석 및 상품 수집"""
        try:
            data = state.get("data", {})
            keyword = data.get("current_keyword")
            current_page = data.get("current_page", 1)

            instruction = f"""
            Analyze the current Coupang search results page for "{keyword}" (Page {current_page}):

            Look at the product listings and identify products that do NOT have:
            - 로켓배송 badge (blue rocket icon)
            - 로켓직구 badge
            - 로켓와우 badge

            These are "일반배송" (regular delivery) products.

            Steps:
            1. Scroll down slowly to see all products using scroll(500, 500, "down", 2)
            2. Look for products without rocket badges
            3. For each non-rocket product, extract:
               - Product name (상품명)
               - Price (가격)
               - Product URL or ID if visible
               - Seller name if visible (판매자)
            4. Continue scrolling to see more products
            5. Return a list of found products in JSON format

            Return the products as a JSON array with keys: name, price, url, seller
            """

            success, data_result, error = await self._run_vlm_instruction(
                instruction, state, "analyze_page"
            )

            if not success:
                return NodeResult(success=False, error=error)

            # VLM이 수집한 상품 목록 (실제로는 파싱 필요)
            collected_products = data_result.get("products", [])

            return NodeResult(
                success=True,
                data={
                    "pending_products": collected_products,  # 저장 대기 상품
                    "pages_analyzed": data.get("pages_analyzed", 0) + 1,
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _save_products(self, state: WorkflowState) -> NodeResult:
        """수집한 상품을 DB에 저장"""
        try:
            data = state.get("data", {})
            parameters = state.get("parameters", {})
            keyword = data.get("current_keyword") or parameters.get("keyword")
            pending_products = data.get("pending_products", [])
            collected_count = data.get("collected_count", 0)

            saved_count = 0

            for product_data in pending_products:
                try:
                    # CoupangProduct 모델로 변환
                    product = CoupangProduct(
                        name=product_data.get("name", "Unknown"),
                        price=product_data.get("price", 0),
                        url=product_data.get("url", ""),
                        seller=product_data.get("seller", ""),
                        keyword=keyword,
                        is_rocket=False,  # 비로켓 상품만 수집
                    )

                    # DB에 저장
                    self._db.add_product(product)
                    saved_count += 1

                except Exception as e:
                    print(f"[CoupangWorkflow] 상품 저장 실패: {e}")
                    continue

            return NodeResult(
                success=True,
                data={
                    "collected_count": collected_count + saved_count,
                    "last_saved_count": saved_count,
                    "pending_products": [],  # 저장 완료 후 비우기
                }
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _check_next_page(self, state: WorkflowState) -> NodeResult:
        """다음 페이지 확인 및 이동"""
        try:
            data = state.get("data", {})
            parameters = state.get("parameters", {})

            current_page = data.get("current_page", 1)
            max_pages = parameters.get("max_pages", 5)

            # 최대 페이지 도달 확인
            if current_page >= max_pages:
                return NodeResult(
                    success=False,  # on_failure -> find_related
                    data={"max_pages_reached": True}
                )

            instruction = """
            Check if there is a next page and navigate to it:

            Steps:
            1. Scroll to the bottom of the page to find pagination
            2. Look for pagination controls (page numbers or "다음" button)
            3. If next page exists, click on the next page number or "다음" button
            4. Wait for the new page to load using wait(3)
            5. If no next page available, report "no_next_page"
            """

            success, data_result, error = await self._run_vlm_instruction(
                instruction, state, "check_next_page"
            )

            if not success or data_result.get("no_next_page"):
                return NodeResult(
                    success=False,  # on_failure -> find_related
                    data={"no_more_pages": True}
                )

            return NodeResult(
                success=True,  # on_success -> analyze_page
                data={"current_page": current_page + 1}
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _find_related(self, state: WorkflowState) -> NodeResult:
        """연관 키워드 탐색"""
        try:
            parameters = state.get("parameters", {})
            data = state.get("data", {})

            # 연관 키워드 탐색 비활성화 시
            if not parameters.get("follow_related", True):
                return NodeResult(
                    success=False,  # on_failure -> complete
                    data={"follow_related_disabled": True}
                )

            searched_keywords = data.get("searched_keywords", [])

            instruction = f"""
            Find related search keywords on Coupang:

            Steps:
            1. Look for "연관 검색어" or "관련 검색어" section on the page
            2. Or look for suggested keywords at the top or bottom of results
            3. Find a new keyword that is NOT in this list: {searched_keywords}
            4. If you find a good related keyword, click on it
            5. If no new related keywords exist, report "no_related_found"
            """

            success, data_result, error = await self._run_vlm_instruction(
                instruction, state, "find_related"
            )

            if not success or data_result.get("no_related_found"):
                return NodeResult(
                    success=False,  # on_failure -> complete
                    data={"no_related_keywords": True}
                )

            new_keyword = data_result.get("related_keyword")
            if new_keyword and new_keyword not in searched_keywords:
                return NodeResult(
                    success=True,  # on_success -> search_keyword
                    data={
                        "current_keyword": new_keyword,
                        "current_page": 1,
                    }
                )

            return NodeResult(
                success=False,  # on_failure -> complete
                data={"no_related_keywords": True}
            )
        except Exception as e:
            return NodeResult(success=False, error=str(e))

    async def _complete(self, state: WorkflowState) -> NodeResult:
        """수집 완료"""
        data = state.get("data", {})

        return NodeResult(
            success=True,
            data={
                "final_collected_count": data.get("collected_count", 0),
                "keywords_searched": data.get("searched_keywords", []),
                "pages_analyzed": data.get("pages_analyzed", 0),
                "completed": True,
            }
        )

    async def _error_handler(self, state: WorkflowState) -> NodeResult:
        """에러 처리 및 복구"""
        error = state.get("error", "Unknown error")
        print(f"[CoupangWorkflow] Error handled: {error}")

        # 복구 시도: 에러를 기록하고 완료 단계로 이동하여 부분 성공 처리 (Graceful Shutdown)
        return NodeResult(
            success=True,
            data={"error_handled": True, "original_error": error},
            next_node="complete"
        )
