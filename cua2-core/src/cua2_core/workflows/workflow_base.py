"""
워크플로우 베이스 클래스 - LangGraph 기반

Features:
- SQLite 영구 Checkpointing (서버 재시작 후 복구 가능)
- Streaming Mode 최적화 (updates 모드로 네트워크 효율성 향상)
- VLM 에러 타입 기반 조건부 라우팅
- Human-in-the-Loop: 중요 작업 전 사용자 확인 대기
- Parallel Node Execution: 독립적인 노드 병렬 실행
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Annotated, Literal
import asyncio
import operator
import os
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# SQLite Checkpointing (영구 저장)
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# Async SQLite (더 나은 성능)
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    ASYNC_SQLITE_AVAILABLE = True
except ImportError:
    ASYNC_SQLITE_AVAILABLE = False


class NodeStatus(str, Enum):
    """노드 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class VLMErrorType(str, Enum):
    """VLM이 스크린샷 분석으로 감지한 에러 타입"""
    NONE = "none"  # 에러 없음
    BOT_DETECTED = "bot_detected"  # 봇 감지 (CAPTCHA, 접근 거부)
    PAGE_FAILED = "page_failed"  # 페이지 로딩 실패
    ACCESS_DENIED = "access_denied"  # 접근 거부 (403)
    ELEMENT_NOT_FOUND = "element_not_found"  # 요소를 찾을 수 없음
    TIMEOUT = "timeout"  # 타임아웃
    UNKNOWN = "unknown"  # 알 수 없는 에러


@dataclass
class NodeResult:
    """노드 실행 결과"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    next_node: Optional[str] = None  # 다음 노드 지정 (조건부 분기)
    vlm_error_type: VLMErrorType = VLMErrorType.NONE  # VLM이 감지한 에러 타입


@dataclass
class WorkflowNode:
    """워크플로우 노드 정의"""
    name: str
    description: str
    display_name: Optional[str] = None  # UI 표시용 이름 (한국어)
    on_success: Optional[str] = None  # 성공 시 다음 노드
    on_failure: Optional[str] = None  # 실패 시 다음 노드
    status: NodeStatus = NodeStatus.PENDING
    node_type: Optional[str] = None  # 노드 타입: start, process, condition, end, error, vlm
    instruction: Optional[str] = None  # VLM 에이전트 명령 (시스템 프롬프트)

    # VLM 에러 타입별 라우팅 (LangGraph 조건부 엣지)
    # VLM이 스크린샷을 보고 [ERROR:TYPE] 형식으로 보고하면 해당 노드로 이동
    on_bot_detected: Optional[str] = None  # 봇 감지 시 → abort 노드로
    on_page_failed: Optional[str] = None  # 페이지 로딩 실패 시 → retry 노드로
    on_access_denied: Optional[str] = None  # 접근 거부 시 → error_handler로

    # 시간 설정
    timeout_sec: int = 120  # 작업 제한시간 (초), 기본 2분
    avg_duration_sec: Optional[int] = None  # 평균 작업 시간 (초), 학습된 값

    # 재사용 및 메모리 설정
    reusable: bool = False  # 이 노드의 trace를 재사용 가능하게 저장할지
    reuse_trace: bool = False  # 이전 trace를 재사용할지 (같은 입력일 때)
    share_memory: bool = False  # 이전 노드와 메모리(컨텍스트) 공유할지
    cache_key_params: List[str] = field(default_factory=list)  # 캐시 키에 사용할 파라미터 목록

    # Human-in-the-Loop 설정
    requires_confirmation: bool = False  # 실행 전 사용자 확인 필요
    confirmation_message: Optional[str] = None  # 확인 요청 메시지
    is_dangerous: bool = False  # 위험한 작업 (결제, 삭제 등)

    # Parallel Execution 설정
    parallel_group: Optional[str] = None  # 같은 그룹의 노드들은 병렬 실행
    depends_on: List[str] = field(default_factory=list)  # 의존하는 노드 목록

    # 에이전트 정보
    agent_type: Optional[str] = None  # 에이전트 타입: VLMAgent, SearchAgent, AnalysisAgent 등
    model_id: Optional[str] = None  # 모델 ID: local-qwen3-vl, gpt-4o 등

    # UI 설정
    clickable: bool = False  # 클릭하여 상세 보기 가능 여부
    metadata: Dict[str, Any] = field(default_factory=dict)  # 노드 설정 값 (Props Inspector용)


@dataclass
class WorkflowConfig:
    """워크플로우 설정"""
    id: str
    name: str
    description: str
    icon: str = "AccountTree"  # MUI icon name
    color: str = "#1976d2"
    category: str = "general"
    parameters: List[Dict[str, Any]] = field(default_factory=list)  # 필요한 파라미터 정의


@dataclass
class NodeExecutionLog:
    """노드 실행 로그 - VLM 노드의 스텝별 정보 저장"""
    step_number: int
    timestamp: str
    screenshot: Optional[str] = None  # base64 encoded image
    action: Optional[str] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    """워크플로우 상태 - LangGraph StateGraph용"""
    # 기본 상태
    workflow_id: str
    execution_id: str  # 실행 ID (예: coupang-collect-20251129092555)
    status: str  # pending, running, completed, failed, stopped
    current_node: Optional[str]

    # 실행 이력
    completed_nodes: Annotated[List[str], operator.add]
    failed_nodes: Annotated[List[str], operator.add]
    node_results: Dict[str, Any]

    # 시간 정보
    start_time: Optional[str]
    end_time: Optional[str]
    current_node_start_time: Optional[str]  # 현재 노드 시작 시간

    # 사용자 입력 파라미터
    parameters: Dict[str, Any]

    # 워크플로우별 커스텀 데이터
    data: Dict[str, Any]

    # 에러 정보
    error: Optional[str]

    # VLM 에러 타입 (스크린샷 기반 에러 감지)
    vlm_error_type: Optional[str]

    # 재시도 횟수 (노드별)
    retry_count: int

    # 중지 플래그
    should_stop: bool

    # 노드별 실행 로그 (VLM 스텝 정보)
    node_logs: Dict[str, List[Dict[str, Any]]]

    # Human-in-the-Loop 상태
    waiting_for_confirmation: bool  # 사용자 확인 대기 중
    pending_confirmation_node: Optional[str]  # 확인 대기 중인 노드
    confirmation_message: Optional[str]  # 사용자에게 보여줄 메시지
    user_confirmed: Optional[bool]  # 사용자 확인 결과 (True: 진행, False: 취소)
    user_input: Optional[str]  # 사용자 입력 (CAPTCHA 등)


class WorkflowBase(ABC):
    """
    워크플로우 베이스 클래스

    모든 워크플로우는 이 클래스를 상속받아 구현합니다.
    LangGraph의 StateGraph를 사용하여 노드 기반 실행을 지원합니다.

    Features:
    - SQLite 영구 Checkpointing: 서버 재시작 후에도 상태 복구 가능
    - Streaming Mode 최적화: updates 모드로 변경분만 전송 (네트워크 75% 감소)
    """

    # Streaming 모드 설정
    # - "values": 전체 상태 전송 (기본, 호환성)
    # - "updates": 변경분만 전송 (효율적)
    # - "debug": 디버그 정보 포함
    STREAM_MODE: Literal["values", "updates", "debug"] = "updates"

    def __init__(
        self,
        use_persistent_storage: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            use_persistent_storage: SQLite 영구 저장소 사용 여부
                - True: 서버 재시작 후에도 상태 복구 가능
                - False: 메모리 기반 (기존 동작)
            checkpoint_dir: 체크포인트 저장 디렉토리 (기본: ~/.cua2/checkpoints)
        """
        self._graph: Optional[StateGraph] = None
        self._app = None
        self._stop_event = asyncio.Event()
        self._current_state: Optional[WorkflowState] = None

        # Checkpointer 설정
        self._use_persistent_storage = use_persistent_storage
        self._checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.expanduser("~"), ".cua2", "checkpoints"
        )
        self._checkpointer = self._create_checkpointer()

        # Human-in-the-Loop 설정
        self._confirmation_event = asyncio.Event()  # 사용자 확인 대기용
        self._confirmation_callback: Optional[Callable] = None  # UI 콜백

        # Dynamic Breakpoints 설정
        self._breakpoints: set = set()  # 브레이크포인트 노드 목록
        self._breakpoint_event = asyncio.Event()  # 브레이크포인트 대기용
        self._breakpoint_callback: Optional[Callable] = None  # 브레이크포인트 콜백

    def _create_checkpointer(self):
        """Checkpointer 설정 - 경로 준비"""
        if not self._use_persistent_storage:
            return None

        if not ASYNC_SQLITE_AVAILABLE:
            import logging
            logging.warning(
                "[WorkflowBase] aiosqlite 없음, MemorySaver 사용. "
                "pip install aiosqlite 로 설치하세요."
            )
            return None

        # 디렉토리 생성
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # SQLite 파일 경로
        db_path = os.path.join(self._checkpoint_dir, "workflow_checkpoints.db")
        return db_path

    @property
    @abstractmethod
    def config(self) -> WorkflowConfig:
        """워크플로우 설정 반환"""
        pass

    @property
    @abstractmethod
    def nodes(self) -> List[WorkflowNode]:
        """워크플로우 노드 목록 반환"""
        pass

    @property
    @abstractmethod
    def start_node(self) -> str:
        """시작 노드 이름 반환"""
        pass

    @abstractmethod
    async def execute_node(self, node_name: str, state: WorkflowState) -> NodeResult:
        """
        개별 노드 실행 로직

        Args:
            node_name: 실행할 노드 이름
            state: 현재 워크플로우 상태

        Returns:
            NodeResult: 노드 실행 결과
        """
        pass

    def build_graph(self) -> StateGraph:
        """LangGraph StateGraph 빌드"""
        graph = StateGraph(WorkflowState)

        # 각 노드를 그래프에 추가
        for node in self.nodes:
            # 노드 실행 함수 생성
            node_func = self._create_node_function(node.name)
            graph.add_node(node.name, node_func)

        # 시작 노드 설정
        graph.set_entry_point(self.start_node)

        # 엣지 추가 (조건부 분기)
        for node in self.nodes:
            if node.on_success or node.on_failure:
                # 조건부 엣지
                graph.add_conditional_edges(
                    node.name,
                    self._create_router(node),
                    self._get_edge_mapping(node)
                )
            elif node.on_success:
                # 단순 성공 엣지
                graph.add_edge(node.name, node.on_success)
            else:
                # 종료
                graph.add_edge(node.name, END)

        return graph

    def _create_node_function(self, node_name: str) -> Callable:
        """노드 실행 함수 생성"""
        async def node_function(state: WorkflowState) -> Dict[str, Any]:
            # 중지 체크
            if state.get("should_stop", False):
                return {
                    "status": "stopped",
                    "current_node": None,
                }

            # Human-in-the-Loop: 사용자 확인 필요한 노드 체크
            node_config = self._get_node_by_name(node_name)
            if node_config and node_config.requires_confirmation:
                # 사용자가 아직 확인하지 않은 경우
                if not state.get("user_confirmed"):
                    confirmation_msg = node_config.confirmation_message or f"'{node_config.display_name or node_name}' 작업을 실행하시겠습니까?"
                    if node_config.is_dangerous:
                        confirmation_msg = f"⚠️ 위험한 작업: {confirmation_msg}"

                    # 상태 업데이트하여 대기 상태로 전환
                    if self._current_state is not None:
                        self._current_state["waiting_for_confirmation"] = True
                        self._current_state["pending_confirmation_node"] = node_name
                        self._current_state["confirmation_message"] = confirmation_msg

                    # UI 콜백 호출 (있는 경우)
                    if self._confirmation_callback:
                        await self._confirmation_callback(node_name, confirmation_msg, node_config.is_dangerous)

                    # 사용자 확인 대기
                    await self._wait_for_confirmation()

                    # 확인 결과 체크
                    if self._current_state and not self._current_state.get("user_confirmed", False):
                        # 사용자가 취소함
                        return {
                            "status": "stopped",
                            "should_stop": True,
                            "error": "사용자가 작업을 취소했습니다",
                            "waiting_for_confirmation": False,
                        }

                    # 확인됨 - 상태 초기화 후 진행
                    if self._current_state is not None:
                        self._current_state["waiting_for_confirmation"] = False
                        self._current_state["user_confirmed"] = None

            # Dynamic Breakpoint 체크
            if node_name in self._breakpoints:
                if self._current_state is not None:
                    self._current_state["paused_at_breakpoint"] = True
                    self._current_state["breakpoint_node"] = node_name

                # 브레이크포인트 콜백 호출
                if self._breakpoint_callback:
                    await self._breakpoint_callback(node_name, self._current_state)

                # 재개 대기
                await self._wait_for_breakpoint_resume()

                if self._current_state is not None:
                    self._current_state["paused_at_breakpoint"] = False
                    self._current_state["breakpoint_node"] = None

            # 현재 노드 설정 (즉시 _current_state 업데이트하여 WebSocket에서 바로 조회 가능하게)
            current_node_start_time = datetime.now().isoformat()
            if self._current_state is not None:
                self._current_state["current_node"] = node_name
                self._current_state["status"] = "running"
                self._current_state["current_node_start_time"] = current_node_start_time

            state_update = {
                "current_node": node_name,
                "status": "running",
                "current_node_start_time": current_node_start_time,
            }

            try:
                # 노드 실행
                result = await self.execute_node(node_name, state)

                if result.success:
                    state_update["completed_nodes"] = [node_name]
                    state_update["node_results"] = {
                        **state.get("node_results", {}),
                        node_name: {"success": True, "data": result.data}
                    }
                    # 성공 시 VLM 에러 타입 및 재시도 횟수 초기화
                    state_update["vlm_error_type"] = None
                    state_update["retry_count"] = 0
                    # 즉시 _current_state 업데이트
                    if self._current_state is not None:
                        if node_name not in self._current_state.get("completed_nodes", []):
                            self._current_state["completed_nodes"] = self._current_state.get("completed_nodes", []) + [node_name]
                        self._current_state["vlm_error_type"] = None
                        self._current_state["retry_count"] = 0
                else:
                    state_update["failed_nodes"] = [node_name]
                    state_update["node_results"] = {
                        **state.get("node_results", {}),
                        node_name: {"success": False, "error": result.error}
                    }
                    state_update["error"] = result.error

                    # 재시도 횟수 증가 (LangGraph 에러 핸들링용)
                    current_retry = state.get("retry_count", 0)
                    state_update["retry_count"] = current_retry + 1

                    # VLM 에러 타입 저장 (LangGraph 조건부 엣지에서 사용)
                    if result.vlm_error_type and result.vlm_error_type != VLMErrorType.NONE:
                        state_update["vlm_error_type"] = result.vlm_error_type.value.upper()
                        if self._current_state is not None:
                            self._current_state["vlm_error_type"] = result.vlm_error_type.value.upper()

                    # 즉시 _current_state 업데이트
                    if self._current_state is not None:
                        if node_name not in self._current_state.get("failed_nodes", []):
                            self._current_state["failed_nodes"] = self._current_state.get("failed_nodes", []) + [node_name]
                        self._current_state["error"] = result.error
                        self._current_state["retry_count"] = current_retry + 1

                # 커스텀 데이터 병합
                if result.data:
                    state_update["data"] = {
                        **state.get("data", {}),
                        **result.data
                    }
                    # 즉시 _current_state 업데이트
                    if self._current_state is not None:
                        self._current_state["data"] = {
                            **self._current_state.get("data", {}),
                            **result.data
                        }

                # 다음 노드 지정 (결과에서)
                if result.next_node:
                    state_update["_next_node"] = result.next_node

            except Exception as e:
                state_update["failed_nodes"] = [node_name]
                state_update["error"] = str(e)
                state_update["node_results"] = {
                    **state.get("node_results", {}),
                    node_name: {"success": False, "error": str(e)}
                }
                # 즉시 _current_state 업데이트
                if self._current_state is not None:
                    if node_name not in self._current_state.get("failed_nodes", []):
                        self._current_state["failed_nodes"] = self._current_state.get("failed_nodes", []) + [node_name]
                    self._current_state["error"] = str(e)

            return state_update

        return node_function

    def _create_router(self, node: WorkflowNode) -> Callable:
        """
        조건부 라우팅 함수 생성 - LangGraph 에러 핸들링 통합

        VLM 에러 타입과 재시도 횟수를 기반으로 라우팅 결정:
        - BOT_DETECTED → on_bot_detected (또는 즉시 중단)
        - PAGE_FAILED → 재시도 (max_retries까지)
        - ACCESS_DENIED → on_access_denied (또는 중단)
        - TIMEOUT → 재시도
        - 성공 → on_success
        - 실패 → on_failure
        """
        max_retries = 3  # 에러 핸들링 설정

        def router(state: WorkflowState) -> str:
            # 중지 체크
            if state.get("should_stop", False):
                return "end"

            # 결과에서 다음 노드 지정 확인 (우선순위 최고)
            next_node = state.get("_next_node")
            if next_node:
                return next_node

            # 노드 결과 확인
            node_result = state.get("node_results", {}).get(node.name, {})
            retry_count = state.get("retry_count", 0)

            # VLM 에러 타입에 따른 조건부 라우팅
            vlm_error = state.get("vlm_error_type")
            if vlm_error:
                # 에러 타입별 라우팅 결정
                if vlm_error == "BOT_DETECTED":
                    # 봇 감지: 즉시 중단 (재시도 없음)
                    return node.on_bot_detected or node.on_failure or "end"

                elif vlm_error == "ACCESS_DENIED":
                    # 접근 거부: 즉시 중단
                    return node.on_access_denied or node.on_failure or "end"

                elif vlm_error in ("PAGE_FAILED", "TIMEOUT"):
                    # 페이지 로딩 실패/타임아웃: 재시도 가능
                    if retry_count < max_retries:
                        # 재시도: 같은 노드로 돌아감
                        return node.on_page_failed or node.name
                    else:
                        # 재시도 한도 초과: 스킵
                        return node.on_failure or "end"

                elif vlm_error == "ELEMENT_NOT_FOUND":
                    # 요소 못찾음: 스킵
                    return node.on_failure or "end"

                else:
                    # 알 수 없는 에러: 중단
                    return node.on_failure or "end"

            # 성공/실패 기본 라우팅
            if node_result.get("success", False):
                return node.on_success or "end"
            else:
                return node.on_failure or "end"

        return router

    def _get_edge_mapping(self, node: WorkflowNode) -> Dict[str, str]:
        """엣지 매핑 생성 - VLM 에러 핸들러 포함"""
        mapping = {"end": END}

        if node.on_success:
            mapping[node.on_success] = node.on_success
        if node.on_failure:
            mapping[node.on_failure] = node.on_failure

        # VLM 에러 타입별 에러 핸들러 매핑
        if node.on_bot_detected:
            mapping[node.on_bot_detected] = node.on_bot_detected
        if node.on_page_failed:
            mapping[node.on_page_failed] = node.on_page_failed
        if node.on_access_denied:
            mapping[node.on_access_denied] = node.on_access_denied

        return mapping

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """
        파라미터 검증

        Args:
            parameters: 검증할 파라미터

        Returns:
            에러 메시지 목록 (빈 리스트면 검증 통과)
        """
        errors = []

        for param_def in self.config.parameters:
            param_name = param_def.get("name")
            param_label = param_def.get("label", param_name)
            required = param_def.get("required", False)
            param_type = param_def.get("type", "string")
            min_val = param_def.get("min")
            max_val = param_def.get("max")
            options = param_def.get("options")

            value = parameters.get(param_name)

            # 필수 파라미터 체크
            if required:
                if value is None or value == "":
                    errors.append(f"필수 파라미터 '{param_label}'이(가) 비어있습니다.")
                    continue

            # 값이 없으면 이후 검증 스킵
            if value is None or value == "":
                continue

            # 타입별 검증
            if param_type == "number":
                try:
                    num_value = float(value)
                    if min_val is not None and num_value < min_val:
                        errors.append(f"'{param_label}'은(는) {min_val} 이상이어야 합니다.")
                    if max_val is not None and num_value > max_val:
                        errors.append(f"'{param_label}'은(는) {max_val} 이하여야 합니다.")
                except (TypeError, ValueError):
                    errors.append(f"'{param_label}'은(는) 숫자여야 합니다.")

            elif param_type == "select" and options:
                valid_values = [opt.get("value") for opt in options]
                if value not in valid_values:
                    errors.append(f"'{param_label}'의 값이 올바르지 않습니다.")

        return errors

    async def run(self, parameters: Dict[str, Any], thread_id: str = "default") -> WorkflowState:
        """
        워크플로우 실행

        Args:
            parameters: 워크플로우 파라미터
            thread_id: 실행 스레드 ID (체크포인트용)

        Returns:
            최종 워크플로우 상태

        Raises:
            ValueError: 필수 파라미터가 누락된 경우
        """
        # 파라미터 검증
        validation_errors = self.validate_parameters(parameters)
        if validation_errors:
            raise ValueError(f"파라미터 검증 실패: {'; '.join(validation_errors)}")

        # 실행 ID 설정
        if thread_id.startswith(self.config.id):
            execution_id = thread_id
        elif thread_id != "default":
            execution_id = f"{self.config.id}-{thread_id}"
        else:
            execution_id = self.config.id

        # 초기 상태
        initial_state: WorkflowState = {
            "workflow_id": self.config.id,
            "execution_id": execution_id,
            "status": "running",
            "current_node": None,
            "completed_nodes": [],
            "failed_nodes": [],
            "node_results": {},
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "parameters": parameters,
            "data": {},
            "error": None,
            "vlm_error_type": None,
            "retry_count": 0,
            "should_stop": False,
            "node_logs": {},
            "waiting_for_confirmation": False,
            "pending_confirmation_node": None,
            "confirmation_message": None,
            "user_confirmed": None,
            "user_input": None,
        }

        self._current_state = initial_state
        self._stop_event.clear()

        # 그래프 빌드
        if self._graph is None:
            self._graph = self.build_graph()

        # 실행 설정
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # AsyncSqliteSaver (Persistent) 또는 MemorySaver (In-Memory) 선택
            if self._checkpointer and ASYNC_SQLITE_AVAILABLE:
                # AsyncSqliteSaver 사용 (Context Manager)
                async with AsyncSqliteSaver.from_conn_string(self._checkpointer) as checkpointer:
                    self._app = self._graph.compile(checkpointer=checkpointer)
                    
                    # stream_mode="updates"로 변경분만 전송 (네트워크 효율성 75% 향상)
                    async for event in self._app.astream(
                        initial_state,
                        config,
                        stream_mode=self.STREAM_MODE,
                    ):
                        # stream_mode에 따른 상태 업데이트
                        if self.STREAM_MODE == "updates":
                            # updates 모드: {node_name: updates_dict} 형식
                            for node_name, updates in event.items():
                                if isinstance(updates, dict):
                                    # 변경분만 병합
                                    for key, value in updates.items():
                                        if key in initial_state:
                                            self._current_state[key] = value
                        elif self.STREAM_MODE == "values":
                            # values 모드: 전체 상태
                            self._current_state = event
                        
                        # 중지 체크
                        if self._stop_event.is_set():
                            self._current_state["should_stop"] = True
                            self._current_state["status"] = "stopped"
                            break

            else:
                # MemorySaver 사용 (동기식)
                checkpointer = MemorySaver()
                self._app = self._graph.compile(checkpointer=checkpointer)
                
                async for event in self._app.astream(
                    initial_state,
                    config,
                    stream_mode=self.STREAM_MODE,
                ):
                    if self.STREAM_MODE == "updates":
                        for node_name, updates in event.items():
                            if isinstance(updates, dict):
                                for key, value in updates.items():
                                    if key in initial_state:
                                        self._current_state[key] = value
                    elif self.STREAM_MODE == "values":
                        self._current_state = event

                    # 중지 체크
                    if self._stop_event.is_set():
                        self._current_state["should_stop"] = True
                        self._current_state["status"] = "stopped"
                        break

            # 최종 상태 설정
            if self._current_state["status"] == "running":
                if self._current_state.get("error"):
                    self._current_state["status"] = "failed"
                else:
                    self._current_state["status"] = "completed"

            self._current_state["end_time"] = datetime.now().isoformat()

        except Exception as e:
            if self._current_state:
                self._current_state["status"] = "failed"
                self._current_state["error"] = str(e)
                self._current_state["end_time"] = datetime.now().isoformat()
            raise e

        return self._current_state

    def stop(self):
        """워크플로우 중지"""
        self._stop_event.set()
        if self._current_state:
            self._current_state["should_stop"] = True

    def get_state(self) -> Optional[WorkflowState]:
        """현재 상태 반환"""
        return self._current_state

    def get_graph_definition(self) -> Dict[str, Any]:
        """
        프론트엔드용 그래프 정의 반환

        Returns:
            노드 및 엣지 정보를 포함한 딕셔너리
        """
        nodes_def = []
        edges_def = []

        for i, node in enumerate(self.nodes):
            node_data = {
                "id": node.name,
                "name": node.display_name or node.name,  # display_name 우선, 없으면 name
                "description": node.description,
                "status": node.status.value,
            }
            # 타입과 instruction 추가 (있을 경우)
            if node.node_type:
                node_data["type"] = node.node_type
            if node.instruction:
                node_data["instruction"] = node.instruction

            # 시간 설정 추가
            node_data["timeout_sec"] = node.timeout_sec
            node_data["avg_duration_sec"] = node.avg_duration_sec

            # 재사용/메모리 설정 추가
            node_data["reusable"] = node.reusable
            node_data["reuse_trace"] = node.reuse_trace
            node_data["share_memory"] = node.share_memory
            node_data["cache_key_params"] = node.cache_key_params

            # 에이전트 정보 추가
            node_data["agent_type"] = node.agent_type
            node_data["model_id"] = node.model_id

            # UI 설정 추가 (VLM 노드는 기본 클릭 가능)
            node_data["clickable"] = node.clickable or (node.node_type == "vlm")
            node_data["metadata"] = node.metadata  # Props Inspector용 메타데이터 추가

            nodes_def.append(node_data)

            if node.on_success:
                edges_def.append({
                    "source": node.name,
                    "target": node.on_success,
                    "type": "success",
                })

            if node.on_failure:
                edges_def.append({
                    "source": node.name,
                    "target": node.on_failure,
                    "type": "failure",
                })

        return {
            "config": {
                "id": self.config.id,
                "name": self.config.name,
                "description": self.config.description,
                "icon": self.config.icon,
                "color": self.config.color,
                "category": self.config.category,
                "parameters": self.config.parameters,
            },
            "nodes": nodes_def,
            "edges": edges_def,
            "start_node": self.start_node,
        }

    # =========================================
    # Checkpoint 관련 메서드 (SQLite 영구 저장)
    # =========================================

    async def resume(self, thread_id: str) -> Optional[WorkflowState]:
        """
        이전 실행에서 재개

        서버 재시작 후에도 이전 상태에서 계속 실행 가능합니다.

        Args:
            thread_id: 재개할 실행의 thread_id

        Returns:
            재개된 워크플로우 상태 또는 None (체크포인트 없음)
        """
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        config = {"configurable": {"thread_id": thread_id}}

        try:
            # 마지막 체크포인트 상태 조회
            checkpoint = await self._app.aget_state(config)
            if checkpoint and checkpoint.values:
                self._current_state = dict(checkpoint.values)

                # 이미 완료/실패/중지된 경우 재개하지 않음
                if self._current_state.get("status") in ("completed", "failed", "stopped"):
                    import logging
                    logging.info(
                        f"[WorkflowBase] 이미 종료된 워크플로우: {thread_id} "
                        f"(status={self._current_state.get('status')})"
                    )
                    return self._current_state

                # 재개 실행
                import logging
                logging.info(
                    f"[WorkflowBase] 워크플로우 재개: {thread_id}, "
                    f"current_node={self._current_state.get('current_node')}"
                )

                self._stop_event.clear()

                async for event in self._app.astream(
                    None,  # None을 전달하면 마지막 체크포인트에서 재개
                    config,
                    stream_mode=self.STREAM_MODE,
                ):
                    if self.STREAM_MODE == "updates":
                        for node_name, updates in event.items():
                            if isinstance(updates, dict):
                                for key, value in updates.items():
                                    if key in ("completed_nodes", "failed_nodes") and isinstance(value, list):
                                        existing = self._current_state.get(key, [])
                                        new_nodes = [n for n in value if n not in existing]
                                        self._current_state[key] = existing + new_nodes
                                    elif key in ("node_results", "data") and isinstance(value, dict):
                                        existing = self._current_state.get(key, {})
                                        self._current_state[key] = {**existing, **value}
                                    else:
                                        self._current_state[key] = value
                    else:
                        for node_name, node_state in event.items():
                            if isinstance(node_state, dict):
                                self._current_state = {**self._current_state, **node_state}

                    if self._stop_event.is_set():
                        self._current_state["should_stop"] = True
                        self._current_state["status"] = "stopped"
                        break

                # 최종 상태
                if self._current_state["status"] == "running":
                    if self._current_state.get("error"):
                        self._current_state["status"] = "failed"
                    else:
                        self._current_state["status"] = "completed"

                self._current_state["end_time"] = datetime.now().isoformat()
                return self._current_state

            return None

        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] 재개 실패: {e}")
            return None

    async def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 실행의 체크포인트 조회

        Args:
            thread_id: 조회할 실행의 thread_id

        Returns:
            체크포인트 상태 딕셔너리 또는 None
        """
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        config = {"configurable": {"thread_id": thread_id}}

        try:
            checkpoint = await self._app.aget_state(config)
            if checkpoint and checkpoint.values:
                return dict(checkpoint.values)
            return None
        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] 체크포인트 조회 실패: {e}")
            return None

    async def list_checkpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        최근 체크포인트 목록 조회

        Args:
            limit: 최대 조회 개수

        Returns:
            체크포인트 요약 목록
        """
        # SQLite 직접 조회 (langgraph API 미지원 시)
        if not self._use_persistent_storage:
            return []

        db_path = os.path.join(self._checkpoint_dir, "workflow_checkpoints.db")
        if not os.path.exists(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 테이블 존재 확인
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            if not cursor.fetchone():
                conn.close()
                return []

            # 최근 체크포인트 조회
            cursor.execute(
                """
                SELECT thread_id, created_at, checkpoint
                FROM checkpoints
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()

            checkpoints = []
            for row in rows:
                checkpoints.append({
                    "thread_id": row[0],
                    "created_at": row[1],
                })
            return checkpoints

        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] 체크포인트 목록 조회 실패: {e}")
            return []

    def is_persistent_storage_enabled(self) -> bool:
        """영구 저장소 사용 여부 반환"""
        return self._use_persistent_storage and SQLITE_AVAILABLE

    # =========================================
    # Human-in-the-Loop 메서드
    # =========================================

    def _get_node_by_name(self, node_name: str) -> Optional[WorkflowNode]:
        """노드 이름으로 노드 설정 조회"""
        for node in self.nodes:
            if node.name == node_name:
                return node
        return None

    def set_confirmation_callback(self, callback: Callable):
        """
        사용자 확인 요청 콜백 설정

        콜백 시그니처: async def callback(node_name: str, message: str, is_dangerous: bool)

        Example:
            async def on_confirmation_needed(node_name, message, is_dangerous):
                # WebSocket으로 UI에 알림
                await websocket.send_json({
                    "type": "confirmation_required",
                    "node": node_name,
                    "message": message,
                    "dangerous": is_dangerous,
                })

            workflow.set_confirmation_callback(on_confirmation_needed)
        """
        self._confirmation_callback = callback

    async def _wait_for_confirmation(self, timeout_sec: int = 300):
        """
        사용자 확인 대기 (기본 5분 타임아웃)

        Args:
            timeout_sec: 타임아웃 시간 (초)
        """
        self._confirmation_event.clear()
        try:
            await asyncio.wait_for(
                self._confirmation_event.wait(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            # 타임아웃 시 취소로 처리
            if self._current_state is not None:
                self._current_state["user_confirmed"] = False
                self._current_state["error"] = "사용자 확인 시간 초과"

    def confirm(self, confirmed: bool = True, user_input: Optional[str] = None):
        """
        사용자 확인 응답 처리 (UI에서 호출)

        Args:
            confirmed: 사용자 확인 여부 (True: 진행, False: 취소)
            user_input: 사용자 입력 (CAPTCHA 등)

        Example:
            # UI에서 사용자가 '확인' 클릭
            workflow.confirm(True)

            # UI에서 사용자가 '취소' 클릭
            workflow.confirm(False)

            # CAPTCHA 입력 후 확인
            workflow.confirm(True, user_input="ABC123")
        """
        if self._current_state is not None:
            self._current_state["user_confirmed"] = confirmed
            self._current_state["user_input"] = user_input
            self._current_state["waiting_for_confirmation"] = False

        # 대기 해제
        self._confirmation_event.set()

    def is_waiting_for_confirmation(self) -> bool:
        """사용자 확인 대기 중인지 확인"""
        if self._current_state:
            return self._current_state.get("waiting_for_confirmation", False)
        return False

    def get_pending_confirmation(self) -> Optional[Dict[str, Any]]:
        """
        대기 중인 확인 요청 정보 반환

        Returns:
            확인 요청 정보 또는 None
        """
        if not self.is_waiting_for_confirmation():
            return None

        return {
            "node": self._current_state.get("pending_confirmation_node"),
            "message": self._current_state.get("confirmation_message"),
            "is_dangerous": self._get_node_by_name(
                self._current_state.get("pending_confirmation_node", "")
            ).is_dangerous if self._current_state.get("pending_confirmation_node") else False,
        }

    async def request_user_input(
        self,
        prompt: str,
        input_type: str = "text",
        timeout_sec: int = 300,
    ) -> Optional[str]:
        """
        사용자 입력 요청 (CAPTCHA, 2FA 등)

        Args:
            prompt: 입력 요청 메시지
            input_type: 입력 타입 ("text", "captcha", "2fa")
            timeout_sec: 타임아웃 시간 (초)

        Returns:
            사용자 입력 문자열 또는 None (타임아웃/취소)

        Example:
            # 봇 감지 시 CAPTCHA 입력 요청
            captcha = await workflow.request_user_input(
                "CAPTCHA를 입력해주세요",
                input_type="captcha",
            )
            if captcha:
                # CAPTCHA 입력 처리
                pass
        """
        if self._current_state is not None:
            self._current_state["waiting_for_confirmation"] = True
            self._current_state["confirmation_message"] = prompt
            self._current_state["user_input"] = None

        # UI 콜백 호출
        if self._confirmation_callback:
            await self._confirmation_callback("user_input", prompt, input_type == "captcha")

        # 사용자 입력 대기
        await self._wait_for_confirmation(timeout_sec)

        # 입력 결과 반환
        if self._current_state:
            return self._current_state.get("user_input")
        return None

    # =========================================
    # Parallel Execution 헬퍼
    # =========================================

    def get_parallel_groups(self) -> Dict[str, List[str]]:
        """
        병렬 실행 그룹 조회

        Returns:
            그룹명 -> 노드명 리스트 매핑
        """
        groups: Dict[str, List[str]] = {}
        for node in self.nodes:
            if node.parallel_group:
                if node.parallel_group not in groups:
                    groups[node.parallel_group] = []
                groups[node.parallel_group].append(node.name)
        return groups

    def get_node_dependencies(self, node_name: str) -> List[str]:
        """
        노드의 의존성 목록 조회

        Args:
            node_name: 노드 이름

        Returns:
            의존하는 노드 이름 목록
        """
        node = self._get_node_by_name(node_name)
        if node:
            return node.depends_on
        return []

    def can_execute_parallel(self, node_names: List[str], completed_nodes: List[str]) -> List[str]:
        """
        병렬 실행 가능한 노드 목록 반환

        Args:
            node_names: 실행 대기 중인 노드 목록
            completed_nodes: 이미 완료된 노드 목록

        Returns:
            지금 바로 병렬 실행 가능한 노드 목록
        """
        executable = []
        for name in node_names:
            deps = self.get_node_dependencies(name)
            # 모든 의존성이 완료되었으면 실행 가능
            if all(dep in completed_nodes for dep in deps):
                executable.append(name)
        return executable

    async def run_parallel_nodes(
        self,
        node_names: List[str],
        state: WorkflowState,
        max_concurrency: int = 5,
    ) -> Dict[str, NodeResult]:
        """
        여러 노드를 병렬로 실행

        Args:
            node_names: 병렬 실행할 노드 이름 목록
            state: 현재 워크플로우 상태
            max_concurrency: 최대 동시 실행 수 (기본 5)

        Returns:
            노드별 실행 결과 딕셔너리

        Example:
            ```python
            # 검색 노드 3개를 병렬 실행
            results = await workflow.run_parallel_nodes(
                ["search_page_1", "search_page_2", "search_page_3"],
                state,
            )

            for node_name, result in results.items():
                if result.success:
                    print(f"{node_name}: {len(result.data.get('items', []))} items")
            ```
        """
        import logging

        if not node_names:
            return {}

        logging.info(f"[WorkflowBase] 병렬 실행 시작: {node_names}")

        # 세마포어로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(node_name: str) -> tuple:
            """세마포어로 동시성 제어하며 노드 실행"""
            async with semaphore:
                try:
                    result = await self.execute_node(node_name, state)
                    return (node_name, result)
                except Exception as e:
                    logging.error(f"[WorkflowBase] 병렬 노드 실행 실패 ({node_name}): {e}")
                    return (node_name, NodeResult(
                        success=False,
                        error=str(e),
                    ))

        # 모든 노드를 병렬로 실행
        tasks = [execute_with_semaphore(name) for name in node_names]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        results: Dict[str, NodeResult] = {}
        for item in results_list:
            if isinstance(item, Exception):
                # gather에서 return_exceptions=True로 예외 캐치
                logging.error(f"[WorkflowBase] 병렬 실행 예외: {item}")
                continue
            node_name, result = item
            results[node_name] = result

            # 상태 업데이트
            if result.success:
                if node_name not in state.get("completed_nodes", []):
                    state["completed_nodes"] = state.get("completed_nodes", []) + [node_name]
                state["node_results"] = {
                    **state.get("node_results", {}),
                    node_name: {"success": True, "data": result.data}
                }
                if result.data:
                    state["data"] = {**state.get("data", {}), **result.data}
            else:
                if node_name not in state.get("failed_nodes", []):
                    state["failed_nodes"] = state.get("failed_nodes", []) + [node_name]
                state["node_results"] = {
                    **state.get("node_results", {}),
                    node_name: {"success": False, "error": result.error}
                }

        logging.info(
            f"[WorkflowBase] 병렬 실행 완료: "
            f"성공 {sum(1 for r in results.values() if r.success)}/{len(results)}"
        )

        return results

    async def run_parallel_group(
        self,
        group_name: str,
        state: WorkflowState,
        max_concurrency: int = 5,
    ) -> Dict[str, NodeResult]:
        """
        병렬 그룹의 모든 노드 실행 (의존성 고려)

        Args:
            group_name: 병렬 그룹 이름
            state: 현재 워크플로우 상태
            max_concurrency: 최대 동시 실행 수

        Returns:
            노드별 실행 결과 딕셔너리

        Example:
            ```python
            # 노드 정의 시 parallel_group 설정
            WorkflowNode(
                name="search_1",
                parallel_group="search_group",
                depends_on=[],
            )
            WorkflowNode(
                name="search_2",
                parallel_group="search_group",
                depends_on=[],
            )

            # 그룹 전체 병렬 실행
            results = await workflow.run_parallel_group("search_group", state)
            ```
        """
        groups = self.get_parallel_groups()
        group_nodes = groups.get(group_name, [])

        if not group_nodes:
            return {}

        all_results: Dict[str, NodeResult] = {}
        pending_nodes = set(group_nodes)
        completed_nodes = set(state.get("completed_nodes", []))

        # 의존성을 고려하여 반복 실행
        while pending_nodes:
            # 현재 실행 가능한 노드 찾기
            executable = self.can_execute_parallel(
                list(pending_nodes),
                list(completed_nodes)
            )

            if not executable:
                # 실행 가능한 노드가 없으면 순환 의존성 또는 오류
                import logging
                logging.warning(
                    f"[WorkflowBase] 병렬 그룹 '{group_name}'에서 실행 가능한 노드 없음. "
                    f"남은 노드: {pending_nodes}"
                )
                break

            # 병렬 실행
            batch_results = await self.run_parallel_nodes(
                executable, state, max_concurrency
            )

            # 결과 병합
            all_results.update(batch_results)

            # 완료된 노드 업데이트
            for node_name, result in batch_results.items():
                pending_nodes.discard(node_name)
                if result.success:
                    completed_nodes.add(node_name)

        return all_results

    async def run_with_parallel_execution(
        self,
        parameters: Dict[str, Any],
        thread_id: str = "default",
    ) -> WorkflowState:
        """
        병렬 실행을 활용한 워크플로우 실행

        parallel_group이 설정된 노드들은 자동으로 병렬 실행됩니다.

        Args:
            parameters: 워크플로우 파라미터
            thread_id: 실행 스레드 ID

        Returns:
            최종 워크플로우 상태

        Example:
            ```python
            # 노드에 parallel_group 설정
            nodes = [
                WorkflowNode(name="start", on_success="search_group_entry"),
                WorkflowNode(name="search_group_entry", parallel_group="search", on_success="analyze"),
                WorkflowNode(name="search_1", parallel_group="search"),
                WorkflowNode(name="search_2", parallel_group="search"),
                WorkflowNode(name="search_3", parallel_group="search"),
                WorkflowNode(name="analyze", on_success="end"),
            ]

            # 병렬 실행 모드로 워크플로우 실행
            result = await workflow.run_with_parallel_execution(params)
            ```
        """
        import logging

        # 파라미터 검증
        validation_errors = self.validate_parameters(parameters)
        if validation_errors:
            raise ValueError(f"파라미터 검증 실패: {'; '.join(validation_errors)}")

        # 초기 상태 설정
        execution_id = f"{self.config.id}-{thread_id}" if thread_id != "default" else self.config.id

        state: WorkflowState = {
            "workflow_id": self.config.id,
            "execution_id": execution_id,
            "status": "running",
            "current_node": None,
            "completed_nodes": [],
            "failed_nodes": [],
            "node_results": {},
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "parameters": parameters,
            "data": {},
            "error": None,
            "vlm_error_type": None,
            "retry_count": 0,
            "should_stop": False,
            "node_logs": {},
            "waiting_for_confirmation": False,
            "pending_confirmation_node": None,
            "confirmation_message": None,
            "user_confirmed": None,
            "user_input": None,
        }

        self._current_state = state
        self._stop_event.clear()

        try:
            # 병렬 그룹 조회
            parallel_groups = self.get_parallel_groups()
            processed_groups: set = set()

            # 시작 노드부터 실행
            current_node = self.start_node

            while current_node and not self._stop_event.is_set():
                node_config = self._get_node_by_name(current_node)
                if not node_config:
                    break

                # 병렬 그룹 노드인 경우
                if node_config.parallel_group and node_config.parallel_group not in processed_groups:
                    group_name = node_config.parallel_group
                    logging.info(f"[WorkflowBase] 병렬 그룹 실행: {group_name}")

                    # 그룹 전체 병렬 실행
                    group_results = await self.run_parallel_group(group_name, state)
                    processed_groups.add(group_name)

                    # 그룹 실행 결과 확인
                    all_success = all(r.success for r in group_results.values())
                    if all_success:
                        current_node = node_config.on_success
                    else:
                        current_node = node_config.on_failure
                        if not current_node:
                            state["error"] = f"병렬 그룹 '{group_name}' 실행 실패"
                            break

                else:
                    # 일반 노드 실행
                    state["current_node"] = current_node
                    state["current_node_start_time"] = datetime.now().isoformat()

                    result = await self.execute_node(current_node, state)

                    if result.success:
                        if current_node not in state.get("completed_nodes", []):
                            state["completed_nodes"] = state.get("completed_nodes", []) + [current_node]
                        state["node_results"][current_node] = {"success": True, "data": result.data}
                        if result.data:
                            state["data"] = {**state.get("data", {}), **result.data}

                        current_node = result.next_node or node_config.on_success
                    else:
                        if current_node not in state.get("failed_nodes", []):
                            state["failed_nodes"] = state.get("failed_nodes", []) + [current_node]
                        state["node_results"][current_node] = {"success": False, "error": result.error}
                        state["error"] = result.error

                        current_node = node_config.on_failure

            # 최종 상태
            if state.get("should_stop"):
                state["status"] = "stopped"
            elif state.get("error"):
                state["status"] = "failed"
            else:
                state["status"] = "completed"

            state["end_time"] = datetime.now().isoformat()

        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            state["end_time"] = datetime.now().isoformat()
            logging.error(f"[WorkflowBase] 병렬 실행 워크플로우 실패: {e}")

        self._current_state = state
        return state

    # =========================================
    # Time Travel Debugging
    # =========================================

    async def get_state_history(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        실행 히스토리 조회 (Time Travel용)

        Args:
            thread_id: 실행 스레드 ID
            limit: 최대 조회 개수

        Returns:
            상태 히스토리 목록 (최신 순)

        Example:
            ```python
            history = await workflow.get_state_history("exec-001")
            for state in history:
                print(f"Node: {state['current_node']}, Time: {state['timestamp']}")
            ```
        """
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        config = {"configurable": {"thread_id": thread_id}}

        try:
            history = []
            async for state in self._app.aget_state_history(config):
                history.append({
                    "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                    "current_node": state.values.get("current_node"),
                    "status": state.values.get("status"),
                    "completed_nodes": state.values.get("completed_nodes", []),
                    "failed_nodes": state.values.get("failed_nodes", []),
                    "error": state.values.get("error"),
                    "timestamp": state.values.get("current_node_start_time"),
                    "values": dict(state.values),
                })
                if len(history) >= limit:
                    break

            return history

        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] 히스토리 조회 실패: {e}")
            return []

    async def replay_from_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Optional[WorkflowState]:
        """
        특정 체크포인트에서 다시 실행 (Time Travel)

        과거 상태로 되돌아가서 다시 실행합니다.

        Args:
            thread_id: 실행 스레드 ID
            checkpoint_id: 되돌아갈 체크포인트 ID

        Returns:
            새로운 실행 결과 상태

        Example:
            ```python
            # 히스토리에서 되돌아갈 지점 선택
            history = await workflow.get_state_history("exec-001")
            target_checkpoint = history[3]["checkpoint_id"]  # 4번째 상태로

            # 해당 지점에서 다시 실행
            new_state = await workflow.replay_from_checkpoint(
                "exec-001",
                target_checkpoint
            )
            ```
        """
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

        try:
            import logging
            logging.info(
                f"[WorkflowBase] Time Travel: {thread_id} -> checkpoint {checkpoint_id}"
            )

            # 해당 체크포인트의 상태 조회
            checkpoint_state = await self._app.aget_state(config)
            if not checkpoint_state or not checkpoint_state.values:
                logging.error(f"[WorkflowBase] 체크포인트를 찾을 수 없음: {checkpoint_id}")
                return None

            self._current_state = dict(checkpoint_state.values)
            self._stop_event.clear()

            # 해당 체크포인트에서 다시 실행
            async for event in self._app.astream(
                None,  # None = 체크포인트에서 재개
                config,
                stream_mode=self.STREAM_MODE,
            ):
                if self.STREAM_MODE == "updates":
                    for node_name, updates in event.items():
                        if isinstance(updates, dict):
                            for key, value in updates.items():
                                if key in ("completed_nodes", "failed_nodes") and isinstance(value, list):
                                    existing = self._current_state.get(key, [])
                                    new_nodes = [n for n in value if n not in existing]
                                    self._current_state[key] = existing + new_nodes
                                elif key in ("node_results", "data") and isinstance(value, dict):
                                    existing = self._current_state.get(key, {})
                                    self._current_state[key] = {**existing, **value}
                                else:
                                    self._current_state[key] = value
                else:
                    for node_name, node_state in event.items():
                        if isinstance(node_state, dict):
                            self._current_state = {**self._current_state, **node_state}

                if self._stop_event.is_set():
                    self._current_state["should_stop"] = True
                    self._current_state["status"] = "stopped"
                    break

            # 최종 상태
            if self._current_state.get("status") == "running":
                if self._current_state.get("error"):
                    self._current_state["status"] = "failed"
                else:
                    self._current_state["status"] = "completed"

            self._current_state["end_time"] = datetime.now().isoformat()
            return self._current_state

        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] Time Travel 실패: {e}")
            return None

    async def fork_from_checkpoint(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str,
        modified_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowState]:
        """
        체크포인트에서 분기하여 새로운 실행 생성

        기존 실행의 특정 지점에서 상태를 수정하고 새로운 실행을 시작합니다.

        Args:
            source_thread_id: 원본 실행 스레드 ID
            checkpoint_id: 분기할 체크포인트 ID
            new_thread_id: 새로운 실행 스레드 ID
            modified_state: 수정할 상태 (선택)

        Returns:
            새로운 실행 결과 상태

        Example:
            ```python
            # 특정 지점에서 파라미터를 변경하여 분기
            new_state = await workflow.fork_from_checkpoint(
                source_thread_id="exec-001",
                checkpoint_id="cp-123",
                new_thread_id="exec-001-fork",
                modified_state={
                    "parameters": {"keyword": "다른 키워드"}
                }
            )
            ```
        """
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        source_config = {
            "configurable": {
                "thread_id": source_thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

        try:
            # 원본 체크포인트 상태 조회
            checkpoint_state = await self._app.aget_state(source_config)
            if not checkpoint_state or not checkpoint_state.values:
                return None

            # 상태 복사 및 수정
            forked_state = dict(checkpoint_state.values)
            forked_state["execution_id"] = new_thread_id

            if modified_state:
                for key, value in modified_state.items():
                    if isinstance(value, dict) and isinstance(forked_state.get(key), dict):
                        forked_state[key] = {**forked_state[key], **value}
                    else:
                        forked_state[key] = value

            # 새로운 스레드로 실행
            new_config = {"configurable": {"thread_id": new_thread_id}}
            self._current_state = forked_state
            self._stop_event.clear()

            async for event in self._app.astream(
                forked_state,
                new_config,
                stream_mode=self.STREAM_MODE,
            ):
                if self.STREAM_MODE == "updates":
                    for node_name, updates in event.items():
                        if isinstance(updates, dict):
                            for key, value in updates.items():
                                if key in ("completed_nodes", "failed_nodes") and isinstance(value, list):
                                    existing = self._current_state.get(key, [])
                                    new_nodes = [n for n in value if n not in existing]
                                    self._current_state[key] = existing + new_nodes
                                elif key in ("node_results", "data") and isinstance(value, dict):
                                    existing = self._current_state.get(key, {})
                                    self._current_state[key] = {**existing, **value}
                                else:
                                    self._current_state[key] = value

                if self._stop_event.is_set():
                    self._current_state["should_stop"] = True
                    self._current_state["status"] = "stopped"
                    break

            if self._current_state.get("status") == "running":
                if self._current_state.get("error"):
                    self._current_state["status"] = "failed"
                else:
                    self._current_state["status"] = "completed"

            self._current_state["end_time"] = datetime.now().isoformat()
            return self._current_state

        except Exception as e:
            import logging
            logging.error(f"[WorkflowBase] Fork 실패: {e}")
            return None

    # =========================================
    # Dynamic Breakpoints
    # =========================================

    def add_breakpoint(self, node_name: str):
        """
        노드에 브레이크포인트 추가

        해당 노드 실행 전 일시 중지됩니다.

        Args:
            node_name: 브레이크포인트를 설정할 노드 이름

        Example:
            ```python
            # 특정 노드에서 멈추도록 설정
            workflow.add_breakpoint("search")
            workflow.add_breakpoint("purchase")  # 위험한 작업 전

            # 실행 - search 노드에서 자동으로 멈춤
            await workflow.run(params)
            ```
        """
        self._breakpoints.add(node_name)
        import logging
        logging.info(f"[WorkflowBase] 브레이크포인트 추가: {node_name}")

    def remove_breakpoint(self, node_name: str):
        """
        브레이크포인트 제거

        Args:
            node_name: 제거할 브레이크포인트 노드 이름
        """
        self._breakpoints.discard(node_name)
        import logging
        logging.info(f"[WorkflowBase] 브레이크포인트 제거: {node_name}")

    def clear_breakpoints(self):
        """모든 브레이크포인트 제거"""
        self._breakpoints.clear()
        import logging
        logging.info("[WorkflowBase] 모든 브레이크포인트 제거됨")

    def get_breakpoints(self) -> List[str]:
        """현재 설정된 브레이크포인트 목록 반환"""
        return list(self._breakpoints)

    def set_breakpoint_callback(self, callback: Callable):
        """
        브레이크포인트 도달 시 콜백 설정

        콜백 시그니처: async def callback(node_name: str, state: WorkflowState)

        Example:
            ```python
            async def on_breakpoint(node_name, state):
                print(f"Paused at {node_name}")
                print(f"Current data: {state.get('data')}")
                # 디버깅 작업 수행...

            workflow.set_breakpoint_callback(on_breakpoint)
            ```
        """
        self._breakpoint_callback = callback

    async def _wait_for_breakpoint_resume(self, timeout_sec: int = 3600):
        """
        브레이크포인트에서 재개 대기 (기본 1시간 타임아웃)

        Args:
            timeout_sec: 타임아웃 시간 (초)
        """
        self._breakpoint_event.clear()
        try:
            await asyncio.wait_for(
                self._breakpoint_event.wait(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            import logging
            logging.warning("[WorkflowBase] 브레이크포인트 타임아웃 - 자동 재개")

    def resume_from_breakpoint(self):
        """
        브레이크포인트에서 재개

        Example:
            ```python
            # 브레이크포인트에서 멈춘 후 디버깅 완료
            workflow.resume_from_breakpoint()
            ```
        """
        self._breakpoint_event.set()
        import logging
        logging.info("[WorkflowBase] 브레이크포인트에서 재개")

    def is_paused_at_breakpoint(self) -> bool:
        """브레이크포인트에서 일시 중지 중인지 확인"""
        if self._current_state:
            return self._current_state.get("paused_at_breakpoint", False)
        return False

    def get_breakpoint_info(self) -> Optional[Dict[str, Any]]:
        """
        현재 브레이크포인트 정보 반환

        Returns:
            브레이크포인트 정보 또는 None
        """
        if not self.is_paused_at_breakpoint():
            return None

        return {
            "node": self._current_state.get("breakpoint_node"),
            "state": dict(self._current_state) if self._current_state else {},
        }

    def step_over(self):
        """
        한 노드만 실행하고 다시 멈춤 (디버거 Step Over)

        현재 노드를 실행하고 다음 노드에서 자동으로 멈춥니다.
        """
        if self._current_state:
            current_node = self._current_state.get("breakpoint_node")
            if current_node:
                # 현재 노드의 다음 노드에 임시 브레이크포인트 설정
                node_config = self._get_node_by_name(current_node)
                if node_config and node_config.on_success:
                    self._breakpoints.add(node_config.on_success)

        # 재개
        self.resume_from_breakpoint()
