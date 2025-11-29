"""
워크플로우 베이스 클래스 - LangGraph 기반
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict, Annotated
import asyncio
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class NodeStatus(str, Enum):
    """노드 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """노드 실행 결과"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    next_node: Optional[str] = None  # 다음 노드 지정 (조건부 분기)


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

    # 시간 설정
    timeout_sec: int = 120  # 작업 제한시간 (초), 기본 2분
    avg_duration_sec: Optional[int] = None  # 평균 작업 시간 (초), 학습된 값

    # 재사용 및 메모리 설정
    reusable: bool = False  # 이 노드의 trace를 재사용 가능하게 저장할지
    reuse_trace: bool = False  # 이전 trace를 재사용할지 (같은 입력일 때)
    share_memory: bool = False  # 이전 노드와 메모리(컨텍스트) 공유할지
    cache_key_params: List[str] = field(default_factory=list)  # 캐시 키에 사용할 파라미터 목록


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

    # 중지 플래그
    should_stop: bool

    # 노드별 실행 로그 (VLM 스텝 정보)
    node_logs: Dict[str, List[Dict[str, Any]]]


class WorkflowBase(ABC):
    """
    워크플로우 베이스 클래스

    모든 워크플로우는 이 클래스를 상속받아 구현합니다.
    LangGraph의 StateGraph를 사용하여 노드 기반 실행을 지원합니다.
    """

    def __init__(self):
        self._graph: Optional[StateGraph] = None
        self._app = None
        self._checkpointer = MemorySaver()
        self._stop_event = asyncio.Event()
        self._current_state: Optional[WorkflowState] = None

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
                    # 즉시 _current_state 업데이트
                    if self._current_state is not None:
                        if node_name not in self._current_state.get("completed_nodes", []):
                            self._current_state["completed_nodes"] = self._current_state.get("completed_nodes", []) + [node_name]
                else:
                    state_update["failed_nodes"] = [node_name]
                    state_update["node_results"] = {
                        **state.get("node_results", {}),
                        node_name: {"success": False, "error": result.error}
                    }
                    state_update["error"] = result.error
                    # 즉시 _current_state 업데이트
                    if self._current_state is not None:
                        if node_name not in self._current_state.get("failed_nodes", []):
                            self._current_state["failed_nodes"] = self._current_state.get("failed_nodes", []) + [node_name]
                        self._current_state["error"] = result.error

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
        """조건부 라우팅 함수 생성"""
        def router(state: WorkflowState) -> str:
            # 중지 체크
            if state.get("should_stop", False):
                return "end"

            # 결과에서 다음 노드 지정 확인
            next_node = state.get("_next_node")
            if next_node:
                return next_node

            # 노드 결과 확인
            node_result = state.get("node_results", {}).get(node.name, {})

            if node_result.get("success", False):
                return node.on_success or "end"
            else:
                return node.on_failure or "end"

        return router

    def _get_edge_mapping(self, node: WorkflowNode) -> Dict[str, str]:
        """엣지 매핑 생성"""
        mapping = {"end": END}

        if node.on_success:
            mapping[node.on_success] = node.on_success
        if node.on_failure:
            mapping[node.on_failure] = node.on_failure

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

        # 그래프 빌드
        if self._graph is None:
            self._graph = self.build_graph()
            self._app = self._graph.compile(checkpointer=self._checkpointer)

        # 실행 ID 설정
        # workflow_registry에서 execution_id를 thread_id로 전달 (예: "coupang-collect-20251129092555")
        # thread_id가 이미 workflow_id를 포함하면 그대로 사용
        if thread_id.startswith(self.config.id):
            execution_id = thread_id  # 이미 완전한 execution_id
        elif thread_id != "default":
            execution_id = f"{self.config.id}-{thread_id}"  # 타임스탬프만 있는 경우
        else:
            execution_id = self.config.id  # 기본값

        # 초기 상태
        initial_state: WorkflowState = {
            "workflow_id": self.config.id,
            "execution_id": execution_id,  # 실행별 고유 ID
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
            "should_stop": False,
            "node_logs": {},  # 노드별 실행 로그
        }

        self._current_state = initial_state
        self._stop_event.clear()

        # 실행
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in self._app.astream(initial_state, config):
                # 상태 업데이트
                for node_name, node_state in event.items():
                    if isinstance(node_state, dict):
                        self._current_state = {**self._current_state, **node_state}

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
            self._current_state["status"] = "failed"
            self._current_state["error"] = str(e)
            self._current_state["end_time"] = datetime.now().isoformat()

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
