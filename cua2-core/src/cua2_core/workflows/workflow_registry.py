"""
워크플로우 레지스트리 - 모든 워크플로우 관리
"""

from typing import Dict, List, Optional, Type
import asyncio
from datetime import datetime

from .workflow_base import WorkflowBase, WorkflowState


class WorkflowRegistry:
    """
    워크플로우 레지스트리

    모든 사용 가능한 워크플로우를 등록하고 관리합니다.
    싱글톤 패턴으로 구현되어 앱 전체에서 공유됩니다.
    """

    _instance: Optional["WorkflowRegistry"] = None

    def __init__(self):
        self._workflows: Dict[str, Type[WorkflowBase]] = {}
        self._running_instances: Dict[str, WorkflowBase] = {}
        self._execution_history: Dict[str, List[WorkflowState]] = {}

    @classmethod
    def get_instance(cls) -> "WorkflowRegistry":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_default_workflows()
        return cls._instance

    def _register_default_workflows(self):
        """기본 워크플로우 등록"""
        from .coupang_workflow import CoupangCollectWorkflow
        from .youtube_workflow import YouTubeContentWorkflow
        self.register(CoupangCollectWorkflow)
        self.register(YouTubeContentWorkflow)

    def register(self, workflow_class: Type[WorkflowBase]):
        """워크플로우 클래스 등록"""
        # 임시 인스턴스로 config 가져오기
        temp_instance = workflow_class()
        workflow_id = temp_instance.config.id
        self._workflows[workflow_id] = workflow_class

    def unregister(self, workflow_id: str):
        """워크플로우 등록 해제"""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]

    def get_workflow_class(self, workflow_id: str) -> Optional[Type[WorkflowBase]]:
        """워크플로우 클래스 가져오기"""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict]:
        """등록된 모든 워크플로우 목록"""
        result = []
        for workflow_id, workflow_class in self._workflows.items():
            temp_instance = workflow_class()
            graph_def = temp_instance.get_graph_definition()
            result.append(graph_def)
        return result

    def get_workflow_detail(self, workflow_id: str) -> Optional[Dict]:
        """워크플로우 상세 정보"""
        workflow_class = self._workflows.get(workflow_id)
        if workflow_class:
            temp_instance = workflow_class()
            return temp_instance.get_graph_definition()
        return None

    async def start_workflow(
        self,
        workflow_id: str,
        parameters: Dict,
        execution_id: Optional[str] = None,
        agent_runner=None,
    ) -> str:
        """
        워크플로우 시작

        Args:
            workflow_id: 워크플로우 ID
            parameters: 워크플로우 파라미터
            execution_id: 실행 ID (지정하지 않으면 자동 생성)
            agent_runner: VLM 에이전트 실행기

        Returns:
            실행 ID
        """
        workflow_class = self._workflows.get(workflow_id)
        if not workflow_class:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        # 실행 ID 생성
        if not execution_id:
            execution_id = f"{workflow_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 인스턴스 생성
        instance = workflow_class(agent_runner=agent_runner) if agent_runner else workflow_class()
        self._running_instances[execution_id] = instance

        # 비동기 실행
        asyncio.create_task(self._run_workflow(execution_id, instance, parameters))

        return execution_id

    async def _run_workflow(
        self,
        execution_id: str,
        instance: WorkflowBase,
        parameters: Dict
    ):
        """워크플로우 실행 (내부)"""
        try:
            final_state = await instance.run(parameters, thread_id=execution_id)

            # 히스토리에 저장
            workflow_id = instance.config.id
            if workflow_id not in self._execution_history:
                self._execution_history[workflow_id] = []
            self._execution_history[workflow_id].append(final_state)

        except ValueError as e:
            # 파라미터 검증 실패
            print(f"[WorkflowRegistry] Workflow {execution_id} validation failed: {e}")
            # 에러 상태 설정
            if instance._current_state is None:
                instance._current_state = {
                    "workflow_id": instance.config.id,
                    "execution_id": execution_id,  # 실행 ID 추가
                    "status": "failed",
                    "error": str(e),
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "completed_nodes": [],
                    "failed_nodes": [],
                    "parameters": parameters,
                    "data": {},
                    "node_logs": {},
                }
            else:
                instance._current_state["status"] = "failed"
                instance._current_state["error"] = str(e)
                instance._current_state["end_time"] = datetime.now().isoformat()

        except Exception as e:
            print(f"[WorkflowRegistry] Workflow {execution_id} failed: {e}")
            # 에러 상태 설정
            if instance._current_state:
                instance._current_state["status"] = "failed"
                instance._current_state["error"] = str(e)
                instance._current_state["end_time"] = datetime.now().isoformat()

        finally:
            # 실행 완료 후 정리 (상태 조회를 위해 일정 시간 유지)
            await asyncio.sleep(60)  # 1분 후 정리
            if execution_id in self._running_instances:
                del self._running_instances[execution_id]

    def stop_workflow(self, execution_id: str) -> bool:
        """워크플로우 중지"""
        instance = self._running_instances.get(execution_id)
        if instance:
            instance.stop()
            return True
        return False

    def get_execution_state(self, execution_id: str) -> Optional[WorkflowState]:
        """실행 중인 워크플로우 상태 조회"""
        instance = self._running_instances.get(execution_id)
        if instance:
            return instance.get_state()
        return None

    def list_running_workflows(self) -> List[Dict]:
        """실행 중인 워크플로우 목록"""
        result = []
        for execution_id, instance in self._running_instances.items():
            state = instance.get_state()
            if state:
                result.append({
                    "execution_id": execution_id,
                    "workflow_id": instance.config.id,
                    "workflow_name": instance.config.name,
                    "status": state.get("status"),
                    "current_node": state.get("current_node"),
                    "start_time": state.get("start_time"),
                })
        return result

    def get_execution_history(self, workflow_id: str, limit: int = 10) -> List[WorkflowState]:
        """워크플로우 실행 히스토리"""
        history = self._execution_history.get(workflow_id, [])
        return history[-limit:]


# 싱글톤 인스턴스 가져오기
def get_workflow_registry() -> WorkflowRegistry:
    return WorkflowRegistry.get_instance()
