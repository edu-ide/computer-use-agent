"""
Letta 메모리 서비스 - 워크플로우 에이전트의 자율적 메모리 관리

Letta(MemGPT)를 사용하여 에이전트가 스스로 메모리를 관리하도록 합니다.

메모리 블록 구조:
- workflow_state: 현재 워크플로우 상태 (현재 노드, 진행률 등)
- task_context: 현재 태스크 컨텍스트 (무엇을 하고 있는지)
- collected_data: 수집한 데이터 요약 (상품 목록 등)
- observations: 관찰 기록 (페이지 상태, 발견한 것들)
- error_history: 에러 히스토리 (어떤 에러가 있었고 어떻게 해결했는지)
"""

import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """메모리 블록 정의"""
    label: str
    value: str
    description: str
    limit: int = 5000
    read_only: bool = False


@dataclass
class NodeReuseSetting:
    """노드별 재사용 설정 (VLM이 학습하여 자동 설정)"""
    node_id: str
    reusable: bool = False
    reuse_trace: bool = False
    share_memory: bool = False
    cache_key_params: List[str] = field(default_factory=list)
    confidence: float = 0.0  # VLM의 확신도 (0~1)
    reason: str = ""  # 설정 이유


@dataclass
class WorkflowMemoryConfig:
    """워크플로우용 메모리 설정"""
    workflow_id: str
    workflow_name: str
    initial_blocks: List[MemoryBlock] = field(default_factory=list)

    @classmethod
    def for_coupang(cls, keyword: str = "") -> "WorkflowMemoryConfig":
        """쿠팡 수집 워크플로우용 메모리 설정"""
        return cls(
            workflow_id="coupang-collect",
            workflow_name="쿠팡 상품 수집",
            initial_blocks=[
                MemoryBlock(
                    label="workflow_state",
                    description="현재 워크플로우 진행 상태. 현재 노드, 완료된 노드, 진행률을 기록합니다.",
                    value=f"""## 워크플로우 상태
- 워크플로우: 쿠팡 상품 수집
- 검색 키워드: {keyword}
- 현재 단계: 시작 전
- 완료된 노드: 없음
- 진행률: 0%
""",
                ),
                MemoryBlock(
                    label="task_context",
                    description="현재 수행 중인 태스크의 상세 컨텍스트. 무엇을 하고 있고 다음에 뭘 해야 하는지 기록합니다.",
                    value="""## 현재 태스크
- 상태: 대기 중
- 목표: 쿠팡에서 비로켓배송 상품 수집
- 다음 액션: 쿠팡 웹사이트 열기
""",
                ),
                MemoryBlock(
                    label="collected_data",
                    description="수집한 데이터 요약. 찾은 상품, 가격대, 판매자 정보 등을 기록합니다.",
                    value="""## 수집 데이터 요약
- 총 수집 상품: 0개
- 분석한 페이지: 0개
- 가격대: -
- 주요 판매자: -
""",
                ),
                MemoryBlock(
                    label="observations",
                    description="페이지 관찰 기록. 현재 화면에서 보이는 것, 발견한 패턴, 주의할 점을 기록합니다.",
                    value="""## 관찰 기록
- 현재 페이지: 없음
- 발견한 패턴: -
- 주의할 점: -
""",
                ),
                MemoryBlock(
                    label="error_history",
                    description="에러 및 해결 히스토리. 발생한 문제와 해결 방법을 기록하여 반복을 피합니다.",
                    value="""## 에러 히스토리
최근 에러 없음.
""",
                ),
                MemoryBlock(
                    label="reuse_learning",
                    description="노드별 재사용 학습 데이터. VLM이 실행 결과를 분석하여 재사용 설정을 자동으로 학습합니다.",
                    value="""## 재사용 학습 데이터
아직 학습된 데이터가 없습니다.

### 학습 기준:
- reusable: 노드 결과가 일관적이고 외부 상태에 의존하지 않을 때
- reuse_trace: 동일 입력에 동일 출력이 보장될 때
- share_memory: 이전 노드의 컨텍스트가 필요할 때
- cache_key_params: 결과에 영향을 주는 파라미터 목록
""",
                ),
            ],
        )


class LettaMemoryService:
    """
    Letta 메모리 서비스

    워크플로우 에이전트의 메모리를 Letta를 통해 관리합니다.
    Letta 서버가 없으면 로컬 폴백 모드로 동작합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: Letta API 키 (없으면 환경변수에서 읽음)
            base_url: Letta 서버 URL (셀프호스팅 시 사용)
        """
        self._api_key = api_key or os.getenv("LETTA_API_KEY")
        self._base_url = base_url or os.getenv("LETTA_BASE_URL", "http://localhost:8283")
        self._client = None
        self._agents: Dict[str, str] = {}  # workflow_id -> agent_id
        self._fallback_mode = False
        self._fallback_memory: Dict[str, Dict[str, str]] = {}  # 로컬 폴백 메모리

        self._initialize_client()

    def _initialize_client(self):
        """Letta 클라이언트 초기화"""
        try:
            from letta_client import Letta

            if self._api_key:
                self._client = Letta(api_key=self._api_key)
            else:
                # 로컬 서버 연결 시도
                self._client = Letta(base_url=self._base_url)

            # 연결 테스트
            # self._client.agents.list()  # 연결 확인
            logger.info(f"Letta 클라이언트 초기화 완료: {self._base_url}")

        except ImportError:
            logger.warning("letta-client 패키지가 설치되지 않음. 폴백 모드 사용.")
            self._fallback_mode = True

        except Exception as e:
            logger.warning(f"Letta 서버 연결 실패: {e}. 폴백 모드 사용.")
            self._fallback_mode = True

    async def create_workflow_agent(
        self,
        config: WorkflowMemoryConfig,
        model: str = "openai/gpt-4o-mini",
        embedding_model: str = "openai/text-embedding-3-small",
    ) -> str:
        """
        워크플로우용 에이전트 생성

        Args:
            config: 워크플로우 메모리 설정
            model: 사용할 LLM 모델
            embedding_model: 임베딩 모델

        Returns:
            agent_id: 생성된 에이전트 ID
        """
        if self._fallback_mode:
            return self._create_fallback_agent(config)

        try:
            # 메모리 블록 준비
            memory_blocks = [
                {
                    "label": block.label,
                    "value": block.value,
                    "description": block.description,
                    "limit": block.limit,
                }
                for block in config.initial_blocks
            ]

            # 에이전트 생성
            agent_state = self._client.agents.create(
                name=f"{config.workflow_id}-agent",
                model=model,
                embedding=embedding_model,
                memory_blocks=memory_blocks,
                system=self._get_system_prompt(config),
            )

            agent_id = agent_state.id
            self._agents[config.workflow_id] = agent_id

            logger.info(f"Letta 에이전트 생성: {agent_id} for {config.workflow_id}")
            return agent_id

        except Exception as e:
            logger.error(f"에이전트 생성 실패: {e}")
            return self._create_fallback_agent(config)

    def _create_fallback_agent(self, config: WorkflowMemoryConfig) -> str:
        """폴백 모드에서 로컬 메모리로 에이전트 생성"""
        agent_id = f"fallback-{config.workflow_id}"
        self._fallback_memory[agent_id] = {
            block.label: block.value for block in config.initial_blocks
        }
        self._agents[config.workflow_id] = agent_id
        logger.info(f"폴백 에이전트 생성: {agent_id}")
        return agent_id

    def _get_system_prompt(self, config: WorkflowMemoryConfig) -> str:
        """워크플로우용 시스템 프롬프트"""
        return f"""You are an AI agent managing a workflow: {config.workflow_name}

Your role is to:
1. Execute tasks step by step
2. Update your memory blocks as you progress
3. Track collected data and observations
4. Learn from errors and avoid repeating them

Memory Management Rules:
- Update 'workflow_state' when moving to a new node or completing tasks
- Update 'task_context' with current objectives and next steps
- Update 'collected_data' when you gather new information
- Update 'observations' with what you see on the screen
- Update 'error_history' when errors occur and how you resolved them

Always keep your memory up-to-date so you can resume work efficiently.
"""

    async def get_memory_block(
        self,
        workflow_id: str,
        block_label: str,
    ) -> Optional[str]:
        """
        메모리 블록 조회

        Args:
            workflow_id: 워크플로우 ID
            block_label: 블록 레이블

        Returns:
            블록 내용 (없으면 None)
        """
        agent_id = self._agents.get(workflow_id)
        if not agent_id:
            return None

        if self._fallback_mode or agent_id.startswith("fallback-"):
            return self._fallback_memory.get(agent_id, {}).get(block_label)

        try:
            block = self._client.agents.blocks.retrieve(
                agent_id=agent_id,
                block_label=block_label,
            )
            return block.value
        except Exception as e:
            logger.error(f"메모리 블록 조회 실패: {e}")
            return None

    async def update_memory_block(
        self,
        workflow_id: str,
        block_label: str,
        value: str,
    ) -> bool:
        """
        메모리 블록 업데이트

        Args:
            workflow_id: 워크플로우 ID
            block_label: 블록 레이블
            value: 새 값

        Returns:
            성공 여부
        """
        agent_id = self._agents.get(workflow_id)
        if not agent_id:
            return False

        if self._fallback_mode or agent_id.startswith("fallback-"):
            if agent_id in self._fallback_memory:
                self._fallback_memory[agent_id][block_label] = value
                return True
            return False

        try:
            self._client.agents.blocks.update(
                agent_id=agent_id,
                block_label=block_label,
                value=value,
            )
            return True
        except Exception as e:
            logger.error(f"메모리 블록 업데이트 실패: {e}")
            return False

    async def get_all_memory(self, workflow_id: str) -> Dict[str, str]:
        """
        모든 메모리 블록 조회

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            {label: value} 딕셔너리
        """
        agent_id = self._agents.get(workflow_id)
        if not agent_id:
            return {}

        if self._fallback_mode or agent_id.startswith("fallback-"):
            return self._fallback_memory.get(agent_id, {}).copy()

        try:
            blocks = self._client.agents.blocks.list(agent_id=agent_id)
            return {block.label: block.value for block in blocks}
        except Exception as e:
            logger.error(f"메모리 조회 실패: {e}")
            return {}

    async def update_workflow_state(
        self,
        workflow_id: str,
        current_node: str,
        completed_nodes: List[str],
        progress: int,
        additional_info: str = "",
    ):
        """워크플로우 상태 업데이트 헬퍼"""
        value = f"""## 워크플로우 상태
- 현재 단계: {current_node}
- 완료된 노드: {', '.join(completed_nodes) if completed_nodes else '없음'}
- 진행률: {progress}%
{additional_info}
"""
        await self.update_memory_block(workflow_id, "workflow_state", value)

    async def update_task_context(
        self,
        workflow_id: str,
        status: str,
        current_goal: str,
        next_action: str,
    ):
        """태스크 컨텍스트 업데이트 헬퍼"""
        value = f"""## 현재 태스크
- 상태: {status}
- 목표: {current_goal}
- 다음 액션: {next_action}
"""
        await self.update_memory_block(workflow_id, "task_context", value)

    async def update_collected_data(
        self,
        workflow_id: str,
        total_items: int,
        pages_analyzed: int,
        price_range: str = "-",
        top_sellers: str = "-",
    ):
        """수집 데이터 요약 업데이트 헬퍼"""
        value = f"""## 수집 데이터 요약
- 총 수집 상품: {total_items}개
- 분석한 페이지: {pages_analyzed}개
- 가격대: {price_range}
- 주요 판매자: {top_sellers}
"""
        await self.update_memory_block(workflow_id, "collected_data", value)

    async def add_observation(
        self,
        workflow_id: str,
        current_page: str,
        observation: str,
    ):
        """관찰 기록 추가"""
        current = await self.get_memory_block(workflow_id, "observations") or ""

        # 최근 관찰 5개만 유지
        lines = current.split("\n")
        observations = [l for l in lines if l.startswith("- ")][-4:]

        value = f"""## 관찰 기록
- 현재 페이지: {current_page}
{''.join(observations)}
- {observation}
"""
        await self.update_memory_block(workflow_id, "observations", value)

    async def add_error(
        self,
        workflow_id: str,
        error: str,
        resolution: str = "",
    ):
        """에러 기록 추가"""
        current = await self.get_memory_block(workflow_id, "error_history") or ""

        # 최근 에러 3개만 유지
        lines = current.split("\n")
        errors = [l for l in lines if l.startswith("- ")][-2:]

        value = f"""## 에러 히스토리
{''.join(errors)}
- 에러: {error}
  해결: {resolution or '미해결'}
"""
        await self.update_memory_block(workflow_id, "error_history", value)

    async def update_reuse_learning(
        self,
        workflow_id: str,
        learning_data: str,
    ):
        """재사용 학습 데이터 업데이트"""
        await self.update_memory_block(workflow_id, "reuse_learning", learning_data)

    async def get_reuse_learning(
        self,
        workflow_id: str,
    ) -> Optional[str]:
        """재사용 학습 데이터 조회"""
        return await self.get_memory_block(workflow_id, "reuse_learning")

    async def update_node_reuse_settings(
        self,
        workflow_id: str,
        node_id: str,
        settings: Dict[str, Any],
    ):
        """개별 노드의 재사용 설정 업데이트"""
        current = await self.get_memory_block(workflow_id, "reuse_learning") or ""

        # 기존 데이터 파싱
        import re
        node_section_pattern = rf"### {re.escape(node_id)}\n.*?(?=###|\Z)"
        node_section = re.search(node_section_pattern, current, re.DOTALL)

        # 새 노드 섹션 생성
        new_section = f"""### {node_id}
- 결정: {settings.get('decision', 'uncertain')} (확신도: {settings.get('confidence', 0):.1%})
- 이유: {settings.get('reason', '-')}
- 추천 설정: reusable={settings.get('reusable', False)}, reuse_trace={settings.get('reuse_trace', False)}, share_memory={settings.get('share_memory', False)}
- cache_key_params: {settings.get('cache_key_params', [])}
"""

        if node_section:
            # 기존 섹션 교체
            updated = re.sub(node_section_pattern, new_section, current, flags=re.DOTALL)
        else:
            # 새 섹션 추가
            if "## 재사용 학습 데이터" in current:
                # 헤더 다음에 추가
                updated = current.replace(
                    "## 재사용 학습 데이터\n",
                    f"## 재사용 학습 데이터\n\n{new_section}"
                )
            else:
                updated = f"## 재사용 학습 데이터\n\n{new_section}"

        await self.update_memory_block(workflow_id, "reuse_learning", updated)

    async def get_context_for_agent(self, workflow_id: str) -> str:
        """
        에이전트에 전달할 컨텍스트 문자열 생성

        모든 메모리 블록을 합쳐서 에이전트가 이해할 수 있는 형식으로 반환
        """
        memory = await self.get_all_memory(workflow_id)

        if not memory:
            return ""

        context_parts = []
        for label, value in memory.items():
            context_parts.append(f"=== {label.upper()} ===\n{value}")

        return "\n\n".join(context_parts)

    async def cleanup_agent(self, workflow_id: str):
        """에이전트 정리"""
        agent_id = self._agents.pop(workflow_id, None)
        if not agent_id:
            return

        if self._fallback_mode or agent_id.startswith("fallback-"):
            self._fallback_memory.pop(agent_id, None)
            return

        try:
            self._client.agents.delete(agent_id=agent_id)
            logger.info(f"Letta 에이전트 삭제: {agent_id}")
        except Exception as e:
            logger.error(f"에이전트 삭제 실패: {e}")


# 싱글톤 인스턴스
_letta_service: Optional[LettaMemoryService] = None


def get_letta_memory_service() -> LettaMemoryService:
    """Letta 메모리 서비스 싱글톤 반환"""
    global _letta_service
    if _letta_service is None:
        _letta_service = LettaMemoryService()
    return _letta_service
