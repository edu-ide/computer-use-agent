"""
작업 체인 서비스 - n8n 스타일의 태스크 체이닝
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional
from dataclasses import asdict

from cua2_core.models.coupang_models import Task, TaskChain, ChainExecutionState

logger = logging.getLogger(__name__)


class TaskChainService:
    """작업 체인 관리 및 실행 서비스"""

    def __init__(self):
        self.chains: dict[str, TaskChain] = {}
        self.execution_states: dict[str, ChainExecutionState] = {}
        self._running_chains: set[str] = set()
        self._stop_flags: dict[str, bool] = {}
        self._callbacks: dict[str, list[Callable]] = {}

    def register_chain(self, chain: TaskChain):
        """체인 등록"""
        self.chains[chain.name] = chain
        logger.info(f"체인 등록: {chain.name} ({len(chain.tasks)}개 태스크)")

    def get_chain(self, name: str) -> Optional[TaskChain]:
        """체인 조회"""
        return self.chains.get(name)

    def list_chains(self) -> list[str]:
        """등록된 체인 목록"""
        return list(self.chains.keys())

    def add_callback(self, chain_name: str, callback: Callable):
        """체인 상태 변경 콜백 등록"""
        if chain_name not in self._callbacks:
            self._callbacks[chain_name] = []
        self._callbacks[chain_name].append(callback)

    async def _notify_callbacks(self, chain_name: str, event: str, data: dict):
        """콜백 호출"""
        for callback in self._callbacks.get(chain_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"콜백 오류: {e}")

    async def run_chain(
        self,
        chain_name: str,
        initial_state: dict,
        task_executor: Callable[[str, dict], dict]
    ) -> ChainExecutionState:
        """
        체인 실행

        Args:
            chain_name: 실행할 체인 이름
            initial_state: 초기 상태
            task_executor: 태스크 실행 함수 (instruction, state) -> result

        Returns:
            최종 실행 상태
        """
        chain = self.chains.get(chain_name)
        if not chain:
            raise ValueError(f"체인을 찾을 수 없음: {chain_name}")

        if chain_name in self._running_chains:
            raise RuntimeError(f"체인이 이미 실행 중: {chain_name}")

        # 실행 상태 초기화
        self._running_chains.add(chain_name)
        self._stop_flags[chain_name] = False

        state = ChainExecutionState(
            chain_name=chain_name,
            current_task=None,
            completed_tasks=[],
            failed_tasks=[],
            state=initial_state.copy(),
            status="running",
            start_time=datetime.utcnow().isoformat(),
        )
        self.execution_states[chain_name] = state

        # 모든 태스크 초기화
        for task in chain.tasks.values():
            task.status = "pending"
            task.result = None
            task.error = None

        await self._notify_callbacks(chain_name, "chain_started", asdict(state))

        try:
            current_task_name = chain.start_task

            while current_task_name and not self._stop_flags.get(chain_name, False):
                task = chain.tasks.get(current_task_name)
                if not task:
                    logger.error(f"태스크를 찾을 수 없음: {current_task_name}")
                    break

                # 태스크 실행
                state.current_task = current_task_name
                task.status = "running"

                await self._notify_callbacks(chain_name, "task_started", {
                    "task_name": current_task_name,
                    "state": state.state,
                })

                try:
                    # instruction에서 state 변수 치환
                    instruction = task.instruction
                    for key, value in state.state.items():
                        instruction = instruction.replace(f"{{{key}}}", str(value))

                    # 태스크 실행
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, task_executor, instruction, state.state
                    )

                    task.status = "success"
                    task.result = result
                    state.completed_tasks.append(current_task_name)

                    # 결과를 state에 저장
                    if task.save_to_state and result:
                        for key in task.save_to_state:
                            if key in result:
                                state.state[key] = result[key]

                    await self._notify_callbacks(chain_name, "task_completed", {
                        "task_name": current_task_name,
                        "result": result,
                        "state": state.state,
                    })

                    # 다음 태스크 결정
                    current_task_name = task.on_success

                except Exception as e:
                    logger.error(f"태스크 실행 오류 ({current_task_name}): {e}")
                    task.status = "failed"
                    task.error = str(e)
                    state.failed_tasks.append(current_task_name)

                    await self._notify_callbacks(chain_name, "task_failed", {
                        "task_name": current_task_name,
                        "error": str(e),
                    })

                    # 실패 시 다음 태스크
                    current_task_name = task.on_failure

            # 체인 완료
            if self._stop_flags.get(chain_name, False):
                state.status = "stopped"
            else:
                state.status = "completed"

        except Exception as e:
            logger.error(f"체인 실행 오류: {e}")
            state.status = "failed"

        finally:
            state.end_time = datetime.utcnow().isoformat()
            state.current_task = None
            self._running_chains.discard(chain_name)

            await self._notify_callbacks(chain_name, "chain_completed", asdict(state))

        return state

    def stop_chain(self, chain_name: str):
        """체인 실행 중지"""
        if chain_name in self._running_chains:
            self._stop_flags[chain_name] = True
            logger.info(f"체인 중지 요청: {chain_name}")

    def get_execution_state(self, chain_name: str) -> Optional[ChainExecutionState]:
        """실행 상태 조회"""
        return self.execution_states.get(chain_name)

    def is_running(self, chain_name: str) -> bool:
        """체인 실행 중 여부"""
        return chain_name in self._running_chains


# 쿠팡 수집 체인 정의
def create_coupang_collection_chain(keyword: str, max_pages: int = 5) -> TaskChain:
    """쿠팡 상품 수집 체인 생성"""
    return TaskChain(
        name=f"coupang_collect_{keyword}",
        start_task="set_keyword",
        tasks={
            "set_keyword": Task(
                name="set_keyword",
                instruction=f"set_search_keyword('{keyword}')를 호출하여 키워드를 설정하세요.",
                on_success="open_coupang",
                save_to_state=["keyword"],
            ),
            "open_coupang": Task(
                name="open_coupang",
                instruction="쿠팡 홈페이지(https://www.coupang.com)를 열어주세요.",
                on_success="search_keyword",
                on_failure="open_coupang",  # 실패 시 재시도
            ),
            "search_keyword": Task(
                name="search_keyword",
                instruction=f"검색창에 '{keyword}'를 입력하고 검색하세요.",
                on_success="collect_products",
                on_failure="search_keyword",
            ),
            "collect_products": Task(
                name="collect_products",
                instruction="""현재 페이지의 상품들을 확인하세요:
1. 각 상품에 로켓배송 아이콘/텍스트가 있는지 확인
2. 로켓배송이 아닌 상품은 save_non_rocket_product()로 저장
3. 상품명, 가격, URL을 정확히 추출하세요""",
                on_success="scroll_and_check",
                save_to_state=["collected_count"],
            ),
            "scroll_and_check": Task(
                name="scroll_and_check",
                instruction="페이지를 스크롤하여 더 많은 상품을 확인하세요. 모든 상품을 확인했으면 다음 페이지로 이동하세요.",
                on_success="collect_products",
                on_failure="find_related_keywords",
            ),
            "find_related_keywords": Task(
                name="find_related_keywords",
                instruction="연관검색어 섹션을 찾아 새로운 키워드를 확인하세요. 발견된 키워드를 보고하세요.",
                on_success="report_results",
                save_to_state=["related_keywords"],
            ),
            "report_results": Task(
                name="report_results",
                instruction="get_collection_stats()를 호출하고 mark_keyword_done()으로 수집을 완료하세요. 최종 결과를 보고하세요.",
                on_success=None,  # 체인 종료
            ),
        },
    )


# 싱글톤 인스턴스
_chain_service: Optional[TaskChainService] = None


def get_chain_service() -> TaskChainService:
    """체인 서비스 싱글톤 인스턴스"""
    global _chain_service
    if _chain_service is None:
        _chain_service = TaskChainService()
    return _chain_service
