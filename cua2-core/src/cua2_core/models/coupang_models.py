"""
쿠팡 상품 수집 관련 모델
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class CoupangProduct:
    """쿠팡 상품 정보"""
    keyword: str
    name: str
    price: int
    seller_type: str  # 일반배송, 로켓직구, 판매자로켓
    url: str
    id: Optional[int] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    thumbnail: Optional[str] = None
    rank: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Task:
    """작업 체인의 개별 태스크"""
    name: str
    instruction: str
    on_success: Optional[str] = None  # 성공 시 다음 태스크 이름
    on_failure: Optional[str] = None  # 실패 시 다음 태스크 이름
    save_to_state: list[str] = field(default_factory=list)  # state에 저장할 키
    status: str = "pending"  # pending, running, success, failed, skipped
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class TaskChain:
    """작업 체인 - 여러 태스크를 연결"""
    name: str
    tasks: dict[str, Task]  # name -> Task
    start_task: str  # 시작 태스크 이름
    current_task: Optional[str] = None
    state: dict = field(default_factory=dict)  # 태스크 간 공유 상태
    status: str = "pending"  # pending, running, completed, failed, stopped


@dataclass
class ChainExecutionState:
    """체인 실행 상태"""
    chain_name: str
    current_task: Optional[str]
    completed_tasks: list[str]
    failed_tasks: list[str]
    state: dict
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
