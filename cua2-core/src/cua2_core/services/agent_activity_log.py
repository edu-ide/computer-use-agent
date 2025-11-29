"""
에이전트 활동 로그 서비스

각 에이전트(Orchestrator, VLM 등)의 작업 내역을 기록하고
UI에 실시간으로 전달합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import deque


class AgentType(Enum):
    """에이전트 유형"""
    ORCHESTRATOR = "orchestrator"  # Orchestrator-8B
    VLM = "vlm"  # Qwen3-VL
    MEMORY = "memory"  # Letta Memory
    TRACE = "trace"  # Trace Store


class ActivityType(Enum):
    """활동 유형"""
    DECISION = "decision"  # 판단/결정
    EXECUTION = "execution"  # 실행
    CACHE_HIT = "cache_hit"  # 캐시 히트
    CACHE_MISS = "cache_miss"  # 캐시 미스
    API_CALL = "api_call"  # API 호출
    ERROR = "error"  # 에러
    WARNING = "warning"  # 경고
    INFO = "info"  # 정보


@dataclass
class AgentActivity:
    """에이전트 활동 기록"""
    id: str
    agent_type: str  # AgentType.value
    activity_type: str  # ActivityType.value
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "activity_type": self.activity_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "execution_id": self.execution_id,
            "node_id": self.node_id,
            "duration_ms": self.duration_ms,
            "time_ago": self._get_time_ago(),
        }

    def _get_time_ago(self) -> str:
        """시간 경과 문자열 (예: '3초 전', '2분 전')"""
        elapsed = time.time() - self.timestamp

        if elapsed < 1:
            return "방금"
        elif elapsed < 60:
            return f"{int(elapsed)}초 전"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}분 전"
        elif elapsed < 86400:
            return f"{int(elapsed / 3600)}시간 전"
        else:
            return f"{int(elapsed / 86400)}일 전"


class AgentActivityLog:
    """
    에이전트 활동 로그 관리자

    모든 에이전트의 활동을 기록하고 UI에 전달합니다.
    """

    MAX_LOGS = 100  # 최대 로그 수

    def __init__(self):
        self._logs: deque[AgentActivity] = deque(maxlen=self.MAX_LOGS)
        self._log_id_counter = 0
        self._listeners: List[Callable[[AgentActivity], None]] = []
        self._async_listeners: List[Callable[[AgentActivity], Any]] = []

        # 에이전트별 최신 활동
        self._latest_by_agent: Dict[str, AgentActivity] = {}

    def _generate_id(self) -> str:
        """고유 ID 생성"""
        self._log_id_counter += 1
        return f"act-{int(time.time() * 1000)}-{self._log_id_counter}"

    def log(
        self,
        agent_type: AgentType,
        activity_type: ActivityType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        node_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> AgentActivity:
        """
        활동 로그 기록

        Args:
            agent_type: 에이전트 유형
            activity_type: 활동 유형
            message: 활동 메시지
            details: 추가 상세 정보
            execution_id: 워크플로우 실행 ID
            node_id: 노드 ID
            duration_ms: 소요 시간 (밀리초)

        Returns:
            생성된 AgentActivity
        """
        activity = AgentActivity(
            id=self._generate_id(),
            agent_type=agent_type.value,
            activity_type=activity_type.value,
            message=message,
            details=details or {},
            execution_id=execution_id,
            node_id=node_id,
            duration_ms=duration_ms,
        )

        # 로그 저장
        self._logs.append(activity)
        self._latest_by_agent[agent_type.value] = activity

        # 리스너들에게 알림
        self._notify_listeners(activity)

        return activity

    def _notify_listeners(self, activity: AgentActivity):
        """리스너들에게 활동 알림"""
        # 동기 리스너
        for listener in self._listeners:
            try:
                listener(activity)
            except Exception as e:
                print(f"[AgentActivityLog] 리스너 오류: {e}")

        # 비동기 리스너
        for async_listener in self._async_listeners:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(async_listener(activity))
            except RuntimeError:
                # 이벤트 루프 없으면 스킵
                pass

    def add_listener(self, listener: Callable[[AgentActivity], None]):
        """동기 리스너 추가"""
        self._listeners.append(listener)

    def add_async_listener(self, listener: Callable[[AgentActivity], Any]):
        """비동기 리스너 추가"""
        self._async_listeners.append(listener)

    def remove_listener(self, listener: Callable):
        """리스너 제거"""
        if listener in self._listeners:
            self._listeners.remove(listener)
        if listener in self._async_listeners:
            self._async_listeners.remove(listener)

    def get_logs(
        self,
        limit: int = 50,
        agent_type: Optional[AgentType] = None,
        execution_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        로그 조회

        Args:
            limit: 최대 개수
            agent_type: 필터링할 에이전트 유형
            execution_id: 필터링할 실행 ID

        Returns:
            활동 로그 목록 (최신순)
        """
        logs = list(self._logs)

        # 필터링
        if agent_type:
            logs = [l for l in logs if l.agent_type == agent_type.value]
        if execution_id:
            logs = [l for l in logs if l.execution_id == execution_id]

        # 최신순 정렬 및 제한
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]

        # time_ago 업데이트하여 반환
        return [l.to_dict() for l in logs]

    def get_latest_by_agent(self) -> Dict[str, Dict[str, Any]]:
        """
        에이전트별 최신 활동 조회

        Returns:
            {agent_type: activity_dict} 형태
        """
        result = {}
        for agent_type, activity in self._latest_by_agent.items():
            result[agent_type] = activity.to_dict()
        return result

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """
        에이전트 상태 요약

        UI에 표시할 에이전트별 상태 정보
        """
        agents = [
            {
                "type": AgentType.ORCHESTRATOR.value,
                "name": "Orchestrator-8B",
                "icon": "🧠",
                "color": "#8B5CF6",  # purple
            },
            {
                "type": AgentType.VLM.value,
                "name": "Qwen3-VL",
                "icon": "👁️",
                "color": "#3B82F6",  # blue
            },
            {
                "type": AgentType.MEMORY.value,
                "name": "Letta Memory",
                "icon": "💾",
                "color": "#10B981",  # green
            },
            {
                "type": AgentType.TRACE.value,
                "name": "Trace Store",
                "icon": "📝",
                "color": "#F59E0B",  # amber
            },
        ]

        for agent in agents:
            latest = self._latest_by_agent.get(agent["type"])
            if latest:
                agent["latest_activity"] = latest.to_dict()
                agent["status"] = "active"
            else:
                agent["latest_activity"] = None
                agent["status"] = "idle"

        return agents

    def clear(self):
        """로그 초기화"""
        self._logs.clear()
        self._latest_by_agent.clear()


# 싱글톤 인스턴스
_activity_log: Optional[AgentActivityLog] = None


def get_agent_activity_log() -> AgentActivityLog:
    """에이전트 활동 로그 싱글톤 반환"""
    global _activity_log
    if _activity_log is None:
        _activity_log = AgentActivityLog()
    return _activity_log


# 편의 함수들
def log_orchestrator(
    activity_type: ActivityType,
    message: str,
    **kwargs
) -> AgentActivity:
    """Orchestrator 활동 로그"""
    return get_agent_activity_log().log(
        AgentType.ORCHESTRATOR,
        activity_type,
        message,
        **kwargs
    )


def log_vlm(
    activity_type: ActivityType,
    message: str,
    **kwargs
) -> AgentActivity:
    """VLM 활동 로그"""
    return get_agent_activity_log().log(
        AgentType.VLM,
        activity_type,
        message,
        **kwargs
    )


def log_memory(
    activity_type: ActivityType,
    message: str,
    **kwargs
) -> AgentActivity:
    """Memory 활동 로그"""
    return get_agent_activity_log().log(
        AgentType.MEMORY,
        activity_type,
        message,
        **kwargs
    )


def log_trace(
    activity_type: ActivityType,
    message: str,
    **kwargs
) -> AgentActivity:
    """Trace 활동 로그"""
    return get_agent_activity_log().log(
        AgentType.TRACE,
        activity_type,
        message,
        **kwargs
    )
