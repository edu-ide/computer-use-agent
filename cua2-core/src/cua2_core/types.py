"""
Magentic-UI 스타일 타입 정의

Plan, PlanStep, SentinelPlanStep, Ledger 등
Orchestrator에서 사용하는 핵심 타입들
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel


class PlanStep(BaseModel):
    """
    플랜의 단일 스텝을 나타내는 클래스.

    Attributes:
        title (str): 스텝의 제목.
        details (str): 스텝의 상세 설명.
        agent_name (str): 이 스텝을 담당할 에이전트 이름.
    """

    title: str
    details: str
    agent_name: str


class SentinelPlanStep(PlanStep):
    """
    장기 실행 모니터링 또는 주기적 작업을 나타내는 클래스.

    Attributes:
        title (str): 스텝의 제목.
        details (str): 스텝의 상세 설명.
        agent_name (str): 이 스텝을 담당할 에이전트 이름.
        sleep_duration (int): 체크 사이의 대기 시간(초).
        condition (Union[int, str]):
            - 정수: 수행할 반복 횟수
            - 문자열: 완료 조건을 설명하는 자연어
    """

    sleep_duration: int
    condition: Union[int, str]


class Plan(BaseModel):
    """
    여러 스텝으로 구성된 플랜을 나타내는 클래스.

    Attributes:
        task (str, optional): 작업의 이름. 기본값: None
        steps (List[PlanStep]): 작업을 완료하기 위한 스텝 목록.

    Example:
        plan = Plan(
            task="웹사이트 열기",
            steps=[PlanStep(title="구글 열기", details="google.com으로 이동")]
        )
    """

    task: Optional[str] = None
    steps: Sequence[PlanStep] = []

    def __getitem__(self, index: int) -> PlanStep:
        return self.steps[index]

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        """플랜의 문자열 표현 반환."""
        plan_str = ""
        if self.task is not None:
            plan_str += f"Task: {self.task}\n"
        for i, step in enumerate(self.steps):
            plan_str += f"{i}. {step.agent_name}: {step.title}\n   {step.details}\n"
            if isinstance(step, SentinelPlanStep):
                condition_str = str(step.condition)
                plan_str += f"   [Sentinel: every {step.sleep_duration}s, condition: {condition_str}]\n"
        return plan_str

    @classmethod
    def from_list_of_dicts_or_str(
        cls, plan_dict: Union[List[Dict[str, str]], str, List[Any], Dict[str, Any]]
    ) -> Optional["Plan"]:
        """딕셔너리 목록 또는 JSON 문자열에서 Plan 로드."""
        if isinstance(plan_dict, str):
            plan_dict = json.loads(plan_dict)
        if len(plan_dict) == 0:
            return None
        assert isinstance(plan_dict, (list, dict))

        task = None
        if isinstance(plan_dict, dict):
            task = plan_dict.get("task", None)
            plan_dict = plan_dict.get("steps", [])

        steps: List[PlanStep] = []
        for raw_step in plan_dict:
            if isinstance(raw_step, dict):
                step: dict[str, Any] = raw_step

                # condition과 sleep_duration 필드가 있으면 sentinel step
                if "condition" in step and "sleep_duration" in step:
                    steps.append(
                        SentinelPlanStep(
                            title=step.get("title", "Untitled Step"),
                            details=step.get("details", "No details provided."),
                            agent_name=step.get("agent_name", "agent"),
                            sleep_duration=step.get("sleep_duration", 0),
                            condition=step.get("condition", "indefinite"),
                        )
                    )
                else:
                    steps.append(
                        PlanStep(
                            title=step.get("title", "Untitled Step"),
                            details=step.get("details", "No details provided."),
                            agent_name=step.get("agent_name", "agent"),
                        )
                    )
        return cls(task=task, steps=steps) if steps else None


class HumanInputFormat(BaseModel):
    """
    사용자 입력 형식을 나타내고 검증하는 클래스.

    Attributes:
        content (str): 입력 내용.
        accepted (bool, optional): 입력이 수락되었는지 여부. 기본값: False
        plan (Plan, optional): 플랜 객체.
    """

    content: str
    accepted: bool = False
    plan: Optional[Plan] = None

    @classmethod
    def from_str(cls, input_str: str) -> "HumanInputFormat":
        """문자열에서 HumanInputFormat 로드 (검증 후)."""
        try:
            data = json.loads(input_str)
            if not isinstance(data, dict):
                raise ValueError("Input string must be a JSON object")
        except (json.JSONDecodeError, ValueError):
            data = {"content": input_str}
        assert isinstance(data, dict)

        return cls(
            content=str(data.get("content", "")),
            accepted=bool(data.get("accepted", False)),
            plan=Plan.from_list_of_dicts_or_str(data.get("plan", [])),
        )

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "HumanInputFormat":
        """딕셔너리에서 HumanInputFormat 로드."""
        return cls(
            content=str(input_dict.get("content", "")),
            accepted=bool(input_dict.get("accepted", False)),
            plan=input_dict.get("plan", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """입력의 딕셔너리 표현 반환."""
        return self.model_dump()

    def to_str(self) -> str:
        """입력의 문자열 표현 반환."""
        return json.dumps(self.model_dump())


@dataclass
class TaskLedger:
    """
    Task Ledger - 외부 루프 (전체 작업 상태)

    Magentic-One의 Dual-Loop 시스템에서 외부 루프를 담당.
    전체 작업에 대한 facts, guesses, 현재 플랜을 관리.
    """
    # 확인된 사실들
    facts: List[str] = field(default_factory=list)
    # 추측/가정들
    guesses: List[str] = field(default_factory=list)
    # 현재 플랜
    plan: Optional[Plan] = None
    # 원본 작업 설명
    original_task: str = ""
    # 최종 답변
    final_answer: Optional[str] = None
    # 작업 완료 여부
    is_complete: bool = False

    def add_fact(self, fact: str) -> None:
        """새로운 fact 추가."""
        if fact not in self.facts:
            self.facts.append(fact)

    def add_guess(self, guess: str) -> None:
        """새로운 guess 추가."""
        if guess not in self.guesses:
            self.guesses.append(guess)

    def update_plan(self, plan: Plan) -> None:
        """플랜 업데이트."""
        self.plan = plan

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "facts": self.facts,
            "guesses": self.guesses,
            "plan": self.plan.model_dump() if self.plan else None,
            "original_task": self.original_task,
            "final_answer": self.final_answer,
            "is_complete": self.is_complete,
        }


@dataclass
class ProgressLedger:
    """
    Progress Ledger - 내부 루프 (스텝 실행 상태)

    Magentic-One의 Dual-Loop 시스템에서 내부 루프를 담당.
    현재 스텝의 진행 상황과 에이전트 상태를 추적.
    """
    # 현재 스텝 인덱스
    current_step_index: int = 0
    # 현재 스텝 시작 시간
    step_start_time: float = 0.0
    # 스텝별 실행 기록
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    # 현재 에이전트
    current_agent: str = ""
    # 에이전트 응답 히스토리
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    # 재시도 횟수
    retry_count: int = 0
    # 최대 재시도 횟수
    max_retries: int = 3
    # 현재 스텝 상태
    step_status: str = "pending"  # pending, running, completed, failed

    def start_step(self, step_index: int, agent_name: str) -> None:
        """새 스텝 시작."""
        import time
        self.current_step_index = step_index
        self.current_agent = agent_name
        self.step_start_time = time.time()
        self.step_status = "running"
        self.retry_count = 0

    def record_response(self, response: Dict[str, Any]) -> None:
        """에이전트 응답 기록."""
        import time
        self.agent_responses.append({
            **response,
            "timestamp": time.time(),
            "step_index": self.current_step_index,
        })

    def complete_step(self, success: bool, summary: str = "") -> None:
        """스텝 완료 기록."""
        import time
        self.step_status = "completed" if success else "failed"
        self.step_history.append({
            "step_index": self.current_step_index,
            "agent": self.current_agent,
            "success": success,
            "summary": summary,
            "duration": time.time() - self.step_start_time,
            "retry_count": self.retry_count,
        })

    def should_retry(self) -> bool:
        """재시도 가능 여부 확인."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """재시도 횟수 증가."""
        self.retry_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "current_step_index": self.current_step_index,
            "step_start_time": self.step_start_time,
            "step_history": self.step_history,
            "current_agent": self.current_agent,
            "agent_responses": self.agent_responses,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "step_status": self.step_status,
        }


@dataclass
class SentinelState:
    """
    Sentinel 스텝 실행 상태.

    장기 실행 모니터링 작업의 상태를 추적.
    """
    # 체크 횟수
    checks_done: int = 0
    # 마지막 체크 시간
    last_check_time: float = 0.0
    # 시작 시간
    start_time: float = 0.0
    # 현재 sleep duration (동적 조정 가능)
    current_sleep_duration: int = 60
    # 조건 충족 여부
    condition_met: bool = False
    # 에러 발생 여부
    error_encountered: bool = False
    # 마지막 체크 결과
    last_result: Optional[Dict[str, Any]] = None

    def record_check(self, result: Dict[str, Any]) -> None:
        """체크 결과 기록."""
        import time
        self.checks_done += 1
        self.last_check_time = time.time()
        self.last_result = result
        self.condition_met = result.get("condition_met", False)
        self.error_encountered = result.get("error_encountered", False)
        # 동적 sleep duration 조정
        if "sleep_duration" in result:
            self.current_sleep_duration = result["sleep_duration"]

    def should_continue(self, condition: Union[int, str]) -> bool:
        """계속 실행해야 하는지 확인."""
        if self.condition_met:
            return False
        if isinstance(condition, int):
            return self.checks_done < condition
        return True  # 문자열 조건은 condition_met으로 판단

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "checks_done": self.checks_done,
            "last_check_time": self.last_check_time,
            "start_time": self.start_time,
            "current_sleep_duration": self.current_sleep_duration,
            "condition_met": self.condition_met,
            "error_encountered": self.error_encountered,
            "last_result": self.last_result,
        }
