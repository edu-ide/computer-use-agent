"""
Magentic-One 스타일 Orchestrator

Task Ledger (외부 루프)와 Progress Ledger (내부 루프)를 사용하여
복잡한 작업을 계획하고 실행하는 Orchestrator.

로컬 LLM (Orchestrator-8B, Fara-7B)과 통합.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from ..types import (
    Plan,
    PlanStep,
    ProgressLedger,
    SentinelPlanStep,
    SentinelState,
    TaskLedger,
)
from .config import OrchestratorConfig
from .prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_SYSTEM_MESSAGE_EXECUTION,
    ORCHESTRATOR_TASK_LEDGER_FULL_FORMAT,
    get_orchestrator_plan_prompt_json,
    get_orchestrator_plan_replan_json,
    get_orchestrator_progress_ledger_prompt,
    get_orchestrator_system_message_planning,
    validate_ledger_json,
    validate_plan_json,
)
from .sentinel_prompts import (
    ORCHESTRATOR_SENTINEL_CONDITION_CHECK_PROMPT,
    validate_sentinel_condition_check_json,
)


@dataclass
class AgentInfo:
    """에이전트 정보."""
    name: str
    description: str
    execute_fn: Optional[Callable] = None


@dataclass
class OrchestratorState:
    """Orchestrator 상태."""
    task: str = ""
    plan: Optional[Plan] = None
    task_ledger: TaskLedger = field(default_factory=TaskLedger)
    progress_ledger: ProgressLedger = field(default_factory=ProgressLedger)
    current_step_idx: int = 0
    n_rounds: int = 0
    n_replans: int = 0
    in_planning_mode: bool = True
    is_complete: bool = False
    final_answer: Optional[str] = None
    message_history: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        """상태 초기화."""
        self.task = ""
        self.plan = None
        self.task_ledger = TaskLedger()
        self.progress_ledger = ProgressLedger()
        self.current_step_idx = 0
        self.n_rounds = 0
        self.n_replans = 0
        self.in_planning_mode = True
        self.is_complete = False
        self.final_answer = None
        self.message_history = []


class LLMClient:
    """로컬 LLM 클라이언트 (SGLang/OpenAI 호환 API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        model: str = "Orchestrator-8B",
        api_key: str = "EMPTY",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """채팅 완성 요청."""
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM 요청 실패: {e}")
            raise

    async def close(self):
        """클라이언트 종료."""
        await self.client.aclose()


class Orchestrator:
    """
    Magentic-One 스타일 Orchestrator.

    Task Ledger와 Progress Ledger를 사용한 Dual-Loop 시스템 구현.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        orchestrator_llm: Optional[LLMClient] = None,
        agents: Optional[Dict[str, AgentInfo]] = None,
    ):
        self.config = config
        self.state = OrchestratorState()

        # LLM 클라이언트 설정
        self.llm = orchestrator_llm or LLMClient()

        # 에이전트 등록
        self.agents: Dict[str, AgentInfo] = agents or {}

        # Sentinel 상태
        self._sentinel_states: Dict[int, SentinelState] = {}

    def register_agent(
        self,
        name: str,
        description: str,
        execute_fn: Optional[Callable] = None,
    ) -> None:
        """에이전트 등록."""
        self.agents[name] = AgentInfo(
            name=name,
            description=description,
            execute_fn=execute_fn,
        )

    def _get_team_description(self) -> str:
        """팀 설명 생성."""
        descriptions = []
        for agent in self.agents.values():
            descriptions.append(f"{agent.name}: {agent.description}")
        return "\n".join(descriptions)

    async def create_plan(self, task: str) -> Plan:
        """
        Task Ledger - 플랜 생성 (외부 루프).

        사용자 작업을 분석하고 실행 계획을 생성.
        """
        self.state.task = task
        self.state.task_ledger.original_task = task
        self.state.in_planning_mode = True

        date_today = datetime.now().strftime("%Y-%m-%d")
        team_description = self._get_team_description()

        # 시스템 메시지 생성
        system_message = get_orchestrator_system_message_planning(
            enable_sentinel_steps=self.config.sentinel_plan.enable_sentinel_steps
        ).format(
            date_today=date_today,
            team=team_description,
        )

        # 플랜 생성 프롬프트
        plan_prompt = get_orchestrator_plan_prompt_json(
            enable_sentinel_steps=self.config.sentinel_plan.enable_sentinel_steps
        ).format(
            team=team_description,
            additional_instructions="",
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Task: {task}\n\n{plan_prompt}"},
        ]

        # LLM 호출
        response = await self.llm.chat_completion(messages)

        # JSON 파싱
        plan = self._parse_plan_response(response)

        if plan:
            self.state.plan = plan
            self.state.task_ledger.update_plan(plan)
            self.state.in_planning_mode = False

        return plan

    async def replan(self, reason: str = "") -> Optional[Plan]:
        """
        재계획 생성.

        현재 진행 상황을 고려하여 새로운 계획 생성.
        """
        if not self.config.allow_for_replans:
            return None

        if self.config.max_replans and self.state.n_replans >= self.config.max_replans:
            return None

        self.state.n_replans += 1

        team_description = self._get_team_description()
        current_plan_str = str(self.state.plan) if self.state.plan else ""

        replan_prompt = get_orchestrator_plan_replan_json(
            enable_sentinel_steps=self.config.sentinel_plan.enable_sentinel_steps
        ).format(
            task=self.state.task,
            team=team_description,
            plan=current_plan_str,
            additional_instructions=reason,
        )

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE_EXECUTION},
            {"role": "user", "content": replan_prompt},
        ]

        response = await self.llm.chat_completion(messages)
        plan = self._parse_plan_response(response)

        if plan:
            self.state.plan = plan
            self.state.task_ledger.update_plan(plan)

        return plan

    async def execute_step(self) -> Dict[str, Any]:
        """
        Progress Ledger - 현재 스텝 실행 (내부 루프).

        현재 플랜 스텝을 실행하고 결과를 반환.
        """
        if not self.state.plan or self.state.current_step_idx >= len(self.state.plan):
            return {"status": "complete", "message": "모든 스텝 완료"}

        step = self.state.plan[self.state.current_step_idx]

        # Progress Ledger 업데이트
        self.progress_ledger.start_step(
            step_index=self.state.current_step_idx,
            agent_name=step.agent_name,
        )

        # Sentinel 스텝인 경우 특별 처리
        if isinstance(step, SentinelPlanStep):
            return await self._execute_sentinel_step(step)

        # 일반 스텝 실행
        return await self._execute_normal_step(step)

    async def _execute_normal_step(self, step: PlanStep) -> Dict[str, Any]:
        """일반 스텝 실행."""
        agent = self.agents.get(step.agent_name)

        if not agent:
            return {
                "status": "error",
                "message": f"에이전트를 찾을 수 없음: {step.agent_name}",
            }

        result = {"status": "pending"}

        if agent.execute_fn:
            try:
                result = await agent.execute_fn(step)
                self.state.progress_ledger.complete_step(
                    success=result.get("status") == "success",
                    summary=result.get("message", ""),
                )
            except Exception as e:
                result = {"status": "error", "message": str(e)}
                if self.state.progress_ledger.should_retry():
                    self.state.progress_ledger.increment_retry()
                    return await self._execute_normal_step(step)
                self.state.progress_ledger.complete_step(success=False, summary=str(e))

        # 다음 스텝으로 이동
        self.state.current_step_idx += 1
        self.state.n_rounds += 1

        return result

    async def _execute_sentinel_step(self, step: SentinelPlanStep) -> Dict[str, Any]:
        """
        Sentinel 스텝 실행 (장기 모니터링 작업).
        """
        step_idx = self.state.current_step_idx

        # Sentinel 상태 초기화
        if step_idx not in self._sentinel_states:
            self._sentinel_states[step_idx] = SentinelState(
                start_time=time.time(),
                current_sleep_duration=step.sleep_duration,
            )

        sentinel_state = self._sentinel_states[step_idx]

        # 에이전트 실행
        agent = self.agents.get(step.agent_name)
        if not agent or not agent.execute_fn:
            return {"status": "error", "message": f"에이전트를 찾을 수 없음: {step.agent_name}"}

        try:
            result = await agent.execute_fn(step)
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        # 조건 체크
        condition_result = await self._check_sentinel_condition(step, result)
        sentinel_state.record_check(condition_result)

        # 계속 실행해야 하는지 확인
        if sentinel_state.should_continue(step.condition):
            # 동적 sleep duration 적용
            sleep_duration = sentinel_state.current_sleep_duration
            await asyncio.sleep(sleep_duration)
            return await self._execute_sentinel_step(step)

        # 완료
        self.state.progress_ledger.complete_step(
            success=sentinel_state.condition_met,
            summary=f"Sentinel 완료: {sentinel_state.checks_done}회 체크",
        )
        self.state.current_step_idx += 1

        return {
            "status": "success" if sentinel_state.condition_met else "timeout",
            "checks_done": sentinel_state.checks_done,
            "condition_met": sentinel_state.condition_met,
        }

    async def _check_sentinel_condition(
        self, step: SentinelPlanStep, agent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sentinel 조건 체크.

        LLM을 사용하여 조건 충족 여부 판단.
        """
        sentinel_state = self._sentinel_states[self.state.current_step_idx]

        prompt = ORCHESTRATOR_SENTINEL_CONDITION_CHECK_PROMPT.format(
            step_description=step.details,
            current_sleep_duration=sentinel_state.current_sleep_duration,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time_since_started=time.time() - sentinel_state.start_time,
            checks_done=sentinel_state.checks_done,
            time_since_last_check=time.time() - sentinel_state.last_check_time
            if sentinel_state.last_check_time > 0
            else 0,
            condition=step.condition,
        )

        messages = [
            {"role": "system", "content": "You are evaluating sentinel conditions."},
            {
                "role": "user",
                "content": f"Agent result: {json.dumps(agent_result)}\n\n{prompt}",
            },
        ]

        response = await self.llm.chat_completion(messages, temperature=0.3)

        try:
            result = self._extract_json(response)
            if validate_sentinel_condition_check_json(result):
                return result
        except Exception:
            pass

        # 기본값 반환
        return {
            "condition_met": False,
            "reason": "조건 체크 실패",
            "sleep_duration": sentinel_state.current_sleep_duration,
            "sleep_duration_reason": "기본값 유지",
            "error_encountered": True,
        }

    async def get_final_answer(self) -> str:
        """
        최종 답변 생성.
        """
        prompt = ORCHESTRATOR_FINAL_ANSWER_PROMPT.format(task=self.state.task)

        # 대화 히스토리 구성
        history_str = ""
        for msg in self.state.message_history[-10:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE_EXECUTION},
            {"role": "user", "content": f"{history_str}\n\n{prompt}"},
        ]

        response = await self.llm.chat_completion(messages)

        self.state.final_answer = response
        self.state.is_complete = True
        self.state.task_ledger.final_answer = response
        self.state.task_ledger.is_complete = True

        return response

    async def run(self, task: str) -> Dict[str, Any]:
        """
        전체 작업 실행.

        1. 플랜 생성 (Task Ledger)
        2. 스텝별 실행 (Progress Ledger)
        3. 최종 답변 생성
        """
        # 플랜 생성
        plan = await self.create_plan(task)
        if not plan:
            return {"status": "error", "message": "플랜 생성 실패"}

        # 스텝 실행
        while self.state.current_step_idx < len(plan):
            if self.config.max_turns and self.state.n_rounds >= self.config.max_turns:
                break

            result = await self.execute_step()

            if result.get("status") == "error":
                # 에러 시 재계획 시도
                if self.config.allow_for_replans:
                    new_plan = await self.replan(reason=result.get("message", ""))
                    if new_plan:
                        continue
                break

        # 최종 답변
        final_answer = await self.get_final_answer()

        return {
            "status": "complete",
            "task": task,
            "plan": str(plan),
            "final_answer": final_answer,
            "rounds": self.state.n_rounds,
            "replans": self.state.n_replans,
        }

    def _parse_plan_response(self, response: str) -> Optional[Plan]:
        """LLM 응답에서 플랜 파싱."""
        try:
            plan_data = self._extract_json(response)
            if validate_plan_json(plan_data):
                return Plan.from_list_of_dicts_or_str(plan_data)
        except Exception as e:
            print(f"플랜 파싱 실패: {e}")
        return None

    def _extract_json(self, text: str) -> Any:
        """텍스트에서 JSON 추출."""
        # JSON 블록 찾기
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # 중괄호/대괄호 찾기
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in text:
                start = text.find(start_char)
                depth = 0
                for i, c in enumerate(text[start:]):
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            text = text[start : start + i + 1]
                            break

        return json.loads(text)

    @property
    def progress_ledger(self) -> ProgressLedger:
        """Progress Ledger 접근."""
        return self.state.progress_ledger

    @property
    def task_ledger(self) -> TaskLedger:
        """Task Ledger 접근."""
        return self.state.task_ledger
