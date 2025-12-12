"""
VLM Agent Runner - 워크플로우에서 VLM 에이전트 실행을 위한 래퍼
"""

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageDraw
from smolagents import ActionStep, AgentMaxStepsError

from cua2_core.services.agent_utils.get_model import get_model
from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent
from cua2_core.services.agent_utils.function_parser import parse_function_call
from cua2_core.services.local_sandbox_service import LocalSandboxService
from cua2_core.services.headless_desktop import HeadlessDesktop
from cua2_core.services.utils import compress_image_to_max_size
from cua2_core.services.trace_store import get_trace_store
from cua2_core.services.orchestrator_service import (
    OrchestratorService,
    StepAction,
    StepFeedback,
    ExecutionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class VLMStepLog:
    """VLM 스텝 로그"""
    step_number: int
    timestamp: str
    screenshot: Optional[str] = None  # base64 encoded (Before Action)
    screenshot_after: Optional[str] = None  # base64 encoded (After Action)
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    orchestrator_feedback: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    # 에이전트/모델 정보
    agent_type: Optional[str] = None  # "VLMAgent", "SearchAgent" 등
    model_id: Optional[str] = None  # "local-qwen3-vl", "gpt-4o" 등
    node_name: Optional[str] = None  # 워크플로우 노드 이름


@dataclass
class VLMInstructionResult:
    """VLM 명령 실행 결과"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    steps: List[VLMStepLog] = field(default_factory=list)
    # 에러 추적
    consecutive_errors: int = 0  # 연속 에러 횟수
    early_stopped: bool = False  # 얼리 스탑 여부
    early_stop_reason: Optional[str] = None  # 얼리 스탑 사유


class AgentStopException(Exception):
    """에이전트 중지 예외"""
    pass


class EarlyStopException(Exception):
    """연속 에러로 인한 얼리 스탑 예외"""
    def __init__(self, message: str, consecutive_errors: int, last_error: str):
        super().__init__(message)
        self.consecutive_errors = consecutive_errors
        self.last_error = last_error


class BotDetectionException(Exception):
    """봇 감지로 인한 즉시 중단 예외"""
    def __init__(self, message: str, detected_pattern: str):
        super().__init__(message)
        self.detected_pattern = detected_pattern


class VLMAgentRunner:
    """
    VLM 에이전트 실행기 - 워크플로우에서 사용

    LocalAgentService의 핵심 로직을 워크플로우에서 사용할 수 있도록 추상화
    """

    # 얼리 스탑 설정
    MAX_CONSECUTIVE_ERRORS = 3  # 연속 에러 허용 횟수

    # Note: Rule-based 패턴 체크 완전 제거됨
    # VLM이 스크린샷을 보고 직접 [ERROR:TYPE] 형식으로 보고함
    # LangGraph 조건부 엣지에서 에러 타입별 라우팅 처리

    def __init__(
        self,
        model_id: str = "local-qwen3-vl",
        max_steps: int = 15,
        data_dir: Optional[str] = None,
        max_consecutive_errors: int = 3,  # 연속 에러 허용 횟수
        agent_type: str = "vlm",  # "vlm" 또는 "web"
        orchestrator: Optional[OrchestratorService] = None,  # Orchestrator 연동 (전략 선택용)
        planning_interval: Optional[int] = None,  # None=계획 재수립 비활성화 (빠른 추론)
        check_bot_detection: bool = True,  # Deprecated: VLM이 직접 [ERROR:TYPE] 보고
    ):
        self.model_id = model_id
        self.max_steps = max_steps
        self.data_dir = data_dir or "/tmp/vlm_agent_runner"
        self.max_consecutive_errors = max_consecutive_errors
        self.agent_type = agent_type
        self._orchestrator = orchestrator
        self.planning_interval = planning_interval  # smolagents planning_interval
        # Note: check_bot_detection은 하위 호환성을 위해 파라미터만 유지, 실제로 사용하지 않음
        # VLM이 스크린샷을 보고 직접 [ERROR:BOT_DETECTED] 형식으로 보고함

        # SandboxService는 'headless_browser' 모드에서는 사용하지 않음
        if self.agent_type != "headless_browser":
            self._sandbox_service = LocalSandboxService(max_sandboxes=1)
        else:
            self._sandbox_service = None
        self._current_sandbox = None
        self._current_agent: Optional[LocalVisionAgent] = None
        self._session_id: Optional[str] = None
        self._steps: List[VLMStepLog] = []
        self._last_screenshot: Optional[tuple[Image.Image, str]] = None
        self._should_stop = False

        # Orchestrator 컨텍스트
        self._workflow_id: Optional[str] = None
        self._node_id: Optional[str] = None
        self._injected_prompt: Optional[str] = None  # 다음 스텝에 주입할 프롬프트
        self._pending_memory_save: Optional[Dict[str, str]] = None  # 메모리 저장 대기

        # 연속 에러 추적
        self._consecutive_errors: int = 0
        self._last_error: Optional[str] = None
        self._current_action: Optional[str] = None
        self._current_observation: Optional[str] = None
        self._current_thought: Optional[str] = None

        # 추론 로그 (전체 과정 기록)
        self._reasoning_logs: List[Dict[str, Any]] = []

    def _check_vlm_error_report(self, final_answer: Optional[str]) -> Optional[str]:
        """
        VLM이 스크린샷 기반으로 보고한 에러 체크

        VLM이 final_answer에서 [ERROR:TYPE] 형식으로 에러를 보고하면 감지
        - [ERROR:BOT_DETECTED] - 봇 감지 (CAPTCHA, 접근 거부 등)
        - [ERROR:PAGE_FAILED] - 페이지 로딩 실패
        - [ERROR:ACCESS_DENIED] - 접근 거부

        Returns:
            에러 타입 문자열 또는 None
        """
        if not final_answer:
            return None

        import re
        # [ERROR:TYPE] 패턴 매칭
        match = re.search(r'\[ERROR:([A-Z_]+)\]', final_answer)
        if match:
            return match.group(1)

        return None

    async def initialize(self, session_id: str) -> bool:
        """샌드박스와 에이전트 초기화"""
        self._session_id = session_id
        self._should_stop = False

        # 데이터 디렉토리 생성
        session_data_dir = os.path.join(self.data_dir, session_id)
        os.makedirs(session_data_dir, exist_ok=True)

        # 샌드박스 획득 (Headless Browser 모드 지원)
        if self.agent_type == "headless_browser":
            if self._current_sandbox is None:
                try:
                    logger.info("Headless Browser 에이전트 모드: HeadlessDesktop 초기화 중...")
                    self._current_sandbox = HeadlessDesktop()
                except Exception as e:
                    logger.error(f"Headless Desktop 초기화 실패: {e}")
                    return False
        else:
            # 기본 VLM 모드: Xvfb 기반 LocalSandboxService 사용
            max_attempts = 30
            for attempt in range(max_attempts):
                response = await self._sandbox_service.acquire_sandbox(session_id)

                if response.error:
                    logger.error(f"샌드박스 획득 실패: {response.error}")
                    await asyncio.sleep(1)
                    continue

                if response.sandbox is not None and response.state == "ready":
                    self._current_sandbox = response.sandbox
                    break

                if response.state == "max_sandboxes_reached":
                    await asyncio.sleep(1)
                    continue

                await asyncio.sleep(1)

        if self._current_sandbox is None:
            logger.error(f"샌드박스를 획득할 수 없음: {session_id}")
            return False

        # 모델 가져오기
        model = get_model(self.model_id)

        # 에이전트 생성 (planning_interval로 매 스텝마다 계획 재수립)
        self._current_agent = LocalVisionAgent(
            model=model,
            data_dir=session_data_dir,
            desktop=self._current_sandbox,
            max_steps=self.max_steps,
            planning_interval=self.planning_interval,  # 적응형 실행: 매 스텝마다 재계획
        )

        logger.info(
            f"VLM 에이전트 초기화 완료: {session_id} "
            f"(planning_interval={self.planning_interval})"
        )
        return True

    async def run_instruction(
        self,
        instruction: str,
        on_step: Optional[Callable[[VLMStepLog], None]] = None,
        # Trace 재사용 설정
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        reuse_trace: bool = False,
        reusable: bool = False,
        cache_key_params: Optional[List[str]] = None,
        share_memory: bool = False,
    ) -> VLMInstructionResult:
        """
        VLM 에이전트에 명령 실행

        Args:
            instruction: 실행할 명령
            on_step: 스텝 완료 시 콜백
            workflow_id: 워크플로우 ID (trace 캐시용)
            node_id: 노드 ID (trace 캐시용)
            params: 파라미터 (trace 캐시 키 생성용)
            reuse_trace: 이전 trace 재사용 여부
            reusable: 이 trace를 저장할지 여부
            cache_key_params: 캐시 키에 사용할 파라미터 목록
            share_memory: 이전 노드의 메모리와 공유할지 (현재 미사용, 추후 구현)

        Returns:
            VLMInstructionResult: 실행 결과
        """
        if self._current_agent is None:
            return VLMInstructionResult(
                success=False,
                error="에이전트가 초기화되지 않았습니다"
            )

        # Trace 재사용 체크
        if reuse_trace and workflow_id and node_id:
            trace_store = get_trace_store()
            cached_trace = trace_store.get_reusable_trace(
                workflow_id=workflow_id,
                node_id=node_id,
                params=params or {},
                key_params=cache_key_params,
            )

            if cached_trace and cached_trace.success:
                logger.info(f"[TraceReuse] {node_id}: 캐시된 trace 재사용 (used_count: {cached_trace.used_count})")

                # 캐시된 스텝을 콜백으로 전달
                self._steps = []
                for step_data in cached_trace.steps:
                    step_log = VLMStepLog(
                        step_number=step_data.get("step_number", 0),
                        timestamp=step_data.get("timestamp", datetime.now().isoformat()),
                        screenshot=step_data.get("screenshot"),
                        thought=step_data.get("thought"),
                        action=step_data.get("action"),
                        observation=step_data.get("observation"),
                        error=step_data.get("error"),
                        tool_calls=step_data.get("tool_calls", []),
                    )
                    self._steps.append(step_log)
                    if on_step:
                        on_step(step_log)

                return VLMInstructionResult(
                    success=True,
                    data={
                        "reused_from_cache": True,
                        "cache_key": cached_trace.cache_key,
                        "steps_count": len(self._steps),
                        **cached_trace.data,
                    },
                    steps=self._steps,
                )

        self._steps = []
        step_number = 0

        # Orchestrator 컨텍스트 설정
        self._workflow_id = workflow_id
        self._node_id = node_id
        self._injected_prompt = None
        self._pending_memory_save = None

        # 연속 에러 카운트 초기화
        self._consecutive_errors = 0
        self._last_error = None
        self._current_thought = None
        self._current_action = None
        self._current_observation = None

        # Orchestrator로부터 동적 시스템 프롬프트 및 전략 결정
        final_instruction = instruction
        if self._orchestrator and node_id:
            # 1. 전략 결정 (Max Tokens 조절용)
            # Orchestrator가 복잡도를 분석하여 적절한 전략을 제안함
            decision = self._orchestrator.decide(
                workflow_id=workflow_id or "default",
                node_id=node_id,
                instruction=instruction,
                params=params or {},
            )
            
            # 전략에 따른 Max Tokens 설정
            # 복잡한 작업일수록 더 긴 사고 과정(Chain of Thought)이 필요하므로 토큰 수 증가
            new_max_tokens = 1024  # 기본값
            if decision.strategy == ExecutionStrategy.CLOUD_HEAVY:
                new_max_tokens = 4096  # 복잡한 작업 (Thinking 모델 등)은 긴 출력 필요
            elif decision.strategy == ExecutionStrategy.CLOUD_LIGHT:
                new_max_tokens = 1024
            elif decision.strategy == ExecutionStrategy.LOCAL_MODEL:
                new_max_tokens = 1024
            
            # 모델 업데이트 (Max Tokens 적용)
            if self._current_agent:
                # 기존 모델 ID 유지하면서 max_tokens만 변경하여 새 모델 인스턴스 생성
                self._current_agent.model = get_model(self.model_id, max_tokens=new_max_tokens)
                logger.info(f"[Orchestrator] 전략: {decision.strategy.value}, Max Tokens: {new_max_tokens}로 조정")

            # 2. 동적 프롬프트 생성
            final_instruction = self._orchestrator.get_dynamic_system_prompt(
                node_id=node_id,
                base_instruction=instruction,
                step_number=0,
            )
            logger.info(f"[Orchestrator] 동적 프롬프트 적용 (원본 {len(instruction)}자 → {len(final_instruction)}자)")

        # 초기 스크린샷
        try:
            screenshot_image = self._current_agent.desktop.screenshot()
            step_filename = f"step-0"
            self._last_screenshot = (screenshot_image, step_filename)
        except Exception as e:
            logger.error(f"초기 스크린샷 실패: {e}")

        # Timing tracking
        last_step_time = time.time()

        def step_callback(memory_step: ActionStep, agent: LocalVisionAgent):
            nonlocal step_number, last_step_time

            # Calculate duration
            current_time = time.time()
            step_duration_ms = int((current_time - last_step_time) * 1000)
            last_step_time = current_time

            if memory_step.step_number is None:
                return

            step_number = memory_step.step_number

            if step_number > agent.max_steps:
                raise AgentStopException("Max steps reached")

            if self._should_stop:
                raise AgentStopException("Stopped by user")

            # 모델 출력 파싱
            model_output = (
                memory_step.model_output_message.content
                if memory_step.model_output_message
                else None
            )

            if isinstance(memory_step.error, AgentMaxStepsError):
                model_output = memory_step.action_output

            # Thought 추출 (여러 형식 지원)
            thought = None
            if model_output and (
                memory_step.error is None or isinstance(memory_step.error, AgentMaxStepsError)
            ):
                # ```로 분리된 경우
                if "```" in model_output:
                    # Action 블록 앞부분을 모두 thought로 간주 (Verification 포함)
                    thought = model_output.split("```")[0].replace("\nAction:\n", "").strip()
                else:
                    # ```가 없으면 전체를 thought로
                    thought = model_output.strip()

                # thought가 비어있으면 전체 출력 사용
                if not thought and model_output:
                    thought = model_output[:500]  # 최대 500자

            # Action 추출
            action_sequence = None
            if model_output:
                parts = model_output.split("```")
                if len(parts) > 1:
                    action_sequence = parts[1]
                    # action이 비어있으면 다음 부분 시도
                    if not action_sequence.strip() and len(parts) > 2:
                        action_sequence = parts[2]

            # Action 정리 (python 태그 제거)
            if action_sequence:
                action_sequence = action_sequence.strip()
                if action_sequence.startswith("python"):
                    action_sequence = action_sequence[6:].strip()
                elif action_sequence.startswith("py"):
                    action_sequence = action_sequence[2:].strip()

            # 스크린샷 처리
            time.sleep(2)  # 액션 후 대기

            screenshot_base64 = None
            if self._last_screenshot:
                image, _ = self._last_screenshot
                try:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    screenshot_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                except Exception as e:
                    logger.error(f"스크린샷 인코딩 오류: {e}")

            # 에러 추적 및 얼리 스탑핑 체크
            step_error = memory_step.error.message if memory_step.error else None

            if step_error:
                self._consecutive_errors += 1
                self._last_error = step_error
                logger.warning(f"[EarlyStop] 에러 발생 ({self._consecutive_errors}/{self.max_consecutive_errors}): {step_error}")

                # 연속 에러 임계값 도달 시 얼리 스탑
                if self._consecutive_errors >= self.max_consecutive_errors:
                    logger.error(f"[EarlyStop] 연속 {self._consecutive_errors}회 에러 발생, 얼리 스탑핑!")
                    raise EarlyStopException(
                        f"연속 {self._consecutive_errors}회 에러 발생으로 중단",
                        consecutive_errors=self._consecutive_errors,
                        last_error=self._last_error,
                    )
            else:
                # 성공 시 연속 에러 카운트 리셋
                if self._consecutive_errors > 0:
                    logger.info(f"[EarlyStop] 에러 복구, 연속 에러 카운트 리셋 (이전: {self._consecutive_errors})")
                self._consecutive_errors = 0

            # Observation 추출 (여러 속성 시도)
            observation = None
            if hasattr(memory_step, 'action_output') and memory_step.action_output:
                observation = str(memory_step.action_output)
            elif hasattr(memory_step, 'observations') and memory_step.observations:
                observation = str(memory_step.observations)
            elif hasattr(memory_step, 'tool_result') and memory_step.tool_result:
                observation = str(memory_step.tool_result)

            # Note: Rule-based 봇 감지 체크 제거됨
            # VLM이 스크린샷을 보고 직접 [ERROR:BOT_DETECTED] 형식으로 보고함

            # Orchestrator 스텝 평가 (연동된 경우)
            step_feedback_dict = None
            if self._orchestrator and self._workflow_id and self._node_id:
                eval_start = time.time()
                feedback = self._orchestrator.evaluate_step(
                    workflow_id=self._workflow_id,
                    node_id=self._node_id,
                    step_number=step_number,
                    thought=thought,
                    action=action_sequence,
                    observation=observation,
                )
                eval_time_ms = int((time.time() - eval_start) * 1000)

                # 피드백을 dict로 변환하여 저장
                step_feedback_dict = {
                    "action": feedback.action.value,
                    "reason": feedback.reason,
                    "injected_prompt": feedback.injected_prompt,
                    "learned_pattern": feedback.learned_pattern,
                    "next_step_hint": feedback.next_step_hint,
                }

                # 평가 결과 로그 (시간 포함)
                logger.info(
                    f"[Orchestrator] 스텝 {step_number} 평가: "
                    f"action={feedback.action.value}, "
                    f"reason='{feedback.reason}', "
                    f"eval_time={eval_time_ms}ms"
                )

                # 피드백에 따른 처리
                if feedback.action == StepAction.STOP:
                    logger.error(f"[Orchestrator] 스텝 중단 지시: {feedback.reason}")
                    if feedback.save_to_memory:
                        self._pending_memory_save = feedback.save_to_memory
                    raise BotDetectionException(
                        f"Orchestrator 중단: {feedback.reason}",
                        detected_pattern=feedback.learned_pattern or "unknown",
                    )
                elif feedback.action == StepAction.INJECT_PROMPT:
                    # 다음 스텝에 프롬프트 주입 예약
                    if feedback.injected_prompt:
                        self._injected_prompt = feedback.injected_prompt
                        logger.info(f"[Orchestrator] 프롬프트 주입 예약: {feedback.injected_prompt[:50]}...")

                # 학습 (시간 측정)
                learn_start = time.time()
                self._orchestrator.learn_from_step(
                    node_id=self._node_id,
                    step_number=step_number,
                    success=step_error is None,
                    thought=thought,
                    action=action_sequence,
                    observation=observation,
                )
                learn_time_ms = int((time.time() - learn_start) * 1000)
                if learn_time_ms > 10:  # 10ms 이상일 때만 로그
                    logger.debug(f"[Orchestrator] 학습 저장: {learn_time_ms}ms")

            # 현재 상태 저장 (외부에서 조회용)
            self._current_thought = thought
            self._current_action = action_sequence
            self._current_observation = observation

            # 추론 로그 저장 (전체 모델 출력 포함)
            self._reasoning_logs.append({
                "step_number": step_number,
                "timestamp": datetime.now().isoformat(),
                "model_raw_output": model_output,  # 모델의 원본 출력
                "thought": thought,
                "action": action_sequence,
                "observation": observation,
                "error": step_error,
                "orchestrator_feedback": step_feedback_dict,
                "duration_ms": step_duration_ms,
            })

            # 스텝 로그 생성 (screenshot_after는 아직 None)
            step_log = VLMStepLog(
                step_number=step_number,
                timestamp=datetime.now().isoformat(),
                screenshot=screenshot_base64,
                thought=thought,
                action=action_sequence,
                observation=observation,
                error=step_error,
                tool_calls=parse_function_call(action_sequence) if action_sequence else [],
                orchestrator_feedback=step_feedback_dict,
                duration_ms=step_duration_ms,
                # 에이전트/모델 정보
                agent_type="VLMAgent",
                model_id=self.model_id,
                node_name=self._node_id,
            )

            # 다음 스크린샷 (Action After)
            try:
                screenshot_image = agent.desktop.screenshot()
                logger.info(f"Screenshot captured: {screenshot_image.size}")

                # Click Visualization (액션 위치 표시)
                if hasattr(agent, "click_coordinates") and agent.click_coordinates:
                    try:
                        draw = ImageDraw.Draw(screenshot_image)
                        cx, cy = agent.click_coordinates
                        radius = 15
                        # 빨간색 원과 십자선 그리기
                        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="red", width=3)
                        draw.line((cx - 25, cy, cx + 25, cy), fill="red", width=2)
                        draw.line((cx, cy - 25, cx, cy + 25), fill="red", width=2)
                    except Exception as e:
                        logger.error(f"클릭 시각화 실패: {e}")
                    finally:
                        # 다음 스텝을 위해 초기화
                        agent.click_coordinates = None

                # BBox Visualization (입력 영역 표시)
                if hasattr(agent, "last_action_bbox") and agent.last_action_bbox:
                    try:
                        draw = ImageDraw.Draw(screenshot_image)
                        bbox = agent.last_action_bbox
                        x, y = bbox["x"], bbox["y"]
                        w, h = bbox["width"], bbox["height"]
                        
                        # 초록색 사각형 그리기
                        draw.rectangle((x, y, x + w, y + h), outline="green", width=3)
                        
                        # "Input" 텍스트 표시 (선택 사항)
                        # draw.text((x, y - 15), "Input", fill="green")
                    except Exception as e:
                        logger.error(f"BBox 시각화 실패: {e}")
                    finally:
                        agent.last_action_bbox = None

                compressed = compress_image_to_max_size(screenshot_image, max_size_kb=300)
                
                # Base64 변환 for screenshot_after
                buffered = BytesIO()
                compressed.save(buffered, format="PNG")
                step_log.screenshot_after = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

                step_filename = f"step-{step_number + 1}"

                # 메모리 정리
                for prev_step in agent.memory.steps:
                    if isinstance(prev_step, ActionStep):
                        prev_step.observations_images = None

                memory_step.observations_images = [compressed.copy()]
                self._last_screenshot = (compressed, step_filename)
            except Exception as e:
                logger.error(f"스크린샷 업데이트 오류: {e}")

            # 로그 저장 및 콜백 (스크린샷 확보 후)
            self._steps.append(step_log)

            if on_step:
                on_step(step_log)

        # 에이전트에 콜백 추가 (smolagents CallbackRegistry 사용)
        self._current_agent.step_callbacks.register(ActionStep, step_callback)

        result: Optional[VLMInstructionResult] = None

        try:
            # 초기 스크린샷 캡처 및 전달 (첫 스텝에서 화면을 볼 수 있도록)
            initial_images = []
            try:
                if hasattr(self._current_agent, "desktop"):
                    screenshot = self._current_agent.desktop.screenshot()
                    compressed = compress_image_to_max_size(screenshot, max_size_kb=300)
                    initial_images = [compressed]
                    
                    # _last_screenshot 업데이트 (시각화 등을 위해)
                    self._last_screenshot = (compressed, "step-0")
            except Exception as e:
                logger.error(f"초기 스크린샷 캡처 실패: {e}")

            # 에이전트 실행 (동적 프롬프트 적용)
            agent_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._current_agent.run, 
                    final_instruction, 
                    images=initial_images
                ),
                timeout=600,  # 10분 타임아웃
            )

            # final_answer 결과 처리
            final_answer_text = str(agent_result) if agent_result else ""

            # VLM이 스크린샷 기반으로 감지한 에러 패턴 체크
            # [ERROR:BOT_DETECTED], [ERROR:PAGE_FAILED], [ERROR:ACCESS_DENIED] 등
            vlm_error_type = self._check_vlm_error_report(final_answer_text)

            if vlm_error_type:
                logger.warning(f"[VLM Error] VLM이 에러 감지: type={vlm_error_type}, msg='{final_answer_text[:100]}...'")
                result = VLMInstructionResult(
                    success=False,
                    error=f"VLM 에러 감지: {vlm_error_type}",
                    data={"steps_count": len(self._steps), "final_answer": final_answer_text, "error_type": vlm_error_type},
                    steps=self._steps,
                    early_stopped=True,
                    early_stop_reason=f"VLM 에러 감지: {vlm_error_type}",
                )
            else:
                result = VLMInstructionResult(
                    success=True,
                    data={"steps_count": len(self._steps), "final_answer": final_answer_text},
                    steps=self._steps,
                )

        except asyncio.TimeoutError:
            logger.error("에이전트 실행 타임아웃")
            result = VLMInstructionResult(
                success=False,
                error="타임아웃",
                steps=self._steps,
            )

        except AgentStopException as e:
            reason = str(e)
            if reason == "Max steps reached":
                result = VLMInstructionResult(
                    success=True,
                    data={"max_steps_reached": True, "steps_count": len(self._steps)},
                    steps=self._steps,
                )
            else:
                result = VLMInstructionResult(
                    success=False,
                    error=reason,
                    steps=self._steps,
                )

        except EarlyStopException as e:
            logger.error(f"[EarlyStop] 얼리 스탑 발생: {e}")
            result = VLMInstructionResult(
                success=False,
                error=str(e),
                steps=self._steps,
                consecutive_errors=e.consecutive_errors,
                early_stopped=True,
                early_stop_reason=f"연속 {e.consecutive_errors}회 에러: {e.last_error}",
            )

        except BotDetectionException as e:
            logger.error(f"[BotDetection] 봇 감지로 즉시 중단: {e}")

            # Save failure pattern to memory if requested
            if self._pending_memory_save and self._orchestrator and self._workflow_id and self._node_id:
                try:
                    await self._orchestrator.save_failure_to_memory(
                        self._workflow_id,
                        self._node_id,
                        self._pending_memory_save["pattern"],
                        self._pending_memory_save["reason"]
                    )
                    logger.info(f"[Memory] 실패 패턴 저장됨: {self._pending_memory_save}")
                except Exception as me:
                    logger.warning(f"[Memory] 실패 패턴 저장 실패: {me}")

            result = VLMInstructionResult(
                success=False,
                error=str(e),
                steps=self._steps,
                early_stopped=True,
                early_stop_reason=f"봇 감지: {e.detected_pattern}",
            )

        except Exception as e:
            import traceback
            logger.error(f"에이전트 실행 오류: {traceback.format_exc()}")
            result = VLMInstructionResult(
                success=False,
                error=str(e),
                steps=self._steps,
            )

        # 성공한 결과를 캐시에 저장 (reusable이 true일 때)
        if result and result.success and reusable and workflow_id and node_id:
            try:
                trace_store = get_trace_store()
                steps_data = [
                    {
                        "step_number": s.step_number,
                        "timestamp": s.timestamp,
                        "screenshot": s.screenshot,
                        "thought": s.thought,
                        "action": s.action,
                        "observation": s.observation,
                        "error": s.error,
                        "tool_calls": s.tool_calls,
                    }
                    for s in result.steps
                ]

                cache_key = trace_store.generate_cache_key(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    params=params or {},
                    key_params=cache_key_params,
                )

                trace_store.save_trace(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    cache_key=cache_key,
                    success=True,
                    steps=steps_data,
                    data=result.data,
                )
                logger.info(f"[TraceSave] {node_id}: trace 저장 완료 (cache_key: {cache_key})")
            except Exception as e:
                logger.error(f"[TraceSave] trace 저장 실패: {e}")

        # CallbackRegistry는 unregister 메서드가 없으므로 내부 리스트에서 제거
        try:
            if ActionStep in self._current_agent.step_callbacks._callbacks:
                callbacks = self._current_agent.step_callbacks._callbacks[ActionStep]
                if step_callback in callbacks:
                    callbacks.remove(step_callback)
        except Exception as e:
            logger.debug(f"콜백 정리 실패 (무시 가능): {e}")

        return result

    def stop(self):
        """에이전트 중지"""
        self._should_stop = True

    async def cleanup(self):
        """리소스 정리"""
        if self._session_id:
            await self._sandbox_service.release_sandbox(self._session_id)
        self._current_sandbox = None
        self._current_agent = None
        self._session_id = None
        self._steps = []
        self._last_screenshot = None

    def get_steps(self) -> List[VLMStepLog]:
        """현재까지의 스텝 로그 반환"""
        return self._steps.copy()

    def get_last_screenshot(self) -> Optional[str]:
        """마지막 스크린샷 반환 (base64)"""
        if self._last_screenshot:
            image, _ = self._last_screenshot
            try:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            except Exception:
                pass
        return None

    def get_current_state(self) -> Dict[str, Any]:
        """현재 에이전트 상태 반환 (실시간 모니터링용)"""
        return {
            "consecutive_errors": self._consecutive_errors,
            "last_error": self._last_error,
            "current_thought": self._current_thought,
            "current_action": self._current_action,
            "current_observation": self._current_observation,
            "step_count": len(self._steps),
            "max_consecutive_errors": self.max_consecutive_errors,
        }

    def get_reasoning_logs(self) -> List[Dict[str, Any]]:
        """
        전체 추론 로그 반환 (모델의 원본 출력 포함)

        비효율적인 추론 패턴 분석용으로 사용
        """
        return self._reasoning_logs.copy()

    def get_reasoning_logs_text(self) -> str:
        """추론 로그를 텍스트 형식으로 반환 (다운로드용)"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"VLM 추론 로그 - {self._node_id or 'Unknown Node'}")
        lines.append(f"모델: {self.model_id}")
        lines.append(f"총 스텝: {len(self._reasoning_logs)}")
        lines.append("=" * 80)
        lines.append("")

        for log in self._reasoning_logs:
            lines.append(f"--- Step {log['step_number']} ({log['timestamp']}) ---")
            lines.append(f"Duration: {log.get('duration_ms', 'N/A')}ms")
            lines.append("")

            # 모델 원본 출력
            if log.get("model_raw_output"):
                lines.append("[Model Raw Output]")
                lines.append(log["model_raw_output"])
                lines.append("")

            # 추출된 정보
            if log.get("thought"):
                lines.append("[Thought]")
                lines.append(log["thought"])
                lines.append("")

            if log.get("action"):
                lines.append("[Action]")
                lines.append(log["action"])
                lines.append("")

            if log.get("observation"):
                lines.append("[Observation]")
                lines.append(str(log["observation"]))
                lines.append("")

            if log.get("error"):
                lines.append("[Error]")
                lines.append(log["error"])
                lines.append("")

            if log.get("orchestrator_feedback"):
                lines.append("[Orchestrator Feedback]")
                lines.append(str(log["orchestrator_feedback"]))
                lines.append("")

            lines.append("")

        return "\n".join(lines)

