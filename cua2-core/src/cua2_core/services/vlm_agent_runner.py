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

from PIL import Image
from smolagents import ActionStep, AgentMaxStepsError

from cua2_core.services.agent_utils.get_model import get_model
from cua2_core.services.agent_utils.local_desktop_agent import LocalVisionAgent
from cua2_core.services.agent_utils.function_parser import parse_function_call
from cua2_core.services.local_sandbox_service import LocalSandboxService
from cua2_core.services.utils import compress_image_to_max_size
from cua2_core.services.trace_store import get_trace_store

logger = logging.getLogger(__name__)


@dataclass
class VLMStepLog:
    """VLM 스텝 로그"""
    step_number: int
    timestamp: str
    screenshot: Optional[str] = None  # base64 encoded
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VLMInstructionResult:
    """VLM 명령 실행 결과"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    steps: List[VLMStepLog] = field(default_factory=list)


class AgentStopException(Exception):
    """에이전트 중지 예외"""
    pass


class VLMAgentRunner:
    """
    VLM 에이전트 실행기 - 워크플로우에서 사용

    LocalAgentService의 핵심 로직을 워크플로우에서 사용할 수 있도록 추상화
    """

    def __init__(
        self,
        model_id: str = "local-qwen3-vl",
        max_steps: int = 15,
        data_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.max_steps = max_steps
        self.data_dir = data_dir or "/tmp/vlm_agent_runner"

        self._sandbox_service = LocalSandboxService(max_sandboxes=1)
        self._current_sandbox = None
        self._current_agent: Optional[LocalVisionAgent] = None
        self._session_id: Optional[str] = None
        self._steps: List[VLMStepLog] = []
        self._last_screenshot: Optional[tuple[Image.Image, str]] = None
        self._should_stop = False

    async def initialize(self, session_id: str) -> bool:
        """샌드박스와 에이전트 초기화"""
        self._session_id = session_id
        self._should_stop = False

        # 데이터 디렉토리 생성
        session_data_dir = os.path.join(self.data_dir, session_id)
        os.makedirs(session_data_dir, exist_ok=True)

        # 샌드박스 획득
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

        # 에이전트 생성
        self._current_agent = LocalVisionAgent(
            model=model,
            data_dir=session_data_dir,
            desktop=self._current_sandbox,
            max_steps=self.max_steps,
        )

        logger.info(f"VLM 에이전트 초기화 완료: {session_id}")
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

        # 초기 스크린샷
        try:
            screenshot_image = self._current_agent.desktop.screenshot()
            step_filename = f"step-0"
            self._last_screenshot = (screenshot_image, step_filename)
        except Exception as e:
            logger.error(f"초기 스크린샷 실패: {e}")

        def step_callback(memory_step: ActionStep, agent: LocalVisionAgent):
            nonlocal step_number

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

            # Thought 추출
            thought = None
            if model_output and (
                memory_step.error is None or isinstance(memory_step.error, AgentMaxStepsError)
            ):
                thought = model_output.split("```")[0].replace("\nAction:\n", "").strip()

            # Action 추출
            action_sequence = None
            if model_output:
                parts = model_output.split("```")
                if len(parts) > 1:
                    action_sequence = parts[1]

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

            # 스텝 로그 생성
            step_log = VLMStepLog(
                step_number=step_number,
                timestamp=datetime.now().isoformat(),
                screenshot=screenshot_base64,
                thought=thought,
                action=action_sequence,
                observation=memory_step.action_output if hasattr(memory_step, 'action_output') else None,
                error=memory_step.error.message if memory_step.error else None,
                tool_calls=parse_function_call(action_sequence) if action_sequence else [],
            )

            self._steps.append(step_log)

            if on_step:
                on_step(step_log)

            # 다음 스크린샷
            try:
                screenshot_image = agent.desktop.screenshot()
                compressed = compress_image_to_max_size(screenshot_image, max_size_kb=500)
                step_filename = f"step-{step_number + 1}"

                # 메모리 정리
                for prev_step in agent.memory.steps:
                    if isinstance(prev_step, ActionStep):
                        prev_step.observations_images = None

                memory_step.observations_images = [compressed.copy()]
                self._last_screenshot = (compressed, step_filename)
            except Exception as e:
                logger.error(f"스크린샷 업데이트 오류: {e}")

        # 에이전트에 콜백 추가 (smolagents CallbackRegistry 사용)
        self._current_agent.step_callbacks.register(ActionStep, step_callback)

        result: Optional[VLMInstructionResult] = None

        try:
            # 에이전트 실행
            await asyncio.wait_for(
                asyncio.to_thread(self._current_agent.run, instruction),
                timeout=600,  # 10분 타임아웃
            )

            result = VLMInstructionResult(
                success=True,
                data={"steps_count": len(self._steps)},
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
