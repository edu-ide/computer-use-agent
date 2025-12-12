"""
Fara WebSurfer - Fara-7B 모델 전용 웹 서핑 에이전트

PlaywrightController를 사용하여 웹 브라우저를 제어하고,
Fara-7B 모델의 출력을 파싱하여 액션을 실행.
"""

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx
from PIL import Image

from .playwright_controller import PlaywrightController
from .prompts import FaraComputerUse, get_computer_use_system_prompt


@dataclass
class WebSurferConfig:
    """WebSurfer 설정."""
    # 뷰포트 설정
    viewport_width: int = 1440
    viewport_height: int = 900
    # MLM 이미지 크기
    mlm_width: int = 1440
    mlm_height: int = 900
    # 히스토리 설정
    max_n_images: int = 3
    max_rounds: int = 100
    # 모델 설정
    model_call_timeout: int = 60
    # 시작 페이지
    start_page: str = "https://www.google.com/"
    # 다운로드 폴더
    downloads_folder: str = "/tmp/downloads"


class FaraWebSurfer:
    """
    Fara-7B 모델 전용 웹 서핑 에이전트.

    - 스크린샷 기반 웹 페이지 분석
    - Fara-7B 모델의 tool call 출력 파싱
    - PlaywrightController를 통한 브라우저 제어
    """

    DEFAULT_DESCRIPTION = (
        "웹 브라우저에 접근할 수 있는 도움되는 어시스턴트입니다. "
        "웹 검색, 페이지 열기, 콘텐츠 상호작용(링크 클릭, 스크롤, 폼 입력 등)을 수행합니다. "
        "페이지 요약 또는 페이지 콘텐츠 기반 질문 답변도 가능합니다."
    )

    def __init__(
        self,
        config: Optional[WebSurferConfig] = None,
        llm_base_url: str = "http://localhost:30001/v1",
        llm_model: str = "/mnt/sda1/models/llm/GELab-Zero-4B-preview",
        llm_api_key: str = "EMPTY",
    ):
        self.config = config or WebSurferConfig()
        self.logger = logging.getLogger(__name__)

        # LLM 클라이언트
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        
        # litellm 대신 httpx 직접 사용
        import httpx
        self._client = httpx.AsyncClient(timeout=self.config.model_call_timeout)

        # PlaywrightController
        self._controller: Optional[PlaywrightController] = None

        # 히스토리
        self._action_history: List[Dict[str, Any]] = []
        self._message_history: List[Dict[str, Any]] = []
        self._round_count: int = 0

    async def initialize(self, page=None) -> None:
        """브라우저 초기화."""
        self._controller = PlaywrightController(
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
        )

        if page:
            await self._controller.set_page(page)
        else:
            await self._controller.start()
            await self._controller.visit_url(self.config.start_page)

    async def close(self) -> None:
        """리소스 정리."""
        if self._controller:
            await self._controller.close()
        await self._client.aclose()

    async def execute_task(self, task: str) -> Dict[str, Any]:
        """
        주어진 작업을 실행.

        Args:
            task: 수행할 작업 설명

        Returns:
            실행 결과
        """
        if not self._controller:
            await self.initialize()

        self._round_count = 0
        self._action_history = []

        while self._round_count < self.config.max_rounds:
            self._round_count += 1

            # 스크린샷 촬영
            screenshot = await self._controller.screenshot()
            if screenshot is None:
                return {"status": "error", "message": "스크린샷 실패"}

            # 모델 호출
            action = await self._get_next_action(task, screenshot)

            if action is None:
                return {"status": "error", "message": "액션 파싱 실패"}

            # 완료 체크
            if action.get("name") == "done":
                return {
                    "status": "success",
                    "message": action.get("arguments", {}).get("message", "완료"),
                    "rounds": self._round_count,
                    "actions": self._action_history,
                }

            # 액션 실행
            result = await self._execute_action(action)
            self._action_history.append({
                "action": action,
                "result": result,
                "round": self._round_count,
            })

            if result.get("status") == "error":
                # 에러 시 재시도 또는 중단
                self.logger.warning(f"액션 실행 실패: {result.get('message')}")

        return {
            "status": "timeout",
            "message": f"최대 라운드({self.config.max_rounds}) 도달",
            "rounds": self._round_count,
            "actions": self._action_history,
        }

    async def _get_next_action(
        self, task: str, screenshot: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """
        다음 액션 결정.

        스크린샷과 작업 설명을 Fara-7B 모델에 전달하고
        다음 수행할 액션을 결정.
        """
        # 시스템 프롬프트 생성
        system_prompt_data = get_computer_use_system_prompt(
            image=screenshot,
            processor_im_cfg={
                "min_pixels": 3136,
                # 이미지 토큰 수 제한을 위해 해상도 초소형으로 축소 (360*360)
                # 시스템 프롬프트가 매우 길어서(5800자+) 이미지 여유 공간이 거의 없음
                "max_pixels": 360 * 360, 
                "patch_size": 14,
                "merge_size": 2,
            },
        )
        
        # [중요] 계산된 크기로 이미지 실제로 리사이징
        # 이를 수행하지 않으면 원본 고해상도 이미지가 전송되어 토큰 초과 발생
        target_w, target_h = system_prompt_data["im_size"]
        if (target_w, target_h) != screenshot.size:
            self.logger.info(f"Resizing screenshot from {screenshot.size} to {(target_w, target_h)}")
            screenshot = screenshot.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 스크린샷을 base64로 인코딩
        img_buffer = io.BytesIO()
        screenshot.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        # 시스템 프롬프트 추출 및 문자열 변환
        
        # 시스템 프롬프트 추출 및 문자열 변환 (SGLang 호환성)
        raw_content = system_prompt_data.get("conversation", [{}])[0].get("content", "")
        system_prompt = ""
        
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    system_prompt += item["text"]
        elif isinstance(raw_content, str):
            system_prompt = raw_content
            
        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # 히스토리 추가 (최근 N개만)
        for hist in self._message_history[-self.config.max_n_images:]:
            messages.append(hist)

        # 현재 작업과 스크린샷
        user_content = [
            {"type": "text", "text": f"Task: {task}\n\nHere is the current screenshot. What action should I take next?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
        ]
        messages.append({"role": "user", "content": user_content})

        # 모델 호출
        try:
            self.logger.info(f"Sending request to model (System prompt len: {len(system_prompt)})")
            
            response = await self._client.post(
                f"{self.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.1, # 낮은 temperature로 결정적이고 빠른 출력
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            self.logger.info(f"Model Raw Response: {content}")
            print(f"\n[DEBUG] Model Raw Response:\n{content}\n[DEBUG END]\n")

            # 응답 파싱
            thoughts, action = self._parse_thoughts_and_action(content)
            
            if not action:
                print(f"[DEBUG] Action parsing failed for content: {content[:100]}...")
                return None

            # 히스토리에 추가
            self._message_history.append({"role": "assistant", "content": content})

            return action

        except Exception as e:
            self.logger.error(f"모델 호출 실패: {e}")
            return None

    def _parse_thoughts_and_action(
        self, message: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Fara-7B 출력 파싱.

        Fara-7B는 다양한 형식으로 응답할 수 있음:
        1. 바로 JSON: {"action": "type", "text": "...", "coordinate": [...]}
        2. <tool_call> 태그 포함: <tool_call>{"action": ...}</tool_call>
        3. JSON 코드 블록: ```json {...} ```
        """
        thoughts = ""
        action = None

        try:
            # thoughts 추출 (있는 경우)
            if "<thoughts>" in message and "</thoughts>" in message:
                start = message.find("<thoughts>") + len("<thoughts>")
                end = message.find("</thoughts>")
                thoughts = message[start:end].strip()

            # 방법 1: tool_call 태그 내의 JSON
            if "<tool_call>" in message:
                if "</tool_call>" in message:
                    start = message.find("<tool_call>") + len("<tool_call>")
                    end = message.find("</tool_call>")
                    action_text = message[start:end].strip()
                else:
                    # 불완전한 tool_call 태그 - 태그 앞의 JSON을 찾음
                    start = message.find("<tool_call>")
                    action_text = message[:start].strip()
                if action_text:
                    raw_action = json.loads(action_text)
                    action = self._normalize_action(raw_action)

            # 방법 2: JSON 코드 블록
            elif "```json" in message:
                start = message.find("```json") + 7
                end = message.find("```", start)
                action_text = message[start:end].strip()
                raw_action = json.loads(action_text)
                action = self._normalize_action(raw_action)

            # 방법 3: 바로 JSON 객체로 시작
            elif message.strip().startswith("{"):
                # JSON 끝까지 찾기
                brace_count = 0
                json_end = 0
                for i, char in enumerate(message):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                action_text = message[:json_end].strip()
                raw_action = json.loads(action_text)
                action = self._normalize_action(raw_action)
            
            # 방법 4 (Fallback): 텍스트 내에 포함된 첫 번째 JSON 객체 찾기 (Regex)
            else:
                import re
                # 가장 바깥쪽 중괄호 쌍을 찾음 (Greedy 하지 않게)
                # 주의: 중첩된 JSON은 단순 regex로 완벽하지 않지만, 일반적인 단일 액션에는 충분함
                json_match = re.search(r'\{[\s\S]*\}', message)
                if json_match:
                    try:
                        potential_json = json_match.group(0)
                        # JSON 파싱 시도
                        raw_action = json.loads(potential_json)
                        # action 키가 있는지 확인하여 유효성 검증
                        if "action" in raw_action:
                            action = self._normalize_action(raw_action)
                            # thoughts가 아직 없다면 JSON 앞부분을 thoughts로 간주
                            if not thoughts:
                                thoughts = message[:json_match.start()].strip()
                    except json.JSONDecodeError:
                        pass # Regex 매칭이 유효한 JSON이 아니면 무시

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패: {e}, message: {message[:200]}")
        except Exception as e:
            self.logger.warning(f"파싱 실패: {e}")

        return thoughts, action

    def _normalize_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fara-7B 액션 형식을 표준 형식으로 변환.

        Fara 형식: {"action": "type", "text": "...", "coordinate": [...]}
        또는 중첩 형식: {"name": "computer_use", "arguments": {"action": "type", ...}}
        표준 형식: {"name": "type", "arguments": {"text": "...", "x": ..., "y": ...}}
        """
        # 중첩된 arguments 구조 처리 (Fara-7B의 tool_call 형식)
        if raw_action.get("name") == "computer_use" and "arguments" in raw_action:
            raw_action = raw_action["arguments"]

        action_type = raw_action.get("action", "")

        # 액션 타입 매핑
        action_map = {
            "type": "type",
            "left_click": "click",
            "mouse_move": "click",
            "scroll": "scroll",
            "visit_url": "goto",
            "web_search": "goto",
            "wait": "wait",
            "key": "key",
            "terminate": "done",
        }

        name = action_map.get(action_type, action_type)

        # 인자 구성
        arguments = {}

        if name == "type":
            arguments["text"] = raw_action.get("text", "")
            if raw_action.get("press_enter"):
                arguments["press_enter"] = True
            # 좌표가 있으면 추가
            coord = raw_action.get("coordinate", [])
            if coord and len(coord) >= 2:
                arguments["x"] = coord[0]
                arguments["y"] = coord[1]
            if raw_action.get("delete_existing_text"):
                arguments["delete_existing_text"] = True

        elif name == "click":
            coord = raw_action.get("coordinate", [0, 0])
            arguments["x"] = coord[0] if len(coord) > 0 else 0
            arguments["y"] = coord[1] if len(coord) > 1 else 0

        elif name == "scroll":
            pixels = raw_action.get("pixels", -400)
            arguments["direction"] = "up" if pixels > 0 else "down"

        elif name == "goto":
            arguments["url"] = raw_action.get("url", raw_action.get("query", ""))

        elif name == "wait":
            arguments["seconds"] = raw_action.get("time", 2)

        elif name == "key":
            keys = raw_action.get("keys", [])
            arguments["key"] = keys[0] if keys else "Enter"

        elif name == "done":
            arguments["message"] = raw_action.get("status", "완료")

        return {"name": name, "arguments": arguments}

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        액션 실행.

        Fara 도구 형식:
        - click: 좌표 클릭
        - type: 텍스트 입력
        - scroll: 스크롤
        - goto: URL 이동
        - wait: 대기
        - done: 완료
        """
        name = action.get("name", "")
        args = action.get("arguments", {})

        try:
            if name == "click":
                x = args.get("x", 0)
                y = args.get("y", 0)
                await self._controller._click_coords_stateful(x, y)
                return {"status": "success", "action": "click", "x": x, "y": y}

            elif name == "type":
                text = args.get("text", "")
                
                # [Workaround] 모델이 "가방"을 "기럽"(\uae30\ub7fd)으로 잘못 출력하는 환각 현상 수정
                # 특정 모델 버전에 종속적인 이슈로 보임
                if text == "\uae30\ub7fd" or text == "기럽":
                    self.logger.warning(f"Detected garbled text '{text}', correcting to '가방'")
                    text = "가방"
                
                # 좌표가 있으면 먼저 클릭
                if "x" in args and "y" in args:
                    x = args["x"]
                    y = args["y"]
                    await self._controller._click_coords_stateful(x, y)
                    # 약간 대기
                    await asyncio.sleep(0.2)
                
                # 기존 텍스트 삭제 요청 시
                if args.get("delete_existing_text"):
                    # Ctrl+A 후 Backspace 입력
                    # PlaywrightController.send_keys는 리스트를 받아 조합키 처리
                    try:
                        await self._controller.send_keys(["Control", "A"])
                        await asyncio.sleep(0.1)
                        await self._controller.send_keys("Backspace")
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        self.logger.warning(f"텍스트 삭제 실패: {e}")

                await self._controller.type_text(text)
                if args.get("press_enter"):
                    await self._controller.send_keys("Enter")
                    # 페이지 로딩 등 대기 시간을 위해 충분히 기다림
                    await asyncio.sleep(3.0)
                
                return {"status": "success", "action": "type", "text": text}

            elif name == "scroll":
                direction = args.get("direction", "down")
                if direction == "down":
                    await self._controller._page_down_stateful()
                else:
                    await self._controller._page_up_stateful()
                return {"status": "success", "action": "scroll", "direction": direction}

            elif name == "goto":
                url = args.get("url", "")
                await self._controller.visit_url(url)
                return {"status": "success", "action": "goto", "url": url}

            elif name == "wait":
                seconds = args.get("seconds", 2)
                await asyncio.sleep(seconds)
                return {"status": "success", "action": "wait", "seconds": seconds}

            elif name == "key":
                key = args.get("key", "")
                await self._controller.send_keys(key)
                return {"status": "success", "action": "key", "key": key}

            elif name == "done":
                return {"status": "done", "message": args.get("message", "완료")}

            else:
                return {"status": "error", "message": f"알 수 없는 액션: {name}"}

        except Exception as e:
            return {"status": "error", "message": str(e), "action": name}

    async def get_page_info(self) -> Dict[str, Any]:
        """현재 페이지 정보 반환."""
        if not self._controller:
            return {"status": "error", "message": "초기화되지 않음"}

        page = self._controller._page
        if not page:
            return {"status": "error", "message": "페이지 없음"}

        return {
            "url": page.url,
            "title": await page.title(),
        }

    def get_action_history(self) -> List[Dict[str, Any]]:
        """액션 히스토리 반환."""
        return self._action_history.copy()
