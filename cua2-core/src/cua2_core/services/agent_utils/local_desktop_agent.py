"""
로컬 데스크톱 에이전트 - E2B 대신 로컬 pyautogui 사용
"""

import os
import time
import unicodedata

from cua2_core.services.agent_utils.prompt import E2B_SYSTEM_PROMPT_TEMPLATE
from cua2_core.services.local_desktop import LocalDesktop

from smolagents import CodeAgent, Model, tool
from smolagents.monitoring import LogLevel


class LocalVisionAgent(CodeAgent):
    """로컬 데스크톱 자동화 에이전트 with Qwen3-VL 비전"""

    def __init__(
        self,
        model: Model,
        data_dir: str,
        desktop: LocalDesktop,
        max_steps: int = 30,
        verbosity_level: LogLevel = 2,
        planning_interval: int | None = None,
        use_v1_prompt: bool = False,
        qwen_normalization: bool = True,
        **kwargs,
    ):
        self.desktop = desktop
        self.data_dir = data_dir
        self.planning_interval = planning_interval
        self.qwen_normalization = qwen_normalization

        # 화면 크기 가져오기
        self.width, self.height = self.desktop.get_screen_size()
        print(f"화면 크기: {self.width}x{self.height}")

        # 임시 디렉토리 설정
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"스크린샷 저장 경로: {self.data_dir}")

        self.use_v1_prompt = use_v1_prompt

        # 기본 에이전트 초기화
        super().__init__(
            tools=[],
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=self.planning_interval,
            stream_outputs=True,
            **kwargs,
        )

        self.prompt_templates["system_prompt"] = E2B_SYSTEM_PROMPT_TEMPLATE.replace(
            "<<resolution_x>>", str(self.width)
        ).replace("<<resolution_y>>", str(self.height))

        # 상태에 화면 정보 추가
        self.state["screen_width"] = self.width
        self.state["screen_height"] = self.height

        # 도구 설정
        self.logger.log("에이전트 도구 설정 중...")
        self._setup_desktop_tools()

        # CodeAgent의 python_executor 래핑 (잘못된 코드 수정용)
        # 모델이 "python open_url(...)" 처럼 출력하거나,
        # "open_url(...) wait(...)" 처럼 한 줄에 여러 명령을 쓰는 경우 수정
        original_executor = self.python_executor

        self.python_executor = SanitizedExecutorProxy(original_executor)

        self.python_executor = SanitizedExecutorProxy(original_executor)

    def _qwen_unnormalization(self, arguments: dict[str, int]) -> dict[str, int]:
        """
        좌표를 0-999 범위에서 실제 픽셀 좌표로 변환
        """
        unnormalized: dict[str, int] = {}
        for key, value in arguments.items():
            if "x" in key.lower() and "y" not in key.lower():
                unnormalized[key] = int((value / 1000) * self.width)
            elif "y" in key.lower():
                unnormalized[key] = int((value / 1000) * self.height)
            else:
                unnormalized[key] = value
        return unnormalized

    def _setup_desktop_tools(self):
        """데스크톱 도구 등록"""

        @tool
        def click(x: int, y: int) -> str:
            """
            지정된 좌표에서 왼쪽 클릭 수행
            Args:
                x: x 좌표 (가로 위치)
                y: y 좌표 (세로 위치)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                new_x, new_y = coords["x"], coords["y"]
                print(f"Coordinate Conversion: ({x}, {y}) -> ({new_x}, {new_y}) [Screen: {self.width}x{self.height}]")
                x, y = new_x, new_y
            
            self.desktop.move_mouse(x, y)
            self.desktop.left_click()
            time.sleep(0.5)  # UI 반응 대기 (배치 실행 시 안전성 확보)
            self.click_coordinates = [x, y]
            self.logger.log(f"클릭: ({x}, {y})")
            print(f"Executed: Clicked at ({x}, {y})")
            return f"Clicked at coordinates ({x}, {y})"

        @tool
        def click_pixels(x: int, y: int) -> str:
            """
            지정된 픽셀 좌표에서 왼쪽 클릭 수행 (좌표 변환 없음)
            Args:
                x: x 픽셀 좌표
                y: y 픽셀 좌표
            """
            self.desktop.move_mouse(x, y)
            self.desktop.left_click()
            time.sleep(0.5)
            self.click_coordinates = [x, y]
            self.logger.log(f"클릭(픽셀): ({x}, {y})")
            print(f"Executed: Clicked (pixels) at ({x}, {y})")
            return f"Clicked at pixel coordinates ({x}, {y})"

        @tool
        def right_click(x: int, y: int) -> str:
            """
            지정된 좌표에서 오른쪽 클릭 수행
            Args:
                x: x 좌표 (가로 위치)
                y: y 좌표 (세로 위치)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.right_click()
            time.sleep(0.5)  # UI 반응 대기
            self.click_coordinates = [x, y]
            self.logger.log(f"오른쪽 클릭: ({x}, {y})")
            print(f"Executed: Right-clicked at ({x}, {y})")
            return f"Right-clicked at coordinates ({x}, {y})"

        @tool
        def double_click(x: int, y: int) -> str:
            """
            지정된 좌표에서 더블 클릭 수행
            Args:
                x: x 좌표 (가로 위치)
                y: y 좌표 (세로 위치)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.double_click()
            time.sleep(0.5)  # UI 반응 대기
            self.click_coordinates = [x, y]
            self.logger.log(f"더블 클릭: ({x}, {y})")
            print(f"Executed: Double-clicked at ({x}, {y})")
            return f"Double-clicked at coordinates ({x}, {y})"

        @tool
        def move_mouse(x: int, y: int) -> str:
            """
            마우스 커서를 지정된 좌표로 이동
            Args:
                x: x 좌표 (가로 위치)
                y: y 좌표 (세로 위치)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.logger.log(f"마우스 이동: ({x}, {y})")
            print(f"Executed: Moved mouse to ({x}, {y})")
            return f"Moved mouse to coordinates ({x}, {y})"

        def normalize_text(text):
            return "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if not unicodedata.combining(c)
            )

        @tool
        def write(text: str) -> str:
            """
            현재 커서 위치에 텍스트 입력
            Args:
                text: 입력할 텍스트
            """
            self.desktop.write(text)
            self.logger.log(f"텍스트 입력: {text}")
            print(f"Executed: Typed '{text}'")
            
            # 입력 후 활성 요소의 BBox 가져오기 (시각화용)
            try:
                # 활성 요소의 위치와 크기 가져오기
                js_code = """
                (() => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return null;
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    };
                })()
                """
                # evaluate_script는 문자열로 결과를 반환하므로 파싱 필요
                # LocalDesktop.evaluate_script 구현에 따라 다를 수 있음
                # 여기서는 간단히 시도하고 실패하면 무시
                bbox_json = self.desktop.evaluate_script(js_code)
                
                import json
                if bbox_json and not bbox_json.startswith("Error") and not bbox_json.startswith("JS Exception"):
                    bbox = json.loads(bbox_json)
                    if bbox:
                        # 좌표 보정 (브라우저 내부 좌표 -> 스크린 좌표)
                        # 주의: 브라우저가 전체 화면이 아닐 수 있으므로 오차가 있을 수 있음
                        # 하지만 현재 환경(XFCE)에서는 브라우저가 보통 최대화되어 있다고 가정
                        # 정확한 매핑을 위해서는 브라우저 창 위치를 알아야 하지만, 
                        # 여기서는 간단히 브라우저 툴바 높이 등을 고려한 오프셋을 적용하거나
                        # 단순히 시각적 피드백용으로 사용
                        
                        # 상단 툴바/주소창 높이 대략 80px 가정 (필요시 조정)
                        browser_offset_y = 80 
                        
                        self.last_action_bbox = {
                            "x": bbox["x"],
                            "y": bbox["y"] + browser_offset_y,
                            "width": bbox["width"],
                            "height": bbox["height"]
                        }
            except Exception as e:
                print(f"BBox capture failed: {e}")
                self.last_action_bbox = None

            return f"Typed text: '{text}'"

        @tool
        def press(keys: list[str]) -> str:
            """
            키보드 키 입력
            Args:
                keys: 입력할 키 (예: ["enter", "space", "backspace"])
            """
            self.desktop.press(keys)
            self.logger.log(f"키 입력: {keys}")
            print(f"Executed: Pressed keys {keys}")
            return f"Pressed keys: {keys}"

        @tool
        def go_back() -> str:
            """
            브라우저에서 이전 페이지로 이동
            """
            self.desktop.press(["alt", "left"])
            self.logger.log("이전 페이지로 이동")
            print("Executed: Went back one page")
            return "Went back one page"

        @tool
        def drag(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            [x1, y1]에서 클릭 후 [x2, y2]로 드래그
            Args:
                x1: 시작 x 좌표
                y1: 시작 y 좌표
                x2: 끝 x 좌표
                y2: 끝 y 좌표
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                )
                x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            self.desktop.drag([x1, y1], [x2, y2])
            self.click_coordinates = [x2, y2]
            message = f"드래그: [{x1}, {y1}] → [{x2}, {y2}]"
            self.logger.log(message)
            print(f"Executed: Dragged from [{x1}, {y1}] to [{x2}, {y2}]")
            return f"Dragged from [{x1}, {y1}] to [{x2}, {y2}]"

        @tool
        def drag_pixels(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            [x1, y1]에서 [x2, y2]로 드래그 (픽셀 좌표)
            Args:
                x1: 시작 x 픽셀
                y1: 시작 y 픽셀
                x2: 끝 x 픽셀
                y2: 끝 y 픽셀
            """
            self.desktop.drag([x1, y1], [x2, y2])
            self.click_coordinates = [x2, y2]
            message = f"드래그(픽셀): [{x1}, {y1}] → [{x2}, {y2}]"
            self.logger.log(message)
            print(f"Executed: Dragged (pixels) from [{x1}, {y1}] to [{x2}, {y2}]")
            return f"Dragged from pixel [{x1}, {y1}] to [{x2}, {y2}]"

        @tool
        def scroll(x: int, y: int, direction: str = "down", amount: int = 2) -> str:
            """
            지정된 좌표에서 스크롤
            Args:
                x: x 좌표
                y: y 좌표
                direction: 스크롤 방향 ("up" 또는 "down")
                amount: 스크롤 양 (1 또는 2 권장)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.scroll(direction=direction, amount=amount)
            message = f"스크롤 {direction} ({amount})"
            self.logger.log(message)
            print(f"Executed: Scrolled {direction} by {amount}")
            return f"Scrolled {direction} by {amount}"

        @tool
        def wait(seconds: float) -> str:
            """
            지정된 시간만큼 대기
            Args:
                seconds: 대기 시간 (초), 보통 3초면 충분
            """
            time.sleep(seconds)
            self.logger.log(f"{seconds}초 대기")
            print(f"Executed: Waited for {seconds} seconds")
            return f"Waited for {seconds} seconds"

        @tool
        def open_url(url: str) -> str:
            """
            브라우저로 URL 열기
            Args:
                url: 열 URL
            """
            if not url.startswith("http") and not url.startswith("https"):
                url = f"https://{url}"
            self.desktop.open(url)
            time.sleep(2)
            self.logger.log(f"URL 열기: {url}")
            print(f"Executed: Opened URL {url}")
            return f"Opened URL: {url}"

        @tool
        def launch(app: str) -> str:
            """
            애플리케이션 실행
            Args:
                app: 실행할 애플리케이션
            """
            self.desktop.commands.run(f"{app}", background=True)
            print(f"Executed: Launched application {app}")
            return f"Launched application: {app}"

        @tool
        def run_javascript(script: str) -> str:
            """
            현재 페이지에서 자바스크립트 실행 (데이터 추출, 페이지 조작 등)
            Args:
                script: 실행할 자바스크립트 코드 (예: "document.title", "document.body.innerText")
            Returns:
                실행 결과
            """
            result = self.desktop.evaluate_script(script)
            self.logger.log(f"JS 실행: {script[:50]}... -> {result[:50]}...")
            print(f"Executed: Run JS '{script[:30]}...' -> '{result[:30]}...'")
            return f"JS Result: {result}"

        # 도구 등록
        self.tools["click"] = click
        self.tools["click_pixels"] = click_pixels
        self.tools["right_click"] = right_click
        self.tools["double_click"] = double_click
        self.tools["move_mouse"] = move_mouse
        self.tools["write"] = write
        self.tools["press"] = press
        self.tools["scroll"] = scroll
        self.tools["wait"] = wait
        self.tools["open_url"] = open_url
        self.tools["launch"] = launch
        self.tools["go_back"] = go_back
        self.tools["drag"] = drag
        self.tools["drag_pixels"] = drag_pixels
        self.tools["run_javascript"] = run_javascript

class SanitizedExecutorProxy:
    def __init__(self, executor):
        self.executor = executor
    
    def __getattr__(self, name):
        return getattr(self.executor, name)

    def __call__(self, code: str, *args, **kwargs):
        import re
        
        # 1. python/py 태그/접두사 제거
        cleaned_code = code.strip()
        if cleaned_code.startswith("python "):
            cleaned_code = cleaned_code[7:].strip()
        elif cleaned_code.startswith("py "):
            cleaned_code = cleaned_code[3:].strip()
        
        # 2. 한 줄에 여러 함수 호출이 있는 경우 세미콜론 삽입
        # 패턴: 닫는 괄호 ')' 뒤에 공백이 있고, 바로 알파벳(함수명)이 시작되는 경우
        # 예: open_url("...") wait(2) -> open_url("..."); wait(2)
        # 단, 이미 세미콜론이 있거나 줄바꿈이 있는 경우는 제외
        if "\n" not in cleaned_code and ";" not in cleaned_code:
            cleaned_code = re.sub(r'(\))(\s+)(?=[a-zA-Z_])', r'\1;\2', cleaned_code)
        
        # 3. 픽셀 좌표 모드 지원 (# pixels 주석이 있는 경우)
        # 사용자가 수동 테스트 시 click(500, 100) 처럼 픽셀 좌표를 사용하고 싶을 때
        if "# pixels" in code or "# pixel" in code:
            cleaned_code = cleaned_code.replace("click(", "click_pixels(")
            cleaned_code = cleaned_code.replace("drag(", "drag_pixels(")
            print("[Sanitizer] Pixel mode enabled: Redirecting click/drag to pixel versions")

        # 디버깅 로깅
        if cleaned_code != code:
            print(f"[Sanitizer] 코드 자동 수정됨:\nOriginal: {code}\nModified: {cleaned_code}")
        
        try:
            return self.executor(cleaned_code, *args, **kwargs)
        except Exception as e:
            print(f"[Sanitizer] 코드 실행 중 오류 발생: {e}")
            # 오류 발생 시 원본 코드로 재시도? 아니면 오류 전파?
            # 구문 오류일 가능성이 높으므로 전파
            raise e
