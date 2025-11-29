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
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.left_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"클릭: ({x}, {y})")
            return f"Clicked at coordinates ({x}, {y})"

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
            self.click_coordinates = [x, y]
            self.logger.log(f"오른쪽 클릭: ({x}, {y})")
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
            self.click_coordinates = [x, y]
            self.logger.log(f"더블 클릭: ({x}, {y})")
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
            self.desktop.write(text, delay_in_ms=75)
            self.logger.log(f"텍스트 입력: '{text}'")
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
            return f"Pressed keys: {keys}"

        @tool
        def go_back() -> str:
            """
            브라우저에서 이전 페이지로 이동
            """
            self.desktop.press(["alt", "left"])
            self.logger.log("이전 페이지로 이동")
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
            message = f"드래그: [{x1}, {y1}] → [{x2}, {y2}]"
            self.logger.log(message)
            return f"Dragged from [{x1}, {y1}] to [{x2}, {y2}]"

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
            return f"Opened URL: {url}"

        @tool
        def launch(app: str) -> str:
            """
            애플리케이션 실행
            Args:
                app: 실행할 애플리케이션
            """
            self.desktop.commands.run(f"{app}", background=True)
            return f"Launched application: {app}"

        # 도구 등록
        self.tools["click"] = click
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
