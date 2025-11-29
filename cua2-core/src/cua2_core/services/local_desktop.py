"""
로컬 데스크톱 제어 모듈 - E2B 샌드박스 대체
Xvfb 가상 디스플레이 + Xdotool 사용
"""

import os
import subprocess
import time
import signal
import tempfile
from typing import Literal

from PIL import Image


class LocalDesktop:
    """로컬 데스크톱 제어 클래스 - Xvfb 가상 디스플레이 사용"""

    def __init__(self, width: int = 1280, height: int = 720, display_num: int = 99):
        self.width = width
        self.height = height
        self.display_num = display_num
        self.display = f":{display_num}"
        self._xvfb_proc = None
        self._browser_proc = None
        self._stream_url = None
        self._browser_started = False

        # Xvfb 전용 프로필 디렉토리 (기존 브라우저와 분리)
        self.profile_dir = f"/tmp/cua-xvfb-profile-{display_num}"
        os.makedirs(self.profile_dir, exist_ok=True)

        # Xvfb 가상 디스플레이 시작
        self._start_xvfb()

    def _start_xvfb(self):
        """Xvfb 가상 디스플레이 시작"""
        # 기존 Xvfb 프로세스 정리
        subprocess.run(["pkill", "-f", f"Xvfb {self.display}"], capture_output=True)
        time.sleep(0.5)

        # Xvfb 시작
        self._xvfb_proc = subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0", f"{self.width}x{self.height}x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        print(f"Xvfb 가상 디스플레이 시작: {self.display} ({self.width}x{self.height})")

    def _get_env(self):
        """DISPLAY 환경변수가 설정된 환경 반환"""
        return {**os.environ, "DISPLAY": self.display}

    def get_screen_size(self) -> tuple[int, int]:
        """화면 크기 반환"""
        return (self.width, self.height)

    def screenshot(self) -> "Image.Image":
        """스크린샷 캡처 (Xvfb에서)"""
        import io

        # import 명령으로 스크린샷 캡처
        result = subprocess.run(
            ["import", "-window", "root", "-display", self.display, "png:-"],
            capture_output=True,
            env=self._get_env()
        )

        if result.returncode == 0 and len(result.stdout) > 1000:
            return Image.open(io.BytesIO(result.stdout))
        else:
            # 실패시 빈 이미지 반환
            print(f"스크린샷 캡처 실패: {result.stderr.decode()[:100]}")
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

    def _run_xdotool(self, *args):
        """xdotool 실행 (가상 디스플레이에서)"""
        subprocess.run(["xdotool"] + list(args), env=self._get_env(), check=True)

    def move_mouse(self, x: int, y: int):
        """마우스 이동"""
        self._run_xdotool("mousemove", str(x), str(y))

    def left_click(self):
        """왼쪽 클릭"""
        self._run_xdotool("click", "1")

    def right_click(self):
        """오른쪽 클릭"""
        self._run_xdotool("click", "3")

    def double_click(self):
        """더블 클릭"""
        self._run_xdotool("click", "--repeat", "2", "1")

    def write(self, text: str, delay_in_ms: int = 50):
        """텍스트 입력"""
        self._run_xdotool("type", "--delay", str(delay_in_ms), text)

    def press(self, keys: list[str]):
        """키 입력"""
        key_map = {
            "enter": "Return",
            "return": "Return",
            "space": "space",
            "backspace": "BackSpace",
            "tab": "Tab",
            "escape": "Escape",
            "esc": "Escape",
            "up": "Up",
            "down": "Down",
            "left": "Left",
            "right": "Right",
            "ctrl": "ctrl",
            "alt": "alt",
            "shift": "shift",
            "super": "super",
        }
        mapped_keys = [key_map.get(k.lower(), k) for k in keys]

        if len(mapped_keys) == 1:
            self._run_xdotool("key", mapped_keys[0])
        else:
            combo = "+".join(mapped_keys)
            self._run_xdotool("key", combo)

    def scroll(self, direction: Literal["up", "down"] = "down", amount: int = 3):
        """스크롤"""
        button = "4" if direction == "up" else "5"
        for _ in range(amount):
            self._run_xdotool("click", button)

    def drag(self, start: list[int], end: list[int]):
        """드래그"""
        self._run_xdotool("mousemove", str(start[0]), str(start[1]))
        self._run_xdotool("mousedown", "1")
        self._run_xdotool("mousemove", str(end[0]), str(end[1]))
        self._run_xdotool("mouseup", "1")

    def open(self, url: str):
        """URL 열기 (가상 디스플레이에서 브라우저 실행)"""
        env = self._get_env()

        # 기존 브라우저 프로세스 종료 (같은 프로필 사용하는 것만)
        subprocess.run(
            ["pkill", "-f", f"--user-data-dir={self.profile_dir}"],
            capture_output=True
        )
        time.sleep(0.5)

        # Chromium 실행 (새 인스턴스로 강제 실행)
        for browser in ["chromium-browser", "chromium", "google-chrome"]:
            try:
                self._browser_proc = subprocess.Popen(
                    [
                        browser,
                        f"--user-data-dir={self.profile_dir}",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-sync",
                        "--disable-popup-blocking",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--disable-software-rasterizer",
                        "--disable-extensions",
                        "--start-maximized",
                        "--new-window",
                        url
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"{browser} 시작 (Xvfb {self.display}) -> {url}")
                self._browser_started = True
                # 브라우저 로딩 대기
                time.sleep(5)
                return
            except FileNotFoundError:
                continue

        # Firefox
        firefox_profile = f"/tmp/cua-xvfb-firefox-{self.display_num}"
        os.makedirs(firefox_profile, exist_ok=True)
        try:
            self._browser_proc = subprocess.Popen(
                ["firefox", "--new-instance", "-profile", firefox_profile, url],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Firefox 시작 (Xvfb {self.display}) -> {url}")
            self._browser_started = True
            time.sleep(5)
            return
        except FileNotFoundError:
            pass

        print("브라우저를 찾을 수 없습니다")

    @property
    def commands(self):
        """명령 실행 인터페이스"""
        return self

    def run(self, command: str, background: bool = False):
        """명령 실행 (가상 디스플레이에서)"""
        env = self._get_env()
        if background:
            subprocess.Popen(command, shell=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(command, shell=True, env=env)

    def kill(self):
        """Xvfb 프로세스 정리"""
        # 브라우저 종료
        subprocess.run(["pkill", "-f", f"--user-data-dir={self.profile_dir}"],
                      capture_output=True)

        # Xvfb 종료
        if self._xvfb_proc:
            self._xvfb_proc.terminate()
            try:
                self._xvfb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._xvfb_proc.kill()
            print(f"Xvfb 종료: {self.display}")

    @property
    def stream(self):
        """스트림 인터페이스 (로컬은 더미)"""
        return self

    def start(self, require_auth: bool = False):
        """스트림 시작 (로컬은 더미)"""
        pass

    def get_url(self, **kwargs) -> str:
        """스트림 URL (로컬은 빈 문자열)"""
        return ""

    def get_auth_key(self) -> str:
        """인증 키 (로컬은 빈 문자열)"""
        return ""


def create_local_sandbox(resolution: tuple[int, int] = (1280, 720), **kwargs) -> LocalDesktop:
    """로컬 데스크톱 샌드박스 생성"""
    return LocalDesktop(width=resolution[0], height=resolution[1])
