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
        # Snap 패키지 호환성을 위해 /tmp 대신 홈 디렉토리 사용
        self.base_profile_dir = os.path.expanduser("~/.cua/profiles")
        os.makedirs(self.base_profile_dir, exist_ok=True)
        self.profile_dir = os.path.join(self.base_profile_dir, f"xvfb-profile-{display_num}")
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

        env = self._get_env()

        # 방법 1: xwd + convert (ImageMagick)
        try:
            # xwd로 캡처 후 convert로 PNG 변환
            xwd_result = subprocess.run(
                ["xwd", "-root", "-display", self.display],
                capture_output=True,
                env=env
            )
            if xwd_result.returncode == 0 and len(xwd_result.stdout) > 1000:
                # convert가 있으면 사용
                convert_result = subprocess.run(
                    ["convert", "xwd:-", "png:-"],
                    input=xwd_result.stdout,
                    capture_output=True
                )
                if convert_result.returncode == 0:
                    return Image.open(io.BytesIO(convert_result.stdout))
        except FileNotFoundError:
            pass

        # 방법 2: import 명령 (ImageMagick)
        try:
            result = subprocess.run(
                ["import", "-window", "root", "-display", self.display, "png:-"],
                capture_output=True,
                env=env
            )
            if result.returncode == 0 and len(result.stdout) > 1000:
                return Image.open(io.BytesIO(result.stdout))
        except FileNotFoundError:
            pass

        # 방법 3: scrot
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
            subprocess.run(
                ["scrot", "-d", "0", temp_path],
                env=env,
                check=True
            )
            img = Image.open(temp_path)
            os.unlink(temp_path)
            return img
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 방법 4: gnome-screenshot
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
            subprocess.run(
                ["gnome-screenshot", "-f", temp_path],
                env=env,
                check=True
            )
            img = Image.open(temp_path)
            os.unlink(temp_path)
            return img
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 모두 실패시 빈 이미지 반환
        print("스크린샷 캡처 실패: 사용 가능한 도구 없음 (imagemagick, scrot, gnome-screenshot 중 하나 설치 필요)")
        return Image.new("RGB", (self.width, self.height), (0, 0, 0))

    def _run_xdotool(self, *args):
        """xdotool 실행 (가상 디스플레이에서)"""
        subprocess.run(["xdotool"] + list(args), env=self._get_env(), check=True)

    def move_mouse(self, x: int, y: int):
        """마우스 이동"""
        self._run_xdotool("mousemove", "--sync", str(x), str(y))

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

    def _copy_chrome_profile(self):
        """기존 Chrome 프로필을 복사하여 독립된 프로필 생성"""
        import shutil

        source_profile = os.path.expanduser("~/.config/google-chrome")
        target_profile = os.path.join(self.base_profile_dir, f"chrome-{self.display_num}")

        # 기존 복사본 삭제 (락 파일 문제 방지)
        if os.path.exists(target_profile):
            try:
                shutil.rmtree(target_profile)
            except Exception as e:
                print(f"기존 프로필 삭제 실패: {e}")

        # 필요한 파일만 복사 (쿠키, 로그인 상태 등)
        os.makedirs(target_profile, exist_ok=True)

        files_to_copy = [
            "Default/Cookies",
            "Default/Login Data",
            "Default/Web Data",
            "Default/Preferences",
            "Default/Secure Preferences",
            "Default/Local Storage",
            "Default/Session Storage",
            "Default/IndexedDB",
            "Local State",
        ]

        for file_path in files_to_copy:
            src = os.path.join(source_profile, file_path)
            dst = os.path.join(target_profile, file_path)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(f"파일 복사 실패 {file_path}: {e}")

        return target_profile

    def open(self, url: str, browser_type: str = "chrome"):
        """URL 열기 (가상 디스플레이에서 브라우저 실행)

        Args:
            url: 열 URL
            browser_type: "chrome" (기본, 프로필 복사하여 독립 실행) 또는 "firefox"
        """
        env = self._get_env()

        # Chrome (프로필 복사하여 독립 인스턴스로 실행)
        if browser_type == "chrome":
            # 기존 프로필 복사
            chrome_profile_dir = self._copy_chrome_profile()
            print(f"Chrome 프로필 복사 완료: {chrome_profile_dir}")

            for browser in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
                try:
                    self._browser_proc = subprocess.Popen(
                        [
                            browser,
                            f"--user-data-dir={chrome_profile_dir}",
                            "--profile-directory=Default",
                            f"--window-size={self.width},{self.height}",
                            "--no-first-run",
                            "--no-default-browser-check",
                            "--disable-background-networking",
                            "--disable-sync",
                            "--no-sandbox",
                            "--disable-gpu",
                            "--remote-debugging-port=9222",  # CDP 포트 활성화
                            url
                        ],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"{browser} 시작 (Xvfb {self.display}, 독립 프로필, CDP:9222) -> {url}")
                    self._browser_started = True
                    time.sleep(5)
                    return
                except FileNotFoundError:
                    continue

            print("Chrome을 찾을 수 없음, Firefox로 시도...")
            browser_type = "firefox"

    def evaluate_script(self, script: str) -> str:
        """
        CDP를 사용하여 자바스크립트 실행
        Args:
            script: 실행할 자바스크립트 코드
        Returns:
            실행 결과 (문자열)
        """
        import requests
        import json
        from websockets.sync.client import connect

        try:
            # 1. CDP WebSocket URL 가져오기
            response = requests.get("http://localhost:9222/json")
            pages = response.json()
            
            # 'page' 타입의 탭 찾기
            ws_url = None
            for page in pages:
                if page.get("type") == "page":
                    ws_url = page.get("webSocketDebuggerUrl")
                    break
            
            if not ws_url:
                return "Error: No active page found for CDP"

            # 2. WebSocket 연결 및 스크립트 실행
            with connect(ws_url) as websocket:
                # 초기 시도: 원본 스크립트 실행
                # (IIFE나 표현식인 경우 그대로 실행됨)
                message = {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": script,
                        "returnByValue": True,
                        "awaitPromise": True
                    }
                }
                websocket.send(json.dumps(message))
                
                # 응답 수신
                response = websocket.recv()
                result = json.loads(response)
                
                # "Illegal return statement" 에러 발생 시 래핑하여 재시도
                # (top-level return 문이 있는 경우)
                if "exceptionDetails" in result["result"]:
                    exception = result["result"]["exceptionDetails"]
                    if exception.get("exception", {}).get("description", "").startswith("SyntaxError: Illegal return statement"):
                        # 래핑하여 재시도
                        wrapped_script = f"(async function() {{ {script} }})()"
                        message["params"]["expression"] = wrapped_script
                        message["id"] = 2
                        websocket.send(json.dumps(message))
                        
                        response = websocket.recv()
                        result = json.loads(response)

                if "error" in result:
                    return f"CDP Error: {result['error']['message']}"
                
                if "exceptionDetails" in result["result"]:
                    return f"JS Exception: {result['result']['exceptionDetails']}"
                
                value = result["result"]["result"].get("value", "None")
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return str(value)

        except Exception as e:
            return f"Error executing script: {e}"

        # Firefox (fallback)
        if browser_type == "firefox":
            import shutil
            firefox_profile = os.path.join(self.base_profile_dir, f"firefox-{self.display_num}")
            if os.path.exists(firefox_profile):
                try:
                    shutil.rmtree(firefox_profile)
                except Exception as e:
                    print(f"Firefox 프로필 삭제 실패: {e}")
            os.makedirs(firefox_profile, exist_ok=True)

            try:
                self._browser_proc = subprocess.Popen(
                    [
                        "firefox",
                        "--new-instance",
                        "--no-remote",
                        "-profile", firefox_profile,
                        f"--width={self.width}",
                        f"--height={self.height}",
                        url
                    ],
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
