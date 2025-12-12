"""
Headless Desktop - CDP 기반 헤드리스 브라우저 제어 (LocalDesktop 인터페이스 호환)
Xvfb 대신 Chrome DevTools Protocol(CDP)을 사용하여 헤드리스 브라우저를 직접 제어합니다.
"""

import os
import subprocess
import time
import json
import base64
import requests
import logging
from io import BytesIO
from typing import Literal, Tuple, List, Optional
from PIL import Image
from websockets.sync.client import connect

logger = logging.getLogger(__name__)

class HeadlessDesktop:
    """
    CDP(Chrome DevTools Protocol)를 사용하는 헤드리스 브라우저 컨트롤러.
    LocalVisionAgent와 호환되는 인터페이스를 제공.
    """

    def __init__(self, width: int = 1920, height: int = 1080, port: int = 9222):
        self.width = width
        self.height = height
        self.port = port
        self.browser_proc = None
        self.ws_url = None
        self._msg_id = 0
        self._start_browser()

    def _start_browser(self):
        """Chrome 브라우저를 headless=new 모드로 실행"""
        # 기존 프로세스 정리
        subprocess.run(["pkill", "-f", f"remote-debugging-port={self.port}"], capture_output=True)
        time.sleep(1)

        chrome_cmd = [
            "google-chrome",
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--remote-debugging-port={self.port}",
            f"--window-size={self.width},{self.height}",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "about:blank"
        ]

        logger.info(f"Starting Headless Chrome on port {self.port}")
        self.browser_proc = subprocess.Popen(
            chrome_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 브라우저 시작 대기 및 연결
        for _ in range(10):
            time.sleep(1)
            if self._connect_cdp():
                return
        
        logger.error("Failed to connect to Headless Chrome CDP")

    def _connect_cdp(self) -> bool:
        """CDP 접속 URL 확보 (Page 타겟)"""
        try:
            # 탭 목록 조회
            response = requests.get(f"http://localhost:{self.port}/json/list", timeout=1)
            targets = response.json()
            
            # 'page' 타입의 첫 번째 타겟 찾기
            for target in targets:
                if target.get("type") == "page":
                    self.ws_url = target.get("webSocketDebuggerUrl")
                    logger.info(f"Connected to Page target: {target.get('url')}")
                    return True
            
            # 없으면 브라우저 타겟이라도 잡아서 새 탭 생성 시도? 
            # 일단 Headless start시 about:blank가 있으므로 페이지는 하나 있어야 함.
            return False
        except Exception as e:
            logger.error(f"CDP Connect Error: {e}")
            return False

    def _send_cdp(self, method: str, params: Optional[dict] = None) -> dict:
        """CDP 명령 전송 (동기식)"""
        if not self.ws_url:
            if not self._connect_cdp():
                return {}
        
        try:
            with connect(self.ws_url) as websocket:
                self._msg_id += 1
                message = {
                    "id": self._msg_id,
                    "method": method,
                    "params": params or {}
                }
                websocket.send(json.dumps(message))
                response = websocket.recv()
                return json.loads(response)
        except Exception as e:
            logger.error(f"CDP Error ({method}): {e}")
            return {}

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def screenshot(self) -> Image.Image:
        """Page.captureScreenshot 사용"""
        res = self._send_cdp("Page.captureScreenshot", {"format": "png"})
        if "result" in res and "data" in res["result"]:
            img_data = base64.b64decode(res["result"]["data"])
            image = Image.open(BytesIO(img_data))
            
            # Resize if too large (Max width 1280) to save tokens (26k -> ~8k)
            MAX_WIDTH = 1280
            if image.width > MAX_WIDTH:
                ratio = MAX_WIDTH / image.width
                new_height = int(image.height * ratio)
                image = image.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
                
            return image
        
        logger.error(f"Screenshot failed: {res.get('error')}")
        return Image.new("RGB", (self.width, self.height), (255, 255, 255))

    # 좌표 저장용
    _last_x = 0
    _last_y = 0

    def move_mouse(self, x: int, y: int):
        self._last_x = x
        self._last_y = y
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x,
            "y": y
        })

    def left_click(self):
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 1
        })
        time.sleep(0.05)
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 1
        })

    def right_click(self):
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": self._last_x,
            "y": self._last_y,
            "button": "right",
            "clickCount": 1
        })
        time.sleep(0.05)
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": self._last_x,
            "y": self._last_y,
            "button": "right",
            "clickCount": 1
        })

    def double_click(self):
        # 1st click
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 1
        })
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 1
        })
        time.sleep(0.1)
        # 2nd click
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 2
        })
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": self._last_x,
            "y": self._last_y,
            "button": "left",
            "clickCount": 2
        })

    def write(self, text: str, delay_in_ms: int = 50):
        """Input.dispatchKeyEvent (char)"""
        for char in text:
            self._send_cdp("Input.dispatchKeyEvent", {
                "type": "char",
                "text": char
            })
            time.sleep(delay_in_ms / 1000.0)

    def press(self, keys: List[str]):
        """Input.dispatchKeyEvent (raw key) - Simplified"""
        # CDP Key handling is complex. For basic nav, use raw key codes or text.
        # Enter: 
        key_map = {
            "enter": "Enter",
            "return": "Enter",
            "tab": "Tab",
            "backspace": "Backspace",
            "escape": "Escape",
        }
        for k in keys:
            cdp_key = key_map.get(k.lower())
            if cdp_key:
                 self._send_cdp("Input.dispatchKeyEvent", {"type": "keyDown", "text": "\r" if cdp_key=="Enter" else "" ,"unmodifiedText": "\r" if cdp_key=="Enter" else "", "key": cdp_key, "code": cdp_key, "windowsVirtualKeyCode": 13 if cdp_key=="Enter" else 0})
                 self._send_cdp("Input.dispatchKeyEvent", {"type": "keyUp", "key": cdp_key, "code": cdp_key})

    def scroll(self, direction: Literal["up", "down"] = "down", amount: int = 3):
        delta_y = 100 * amount if direction == "down" else -100 * amount
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": self._last_x,
            "y": self._last_y,
            "deltaY": delta_y
        })

    def drag(self, start: List[int], end: List[int]):
        self.move_mouse(start[0], start[1])
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": start[0],
            "y": start[1],
            "button": "left",
            "clickCount": 1
        })
        time.sleep(0.1)
        # Drag move
        steps = 5
        dx = (end[0] - start[0]) / steps
        dy = (end[1] - start[1]) / steps
        for i in range(steps):
             self._send_cdp("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": start[0] + dx * (i+1),
                "y": start[1] + dy * (i+1),
                "button": "left"
            })
        
        self._send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": end[0],
            "y": end[1],
            "button": "left",
            "clickCount": 1
        })

    def open(self, url: str, browser_type: str = "chrome"):
        if not url.startswith("http"):
            url = "https://" + url
        self._send_cdp("Page.navigate", {"url": url})
        logger.info(f"Navigated to {url}")

    def evaluate_script(self, script: str) -> str:
        res = self._send_cdp("Runtime.evaluate", {
            "expression": script,
            "returnByValue": True
        })
        if "result" in res and "result" in res["result"]:
            val = res["result"]["result"].get("value")
            return str(val)
        return str(res.get("error"))

    @property
    def commands(self):
        return self

    def run(self, command: str, background: bool = False):
        print(f"Ignored system command in headless: {command}")

    def kill(self):
        if self.browser_proc:
            self.browser_proc.kill()
        subprocess.run(["pkill", "-f", f"remote-debugging-port={self.port}"], capture_output=True)
