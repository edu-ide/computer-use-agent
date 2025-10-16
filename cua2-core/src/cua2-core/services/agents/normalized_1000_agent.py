import time
import unicodedata
from typing import List, Literal

# SmolaAgents imports
from smolagents import Model, Tool, tool
from smolagents.monitoring import LogLevel

from backend.models.models import AgentType
from backend.services.agents.prompt import Normalized1000CoordinatesSystemPrompt
from computer_use_studio import DesktopAgentBase, Sandbox


class Normalized1000Agent(DesktopAgentBase):
    """Agent for desktop automation with normalized coordinates (0 to 1000)"""

    AGENT_TYPE = AgentType.NORMALIZED_1000_COORDINATES

    def __init__(
        self,
        model: Model,
        data_dir: str,
        desktop: Sandbox,
        system_prompt: Normalized1000CoordinatesSystemPrompt,
        tools: List[Tool] | None = None,
        max_steps: int = 20,
        verbosity_level: LogLevel = LogLevel.INFO,
        planning_interval: int | None = None,
        use_v1_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_dir=data_dir,
            desktop=desktop,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=planning_interval,
            use_v1_prompt=use_v1_prompt,
            **kwargs,
        )

    def _normalize_to_pixel(self, norm_x: int, norm_y: int) -> tuple[int, int]:
        """
        Convert normalized coordinates (0-1000) to pixel coordinates
        Args:
            norm_x: Normalized x coordinate (0 to 1000)
            norm_y: Normalized y coordinate (0 to 1000)
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Clamp values to valid range
        norm_x = max(0, min(1000, norm_x))
        norm_y = max(0, min(1000, norm_y))

        # Convert from 0-1000 range to 0-1 range, then to pixels
        norm_x_float = norm_x / 1000.0
        norm_y_float = norm_y / 1000.0

        pixel_x = int(norm_x_float * self.width)
        pixel_y = int(norm_y_float * self.height)

        # Ensure we don't go outside screen bounds
        pixel_x = max(0, min(self.width - 1, pixel_x))
        pixel_y = max(0, min(self.height - 1, pixel_y))

        return pixel_x, pixel_y

    def _setup_desktop_tools(self):
        """Register all desktop tools with normalized coordinate support (0-1000)"""

        @tool
        def click(x: int, y: int) -> str:
            """
            Performs a left-click at the specified normalized coordinates
            Args:
                x: The normalized x coordinate (0 to 1000, where 0 is left edge, 1000 is right edge)
                y: The normalized y coordinate (0 to 1000, where 0 is top edge, 1000 is bottom edge)
            """
            pixel_x, pixel_y = self._normalize_to_pixel(x, y)
            self.desktop.left_click(pixel_x, pixel_y)
            self.click_coordinates = (pixel_x, pixel_y)
            self.logger.log(
                f"Clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"
            )
            time.sleep(1)
            return f"Clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"

        @tool
        def right_click(x: int, y: int) -> str:
            """
            Performs a right-click at the specified normalized coordinates
            Args:
                x: The normalized x coordinate (0 to 1000, where 0 is left edge, 1000 is right edge)
                y: The normalized y coordinate (0 to 1000, where 0 is top edge, 1000 is bottom edge)
            """
            pixel_x, pixel_y = self._normalize_to_pixel(x, y)
            self.desktop.right_click(pixel_x, pixel_y)
            self.click_coordinates = (pixel_x, pixel_y)
            self.logger.log(
                f"Right-clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"
            )
            return f"Right-clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"

        @tool
        def double_click(x: int, y: int) -> str:
            """
            Performs a double-click at the specified normalized coordinates
            Args:
                x: The normalized x coordinate (0 to 1000, where 0 is left edge, 1000 is right edge)
                y: The normalized y coordinate (0 to 1000, where 0 is top edge, 1000 is bottom edge)
            """
            pixel_x, pixel_y = self._normalize_to_pixel(x, y)
            self.desktop.double_click(pixel_x, pixel_y)
            self.click_coordinates = (pixel_x, pixel_y)
            self.logger.log(
                f"Double-clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"
            )
            return f"Double-clicked at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"

        @tool
        def move_mouse(x: int, y: int) -> str:
            """
            Moves the mouse cursor to the specified normalized coordinates
            Args:
                x: The normalized x coordinate (0 to 1000, where 0 is left edge, 1000 is right edge)
                y: The normalized y coordinate (0 to 1000, where 0 is top edge, 1000 is bottom edge)
            """
            pixel_x, pixel_y = self._normalize_to_pixel(x, y)
            self.desktop.move_mouse(pixel_x, pixel_y)
            self.logger.log(
                f"Moved mouse to normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"
            )
            return f"Moved mouse to normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"

        def normalize_text(text):
            return "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if not unicodedata.combining(c)
            )

        @tool
        def write(text: str) -> str:
            """
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
            """
            # clean_text = normalize_text(text)
            self.desktop.write(text, delay_in_ms=10)
            self.logger.log(f"Typed text: '{text}'")
            time.sleep(1)
            return f"Typed text: '{text}'"

        @tool
        def press(key: str) -> str:
            """
            Presses a keyboard key or combination of keys
            Args:
                key: The key to press (e.g. "enter", "space", "backspace", etc.) or a multiple keys string to press, for example "ctrl+a" or "ctrl+shift+a".
            """
            self.desktop.press(key)
            self.logger.log(f"Pressed key: {key}")
            time.sleep(0.1)
            return f"Pressed key: {key}"

        @tool
        def drag(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Clicks at normalized coordinates [x1, y1], drags mouse to [x2, y2], then release click.
            Args:
                x1: origin normalized x coordinate (0 to 1000)
                y1: origin normalized y coordinate (0 to 1000)
                x2: end normalized x coordinate (0 to 1000)
                y2: end normalized y coordinate (0 to 1000)
            """
            pixel_x1, pixel_y1 = self._normalize_to_pixel(x1, y1)
            pixel_x2, pixel_y2 = self._normalize_to_pixel(x2, y2)
            self.desktop.drag((pixel_x1, pixel_y1), (pixel_x2, pixel_y2))
            message = f"Dragged and dropped from normalized [{x1}, {y1}] to [{x2}, {y2}] -> pixels [{pixel_x1}, {pixel_y1}] to [{pixel_x2}, {pixel_y2}]"
            self.logger.log(message)
            return message

        @tool
        def scroll(
            x: int,
            y: int,
            direction: Literal["up", "down"] = "down",
            amount: int = 2,
        ) -> str:
            """
            Moves the mouse to selected normalized coordinates, then uses the scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
            Args:
                x: The normalized x coordinate (0 to 1000) of the element to scroll/zoom
                y: The normalized y coordinate (0 to 1000) of the element to scroll/zoom
                direction: The direction to scroll ("up" or "down"), defaults to "down". For zoom, "up" zooms in, "down" zooms out.
                amount: The amount to scroll. A good amount is 1 or 2.
            """
            pixel_x, pixel_y = self._normalize_to_pixel(x, y)
            self.desktop.move_mouse(pixel_x, pixel_y)
            self.desktop.scroll(direction=direction, amount=amount)
            message = f"Scrolled {direction} by {amount} at normalized coordinates ({x}, {y}) -> pixels ({pixel_x}, {pixel_y})"
            self.logger.log(message)
            return message

        @tool
        def wait(seconds: float) -> str:
            """
            Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
            Args:
                seconds: Number of seconds to wait, generally 3 is enough.
            """
            time.sleep(seconds)
            self.logger.log(f"Waited for {seconds} seconds")
            return f"Waited for {seconds} seconds"

        @tool
        def open(file_or_url: str) -> str:
            """
            Directly opens a browser with the specified url or opens a file with the default application: use this at start of web searches rather than trying to click the browser or open a file by clicking.
            Args:
                file_or_url: The URL or file to open
            """

            self.desktop.open(file_or_url)
            # Give it time to load
            time.sleep(2)
            self.logger.log(f"Opening: {file_or_url}")
            return f"Opened: {file_or_url}"

        @tool
        def launch_app(app_name: str) -> str:
            """
            Launches the specified application.
            Args:
                app_name: the name of the application to launch
            """
            self.desktop.launch(app_name)
            self.logger.log(f"Launched app: {app_name}")
            return f"Launched app: {app_name}"

        @tool
        def execute(command: str) -> str:
            """
            Executes a terminal command in the desktop environment.
            Args:
                command: The command to execute
            """
            self.desktop.execute_command(command)
            self.logger.log(f"Executed command: {command}")
            return f"Executed command: {command}"

        @tool
        def refresh() -> str:
            """
            Refreshes the current web page if you're in a browser.
            """
            self.desktop.press(["ctrl", "r"])
            self.logger.log("Refreshed the current page")
            return "Refreshed the current page"

        @tool
        def go_back() -> str:
            """
            Goes back to the previous page in the browser. If using this tool doesn't work, just click the button directly.
            Args:
            """
            self.desktop.press(["alt", "left"])
            self.logger.log("Went back one page")
            return "Went back one page"

        # Register the tools
        self.tools["click"] = click
        self.tools["right_click"] = right_click
        self.tools["double_click"] = double_click
        self.tools["move_mouse"] = move_mouse
        self.tools["write"] = write
        self.tools["press"] = press
        self.tools["scroll"] = scroll
        self.tools["wait"] = wait
        self.tools["open"] = open
        self.tools["go_back"] = go_back
        self.tools["drag"] = drag
        self.tools["launch_app"] = launch_app
        self.tools["execute"] = execute
        self.tools["refresh"] = refresh
        self.tools["refresh"] = refresh
        self.tools["execute"] = execute
        self.tools["refresh"] = refresh
        self.tools["refresh"] = refresh
