import os
import time
import unicodedata

from cua2_core.services.agent_utils.prompt import E2B_SYSTEM_PROMPT_TEMPLATE

# E2B imports
from e2b_desktop import Sandbox

# SmolaAgents imports
from smolagents import CodeAgent, Model, tool
from smolagents.monitoring import LogLevel


class E2BVisionAgent(CodeAgent):
    """Agent for e2b desktop automation with Qwen2.5VL vision capabilities"""

    def __init__(
        self,
        model: Model,
        data_dir: str,
        desktop: Sandbox,
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
        # Initialize Desktop
        self.width, self.height = self.desktop.get_screen_size()
        print(f"Screen size: {self.width}x{self.height}")

        # Set up temp directory
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Screenshots and steps will be saved to: {self.data_dir}")

        self.use_v1_prompt = use_v1_prompt
        # Initialize base agent
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

        # Add screen info to state
        self.state["screen_width"] = self.width
        self.state["screen_height"] = self.height

        # Add default tools
        self.logger.log("Setting up agent tools...")
        self._setup_desktop_tools()

    def _qwen_unnormalization(self, arguments: dict[str, int]) -> dict[str, int]:
        """
        Unnormalize coordinates from 0-999 range to actual screen pixel coordinates.
        Coordinates are identified by keys containing 'x' or 'y'.

        Args:
            arguments: Dictionary with coordinate parameters (keys containing 'x' or 'y')

        Returns:
            Dictionary with unnormalized pixel coordinates
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
        """Register all desktop tools"""

        @tool
        def click(x: int, y: int) -> str:
            """
            Performs a left-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.left_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Clicked at coordinates ({x}, {y})")
            return f"Clicked at coordinates ({x}, {y})"

        @tool
        def right_click(x: int, y: int) -> str:
            """
            Performs a right-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.right_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Right-clicked at coordinates ({x}, {y})")
            return f"Right-clicked at coordinates ({x}, {y})"

        @tool
        def double_click(x: int, y: int) -> str:
            """
            Performs a double-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.double_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Double-clicked at coordinates ({x}, {y})")
            return f"Double-clicked at coordinates ({x}, {y})"

        @tool
        def move_mouse(x: int, y: int) -> str:
            """
            Moves the mouse cursor to the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.logger.log(f"Moved mouse to coordinates ({x}, {y})")
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
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
            """
            clean_text = normalize_text(text)
            self.desktop.write(clean_text, delay_in_ms=75)
            self.logger.log(f"Typed text: '{clean_text}'")
            return f"Typed text: '{clean_text}'"

        @tool
        def press(keys: list[str]) -> str:
            """
            Presses a keyboard key
            Args:
                keys: The keys to press (e.g. ["enter", "space", "backspace", etc.]).
            """
            self.desktop.press(keys)
            self.logger.log(f"Pressed keys: {keys}")
            return f"Pressed keys: {keys}"

        @tool
        def go_back() -> str:
            """
            Goes back to the previous page in the browser. If using this tool doesn't work, just click the button directly.
            Args:
            """
            self.desktop.press(["alt", "left"])
            self.logger.log("Went back one page")
            return "Went back one page"

        @tool
        def drag(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Clicks [x1, y1], drags mouse to [x2, y2], then release click.
            Args:
                x1: origin x coordinate
                y1: origin y coordinate
                x2: end x coordinate
                y2: end y coordinate
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                )
                x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            self.desktop.drag([x1, y1], [x2, y2])
            message = f"Dragged and dropped from [{x1}, {y1}] to [{x2}, {y2}]"
            self.logger.log(message)
            return message

        @tool
        def scroll(x: int, y: int, direction: str = "down", amount: int = 2) -> str:
            """
            Moves the mouse to selected coordinates, then uses the scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
            Args:
                x: The x coordinate (horizontal position) of the element to scroll/zoom
                y: The y coordinate (vertical position) of the element to scroll/zoom
                direction: The direction to scroll ("up" or "down"), defaults to "down". For zoom, "up" zooms in, "down" zooms out.
                amount: The amount to scroll. A good amount is 1 or 2.
            """
            if self.qwen_normalization:
                coords = self._qwen_unnormalization({"x": x, "y": y})
                x, y = coords["x"], coords["y"]
            self.desktop.move_mouse(x, y)
            self.desktop.scroll(direction=direction, amount=amount)
            message = f"Scrolled {direction} by {amount}"
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
        def open_url(url: str) -> str:
            """
            Directly opens a browser with the specified url: use this at start of web searches rather than trying to click the browser.
            Args:
                url: The URL to open
            """
            if not url.startswith("http") and not url.startswith("https"):
                url = f"https://{url}"
            self.desktop.open(url)

            time.sleep(2)
            self.logger.log(f"Opening URL: {url}")
            return f"Opened URL: {url}"

        @tool
        def launch(app: str) -> str:
            """
            Launches the specified application
            Args:
                app: The application to launch
            """
            self.desktop.commands.run(f"{app}", background=True)
            return f"Launched application: {app}"

        # Register the tools
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
        self.tools["scroll"] = scroll
