import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

from e2b_desktop import Sandbox

logger = logging.getLogger(__name__)

SANDBOX_METADATA: dict[str, dict[str, Any]] = {}
SANDBOX_TIMEOUT = 300
WIDTH = 1280
HEIGHT = 960


class SandboxService:
    def __init__(self, max_sandboxes: int = 50):
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is not set")
        self.max_sandboxes = max_sandboxes
        self.sandboxes: dict[str, Sandbox] = {}
        self.sandbox_metadata: dict[str, dict[str, Any]] = {}
        self.sandbox_lock = asyncio.Lock()

    async def acquire_sandbox(self, session_hash: str) -> Sandbox | None:
        async with self.sandbox_lock:
            current_time = datetime.now()

            if (
                session_hash in self.sandboxes
                and session_hash in self.sandbox_metadata
                and current_time - self.sandbox_metadata[session_hash]["created_at"]
                < SANDBOX_TIMEOUT
            ):
                print(f"Reusing Sandbox for session {session_hash}")
                self.sandbox_metadata[session_hash]["last_accessed"] = current_time
                return self.sandboxes[session_hash]

            if session_hash in self.sandboxes:
                try:
                    print(f"Closing expired sandbox for session {session_hash}")
                    await asyncio.to_thread(self.sandboxes[session_hash].kill)
                except Exception as e:
                    print(f"Error closing expired sandbox: {str(e)}")
            elif len(self.sandboxes) >= self.max_sandboxes:
                return None

            print(f"Creating new sandbox for session {session_hash}")

            def create_and_setup_sandbox():
                desktop = Sandbox.create(
                    api_key=os.getenv("E2B_API_KEY"),
                    resolution=(WIDTH, HEIGHT),
                    dpi=96,
                    timeout=SANDBOX_TIMEOUT,
                    template="k0wmnzir0zuzye6dndlw",
                )
                desktop.stream.start(require_auth=True)
                setup_cmd = """sudo mkdir -p /usr/lib/firefox-esr/distribution && echo '{"policies":{"OverrideFirstRunPage":"","OverridePostUpdatePage":"","DisableProfileImport":true,"DontCheckDefaultBrowser":true}}' | sudo tee /usr/lib/firefox-esr/distribution/policies.json > /dev/null"""
                desktop.commands.run(setup_cmd)
                time.sleep(3)
                return desktop

            desktop = await asyncio.to_thread(create_and_setup_sandbox)

            print(f"Sandbox ID for session {session_hash} is {desktop.sandbox_id}.")

            self.sandboxes[session_hash] = desktop
            self.sandbox_metadata[session_hash] = {
                "created_at": current_time,
                "last_accessed": current_time,
            }
            return desktop

    async def release_sandbox(self, session_hash: str):
        async with self.sandbox_lock:
            if session_hash in self.sandboxes:
                print(f"Releasing sandbox for session {session_hash}")
                await asyncio.to_thread(self.sandboxes[session_hash].kill)
                del self.sandboxes[session_hash]
                del self.sandbox_metadata[session_hash]

    async def cleanup_sandboxes(self):
        async with self.sandbox_lock:
            for session_hash in list(self.sandboxes.keys()):
                await asyncio.to_thread(self.sandboxes[session_hash].kill)
                del self.sandboxes[session_hash]
                del self.sandbox_metadata[session_hash]


if __name__ == "__main__":
    desktop: Sandbox = Sandbox.create(
        api_key=os.getenv("E2B_API_KEY"),
        resolution=(WIDTH, HEIGHT),
        dpi=96,
        timeout=SANDBOX_TIMEOUT,
        template="k0wmnzir0zuzye6dndlw",
    )
    desktop.stream.start(require_auth=True)
    setup_cmd = """sudo mkdir -p /usr/lib/firefox-esr/distribution && echo '{"policies":{"OverrideFirstRunPage":"","OverridePostUpdatePage":"","DisableProfileImport":true,"DontCheckDefaultBrowser":true}}' | sudo tee /usr/lib/firefox-esr/distribution/policies.json > /dev/null"""
    desktop.commands.run(setup_cmd)
    print(
        desktop.stream.get_url(
            auto_connect=True,
            view_only=False,
            resize="scale",
            auth_key=desktop.stream.get_auth_key(),
        )
    )
    try:
        while True:
            application = input("Enter application to launch: ")
            desktop.commands.run(f"{application} &")
    except (KeyboardInterrupt, Exception):
        pass

    desktop.kill()
