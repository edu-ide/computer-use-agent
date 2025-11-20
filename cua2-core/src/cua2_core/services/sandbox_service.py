import asyncio
import os
import time
from datetime import datetime
from typing import Any, Literal

from e2b_desktop import Sandbox
from pydantic import BaseModel

SANDBOX_METADATA: dict[str, dict[str, Any]] = {}
SANDBOX_TIMEOUT = 500
SANDBOX_CREATION_TIMEOUT = 200
SANDBOX_CREATION_MAX_TIME = (
    300  # Maximum time a sandbox can be in "creating" state (5 minutes)
)
WIDTH = 1280
HEIGHT = 960


class SandboxResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sandbox: Sandbox | None
    state: Literal["creating", "ready", "max_sandboxes_reached"]


class SandboxService:
    def __init__(self, max_sandboxes: int = 50):
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is not set")
        self.max_sandboxes = max_sandboxes
        self.sandboxes: dict[str, Sandbox] = {}
        self.sandbox_metadata: dict[str, dict[str, Any]] = {}
        self.sandbox_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def _create_sandbox_background(
        self, session_hash: str, expired_sandbox: Sandbox | None
    ):
        """Background task to create and setup a sandbox."""
        # Kill expired sandbox first
        if expired_sandbox:
            try:
                print(f"Closing expired sandbox for session {session_hash}")
                await asyncio.to_thread(expired_sandbox.kill)
            except Exception as e:
                print(f"Error closing expired sandbox: {str(e)}")

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

        try:
            desktop = await asyncio.to_thread(create_and_setup_sandbox)
            print(f"Sandbox ID for session {session_hash} is {desktop.sandbox_id}.")

            # Update sandbox state under lock
            async with self.sandbox_lock:
                # Double-check metadata still exists and is in "creating" state
                # (it might have been released while we were creating)
                if (
                    session_hash in self.sandbox_metadata
                    and self.sandbox_metadata[session_hash].get("state") == "creating"
                ):
                    self.sandboxes[session_hash] = desktop
                    self.sandbox_metadata[session_hash]["state"] = "ready"
                else:
                    # Sandbox was released while creating, kill it immediately
                    print(
                        f"Sandbox {session_hash} was released during creation, killing it"
                    )
                    try:
                        await asyncio.to_thread(desktop.kill)
                    except Exception as kill_error:
                        print(f"Error killing orphaned sandbox: {str(kill_error)}")

        except Exception as e:
            print(f"Error creating sandbox for session {session_hash}: {str(e)}")
            # Clean up metadata on failure
            async with self.sandbox_lock:
                if session_hash in self.sandbox_metadata:
                    del self.sandbox_metadata[session_hash]

    async def _periodic_cleanup(self):
        """Background task to periodically clean up stuck creating sandboxes"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self.cleanup_stuck_creating_sandboxes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic cleanup: {str(e)}")

    def start_periodic_cleanup(self):
        """Start the periodic cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    def stop_periodic_cleanup(self):
        """Stop the periodic cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def acquire_sandbox(self, session_hash: str) -> SandboxResponse:
        current_time = datetime.now()
        should_create = False
        expired_sandbox = None

        # Quick check under lock - only check state and mark creation
        async with self.sandbox_lock:
            # Check if sandbox exists and is ready
            if (
                session_hash in self.sandboxes
                and session_hash in self.sandbox_metadata
                and self.sandbox_metadata[session_hash].get("state") == "ready"
                and (
                    current_time - self.sandbox_metadata[session_hash]["created_at"]
                ).total_seconds()
                < SANDBOX_CREATION_TIMEOUT
            ):
                print(f"Reusing Sandbox for session {session_hash}")
                self.sandbox_metadata[session_hash]["last_accessed"] = current_time
                return SandboxResponse(
                    sandbox=self.sandboxes[session_hash], state="ready"
                )

            # Check if sandbox is already being created
            if (
                session_hash in self.sandbox_metadata
                and self.sandbox_metadata[session_hash].get("state") == "creating"
            ):
                print(f"Sandbox for session {session_hash} is already being created")
                return SandboxResponse(sandbox=None, state="creating")

            # Mark expired sandbox for cleanup (remove from dict within lock)
            if session_hash in self.sandboxes:
                print(f"Marking expired sandbox for session {session_hash} for cleanup")
                expired_sandbox = self.sandboxes[session_hash]
                del self.sandboxes[session_hash]
                if session_hash in self.sandbox_metadata:
                    del self.sandbox_metadata[session_hash]

            # Check if we have capacity
            # Count both ready sandboxes and sandboxes in "creating" state
            # We count BEFORE adding this one to ensure we don't exceed the limit
            creating_count = sum(
                1
                for meta in self.sandbox_metadata.values()
                if meta.get("state") == "creating"
            )
            # Check capacity BEFORE adding this session_hash to metadata
            if len(self.sandboxes) + creating_count >= self.max_sandboxes:
                return SandboxResponse(sandbox=None, state="max_sandboxes_reached")

            # Mark that we're creating this sandbox
            # This happens atomically within the lock, so no race condition
            print(f"Creating new sandbox for session {session_hash}")
            self.sandbox_metadata[session_hash] = {
                "state": "creating",
                "created_at": current_time,
                "last_accessed": current_time,
            }
            should_create = True

        # Start sandbox creation in background without waiting
        if should_create:
            asyncio.create_task(
                self._create_sandbox_background(session_hash, expired_sandbox)
            )

        # Check state after starting background task (it might complete very quickly)
        async with self.sandbox_lock:
            if session_hash in self.sandbox_metadata:
                state = self.sandbox_metadata[session_hash].get("state")
                if state == "creating":
                    return SandboxResponse(sandbox=None, state="creating")
                if state == "ready" and session_hash in self.sandboxes:
                    return SandboxResponse(
                        sandbox=self.sandboxes[session_hash], state="ready"
                    )

        # If metadata doesn't exist, it means creation failed immediately
        # Return "creating" anyway as the caller will retry
        return SandboxResponse(sandbox=None, state="creating")

    async def release_sandbox(self, session_hash: str):
        sandbox_to_kill = None

        # Remove from dictionaries under lock
        async with self.sandbox_lock:
            if session_hash in self.sandboxes:
                print(f"Releasing sandbox for session {session_hash}")
                sandbox_to_kill = self.sandboxes[session_hash]
                del self.sandboxes[session_hash]
            # Always clean up metadata, even if sandbox is still in "creating" state
            if session_hash in self.sandbox_metadata:
                state = self.sandbox_metadata[session_hash].get("state")
                if state == "creating":
                    print(
                        f"Cleaning up stuck 'creating' sandbox for session {session_hash}"
                    )
                del self.sandbox_metadata[session_hash]

        # Kill sandbox outside of lock
        if sandbox_to_kill:
            try:
                await asyncio.to_thread(sandbox_to_kill.kill)
            except Exception as e:
                print(f"Error killing sandbox for session {session_hash}: {str(e)}")

    async def cleanup_stuck_creating_sandboxes(self):
        """Clean up sandboxes that have been stuck in 'creating' state for too long"""
        current_time = datetime.now()
        stuck_sandboxes_to_kill = []

        async with self.sandbox_lock:
            for session_hash, metadata in list(self.sandbox_metadata.items()):
                if metadata.get("state") == "creating":
                    created_at = metadata.get("created_at")
                    if (
                        created_at
                        and (current_time - created_at).total_seconds()
                        > SANDBOX_CREATION_MAX_TIME
                    ):
                        print(
                            f"Cleaning up stuck 'creating' sandbox for session {session_hash} "
                            f"(stuck for {(current_time - created_at).total_seconds():.1f}s)"
                        )
                        # Collect sandbox to kill if it exists
                        if session_hash in self.sandboxes:
                            stuck_sandboxes_to_kill.append(
                                (session_hash, self.sandboxes[session_hash])
                            )
                            del self.sandboxes[session_hash]
                        del self.sandbox_metadata[session_hash]

        # Kill stuck sandboxes outside of lock
        for session_hash, sandbox in stuck_sandboxes_to_kill:
            try:
                await asyncio.to_thread(sandbox.kill)
                print(f"Killed stuck sandbox for session {session_hash}")
            except Exception as e:
                print(
                    f"Error killing stuck sandbox for session {session_hash}: {str(e)}"
                )

        return len(stuck_sandboxes_to_kill)

    async def cleanup_sandboxes(self):
        sandboxes_to_kill = []

        # Collect sandboxes under lock
        async with self.sandbox_lock:
            for session_hash in list(self.sandboxes.keys()):
                sandboxes_to_kill.append((session_hash, self.sandboxes[session_hash]))
                del self.sandboxes[session_hash]
                if session_hash in self.sandbox_metadata:
                    del self.sandbox_metadata[session_hash]

        # Kill all sandboxes outside of lock
        for session_hash, sandbox in sandboxes_to_kill:
            try:
                await asyncio.to_thread(sandbox.kill)
            except Exception as e:
                print(f"Error killing sandbox for session {session_hash}: {str(e)}")


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
