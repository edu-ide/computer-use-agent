import asyncio
import os
import time
from datetime import datetime
from typing import Any, Literal

from e2b_desktop import Sandbox
from pydantic import BaseModel

SANDBOX_METADATA: dict[str, dict[str, Any]] = {}
SANDBOX_TIMEOUT = 500
SANDBOX_READY_TIMEOUT = 200
SANDBOX_CREATION_MAX_TIME = (
    90  # Maximum time a sandbox can be in "creating" state (90 seconds)
    # Reduced from 120 to be more aggressive and clean up stuck sandboxes
    # before the agent_service loop times out (which waits 60 attempts * 2s = 120s)
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

        desktop = None
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
                    print(f"Sandbox {session_hash} is now ready")
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
            # Clean up metadata on failure - CRITICAL to prevent leaks
            async with self.sandbox_lock:
                if session_hash in self.sandbox_metadata:
                    state = self.sandbox_metadata[session_hash].get("state")
                    print(
                        f"Cleaning up failed sandbox creation for {session_hash} (state was: {state})"
                    )
                    del self.sandbox_metadata[session_hash]
                # Also remove from sandboxes dict if it somehow got added
                if session_hash in self.sandboxes:
                    del self.sandboxes[session_hash]
            # Kill the sandbox if it was partially created
            if desktop is not None:
                try:
                    await asyncio.to_thread(desktop.kill)
                    print(f"Killed partially created sandbox for {session_hash}")
                except Exception as kill_error:
                    print(f"Error killing partially created sandbox: {str(kill_error)}")

    async def _periodic_cleanup(self):
        """Background task to periodically clean up stuck creating sandboxes and expired ready sandboxes"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds (more aggressive)
                cleaned_creating = await self.cleanup_stuck_creating_sandboxes()
                cleaned_expired = await self.cleanup_expired_ready_sandboxes()
                if cleaned_creating > 0 or cleaned_expired > 0:
                    print(
                        f"Periodic cleanup: removed {cleaned_creating} stuck creating + "
                        f"{cleaned_expired} expired ready = {cleaned_creating + cleaned_expired} total"
                    )
                # Log sandbox pool state periodically for debugging
                async with self.sandbox_lock:
                    ready_count = len(self.sandboxes)
                    creating_count = sum(
                        1
                        for meta in self.sandbox_metadata.values()
                        if meta.get("state") == "creating"
                    )
                    total_count = ready_count + creating_count
                    if total_count > 0:
                        print(
                            f"Sandbox pool state: {ready_count} ready, {creating_count} creating, "
                            f"{total_count}/{self.max_sandboxes} total"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic cleanup: {str(e)}")

    def start_periodic_cleanup(self):
        """Start the periodic cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            except RuntimeError as e:
                # If called outside event loop, log but don't crash
                print(f"Warning: Cannot start periodic cleanup (no event loop): {e}")

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
                < SANDBOX_READY_TIMEOUT
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
                # Check if this sandbox has been stuck in "creating" state for too long
                created_at = self.sandbox_metadata[session_hash].get("created_at")
                if created_at:
                    stuck_duration = (current_time - created_at).total_seconds()
                    if stuck_duration > SANDBOX_CREATION_MAX_TIME:
                        # This sandbox is stuck - clean it up immediately
                        print(
                            f"Sandbox for session {session_hash} has been stuck in 'creating' state "
                            f"for {stuck_duration:.1f}s (threshold: {SANDBOX_CREATION_MAX_TIME}s) - cleaning up"
                        )
                        # Remove from metadata and sandboxes dict
                        if session_hash in self.sandboxes:
                            # Schedule kill outside of lock with error handling
                            stuck_sandbox = self.sandboxes[session_hash]
                            del self.sandboxes[session_hash]

                            async def kill_stuck():
                                try:
                                    await asyncio.to_thread(stuck_sandbox.kill)
                                except Exception as e:
                                    print(
                                        f"Error killing stuck sandbox for {session_hash}: {str(e)}"
                                    )

                            asyncio.create_task(kill_stuck())
                        del self.sandbox_metadata[session_hash]
                        # Fall through to create a new sandbox
                    else:
                        print(
                            f"Sandbox for session {session_hash} is already being created"
                        )
                        return SandboxResponse(sandbox=None, state="creating")
                else:
                    # Missing created_at - corrupted metadata, clean it up
                    print(
                        f"WARNING: Sandbox {session_hash} in 'creating' state has no 'created_at' - cleaning up"
                    )
                    if session_hash in self.sandboxes:
                        stuck_sandbox = self.sandboxes[session_hash]
                        del self.sandboxes[session_hash]

                        async def kill_stuck():
                            try:
                                await asyncio.to_thread(stuck_sandbox.kill)
                            except Exception as e:
                                print(
                                    f"Error killing stuck sandbox for {session_hash}: {str(e)}"
                                )

                        asyncio.create_task(kill_stuck())
                    del self.sandbox_metadata[session_hash]
                    # Fall through to create a new sandbox

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
            ready_count = len(self.sandboxes)
            total_count = ready_count + creating_count
            # Check capacity BEFORE adding this session_hash to metadata
            if total_count >= self.max_sandboxes:
                print(
                    f"Sandbox pool at capacity: {ready_count} ready + {creating_count} creating = "
                    f"{total_count}/{self.max_sandboxes}"
                )
                # CRITICAL: If we have an expired sandbox but can't create a new one,
                # we must still kill the expired sandbox to prevent leaks
                if expired_sandbox:
                    print(
                        f"Killing expired sandbox for {session_hash} even though pool is at capacity"
                    )

                    async def kill_expired():
                        try:
                            await asyncio.to_thread(expired_sandbox.kill)
                        except Exception as e:
                            print(
                                f"Error killing expired sandbox for {session_hash}: {str(e)}"
                            )

                    asyncio.create_task(kill_expired())
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
        elif expired_sandbox:
            # If we're not creating but have an expired sandbox, kill it
            # This shouldn't normally happen, but handle it defensively
            print(f"Killing expired sandbox for {session_hash} (not creating new one)")
            try:
                await asyncio.to_thread(expired_sandbox.kill)
            except Exception as e:
                print(f"Error killing expired sandbox: {str(e)}")

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
                    # Clean up if:
                    # 1. created_at exists and is older than threshold, OR
                    # 2. created_at is missing (corrupted metadata - should never happen but handle it)
                    should_cleanup = False
                    stuck_duration = 0.0

                    if created_at:
                        stuck_duration = (current_time - created_at).total_seconds()
                        if stuck_duration > SANDBOX_CREATION_MAX_TIME:
                            should_cleanup = True
                    else:
                        # Missing created_at is a bug, but clean it up anyway
                        print(
                            f"WARNING: Sandbox {session_hash} in 'creating' state has no 'created_at' timestamp - cleaning up"
                        )
                        should_cleanup = True
                        stuck_duration = float("inf")

                    if should_cleanup:
                        print(
                            f"Cleaning up stuck 'creating' sandbox for session {session_hash} "
                            f"(stuck for {stuck_duration:.1f}s)"
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

    async def cleanup_expired_ready_sandboxes(self):
        """Clean up ready sandboxes that have expired (not accessed for too long)"""
        current_time = datetime.now()
        expired_sandboxes_to_kill = []

        async with self.sandbox_lock:
            for session_hash, metadata in list(self.sandbox_metadata.items()):
                if metadata.get("state") == "ready" and session_hash in self.sandboxes:
                    created_at = metadata.get("created_at")
                    if (
                        created_at
                        and (current_time - created_at).total_seconds()
                        >= SANDBOX_READY_TIMEOUT
                    ):
                        print(
                            f"Cleaning up expired ready sandbox for session {session_hash} "
                            f"(age: {(current_time - created_at).total_seconds():.1f}s)"
                        )
                        expired_sandboxes_to_kill.append(
                            (session_hash, self.sandboxes[session_hash])
                        )
                        del self.sandboxes[session_hash]
                        del self.sandbox_metadata[session_hash]

        # Kill expired sandboxes outside of lock
        for session_hash, sandbox in expired_sandboxes_to_kill:
            try:
                await asyncio.to_thread(sandbox.kill)
                print(f"Killed expired ready sandbox for session {session_hash}")
            except Exception as e:
                print(
                    f"Error killing expired ready sandbox for session {session_hash}: {str(e)}"
                )

        return len(expired_sandboxes_to_kill)

    async def get_sandbox_counts(self) -> tuple[int, int]:
        """
        Get the count of available (ready) and non-available (creating) sandboxes.

        Returns:
            Tuple of (available_count, non_available_count)
        """
        async with self.sandbox_lock:
            ready_count = len(self.sandboxes)
            creating_count = sum(
                1
                for meta in self.sandbox_metadata.values()
                if meta.get("state") == "creating"
            )
            return (ready_count, creating_count)

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
