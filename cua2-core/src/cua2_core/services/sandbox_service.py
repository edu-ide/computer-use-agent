import asyncio
import os
import time
from datetime import datetime
from typing import Literal

from e2b_desktop import Sandbox
from pydantic import BaseModel

SANDBOX_TIMEOUT = 500
SANDBOX_READY_TIMEOUT = 200  # Seconds before a sandbox expires
WIDTH = 1280
HEIGHT = 960


class SandboxResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sandbox: Sandbox | None
    state: Literal["creating", "ready", "max_sandboxes_reached"]
    error: str | None = None  # Error message if creation failed


class SandboxEntry:
    """Simple container for sandbox and its metadata"""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if sandbox has expired"""
        age = (datetime.now() - self.created_at).total_seconds()
        return age >= SANDBOX_READY_TIMEOUT

    def update_access(self):
        """Update last access time"""
        self.last_accessed = datetime.now()


class SandboxService:
    """
    Simplified sandbox service for production use.

    Key simplifications:
    - Non-blocking sandbox creation (background tasks)
    - Simple expiration-based cleanup
    - Minimal state tracking (pending vs ready)
    - Straightforward locking
    """

    def __init__(self, max_sandboxes: int = 50):
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is not set")
        self.max_sandboxes = max_sandboxes
        self.sandboxes: dict[str, SandboxEntry] = {}  # Ready sandboxes
        self.pending: set[str] = set()  # Session hashes currently being created
        self.creation_errors: dict[
            str, str
        ] = {}  # Track creation errors by session_hash
        self.lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _create_and_setup_sandbox(self) -> Sandbox:
        """Create and setup a sandbox (synchronous operation)"""
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

    async def acquire_sandbox(self, session_hash: str) -> SandboxResponse:
        """
        Acquire a sandbox for a session.
        Returns immediately - either with ready sandbox, or "creating" if one is being created.
        """
        async with self.lock:
            # Check if we have a valid sandbox for this session
            if session_hash in self.sandboxes:
                entry = self.sandboxes[session_hash]
                if not entry.is_expired():
                    entry.update_access()
                    print(f"Reusing sandbox for session {session_hash}")
                    return SandboxResponse(sandbox=entry.sandbox, state="ready")
                else:
                    # Expired - remove it
                    print(f"Removing expired sandbox for session {session_hash}")
                    old_entry = self.sandboxes.pop(session_hash)
                    # Schedule cleanup outside lock
                    asyncio.create_task(
                        self._kill_sandbox_safe(old_entry.sandbox, session_hash)
                    )

            # Check if already being created
            if session_hash in self.pending:
                print(f"Sandbox for session {session_hash} is already being created")
                # Check if there was a creation error
                if session_hash in self.creation_errors:
                    error_msg = self.creation_errors.pop(session_hash)
                    return SandboxResponse(
                        sandbox=None, state="creating", error=error_msg
                    )
                return SandboxResponse(sandbox=None, state="creating")

            # Check if there's a previous creation error (shouldn't happen, but handle it)
            if session_hash in self.creation_errors:
                error_msg = self.creation_errors.pop(session_hash)
                return SandboxResponse(sandbox=None, state="creating", error=error_msg)

            # Check capacity (count both ready and pending)
            total_count = len(self.sandboxes) + len(self.pending)
            if total_count >= self.max_sandboxes:
                print(
                    f"Sandbox pool at capacity: {len(self.sandboxes)} ready + {len(self.pending)} pending = {total_count}/{self.max_sandboxes}"
                )
                # Try to clean up expired sandboxes first
                await self._cleanup_expired_internal()
                # Recheck capacity after cleanup
                total_count = len(self.sandboxes) + len(self.pending)
                if total_count >= self.max_sandboxes:
                    return SandboxResponse(sandbox=None, state="max_sandboxes_reached")

            # Mark as pending and start creation in background
            self.pending.add(session_hash)
            print(f"Starting creation of sandbox for session {session_hash}")

        # Start creation in background (non-blocking)
        asyncio.create_task(self._create_sandbox_background(session_hash))
        return SandboxResponse(sandbox=None, state="creating")

    async def _create_sandbox_background(self, session_hash: str):
        """Background task to create a sandbox"""
        desktop = None
        try:
            desktop = await asyncio.to_thread(self._create_and_setup_sandbox)
            print(
                f"Sandbox created for session {session_hash}, ID: {desktop.sandbox_id}"
            )

            async with self.lock:
                # Check if session was released while creating (removed from pending)
                was_released = session_hash not in self.pending
                self.pending.discard(session_hash)

                if was_released:
                    # Session was released - kill the sandbox
                    print(
                        f"Session {session_hash} was released during creation, killing sandbox"
                    )
                    asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))
                    return

                # Check capacity before adding
                if len(self.sandboxes) >= self.max_sandboxes:
                    print(
                        f"Pool at capacity, killing newly created sandbox for {session_hash}"
                    )
                    asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))
                    return

                # Add to pool
                self.sandboxes[session_hash] = SandboxEntry(desktop)
                print(f"Sandbox {session_hash} is now ready")

        except Exception as e:
            error_msg = str(e)
            import traceback

            error_details = traceback.format_exc()
            print(f"Error creating sandbox for session {session_hash}: {error_msg}")
            print(f"Full traceback: {error_details}")

            async with self.lock:
                self.pending.discard(session_hash)
                # Store error so agent service can retrieve it
                self.creation_errors[session_hash] = error_msg
            if desktop:
                asyncio.create_task(self._kill_sandbox_safe(desktop, session_hash))

    async def release_sandbox(self, session_hash: str):
        """Release a sandbox for a session"""
        sandbox = None
        async with self.lock:
            if session_hash in self.sandboxes:
                entry = self.sandboxes.pop(session_hash)
                sandbox = entry.sandbox
            # Also remove from pending if it's there (creation will be cancelled)
            self.pending.discard(session_hash)
            # Clean up any stored errors
            self.creation_errors.pop(session_hash, None)

        if sandbox:
            await self._kill_sandbox_safe(sandbox, session_hash)
            print(f"Released sandbox for session {session_hash}")

    async def _kill_sandbox_safe(self, sandbox: Sandbox, session_hash: str):
        """Safely kill a sandbox with error handling"""
        try:
            await asyncio.to_thread(sandbox.kill)
        except Exception as e:
            print(f"Error killing sandbox for session {session_hash}: {str(e)}")

    async def _cleanup_expired_internal(self) -> int:
        """Internal cleanup of expired sandboxes (must be called with lock held)"""
        expired = []
        for session_hash, entry in list(self.sandboxes.items()):
            if entry.is_expired():
                expired.append((session_hash, entry.sandbox))
                del self.sandboxes[session_hash]

        # Kill expired sandboxes outside lock
        for session_hash, sandbox in expired:
            await self._kill_sandbox_safe(sandbox, session_hash)
            print(f"Cleaned up expired sandbox for session {session_hash}")

        return len(expired)

    async def cleanup_expired_ready_sandboxes(self) -> int:
        """Clean up expired ready sandboxes"""
        async with self.lock:
            return await self._cleanup_expired_internal()

    async def get_sandbox_counts(self) -> tuple[int, int]:
        """
        Get the count of available (ready) and non-available (pending) sandboxes.
        Returns: (available_count, non_available_count)
        """
        async with self.lock:
            available = len(self.sandboxes)
            non_available = len(self.pending)
            return (available, non_available)

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                async with self.lock:
                    cleaned = await self._cleanup_expired_internal()
                    if cleaned > 0:
                        print(f"Periodic cleanup: removed {cleaned} expired sandboxes")
                    # Log pool state
                    ready_count = len(self.sandboxes)
                    pending_count = len(self.pending)
                    total = ready_count + pending_count
                    if total > 0:
                        print(
                            f"Sandbox pool: {ready_count} ready, {pending_count} pending, {total}/{self.max_sandboxes} total"
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
                print(f"Warning: Cannot start periodic cleanup (no event loop): {e}")

    def stop_periodic_cleanup(self):
        """Stop the periodic cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    async def cleanup_sandboxes(self):
        """Clean up all sandboxes"""
        async with self.lock:
            sandboxes_to_kill = list(self.sandboxes.values())
            self.sandboxes.clear()

        for entry in sandboxes_to_kill:
            await self._kill_sandbox_safe(entry.sandbox, "cleanup")


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
