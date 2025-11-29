"""
로컬 샌드박스 서비스 - E2B SandboxService 대체
로컬 데스크톱을 사용하여 에이전트 실행
"""

import asyncio
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from cua2_core.services.local_desktop import LocalDesktop


WIDTH = 1920
HEIGHT = 1080


class SandboxResponse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sandbox: LocalDesktop | None
    state: Literal["creating", "ready", "max_sandboxes_reached"]
    error: str | None = None


class SandboxEntry:
    """샌드박스 컨테이너"""

    def __init__(self, sandbox: LocalDesktop):
        self.sandbox = sandbox
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """로컬은 만료 없음"""
        return False

    def update_access(self):
        self.last_accessed = datetime.now()


class LocalSandboxService:
    """
    로컬 데스크톱 샌드박스 서비스
    E2B API_KEY 없이 로컬 데스크톱 사용
    """

    def __init__(self, max_sandboxes: int = 1):
        self.max_sandboxes = max_sandboxes
        self.sandboxes: dict[str, SandboxEntry] = {}
        self.pending: set[str] = set()
        self.creation_errors: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _create_and_setup_sandbox(self) -> LocalDesktop:
        """로컬 데스크톱 생성"""
        desktop = LocalDesktop(width=WIDTH, height=HEIGHT)
        return desktop

    async def acquire_sandbox(self, session_hash: str) -> SandboxResponse:
        """샌드박스 획득"""
        async with self.lock:
            # 기존 샌드박스 재사용
            if session_hash in self.sandboxes:
                entry = self.sandboxes[session_hash]
                entry.update_access()
                print(f"기존 로컬 데스크톱 재사용: {session_hash}")
                return SandboxResponse(sandbox=entry.sandbox, state="ready")

            # 생성 중인지 확인
            if session_hash in self.pending:
                if session_hash in self.creation_errors:
                    error_msg = self.creation_errors.pop(session_hash)
                    return SandboxResponse(sandbox=None, state="creating", error=error_msg)
                return SandboxResponse(sandbox=None, state="creating")

            # 용량 확인
            total_count = len(self.sandboxes) + len(self.pending)
            if total_count >= self.max_sandboxes:
                return SandboxResponse(sandbox=None, state="max_sandboxes_reached")

            self.pending.add(session_hash)
            print(f"로컬 데스크톱 생성 시작: {session_hash}")

        # 백그라운드에서 생성
        asyncio.create_task(self._create_sandbox_background(session_hash))
        return SandboxResponse(sandbox=None, state="creating")

    async def _create_sandbox_background(self, session_hash: str):
        """백그라운드에서 샌드박스 생성"""
        try:
            desktop = await asyncio.to_thread(self._create_and_setup_sandbox)
            print(f"로컬 데스크톱 생성 완료: {session_hash}")

            async with self.lock:
                self.pending.discard(session_hash)
                self.sandboxes[session_hash] = SandboxEntry(desktop)
                print(f"로컬 데스크톱 준비됨: {session_hash}")

        except Exception as e:
            error_msg = str(e)
            print(f"로컬 데스크톱 생성 오류 ({session_hash}): {error_msg}")

            async with self.lock:
                self.pending.discard(session_hash)
                self.creation_errors[session_hash] = error_msg

    async def release_sandbox(self, session_hash: str):
        """샌드박스 해제"""
        async with self.lock:
            if session_hash in self.sandboxes:
                entry = self.sandboxes.pop(session_hash)
                entry.sandbox.kill()
            self.pending.discard(session_hash)
            self.creation_errors.pop(session_hash, None)
        print(f"로컬 데스크톱 해제: {session_hash}")

    async def get_sandbox_counts(self) -> tuple[int, int]:
        """샌드박스 수 반환"""
        async with self.lock:
            return (len(self.sandboxes), len(self.pending))

    async def cleanup_expired_ready_sandboxes(self) -> int:
        """만료된 샌드박스 정리 (로컬은 만료 없음)"""
        return 0

    def start_periodic_cleanup(self):
        """주기적 정리 시작 (로컬은 불필요)"""
        pass

    def stop_periodic_cleanup(self):
        """주기적 정리 중지"""
        pass

    async def cleanup_sandboxes(self):
        """모든 샌드박스 정리"""
        async with self.lock:
            for entry in self.sandboxes.values():
                entry.sandbox.kill()
            self.sandboxes.clear()
