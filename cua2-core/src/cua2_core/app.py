import os
from contextlib import asynccontextmanager

from cua2_core.websocket.websocket_manager import WebSocketManager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# 로컬 모드 여부 확인
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    print("서비스 초기화 중...")

    websocket_manager = WebSocketManager()

    if USE_LOCAL:
        # 로컬 모드 - E2B 없이 로컬 데스크톱 사용
        print("로컬 모드로 실행 (E2B 없음)")
        from cua2_core.services.local_agent_service import LocalAgentService

        max_sandboxes = 1  # 로컬은 1개만
        agent_service = LocalAgentService(websocket_manager, max_sandboxes)
        sandbox_service = agent_service.sandbox_service

    else:
        # E2B 클라우드 모드
        print("E2B 클라우드 모드로 실행")
        if not os.getenv("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is not set")
        if not os.getenv("HF_TOKEN"):
            raise ValueError("HF_TOKEN is not set")

        from cua2_core.services.agent_service import AgentService
        from cua2_core.services.sandbox_service import SandboxService

        max_sandboxes = 600
        sandbox_service = SandboxService(max_sandboxes=max_sandboxes)
        agent_service = AgentService(websocket_manager, sandbox_service, max_sandboxes)
        sandbox_service.start_periodic_cleanup()

    # Store services in app state
    app.state.websocket_manager = websocket_manager
    app.state.sandbox_service = sandbox_service
    app.state.agent_service = agent_service
    app.state.use_local = USE_LOCAL

    print("서비스 초기화 완료")

    yield

    print("서비스 종료 중...")
    await agent_service.cleanup()
    if not USE_LOCAL:
        sandbox_service.stop_periodic_cleanup()
        await sandbox_service.cleanup_sandboxes()
    print("서비스 종료 완료")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Computer Use Backend (Local)",
    description="Backend API for Computer Use - AI-powered automation interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
