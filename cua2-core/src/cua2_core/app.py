import os
from contextlib import asynccontextmanager

from cua2_core.services.agent_service import AgentService
from cua2_core.services.sandbox_service import SandboxService
from cua2_core.websocket.websocket_manager import WebSocketManager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Initialize services
    print("Initializing services...")

    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN is not set")

    num_workers = int(os.getenv("NUM_WORKERS", "12"))
    max_sandboxes = int(600 / num_workers)

    websocket_manager = WebSocketManager()

    sandbox_service = SandboxService(max_sandboxes=max_sandboxes)

    agent_service = AgentService(websocket_manager, sandbox_service, max_sandboxes)

    # Start periodic cleanup of stuck sandboxes
    sandbox_service.start_periodic_cleanup()

    # Store services in app state for access in routes
    app.state.websocket_manager = websocket_manager
    app.state.sandbox_service = sandbox_service
    app.state.agent_service = agent_service

    print("Services initialized successfully")

    yield

    print("Shutting down services...")
    sandbox_service.stop_periodic_cleanup()
    await agent_service.cleanup()
    await sandbox_service.cleanup_sandboxes()
    print("Services shut down successfully")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Computer Use Backend",
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
