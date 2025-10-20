from contextlib import asynccontextmanager

from cua2_core.services.agent_service import AgentService
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

    # Initialize WebSocket manager
    websocket_manager = WebSocketManager()

    # Initialize agent service with websocket manager dependency
    agent_service = AgentService(websocket_manager)

    # Store services in app state for access in routes
    app.state.websocket_manager = websocket_manager
    app.state.agent_service = agent_service

    print("Services initialized successfully")

    yield

    # Shutdown: Clean up resources
    print("Shutting down services...")
    # Add any cleanup logic here if needed
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
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
