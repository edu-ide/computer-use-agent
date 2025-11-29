import os

import uvicorn
from cua2_core.app import app
from cua2_core.routes.routes import router
from cua2_core.routes.websocket import router as websocket_router
from cua2_core.routes.coupang_routes import router as coupang_router
from cua2_core.routes.workflow_routes import router as workflow_router
from cua2_core.routes.trace_routes import router as trace_router
from cua2_core.routes.workflow_websocket import router as workflow_ws_router
from cua2_core.routes.agent_activity_routes import router as agent_activity_router

# Include routes
app.include_router(router, prefix="/api")
app.include_router(websocket_router)
app.include_router(coupang_router)  # /api/coupang/* 엔드포인트
app.include_router(workflow_router)  # /api/workflows/* 엔드포인트
app.include_router(trace_router)  # /api/traces/* 엔드포인트
app.include_router(workflow_ws_router)  # /ws/workflow/* WebSocket 엔드포인트
app.include_router(agent_activity_router)  # /api/agents/* 에이전트 활동 로그


# Health check endpoint (without prefix)
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "cua2-core"}


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting Computer Use Backend on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"WebSocket endpoint: ws://{host}:{port}/ws")

    uvicorn.run(
        "cua2_core.main:app",
        host=host,
        port=port,
        # reload=debug,
        reload=True,
        log_level="info" if not debug else "debug",
    )
