import os

import uvicorn

from backend.app import app
from backend.routes.routes import router
from backend.routes.websocket import router as websocket_router

# Include routes
app.include_router(router, prefix="/api/v1")
app.include_router(websocket_router)


# Health check endpoint (without prefix)
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "computer-use-studio-backend"}


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting Computer Use Studio Backend on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"WebSocket endpoint: ws://{host}:{port}/ws")

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
