import os

import uvicorn
from cua2_core.app import app
from cua2_core.routes.routes import router
from cua2_core.routes.websocket import router as websocket_router

# Include routes
app.include_router(router, prefix="/api")
app.include_router(websocket_router)


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
        "cua2_core.app:app",
        host=host,
        port=port,
        # reload=debug,
        reload=True,
        log_level="info" if not debug else "debug",
    )
