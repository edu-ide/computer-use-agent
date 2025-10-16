import asyncio
import json
from typing import Dict, Optional, Set

from fastapi import WebSocket

from backend.models.models import AgentMetadata, WebSocketEvent


class WebSocketManager:
    """Manages WebSocket connections and broadcasting"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        if websocket in self.connection_tasks:
            self.connection_tasks[websocket].cancel()
            del self.connection_tasks[websocket]
        print(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(
        self, message: WebSocketEvent, websocket: WebSocket
    ):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message.model_dump()))
        except Exception as e:
            print(f"Error sending personal message: {e}")
            # Only disconnect if the connection is still in our set
            if websocket in self.active_connections:
                self.disconnect(websocket)

    async def broadcast(self, message: WebSocketEvent):
        """Broadcast a message to all connected WebSockets"""
        if not self.active_connections:
            return

        # Create a list of connections to remove if they fail
        disconnected = []

        for connection in self.active_connections.copy():
            try:
                await connection.send_text(json.dumps(message.model_dump()))
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Remove failed connections
        for connection in disconnected:
            if connection in self.active_connections:
                self.disconnect(connection)

    async def send_agent_start(self, content: str, message_id: str):
        """Send agent start event"""
        event = WebSocketEvent(
            type="agent_start", content=content, messageId=message_id
        )
        await self.broadcast(event)

    async def send_agent_progress(self, content: str, message_id: str):
        """Send agent progress event"""
        event = WebSocketEvent(
            type="agent_progress", content=content, messageId=message_id
        )
        await self.broadcast(event)

    async def send_agent_complete(
        self, content: str, message_id: str, metadata: Optional[AgentMetadata] = None
    ):
        """Send agent complete event"""
        event = WebSocketEvent(
            type="agent_complete",
            content=content,
            messageId=message_id,
            metadata=metadata,
        )
        await self.broadcast(event)

    async def send_agent_error(self, content: str, message_id: Optional[str] = None):
        """Send agent error event"""
        event = WebSocketEvent(
            type="agent_error", content=content, messageId=message_id
        )
        await self.broadcast(event)

    async def send_vnc_url_set(self, vnc_url: str, content: Optional[str] = None):
        """Send VNC URL set event"""
        event = WebSocketEvent(
            type="vnc_url_set",
            content=content or f"VNC stream available at: {vnc_url}",
            vncUrl=vnc_url,
        )
        await self.broadcast(event)

    async def send_vnc_url_unset(self, content: Optional[str] = None):
        """Send VNC URL unset event (reset to default display)"""
        event = WebSocketEvent(
            type="vnc_url_unset",
            content=content or "VNC stream disconnected, showing default display",
        )
        await self.broadcast(event)

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
