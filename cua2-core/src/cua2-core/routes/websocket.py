import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Get services from app state
from backend.app import app
from backend.models.models import UserTaskMessage, WebSocketEvent

# Create router
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""

    websocket_manager = app.state.websocket_manager
    agent_service = app.state.agent_service

    await websocket_manager.connect(websocket)

    try:
        welcome_message = WebSocketEvent(
            type="heartbeat",
            content="WebSocket connection established successfully",
            messageId="connection_welcome",
        )
        await websocket_manager.send_personal_message(welcome_message, websocket)

        # Keep the connection alive and wait for messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()

                try:
                    # Parse the message
                    message_data = json.loads(data)
                    message = UserTaskMessage(**message_data)

                    # Process the user task
                    if message.type == "user_task":
                        message_id = await agent_service.process_user_task(
                            message.content, message.model_id
                        )

                        # Send acknowledgment back to the client
                        response = WebSocketEvent(
                            type="agent_start",
                            content=f"Received task: {message.content}",
                            messageId=message_id,
                        )
                        await websocket_manager.send_personal_message(
                            response, websocket
                        )

                except json.JSONDecodeError:
                    error_response = WebSocketEvent(
                        type="agent_error", content="Invalid JSON format"
                    )
                    await websocket_manager.send_personal_message(
                        error_response, websocket
                    )

                except Exception as e:
                    print(f"Error processing message: {e}")
                    error_response = WebSocketEvent(
                        type="agent_error",
                        content=f"Error processing message: {str(e)}",
                    )
                    await websocket_manager.send_personal_message(
                        error_response, websocket
                    )

            except Exception as e:
                print(f"Error receiving WebSocket message: {e}")
                # If we can't receive messages, the connection is likely broken
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected normally")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        # Ensure cleanup happens
        websocket_manager.disconnect(websocket)
