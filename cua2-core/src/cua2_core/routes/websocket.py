import json

# Get services from app state
from cua2_core.app import app
from cua2_core.models.models import AgentTrace, HeartbeatEvent
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Create router
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""

    websocket_manager = app.state.websocket_manager
    agent_service = app.state.agent_service

    await websocket_manager.connect(websocket)

    try:
        # Send welcome heartbeat
        welcome_message = HeartbeatEvent(type="heartbeat")
        await websocket_manager.send_personal_message(welcome_message, websocket)

        # Keep the connection alive and wait for messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()

                try:
                    # Parse the message
                    message_data = json.loads(data)
                    print(f"Received message: {message_data}")

                    # Check if it's a user task message
                    if message_data.get("type") == "user_task":
                        # Extract and parse the trace
                        trace_data = message_data.get("trace")
                        if trace_data:
                            # Convert timestamp string to datetime if needed
                            if isinstance(trace_data.get("timestamp"), str):
                                from datetime import datetime

                                trace_data["timestamp"] = datetime.fromisoformat(
                                    trace_data["timestamp"].replace("Z", "+00:00")
                                )

                            trace = AgentTrace(**trace_data)

                            # Process the user task with the trace
                            trace_id = await agent_service.process_user_task(trace)
                            print(f"Started processing trace: {trace_id}")
                        else:
                            print("No trace data in message")

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    from cua2_core.models.models import AgentErrorEvent

                    error_response = AgentErrorEvent(
                        type="agent_error", error="Invalid JSON format"
                    )
                    await websocket_manager.send_personal_message(
                        error_response, websocket
                    )

                except Exception as e:
                    print(f"Error processing message: {e}")
                    import traceback

                    traceback.print_exc()
                    from cua2_core.models.models import AgentErrorEvent

                    error_response = AgentErrorEvent(
                        type="agent_error", error=f"Error processing message: {str(e)}"
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
