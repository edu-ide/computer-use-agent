from unittest.mock import Mock

import pytest
from cua2_core.models.models import AvailableModelsResponse, UpdateStepResponse
from cua2_core.routes.routes import router
from cua2_core.services.agent_service import AgentService
from cua2_core.services.agent_utils.get_model import AVAILABLE_MODELS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


@pytest.fixture
def mock_agent_service():
    """Fixture to create a mocked AgentService"""
    service = Mock(spec=AgentService)
    service.active_tasks = {}
    service.update_trace_step = Mock()
    return service


@pytest.fixture
def mock_websocket_manager():
    """Fixture to create a mocked WebSocketManager"""
    manager = Mock()
    manager.get_connection_count = Mock(return_value=0)
    return manager


@pytest.fixture
def app(mock_agent_service, mock_websocket_manager):
    """Fixture to create FastAPI app with mocked services"""
    # Create a test FastAPI app
    test_app = FastAPI(title="Test App")

    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the router
    test_app.include_router(router)

    # Mock the services in app state
    test_app.state.agent_service = mock_agent_service
    test_app.state.websocket_manager = mock_websocket_manager

    return test_app


@pytest.fixture
def client(app):
    """Fixture to create test client"""
    return TestClient(app)


class TestGetAvailableModels:
    """Test suite for GET /models endpoint"""

    def test_get_available_models_success(self, client):
        """Test successful retrieval of available models"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

        # Verify models match AVAILABLE_MODELS
        assert data["models"] == AVAILABLE_MODELS

    def test_get_available_models_structure(self, client):
        """Test that response matches AvailableModelsResponse schema"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        # Validate against Pydantic model
        models_response = AvailableModelsResponse(**data)
        assert models_response.models == AVAILABLE_MODELS

    def test_get_available_models_content(self, client):
        """Test that specific expected models are included"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        # Check for some specific models
        expected_models = [
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ]

        for model in expected_models:
            assert model in data["models"]


class TestUpdateTraceStep:
    """Test suite for PATCH /traces/{trace_id}/steps/{step_id} endpoint"""

    def test_update_trace_step_success(self, client, mock_agent_service):
        """Test successful step update"""
        trace_id = "test-trace-123"
        step_id = "1"
        request_data = {"step_evaluation": "like"}

        # Mock the service method to succeed
        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "Step updated successfully"

        # Verify the service was called correctly
        mock_agent_service.update_trace_step.assert_called_once_with(
            trace_id=trace_id, step_id=step_id, step_evaluation="like"
        )

    def test_update_trace_step_with_dislike(self, client, mock_agent_service):
        """Test step update with 'dislike' evaluation"""
        trace_id = "test-trace-456"
        step_id = "2"
        request_data = {"step_evaluation": "dislike"}

        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 200

        mock_agent_service.update_trace_step.assert_called_once_with(
            trace_id=trace_id, step_id=step_id, step_evaluation="dislike"
        )

    def test_update_trace_step_with_neutral(self, client, mock_agent_service):
        """Test step update with 'neutral' evaluation"""
        trace_id = "test-trace-789"
        step_id = "3"
        request_data = {"step_evaluation": "neutral"}

        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 200

        mock_agent_service.update_trace_step.assert_called_once_with(
            trace_id=trace_id, step_id=step_id, step_evaluation="neutral"
        )

    def test_update_trace_step_invalid_evaluation(self, client, mock_agent_service):
        """Test step update with invalid evaluation value"""
        trace_id = "test-trace-123"
        step_id = "1"
        request_data = {"step_evaluation": "invalid"}

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        # Should fail validation
        assert response.status_code == 422

    def test_update_trace_step_value_error(self, client, mock_agent_service):
        """Test step update when service raises ValueError"""
        trace_id = "test-trace-123"
        step_id = "invalid"
        request_data = {"step_evaluation": "like"}

        # Mock the service to raise ValueError
        mock_agent_service.update_trace_step.side_effect = ValueError(
            "Invalid step_id format"
        )

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 400
        assert "Invalid step_id format" in response.json()["detail"]

    def test_update_trace_step_not_found(self, client, mock_agent_service):
        """Test step update when trace is not found"""
        trace_id = "nonexistent-trace"
        step_id = "1"
        request_data = {"step_evaluation": "like"}

        # Mock the service to raise FileNotFoundError
        mock_agent_service.update_trace_step.side_effect = FileNotFoundError(
            "Trace not found"
        )

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 404
        assert "Trace not found" in response.json()["detail"]

    def test_update_trace_step_step_not_found(self, client, mock_agent_service):
        """Test step update when step doesn't exist in trace"""
        trace_id = "test-trace-123"
        step_id = "999"
        request_data = {"step_evaluation": "like"}

        # Mock the service to raise ValueError for step not found
        mock_agent_service.update_trace_step.side_effect = ValueError(
            "Step 999 not found in trace"
        )

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 400
        assert "Step 999 not found in trace" in response.json()["detail"]

    def test_update_trace_step_missing_request_body(self, client, mock_agent_service):
        """Test step update with missing request body"""
        trace_id = "test-trace-123"
        step_id = "1"

        response = client.patch(f"/traces/{trace_id}/steps/{step_id}", json={})

        # Should fail validation
        assert response.status_code == 422

    def test_update_trace_step_with_special_characters(
        self, client, mock_agent_service
    ):
        """Test step update with trace_id containing special characters"""
        trace_id = "trace-01K960P4FA2BVC058EZDXQEB5E-Qwen-Qwen3-VL-30B-A3B-Instruct"
        step_id = "1"
        request_data = {"step_evaluation": "like"}

        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 200

        mock_agent_service.update_trace_step.assert_called_once_with(
            trace_id=trace_id, step_id=step_id, step_evaluation="like"
        )

    def test_update_trace_step_response_structure(self, client, mock_agent_service):
        """Test that response matches UpdateStepResponse schema"""
        trace_id = "test-trace-123"
        step_id = "1"
        request_data = {"step_evaluation": "like"}

        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            f"/traces/{trace_id}/steps/{step_id}", json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        # Validate against Pydantic model
        update_response = UpdateStepResponse(**data)
        assert update_response.success is True
        assert update_response.message == "Step updated successfully"


class TestRoutesIntegration:
    """Integration tests for multiple routes"""

    def test_models_endpoint_available(self, client):
        """Test that models endpoint is available"""
        response = client.get("/models")
        assert response.status_code == 200

    def test_update_step_endpoint_available(self, client, mock_agent_service):
        """Test that update step endpoint is available"""
        mock_agent_service.update_trace_step.return_value = None

        response = client.patch(
            "/traces/test/steps/1", json={"step_evaluation": "like"}
        )
        assert response.status_code == 200

    def test_invalid_route(self, client):
        """Test accessing an invalid route"""
        response = client.get("/invalid-route")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
