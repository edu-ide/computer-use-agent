from smolagents import Model

from backend.models.models import AgentType
from backend.services.models.anthropic import AnthropicModel


def get_model(model_id: str) -> tuple[Model, AgentType]:
    """Get the model"""
    if "sonnet" in model_id:
        return AnthropicModel(model_id=model_id), AgentType.PIXEL_COORDINATES
    else:
        raise ValueError(f"Model {model_id} not found")
