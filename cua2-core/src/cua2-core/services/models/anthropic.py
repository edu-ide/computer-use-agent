from smolagents import LiteLLMModel


class AnthropicModel(LiteLLMModel):
    """Anthropic model"""

    MODEL_TYPE = "anthropic"

    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
