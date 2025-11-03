from smolagents import InferenceClientModel, Model

# Available model IDs
AVAILABLE_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
]


def get_model(model_id: str) -> Model:
    """Get the model"""
    return InferenceClientModel(model_id=model_id)
