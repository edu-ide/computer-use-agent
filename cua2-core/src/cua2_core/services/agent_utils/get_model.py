import os
from smolagents import LiteLLMModel, Model

# 로컬 모드 여부
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"

# Available model IDs
AVAILABLE_MODELS = [
    "local-qwen3-vl",
] if USE_LOCAL else [
    "local-qwen3-vl",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
]

# 로컬 llama.cpp 서버 설정 (환경변수로 설정 가능)
LOCAL_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:8080/v1")


def get_model(model_id: str, max_tokens: int = 1024) -> Model:
    """
    Get the model with optimized settings for faster inference.

    Args:
        model_id: 모델 ID
        max_tokens: 최대 출력 토큰 수 (기본값: 1024, 장황한 출력 방지)
    """
    # 로컬 모드면 무조건 로컬 모델 사용
    if USE_LOCAL or model_id == "local-qwen3-vl":
        return LiteLLMModel(
            model_id="openai/qwen3-vl",
            api_base=LOCAL_API_BASE,
            api_key="none",
            max_tokens=max_tokens,  # 출력 토큰 제한
            temperature=0.1,  # 낮은 temperature로 결정적이고 빠른 출력
        )
    else:
        # HuggingFace Inference API
        from smolagents import InferenceClientModel
        return InferenceClientModel(bill_to="smolagents", model_id=model_id)
