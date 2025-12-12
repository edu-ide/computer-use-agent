import os
from smolagents import LiteLLMModel, Model

# 로컬 모드 여부
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"

# Available model IDs
AVAILABLE_MODELS = [
    "local-fara-7b",
] if USE_LOCAL else [
    "local-fara-7b",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
]

# SGLang Fara-7B 서버 설정 (환경변수로 설정 가능)
# Fara-7B는 포트 30001에서 실행
LOCAL_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:30001/v1")


def get_model(model_id: str, max_tokens: int = 1024) -> Model:
    """
    Get the model with optimized settings for faster inference.

    Args:
        model_id: 모델 ID
        max_tokens: 최대 출력 토큰 수 (기본값: 1024, 장황한 출력 방지)
    """
    # 로컬 모드면 SGLang 서버 사용 (LiteLLM Model)
    # model_id가 경로이거나 로컬 ID면 openai/ 접두어 붙여서 SGLang 호출
    if USE_LOCAL or model_id.startswith("/") or "local" in model_id:
        # LiteLLM에서는 custom provider pattern으로 "openai/custom_model_name" 형식을 사용
        # SGLang 서버는 model name을 무시하거나 로딩된 모델을 사용하므로
        # client 쪽에서 식별 가능한 임의의 이름이나 전달받은 ID를 사용
        
        # 만약 model_id가 전체 경로라면, SGLang이 로드한 모델명을 매칭하기 어려울 수 있으나
        # v1/chat/completions 호출 시 body의 model 필드로 전달됨.
        # SGLang은 요청된 model 필드가 서버에 로드된 모델과 다르면 에러를 낼 수도 있음.
        # 하지만 보통 단일 모델 서빙 시에는 유연하게 처리되기도 함.
        # 안전하게 "default" 또는 전달받은 model_id 사용.
        
        # 경로인 경우 파일명만 추출해서 사용해볼 수도 있음 (예: GELab-Zero-4B-preview)
        if "/" in model_id:
            safe_model_id = model_id.split("/")[-1]
        else:
            safe_model_id = model_id

        return LiteLLMModel(
            model_id=f"openai/{safe_model_id}",
            api_base=LOCAL_API_BASE,
            api_key="none",
            max_tokens=max_tokens,
            temperature=0.1,
        )
    else:
        # HuggingFace Inference API
        from smolagents import InferenceClientModel
        return InferenceClientModel(bill_to="smolagents", model_id=model_id)
