#!/bin/bash

# SGLang 환경 경로
SGLANG_ENV="/mnt/sda1/sglang-env"
SGLANG_PYTHON="$SGLANG_ENV/bin/python"
SGLANG_PIP="$SGLANG_ENV/bin/pip"
MODEL_PATH="/mnt/sda1/models/llm/GELab-Zero-4B-preview"

# 라이브러리 경로 강제 지정 (CuDNN 인식 문제 해결 시도)
export LD_LIBRARY_PATH="$SGLANG_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$SGLANG_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

echo "=================================================="
echo "   🚀 SGLang Model Server (Interactive Mode)"
echo "=================================================="

# 1. 기존 프로세스 정리
echo "🛑 기존 프로세스 정리..."
pkill -f "sglang.launch_server"
sleep 2

# 2. 서버 시작 (Interactive)
echo "🚀 SGLang 서버 시작..."
echo "   - Model: $MODEL_PATH"
echo "   - Context Length: 32768"
echo "   - Port: 30001"
echo "--------------------------------------------------"

# 포그라운드 실행 (로그를 화면에 직접 출력)
$SGLANG_PYTHON -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port 30001 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --context-length 32768
