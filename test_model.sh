#!/bin/bash

# 설정
API_URL="http://localhost:30001/v1/chat/completions"
MODEL="Fara-7B"

# 1x1 빨간색 픽셀 이미지 (Base64)
IMAGE_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

echo "=================================================="
echo "   Fara-7B Model Test Script"
echo "=================================================="
echo "Target: $API_URL"
echo ""

# 1. 텍스트 전용 테스트
echo "[Test 1] Basic Text Chat..."
RESPONSE_TEXT=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"Hello! Are you working?\"}
    ],
    \"max_tokens\": 50
  }")

echo "Response:"
echo "$RESPONSE_TEXT"
echo ""

# 2. 멀티모달(비전) 테스트
echo "[Test 2] Multimodal Vision..."
RESPONSE_VISION=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"What color is this image?\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_BASE64\"}}
        ]
      }
    ],
    \"max_tokens\": 50
  }")

echo "Response:"
echo "$RESPONSE_VISION"
echo ""

echo "=================================================="
echo "Done."
