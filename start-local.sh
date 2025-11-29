#!/bin/bash
# 로컬 Computer Use Agent 시작 스크립트

BASE_DIR=/home/sk/ws/llm/computer-use-agent

echo "=========================================="
echo "  Computer Use Agent (로컬 모드)"
echo "=========================================="
echo ""

# 환경 변수 설정
export USE_LOCAL=true
export PYTHONPATH="$BASE_DIR/cua2-core/src:$PYTHONPATH"

# 이전 프로세스 종료
echo "[0/3] 이전 프로세스 정리 중..."
# 포트 8000 사용 중인 프로세스 종료
fuser -k 8000/tcp 2>/dev/null && echo "  - 포트 8000 프로세스 종료"
# 포트 5173 사용 중인 프로세스 종료
fuser -k 5173/tcp 2>/dev/null && echo "  - 포트 5173 프로세스 종료"
sleep 1

# 프론트엔드 의존성 설치 (없으면)
if [ ! -d "$BASE_DIR/cua2-front/node_modules" ]; then
    echo "[1/3] 프론트엔드 의존성 설치 중..."
    cd $BASE_DIR/cua2-front && npm install
fi

# 백엔드 시작 (포트 8000)
echo "[2/3] 백엔드 서버 시작 중... (포트 8000)"
cd $BASE_DIR/cua2-core
python3 -m uvicorn cua2_core.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

# 프론트엔드 시작 (포트 5173)
echo "[3/3] 프론트엔드 서버 시작 중... (포트 5173)"
cd $BASE_DIR/cua2-front
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "  서버 시작 완료!"
echo "=========================================="
echo ""
echo "  WebUI:     http://localhost:5173"
echo "  백엔드:    http://localhost:8000"
echo "  API 문서:  http://localhost:8000/docs"
echo ""
echo "  종료: Ctrl+C"
echo "=========================================="

# 종료 시 프로세스 정리
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

wait
