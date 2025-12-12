#!/bin/bash

# 프로젝트 경로 설정
PROJECT_ROOT="/mnt/sda1/projects/computer-use-agent"
cd "$PROJECT_ROOT" || { echo "❌ 프로젝트 경로를 찾을 수 없습니다: $PROJECT_ROOT"; exit 1; }

echo "=================================================="
echo "   🤖 Computer Use Agent Launch Script"
echo "=================================================="

# 1. 기존 프로세스 정리
echo "🧹 기존 프로세스 정리 중..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 5173/tcp 2>/dev/null
sleep 1

# 2. 가상환경 활성화
echo "🔌 가상환경 활성화..."
source .venv/bin/activate

# 3. 환경 변수 설정
export USE_LOCAL=true
export PYTHONPATH="$PROJECT_ROOT/cua2-core/src:$PYTHONPATH"

# 4. 백엔드 시작
echo "🚀 백엔드 서버 시작 (Port 8000)..."
cd cua2-core
# 로그를 backend.log에 저장하고 백그라운드 실행
python3 -m uvicorn cua2_core.main:app --host 0.0.0.0 --port 8000 > "$PROJECT_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!

# 5. 프론트엔드 시작
echo "🚀 프론트엔드 서버 시작 (Port 5173)..."
cd ../cua2-front
# 로그를 frontend.log에 저장하고 백그라운드 실행
npm run dev > "$PROJECT_ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!

# 6. 시작 완료 메시지
echo ""
echo "✅ 서비스가 시작되었습니다!"
echo "--------------------------------------------------"
echo "🖥️  Web UI:   http://localhost:5173"
echo "⚙️  API Docs: http://localhost:8000/docs"
echo "📝 Logs:     $PROJECT_ROOT/backend.log"
echo "             $PROJECT_ROOT/frontend.log"
echo "--------------------------------------------------"
echo "종료하려면 Ctrl+C를 누르세요."

# 7. 종료 처리 (Ctrl+C 감지 시 프로세스 킬)
trap "echo '🛑 서버 종료 중...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

# 8. 대기
wait