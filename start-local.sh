#!/bin/bash
# 로컬 Computer Use Agent 시작 스크립트

# 스크립트 위치 기반으로 BASE_DIR 설정
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BASE_DIR/.venv"

echo "=========================================="
echo "  Computer Use Agent (로컬 모드)"
echo "=========================================="
echo ""
echo "  프로젝트 경로: $BASE_DIR"
echo ""

# 시스템 패키지 확인 및 설치
echo "[0/6] 시스템 패키지 확인 중..."
MISSING_PKGS=""

# 필수 패키지 확인
command -v Xvfb >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS xvfb"
command -v xdotool >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS xdotool"
command -v convert >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS imagemagick"
command -v google-chrome >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS google-chrome-stable"

if [ -n "$MISSING_PKGS" ]; then
    echo "  - 필요한 패키지 설치 중:$MISSING_PKGS"
    sudo apt update && sudo apt install -y $MISSING_PKGS || { echo "시스템 패키지 설치 실패!"; exit 1; }
else
    echo "  - 시스템 패키지 확인 완료"
fi

# 이전 프로세스 종료
echo "[1/6] 이전 프로세스 정리 중..."
fuser -k 8000/tcp 2>/dev/null && echo "  - 포트 8000 프로세스 종료"
fuser -k 5173/tcp 2>/dev/null && echo "  - 포트 5173 프로세스 종료"
sleep 1

# 가상환경 생성 (없으면)
echo "[2/6] Python 가상환경 확인 중..."
if [ ! -d "$VENV_DIR" ]; then
    echo "  - 가상환경 생성 중..."
    python3 -m venv "$VENV_DIR" || { echo "가상환경 생성 실패!"; exit 1; }
else
    echo "  - 가상환경 존재함"
fi

# 가상환경 활성화
source "$VENV_DIR/bin/activate"
echo "  - 가상환경 활성화됨: $VIRTUAL_ENV"

# 환경 변수 설정
export USE_LOCAL=true
export PYTHONPATH="$BASE_DIR/cua2-core/src:$PYTHONPATH"

# Python 패키지 설치 (editable mode)
echo "[3/6] Python 패키지 설치 중..."
cd "$BASE_DIR/cua2-core"
if ! pip show cua2-core &>/dev/null; then
    echo "  - cua2-core 패키지 설치 (editable mode)..."
    pip install -e . || { echo "Python 패키지 설치 실패!"; exit 1; }
else
    echo "  - cua2-core 패키지 이미 설치됨"
fi

# langgraph 설치 확인
if ! pip show langgraph &>/dev/null; then
    echo "  - langgraph 설치 중..."
    pip install langgraph || { echo "langgraph 설치 실패!"; exit 1; }
fi

# aiosqlite 설치 확인 (AsyncSqliteSaver용)
if ! pip show aiosqlite &>/dev/null; then
    echo "  - aiosqlite 설치 중..."
    pip install aiosqlite || { echo "aiosqlite 설치 실패!"; exit 1; }
fi

# 프론트엔드 의존성 설치 (없으면)
echo "[4/6] 프론트엔드 의존성 확인 중..."
if [ ! -d "$BASE_DIR/cua2-front/node_modules" ]; then
    echo "  - npm 패키지 설치 중..."
    cd "$BASE_DIR/cua2-front" && npm install --legacy-peer-deps || { echo "npm 설치 실패!"; exit 1; }
else
    echo "  - node_modules 이미 존재"
fi

# 로그 디렉토리 생성
mkdir -p "$BASE_DIR/logs"

# 백엔드 시작 (포트 8000)
echo "[5/6] 백엔드 서버 시작 중... (포트 8000)"
cd "$BASE_DIR/cua2-core"
# python3 -m uvicorn cua2_core.main:app --reload --host 0.0.0.0 --port 8000 &
# 로그를 파일과 터미널 모두에 출력
python3 -m uvicorn cua2_core.main:app --reload --host 0.0.0.0 --port 8000 2>&1 | tee "$BASE_DIR/logs/backend.log" &
BACKEND_PID=$!

sleep 2

# 프론트엔드 시작 (포트 5173)
echo "[6/6] 프론트엔드 서버 시작 중... (포트 5173)"
cd "$BASE_DIR/cua2-front"
# npm run dev &
# 로그를 파일과 터미널 모두에 출력
npm run dev 2>&1 | tee "$BASE_DIR/logs/frontend.log" &
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
