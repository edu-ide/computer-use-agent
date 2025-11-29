#!/bin/bash
# Xvfb 테스트 스크립트

# 기존 정리
pkill -f "Xvfb :99" 2>/dev/null
pkill -f "cua-xvfb-profile-99" 2>/dev/null
rm -rf /tmp/cua-xvfb-profile-99
sleep 1

echo "1. Xvfb 시작..."
Xvfb :99 -screen 0 1280x720x24 &
XVFB_PID=$!
sleep 2

echo "2. Xvfb PID: $XVFB_PID"

echo "3. Chromium 시작 (새 프로필)..."
mkdir -p /tmp/cua-xvfb-profile-99
DISPLAY=:99 chromium-browser \
    --user-data-dir=/tmp/cua-xvfb-profile-99 \
    --no-first-run \
    --no-default-browser-check \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --disable-software-rasterizer \
    --start-maximized \
    https://google.com &
BROWSER_PID=$!

echo "4. Browser PID: $BROWSER_PID"
echo "5. 브라우저 로딩 대기 (10초)..."
sleep 10

echo "6. 스크린샷 캡처..."
DISPLAY=:99 import -window root /tmp/xvfb_test_result.png

echo "7. 결과:"
ls -la /tmp/xvfb_test_result.png
file /tmp/xvfb_test_result.png

echo ""
echo "테스트 완료. 정리하려면: pkill -f 'Xvfb :99'; pkill -f 'cua-xvfb-profile-99'"
