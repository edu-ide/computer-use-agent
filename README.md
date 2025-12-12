# CUA2 - Computer Use Agent (컴퓨터 사용 에이전트)

CUA2는 실시간 에이전트 작업 처리, VNC 스트리밍, 단계별 실행 시각화 기능을 갖춘 AI 기반 자동화 인터페이스입니다.

## 📋 주요 기능

- **실시간 에이전트 제어**: AI 에이전트가 컴퓨터를 제어하여 작업을 수행합니다.
- **VNC 스트리밍**: 에이전트의 작업 화면을 실시간으로 확인하고 제어할 수 있습니다.
- **단계별 실행**: 에이전트의 사고 과정과 행동을 단계별로 시각화합니다.
- **로컬 및 Docker 환경 지원**: 로컬 개발 환경과 안전한 격리 환경(Docker/E2B)을 모두 지원합니다.

## 🛠️ 시스템 요구 사항

- **운영체제**: Linux (Ubuntu 20.04+ 권장)
- **Python**: 3.10 이상
- **Node.js**: 16 이상
- **기타 패키지**: Xvfb, xdotool, ImageMagick, Google Chrome (또는 Chromium)
  - `start-local.sh` 실행 시 자동으로 설치를 시도합니다.

## 🚀 빠른 시작 (로컬 모드)

가장 간편하게 실행하는 방법은 `start-local.sh` 스크립트를 사용하는 것입니다.

```bash
./start-local.sh
```

이 스크립트는 다음 작업을 자동으로 수행합니다:

1. 시스템 패키지(Xvfb 등) 확인 및 설치 (sudo 권한 필요)
2. Python 가상환경(`.venv`) 생성 및 활성화
3. 백엔드(`cua2-core`) 및 프론트엔드(`cua2-front`) 의존성 설치
4. 백엔드 서버(Port 8000) 및 프론트엔드 서버(Port 5173) 실행

실행이 완료되면 브라우저에서 아래 주소로 접속하세요:

- **Web UI**: [http://localhost:5173](http://localhost:5173)
- **API 문서**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📦 Docker 사용법

Docker를 사용하여 격리된 환경에서 실행할 수도 있습니다.

### 사전 준비

환경 변수 설정이 필요합니다.

- `E2B_API_KEY`: E2B 사용을 위한 API 키
- `HF_TOKEN`: Hugging Face 모델 사용을 위한 토큰

### 빌드 및 실행

```bash
# 이미지 빌드
make docker-build

# 컨테이너 실행 (환경 변수 필요)
export E2B_API_KEY=your_key_here
export HF_TOKEN=your_token_here
make docker-run
```

Docker 실행 시 서비스는 [http://localhost:7860](http://localhost:7860)에서 접근 가능합니다.

## 🔧 수동 설치 및 개발

스크립트를 사용하지 않고 직접 개발 환경을 구성하려면 `Makefile` 명령어를 사용할 수 있습니다.

### 의존성 설치

```bash
# 전체 의존성 동기화
make setup

# 또는 각각 설치
cd cua2-core && pip install -e .
cd cua2-front && npm install
```

### 서버 실행

터미널을 두 개 열어서 실행합니다.

**터미널 1 (백엔드):**

```bash
make dev-backend
# 또는
cd cua2-core && python3 -m uvicorn cua2_core.main:app --reload --host 0.0.0.0 --port 8000
```

**터미널 2 (프론트엔드):**

```bash
make dev-frontend
# 또는
cd cua2-front && npm run dev
```

## 📂 프로젝트 구조

- `cua2-core/`: FastAPI 기반 백엔드 서버 소스 코드
- `cua2-front/`: React/Vite 기반 프론트엔드 소스 코드
- `start-local.sh`: 로컬 개발 환경 자동 시작 스크립트
- `Makefile`: 개발 및 배포 편의를 위한 명령어 모음
- `Dockerfile`: Docker 이미지 빌드 설정

## ❓ 문제 해결

- **포트 충돌**: 8000번 또는 5173번 포트가 이미 사용 중인지 확인하세요.
- **권한 문제**: `start-local.sh` 실행 시 패키지 설치를 위해 sudo 비밀번호를 물어볼 수 있습니다.
- **의존성 오류**: Python 버전이 3.10 이상인지 확인하고 `make clean` 후 다시 시도해보세요
- cd computer-use-agent
  ./start-local.sh
