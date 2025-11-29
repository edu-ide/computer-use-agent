# CUA2-Core

Computer Use Agent 2.0 Core - VLM 기반 워크플로우 자동화 시스템

## 주요 기능

### 핵심 기능
- **SQLite 영구 Checkpointing**: 서버 재시작 후에도 상태 복구 가능
- **Streaming Mode 최적화**: updates 모드로 네트워크 효율성 75% 향상
- **planning_interval**: 매 스텝마다 계획 재수립 (적응형 실행)
- **Multi-Agent System**: 전문화된 에이전트들의 조합
- **Subgraphs**: 재사용 가능한 에러 처리 서브그래프
- **High-Level Tools**: 저수준 도구를 조합한 고수준 도구

### 고급 기능
- **Time Travel Debugging**: 실행 히스토리 조회 및 특정 시점으로 되돌리기
- **Dynamic Breakpoints**: 런타임 브레이크포인트로 디버깅
- **Tool Error Recovery**: 자동 에러 복구 (재시도, 대체 도구, 롤백)
- **Memory Persistence (SQLite)**: 버전 관리되는 영구 메모리 저장소
- **Parallel Node Execution**: 독립 노드 병렬 실행으로 성능 향상
- **WebSocket Human-in-the-Loop**: 실시간 사용자 확인/입력 요청

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                        cua2-core                             │
├─────────────────────────────────────────────────────────────┤
│  Workflows (LangGraph)                                       │
│  ├── workflow_base.py      # StateGraph, SQLite Checkpoint  │
│  ├── common_subgraphs.py   # 재사용 가능한 서브그래프       │
│  ├── error_handling.py     # LangGraph 조건부 에러 라우팅   │
│  └── coupang_workflow.py   # 예시 워크플로우                │
├─────────────────────────────────────────────────────────────┤
│  Services                                                    │
│  ├── orchestrator/         # ToolOrchestra 패턴             │
│  │   ├── strategy_selector.py  # 복잡도 분석 & 모델 선택   │
│  │   ├── workflow_monitor.py   # 실행 추적 & 리포트        │
│  │   └── step_evaluator.py     # 스텝 평가 & 학습          │
│  ├── multi_agent/          # Multi-Agent System             │
│  │   ├── manager_agent.py      # 에이전트 조정자           │
│  │   ├── search_agent.py       # 검색 전문                 │
│  │   ├── analysis_agent.py     # 분석 전문                 │
│  │   └── validation_agent.py   # 검증 전문                 │
│  ├── agent_utils/                                           │
│  │   └── high_level_tools.py   # 고수준 도구 모음          │
│  ├── vlm_agent_runner.py   # smolagents 기반 VLM 실행      │
│  ├── letta_memory_service.py  # SQLite 기반 영구 메모리    │
│  ├── tool_error_recovery.py   # 도구 에러 자동 복구        │
│  └── trace_store.py        # Trace 캐시                    │
├─────────────────────────────────────────────────────────────┤
│  Routes (FastAPI)                                            │
│  └── routes.py             # REST API 엔드포인트            │
└─────────────────────────────────────────────────────────────┘
```

## 주요 컴포넌트

### 1. Orchestrator 모듈 (`/services/orchestrator/`)

NVIDIA ToolOrchestra 논문(arXiv:2511.21689) 기반 구현

#### StrategySelector
- 노드 복잡도 분석 (비전, 추론, 데이터 추출 등)
- 복잡도 기반 모델 선택:
  - 단순 작업 → 로컬 모델 (Qwen-VL)
  - 중간 복잡도 → 경량 클라우드 (GPT-4o-mini)
  - 복잡한 작업 → 고성능 모델 (GPT-4o, Claude)

#### WorkflowMonitor
- 워크플로우 실행 상태 추적
- Stuck 노드 감지
- 실행 리포트 생성

#### StepEvaluator
- VLM 스텝별 실시간 평가
- 반복 실패 감지 및 Early Stop
- 상황별 프롬프트 주입 (팝업, 로딩 등)
- 패턴 학습 및 힌트 제공

### 2. LangGraph 에러 핸들링

VLM이 스크린샷을 보고 직접 에러를 감지하여 `[ERROR:TYPE]` 형식으로 보고합니다.

#### VLM 에러 타입
```python
class VLMErrorType(str, Enum):
    NONE = "none"
    BOT_DETECTED = "bot_detected"      # 봇 감지 → 즉시 중단
    PAGE_FAILED = "page_failed"        # 페이지 실패 → 재시도
    ACCESS_DENIED = "access_denied"    # 접근 거부 → 중단
    ELEMENT_NOT_FOUND = "element_not_found"  # 요소 미발견 → 스킵
    TIMEOUT = "timeout"                # 타임아웃 → 재시도
    UNKNOWN = "unknown"                # 알 수 없음 → 중단
```

#### 조건부 라우팅 (LangGraph)
```python
# workflow_base.py
def _create_router(self, node: WorkflowNode) -> Callable:
    def router(state: WorkflowState) -> str:
        vlm_error = state.get("vlm_error_type")
        retry_count = state.get("retry_count", 0)

        if vlm_error == "BOT_DETECTED":
            return node.on_bot_detected or "end"
        elif vlm_error in ("PAGE_FAILED", "TIMEOUT"):
            if retry_count < max_retries:
                return node.name  # 재시도
            return node.on_failure or "end"
        elif vlm_error == "ELEMENT_NOT_FOUND":
            return node.on_skip or node.on_success
        # ...
```

### 3. VLM Agent Runner (`smolagents`)

- `LocalVisionAgent`를 사용한 VLM 실행
- Playwright MCP 도구 통합
- 스크린샷 기반 행동 결정

### 4. Letta Memory Service

- 구조화된 메모리 블록 (Task, Progress, Failure Patterns)
- 실패 패턴 학습 및 Early Stop에 활용

## 설치

```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -e .
```

## 테스트

```bash
# Orchestrator 모듈 테스트
pytest tests/test_orchestrator.py -v

# 전체 테스트
pytest tests/ -v
```

## 사용 예시

```python
from cua2_core.services.orchestrator import (
    StrategySelector,
    WorkflowMonitor,
    StepEvaluator,
)

# 전략 선택
selector = StrategySelector(prefer_local=True)
decision = selector.decide(
    node_id="search",
    instruction="Search for products",
    params={"keyword": "iPhone"},
)
print(f"Strategy: {decision.strategy.value}, Model: {decision.model_id}")

# 워크플로우 모니터링
monitor = WorkflowMonitor()
monitor.start_workflow_tracking("wf", "exec-001", total_nodes=5)

# 스텝 평가
evaluator = StepEvaluator()
feedback = evaluator.evaluate_step(
    workflow_id="wf",
    node_id="node1",
    step_number=1,
    thought="Looking for button",
    action="click(button)",
    observation="Button clicked",
)
```

## 새 기능 사용 예시

### SQLite Checkpointing (워크플로우 복구)
```python
# 영구 저장소 사용 (기본값)
class MyWorkflow(WorkflowBase):
    def __init__(self):
        super().__init__(
            use_persistent_storage=True,  # SQLite 사용
            checkpoint_dir="~/.cua2/checkpoints",
        )

# 이전 실행에서 재개
state = await workflow.resume(thread_id="previous-execution-id")

# 체크포인트 조회
checkpoint = await workflow.get_checkpoint(thread_id)
```

### Multi-Agent System
```python
from cua2_core.services.multi_agent import ManagerAgent, WorkflowStep

# Manager Agent 생성
manager = ManagerAgent(
    manager_model=get_model("gpt-4o"),  # 조정용 강력한 모델
    agent_model=get_model("local-qwen3-vl"),  # 작업용 로컬 모델
)

# 워크플로우 실행
result = await manager.run_workflow([
    WorkflowStep("search", "Search for iPhone", {"keyword": "iPhone"}),
    WorkflowStep("analysis", "Extract products", depends_on=["search"]),
    WorkflowStep("validation", "Validate data", depends_on=["analysis"]),
], parallel=True)  # 독립적인 스텝은 병렬 실행

# 편의 메서드
result = await manager.search_and_analyze("노트북", pages=3, validate=True)
```

### Common Subgraphs (에러 처리 재사용)
```python
from cua2_core.workflows.common_subgraphs import CommonSubgraphs

class MyWorkflow(WorkflowBase):
    def build_graph(self):
        graph = StateGraph(WorkflowState)

        # 메인 노드
        graph.add_node("main_task", self._main_task)

        # 재사용 가능한 에러 핸들러 추가
        error_handler = CommonSubgraphs.create_error_handler(
            max_retries=3,
            on_bot_detected=self._handle_bot,
        )
        graph.add_node("error_handler", error_handler)

        # 조건부 라우팅
        graph.add_conditional_edges(
            "main_task",
            CommonSubgraphs.route_on_error,
            {"error": "error_handler", "success": END}
        )
```

### High-Level Tools
```python
from cua2_core.services.agent_utils.high_level_tools import create_high_level_tools

# 고수준 도구 생성
tools = create_high_level_tools(desktop)

agent = LocalVisionAgent(
    model=model,
    desktop=desktop,
    tools=tools,  # smart_click, fill_input_field, smooth_scroll 등
)

# 사용 가능한 도구:
# - smart_click(element_description): 텍스트로 요소 찾아 클릭
# - fill_input_field(x, y, text): 필드 클릭 + 입력
# - smooth_scroll(direction, amount): 부드러운 스크롤
# - wait_for_page_load(seconds): 페이지 로딩 대기
# - search_on_page(keyword): 페이지 내 검색 (Ctrl+F)
```

## 디렉토리 구조

```
src/cua2_core/
├── models/           # Pydantic 모델
├── routes/           # FastAPI 라우트
├── services/         # 핵심 서비스
│   ├── orchestrator/ # Orchestrator 모듈
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── strategy_selector.py
│   │   ├── workflow_monitor.py
│   │   └── step_evaluator.py
│   ├── multi_agent/  # Multi-Agent System
│   │   ├── manager_agent.py
│   │   ├── search_agent.py
│   │   ├── analysis_agent.py
│   │   └── validation_agent.py
│   ├── agent_utils/
│   │   └── high_level_tools.py
│   ├── vlm_agent_runner.py
│   ├── letta_memory_service.py
│   ├── trace_store.py
│   └── node_reuse_analyzer.py
└── workflows/        # 워크플로우 정의
    ├── workflow_base.py
    ├── common_subgraphs.py
    ├── error_handling.py
    └── coupang_workflow.py
```

---

## 고급 기능 상세

### 1. Time Travel Debugging

워크플로우 실행 히스토리를 조회하고 과거 시점으로 되돌아가 다시 실행할 수 있습니다.

```python
from cua2_core.workflows import WorkflowBase

# 실행 히스토리 조회
history = await workflow.get_state_history("exec-001", limit=20)
for state in history:
    print(f"Node: {state['current_node']}, Time: {state['timestamp']}")
    print(f"  Completed: {state['completed_nodes']}")

# 특정 체크포인트로 되돌아가서 다시 실행
checkpoint_id = history[3]["checkpoint_id"]
new_state = await workflow.replay_from_checkpoint("exec-001", checkpoint_id)

# 체크포인트에서 분기하여 새로운 실행 생성 (다른 파라미터로)
forked_state = await workflow.fork_from_checkpoint(
    source_thread_id="exec-001",
    checkpoint_id=checkpoint_id,
    new_thread_id="exec-001-fork",
    modified_state={
        "parameters": {"keyword": "다른 검색어"}
    }
)
```

### 2. Dynamic Breakpoints

런타임에 브레이크포인트를 설정하여 워크플로우를 디버깅할 수 있습니다.

```python
# 브레이크포인트 설정
workflow.add_breakpoint("search")
workflow.add_breakpoint("purchase")  # 위험한 작업 전

# 브레이크포인트 콜백 설정
async def on_breakpoint(node_name, state):
    print(f"Paused at {node_name}")
    print(f"Current data: {state.get('data')}")
    # 디버깅 작업 수행...

workflow.set_breakpoint_callback(on_breakpoint)

# 실행 - search 노드에서 자동으로 멈춤
await workflow.run(params)

# 브레이크포인트에서 재개
workflow.resume_from_breakpoint()

# Step Over - 한 노드만 실행하고 다시 멈춤
workflow.step_over()

# 브레이크포인트 관리
workflow.get_breakpoints()  # ["search", "purchase"]
workflow.remove_breakpoint("search")
workflow.clear_breakpoints()
```

#### REST API
```bash
# 브레이크포인트 조회
GET /workflow/{workflow_id}/breakpoints

# 브레이크포인트 추가
POST /workflow/{workflow_id}/breakpoints
{"node_name": "search"}

# 브레이크포인트 제거
DELETE /workflow/{workflow_id}/breakpoints/{node_name}

# 실행 재개
POST /workflow/{workflow_id}/resume

# Step Over
POST /workflow/{workflow_id}/step-over
```

### 3. Tool Error Recovery

도구 실행 중 발생하는 에러를 자동으로 복구합니다.

```python
from cua2_core.services.tool_error_recovery import (
    ToolErrorRecovery,
    RecoveryStrategy,
    get_tool_error_recovery,
)

# 싱글톤 인스턴스 사용
recovery = get_tool_error_recovery()

# 대체 도구 등록
recovery.register_alternative("click", "safe_click")
recovery.register_alternative("type_text", "smart_type")

# 에러 발생 시 복구 전략 결정
action = recovery.decide_recovery(
    tool_name="click",
    error_message="Element not found",
    attempt=1,
)

print(action.strategy)  # RecoveryStrategy.ALTERNATIVE
print(action.tool_name)  # "safe_click"

# 복구 실행
result = await recovery.execute_recovery(
    action,
    retry_func=lambda: retry_click(),
    alternative_func=lambda name: use_alternative(name),
)

if result.success:
    print("복구 성공:", result.new_result)
```

#### 복구 전략
| 전략 | 설명 | 적용 에러 타입 |
|------|------|---------------|
| `RETRY` | Exponential backoff로 재시도 | TIMEOUT, NETWORK_ERROR, RATE_LIMITED |
| `ALTERNATIVE` | 대체 도구 사용 | ELEMENT_NOT_FOUND |
| `ROLLBACK` | 이전 상태로 롤백 | 설정 가능 |
| `SKIP` | 현재 작업 건너뛰기 | INVALID_INPUT |
| `USER_INPUT` | 사용자 입력 요청 | 설정 가능 |
| `ABORT` | 실행 중단 | PERMISSION_DENIED, UNKNOWN |

### 4. Memory Persistence (SQLite)

메모리 블록을 SQLite에 영구 저장하고 버전 히스토리를 관리합니다.

```python
from cua2_core.services.letta_memory_service import (
    LettaMemoryService,
    WorkflowMemoryConfig,
    get_letta_memory_service,
)

# 서비스 인스턴스
memory_service = get_letta_memory_service()

# 워크플로우 에이전트 생성
config = WorkflowMemoryConfig.for_coupang(keyword="노트북")
agent_id = await memory_service.create_workflow_agent(config)

# 메모리 블록 업데이트 (자동으로 히스토리 저장)
await memory_service.update_memory_block(
    workflow_id="coupang-collect",
    block_label="workflow_state",
    value="현재 단계: 검색 중",
    change_reason="검색 시작",  # 변경 이유 기록
)

# 히스토리 조회
history = await memory_service.get_block_history(
    workflow_id="coupang-collect",
    block_label="workflow_state",
    limit=10,
)
for h in history:
    print(f"v{h['version']}: {h['created_at']} - {h['change_reason']}")

# 특정 버전으로 롤백
await memory_service.rollback_block(
    workflow_id="coupang-collect",
    block_label="workflow_state",
    version=3,
)

# 히스토리에서 검색
results = await memory_service.search_memory_history(
    workflow_id="coupang-collect",
    query="에러",
    limit=20,
)

# 통계 조회
stats = await memory_service.get_memory_stats("coupang-collect")
print(f"블록 수: {stats['block_count']}, 히스토리: {stats['history_count']}")
```

### 5. Parallel Node Execution

독립적인 노드들을 병렬로 실행하여 성능을 향상시킵니다.

```python
from cua2_core.workflows import WorkflowBase, WorkflowNode

# 노드 정의 시 parallel_group 설정
class MyWorkflow(WorkflowBase):
    @property
    def nodes(self):
        return [
            WorkflowNode(name="start", on_success="search_1"),
            # 같은 parallel_group의 노드들은 병렬 실행
            WorkflowNode(name="search_1", parallel_group="search"),
            WorkflowNode(name="search_2", parallel_group="search"),
            WorkflowNode(name="search_3", parallel_group="search"),
            # 의존성 설정
            WorkflowNode(
                name="analyze",
                depends_on=["search_1", "search_2", "search_3"],
                on_success="end",
            ),
        ]

# 방법 1: 병렬 실행 모드로 전체 워크플로우 실행
result = await workflow.run_with_parallel_execution(params)

# 방법 2: 특정 노드들만 병렬 실행
results = await workflow.run_parallel_nodes(
    ["search_1", "search_2", "search_3"],
    state,
    max_concurrency=5,  # 최대 동시 실행 수
)

for node_name, result in results.items():
    print(f"{node_name}: {'성공' if result.success else '실패'}")

# 방법 3: 병렬 그룹 실행 (의존성 자동 처리)
group_results = await workflow.run_parallel_group("search", state)
```

### 6. WebSocket Human-in-the-Loop

실시간으로 사용자 확인을 요청하고 응답을 받습니다.

```python
from cua2_core.workflows import WorkflowBase, WorkflowNode

# 확인이 필요한 노드 설정
class MyWorkflow(WorkflowBase):
    @property
    def nodes(self):
        return [
            WorkflowNode(
                name="purchase",
                requires_confirmation=True,  # 사용자 확인 필요
                confirmation_message="결제를 진행하시겠습니까?",
                is_dangerous=True,  # 위험한 작업 표시
            ),
        ]

# UI 콜백 설정 (WebSocket 알림용)
async def on_confirmation_needed(node_name, message, is_dangerous):
    await websocket.send_json({
        "type": "confirmation_required",
        "node": node_name,
        "message": message,
        "dangerous": is_dangerous,
    })

workflow.set_confirmation_callback(on_confirmation_needed)

# 실행 - purchase 노드에서 자동으로 대기
await workflow.run(params)

# UI에서 사용자 응답 처리
workflow.confirm(confirmed=True)  # 승인
workflow.confirm(confirmed=False)  # 취소
workflow.confirm(confirmed=True, user_input="CAPTCHA123")  # 입력과 함께 승인

# CAPTCHA/2FA 입력 요청
user_input = await workflow.request_user_input(
    prompt="CAPTCHA를 입력해주세요",
    input_type="captcha",
    timeout_sec=300,
)
```

#### WebSocket 이벤트 타입
```typescript
// 확인 요청
interface ConfirmationRequiredEvent {
  type: "confirmation_required";
  workflow_id: string;
  node_name: string;
  message: string;
  is_dangerous: boolean;
  input_type?: "text" | "captcha" | "2fa";
}

// 확인 완료
interface ConfirmationReceivedEvent {
  type: "confirmation_received";
  workflow_id: string;
  node_name: string;
  confirmed: boolean;
  user_input?: string;
}

// 워크플로우 상태 업데이트
interface WorkflowStateUpdateEvent {
  type: "workflow_state_update";
  workflow_id: string;
  execution_id: string;
  status: "running" | "completed" | "failed" | "stopped";
  current_node?: string;
  completed_nodes: string[];
  failed_nodes: string[];
  progress_percent: number;
  error?: string;
}

// 브레이크포인트 이벤트
interface BreakpointHitEvent {
  type: "breakpoint_hit";
  workflow_id: string;
  node_name: string;
  state?: object;
}

interface BreakpointResumedEvent {
  type: "breakpoint_resumed";
  workflow_id: string;
  node_name: string;
}
```

#### REST API
```bash
# 확인 대기 상태 조회
GET /workflow/{workflow_id}/confirmation

# 사용자 확인 응답
POST /workflow/{workflow_id}/confirm
{"confirmed": true, "user_input": "optional input"}
```

---

## 참조

- [NVIDIA ToolOrchestra](https://arxiv.org/abs/2511.21689) - 모델 라우팅 패턴
- [LangGraph](https://github.com/langchain-ai/langgraph) - 상태 그래프 기반 워크플로우
- [smolagents](https://github.com/huggingface/smolagents) - VLM 에이전트 실행
