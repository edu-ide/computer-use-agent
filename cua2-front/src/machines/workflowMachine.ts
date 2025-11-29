/**
 * 워크플로우 상태 머신 - XState v5
 *
 * 상태 전환:
 *   idle -> starting -> connecting -> running -> (completed | failed | stopped)
 *                                         |
 *                                         +-> stopping -> stopped
 *
 * 이벤트:
 *   - START: 워크플로우 시작
 *   - WS_CONNECTED: WebSocket 연결됨
 *   - WS_DISCONNECTED: WebSocket 연결 끊김
 *   - WS_STATUS: 백엔드에서 상태 업데이트 수신
 *   - WS_STEP: 새로운 스텝 수신
 *   - WS_COMPLETE: 워크플로우 완료
 *   - WS_ERROR: 에러 발생
 *   - STOP: 사용자가 중지 요청
 *   - RESET: 상태 초기화
 */

import { setup, assign, fromPromise } from 'xstate';
import {
  startWorkflow,
  stopWorkflow,
  WorkflowDefinition,
  ExecutionStatus,
} from '@/services/workflowApi';
import { VLMStep } from '@/components/workflow';

// 컨텍스트 타입
export interface WorkflowContext {
  // 워크플로우 정보
  workflowId: string | null;
  workflow: WorkflowDefinition | null;
  parameters: Record<string, unknown>;

  // 실행 정보
  executionId: string | null;
  executionStatus: ExecutionStatus | null;

  // 노드 상태
  currentNode: string | null;
  completedNodes: string[];
  failedNodes: string[];

  // 현재 노드 상세 정보 (실시간 모니터링용)
  currentNodeStartTime: string | null;
  currentThought: string | null;
  currentAction: string | null;
  currentObservation: string | null;
  stepCount: number | null;

  // 에러 추적 정보
  consecutiveErrors: number | null;
  lastError: string | null;

  // VLM 스텝
  vlmSteps: VLMStep[];
  lastScreenshot: string | null;

  // 에러
  error: string | null;

  // WebSocket 상태
  wsConnected: boolean;
}

// 이벤트 타입
export type WorkflowEvent =
  | { type: 'START'; workflowId: string; parameters: Record<string, unknown> }
  | { type: 'WS_CONNECTED' }
  | { type: 'WS_DISCONNECTED' }
  | {
      type: 'WS_STATUS';
      status: ExecutionStatus;
      allSteps?: VLMStep[];
      lastScreenshot?: string | null;
      // 현재 노드 상세 정보
      currentNodeStartTime?: string | null;
      currentThought?: string | null;
      currentAction?: string | null;
      currentObservation?: string | null;
      stepCount?: number | null;
      // 에러 추적 정보
      consecutiveErrors?: number | null;
      lastError?: string | null;
    }
  | { type: 'WS_STEP'; step: VLMStep }
  | { type: 'WS_COMPLETE'; status: string; error: string | null }
  | { type: 'WS_ERROR'; message: string }
  | { type: 'STOP' }
  | { type: 'RESET' }
  | { type: 'SET_WORKFLOW'; workflow: WorkflowDefinition }
  | { type: 'SET_PARAMETERS'; parameters: Record<string, unknown> };

// 초기 컨텍스트
const initialContext: WorkflowContext = {
  workflowId: null,
  workflow: null,
  parameters: {},
  executionId: null,
  executionStatus: null,
  currentNode: null,
  completedNodes: [],
  failedNodes: [],
  // 현재 노드 상세 정보
  currentNodeStartTime: null,
  currentThought: null,
  currentAction: null,
  currentObservation: null,
  stepCount: null,
  // 에러 추적 정보
  consecutiveErrors: null,
  lastError: null,
  vlmSteps: [],
  lastScreenshot: null,
  error: null,
  wsConnected: false,
};

// 상태 머신 정의
export const workflowMachine = setup({
  types: {
    context: {} as WorkflowContext,
    events: {} as WorkflowEvent,
  },
  actions: {
    setWorkflow: assign({
      workflow: ({ event }) => {
        if (event.type === 'SET_WORKFLOW') return event.workflow;
        return null;
      },
      workflowId: ({ event }) => {
        if (event.type === 'SET_WORKFLOW') return event.workflow.config.id;
        return null;
      },
    }),
    setParameters: assign({
      parameters: ({ event }) => {
        if (event.type === 'SET_PARAMETERS') return event.parameters;
        return {};
      },
    }),
    setExecutionId: assign({
      executionId: (_, params: { executionId: string }) => params.executionId,
    }),
    setWsConnected: assign({
      wsConnected: ({ event }) => event.type === 'WS_CONNECTED',
    }),
    setWsDisconnected: assign({
      wsConnected: false,
    }),
    updateFromStatus: assign(({ context, event }) => {
      if (event.type !== 'WS_STATUS') return {};

      return {
        executionStatus: event.status,
        currentNode: event.status.current_node,
        completedNodes: event.status.completed_nodes,
        failedNodes: event.status.failed_nodes,
        vlmSteps: event.allSteps || context.vlmSteps,
        lastScreenshot: event.lastScreenshot ?? context.lastScreenshot,
        error: event.status.error,
        // 현재 노드 상세 정보
        currentNodeStartTime: event.currentNodeStartTime ?? null,
        currentThought: event.currentThought ?? null,
        currentAction: event.currentAction ?? null,
        currentObservation: event.currentObservation ?? null,
        stepCount: event.stepCount ?? null,
        // 에러 추적 정보
        consecutiveErrors: event.consecutiveErrors ?? null,
        lastError: event.lastError ?? null,
      };
    }),
    addStep: assign({
      vlmSteps: ({ context, event }) => {
        if (event.type !== 'WS_STEP') return context.vlmSteps;
        return [...context.vlmSteps, event.step];
      },
    }),
    setError: assign({
      error: ({ event }) => {
        if (event.type === 'WS_ERROR') return event.message;
        if (event.type === 'WS_COMPLETE') return event.error;
        return null;
      },
    }),
    clearError: assign({
      error: null,
    }),
    resetContext: assign(initialContext),
    prepareForStart: assign({
      vlmSteps: [],
      completedNodes: [],
      failedNodes: [],
      currentNode: null,
      error: null,
      lastScreenshot: null,
      executionStatus: null,
      // 현재 노드 상세 정보 초기화
      currentNodeStartTime: null,
      currentThought: null,
      currentAction: null,
      currentObservation: null,
      stepCount: null,
      consecutiveErrors: null,
      lastError: null,
    }),
  },
  guards: {
    isCompleted: ({ event }) => {
      if (event.type === 'WS_STATUS') {
        return event.status.status === 'completed';
      }
      if (event.type === 'WS_COMPLETE') {
        return event.status === 'completed';
      }
      return false;
    },
    isFailed: ({ event }) => {
      if (event.type === 'WS_STATUS') {
        return event.status.status === 'failed';
      }
      if (event.type === 'WS_COMPLETE') {
        return event.status === 'failed';
      }
      return false;
    },
    isStopped: ({ event }) => {
      if (event.type === 'WS_STATUS') {
        return event.status.status === 'stopped';
      }
      if (event.type === 'WS_COMPLETE') {
        return event.status === 'stopped';
      }
      return false;
    },
    // 종료 상태가 아닌 경우 (running, pending 등)
    isNotTerminal: ({ event }) => {
      if (event.type === 'WS_STATUS') {
        const status = event.status.status;
        return !['completed', 'failed', 'stopped'].includes(status);
      }
      return true;
    },
  },
  actors: {
    startWorkflowActor: fromPromise(
      async ({
        input,
      }: {
        input: { workflowId: string; parameters: Record<string, unknown> };
      }) => {
        const result = await startWorkflow(input.workflowId, input.parameters);
        return result.execution_id;
      }
    ),
    stopWorkflowActor: fromPromise(
      async ({ input }: { input: { executionId: string } }) => {
        await stopWorkflow(input.executionId);
      }
    ),
  },
}).createMachine({
  id: 'workflow',
  initial: 'idle',
  context: initialContext,
  states: {
    // 대기 상태
    idle: {
      on: {
        SET_WORKFLOW: {
          actions: 'setWorkflow',
        },
        SET_PARAMETERS: {
          actions: 'setParameters',
        },
        START: {
          target: 'starting',
          actions: ['prepareForStart', 'clearError'],
        },
      },
    },

    // 워크플로우 시작 요청 중
    starting: {
      invoke: {
        id: 'startWorkflow',
        src: 'startWorkflowActor',
        input: ({ context, event }) => {
          if (event.type === 'START') {
            return {
              workflowId: event.workflowId,
              parameters: event.parameters,
            };
          }
          return {
            workflowId: context.workflowId!,
            parameters: context.parameters,
          };
        },
        onDone: {
          target: 'connecting',
          actions: assign({
            executionId: ({ event }) => event.output,
          }),
        },
        onError: {
          target: 'failed',
          actions: assign({
            error: ({ event }) => String(event.error),
          }),
        },
      },
    },

    // WebSocket 연결 대기
    connecting: {
      on: {
        WS_CONNECTED: {
          target: 'running',
          actions: 'setWsConnected',
        },
        WS_STATUS: {
          // 연결 전에 상태가 먼저 올 수도 있음
          target: 'running',
          actions: ['setWsConnected', 'updateFromStatus'],
        },
        WS_ERROR: {
          target: 'failed',
          actions: 'setError',
        },
      },
      after: {
        // 10초 후 타임아웃
        10000: {
          target: 'failed',
          actions: assign({
            error: 'WebSocket 연결 타임아웃',
          }),
        },
      },
    },

    // 실행 중
    running: {
      on: {
        WS_STATUS: [
          {
            guard: 'isCompleted',
            target: 'completed',
            actions: 'updateFromStatus',
          },
          {
            guard: 'isFailed',
            target: 'failed',
            actions: 'updateFromStatus',
          },
          {
            guard: 'isStopped',
            target: 'stopped',
            actions: 'updateFromStatus',
          },
          {
            // 기본 핸들러: 종료 상태가 아닌 모든 상태 (running, pending 등)
            // guard 없이 항상 실행되어 상태 업데이트
            actions: 'updateFromStatus',
          },
        ],
        WS_STEP: {
          actions: 'addStep',
        },
        WS_COMPLETE: [
          {
            guard: 'isCompleted',
            target: 'completed',
          },
          {
            guard: 'isFailed',
            target: 'failed',
            actions: 'setError',
          },
          {
            guard: 'isStopped',
            target: 'stopped',
          },
        ],
        WS_ERROR: {
          target: 'failed',
          actions: 'setError',
        },
        WS_DISCONNECTED: {
          actions: 'setWsDisconnected',
          // 실행 중 연결이 끊기면 재연결 시도 (WebSocket 훅에서 처리)
        },
        STOP: {
          target: 'stopping',
        },
      },
    },

    // 중지 요청 중
    stopping: {
      invoke: {
        id: 'stopWorkflow',
        src: 'stopWorkflowActor',
        input: ({ context }) => ({
          executionId: context.executionId!,
        }),
        onDone: {
          // 백엔드에서 stopped 상태가 올 때까지 대기
        },
        onError: {
          // 에러가 나도 stopped로 전환
          target: 'stopped',
        },
      },
      on: {
        WS_STATUS: [
          {
            guard: 'isStopped',
            target: 'stopped',
            actions: 'updateFromStatus',
          },
          {
            guard: 'isCompleted',
            target: 'completed',
            actions: 'updateFromStatus',
          },
          {
            guard: 'isFailed',
            target: 'failed',
            actions: 'updateFromStatus',
          },
        ],
        WS_COMPLETE: [
          {
            guard: 'isStopped',
            target: 'stopped',
          },
          {
            guard: 'isCompleted',
            target: 'completed',
          },
          {
            guard: 'isFailed',
            target: 'failed',
            actions: 'setError',
          },
        ],
      },
      after: {
        // 5초 후 강제 stopped
        5000: {
          target: 'stopped',
        },
      },
    },

    // 완료 상태
    completed: {
      on: {
        RESET: {
          target: 'idle',
          actions: 'resetContext',
        },
        START: {
          target: 'starting',
          actions: ['prepareForStart', 'clearError'],
        },
        SET_PARAMETERS: {
          actions: 'setParameters',
        },
      },
    },

    // 실패 상태
    failed: {
      on: {
        RESET: {
          target: 'idle',
          actions: 'resetContext',
        },
        START: {
          target: 'starting',
          actions: ['prepareForStart', 'clearError'],
        },
        SET_PARAMETERS: {
          actions: 'setParameters',
        },
      },
    },

    // 중지됨 상태
    stopped: {
      on: {
        RESET: {
          target: 'idle',
          actions: 'resetContext',
        },
        START: {
          target: 'starting',
          actions: ['prepareForStart', 'clearError'],
        },
        SET_PARAMETERS: {
          actions: 'setParameters',
        },
      },
    },
  },
});

// 상태 타입 (UI에서 사용)
export type WorkflowState =
  | 'idle'
  | 'starting'
  | 'connecting'
  | 'running'
  | 'stopping'
  | 'completed'
  | 'failed'
  | 'stopped';

// 상태 체크 유틸리티
export const isRunningState = (state: WorkflowState): boolean =>
  ['starting', 'connecting', 'running', 'stopping'].includes(state);

export const isFinishedState = (state: WorkflowState): boolean =>
  ['completed', 'failed', 'stopped'].includes(state);
