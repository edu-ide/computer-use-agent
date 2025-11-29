/**
 * 워크플로우 실행 상태 실시간 WebSocket 훅
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ExecutionStatus, VLMStepLog } from '@/services/workflowApi';

// WebSocket 메시지 타입
export interface WorkflowWsStatusMessage {
  type: 'status';
  execution_id: string;
  workflow_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  current_node: string | null;
  completed_nodes: string[];
  failed_nodes: string[];
  data: Record<string, unknown>;
  error: string | null;
  start_time: string | null;
  end_time: string | null;
  last_screenshot?: string | null;
  all_steps: VLMStepLog[];
}

export interface WorkflowWsStepMessage {
  type: 'step';
  step: VLMStepLog & { node_id: string };
}

export interface WorkflowWsCompleteMessage {
  type: 'complete';
  execution_id: string;
  status: string;
  error: string | null;
}

export interface WorkflowWsErrorMessage {
  type: 'error';
  message: string;
}

export type WorkflowWsMessage =
  | WorkflowWsStatusMessage
  | WorkflowWsStepMessage
  | WorkflowWsCompleteMessage
  | WorkflowWsErrorMessage;

interface UseWorkflowWebSocketOptions {
  executionId: string | null;
  onStatus?: (status: WorkflowWsStatusMessage) => void;
  onStep?: (step: VLMStepLog & { node_id: string }) => void;
  onComplete?: (status: string, error: string | null) => void;
  onError?: (message: string) => void;
}

/**
 * 워크플로우 WebSocket URL 생성
 */
const getWorkflowWsUrl = (executionId: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  const port = import.meta.env.VITE_API_PORT || '8000';

  // In production, use same port as frontend (assume reverse proxy)
  if (import.meta.env.PROD) {
    return `${protocol}//${window.location.host}/ws/workflow/${executionId}`;
  }

  // In development, use API port
  return `${protocol}//${host}:${port}/ws/workflow/${executionId}`;
};

export const useWorkflowWebSocket = ({
  executionId,
  onStatus,
  onStep,
  onComplete,
  onError,
}: UseWorkflowWebSocketOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<
    'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'
  >('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 3;

  // 연결
  const connect = useCallback(() => {
    if (!executionId) return;

    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const url = getWorkflowWsUrl(executionId);
    console.log(`워크플로우 WebSocket 연결 시도: ${url}`);

    try {
      setConnectionState('connecting');
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('워크플로우 WebSocket 연결됨');
        setIsConnected(true);
        setConnectionState('connected');
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WorkflowWsMessage;

          switch (data.type) {
            case 'status':
              onStatus?.(data);
              break;
            case 'step':
              onStep?.(data.step);
              break;
            case 'complete':
              onComplete?.(data.status, data.error);
              break;
            case 'error':
              onError?.(data.message);
              break;
          }
        } catch (error) {
          console.error('WebSocket 메시지 파싱 오류:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('워크플로우 WebSocket 오류:', error);
        setConnectionState('error');
      };

      ws.onclose = (event) => {
        console.log('워크플로우 WebSocket 연결 해제:', event.code, event.reason);
        setIsConnected(false);
        setConnectionState('disconnected');

        // 재연결 시도 (비정상 종료 && 최대 횟수 미달)
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 5000);
          console.log(`재연결 시도 예정: ${delay}ms 후`);

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('WebSocket 생성 실패:', error);
      setConnectionState('error');
    }
  }, [executionId, onStatus, onStep, onComplete, onError]);

  // 연결 해제
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    setConnectionState('disconnected');
    reconnectAttemptsRef.current = 0;
  }, []);

  // executionId가 변경되면 재연결
  useEffect(() => {
    if (executionId) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [executionId]);

  return {
    isConnected,
    connectionState,
    connect,
    disconnect,
  };
};
