import { useAgentStore } from '@/stores/agentStore';
import { AgentTrace, AgentTraceMetadata, WebSocketEvent } from '@/types/agent';
import { useCallback, useEffect } from 'react';
import { useWebSocket } from './useWebSocket';

interface UseAgentWebSocketOptions {
  url: string;
}

export const useAgentWebSocket = ({ url }: UseAgentWebSocketOptions) => {
  const {
    setTrace,
    traceId,
    setTraceId,
    updateTraceWithStep,
    completeTrace,
    setIsAgentProcessing,
    setIsConnectingToE2B,
    setVncUrl,
    setError,
    setIsConnected,
    selectedModelId,
    resetAgent,
  } = useAgentStore();

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = useCallback(
    (event: WebSocketEvent) => {
      console.log('WebSocket event received:', event);

      switch (event.type) {
        case 'agent_start': {
          // Clear previous state (especially finalStep)
          resetAgent();

          setIsAgentProcessing(true);
          setIsConnectingToE2B(true); // Start connecting to E2B
          setError(undefined); // Clear any previous error

          // Ensure trace has proper metadata with default maxSteps if not provided
          const traceWithMetadata = {
            ...event.agentTrace,
            traceMetadata: event.agentTrace.traceMetadata ? {
              ...event.agentTrace.traceMetadata,
              maxSteps: event.agentTrace.traceMetadata.maxSteps > 0
                ? event.agentTrace.traceMetadata.maxSteps
                : 200, // Default if backend sends 0
            } : {
              traceId: event.agentTrace.id,
              inputTokensUsed: 0,
              outputTokensUsed: 0,
              duration: 0,
              numberOfSteps: 0,
              maxSteps: 200,
              completed: false,
              final_state: null,
            },
          };

          setTrace(traceWithMetadata);
          console.log('Agent start received:', traceWithMetadata);
          break;
        }

        case 'agent_progress':
          // Add new step from agent trace run with image, generated text, actions, tokens and timestamp
          setIsConnectingToE2B(false); // Connected! First step received
          updateTraceWithStep(event.agentStep, event.traceMetadata);
          console.log('Agent progress received:', event.agentStep);
          break;

        case 'agent_complete':
          setIsAgentProcessing(false);
          setIsConnectingToE2B(false);
          completeTrace(event.traceMetadata, event.final_state);
          console.log('Agent complete received:', event.traceMetadata, 'Final state:', event.final_state);
          break;

        case 'agent_error':
          setIsAgentProcessing(false);
          setIsConnectingToE2B(false);
          setError(event.error);
          console.error('Agent error received:', event.error);
          break;

        case 'vnc_url_set':
          setIsConnectingToE2B(false); // Connected! VNC URL received
          setVncUrl(event.vncUrl);
          console.log('VNC URL set received:', event.vncUrl);
          break;

        case 'vnc_url_unset':
          setVncUrl('');
          console.log('VNC URL unset received');
          break;

        case 'heartbeat':
          console.log('Heartbeat received:', event);
          setTraceId(event.uuid);
          console.log('TraceId set from backend:', event.uuid);
          break;

      }
    },
    [setTrace, updateTraceWithStep, completeTrace, setIsAgentProcessing, setIsConnectingToE2B, setVncUrl, setError, resetAgent, setTraceId, traceId]
  );

  // Handle WebSocket errors
  const handleWebSocketError = useCallback(() => {
    // WebSocket Frontend Error handling
    console.error('WebSocket connection error');
  }, []);

  // Initialize WebSocket connection
  const { isConnected, connectionState, sendMessage, manualReconnect } = useWebSocket({
    url,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
  });

  // Sync connection state to store and clear traceId on disconnect
  useEffect(() => {
    setIsConnected(isConnected);

    // Clear traceId when websocket disconnects
    if (!isConnected) {
      setTraceId(null);
      console.log('WebSocket disconnected - traceId cleared');
    }
  }, [isConnected, setIsConnected, setTraceId]);

  // Create a global sendNewTask function that can be called from anywhere
  useEffect(() => {
    // Store sendNewTask in window for global access
    (window as Window & { __sendNewTask?: (instruction: string, modelId: string) => void }).__sendNewTask = (instruction: string, modelId: string) => {
      // Reset agent state before starting a new task
      resetAgent();

      // Ensure traceId is set before creating trace
      if (!traceId) {
        console.error('Internal error: Cannot send task. TraceId not set. Refreshing page...');
        window.location.reload();
        return;
      }

      const trace: AgentTrace = {
        id: traceId,
        instruction,
        modelId: modelId,
        timestamp: new Date(),
        isRunning: true,
        traceMetadata: {
          traceId: traceId,
          inputTokensUsed: 0,
          outputTokensUsed: 0,
          duration: 0,
          numberOfSteps: 0,
          maxSteps: 200, // Default max steps, will be updated by backend
          completed: false,
          final_state: null,
        } as AgentTraceMetadata,
      };

      setTrace(trace);
      setIsAgentProcessing(true);
      setIsConnectingToE2B(true); // Start connecting when task is sent

      // Send message to Python backend via WebSocket
      sendMessage({
        type: 'user_task',
        trace: trace,
      });

      console.log('Task sent:', trace);
    };
  }, [setTrace, setIsAgentProcessing, setIsConnectingToE2B, sendMessage, resetAgent, traceId]);

  // Function to stop the current task
  const stopCurrentTask = useCallback(() => {
    const trace = useAgentStore.getState().trace;
    if (trace?.id && trace.isRunning) {
      sendMessage({
        type: 'stop_task',
        trace_id: trace.id,
      });
      console.log('Stop task sent for trace:', trace.id);

      // Don't update UI state here - wait for backend to send agent_complete with final_state='stopped'
    }
  }, [sendMessage]);

  return {
    isConnected,
    connectionState,
    manualReconnect,
    stopCurrentTask,
  };
};
