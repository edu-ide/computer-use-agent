import { Header, Metadata, StackSteps, VNCStream } from '@/components/mock';
import { getWebSocketUrl } from '@/config/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentStep, AgentTrace, WebSocketEvent } from '@/types/agent';
import { useState } from 'react';
import { ulid } from 'ulid';

const Index = () => {
  const [trace, setTrace] = useState<AgentTrace>();
  const [isAgentProcessing, setIsAgentProcessing] = useState(false);
  const [vncUrl, setVncUrl] = useState<string>('');
  const [selectedModelId, setSelectedModelId] = useState<string>("Qwen/Qwen3-VL-30B-A3B-Instruct");

  // #################### WebSocket Connection ########################

  // WebSocket connection - Automatically configured based on environment
  const WS_URL = getWebSocketUrl();

  const handleWebSocketMessage = (event: WebSocketEvent) => {
    console.log('WebSocket event received:', event);

    switch (event.type) {
      case 'agent_start':
        setIsAgentProcessing(true);
        setTrace(event.agentTrace);
        console.log('Agent start received:', event.agentTrace);
        break;

      case 'agent_progress':
        // Add new step from a agent trace run with image, generated text, actions, tokens and timestamp
        setTrace(prev => {
          const existingSteps = prev?.steps || [] as AgentStep[];
          const stepExists = existingSteps.some(step => step.stepId === event.agentStep.stepId);

          if (!stepExists) {
            return {
              ...prev,
              steps: [...existingSteps, event.agentStep],
              traceMetadata: event.traceMetadata,
              isRunning: true
            };
          }
          return prev;
        });
        console.log('Agent progress received:', event.agentStep);
        break;

      case 'agent_complete':
        setIsAgentProcessing(false);
        setTrace(trace => {
          return trace.id === event.traceMetadata.traceId
            ? {
              ...trace,
              isRunning: false,
              metadata: event.traceMetadata,
            }
            : trace;
        });
        console.log('Agent complete received:', event.traceMetadata);
        break;

      case 'agent_error':
        setIsAgentProcessing(false);
        // TODO: Handle agent error
        console.log('Agent error received:', event.error);
        break;

      case 'vnc_url_set':
        setVncUrl(event.vncUrl);
        // TODO: Handle VNC URL set
        console.log('VNC URL set received:', event.vncUrl);
        break;

      case 'vnc_url_unset':
        setVncUrl('');
        // TODO: Handle VNC URL unset
        console.log('VNC URL unset received:');
        break;

      case 'heartbeat':
        console.log('Heartbeat received:', event);
        break;
    }
  };

  const handleWebSocketError = () => {
    // WebSocket Frontend Error handling

  };

  const { isConnected, connectionState, sendMessage, manualReconnect } = useWebSocket({
    url: WS_URL,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
  });

  // #################### Frontend Functionality ########################

  const handleModelId = (modelId: string) => {
    setSelectedModelId(modelId);
  };

  const handleSendNewTask = (content: string, modelId: string) => {
    const trace: AgentTrace = {
      id: ulid(),
      instruction: content,
      modelId: selectedModelId,
      timestamp: new Date(),
      isRunning: true,
    };

    setTrace(trace);

    // Send message to Python backend via WebSocket
    sendMessage({
      type: 'user_task',
      trace: trace,
    });
  };

  // #################### Mock Frontend Rendering ########################

  return (
    <div style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', backgroundColor: '#f3f4f6' }}>
      <Header
        isConnected={isConnected}
        isAgentProcessing={isAgentProcessing}
        onSendTask={handleSendNewTask}
      />

      <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', minHeight: 0, padding: '32px' }}>
        <div style={{ width: '100%', height: '100%', maxWidth: '1400px', maxHeight: '900px', display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>
          {/* Left Side: VNC Stream + Metadata */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '20px 12px', gap: '20px', minWidth: 0 }}>
            <VNCStream vncUrl={vncUrl} />
            <Metadata trace={trace} />
          </div>

          {/* Right Side: Stack Steps */}
          <StackSteps trace={trace} />
        </div>
      </div>
    </div>
  );
};

export default Index;
