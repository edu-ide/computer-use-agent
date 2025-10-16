import React from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { AgentMessage, WebSocketEvent } from '@/types/agent';
import { useEffect, useState } from 'react';

const Index = () => {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [isAgentProcessing, setIsAgentProcessing] = useState(false);
  const [vncUrl, setVncUrl] = useState<string>('');

  // WebSocket connection - Use environment variable for flexibility across environments
  // const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
  const WS_URL = 'ws://localhost:8000/ws';

  const handleWebSocketMessage = (event: WebSocketEvent) => {
    console.log('WebSocket event received:', event);

    switch (event.type) {
      case 'agent_start':
        setIsAgentProcessing(true);
        if (event.content) {
          const newMessage: AgentMessage = {
            id: event.messageId,
            type: 'agent',
            instructions: event.instructions,
            modelId: event.modelId,
            timestamp: new Date(),
            isLoading: true,
          };
          setMessages(prev => [...prev, newMessage]);
        }
        break;

      case 'agent_progress':
        if (event.messageId && event.agentStep) {
          // Add new step from a agent trace run with image, generated text, actions, tokens and timestamp
          setMessages(prev =>
            prev.map(msg => {
              if (msg.id === event.agentStep.messageId) {
                const existingSteps = msg.steps || [];
                const stepExists = existingSteps.some(step => step.stepId === event.agentStep.stepId);
                
                if (!stepExists) {
                  return { ...msg, steps: [...existingSteps, event.agentStep], isLoading: true };
                }
                return msg;
              }
              return msg;
            })
          );
        }
        break;

      case 'agent_complete':
        setIsAgentProcessing(false);
        if (event.messageId && event.metadata) {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === event.metadata.messageId
                ? {
                  ...msg,
                  isLoading: false,
                  metadata: event.metadata,
                }
                : msg
            )
          );
        }
        break;

      case 'agent_error':
        setIsAgentProcessing(false);
        // TODO: Handle agent error
        break;

      case 'vnc_url_set':
        if (event.vncUrl) {
          setVncUrl(event.vncUrl);
        }
        // TODO: Handle VNC URL set
        break;

      case 'vnc_url_unset':
        setVncUrl('');
        // TODO: Handle VNC URL unset
        break;

      case 'heartbeat':
        console.log('Heartbeat received:', event);
        break;
    }
  };

  const handleWebSocketError = () => {
    // Error handling is now throttled in the WebSocket hook

  };

  const { isConnected, connectionState, sendMessage, manualReconnect } = useWebSocket({
    url: WS_URL,
    onMessage: handleWebSocketMessage,
    onError: handleWebSocketError,
  });

  const handleSendMessage = (content: string) => {
    const userMessage: AgentMessage = {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    // Send message to Python backend via WebSocket
    sendMessage({
      type: 'user_task',
      content,
      model_id: "anthropic/claude-sonnet-4-5-20250929",
      timestamp: new Date().toISOString(),
    });
  };


  return (
    <div>
      <h1>Hello World</h1>
    </div>
  );
};

export default Index;
