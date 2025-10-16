export interface AgentMessage {
  id: string;
  type: 'user' | 'agent';
  timestamp: Date;
  instructions: string;
  modelId: string;
  steps?: AgentStep[];
  metadata?: AgentMetadata;
  isLoading?: boolean;
}

export interface AgentStep {
  messageId: string;
  stepId: string;
  image: string;
  generatedText: string;
  actions: string[];
  inputTokensUsed: number;
  outputTokensUsed: number;
  timestamp: Date;
}

export interface AgentMetadata {
  messageId: string;
  inputTokensUsed: number;
  outputTokensUsed: number;
  timeTaken: number;
  numberOfSteps: number;
}

export interface WebSocketEvent {
  type: 'agent_start' | 'agent_progress' | 'agent_complete' | 'agent_error' | 'vnc_url_set' | 'vnc_url_unset' | 'heartbeat';
  agentStep?: AgentStep;
  metadata?: AgentMetadata;
  vncUrl?: string;
}
