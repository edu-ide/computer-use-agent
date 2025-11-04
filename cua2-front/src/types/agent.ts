
export interface AgentTrace {
  id: string;
  timestamp: Date;
  instruction: string;
  modelId: string;
  isRunning: boolean;
  steps?: AgentStep[];
  metadata?: AgentTraceMetadata;
}

export interface AgentStep {
  traceId: string;
  stepId: string;
  error: string;
  image: string;
  thought: string;
  actions: string[];
  duration: number;
  inputTokensUsed: number;
  outputTokensUsed: number;
  step_evaluation: 'like' | 'dislike' | 'neutral';
}

export interface AgentTraceMetadata {
  traceId: string;
  inputTokensUsed: number;
  outputTokensUsed: number;
  duration: number;
  numberOfSteps: number;
}

// #################### WebSocket Events Types - Server to Client ########################

interface AgentStartEvent {
  type: 'agent_start';
  agentTrace: AgentTrace;
}

interface AgentProgressEvent {
  type: 'agent_progress';
  agentStep: AgentStep;
  traceMetadata: AgentTraceMetadata;
}

interface AgentCompleteEvent {
  type: 'agent_complete';
  traceMetadata: AgentTraceMetadata;
}

interface AgentErrorEvent {
  type: 'agent_error';
  error: string;
}

interface VncUrlSetEvent {
  type: 'vnc_url_set';
  vncUrl: string;
}

interface VncUrlUnsetEvent {
  type: 'vnc_url_unset';
}

interface HeartbeatEvent {
  type: 'heartbeat';
}

export type WebSocketEvent =
  | AgentStartEvent
  | AgentProgressEvent
  | AgentCompleteEvent
  | AgentErrorEvent
  | VncUrlSetEvent
  | VncUrlUnsetEvent
  | HeartbeatEvent;

// #################### User Task Message Type (Through WebSocket) - Client to Server ########################


export interface UserTaskMessage {
  type: 'user_task';
  trace: AgentTrace;
}

// #################### API Routes Types ########################

export interface AvailableModelsResponse {
  models: string[];
}

export interface UpdateStepRequest {
  step_evaluation: 'like' | 'dislike' | 'neutral';
}

export interface UpdateStepResponse {
  success: boolean;
  message: string;
}

export interface GenerateInstructionRequest {
  model_id: string;
  prompt?: string;
}

export interface GenerateInstructionResponse {
  instruction: string;
  model_id: string;
}
