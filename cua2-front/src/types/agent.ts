export interface AgentTrace {
  id: string;
  timestamp: Date;
  instruction: string;
  modelId: string;
  isRunning: boolean;
  steps?: AgentStep[];
  traceMetadata?: AgentTraceMetadata;
}

export interface AgentAction {
  function_name: string;
  parameters: Record<string, unknown>;
  description: string;
}

export interface AgentStep {
  traceId: string;
  stepId: string;
  error?: string | null;
  image: string;
  thought?: string | null;
  actions?: AgentAction[] | null;
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
  maxSteps: number;
  completed: boolean;
  final_state: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout' | null;
  user_evaluation?: 'success' | 'failed' | 'not_evaluated';
}

export interface FinalStep {
  type: 'success' | 'failure' | 'stopped' | 'max_steps_reached' | 'sandbox_timeout';
  message?: string;
  metadata: AgentTraceMetadata;
}

// #################### WebSocket Events Types - Server to Client ########################

interface AgentStartEvent {
  type: 'agent_start';
  agentTrace: AgentTrace;
  status: 'max_sandboxes_reached' | 'success';
}

interface AgentProgressEvent {
  type: 'agent_progress';
  agentStep: AgentStep;
  traceMetadata: AgentTraceMetadata;
}

interface AgentCompleteEvent {
  type: 'agent_complete';
  traceMetadata: AgentTraceMetadata;
  final_state: 'success' | 'stopped' | 'max_steps_reached' | 'error' | 'sandbox_timeout';
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
  uuid: string;
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


export interface StopTaskMessage {
  type: 'stop_task';
  traceId: string;
}
