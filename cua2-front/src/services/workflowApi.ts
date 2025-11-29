/**
 * 워크플로우 API 서비스 - LangGraph 기반
 */

import { getApiBaseUrl } from '../config/api';

const API_BASE = `${getApiBaseUrl()}/workflows`;

// === 타입 정의 ===

export interface WorkflowParameter {
  name: string;
  type: 'string' | 'number' | 'boolean';
  label: string;
  placeholder?: string;
  default?: string | number | boolean;
  required?: boolean;
  min?: number;
  max?: number;
}

export interface WorkflowConfig {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  category: string;
  parameters: WorkflowParameter[];
}

export interface WorkflowNode {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  type?: 'start' | 'process' | 'condition' | 'end' | 'error' | 'vlm';
  instruction?: string; // VLM 에이전트 명령 (시스템 프롬프트)
  // 재사용/메모리 설정
  reusable?: boolean;  // trace 저장 여부
  reuse_trace?: boolean;  // 이전 trace 재사용 여부
  share_memory?: boolean;  // 메모리 공유 여부
  cache_key_params?: string[];  // 캐시 키에 사용할 파라미터
}

export interface WorkflowEdge {
  source: string;
  target: string;
  type: 'success' | 'failure';
}

export interface WorkflowDefinition {
  config: WorkflowConfig;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  start_node: string;
}

export interface ExecutionStatus {
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
}

export interface RunningExecution {
  execution_id: string;
  workflow_id: string;
  workflow_name: string;
  status: string;
  current_node: string | null;
  start_time: string | null;
}

export interface VLMStepLog {
  step_number: number;
  timestamp: string;
  screenshot: string | null;
  thought: string | null;
  action: string | null;
  observation: string | null;
  error: string | null;
  tool_calls: Array<{
    function: string;
    args: Record<string, unknown>;
  }>;
}

export interface NodeExecutionLogs {
  execution_id: string;
  node_id: string;
  status: 'success' | 'failed';
  error: string | null;
  logs: VLMStepLog[];
  log_count: number;
}

export interface StartWorkflowOptions {
  useVlmAgent?: boolean;
  modelId?: string;
}

// === 워크플로우 정의 API ===

export async function listWorkflows(): Promise<{ workflows: WorkflowDefinition[]; count: number }> {
  const response = await fetch(`${API_BASE}`);
  if (!response.ok) throw new Error('워크플로우 목록 조회 실패');
  return response.json();
}

export async function getWorkflowDetail(workflowId: string): Promise<WorkflowDefinition> {
  const response = await fetch(`${API_BASE}/${encodeURIComponent(workflowId)}`);
  if (!response.ok) throw new Error('워크플로우 상세 조회 실패');
  return response.json();
}

// === 워크플로우 실행 API ===

export async function startWorkflow(
  workflowId: string,
  parameters: Record<string, unknown>,
  executionId?: string,
  options?: StartWorkflowOptions
): Promise<{
  execution_id: string;
  workflow_id: string;
  status: string;
  parameters: Record<string, unknown>;
  vlm_agent_enabled: boolean;
}> {
  const response = await fetch(`${API_BASE}/${encodeURIComponent(workflowId)}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parameters,
      execution_id: executionId,
      use_vlm_agent: options?.useVlmAgent ?? true,
      model_id: options?.modelId ?? 'local-qwen3-vl',
    }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '워크플로우 시작 실패' }));
    throw new Error(error.detail || '워크플로우 시작 실패');
  }
  return response.json();
}

export async function stopWorkflow(executionId: string): Promise<{ execution_id: string; status: string }> {
  const response = await fetch(`${API_BASE}/executions/${encodeURIComponent(executionId)}/stop`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('워크플로우 중지 실패');
  return response.json();
}

export async function getExecutionStatus(executionId: string): Promise<ExecutionStatus> {
  const response = await fetch(`${API_BASE}/executions/${encodeURIComponent(executionId)}/status`);
  if (!response.ok) throw new Error('실행 상태 조회 실패');
  return response.json();
}

export async function listRunningExecutions(): Promise<{ executions: RunningExecution[]; count: number }> {
  const response = await fetch(`${API_BASE}/executions`);
  if (!response.ok) throw new Error('실행 중인 워크플로우 조회 실패');
  return response.json();
}

export async function getWorkflowHistory(
  workflowId: string,
  limit: number = 10
): Promise<{ workflow_id: string; history: ExecutionStatus[]; count: number }> {
  const response = await fetch(`${API_BASE}/${encodeURIComponent(workflowId)}/history?limit=${limit}`);
  if (!response.ok) throw new Error('워크플로우 히스토리 조회 실패');
  return response.json();
}

// === 노드 실행 로그 API ===

export async function getNodeExecutionLogs(
  executionId: string,
  nodeId: string
): Promise<NodeExecutionLogs> {
  const response = await fetch(
    `${API_BASE}/executions/${encodeURIComponent(executionId)}/nodes/${encodeURIComponent(nodeId)}/logs`
  );
  if (!response.ok) throw new Error('노드 실행 로그 조회 실패');
  return response.json();
}

export async function getCurrentScreenshot(
  executionId: string
): Promise<{ execution_id: string; screenshot: string }> {
  const response = await fetch(
    `${API_BASE}/executions/${encodeURIComponent(executionId)}/screenshot`
  );
  if (!response.ok) throw new Error('스크린샷 조회 실패');
  return response.json();
}

// === 트레이스 API ===

const TRACE_API_BASE = `${getApiBaseUrl()}/traces`;

export interface SaveTraceRequest {
  execution_id: string;
  workflow_id: string;
  instruction?: string;
  model_id?: string;
  status: string;
  final_state?: string;
  error_message?: string;
  error_cause?: string;
  user_evaluation: string;
  evaluation_reason?: string;
  steps_count: number;
  max_steps?: number;
  duration_seconds?: number;
  start_time?: string;
  end_time?: string;
  steps: Array<{
    step_id?: string;
    step_number: number;
    image?: string;
    screenshot?: string;
    thought?: string;
    action?: string;
    observation?: string;
    error?: string;
    tool_calls?: Array<{ name: string; args: Record<string, unknown> }>;
    actions?: Array<{ function_name: string; parameters: Record<string, unknown> }>;
    step_evaluation?: string;
    evaluation?: string;
    timestamp?: string;
  }>;
}

export interface TraceData {
  trace_id: string;
  execution_id: string;
  workflow_id: string;
  instruction?: string;
  model_id: string;
  status: string;
  final_state?: string;
  error_message?: string;
  error_cause?: string;
  user_evaluation: string;
  evaluation_reason?: string;
  steps_count: number;
  max_steps: number;
  duration_seconds: number;
  start_time?: string;
  end_time?: string;
  created_at: string;
  updated_at: string;
}

export interface TraceStepData {
  trace_id: string;
  step_id: string;
  step_number: number;
  screenshot_path?: string;
  thought?: string;
  action?: string;
  observation?: string;
  error?: string;
  tool_calls?: Array<{ name?: string; function_name?: string; args?: Record<string, unknown>; parameters?: Record<string, unknown> }>;
  evaluation: string;
  timestamp: string;
}

/**
 * 트레이스 저장
 */
export async function saveTrace(request: SaveTraceRequest): Promise<{
  trace_id: string;
  execution_id: string;
  status: string;
  steps_saved: number;
}> {
  const response = await fetch(TRACE_API_BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '트레이스 저장 실패' }));
    throw new Error(error.detail || '트레이스 저장 실패');
  }
  return response.json();
}

/**
 * 트레이스 평가 업데이트
 */
export async function updateTraceEvaluation(
  traceId: string,
  userEvaluation: string,
  evaluationReason?: string
): Promise<{ trace_id: string; status: string }> {
  const response = await fetch(`${TRACE_API_BASE}/${encodeURIComponent(traceId)}/evaluation`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_evaluation: userEvaluation,
      evaluation_reason: evaluationReason,
    }),
  });
  if (!response.ok) throw new Error('평가 업데이트 실패');
  return response.json();
}

/**
 * 스텝 평가 업데이트
 */
export async function updateStepEvaluation(
  traceId: string,
  stepId: string,
  evaluation: string
): Promise<{ trace_id: string; step_id: string; status: string }> {
  const response = await fetch(
    `${TRACE_API_BASE}/${encodeURIComponent(traceId)}/steps/${encodeURIComponent(stepId)}/evaluation`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ evaluation }),
    }
  );
  if (!response.ok) throw new Error('스텝 평가 업데이트 실패');
  return response.json();
}

/**
 * 트레이스 목록 조회
 */
export async function listTraces(params?: {
  workflow_id?: string;
  status?: string;
  user_evaluation?: string;
  limit?: number;
  offset?: number;
}): Promise<{ traces: TraceData[]; count: number }> {
  const searchParams = new URLSearchParams();
  if (params?.workflow_id) searchParams.append('workflow_id', params.workflow_id);
  if (params?.status) searchParams.append('status', params.status);
  if (params?.user_evaluation) searchParams.append('user_evaluation', params.user_evaluation);
  if (params?.limit) searchParams.append('limit', String(params.limit));
  if (params?.offset) searchParams.append('offset', String(params.offset));

  const response = await fetch(`${TRACE_API_BASE}?${searchParams.toString()}`);
  if (!response.ok) throw new Error('트레이스 목록 조회 실패');
  return response.json();
}

/**
 * 트레이스 상세 조회
 */
export async function getTrace(traceId: string): Promise<{
  trace: TraceData;
  steps: TraceStepData[];
}> {
  const response = await fetch(`${TRACE_API_BASE}/${encodeURIComponent(traceId)}`);
  if (!response.ok) throw new Error('트레이스 조회 실패');
  return response.json();
}

/**
 * 트레이스 통계 조회
 */
export async function getTraceStats(): Promise<{
  total_traces: number;
  by_status: Record<string, number>;
  by_evaluation: Record<string, number>;
  by_workflow: Record<string, number>;
  success_rate: number;
  evaluated_count: number;
}> {
  const response = await fetch(`${TRACE_API_BASE}/stats`);
  if (!response.ok) throw new Error('트레이스 통계 조회 실패');
  return response.json();
}
