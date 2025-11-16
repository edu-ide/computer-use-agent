import { getApiBaseUrl } from '@/config/api';

/**
 * Fetch available models from the backend
 */
export async function fetchAvailableModels(): Promise<string[]> {
  const response = await fetch(`${getApiBaseUrl()}/models`);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  const data = await response.json();
  return data.models;
}

/**
 * Generate a random instruction from the backend
 */
export async function generateRandomQuestion(): Promise<string> {
  const response = await fetch(`${getApiBaseUrl()}/generate-instruction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });
  if (!response.ok) {
    throw new Error('Failed to generate instruction');
  }
  const data = await response.json();
  return data.instruction;
}

/**
 * Update step evaluation (vote)
 */
export async function updateStepEvaluation(
  traceId: string,
  stepId: string,
  evaluation: 'like' | 'dislike' | 'neutral'
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/traces/${traceId}/steps/${stepId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      step_evaluation: evaluation,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to update step evaluation');
  }
}

/**
 * Update trace evaluation (overall task feedback)
 */
export async function updateTraceEvaluation(
  traceId: string,
  evaluation: 'success' | 'failed' | 'not_evaluated'
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/traces/${traceId}/evaluation`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_evaluation: evaluation,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to update trace evaluation');
  }
}
