import { config } from '@/config';

/**
 * Fetch available models from the backend
 */
export async function fetchAvailableModels(): Promise<string[]> {
  const response = await fetch(`${config.apiBaseUrl}/models`);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  const data = await response.json();
  return data.models;
}

/**
 * Generate a random instruction from the backend
 */
export async function generateRandomQuestion(modelId: string): Promise<string> {
  const response = await fetch(`${config.apiBaseUrl}/generate-instruction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_id: modelId,
    }),
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
  const response = await fetch(`${config.apiBaseUrl}/traces/${traceId}/steps/${stepId}`, {
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
