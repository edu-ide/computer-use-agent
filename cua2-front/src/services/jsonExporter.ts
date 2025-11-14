import { AgentStep, AgentTrace, AgentTraceMetadata, FinalStep } from '@/types/agent';

/**
 * Extract final answer from steps
 */
const extractFinalAnswer = (steps: AgentStep[]): string | null => {
  if (!steps || steps.length === 0) {
    return null;
  }

  // Try to find final_answer in any step (iterate backwards)
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];

    if (step.actions && Array.isArray(step.actions)) {
      const finalAnswerAction = step.actions.find(
        (action) => action.function_name === 'final_answer'
      );

      if (finalAnswerAction && finalAnswerAction.parameters?.answer) {
        return String(finalAnswerAction.parameters.answer);
      }
    }
  }

  // Fallback: return the last thought if no final_answer found
  const lastStep = steps[steps.length - 1];
  return lastStep?.thought || null;
};

/**
 * Export the complete trace as JSON
 * @param trace The agent trace
 * @param steps The trace steps
 * @param metadata The final metadata
 * @param finalStep The final step with completion status
 * @returns A JSON object containing the entire trace
 */
export const exportTraceToJson = (
  trace: AgentTrace,
  steps: AgentStep[],
  metadata?: AgentTraceMetadata,
  finalStep?: FinalStep
): string => {
  const exportData = {
    trace: {
      id: trace.id,
      timestamp: trace.timestamp,
      instruction: trace.instruction,
      modelId: trace.modelId,
      isRunning: trace.isRunning,
    },
    completion: finalStep ? {
      status: finalStep.type,
      message: finalStep.message || null,
      finalAnswer: extractFinalAnswer(steps),
    } : null,
    metadata: metadata || trace.traceMetadata,
    steps: steps.map((step) => ({
      traceId: step.traceId,
      stepId: step.stepId,
      error: step.error,
      image: step.image, // Include full base64 image
      thought: step.thought,
      actions: step.actions,
      duration: step.duration,
      inputTokensUsed: step.inputTokensUsed,
      outputTokensUsed: step.outputTokensUsed,
      step_evaluation: step.step_evaluation,
    })),
    exportedAt: new Date().toISOString(),
  };

  return JSON.stringify(exportData, null, 2);
};

/**
 * Download a JSON with a filename
 * @param jsonString JSON string to download
 * @param filename Filename to download
 */
export const downloadJson = (jsonString: string, filename: string = 'trace.json') => {
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
