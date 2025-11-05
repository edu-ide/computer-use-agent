import { AgentTrace, AgentStep, AgentTraceMetadata } from '@/types/agent';

/**
 * Export the complete trace as JSON
 * @param trace The agent trace
 * @param steps The trace steps
 * @param metadata The final metadata
 * @returns A JSON object containing the entire trace
 */
export const exportTraceToJson = (
  trace: AgentTrace,
  steps: AgentStep[],
  metadata?: AgentTraceMetadata
): string => {
  const exportData = {
    trace: {
      id: trace.id,
      timestamp: trace.timestamp,
      instruction: trace.instruction,
      modelId: trace.modelId,
      isRunning: trace.isRunning,
    },
    metadata: metadata || trace.traceMetadata,
    steps: steps.map((step) => ({
      traceId: step.traceId,
      stepId: step.stepId,
      error: step.error,
      thought: step.thought,
      actions: step.actions,
      duration: step.duration,
      inputTokensUsed: step.inputTokensUsed,
      outputTokensUsed: step.outputTokensUsed,
      step_evaluation: step.step_evaluation,
      // Ne pas inclure l'image base64 pour réduire la taille du JSON
      hasImage: !!step.image,
    })),
    exportedAt: new Date().toISOString(),
  };

  return JSON.stringify(exportData, null, 2);
};

/**
 * Télécharge un JSON avec un nom de fichier
 * @param jsonString String JSON à télécharger
 * @param filename Nom du fichier à télécharger
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
