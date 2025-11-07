import { useCallback } from 'react';
import { exportTraceToJson, downloadJson } from '@/services/jsonExporter';
import { AgentTrace, AgentStep, AgentTraceMetadata, FinalStep } from '@/types/agent';

interface UseJsonExporterOptions {
  trace?: AgentTrace;
  steps: AgentStep[];
  metadata?: AgentTraceMetadata;
  finalStep?: FinalStep;
}

interface UseJsonExporterReturn {
  downloadTraceAsJson: () => void;
}

/**
 * Custom hook to export and download a trace as JSON
 */
export const useJsonExporter = ({
  trace,
  steps,
  metadata,
  finalStep,
}: UseJsonExporterOptions): UseJsonExporterReturn => {
  const downloadTraceAsJson = useCallback(() => {
    if (!trace) {
      console.error('No trace available to export');
      return;
    }

    try {
      const jsonString = exportTraceToJson(trace, steps, metadata, finalStep);
      const filename = `trace-${trace.id}.json`;
      downloadJson(jsonString, filename);
    } catch (error) {
      console.error('Error exporting trace to JSON:', error);
    }
  }, [trace, steps, metadata, finalStep]);

  return {
    downloadTraceAsJson,
  };
};
