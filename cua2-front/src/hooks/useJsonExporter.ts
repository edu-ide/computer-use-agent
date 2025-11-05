import { useCallback } from 'react';
import { exportTraceToJson, downloadJson } from '@/services/jsonExporter';
import { AgentTrace, AgentStep, AgentTraceMetadata } from '@/types/agent';

interface UseJsonExporterOptions {
  trace?: AgentTrace;
  steps: AgentStep[];
  metadata?: AgentTraceMetadata;
}

interface UseJsonExporterReturn {
  downloadTraceAsJson: () => void;
}

/**
 * Hook personnalisé pour exporter et télécharger une trace en JSON
 */
export const useJsonExporter = ({
  trace,
  steps,
  metadata,
}: UseJsonExporterOptions): UseJsonExporterReturn => {
  const downloadTraceAsJson = useCallback(() => {
    if (!trace) {
      console.error('No trace available to export');
      return;
    }

    try {
      const jsonString = exportTraceToJson(trace, steps, metadata);
      const filename = `trace-${trace.id}.json`;
      downloadJson(jsonString, filename);
    } catch (error) {
      console.error('Error exporting trace to JSON:', error);
    }
  }, [trace, steps, metadata]);

  return {
    downloadTraceAsJson,
  };
};
