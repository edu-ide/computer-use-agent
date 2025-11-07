import { useState, useCallback } from 'react';
import { generateGif, downloadGif, GifGenerationOptions } from '@/services/gifGenerator';
import { AgentStep } from '@/types/agent';

interface UseGifGeneratorOptions {
  steps: AgentStep[];
  traceId?: string;
}

interface UseGifGeneratorReturn {
  isGenerating: boolean;
  error: string | null;
  generateAndDownloadGif: () => Promise<void>;
}

/**
 * Custom hook to generate and download a GIF from trace steps
 */
export const useGifGenerator = ({
  steps,
  traceId,
}: UseGifGeneratorOptions): UseGifGeneratorReturn => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateAndDownloadGif = useCallback(async () => {
    if (!steps || steps.length === 0) {
      setError('No steps available to generate GIF');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      // Extract images from steps
      const images = steps
        .map((step) => step.image)
        .filter((image): image is string => !!image);

      if (images.length === 0) {
        setError('No images available in steps');
        setIsGenerating(false);
        return;
      }

      // Generate GIF with original image dimensions
      const options: GifGenerationOptions = {
        images,
        interval: 1.5, // 1.5 seconds per frame
        quality: 10, // Medium quality for good size/quality compromise
      };

      const result = await generateGif(options);

      if (!result.success || !result.image) {
        setError(result.error || 'Error generating GIF');
        setIsGenerating(false);
        return;
      }

      // Download the GIF
      const filename = traceId
        ? `trace-${traceId}-replay.gif`
        : `trace-replay-${Date.now()}.gif`;

      downloadGif(result.image, filename);
      setIsGenerating(false);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unexpected error generating GIF'
      );
      setIsGenerating(false);
    }
  }, [steps, traceId]);

  return {
    isGenerating,
    error,
    generateAndDownloadGif,
  };
};
