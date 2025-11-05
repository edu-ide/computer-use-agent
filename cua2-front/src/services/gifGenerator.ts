import gifshot from 'gifshot';

export interface GifGenerationOptions {
  images: string[];
  interval?: number; // Duration of each frame in seconds
  gifWidth?: number;
  gifHeight?: number;
  quality?: number;
}

export interface GifGenerationResult {
  success: boolean;
  image?: string; // GIF data URL
  error?: string;
}

/**
 * Add step counter to an image
 * @param imageSrc Image source (base64 or URL)
 * @param stepNumber Step number
 * @param totalSteps Total number of steps
 * @param width Image width
 * @param height Image height
 * @returns Promise resolved with modified image in base64
 */
const addStepCounter = async (
  imageSrc: string,
  stepNumber: number,
  totalSteps: number,
  width: number,
  height: number
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        reject(new Error('Cannot get canvas context'));
        return;
      }

      // Draw the image
      ctx.drawImage(img, 0, 0, width, height);

      // Configure counter style
      const fontSize = Math.max(12, Math.floor(height * 0.08));
      const padding = Math.max(6, Math.floor(height * 0.03));
      const text = `${stepNumber}/${totalSteps}`;

      ctx.font = `bold ${fontSize}px Arial, sans-serif`;
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;

      // Position at bottom right
      const x = width - textWidth - padding * 2;
      const y = height - padding * 2;

      // Draw semi-transparent rectangle for readability
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillRect(
        x - padding,
        y - textHeight - padding,
        textWidth + padding * 2,
        textHeight + padding * 2
      );

      // Draw black text
      ctx.fillStyle = '#000000';
      ctx.textBaseline = 'top';
      ctx.fillText(text, x, y - textHeight);

      // Convert canvas to base64
      resolve(canvas.toDataURL('image/png'));
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = imageSrc;
  });
};

/**
 * Generate a GIF from a list of images (base64 or URLs)
 * @param options GIF generation options
 * @returns Promise resolved with generation result
 */
export const generateGif = async (
  options: GifGenerationOptions
): Promise<GifGenerationResult> => {
  const {
    images,
    interval = 1.5, // 1.5 seconds per frame by default
    gifWidth = 400,
    gifHeight = 200,
    quality = 10,
  } = options;

  if (!images || images.length === 0) {
    return {
      success: false,
      error: 'No images provided to generate GIF',
    };
  }

  try {
    // Add counter to each image
    const imagesWithCounter = await Promise.all(
      images.map((img, index) =>
        addStepCounter(img, index + 1, images.length, gifWidth, gifHeight)
      )
    );

    return new Promise((resolve) => {
      gifshot.createGIF(
        {
          images: imagesWithCounter,
          interval,
          gifWidth,
          gifHeight,
          numFrames: imagesWithCounter.length,
          frameDuration: interval,
          sampleInterval: quality,
        },
        (obj: { error: boolean; errorMsg?: string; image?: string }) => {
          if (obj.error) {
            resolve({
              success: false,
              error: obj.errorMsg || 'Error during GIF generation',
            });
          } else {
            resolve({
              success: true,
              image: obj.image,
            });
          }
        }
      );
    });
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
};

/**
 * Download a GIF (data URL) with a filename
 * @param dataUrl GIF data URL
 * @param filename Filename to download
 */
export const downloadGif = (dataUrl: string, filename: string = 'trace-replay.gif') => {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
