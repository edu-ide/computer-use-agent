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
      const fontSize = Math.max(11, Math.floor(height * 0.05));
      const padding = Math.max(5, Math.floor(height * 0.02));
      const text = `${stepNumber}/${totalSteps}`;

      ctx.font = `bold ${fontSize}px Arial, sans-serif`;
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;

      // Use actual text metrics for better vertical centering
      const actualHeight = textMetrics.actualBoundingBoxAscent + textMetrics.actualBoundingBoxDescent;

      // Calculate box dimensions
      const boxWidth = textWidth + padding * 2;
      const boxHeight = actualHeight + padding * 2;

      // Position at bottom right with margin
      const margin = Math.max(8, Math.floor(height * 0.015));
      const boxX = width - boxWidth - margin;
      const boxY = height - boxHeight - margin;

      // Draw semi-transparent rounded rectangle for readability
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      const borderRadius = 4;
      ctx.beginPath();
      ctx.roundRect(boxX, boxY, boxWidth, boxHeight, borderRadius);
      ctx.fill();

      // Draw black text centered in the box
      ctx.fillStyle = '#000000';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      // Position text precisely using actual bounding box metrics
      const textX = boxX + boxWidth / 2;
      const textY = boxY + padding + textMetrics.actualBoundingBoxAscent;
      ctx.fillText(text, textX, textY);

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
 * Get the dimensions of an image
 * @param imageSrc Image source (base64 or URL)
 * @returns Promise resolved with image dimensions
 */
const getImageDimensions = (imageSrc: string): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };

    img.onerror = () => {
      reject(new Error('Failed to load image to get dimensions'));
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
    gifWidth,
    gifHeight,
    quality = 10,
  } = options;

  if (!images || images.length === 0) {
    return {
      success: false,
      error: 'No images provided to generate GIF',
    };
  }

  try {
    // Get dimensions from the first image if not specified
    let width = gifWidth;
    let height = gifHeight;

    if (!width || !height) {
      const dimensions = await getImageDimensions(images[0]);
      width = width || dimensions.width;
      height = height || dimensions.height;
    }

    // Add counter to each image
    const imagesWithCounter = await Promise.all(
      images.map((img, index) =>
        addStepCounter(img, index + 1, images.length, width, height)
      )
    );

    return new Promise((resolve) => {
      gifshot.createGIF(
        {
          images: imagesWithCounter,
          interval,
          gifWidth: width,
          gifHeight: height,
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
