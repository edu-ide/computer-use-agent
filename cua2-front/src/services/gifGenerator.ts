import gifshot from 'gifshot';

export interface GifGenerationOptions {
  images: string[];
  interval?: number; // Durée de chaque frame en secondes
  gifWidth?: number;
  gifHeight?: number;
  quality?: number;
}

export interface GifGenerationResult {
  success: boolean;
  image?: string; // Data URL du GIF
  error?: string;
}

/**
 * Ajoute un compteur d'étapes sur une image
 * @param imageSrc Source de l'image (base64 ou URL)
 * @param stepNumber Numéro de l'étape
 * @param totalSteps Nombre total d'étapes
 * @param width Largeur de l'image
 * @param height Hauteur de l'image
 * @returns Promesse résolue avec l'image modifiée en base64
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

      // Dessiner l'image
      ctx.drawImage(img, 0, 0, width, height);

      // Configurer le style du compteur
      const fontSize = Math.max(12, Math.floor(height * 0.08));
      const padding = Math.max(6, Math.floor(height * 0.03));
      const text = `${stepNumber}/${totalSteps}`;

      ctx.font = `bold ${fontSize}px Arial, sans-serif`;
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;

      // Position en bas à droite
      const x = width - textWidth - padding * 2;
      const y = height - padding * 2;

      // Dessiner un rectangle semi-transparent pour la lisibilité
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillRect(
        x - padding,
        y - textHeight - padding,
        textWidth + padding * 2,
        textHeight + padding * 2
      );

      // Dessiner le texte en noir
      ctx.fillStyle = '#000000';
      ctx.textBaseline = 'top';
      ctx.fillText(text, x, y - textHeight);

      // Convertir le canvas en base64
      resolve(canvas.toDataURL('image/png'));
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = imageSrc;
  });
};

/**
 * Génère un GIF à partir d'une liste d'images (base64 ou URLs)
 * @param options Options de génération du GIF
 * @returns Promesse résolue avec le résultat de la génération
 */
export const generateGif = async (
  options: GifGenerationOptions
): Promise<GifGenerationResult> => {
  const {
    images,
    interval = 1.5, // 1.5 secondes par frame par défaut
    gifWidth = 400,
    gifHeight = 200,
    quality = 10,
  } = options;

  if (!images || images.length === 0) {
    return {
      success: false,
      error: 'Aucune image fournie pour générer le GIF',
    };
  }

  try {
    // Ajouter le compteur sur chaque image
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
              error: obj.errorMsg || 'Erreur lors de la génération du GIF',
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
      error: error instanceof Error ? error.message : 'Erreur inconnue',
    };
  }
};

/**
 * Télécharge un GIF (data URL) avec un nom de fichier
 * @param dataUrl Data URL du GIF
 * @param filename Nom du fichier à télécharger
 */
export const downloadGif = (dataUrl: string, filename: string = 'trace-replay.gif') => {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
