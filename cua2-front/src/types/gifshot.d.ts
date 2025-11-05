declare module 'gifshot' {
  export interface GifOptions {
    images?: string[];
    video?: string | string[] | HTMLVideoElement | HTMLVideoElement[];
    gifWidth?: number;
    gifHeight?: number;
    interval?: number;
    numFrames?: number;
    frameDuration?: number;
    sampleInterval?: number;
    quality?: number;
    numWorkers?: number;
    progressCallback?: (progress: number) => void;
    completeCallback?: () => void;
  }

  export interface GifResult {
    error: boolean;
    errorCode?: string;
    errorMsg?: string;
    image?: string;
  }

  export function createGIF(
    options: GifOptions,
    callback?: (result: GifResult) => void
  ): void;

  export function takeSnapShot(
    options: GifOptions,
    callback: (result: GifResult) => void
  ): void;
}
