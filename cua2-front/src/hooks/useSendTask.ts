import { useCallback } from 'react';

export const useSendTask = () => {
  const sendTask = useCallback((instruction: string, modelId: string) => {
    const sendNewTask = (window as Window & { __sendNewTask?: (instruction: string, modelId: string) => void }).__sendNewTask;
    if (sendNewTask) {
      sendNewTask(instruction, modelId);
    } else {
      console.error('WebSocket not initialized');
    }
  }, []);

  return sendTask;
};
