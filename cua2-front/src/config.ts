import { getApiBaseUrl, getWebSocketUrl } from './config/api';

// Application configuration
export const config = {
  // WebSocket URL for backend connection (동적 생성)
  get wsUrl() {
    return getWebSocketUrl();
  },

  // API Base URL (동적 생성)
  get apiBaseUrl() {
    return getApiBaseUrl();
  },

  // Default model (will be overridden by first available model from backend)
  defaultModelId: 'Qwen/Qwen3-VL-8B-Instruct',
} as const;
