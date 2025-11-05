
/**
 * Get the WebSocket URL based on the environment
 * In production (Docker), it uses the current host
 * In development, it uses the configured URL or defaults to localhost:8000
 */
export const getWebSocketUrl = (): string => {
    // Check if we have a configured WebSocket URL from environment
    const envWsUrl = import.meta.env.VITE_WS_URL;

    if (envWsUrl) {
        return envWsUrl;
    }

    // In production (when served from same origin), use relative WebSocket URL
    if (import.meta.env.PROD) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    // Development fallback
    return 'ws://localhost:8000/ws';
};

/**
 * Get the base API URL based on the environment
 * In production (Docker), it uses the current host with /api prefix
 * In development, it uses the configured URL or defaults to localhost:8000/api
 */
export const getApiBaseUrl = (): string => {
    // Check if we have a configured API URL from environment
    const envApiUrl = import.meta.env.VITE_API_URL;

    if (envApiUrl) {
        return envApiUrl;
    }

    // In production (when served from same origin), use relative API URL
    if (import.meta.env.PROD) {
        const protocol = window.location.protocol;
        const host = window.location.host;
        return `${protocol}//${host}/api`;
    }

    // Development fallback
    return 'http://localhost:8000/api';
};
