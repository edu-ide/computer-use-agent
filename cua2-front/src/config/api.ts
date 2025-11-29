/**
 * API 및 WebSocket URL 설정
 * - 환경변수가 설정되어 있으면 사용
 * - 그렇지 않으면 현재 호스트 기반으로 동적 생성
 */

/**
 * Get the WebSocket URL based on the environment
 */
export const getWebSocketUrl = (): string => {
    // Check if we have a configured WebSocket URL from environment
    const envWsUrl = import.meta.env.VITE_WS_URL;

    if (envWsUrl) {
        return envWsUrl;
    }

    // Use current host (works for both local and remote access)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';

    // In production, use same port as frontend (assume reverse proxy)
    if (import.meta.env.PROD) {
        return `${protocol}//${window.location.host}/ws`;
    }

    // In development, use API port
    return `${protocol}//${host}:${port}/ws`;
};

/**
 * Get the base API URL based on the environment
 */
export const getApiBaseUrl = (): string => {
    // Check if we have a configured API URL from environment
    const envApiUrl = import.meta.env.VITE_API_URL;

    if (envApiUrl) {
        return envApiUrl;
    }

    // Use current host (works for both local and remote access)
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';

    // In production, use same port as frontend (assume reverse proxy)
    if (import.meta.env.PROD) {
        return `${protocol}//${window.location.host}/api`;
    }

    // In development, use API port
    return `${protocol}//${host}:${port}/api`;
};
