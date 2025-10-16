import { WebSocketEvent } from '@/types/agent';
import { useCallback, useEffect, useRef, useState } from 'react';

interface UseWebSocketProps {
  url: string;
  onMessage: (event: WebSocketEvent) => void;
  onError?: (error: Event) => void;
}

export const useWebSocket = ({ url, onMessage, onError }: UseWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 3; // Only try three times, then stop
  const baseReconnectDelay = 3000; // Start with 3 seconds
  const maxReconnectDelay = 5000; // Max 5 seconds
  const lastErrorTimeRef = useRef(0);
  const errorThrottleMs = 5000; // Only show error toast once every 5 seconds
  const isInitialConnectionRef = useRef(true); // Track if this is the first connection attempt

  const getReconnectDelay = () => {
    // Exponential backoff with jitter
    const delay = Math.min(
      baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
      maxReconnectDelay
    );
    return delay + Math.random() * 1000; // Add jitter
  };

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return; // Already connected or connecting
    }

    try {
      setConnectionState('connecting');
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionState('connected');
        reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
        isInitialConnectionRef.current = false; // Mark that we've had a successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WebSocketEvent;
          onMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState('error');

        // Don't show error toasts on initial connection failure
        // Only show toasts after we've had a successful connection before
        if (!isInitialConnectionRef.current) {
          // Throttle error notifications
          const now = Date.now();
          if (now - lastErrorTimeRef.current > errorThrottleMs) {
            lastErrorTimeRef.current = now;
            onError?.(error);
          }
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected', { code: event.code, reason: event.reason });
        setIsConnected(false);
        setConnectionState('disconnected');

        // Only attempt to reconnect if it wasn't a manual close (code 1000) and we haven't exceeded max attempts
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = getReconnectDelay();
          console.log(`Attempting to reconnect in ${Math.round(delay)}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.log('Max reconnection attempts reached');
          setConnectionState('error');
        } else if (event.code === 1000) {
          // Normal closure - don't reconnect
          setConnectionState('disconnected');
          console.log('WebSocket closed normally, not reconnecting');
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState('error');
    }
  }, [url, onMessage, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    setConnectionState('disconnected');
    reconnectAttemptsRef.current = 0;
  }, []);

  const manualReconnect = useCallback(() => {
    console.log('Manual reconnect requested');
    disconnect();
    reconnectAttemptsRef.current = 0;
    isInitialConnectionRef.current = false; // Allow error toasts on manual reconnect
    setTimeout(() => connect(), 1000); // Small delay before reconnecting
  }, [disconnect, connect]);

  const sendMessage = (message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
      }
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [url]); // Only depend on url, not the functions

  return {
    isConnected,
    connectionState,
    sendMessage,
    reconnect: connect,
    disconnect,
    manualReconnect
  };
};
