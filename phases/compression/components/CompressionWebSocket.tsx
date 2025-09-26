import React, { useEffect } from 'react';

interface CompressionWebSocketProps {
  onMessage: (data: any) => void;
  wsRef: React.MutableRefObject<WebSocket | null>;
}

export const CompressionWebSocket: React.FC<CompressionWebSocketProps> = ({
  onMessage,
  wsRef
}) => {
  useEffect(() => {
    const initializeWebSocket = () => {
      try {
        // Connect to Python backend WebSocket
        const ws = new WebSocket('ws://localhost:8081/compression');
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('Connected to compression pipeline WebSocket');

          // Send initial status request
          ws.send(JSON.stringify({
            type: 'get_status'
          }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            onMessage(data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = (event) => {
          console.log('WebSocket connection closed:', event.code, event.reason);

          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            if (wsRef.current?.readyState === WebSocket.CLOSED) {
              console.log('Attempting to reconnect...');
              initializeWebSocket();
            }
          }, 3000);
        };
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);

        // Retry connection after 5 seconds
        setTimeout(initializeWebSocket, 5000);
      }
    };

    // Initialize connection
    initializeWebSocket();

    // Cleanup on unmount
    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [onMessage, wsRef]);

  return null; // This component doesn't render anything
};