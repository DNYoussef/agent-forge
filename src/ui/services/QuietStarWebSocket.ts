/**
 * QuietStarWebSocket Service
 * Real-time WebSocket integration for Quiet Star reasoning updates
 */

import { WebSocketMessage, ThoughtStream, ReasoningToken, ConvergenceMetrics, CurriculumStage } from '../types/phases';

export interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
}

export interface QuietStarWebSocketCallbacks {
  onReasoningToken?: (token: ReasoningToken) => void;
  onStreamUpdate?: (streamId: number, updates: Partial<ThoughtStream>) => void;
  onConvergenceUpdate?: (metrics: Partial<ConvergenceMetrics>) => void;
  onCurriculumProgress?: (stageId: number, updates: Partial<CurriculumStage>) => void;
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
}

export class QuietStarWebSocket {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private callbacks: QuietStarWebSocketCallbacks;
  private reconnectAttempts: number = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isConnected: boolean = false;
  private messageQueue: WebSocketMessage[] = [];

  constructor(
    config: Partial<WebSocketConfig> = {},
    callbacks: QuietStarWebSocketCallbacks = {}
  ) {
    this.config = {
      url: 'ws://localhost:8080/quiet-star',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      ...config
    };
    this.callbacks = callbacks;
  }

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }

        this.ws = new WebSocket(this.config.url);

        this.ws.onopen = () => {
          console.log('Connected to Quiet Star WebSocket');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.callbacks.onConnectionChange?.(true);
          this.startHeartbeat();
          this.flushMessageQueue();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
            this.callbacks.onError?.(new Error('Invalid message format'));
          }
        };

        this.ws.onclose = (event) => {
          console.log('Disconnected from Quiet Star WebSocket', event.code, event.reason);
          this.isConnected = false;
          this.callbacks.onConnectionChange?.(false);
          this.stopHeartbeat();

          if (event.code !== 1000 && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.callbacks.onError?.(new Error('WebSocket connection error'));
          reject(error);
        };

        // Connection timeout
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            this.ws?.close();
            reject(new Error('Connection timeout'));
          }
        }, 10000);

      } catch (error) {
        reject(error);
      }
    });
  }

  public disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.isConnected = false;
    this.callbacks.onConnectionChange?.(false);
  }

  public sendMessage(message: Partial<WebSocketMessage>): void {
    const fullMessage: WebSocketMessage = {
      type: 'reasoning_token',
      data: {},
      timestamp: Date.now(),
      ...message
    };

    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(fullMessage));
    } else {
      // Queue message for later if not connected
      this.messageQueue.push(fullMessage);
    }
  }

  public updateCallbacks(newCallbacks: Partial<QuietStarWebSocketCallbacks>): void {
    this.callbacks = { ...this.callbacks, ...newCallbacks };
  }

  public isConnectionOpen(): boolean {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  public getConnectionState(): 'connecting' | 'open' | 'closing' | 'closed' {
    if (!this.ws) return 'closed';

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'open';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'closed';
      default: return 'closed';
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'reasoning_token':
        if (this.isValidReasoningToken(message.data)) {
          this.callbacks.onReasoningToken?.(message.data);
        }
        break;

      case 'stream_update':
        if (this.isValidStreamUpdate(message.data)) {
          this.callbacks.onStreamUpdate?.(message.data.streamId, message.data.updates);
        }
        break;

      case 'convergence_update':
        if (this.isValidConvergenceUpdate(message.data)) {
          this.callbacks.onConvergenceUpdate?.(message.data);
        }
        break;

      case 'curriculum_progress':
        if (this.isValidCurriculumProgress(message.data)) {
          this.callbacks.onCurriculumProgress?.(message.data.stageId, message.data.updates);
        }
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  private isValidReasoningToken(data: any): data is ReasoningToken {
    return (
      typeof data === 'object' &&
      typeof data.id === 'string' &&
      typeof data.content === 'string' &&
      ['thought_start', 'thought_content', 'thought_end', 'regular'].includes(data.type) &&
      typeof data.position === 'number' &&
      typeof data.timestamp === 'number' &&
      typeof data.confidence === 'number' &&
      typeof data.streamId === 'number'
    );
  }

  private isValidStreamUpdate(data: any): data is { streamId: number; updates: Partial<ThoughtStream> } {
    return (
      typeof data === 'object' &&
      typeof data.streamId === 'number' &&
      typeof data.updates === 'object'
    );
  }

  private isValidConvergenceUpdate(data: any): data is Partial<ConvergenceMetrics> {
    return typeof data === 'object' && data !== null;
  }

  private isValidCurriculumProgress(data: any): data is { stageId: number; updates: Partial<CurriculumStage> } {
    return (
      typeof data === 'object' &&
      typeof data.stageId === 'number' &&
      typeof data.updates === 'object'
    );
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.reconnectTimer = setTimeout(() => {
      console.log(`Reconnect attempt ${this.reconnectAttempts}`);
      this.connect().catch((error) => {
        console.error('Reconnect failed:', error);
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
        this.sendMessage({
          type: 'heartbeat' as any,
          data: { timestamp: Date.now() }
        });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.ws?.send(JSON.stringify(message));
      }
    }
  }
}

// Utility functions for creating WebSocket messages
export const createReasoningTokenMessage = (token: ReasoningToken): WebSocketMessage => ({
  type: 'reasoning_token',
  data: token,
  timestamp: Date.now()
});

export const createStreamUpdateMessage = (streamId: number, updates: Partial<ThoughtStream>): WebSocketMessage => ({
  type: 'stream_update',
  data: { streamId, updates },
  timestamp: Date.now()
});

export const createConvergenceUpdateMessage = (metrics: Partial<ConvergenceMetrics>): WebSocketMessage => ({
  type: 'convergence_update',
  data: metrics,
  timestamp: Date.now()
});

export const createCurriculumProgressMessage = (stageId: number, updates: Partial<CurriculumStage>): WebSocketMessage => ({
  type: 'curriculum_progress',
  data: { stageId, updates },
  timestamp: Date.now()
});

// React hook for using WebSocket
export const useQuietStarWebSocket = (
  config?: Partial<WebSocketConfig>,
  callbacks?: QuietStarWebSocketCallbacks
) => {
  const [ws] = React.useState(() => new QuietStarWebSocket(config, callbacks));
  const [connectionState, setConnectionState] = React.useState<'connecting' | 'open' | 'closing' | 'closed'>('closed');

  React.useEffect(() => {
    ws.updateCallbacks({
      ...callbacks,
      onConnectionChange: (connected) => {
        setConnectionState(ws.getConnectionState());
        callbacks?.onConnectionChange?.(connected);
      }
    });
  }, [ws, callbacks]);

  React.useEffect(() => {
    return () => {
      ws.disconnect();
    };
  }, [ws]);

  const connect = React.useCallback(() => ws.connect(), [ws]);
  const disconnect = React.useCallback(() => ws.disconnect(), [ws]);
  const sendMessage = React.useCallback((message: Partial<WebSocketMessage>) => ws.sendMessage(message), [ws]);

  return {
    connect,
    disconnect,
    sendMessage,
    connectionState,
    isConnected: connectionState === 'open'
  };
};

// Mock WebSocket server for development
export class MockQuietStarWebSocket extends QuietStarWebSocket {
  private mockInterval: NodeJS.Timeout | null = null;
  private streamCounter: number = 0;

  public connect(): Promise<void> {
    return new Promise((resolve) => {
      console.log('Mock WebSocket connected');
      this.isConnected = true;
      this.callbacks.onConnectionChange?.(true);
      this.startMockData();
      resolve();
    });
  }

  public disconnect(): void {
    console.log('Mock WebSocket disconnected');
    this.stopMockData();
    this.isConnected = false;
    this.callbacks.onConnectionChange?.(false);
  }

  private startMockData(): void {
    this.mockInterval = setInterval(() => {
      // Generate mock reasoning tokens
      const streamId = Math.floor(Math.random() * 4);
      const tokenTypes: ReasoningToken['type'][] = ['thought_start', 'thought_content', 'thought_end', 'regular'];
      const tokenType = tokenTypes[Math.floor(Math.random() * tokenTypes.length)];

      const mockToken: ReasoningToken = {
        id: `token_${Date.now()}_${Math.random()}`,
        content: this.generateMockContent(tokenType),
        type: tokenType,
        position: this.streamCounter++,
        timestamp: Date.now(),
        confidence: 0.7 + Math.random() * 0.3,
        streamId
      };

      this.callbacks.onReasoningToken?.(mockToken);

      // Occasionally update convergence metrics
      if (Math.random() < 0.1) {
        const mockMetrics: Partial<ConvergenceMetrics> = {
          internalizationRate: Math.min(0.95, 0.5 + Math.random() * 0.4),
          thoughtQuality: Math.min(0.9, 0.6 + Math.random() * 0.3),
          parallelismEfficiency: Math.min(0.85, 0.4 + Math.random() * 0.4)
        };

        this.callbacks.onConvergenceUpdate?.(mockMetrics);
      }

      // Occasionally update curriculum progress
      if (Math.random() < 0.05) {
        const stageId = Math.floor(Math.random() * 6);
        const mockUpdates: Partial<CurriculumStage> = {
          progress: Math.min(100, Math.random() * 100)
        };

        this.callbacks.onCurriculumProgress?.(stageId, mockUpdates);
      }

    }, 500 + Math.random() * 1000); // Random interval between 500-1500ms
  }

  private stopMockData(): void {
    if (this.mockInterval) {
      clearInterval(this.mockInterval);
      this.mockInterval = null;
    }
  }

  private generateMockContent(type: ReasoningToken['type']): string {
    const contents = {
      thought_start: ['<|startofthought|>', 'Beginning reasoning...', 'Let me think about this...'],
      thought_end: ['<|endofthought|>', 'Reasoning complete.', 'Conclusion reached.'],
      thought_content: [
        'This problem requires analyzing the relationship between...',
        'The key insight is that we need to consider...',
        'Given the constraints, the optimal approach would be...',
        'Breaking this down step by step...',
        'The logical conclusion follows from...'
      ],
      regular: ['The', 'answer', 'is', 'based', 'on', 'careful', 'analysis', 'of', 'the', 'problem']
    };

    const typeContents = contents[type];
    return typeContents[Math.floor(Math.random() * typeContents.length)];
  }
}

// Type guards for runtime validation
export const isWebSocketMessage = (data: any): data is WebSocketMessage => {
  return (
    typeof data === 'object' &&
    typeof data.type === 'string' &&
    typeof data.timestamp === 'number' &&
    data.data !== undefined
  );
};

export default QuietStarWebSocket;