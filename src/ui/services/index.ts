/**
 * UI Services Index
 * Central export point for all Agent Forge UI services
 */

// WebSocket Service
export {
  QuietStarWebSocket,
  MockQuietStarWebSocket,
  useQuietStarWebSocket,
  createReasoningTokenMessage,
  createStreamUpdateMessage,
  createConvergenceUpdateMessage,
  createCurriculumProgressMessage,
  isWebSocketMessage
} from './QuietStarWebSocket';

// Re-export types
export type {
  WebSocketConfig,
  QuietStarWebSocketCallbacks
} from './QuietStarWebSocket';