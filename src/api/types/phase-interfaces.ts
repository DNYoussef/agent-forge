/**
 * TypeScript interfaces for Phase API Integration
 * Ensures exact compatibility between simulation and real backend
 */

export enum PhaseStatus {
  IDLE = 'idle',
  INITIALIZING = 'initializing',
  RUNNING = 'running',
  COMPLETED = 'completed',
  ERROR = 'error',
  CANCELLED = 'cancelled',
}

export interface CognateConfig {
  sessionId: string;
  maxIterations?: number;
  convergenceThreshold?: number;
  parallelAgents?: number;
  timeout?: number;
  enableDebugging?: boolean;
  customParams?: Record<string, any>;
}

export interface CognateMetrics {
  iterationsCompleted: number;
  convergenceScore: number;
  activeAgents: number;
  averageResponseTime: number;
  errorCount: number;
  successRate: number;
  lastUpdated: string;
  estimatedCompletion: string;
  throughput: number;
  memoryUsage: number;
}

export interface EvoMergeMetrics {
  generationsCompleted: number;
  populationSize: number;
  fitnessScore: number;
  mutationRate: number;
  crossoverRate: number;
  eliteCount: number;
  averageFitness: number;
  bestFitness: number;
  diversityIndex: number;
  stagnationCount: number;
  lastUpdated: string;
  estimatedCompletion: string;
}

export interface SessionState {
  sessionId: string;
  status: PhaseStatus;
  currentPhase: string | null;
  startTime: string;
  lastActivity: string;
  cognateConfig?: CognateConfig;
  cognateMetrics?: CognateMetrics;
  evoMetrics?: EvoMergeMetrics;
  errorHistory?: string[];
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
  sessionId?: string;
  status?: string;
}

export interface CognateStartResponse {
  success: boolean;
  sessionId: string;
  status: string;
  estimatedDuration: number;
  config: CognateConfig;
  timestamp: string;
}

export interface CognateStatusResponse {
  sessionId: string;
  status: string;
  currentPhase: string | null;
  startTime: string;
  lastActivity: string;
  metrics: CognateMetrics | null;
  config: CognateConfig | null;
  errors: string[];
  isActive: boolean;
}

export interface SessionListResponse {
  sessions: Array<{
    sessionId: string;
    status: string;
    currentPhase: string | null;
    startTime: string;
    lastActivity: string;
    isActive: boolean;
  }>;
  totalSessions: number;
  activeTasks: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  sessions: number;
  active_tasks: number;
}

// API Client Configuration
export interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  enableFallback: boolean;
  fallbackDelay: number;
}

// Error Types
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class NetworkError extends ApiError {
  constructor(message: string, originalError?: Error) {
    super(message, 0, originalError);
    this.name = 'NetworkError';
  }
}

export class TimeoutError extends ApiError {
  constructor(message: string = 'Request timeout') {
    super(message, 408);
    this.name = 'TimeoutError';
  }
}

// Simulation fallback interfaces (preserve existing structure)
export interface SimulationConfig {
  enableLogging?: boolean;
  simulateLatency?: boolean;
  errorRate?: number;
  maxLatency?: number;
}

export interface FallbackOptions {
  enableFallback: boolean;
  fallbackDelay: number;
  maxRetries: number;
  simulationConfig?: SimulationConfig;
}

// BitNet Compression Interfaces
export interface BitNetMetrics {
  compressionProgress: number;
  memoryReduction: number;
  performanceRetention: number;
  quantizedLayers: number;
  totalLayers: number;
  modelSizeMB: number;
  compressionRatio: number;
  sparsityRatio: number;
  quantizationBits: number;
  weightsDistribution: {
    negative: number; // -1 values
    zero: number;     // 0 values
    positive: number; // +1 values
  };
  currentPhase: "initializing" | "calibration" | "quantization" | "fine_tuning" | "completed";
  layerProgress: number;
  avgQuantizationTime: number;
  lastUpdated: string;
  estimatedCompletion: string;
}

export interface BitNetConfig {
  sessionId: string;
  quantizationBits?: number;
  preserveCriticalLayers?: boolean;
  criticalLayerThreshold?: number;
  compressionRatio?: number;
  grokfastEnabled?: boolean;
  grokfastAlpha?: number;
  grokfastLambda?: number;
  learningRate?: number;
  fineTuneEpochs?: number;
  memoryOptimizationLevel?: string;
}
