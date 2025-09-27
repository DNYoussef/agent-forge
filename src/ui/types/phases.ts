/**
 * Type definitions for Agent Forge Phase system
 */

export type PhaseStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export type MergeTechnique =
  | 'linear'
  | 'slerp'
  | 'ties'
  | 'dare'
  | 'frankenmerge'
  | 'dfs'
  | 'task_arithmetic';

export interface EvoMergeConfig {
  generations: number;
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  mergeTechniques: MergeTechnique[];
  eliteSize: number;
  tournamentSize: number;
  domainWeights: {
    code: number;
    math: number;
    multilingual: number;
    structuredData: number;
  };
}

export interface ParetoPoint {
  performance: number;
  efficiency: number;
  isPareto: boolean;
}

export interface FitnessGeneration {
  generation: number;
  best: number;
  avg: number;
  worst: number;
}

export interface EvoMergeMetrics {
  currentGeneration: number;
  bestFitness: number;
  avgFitness: number;
  worstFitness?: number;
  diversityScore: number;
  paretoFront?: ParetoPoint[];
  fitnessHistory?: FitnessGeneration[];
}

export interface CognateConfig {
  modelTypes: ('planner' | 'reasoner' | 'memory')[];
  baseArchitecture: 'transformer' | 'llama' | 'gpt';
  modelSize: 'nano' | 'small' | 'medium' | 'large';
  learningRate: number;
  batchSize: number;
  maxEpochs: number;
  warmupSteps: number;
  enableGrokfast: boolean;
  emaAlpha: number;
  lambdaFactor: number;
}

export interface CognateMetrics {
  trainingLoss: number[];
  validationPerplexity: { [key: string]: number[] };
  grokkingProgress: number;
  memoryUsage: {
    gpu: number;
    cpu: number;
  };
  modelQuality: number;
  tokenThroughput: number;
}

export interface PhaseResponse {
  status: PhaseStatus;
  message?: string;
  data?: any;
}

export interface PhaseMetricsResponse<T> {
  metrics: T;
  timestamp: string;
}

export interface PhaseConfigRequest<T> {
  config: T;
}

// Phase 3: Quiet Star Reasoning Types
export interface ReasoningToken {
  id: string;
  content: string;
  type: 'thought_start' | 'thought_content' | 'thought_end' | 'regular';
  position: number;
  timestamp: number;
  confidence: number;
  streamId: number;
}

export interface ThoughtStream {
  id: number;
  tokens: ReasoningToken[];
  isActive: boolean;
  temperature: number;
  length: number;
  convergenceScore: number;
}

export interface CurriculumStage {
  id: number;
  name: string;
  description: string;
  progress: number;
  isActive: boolean;
  thoughtLength: {
    min: number;
    max: number;
    current: number;
  };
  metrics: {
    accuracy: number;
    efficiency: number;
    internalization: number;
  };
}

export interface ConvergenceMetrics {
  internalizationRate: number;
  thoughtQuality: number;
  parallelismEfficiency: number;
  curriculumProgress: number;
  tokenGeneration: {
    rate: number;
    accuracy: number;
    diversity: number;
  };
  learning: {
    convergenceSpeed: number;
    stabilityScore: number;
    adaptationRate: number;
  };
}

export interface QuietStarConfig {
  parallelStreams: number;
  maxThoughtLength: number;
  temperature: number;
  curriculumEnabled: boolean;
  visualizationMode: '2d' | '3d';
  realTimeUpdates: boolean;
}

export interface QuietStarMetrics {
  streams: ThoughtStream[];
  curriculum: CurriculumStage[];
  convergence: ConvergenceMetrics;
  totalTokensGenerated: number;
  averageThoughtLength: number;
  activeStreams: number;
}

export interface WebSocketMessage {
  type: 'reasoning_token' | 'stream_update' | 'convergence_update' | 'curriculum_progress';
  data: any;
  timestamp: number;
}