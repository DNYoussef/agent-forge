/**
 * UI Components Index
 * Central export point for all Agent Forge UI components
 */

// Phase Controller
export { PhaseController } from './PhaseController';

// Phase 3: Quiet Star Components
export { QuietStarDashboard } from './QuietStarDashboard';
export { ReasoningTokenVisualizer } from './ReasoningTokenVisualizer';
export { ParallelThoughtsViewer } from './ParallelThoughtsViewer';
export { CurriculumProgressTracker } from './CurriculumProgressTracker';
export { ConvergenceMetrics } from './ConvergenceMetrics';
export { default as ThreeJSVisualization } from './ThreeJSVisualization';

// Re-export types
export type {
  PhaseStatus,
  MergeTechnique,
  EvoMergeConfig,
  ParetoPoint,
  FitnessGeneration,
  EvoMergeMetrics,
  CognateConfig,
  CognateMetrics,
  PhaseResponse,
  PhaseMetricsResponse,
  PhaseConfigRequest,
  // Quiet Star types
  ReasoningToken,
  ThoughtStream,
  CurriculumStage,
  ConvergenceMetrics as ConvergenceMetricsType,
  QuietStarConfig,
  QuietStarMetrics,
  WebSocketMessage
} from '../types/phases';