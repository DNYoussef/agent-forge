/**
 * BitNet Phase Simulation Functions
 * Provides fallback simulation for BitNet compression metrics when Python backend is unavailable
 * CRITICAL: Maintains exact response formats for UI compatibility
 */

import { BitNetMetrics } from '../types/phase-interfaces';

interface BitNetSimulationState {
  sessionId: string;
  startTime: Date;
  lastActivity: Date;
  currentPhase: 'initializing' | 'calibration' | 'quantization' | 'fine_tuning' | 'completed';
  layersProcessed: number;
  totalLayers: number;
  isActive: boolean;
}

// In-memory simulation state
const bitnetSimulationStates = new Map<string, BitNetSimulationState>();

/**
 * Get BitNet compression metrics with realistic simulation
 * Maintains exact response format as real backend
 */
export async function getBitNetMetrics(sessionId: string): Promise<BitNetMetrics> {
  console.log('[SIMULATION] Getting BitNet metrics for session:', sessionId);

  let state = bitnetSimulationStates.get(sessionId);

  if (!state) {
    // Initialize new simulation state
    state = {
      sessionId,
      startTime: new Date(),
      lastActivity: new Date(),
      currentPhase: 'initializing',
      layersProcessed: 0,
      totalLayers: 64, // Typical transformer layer count
      isActive: true,
    };
    bitnetSimulationStates.set(sessionId, state);
  }

  // Update state
  state.lastActivity = new Date();

  // Calculate elapsed time for progression
  const elapsedMinutes = (Date.now() - state.startTime.getTime()) / (1000 * 60);
  const progressFactor = Math.min(1.0, elapsedMinutes / 15); // Progress over 15 minutes

  // Update simulation phases based on progress
  if (progressFactor < 0.1) {
    state.currentPhase = 'initializing';
  } else if (progressFactor < 0.3) {
    state.currentPhase = 'calibration';
  } else if (progressFactor < 0.8) {
    state.currentPhase = 'quantization';
    state.layersProcessed = Math.floor(progressFactor * state.totalLayers);
  } else if (progressFactor < 0.95) {
    state.currentPhase = 'fine_tuning';
    state.layersProcessed = state.totalLayers;
  } else {
    state.currentPhase = 'completed';
    state.layersProcessed = state.totalLayers;
    state.isActive = false;
  }

  // Calculate compression metrics with realistic values
  const compressionProgress = Math.min(100, progressFactor * 100);
  const baseCompressionRatio = 8.0; // BitNet typical 8x compression
  const currentCompressionRatio = 1 + (baseCompressionRatio - 1) * progressFactor;

  // Model size reduction
  const originalSizeMB = 6400; // 6.4GB model
  const currentSizeMB = originalSizeMB / currentCompressionRatio;

  // Performance retention (starts at 100%, may dip during compression, recovers with fine-tuning)
  let performanceRetention = 100;
  if (state.currentPhase === 'quantization') {
    performanceRetention = 85 + Math.random() * 10; // 85-95% during quantization
  } else if (state.currentPhase === 'fine_tuning') {
    performanceRetention = 90 + progressFactor * 8; // Recovers to 98%
  } else if (state.currentPhase === 'completed') {
    performanceRetention = 96 + Math.random() * 3; // Final 96-99%
  }

  // Weight distribution - BitNet uses {-1, 0, +1}
  const totalWeights = state.layersProcessed * 1000000; // Approx weights per layer
  const quantizationProgress = state.layersProcessed / state.totalLayers;

  // During quantization, weights get converted to BitNet values
  const negative = Math.floor(totalWeights * 0.35 * quantizationProgress);
  const positive = Math.floor(totalWeights * 0.35 * quantizationProgress);
  const zero = Math.floor(totalWeights * 0.30 * quantizationProgress);

  const weightsDistribution = {
    negative: negative / totalWeights,
    zero: zero / totalWeights,
    positive: positive / totalWeights
  };

  // Calculate sparsity ratio (percentage of zero weights)
  const sparsityRatio = weightsDistribution.zero;

  // Estimate completion time
  const remainingMinutes = Math.max(0, 15 - elapsedMinutes);
  const estimatedCompletion = new Date(Date.now() + remainingMinutes * 60 * 1000).toISOString();

  const metrics: BitNetMetrics = {
    compressionProgress: Math.round(compressionProgress * 10) / 10,
    memoryReduction: Math.round(currentCompressionRatio * 10) / 10,
    performanceRetention: Math.round(performanceRetention * 10) / 10,
    quantizedLayers: state.layersProcessed,
    totalLayers: state.totalLayers,
    modelSizeMB: Math.round(currentSizeMB * 10) / 10,
    compressionRatio: Math.round(currentCompressionRatio * 10) / 10,
    sparsityRatio: Math.round(sparsityRatio * 100) / 100,
    quantizationBits: 1.58,
    weightsDistribution,
    currentPhase: state.currentPhase,
    layerProgress: Math.round((state.layersProcessed / state.totalLayers) * 100),
    avgQuantizationTime: 0.5 + Math.random() * 0.3, // 0.5-0.8 seconds per layer
    lastUpdated: new Date().toISOString(),
    estimatedCompletion
  };

  console.log('[SIMULATION] BitNet metrics generated:', {
    session: sessionId,
    phase: state.currentPhase,
    progress: compressionProgress,
    layers: `${state.layersProcessed}/${state.totalLayers}`
  });

  return metrics;
}

/**
 * Initialize BitNet simulation for a session
 */
export function initializeBitNetSimulation(sessionId: string): void {
  console.log('[SIMULATION] Initializing BitNet simulation for session:', sessionId);

  const state: BitNetSimulationState = {
    sessionId,
    startTime: new Date(),
    lastActivity: new Date(),
    currentPhase: 'initializing',
    layersProcessed: 0,
    totalLayers: 64,
    isActive: true,
  };

  bitnetSimulationStates.set(sessionId, state);
}

/**
 * Stop BitNet simulation for a session
 */
export function stopBitNetSimulation(sessionId: string): void {
  console.log('[SIMULATION] Stopping BitNet simulation for session:', sessionId);

  const state = bitnetSimulationStates.get(sessionId);
  if (state) {
    state.isActive = false;
    state.currentPhase = 'completed';
  }
}

/**
 * Reset BitNet simulation for a session
 */
export function resetBitNetSimulation(sessionId: string): void {
  console.log('[SIMULATION] Resetting BitNet simulation for session:', sessionId);

  const state = bitnetSimulationStates.get(sessionId);
  if (state) {
    state.startTime = new Date();
    state.lastActivity = new Date();
    state.currentPhase = 'initializing';
    state.layersProcessed = 0;
    state.isActive = true;
  }
}

/**
 * Cleanup BitNet simulation state for a session
 */
export function cleanupBitNetSimulation(sessionId: string): void {
  console.log('[SIMULATION] Cleaning up BitNet simulation for session:', sessionId);
  bitnetSimulationStates.delete(sessionId);
}

/**
 * Get all active BitNet simulation sessions
 */
export function getActiveBitNetSessions(): string[] {
  return Array.from(bitnetSimulationStates.entries())
    .filter(([_, state]) => state.isActive)
    .map(([sessionId]) => sessionId);
}