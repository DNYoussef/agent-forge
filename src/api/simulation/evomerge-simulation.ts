/**
 * EvoMerge Phase Simulation Functions
 * Provides fallback simulation for EvoMerge metrics when Python backend is unavailable
 * CRITICAL: Maintains exact response formats for UI compatibility
 */

import { EvoMergeMetrics } from '../types/phase-interfaces';

interface EvoSimulationState {
  sessionId: string;
  startTime: Date;
  lastActivity: Date;
  generation: number;
  isActive: boolean;
}

// In-memory simulation state
const evoSimulationStates = new Map<string, EvoSimulationState>();

/**
 * Get EvoMerge metrics with realistic simulation
 * Maintains exact response format as real backend
 */
export async function getEvoMergeMetrics(sessionId: string): Promise<EvoMergeMetrics> {
  console.log('[SIMULATION] Getting EvoMerge metrics for session:', sessionId);

  let state = evoSimulationStates.get(sessionId);

  if (!state) {
    // Initialize new simulation state
    state = {
      sessionId,
      startTime: new Date(),
      lastActivity: new Date(),
      generation: 0,
      isActive: true,
    };
    evoSimulationStates.set(sessionId, state);
  }

  // Update state
  state.lastActivity = new Date();
  state.generation += 1;

  // Calculate elapsed time for progression
  const elapsedMinutes = (Date.now() - state.startTime.getTime()) / (1000 * 60);
  const progressFactor = Math.min(1.0, elapsedMinutes / 10); // Progress over 10 minutes

  // Realistic genetic algorithm metrics
  const basePopulationSize = 50;
  const maxGenerations = 100;

  // Fitness progression (starts low, improves over time with some randomness)
  const baseFitness = 0.3 + progressFactor * 0.5;
  const fitnessVariation = 0.1 * Math.sin(state.generation * 0.1) * (1 - progressFactor);

  const averageFitness = Math.min(0.95, baseFitness + fitnessVariation);
  const bestFitness = Math.min(0.98, averageFitness + 0.05 + Math.random() * 0.05);

  // Diversity starts high, decreases as population converges
  const diversityIndex = Math.max(0.1, 0.9 - progressFactor * 0.7 + Math.random() * 0.1);

  // Stagnation count increases when fitness doesn't improve much
  const stagnationCount = Math.floor(progressFactor * 5 + Math.random() * 3);

  // Dynamic mutation rate (higher when stagnating)
  const mutationRate = Math.max(0.05, 0.1 + stagnationCount * 0.02);

  // Crossover rate stays relatively stable
  const crossoverRate = 0.7 + Math.random() * 0.1;

  // Elite count based on population size
  const eliteCount = Math.max(3, Math.floor(basePopulationSize * 0.1));

  // Calculate estimated completion
  const remainingGenerations = Math.max(0, maxGenerations - state.generation);
  const etaMinutes = remainingGenerations * 0.5; // 30 seconds per generation
  const eta = new Date(Date.now() + etaMinutes * 60 * 1000);

  const metrics: EvoMergeMetrics = {
    generationsCompleted: Math.min(maxGenerations, state.generation),
    populationSize: basePopulationSize,
    fitnessScore: Number(bestFitness.toFixed(4)),
    mutationRate: Number(mutationRate.toFixed(3)),
    crossoverRate: Number(crossoverRate.toFixed(2)),
    eliteCount,
    averageFitness: Number(averageFitness.toFixed(4)),
    bestFitness: Number(bestFitness.toFixed(4)),
    diversityIndex: Number(diversityIndex.toFixed(3)),
    stagnationCount,
    lastUpdated: new Date().toISOString(),
    estimatedCompletion: eta.toISOString(),
  };

  return metrics;
}

/**
 * Initialize EvoMerge simulation for a session
 */
export function initializeEvoMergeSimulation(sessionId: string): void {
  console.log('[SIMULATION] Initializing EvoMerge simulation for session:', sessionId);

  const state: EvoSimulationState = {
    sessionId,
    startTime: new Date(),
    lastActivity: new Date(),
    generation: 0,
    isActive: true,
  };

  evoSimulationStates.set(sessionId, state);
}

/**
 * Stop EvoMerge simulation for a session
 */
export function stopEvoMergeSimulation(sessionId: string): void {
  console.log('[SIMULATION] Stopping EvoMerge simulation for session:', sessionId);

  const state = evoSimulationStates.get(sessionId);
  if (state) {
    state.isActive = false;
  }
}

/**
 * Clean up EvoMerge simulation data
 */
export function cleanupEvoMergeSimulation(sessionId: string): void {
  console.log('[SIMULATION] Cleaning up EvoMerge simulation for session:', sessionId);
  evoSimulationStates.delete(sessionId);
}

/**
 * Get all active EvoMerge simulations (for debugging)
 */
export function getActiveEvoMergeSimulations(): Array<{
  sessionId: string;
  generation: number;
  isActive: boolean;
  elapsedTime: number;
}> {
  return Array.from(evoSimulationStates.values()).map(state => ({
    sessionId: state.sessionId,
    generation: state.generation,
    isActive: state.isActive,
    elapsedTime: Date.now() - state.startTime.getTime(),
  }));
}

/**
 * Reset EvoMerge simulation for a session
 */
export function resetEvoMergeSimulation(sessionId: string): void {
  console.log('[SIMULATION] Resetting EvoMerge simulation for session:', sessionId);

  const state = evoSimulationStates.get(sessionId);
  if (state) {
    state.generation = 0;
    state.startTime = new Date();
    state.lastActivity = new Date();
    state.isActive = true;
  } else {
    initializeEvoMergeSimulation(sessionId);
  }
}