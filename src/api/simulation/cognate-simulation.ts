/**
 * Cognate Phase Simulation Functions
 * Provides fallback simulation when Python backend is unavailable
 * CRITICAL: Maintains exact response formats for UI compatibility
 */

import {
  CognateConfig,
  CognateMetrics,
  CognateStartResponse,
  CognateStatusResponse,
  PhaseStatus,
} from '../types/phase-interfaces';

interface SimulationSession {
  config: CognateConfig;
  metrics: CognateMetrics;
  status: PhaseStatus;
  startTime: Date;
  lastActivity: Date;
  errorHistory: string[];
}

// In-memory simulation state
const simulationSessions = new Map<string, SimulationSession>();
const simulationTimers = new Map<string, NodeJS.Timeout>();

/**
 * Start Cognate phase simulation
 * Maintains exact response format as real backend
 */
export async function startCognatePhase(
  sessionId: string,
  config: CognateConfig
): Promise<CognateStartResponse> {
  console.log('[SIMULATION] Starting Cognate phase simulation for session:', sessionId);

  // Clean up existing session if present
  if (simulationSessions.has(sessionId)) {
    stopCognatePhaseSimulation(sessionId);
  }

  // Create simulation session
  const session: SimulationSession = {
    config: {
      maxIterations: 10,
      convergenceThreshold: 0.95,
      parallelAgents: 3,
      timeout: 300,
      enableDebugging: false,
      customParams: {},
      ...config,
      sessionId, // Ensure sessionId is always present
    },
    metrics: {
      iterationsCompleted: 0,
      convergenceScore: 0.0,
      activeAgents: 0,
      averageResponseTime: 0.0,
      errorCount: 0,
      successRate: 100.0,
      lastUpdated: new Date().toISOString(),
      estimatedCompletion: '',
      throughput: 0.0,
      memoryUsage: 0.0,
    },
    status: PhaseStatus.INITIALIZING,
    startTime: new Date(),
    lastActivity: new Date(),
    errorHistory: [],
  };

  simulationSessions.set(sessionId, session);

  // Start simulation timer
  const timer = setTimeout(() => {
    runCognateSimulation(sessionId);
  }, 1000); // Start after 1 second

  simulationTimers.set(sessionId, timer);

  // Simulate initialization delay
  await new Promise(resolve => setTimeout(resolve, 500));

  session.status = PhaseStatus.RUNNING;
  session.lastActivity = new Date();

  return {
    success: true,
    sessionId,
    status: PhaseStatus.INITIALIZING,
    estimatedDuration: session.config.maxIterations * 30,
    config: session.config,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Get Cognate phase status and metrics
 * Maintains exact response format as real backend
 */
export async function getCognateStatus(sessionId: string): Promise<CognateStatusResponse> {
  const session = simulationSessions.get(sessionId);

  if (!session) {
    throw new Error(`Session ${sessionId} not found`);
  }

  session.lastActivity = new Date();

  return {
    sessionId,
    status: session.status,
    currentPhase: 'cognate',
    startTime: session.startTime.toISOString(),
    lastActivity: session.lastActivity.toISOString(),
    metrics: { ...session.metrics },
    config: { ...session.config },
    errors: [...session.errorHistory.slice(-5)], // Last 5 errors
    isActive: simulationTimers.has(sessionId),
  };
}

/**
 * Stop Cognate phase simulation
 */
export function stopCognatePhaseSimulation(sessionId: string): boolean {
  const session = simulationSessions.get(sessionId);
  const timer = simulationTimers.get(sessionId);

  if (timer) {
    clearTimeout(timer);
    simulationTimers.delete(sessionId);
  }

  if (session) {
    session.status = PhaseStatus.CANCELLED;
    session.lastActivity = new Date();
  }

  return true;
}

/**
 * Run the actual simulation logic
 */
function runCognateSimulation(sessionId: string): void {
  const session = simulationSessions.get(sessionId);

  if (!session || session.status === PhaseStatus.CANCELLED) {
    return;
  }

  session.status = PhaseStatus.RUNNING;

  const simulationStep = () => {
    const session = simulationSessions.get(sessionId);
    if (!session || session.status !== PhaseStatus.RUNNING) {
      return;
    }

    // Update metrics realistically
    session.metrics.iterationsCompleted += 1;

    const progress = session.metrics.iterationsCompleted / session.config.maxIterations;

    // Realistic convergence curve (sigmoid-like)
    session.metrics.convergenceScore = Math.min(
      0.95,
      0.1 + progress * 0.7 + 0.15 * (1 - Math.exp(-5 * progress))
    );

    // Realistic agent activation
    session.metrics.activeAgents = Math.min(
      session.config.parallelAgents,
      Math.ceil(progress * session.config.parallelAgents)
    );

    // Fluctuating response time
    session.metrics.averageResponseTime = 1.2 + 0.3 * Math.sin(session.metrics.iterationsCompleted);

    // Gradually decreasing success rate (realistic)
    session.metrics.successRate = Math.max(85.0, 100.0 - (progress * 15));

    // Increasing throughput
    session.metrics.throughput = 8.0 + progress * 5.0 + Math.random() * 2.0;

    // Memory usage curve
    session.metrics.memoryUsage = Math.min(95.0, 15.0 + progress * 60.0 + Math.random() * 10.0);

    session.metrics.lastUpdated = new Date().toISOString();
    session.lastActivity = new Date();

    // Calculate ETA
    if (progress > 0) {
      const remainingIterations = session.config.maxIterations - session.metrics.iterationsCompleted;
      const etaSeconds = remainingIterations * 2.5; // Assume 2.5 seconds per iteration
      const eta = new Date(Date.now() + etaSeconds * 1000);
      session.metrics.estimatedCompletion = eta.toISOString();
    }

    // Add occasional errors for realism
    if (session.metrics.iterationsCompleted > 2 &&
        session.errorHistory.length < 3 &&
        Math.random() < 0.1) {
      const agentId = (session.metrics.iterationsCompleted % 3) + 1;
      session.errorHistory.push(
        `Simulation: Timeout on agent ${agentId} at iteration ${session.metrics.iterationsCompleted}`
      );
      session.metrics.errorCount += 1;
    }

    // Check completion conditions
    if (session.metrics.iterationsCompleted >= session.config.maxIterations ||
        session.metrics.convergenceScore >= session.config.convergenceThreshold) {
      session.status = PhaseStatus.COMPLETED;
      session.lastActivity = new Date();

      const timer = simulationTimers.get(sessionId);
      if (timer) {
        clearTimeout(timer);
        simulationTimers.delete(sessionId);
      }

      console.log(`[SIMULATION] Cognate phase completed for session ${sessionId}`);
      return;
    }

    // Schedule next update
    const nextTimer = setTimeout(simulationStep, 2500); // Every 2.5 seconds
    simulationTimers.set(sessionId, nextTimer);
  };

  // Start the simulation loop
  setTimeout(simulationStep, 1000);
}

/**
 * Clean up simulation data for session
 */
export function cleanupSimulationSession(sessionId: string): void {
  stopCognatePhaseSimulation(sessionId);
  simulationSessions.delete(sessionId);
}

/**
 * Get all simulation sessions (for debugging)
 */
export function getSimulationSessions(): Array<{ sessionId: string; status: PhaseStatus }> {
  return Array.from(simulationSessions.entries()).map(([sessionId, session]) => ({
    sessionId,
    status: session.status,
  }));
}