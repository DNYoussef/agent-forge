/**
 * Next.js API Route for EvoMerge Phase Management
 * Integrates with Python backend with fallback to simulation
 * CRITICAL: Maintains exact response formats for UI compatibility
 */

import { NextRequest, NextResponse } from 'next/server';
import { apiUtils, NetworkError } from '../../../../../api/utils/api-client';
import {
  getEvoMergeMetrics as simulateGetEvoMergeMetrics,
  initializeEvoMergeSimulation,
  stopEvoMergeSimulation,
  cleanupEvoMergeSimulation,
  resetEvoMergeSimulation,
} from '../../../../../api/simulation/evomerge-simulation';
import {
  EvoMergeMetrics,
  ApiResponse,
} from '../../../../../api/types/phase-interfaces';

/**
 * GET /api/phases/evomerge?sessionId=xxx - Get EvoMerge Metrics
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  console.log('[API] GET /api/phases/evomerge - Getting metrics for session:', sessionId);

  try {
    if (!sessionId) {
      return NextResponse.json(
        {
          success: false,
          error: 'sessionId parameter is required',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    // Define fallback function that maintains exact response format
    const fallbackToSimulation = async (): Promise<EvoMergeMetrics> => {
      console.warn('[API] Falling back to simulation for EvoMerge metrics');
      return await simulateGetEvoMergeMetrics(sessionId);
    };

    try {
      // Attempt real backend call with automatic fallback
      const result = await apiUtils.getEvoMergeMetrics(sessionId, fallbackToSimulation);

      console.log('[API] EvoMerge metrics retrieved successfully for session:', sessionId);
      return NextResponse.json(result, { status: 200 });

    } catch (error) {
      if (error instanceof NetworkError) {
        // Fallback to simulation
        const fallbackResult = await fallbackToSimulation();
        return NextResponse.json(fallbackResult, { status: 200 });
      }

      throw error; // Re-throw non-network errors
    }

  } catch (error) {
    console.error('[API] Error in EvoMerge phase GET handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
      sessionId,
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}

/**
 * POST /api/phases/evomerge - Initialize EvoMerge Phase
 *
 * Body:
 * {
 *   sessionId: string;
 *   populationSize?: number;
 *   maxGenerations?: number;
 *   mutationRate?: number;
 *   crossoverRate?: number;
 *   eliteCount?: number;
 * }
 */
export async function POST(request: NextRequest) {
  console.log('[API] POST /api/phases/evomerge - Initializing EvoMerge phase');

  try {
    const body = await request.json();

    // Validate required fields
    if (!body.sessionId) {
      return NextResponse.json(
        {
          success: false,
          error: 'sessionId is required',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    const sessionId = body.sessionId;

    try {
      // Try real backend first
      const result = await fetch('http://localhost:8001/api/evomerge/initialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          populationSize: body.populationSize || 50,
          maxGenerations: body.maxGenerations || 100,
          mutationRate: body.mutationRate || 0.1,
          crossoverRate: body.crossoverRate || 0.7,
          eliteCount: body.eliteCount || 5,
        }),
      });

      if (!result.ok) {
        throw new Error('Backend initialization failed');
      }

      const data = await result.json();
      console.log('[API] EvoMerge phase initialized via backend:', sessionId);
      return NextResponse.json(data, { status: 200 });

    } catch (error) {
      console.warn('[API] Falling back to simulation initialization for session:', sessionId);

      // Fallback to simulation initialization
      initializeEvoMergeSimulation(sessionId);

      const response: ApiResponse = {
        success: true,
        sessionId,
        status: 'initialized',
        timestamp: new Date().toISOString(),
      };

      return NextResponse.json(response, { status: 200 });
    }

  } catch (error) {
    console.error('[API] Error in EvoMerge phase POST handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}

/**
 * PUT /api/phases/evomerge?sessionId=xxx - Reset EvoMerge Phase
 */
export async function PUT(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  console.log('[API] PUT /api/phases/evomerge - Resetting session:', sessionId);

  try {
    if (!sessionId) {
      return NextResponse.json(
        {
          success: false,
          error: 'sessionId parameter is required',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    try {
      // Try real backend first
      const result = await fetch(`http://localhost:8001/api/evomerge/reset/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!result.ok) {
        throw new Error('Backend reset request failed');
      }

      const data = await result.json();
      console.log('[API] EvoMerge phase reset via backend:', sessionId);
      return NextResponse.json(data, { status: 200 });

    } catch (error) {
      console.warn('[API] Falling back to simulation reset for session:', sessionId);

      // Fallback to simulation reset
      resetEvoMergeSimulation(sessionId);

      const response: ApiResponse = {
        success: true,
        sessionId,
        status: 'reset',
        timestamp: new Date().toISOString(),
      };

      return NextResponse.json(response, { status: 200 });
    }

  } catch (error) {
    console.error('[API] Error in EvoMerge phase PUT handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
      sessionId,
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}

/**
 * DELETE /api/phases/evomerge?sessionId=xxx - Stop EvoMerge Phase
 */
export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  console.log('[API] DELETE /api/phases/evomerge - Stopping session:', sessionId);

  try {
    if (!sessionId) {
      return NextResponse.json(
        {
          success: false,
          error: 'sessionId parameter is required',
          timestamp: new Date().toISOString(),
        },
        { status: 400 }
      );
    }

    try {
      // Try real backend first
      const result = await fetch(`http://localhost:8001/api/evomerge/stop/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!result.ok) {
        throw new Error('Backend stop request failed');
      }

      const data = await result.json();
      console.log('[API] EvoMerge phase stopped via backend:', sessionId);
      return NextResponse.json(data, { status: 200 });

    } catch (error) {
      console.warn('[API] Falling back to simulation stop for session:', sessionId);

      // Fallback to simulation stop
      stopEvoMergeSimulation(sessionId);
      cleanupEvoMergeSimulation(sessionId);

      const response: ApiResponse = {
        success: true,
        sessionId,
        status: 'stopped',
        timestamp: new Date().toISOString(),
      };

      return NextResponse.json(response, { status: 200 });
    }

  } catch (error) {
    console.error('[API] Error in EvoMerge phase DELETE handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
      sessionId,
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}