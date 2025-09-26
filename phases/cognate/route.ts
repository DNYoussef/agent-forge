/**
 * Next.js API Route for Cognate Phase Management
 * Integrates with Python backend with fallback to simulation
 * CRITICAL: Maintains exact response formats for UI compatibility
 */

import { NextRequest, NextResponse } from 'next/server';
import { apiUtils, NetworkError } from '../../../../../api/utils/api-client';
import {
  startCognatePhase as simulateStartCognatePhase,
  getCognateStatus as simulateGetCognateStatus,
  stopCognatePhaseSimulation,
  cleanupSimulationSession,
} from '../../../../../api/simulation/cognate-simulation';
import {
  CognateConfig,
  CognateStartResponse,
  CognateStatusResponse,
  ApiResponse,
} from '../../../../../api/types/phase-interfaces';

/**
 * POST /api/phases/cognate - Start Cognate Phase
 *
 * Body:
 * {
 *   sessionId: string;
 *   maxIterations?: number;
 *   convergenceThreshold?: number;
 *   parallelAgents?: number;
 *   timeout?: number;
 *   enableDebugging?: boolean;
 *   customParams?: Record<string, any>;
 * }
 */
export async function POST(request: NextRequest) {
  console.log('[API] POST /api/phases/cognate - Starting Cognate phase');

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

    const config: CognateConfig = {
      sessionId: body.sessionId,
      maxIterations: body.maxIterations || 10,
      convergenceThreshold: body.convergenceThreshold || 0.95,
      parallelAgents: body.parallelAgents || 3,
      timeout: body.timeout || 300,
      enableDebugging: body.enableDebugging || false,
      customParams: body.customParams || {},
    };

    // Define fallback function that maintains exact response format
    const fallbackToSimulation = async (): Promise<CognateStartResponse> => {
      console.warn('[API] Falling back to simulation for Cognate phase start');
      return await simulateStartCognatePhase(config.sessionId, config);
    };

    try {
      // Attempt real backend call with automatic fallback
      const result = await apiUtils.startCognatePhase(config, fallbackToSimulation);

      console.log('[API] Cognate phase started successfully:', result.sessionId);
      return NextResponse.json(result, { status: 200 });

    } catch (error) {
      // This shouldn't happen due to fallback, but handle just in case
      console.error('[API] Unexpected error starting Cognate phase:', error);

      // Emergency fallback
      const fallbackResult = await fallbackToSimulation();
      return NextResponse.json(fallbackResult, { status: 200 });
    }

  } catch (error) {
    console.error('[API] Error in Cognate phase POST handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}

/**
 * GET /api/phases/cognate?sessionId=xxx - Get Cognate Phase Status
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  console.log('[API] GET /api/phases/cognate - Getting status for session:', sessionId);

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
    const fallbackToSimulation = async (): Promise<CognateStatusResponse> => {
      console.warn('[API] Falling back to simulation for Cognate status');
      return await simulateGetCognateStatus(sessionId);
    };

    try {
      // Attempt real backend call with automatic fallback
      const result = await apiUtils.getCognateStatus(sessionId, fallbackToSimulation);

      console.log('[API] Cognate status retrieved successfully for session:', sessionId);
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
    console.error('[API] Error in Cognate phase GET handler:', error);

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
 * DELETE /api/phases/cognate?sessionId=xxx - Stop Cognate Phase
 */
export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get('sessionId');

  console.log('[API] DELETE /api/phases/cognate - Stopping session:', sessionId);

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
      const result = await fetch(`http://localhost:8001/api/cognate/stop/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!result.ok) {
        throw new Error('Backend stop request failed');
      }

      const data = await result.json();
      console.log('[API] Cognate phase stopped via backend:', sessionId);
      return NextResponse.json(data, { status: 200 });

    } catch (error) {
      console.warn('[API] Falling back to simulation stop for session:', sessionId);

      // Fallback to simulation stop
      stopCognatePhaseSimulation(sessionId);
      cleanupSimulationSession(sessionId);

      const response: ApiResponse = {
        success: true,
        sessionId,
        status: 'cancelled',
        timestamp: new Date().toISOString(),
      };

      return NextResponse.json(response, { status: 200 });
    }

  } catch (error) {
    console.error('[API] Error in Cognate phase DELETE handler:', error);

    const errorResponse: ApiResponse = {
      success: false,
      error: error.message || 'Internal server error',
      timestamp: new Date().toISOString(),
      sessionId,
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}