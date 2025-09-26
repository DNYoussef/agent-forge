import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

// Pipeline status storage (in production, use a database)
const pipelineStatus = new Map<string, any>();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, phases, config } = body;

    const sessionId = `pipeline-${Date.now()}`;

    switch (action) {
      case 'start': {
        // Store initial status
        pipelineStatus.set(sessionId, {
          id: sessionId,
          status: 'running',
          phases: phases || [1, 2, 3, 4, 5, 6, 7, 8],
          currentPhase: 1,
          startTime: new Date(),
          progress: 0,
          config: config || {}
        });

        // Start the pipeline in background
        startPipelineBackground(sessionId, phases, config);

        return NextResponse.json({
          success: true,
          sessionId,
          message: 'Pipeline started successfully'
        });
      }

      case 'stop': {
        const { sessionId } = body;
        if (pipelineStatus.has(sessionId)) {
          const status = pipelineStatus.get(sessionId);
          status.status = 'stopped';
          status.endTime = new Date();
          pipelineStatus.set(sessionId, status);
        }

        return NextResponse.json({
          success: true,
          message: 'Pipeline stopped'
        });
      }

      case 'pause': {
        const { sessionId } = body;
        if (pipelineStatus.has(sessionId)) {
          const status = pipelineStatus.get(sessionId);
          status.status = 'paused';
          pipelineStatus.set(sessionId, status);
        }

        return NextResponse.json({
          success: true,
          message: 'Pipeline paused'
        });
      }

      case 'resume': {
        const { sessionId } = body;
        if (pipelineStatus.has(sessionId)) {
          const status = pipelineStatus.get(sessionId);
          status.status = 'running';
          pipelineStatus.set(sessionId, status);
        }

        return NextResponse.json({
          success: true,
          message: 'Pipeline resumed'
        });
      }

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('Pipeline API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const sessionId = searchParams.get('sessionId');

  if (sessionId && pipelineStatus.has(sessionId)) {
    const status = pipelineStatus.get(sessionId);

    // Simulate progress update
    if (status.status === 'running' && status.progress < 100) {
      status.progress = Math.min(status.progress + Math.random() * 5, 100);

      // Update current phase based on progress
      const phaseProgress = status.progress / (100 / status.phases.length);
      status.currentPhase = Math.min(
        Math.ceil(phaseProgress) || 1,
        status.phases.length
      );

      if (status.progress >= 100) {
        status.status = 'completed';
        status.endTime = new Date();
      }

      pipelineStatus.set(sessionId, status);
    }

    return NextResponse.json(status);
  }

  // Return all pipeline statuses if no sessionId specified
  const allStatuses = Array.from(pipelineStatus.values());
  return NextResponse.json(allStatuses);
}

// Background pipeline execution (simulated for now)
async function startPipelineBackground(
  sessionId: string,
  phases: number[],
  config: any
) {
  // In production, this would call the actual Python pipeline
  // For now, we simulate progress
  const pythonScriptPath = path.join(
    process.cwd(),
    '..', '..', '..',
    'swarm_cli.py'
  );

  try {
    // Attempt to run actual pipeline if available
    const command = `python "${pythonScriptPath}" execute --phases ${phases.join(',')}`;
    console.log('Executing:', command);

    // Non-blocking execution
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error('Pipeline execution error:', error);
        if (pipelineStatus.has(sessionId)) {
          const status = pipelineStatus.get(sessionId);
          status.status = 'error';
          status.error = error.message;
          pipelineStatus.set(sessionId, status);
        }
      } else {
        console.log('Pipeline output:', stdout);
        if (pipelineStatus.has(sessionId)) {
          const status = pipelineStatus.get(sessionId);
          status.status = 'completed';
          status.progress = 100;
          status.endTime = new Date();
          pipelineStatus.set(sessionId, status);
        }
      }
    });
  } catch (error) {
    console.error('Failed to start pipeline:', error);
  }
}