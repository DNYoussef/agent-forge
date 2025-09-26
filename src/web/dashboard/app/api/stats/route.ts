import { NextResponse } from 'next/server';

export async function GET() {
  const stats = {
    totalAgents: 0,
    successRate: 0,
    activePipelines: 0,
    avgPipelineTime: 0
  };

  return NextResponse.json(stats);
}
