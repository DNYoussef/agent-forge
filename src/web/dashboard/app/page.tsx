'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Activity, Cpu, Zap, Award, Play, Square, Settings, RefreshCw, CheckCircle } from 'lucide-react';

interface DashboardStats {
  totalAgents: number;
  successRate: number;
  activePipelines: number;
  avgPipelineTime: number;
}

interface PipelineStatus {
  id: string;
  status: string;
  phases: number[];
  currentPhase: number;
  progress: number;
  startTime: Date;
}

export default function EnhancedAgentForgeDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [selectedPhases, setSelectedPhases] = useState<number[]>([1, 2, 3, 4, 5, 6, 7, 8]);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [phaseStatuses, setPhaseStatuses] = useState<Record<number, string>>({});

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 3000);

    // Check for existing pipeline status
    fetchPipelineStatus();

    return () => clearInterval(interval);
  }, []);

  const fetchPipelineStatus = async () => {
    try {
      const response = await fetch('/api/pipeline');
      const pipelines = await response.json();
      if (pipelines.length > 0) {
        const latestPipeline = pipelines[pipelines.length - 1];
        setPipelineStatus(latestPipeline);
        setIsPipelineRunning(latestPipeline.status === 'running');

        // Update phase statuses
        const newStatuses: Record<number, string> = {};
        for (let i = 1; i <= 8; i++) {
          if (i < latestPipeline.currentPhase) {
            newStatuses[i] = 'completed';
          } else if (i === latestPipeline.currentPhase) {
            newStatuses[i] = 'running';
          } else {
            newStatuses[i] = 'pending';
          }
        }
        setPhaseStatuses(newStatuses);
      }
    } catch (error) {
      console.error('Failed to fetch pipeline status:', error);
    }
  };

  const startPipeline = async () => {
    try {
      const response = await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'start',
          phases: selectedPhases,
          config: {
            enableGrokfast: true,
            enableEdgeOfChaos: true,
            enableTheaterDetection: true
          }
        })
      });
      const data = await response.json();
      if (data.success) {
        setIsPipelineRunning(true);
        // Start polling for status
        const statusInterval = setInterval(async () => {
          const statusResponse = await fetch(`/api/pipeline?sessionId=${data.sessionId}`);
          const statusData = await statusResponse.json();
          setPipelineStatus(statusData);

          // Update individual phase statuses
          const newStatuses: Record<number, string> = {};
          for (let i = 1; i <= 8; i++) {
            if (i < statusData.currentPhase) {
              newStatuses[i] = 'completed';
            } else if (i === statusData.currentPhase) {
              newStatuses[i] = 'running';
            } else {
              newStatuses[i] = selectedPhases.includes(i) ? 'pending' : 'skipped';
            }
          }
          setPhaseStatuses(newStatuses);

          if (statusData.status === 'completed' || statusData.status === 'stopped') {
            setIsPipelineRunning(false);
            clearInterval(statusInterval);
          }
        }, 1000);
      }
    } catch (error) {
      console.error('Failed to start pipeline:', error);
    }
  };

  const stopPipeline = async () => {
    if (!pipelineStatus) return;
    try {
      await fetch('/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'stop',
          sessionId: pipelineStatus.id
        })
      });
      setIsPipelineRunning(false);
    } catch (error) {
      console.error('Failed to stop pipeline:', error);
    }
  };

  const phases = [
    { id: 1, name: 'Cognate', color: '#3b82f6', icon: '🧠', path: '/phases/cognate', desc: 'Model Creation' },
    { id: 2, name: 'EvoMerge', color: '#a855f7', icon: '🧬', path: '/phases/evomerge', desc: 'Evolution' },
    { id: 3, name: 'Quiet-STaR', color: '#06b6d4', icon: '💭', path: '/phases/quietstar', desc: 'Reasoning' },
    { id: 4, name: 'BitNet', color: '#f97316', icon: '📦', path: '/phases/bitnet', desc: 'Compression' },
    { id: 5, name: 'Forge', color: '#ef4444', icon: '🔥', path: '/phases/forge', desc: 'Training' },
    { id: 6, name: 'Baking', color: '#10b981', icon: '🛠️', path: '/phases/baking', desc: 'Tools' },
    { id: 7, name: 'ADAS', color: '#ec4899', icon: '🔬', path: '/phases/adas', desc: 'Architecture' },
    { id: 8, name: 'Final', color: '#eab308', icon: '✨', path: '/phases/final', desc: 'Production' }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
        <div className="text-center">
          <div className="animate-spin rounded-full h-20 w-20 border-t-4 border-b-4 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400 text-lg">Initializing Agent Forge...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <header className="mb-12">
        <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          Agent Forge
        </h1>
        <p className="text-xl text-gray-400">
          8-Phase AI Agent Creation Pipeline - Enhanced Control Center
        </p>
      </header>

      {/* Pipeline Control Panel */}
      <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Pipeline Control Center</h2>
          <Settings className="w-6 h-6 text-gray-400 cursor-pointer hover:text-white" />
        </div>

        {/* Phase Selection */}
        <div className="mb-4">
          <label className="text-sm text-gray-400 mb-2 block">Select Phases to Run:</label>
          <div className="flex gap-2 flex-wrap">
            {phases.map((phase) => (
              <button
                key={phase.id}
                onClick={() => {
                  if (selectedPhases.includes(phase.id)) {
                    setSelectedPhases(selectedPhases.filter(p => p !== phase.id));
                  } else {
                    setSelectedPhases([...selectedPhases, phase.id].sort());
                  }
                }}
                disabled={isPipelineRunning}
                className={`px-3 py-1 rounded-lg border transition-all ${
                  selectedPhases.includes(phase.id)
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-white/5 border-white/10 text-gray-400 hover:border-white/30'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {phase.icon} {phase.id}
              </button>
            ))}
          </div>
        </div>

        {/* Pipeline Progress */}
        {pipelineStatus && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Pipeline Progress</span>
              <span>{Math.round(pipelineStatus.progress)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${pipelineStatus.progress}%` }}
              />
            </div>
            <div className="mt-2 text-sm text-gray-400">
              Current Phase: {pipelineStatus.currentPhase} / {pipelineStatus.phases.length}
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="flex gap-4">
          {!isPipelineRunning ? (
            <button
              onClick={startPipeline}
              disabled={selectedPhases.length === 0}
              className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <Play className="w-5 h-5" />
              Start Pipeline
            </button>
          ) : (
            <button
              onClick={stopPipeline}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <Square className="w-5 h-5" />
              Stop Pipeline
            </button>
          )}

          <button
            onClick={fetchPipelineStatus}
            className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <Activity className="w-8 h-8 text-blue-400" />
            <div className="text-3xl font-bold text-blue-400">
              {stats?.totalAgents?.toLocaleString() || 0}
            </div>
          </div>
          <div className="text-gray-400">Total Agents Created</div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <Award className="w-8 h-8 text-green-400" />
            <div className="text-3xl font-bold text-green-400">
              {stats?.successRate || 0}%
            </div>
          </div>
          <div className="text-gray-400">Success Rate</div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <Zap className="w-8 h-8 text-yellow-400" />
            <div className="text-3xl font-bold text-yellow-400">
              {stats?.activePipelines || 0}
            </div>
          </div>
          <div className="text-gray-400">Active Pipelines</div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <Cpu className="w-8 h-8 text-purple-400" />
            <div className="text-3xl font-bold text-purple-400">
              {stats?.avgPipelineTime || 0}m
            </div>
          </div>
          <div className="text-gray-400">Avg Pipeline Time</div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
        {phases.map((phase) => (
          <Link key={phase.id} href={phase.path}>
            <div
              className={`bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-white/30 transition-all duration-300 hover:scale-105 cursor-pointer relative ${
                phaseStatuses[phase.id] === 'running' ? 'animate-pulse' : ''
              }`}
              style={{ borderColor: phaseStatuses[phase.id] === 'completed' ? phase.color + '60' : phase.color + '40' }}
            >
              {phaseStatuses[phase.id] === 'completed' && (
                <CheckCircle className="absolute top-2 right-2 w-5 h-5 text-green-400" />
              )}
              <div className="text-5xl mb-4">{phase.icon}</div>
              <div className="text-xl font-bold mb-2" style={{ color: phase.color }}>
                Phase {phase.id}
              </div>
              <div className="text-gray-400 mb-1">{phase.name}</div>
              <div className="text-sm text-gray-500">{phase.desc}</div>
              {phaseStatuses[phase.id] && (
                <div className={`text-xs mt-2 ${
                  phaseStatuses[phase.id] === 'completed' ? 'text-green-400' :
                  phaseStatuses[phase.id] === 'running' ? 'text-yellow-400' :
                  phaseStatuses[phase.id] === 'pending' ? 'text-blue-400' :
                  'text-gray-500'
                }`}>
                  {phaseStatuses[phase.id].toUpperCase()}
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link href="/runs">
          <div className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 rounded-2xl p-6 border border-blue-500/30 hover:border-blue-500/60 transition-all cursor-pointer">
            <h3 className="text-xl font-bold mb-2">Pipeline Runs</h3>
            <p className="text-gray-400">View active and historical pipeline executions</p>
          </div>
        </Link>

        <Link href="/analytics">
          <div className="bg-gradient-to-r from-green-600/20 to-emerald-600/20 rounded-2xl p-6 border border-green-500/30 hover:border-green-500/60 transition-all cursor-pointer">
            <h3 className="text-xl font-bold mb-2">Analytics</h3>
            <p className="text-gray-400">Performance metrics and quality analysis</p>
          </div>
        </Link>
      </div>
    </div>
  );
}