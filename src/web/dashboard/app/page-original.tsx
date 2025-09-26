'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Activity, Cpu, Zap, Award } from 'lucide-react';

interface DashboardStats {
  totalAgents: number;
  successRate: number;
  activePipelines: number;
  avgPipelineTime: number;
}

export default function AgentForgeDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

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
    return () => clearInterval(interval);
  }, []);

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
          8-Phase AI Agent Creation Pipeline
        </p>
      </header>

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
              className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:border-white/30 transition-all duration-300 hover:scale-105 cursor-pointer"
              style={{ borderColor: phase.color + '40' }}
            >
              <div className="text-5xl mb-4">{phase.icon}</div>
              <div className="text-xl font-bold mb-2" style={{ color: phase.color }}>
                Phase {phase.id}
              </div>
              <div className="text-gray-400 mb-1">{phase.name}</div>
              <div className="text-sm text-gray-500">{phase.desc}</div>
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
