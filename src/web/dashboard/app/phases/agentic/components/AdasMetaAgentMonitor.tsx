'use client';

import React, { useMemo } from 'react';
import { Brain, GitMerge, Shuffle, TrendingUp, Archive, Cpu, Zap, Target } from 'lucide-react';

interface AdasMetaAgentMonitorProps {
  searchData?: any;
  discoverySession?: any;
  configurationsArchive?: any[];
}

export const AdasMetaAgentMonitor: React.FC<AdasMetaAgentMonitorProps> = ({
  searchData,
  discoverySession,
  configurationsArchive
}) => {
  // Calculate discovery strategies distribution
  const strategyDistribution = useMemo(() => {
    if (!discoverySession?.discovery_results?.strategy_usage) {
      return {
        random: 25,
        mutation: 25,
        crossover: 25,
        task_specialized: 25
      };
    }
    return discoverySession.discovery_results.strategy_usage;
  }, [discoverySession]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900 to-pink-900 rounded-lg p-6 border border-purple-500">
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Brain className="w-6 h-6" />
          ADAS Meta-Agent Search Monitor
        </h2>
        <p className="text-gray-300">
          Progressive agent invention discovering optimal expert vector configurations
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Search Progress */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Discovery Progress
          </h3>

          {discoverySession?.discovery_results ? (
            <div className="space-y-4">
              {/* Iterations */}
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-gray-400">Iterations Completed</span>
                  <span className="font-bold">
                    {discoverySession.discovery_results.iterations_completed || 0} / {discoverySession.discovery_results.total_iterations || 30}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all"
                    style={{
                      width: `${((discoverySession.discovery_results.iterations_completed || 0) / (discoverySession.discovery_results.total_iterations || 30)) * 100}%`
                    }}
                  />
                </div>
              </div>

              {/* Best Performance */}
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">Best Performance Score</div>
                <div className="text-2xl font-bold text-green-400">
                  {discoverySession.discovery_results.best_performance?.toFixed(4) || '0.0000'}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Found in iteration #{discoverySession.discovery_results.best_iteration || 0}
                </div>
              </div>

              {/* Configurations Tested */}
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Total Tested</div>
                  <div className="text-lg font-bold">{discoverySession.discovery_results.total_tested || 0}</div>
                </div>
                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Valid Found</div>
                  <div className="text-lg font-bold text-blue-400">
                    {discoverySession.discovery_results.valid_configurations || 0}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No active discovery in progress
            </div>
          )}
        </div>

        {/* Discovery Strategies */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <GitMerge className="w-5 h-5 text-purple-400" />
            Discovery Strategies
          </h3>

          <div className="space-y-3">
            {/* Strategy Distribution */}
            {Object.entries(strategyDistribution).map(([strategy, percentage]: [string, any]) => (
              <div key={strategy}>
                <div className="flex justify-between mb-1">
                  <span className="text-sm capitalize">{strategy.replace('_', ' ')}</span>
                  <span className="text-sm text-gray-400">{percentage}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      strategy === 'random' ? 'bg-blue-500' :
                      strategy === 'mutation' ? 'bg-green-500' :
                      strategy === 'crossover' ? 'bg-yellow-500' :
                      'bg-purple-500'
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            ))}

            {/* Strategy Icons */}
            <div className="grid grid-cols-2 gap-2 mt-4">
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <Shuffle className="w-5 h-5 mx-auto mb-1 text-blue-400" />
                <div className="text-xs">Random</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <Zap className="w-5 h-5 mx-auto mb-1 text-green-400" />
                <div className="text-xs">Mutation</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <GitMerge className="w-5 h-5 mx-auto mb-1 text-yellow-400" />
                <div className="text-xs">Crossover</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <Target className="w-5 h-5 mx-auto mb-1 text-purple-400" />
                <div className="text-xs">Specialized</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Archive */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Archive className="w-5 h-5 text-yellow-400" />
          Expert Configuration Archive
        </h3>

        {configurationsArchive && configurationsArchive.length > 0 ? (
          <div className="grid grid-cols-3 gap-3">
            {configurationsArchive.slice(0, 9).map((config: any, idx: number) => (
              <div key={idx} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold">{config.task_specialization}</span>
                  <span className="text-xs text-gray-400">Gen {config.generation || 0}</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Score:</span>
                    <span className="text-green-400">{config.performance_score?.toFixed(3) || '0.000'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">SVD Layers:</span>
                    <span>{Object.keys(config.svd_components || {}).length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Strategy:</span>
                    <span className="text-purple-400">{config.discovery_strategy || 'unknown'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            Archive is empty. Run discovery to populate configurations.
          </div>
        )}
      </div>

      {/* Meta-Agent Intelligence */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-blue-400" />
          Meta-Agent Intelligence
        </h3>

        <div className="grid grid-cols-4 gap-4">
          {/* Code Generation */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Code Generation</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">Active</div>
              <div className="text-xs text-gray-500 mt-1">Python/PyTorch</div>
            </div>
          </div>

          {/* Progressive Learning */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Progressive Learning</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">Enabled</div>
              <div className="text-xs text-gray-500 mt-1">Iterative Improvement</div>
            </div>
          </div>

          {/* Archive Memory */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Archive Memory</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {configurationsArchive?.length || 0}
              </div>
              <div className="text-xs text-gray-500 mt-1">Configurations</div>
            </div>
          </div>

          {/* Search Budget */}
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Search Budget</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {discoverySession?.discovery_results?.budget_remaining || 50}
              </div>
              <div className="text-xs text-gray-500 mt-1">Evaluations</div>
            </div>
          </div>
        </div>
      </div>

      {/* How ADAS Works */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">ADAS Algorithm Process</h3>

        <div className="space-y-3 text-sm">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold">1</div>
            <div>
              <div className="font-semibold">Initialize Archive</div>
              <div className="text-gray-400">Start with random expert vector configurations</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold">2</div>
            <div>
              <div className="font-semibold">Generate New Configurations</div>
              <div className="text-gray-400">Use mutation, crossover, and task-specialized generation</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold">3</div>
            <div>
              <div className="font-semibold">Evaluate Performance</div>
              <div className="text-gray-400">Test configurations on task-specific benchmarks</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold">4</div>
            <div>
              <div className="font-semibold">Update Archive</div>
              <div className="text-gray-400">Keep best performing configurations for next iteration</div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold">5</div>
            <div>
              <div className="font-semibold">Progressive Invention</div>
              <div className="text-gray-400">Learn from patterns to create better configurations</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};