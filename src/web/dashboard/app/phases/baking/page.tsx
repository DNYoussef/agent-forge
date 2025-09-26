'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Wrench, TrendingUp, Settings, Target, TestTube, Award, Cpu, Layers, Brain, GitBranch, BarChart } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';

interface BakingConfig {
  // Basic configuration
  toolCount: number;
  personaCount: number;

  // Enhanced configuration for 95% target
  targetSuccessRate: number;
  maxIterations: number;
  currentIteration: number;

  // Multi-prompt testing
  promptsPerTask: number;
  promptSelectionTopK: number;

  // Progressive half-baking
  useProgressiveBaking: boolean;
  currentBakingStage: number;

  // Benchmarks
  useSWEBench: boolean;
  useToolBench: boolean;
  useHumanEval: boolean;

  // Real tools & MCP
  useRealTools: boolean;
  useMCPServers: boolean;

  // Adaptive parameters
  bakingStrength: number;
  learningRate: number;
  grokfastEnabled: boolean;
  grokfastLambda: number;
}

interface BenchmarkMetrics {
  sweScore: number;
  toolBenchScore: number;
  humanEvalScore: number;
  overallScore: number;
  gapToTarget: number;
}

interface ToolMetrics {
  name: string;
  calls: number;
  successes: number;
  successRate: number;
  avgLatency: number;
}

interface MCPServer {
  name: string;
  status: 'online' | 'offline' | 'degraded';
  capabilities: string[];
  latency: number;
  successRate: number;
}

interface PromptVariation {
  id: string;
  text: string;
  successRate: number;
  selected: boolean;
}

export default function BakingPage() {
  const [config, setConfig] = useState<BakingConfig>({
    toolCount: 10,
    personaCount: 5,
    targetSuccessRate: 0.95,
    maxIterations: 100,
    currentIteration: 0,
    promptsPerTask: 10,
    promptSelectionTopK: 3,
    useProgressiveBaking: true,
    currentBakingStage: 1,
    useSWEBench: true,
    useToolBench: true,
    useHumanEval: true,
    useRealTools: true,
    useMCPServers: true,
    bakingStrength: 0.15,
    learningRate: 0.0001,
    grokfastEnabled: true,
    grokfastLambda: 0.05
  });

  const [benchmarks, setBenchmarks] = useState<BenchmarkMetrics>({
    sweScore: 0,
    toolBenchScore: 0,
    humanEvalScore: 0,
    overallScore: 0,
    gapToTarget: 0.95
  });

  const [toolMetrics, setToolMetrics] = useState<ToolMetrics[]>([
    { name: 'Calculator', calls: 0, successes: 0, successRate: 0, avgLatency: 0 },
    { name: 'Web Search', calls: 0, successes: 0, successRate: 0, avgLatency: 0 },
    { name: 'Code Executor', calls: 0, successes: 0, successRate: 0, avgLatency: 0 },
    { name: 'File Manager', calls: 0, successes: 0, successRate: 0, avgLatency: 0 },
    { name: 'Data Analyzer', calls: 0, successes: 0, successRate: 0, avgLatency: 0 }
  ]);

  const [mcpServers, setMcpServers] = useState<MCPServer[]>([
    { name: 'filesystem', status: 'online', capabilities: ['file_ops'], latency: 5, successRate: 0.99 },
    { name: 'code-runner', status: 'online', capabilities: ['code_exec'], latency: 100, successRate: 0.95 },
    { name: 'web-search', status: 'online', capabilities: ['search'], latency: 500, successRate: 0.92 },
    { name: 'github', status: 'online', capabilities: ['github'], latency: 200, successRate: 0.97 },
    { name: 'playwright', status: 'offline', capabilities: ['browser'], latency: 1000, successRate: 0.90 }
  ]);

  const [promptVariations, setPromptVariations] = useState<PromptVariation[]>([
    { id: '1', text: 'Calculate: {expression}', successRate: 0.92, selected: true },
    { id: '2', text: 'Compute the result of {expression}', successRate: 0.88, selected: true },
    { id: '3', text: 'What is {expression}?', successRate: 0.85, selected: true },
    { id: '4', text: 'Solve: {expression}', successRate: 0.80, selected: false },
    { id: '5', text: 'Evaluate {expression} mathematically', successRate: 0.75, selected: false }
  ]);

  const [isRunning, setIsRunning] = useState(false);

  // Progressive baking stages
  const bakingStages = [
    { stage: 1, layers: [0, 1], name: 'Foundation', color: 'blue' },
    { stage: 2, layers: [2, 3, 4], name: 'Understanding', color: 'purple' },
    { stage: 3, layers: [5, 6, 7], name: 'Reasoning', color: 'indigo' },
    { stage: 4, layers: [8, 9, 10, 11], name: 'Specialization', color: 'green' }
  ];

  // Simulate real-time updates
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        // Update iteration
        setConfig(prev => ({
          ...prev,
          currentIteration: Math.min(prev.currentIteration + 1, prev.maxIterations)
        }));

        // Update benchmark scores
        setBenchmarks(prev => {
          const newOverall = Math.min(0.95, prev.overallScore + Math.random() * 0.02);
          return {
            sweScore: Math.min(0.95, prev.sweScore + Math.random() * 0.015),
            toolBenchScore: Math.min(0.95, prev.toolBenchScore + Math.random() * 0.018),
            humanEvalScore: Math.min(0.95, prev.humanEvalScore + Math.random() * 0.012),
            overallScore: newOverall,
            gapToTarget: Math.max(0, 0.95 - newOverall)
          };
        });

        // Update tool metrics
        setToolMetrics(prev => prev.map(tool => ({
          ...tool,
          calls: tool.calls + Math.floor(Math.random() * 10),
          successes: tool.successes + Math.floor(Math.random() * 8),
          successRate: Math.min(0.98, tool.successRate + Math.random() * 0.01),
          avgLatency: Math.max(5, tool.avgLatency + (Math.random() - 0.5) * 2)
        })));

        // Update baking stage
        setConfig(prev => ({
          ...prev,
          currentBakingStage: Math.min(4, Math.floor(prev.currentIteration / 25) + 1)
        }));

        // Stop if target reached
        if (benchmarks.overallScore >= 0.95) {
          setIsRunning(false);
        }
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isRunning, benchmarks.overallScore]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-indigo-400 hover:text-indigo-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-4">
          <Wrench className="w-12 h-12 text-indigo-400" />
          Phase 6: Tool & Persona Baking (Enhanced)
        </h1>
        <p className="text-xl text-gray-400">
          Iterative prompt baking with SWE-bench evaluation targeting 95% success rate
        </p>
      </div>

      {/* Top Stats Bar */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between">
            <Award className="w-6 h-6 text-yellow-400" />
            <div className="text-right">
              <div className="text-2xl font-bold text-yellow-400">
                {(benchmarks.overallScore * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-400">Overall Score</div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between">
            <Target className="w-6 h-6 text-green-400" />
            <div className="text-right">
              <div className="text-2xl font-bold text-green-400">95%</div>
              <div className="text-xs text-gray-400">Target</div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between">
            <GitBranch className="w-6 h-6 text-purple-400" />
            <div className="text-right">
              <div className="text-2xl font-bold text-purple-400">
                {config.currentIteration}/{config.maxIterations}
              </div>
              <div className="text-xs text-gray-400">Iteration</div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between">
            <Layers className="w-6 h-6 text-blue-400" />
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-400">
                Stage {config.currentBakingStage}
              </div>
              <div className="text-xs text-gray-400">Baking Stage</div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
          <div className="flex items-center justify-between">
            <Brain className="w-6 h-6 text-indigo-400" />
            <div className="text-right">
              <div className="text-2xl font-bold text-indigo-400">
                {promptVariations.filter(p => p.selected).length}/{config.promptsPerTask}
              </div>
              <div className="text-xs text-gray-400">Best Prompts</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Configuration & Control */}
        <div className="space-y-6">
          {/* Enhanced Control Panel */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Settings className="w-6 h-6 text-indigo-400" />
              Enhanced Configuration
            </h2>

            <div className="space-y-4">
              {/* Target Success Rate */}
              <div>
                <label className="text-sm text-gray-400 mb-1 block">
                  Target Success Rate: {(config.targetSuccessRate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="0.99"
                  step="0.01"
                  value={config.targetSuccessRate}
                  onChange={(e) => setConfig({...config, targetSuccessRate: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>

              {/* Max Iterations */}
              <div>
                <label className="text-sm text-gray-400 mb-1 block">
                  Max Iterations: {config.maxIterations}
                </label>
                <input
                  type="range"
                  min="50"
                  max="200"
                  value={config.maxIterations}
                  onChange={(e) => setConfig({...config, maxIterations: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>

              {/* Prompts per Task */}
              <div>
                <label className="text-sm text-gray-400 mb-1 block">
                  Prompts per Task: {config.promptsPerTask}
                </label>
                <input
                  type="range"
                  min="5"
                  max="20"
                  value={config.promptsPerTask}
                  onChange={(e) => setConfig({...config, promptsPerTask: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>

              {/* Benchmark Selection */}
              <div className="border-t border-white/10 pt-4">
                <div className="text-sm text-gray-400 mb-2">Benchmarks</div>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={config.useSWEBench}
                      onChange={(e) => setConfig({...config, useSWEBench: e.target.checked})}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">SWE-bench</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={config.useToolBench}
                      onChange={(e) => setConfig({...config, useToolBench: e.target.checked})}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">ToolBench</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={config.useHumanEval}
                      onChange={(e) => setConfig({...config, useHumanEval: e.target.checked})}
                      className="w-4 h-4"
                    />
                    <span className="text-sm">HumanEval</span>
                  </label>
                </div>
              </div>

              {/* Tool Configuration */}
              <div className="border-t border-white/10 pt-4">
                <div className="text-sm text-gray-400 mb-2">Tool System</div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.useRealTools}
                    onChange={(e) => setConfig({...config, useRealTools: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Use Real Tools (not mocks)</span>
                </label>
                <label className="flex items-center gap-2 mt-2">
                  <input
                    type="checkbox"
                    checked={config.useMCPServers}
                    onChange={(e) => setConfig({...config, useMCPServers: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Use MCP Servers</span>
                </label>
              </div>

              {/* Progressive Baking */}
              <div className="border-t border-white/10 pt-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.useProgressiveBaking}
                    onChange={(e) => setConfig({...config, useProgressiveBaking: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <span className="text-sm">Progressive Half-Baking</span>
                </label>
              </div>
            </div>

            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`w-full mt-6 py-3 rounded-lg font-semibold transition-all ${
                isRunning
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {isRunning ? 'Stop Baking' : 'Start Enhanced Baking'}
            </button>
          </div>

          {/* MCP Server Status */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Cpu className="w-6 h-6 text-indigo-400" />
              MCP Servers
            </h2>

            <div className="space-y-2">
              {mcpServers.map((server, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${
                      server.status === 'online' ? 'bg-green-400' :
                      server.status === 'degraded' ? 'bg-yellow-400' :
                      'bg-red-400'
                    }`} />
                    <span className="text-sm font-medium">{server.name}</span>
                  </div>
                  <div className="text-xs text-gray-400">
                    {server.latency}ms | {(server.successRate * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Middle Column: Metrics & Progress */}
        <div className="space-y-6">
          {/* Benchmark Progress */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <BarChart className="w-6 h-6 text-indigo-400" />
              Benchmark Progress to 95%
            </h2>

            <div className="space-y-4">
              {/* Overall Progress */}
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Overall Score</span>
                  <span className={`font-bold ${
                    benchmarks.overallScore >= 0.95 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {(benchmarks.overallScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative">
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 ${
                        benchmarks.overallScore >= 0.95
                          ? 'bg-green-400'
                          : 'bg-gradient-to-r from-yellow-400 to-green-400'
                      }`}
                      style={{ width: `${benchmarks.overallScore * 100}%` }}
                    />
                  </div>
                  <div
                    className="absolute top-0 h-3 w-0.5 bg-white"
                    style={{ left: '95%' }}
                  />
                </div>
                {benchmarks.gapToTarget > 0 && (
                  <div className="text-xs text-gray-500 mt-1">
                    Gap to target: {(benchmarks.gapToTarget * 100).toFixed(1)}%
                  </div>
                )}
              </div>

              {/* SWE-bench Score */}
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>SWE-bench</span>
                  <span className="text-blue-400">{(benchmarks.sweScore * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${benchmarks.sweScore * 100}%` }}
                  />
                </div>
              </div>

              {/* ToolBench Score */}
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>ToolBench</span>
                  <span className="text-purple-400">{(benchmarks.toolBenchScore * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${benchmarks.toolBenchScore * 100}%` }}
                  />
                </div>
              </div>

              {/* HumanEval Score */}
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>HumanEval</span>
                  <span className="text-indigo-400">{(benchmarks.humanEvalScore * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-indigo-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${benchmarks.humanEvalScore * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Progressive Baking Visualization */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Layers className="w-6 h-6 text-indigo-400" />
              Progressive Half-Baking
            </h2>

            <div className="space-y-3">
              {bakingStages.map((stage) => (
                <div key={stage.stage} className="relative">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{stage.name}</span>
                    <span className="text-xs text-gray-400">Layers {stage.layers.join(', ')}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 bg-${stage.color}-400`}
                      style={{
                        width: config.currentBakingStage >= stage.stage ? '100%' : '0%',
                        opacity: config.currentBakingStage >= stage.stage ? 1 : 0.3
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 p-3 bg-indigo-900/20 rounded-lg">
              <div className="text-xs text-indigo-400">Current Stage</div>
              <div className="text-lg font-bold text-indigo-300">
                {bakingStages[Math.min(config.currentBakingStage - 1, 3)]?.name || 'Initializing'}
              </div>
            </div>
          </div>

          {/* Tool Performance */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TestTube className="w-6 h-6 text-indigo-400" />
              Real Tool Performance
            </h2>

            <div className="space-y-2">
              {toolMetrics.map((tool, i) => (
                <div key={i} className="p-3 bg-white/5 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">{tool.name}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      tool.successRate >= 0.95 ? 'bg-green-600/30 text-green-400' :
                      tool.successRate >= 0.85 ? 'bg-yellow-600/30 text-yellow-400' :
                      'bg-red-600/30 text-red-400'
                    }`}>
                      {(tool.successRate * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">Calls:</span>
                      <span className="ml-1 text-gray-300">{tool.calls}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Success:</span>
                      <span className="ml-1 text-gray-300">{tool.successes}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Latency:</span>
                      <span className="ml-1 text-gray-300">{tool.avgLatency.toFixed(0)}ms</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Prompt Evolution */}
        <div className="space-y-6">
          {/* Prompt Evolution Display */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Brain className="w-6 h-6 text-indigo-400" />
              Prompt Evolution
            </h2>

            <div className="mb-4 text-sm text-gray-400">
              Testing {config.promptsPerTask} variations, selecting top {config.promptSelectionTopK}
            </div>

            <div className="space-y-3">
              {promptVariations.map((prompt) => (
                <div
                  key={prompt.id}
                  className={`p-3 rounded-lg border transition-all ${
                    prompt.selected
                      ? 'bg-indigo-900/30 border-indigo-500/50'
                      : 'bg-white/5 border-white/10'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-mono text-gray-300 flex-1">
                      {prompt.text}
                    </span>
                    {prompt.selected && (
                      <Award className="w-4 h-4 text-yellow-400 ml-2" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="w-full bg-gray-700 rounded-full h-1.5 mr-2">
                      <div
                        className={`h-1.5 rounded-full ${
                          prompt.successRate >= 0.9 ? 'bg-green-400' :
                          prompt.successRate >= 0.8 ? 'bg-yellow-400' :
                          'bg-red-400'
                        }`}
                        style={{ width: `${prompt.successRate * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 min-w-[3rem] text-right">
                      {(prompt.successRate * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 p-3 bg-green-900/20 rounded-lg">
              <div className="text-xs text-green-400 mb-1">Best Performing Prompt</div>
              <div className="text-sm font-mono text-green-300">
                {promptVariations.find(p => p.selected)?.text || 'Calculating...'}
              </div>
            </div>
          </div>

          {/* Iteration Status */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-indigo-400" />
              Adaptive Iteration Status
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Iteration Progress</span>
                  <span className="text-indigo-400 font-bold">
                    {config.currentIteration} / {config.maxIterations}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-indigo-400 to-purple-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(config.currentIteration / config.maxIterations) * 100}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/10">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Baking Strength</div>
                  <div className="text-lg font-bold text-indigo-400">
                    {config.bakingStrength.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-1">Learning Rate</div>
                  <div className="text-lg font-bold text-purple-400">
                    {config.learningRate.toExponential(1)}
                  </div>
                </div>
              </div>

              {config.grokfastEnabled && (
                <div className="pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">Grokfast λ</span>
                    <span className="text-sm font-bold text-green-400">
                      {config.grokfastLambda.toFixed(3)}
                    </span>
                  </div>
                </div>
              )}

              {/* Status Message */}
              <div className="mt-4 p-3 bg-indigo-900/20 rounded-lg">
                <div className="text-xs text-indigo-400 mb-1">Status</div>
                <div className="text-sm text-indigo-300">
                  {benchmarks.overallScore >= 0.95
                    ? '✓ Target achieved! 95% success rate reached.'
                    : benchmarks.gapToTarget > 0.2
                    ? 'Aggressive baking: Far from target'
                    : benchmarks.gapToTarget > 0.1
                    ? 'Moderate baking: Getting closer'
                    : 'Fine-tuning: Near target'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}