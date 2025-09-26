'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Brain, Sparkles, TrendingUp, Settings, Activity } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';
import ThoughtGeneratorVisualization from '@/components/phases/quietstar/ThoughtGeneratorVisualization';
import CoherenceScoreBreakdown from '@/components/phases/quietstar/CoherenceScoreBreakdown';
import PromptBakingProgress from '@/components/phases/quietstar/PromptBakingProgress';
import ThoughtInspector from '@/components/phases/quietstar/ThoughtInspector';
import AttentionWeightVisualizer from '@/components/phases/quietstar/AttentionWeightVisualizer';

interface QuietStarConfig {
  convergenceThreshold: number;
  maxIterations: number;
  cognitiveStrategy: string;
  grokfastEnabled: boolean;
  grokfastAlpha: number;
  grokfastLambda: number;
  learningRate: number;
  batchSize: number;
  thoughtMixingEnabled: boolean;
  edgeOfChaosEnabled: boolean;
}

export default function QuietStarPage() {
  const [config, setConfig] = useState<QuietStarConfig>({
    convergenceThreshold: 0.95,
    maxIterations: 100,
    cognitiveStrategy: 'systems_thinking',
    grokfastEnabled: true,
    grokfastAlpha: 0.98,
    grokfastLambda: 0.05,
    learningRate: 0.0001,
    batchSize: 32,
    thoughtMixingEnabled: true,
    edgeOfChaosEnabled: true
  });

  const [metrics, setMetrics] = useState({
    iteration: 0,
    convergenceScore: 0,
    bakingProgress: 0,
    reasoningDepth: 0,
    thoughtCoherence: 0
  });

  const [activePhase, setActivePhase] = useState(false);
  const [thoughts, setThoughts] = useState([]);
  const [bakingComplete, setBakingComplete] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [injectionData, setInjectionData] = useState(null);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (activePhase) {
      const ws = new WebSocket('ws://localhost:8000/ws/quietstar');

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'thought_generated') {
          setThoughts(prev => [...prev, data.thought]);
        } else if (data.type === 'metrics_update') {
          setMetrics(prev => ({ ...prev, ...data.metrics }));
        } else if (data.type === 'injection_data') {
          setInjectionData(data.injection);
        }
      };

      return () => ws.close();
    }
  }, [activePhase]);

  const cognitiveStrategies = [
    { id: 'systems_thinking', name: 'Systems Thinking', desc: 'Holistic problem analysis' },
    { id: 'first_principles', name: 'First Principles', desc: 'Fundamental reasoning' },
    { id: 'lateral_thinking', name: 'Lateral Thinking', desc: 'Creative problem solving' },
    { id: 'analogical', name: 'Analogical', desc: 'Pattern-based reasoning' },
    { id: 'inductive', name: 'Inductive', desc: 'Specific to general' },
    { id: 'deductive', name: 'Deductive', desc: 'General to specific' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-4">
          <Sparkles className="w-12 h-12 text-purple-400" />
          Phase 3: Quiet-STaR Baking
        </h1>
        <p className="text-xl text-gray-400">
          Iterative prompt baking with reasoning enhancement
        </p>
      </div>

      {/* Prompt Baking Progress */}
      <div className="mb-8">
        <PromptBakingProgress
          isActive={activePhase}
          totalStages={5}
          onBakingComplete={() => setBakingComplete(true)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Settings className="w-6 h-6 text-purple-400" />
            Configuration
          </h2>

          <div className="space-y-4">
            {/* Cognitive Strategy */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Cognitive Strategy</label>
              <select
                value={config.cognitiveStrategy}
                onChange={(e) => setConfig({...config, cognitiveStrategy: e.target.value})}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white"
              >
                {cognitiveStrategies.map(s => (
                  <option key={s.id} value={s.id}>{s.name} - {s.desc}</option>
                ))}
              </select>
            </div>

            {/* Convergence Threshold */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Convergence Threshold: {config.convergenceThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.8"
                max="0.99"
                step="0.01"
                value={config.convergenceThreshold}
                onChange={(e) => setConfig({...config, convergenceThreshold: parseFloat(e.target.value)})}
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

            {/* Learning Rate */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Learning Rate</label>
              <input
                type="number"
                value={config.learningRate}
                onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                step="0.00001"
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white"
              />
            </div>

            {/* Batch Size */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Batch Size</label>
              <select
                value={config.batchSize}
                onChange={(e) => setConfig({...config, batchSize: parseInt(e.target.value)})}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white"
              >
                <option value="16">16</option>
                <option value="32">32</option>
                <option value="64">64</option>
                <option value="128">128</option>
              </select>
            </div>

            {/* Feature Toggles */}
            <div className="border-t border-white/10 pt-4 space-y-2">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.thoughtMixingEnabled}
                  onChange={(e) => setConfig({...config, thoughtMixingEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">ThoughtMixingHead Enhancement</label>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.edgeOfChaosEnabled}
                  onChange={(e) => setConfig({...config, edgeOfChaosEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Edge-of-Chaos Training</label>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.grokfastEnabled}
                  onChange={(e) => setConfig({...config, grokfastEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Grokfast 50x Acceleration</label>
              </div>
            </div>

            {/* Grokfast Settings */}
            {config.grokfastEnabled && (
              <div className="border-t border-white/10 pt-4">
                <div className="mb-2">
                  <label className="text-xs text-gray-500 mb-1 block">
                    EMA Alpha: {config.grokfastAlpha}
                  </label>
                  <input
                    type="range"
                    min="0.9"
                    max="0.99"
                    step="0.01"
                    value={config.grokfastAlpha}
                    onChange={(e) => setConfig({...config, grokfastAlpha: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-xs text-gray-500 mb-1 block">
                    Lambda: {config.grokfastLambda}
                  </label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.25"
                    step="0.01"
                    value={config.grokfastLambda}
                    onChange={(e) => setConfig({...config, grokfastLambda: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Metrics & Visualization */}
        <div className="space-y-6">
          {/* Control Panel */}
          <PhaseController
            phaseName="Quiet-STaR"
            phaseId={3}
            apiEndpoint="/api/phases/quietstar"
            onStart={() => setActivePhase(true)}
            onStop={() => setActivePhase(false)}
          />

          {/* Thought Generator Visualization */}
          <ThoughtGeneratorVisualization
            isActive={activePhase}
            config={{
              maxThoughts: config.maxIterations,
              temperature: 0.8,
              coherenceThreshold: config.convergenceThreshold,
              thoughtLength: config.batchSize
            }}
            onThoughtGenerated={(thought) => setThoughts(prev => [...prev, thought])}
          />

          {/* Real-time Metrics */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-400" />
              Baking Metrics
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Iteration Progress</span>
                  <span className="text-blue-400">{metrics.iteration} / {config.maxIterations}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(metrics.iteration / config.maxIterations) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Convergence Score</span>
                  <span className="text-green-400">{metrics.convergenceScore.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.convergenceScore * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Thoughts Generated</span>
                  <span className="text-purple-400">{thoughts.length}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, (thoughts.length / 50) * 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Reasoning Depth</span>
                  <span className="text-yellow-400">{metrics.reasoningDepth.toFixed(1)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, metrics.reasoningDepth * 10)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Attention Weight Visualizer */}
        <AttentionWeightVisualizer
          isActive={activePhase}
          injectionData={injectionData}
          onVisualizationUpdate={(data) => console.log('Visualization update:', data)}
        />
      </div>

      {/* Thought Inspector */}
      {thoughts.length > 0 && (
        <div className="mt-8">
          <ThoughtInspector
            thoughts={thoughts}
            coherenceThreshold={config.convergenceThreshold}
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
          />
        </div>
      )}

      {/* Coherence Analysis */}
      {bakingComplete && thoughts.length > 0 && (
        <div className="mt-8">
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Brain className="w-6 h-6 text-green-400" />
              Coherence Analysis Summary
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {thoughts.slice(-4).map((thought, index) => (
                <CoherenceScoreBreakdown
                  key={`summary-${index}`}
                  scores={thought.coherence_scores}
                  threshold={config.convergenceThreshold}
                  thoughtId={thoughts.length - 4 + index}
                  isExpanded={false}
                  onToggleExpanded={() => {}}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}