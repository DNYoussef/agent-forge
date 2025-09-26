'use client';

import { useState } from 'react';
import { ArrowLeft, Network, TrendingUp, Settings, Layers, Zap } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';

interface ADASConfig {
  populationSize: number;
  generations: number;
  vectorCompositionEnabled: boolean;
  compositionScale: number;
  nsgaIIEnabled: boolean;
  objectives: string[];
  grokfastEnabled: boolean;
  grokfastAlpha: number;
  grokfastLambda: number;
  mutationRate: number;
  crossoverRate: number;
  techniquePoolSize: number;
}

export default function ADASPage() {
  const [config, setConfig] = useState<ADASConfig>({
    populationSize: 12,
    generations: 30,
    vectorCompositionEnabled: true,
    compositionScale: 0.1,
    nsgaIIEnabled: true,
    objectives: ['accuracy', 'efficiency', 'size'],
    grokfastEnabled: true,
    grokfastAlpha: 0.98,
    grokfastLambda: 0.05,
    mutationRate: 0.15,
    crossoverRate: 0.7,
    techniquePoolSize: 8
  });

  const [metrics, setMetrics] = useState({
    currentGeneration: 0,
    paretoFrontSize: 0,
    bestAccuracy: 0,
    bestEfficiency: 0,
    architectureDiversity: 1.0,
    convergenceScore: 0
  });

  const availableObjectives = [
    { id: 'accuracy', name: 'Accuracy', desc: 'Model performance' },
    { id: 'efficiency', name: 'Efficiency', desc: 'Computational cost' },
    { id: 'size', name: 'Size', desc: 'Model parameters' },
    { id: 'latency', name: 'Latency', desc: 'Inference speed' },
    { id: 'memory', name: 'Memory', desc: 'Memory usage' }
  ];

  const vectorOperations = [
    'Linear Interpolation', 'Spherical Interpolation',
    'Attention Blending', 'Layer Composition',
    'Parameter Averaging', 'Gradient Mixing'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-teal-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-teal-400 hover:text-teal-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-4">
          <Network className="w-12 h-12 text-teal-400" />
          Phase 7: ADAS Architecture Search
        </h1>
        <p className="text-xl text-gray-400">
          Vector composition with multi-objective optimization
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Settings className="w-6 h-6 text-teal-400" />
            Configuration
          </h2>

          <div className="space-y-4">
            {/* Population & Generations */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Population Size: {config.populationSize}
              </label>
              <input
                type="range"
                min="8"
                max="20"
                value={config.populationSize}
                onChange={(e) => setConfig({...config, populationSize: parseInt(e.target.value)})}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Generations: {config.generations}
              </label>
              <input
                type="range"
                min="20"
                max="50"
                value={config.generations}
                onChange={(e) => setConfig({...config, generations: parseInt(e.target.value)})}
                className="w-full"
              />
            </div>

            {/* Vector Composition */}
            <div className="border-t border-white/10 pt-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.vectorCompositionEnabled}
                  onChange={(e) => setConfig({...config, vectorCompositionEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Vector Composition (Transformers Squared)</label>
              </div>

              {config.vectorCompositionEnabled && (
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">
                    Composition Scale: {config.compositionScale.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.3"
                    step="0.05"
                    value={config.compositionScale}
                    onChange={(e) => setConfig({...config, compositionScale: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-600 mt-1">
                    Operations: {vectorOperations.slice(0, 3).join(', ')}
                  </div>
                </div>
              )}
            </div>

            {/* NSGA-II Multi-Objective */}
            <div className="border-t border-white/10 pt-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.nsgaIIEnabled}
                  onChange={(e) => setConfig({...config, nsgaIIEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">NSGA-II Multi-Objective Optimization</label>
              </div>

              {config.nsgaIIEnabled && (
                <div className="space-y-2 mt-2">
                  <div className="text-xs text-gray-500">Optimization Objectives:</div>
                  {availableObjectives.map(obj => (
                    <label key={obj.id} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={config.objectives.includes(obj.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setConfig({...config, objectives: [...config.objectives, obj.id]});
                          } else {
                            setConfig({...config, objectives: config.objectives.filter(o => o !== obj.id)});
                          }
                        }}
                        className="mr-2"
                      />
                      <span className="text-xs">
                        {obj.name}
                        <span className="text-gray-600 ml-1">({obj.desc})</span>
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Genetic Parameters */}
            <div className="border-t border-white/10 pt-4">
              <div className="mb-2">
                <label className="text-sm text-gray-400 mb-1 block">
                  Mutation Rate: {(config.mutationRate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.3"
                  step="0.05"
                  value={config.mutationRate}
                  onChange={(e) => setConfig({...config, mutationRate: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="text-sm text-gray-400 mb-1 block">
                  Crossover Rate: {(config.crossoverRate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="0.9"
                  step="0.1"
                  value={config.crossoverRate}
                  onChange={(e) => setConfig({...config, crossoverRate: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>
            </div>

            {/* Technique Pool */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Technique Pool Size: {config.techniquePoolSize}
              </label>
              <input
                type="range"
                min="4"
                max="12"
                value={config.techniquePoolSize}
                onChange={(e) => setConfig({...config, techniquePoolSize: parseInt(e.target.value)})}
                className="w-full"
              />
            </div>

            {/* Grokfast */}
            <div className="border-t border-white/10 pt-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.grokfastEnabled}
                  onChange={(e) => setConfig({...config, grokfastEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Grokfast Acceleration</label>
              </div>

              {config.grokfastEnabled && (
                <>
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
                </>
              )}
            </div>
          </div>
        </div>

        {/* Metrics Panel */}
        <div className="space-y-6">
          {/* Control Panel */}
          <PhaseController
            phaseName="ADAS"
            phaseId={7}
            apiEndpoint="/api/phases/adas"
          />

          {/* Real-time Metrics */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-teal-400" />
              Search Metrics
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Generation Progress</span>
                  <span className="text-teal-400">{metrics.currentGeneration} / {config.generations}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-teal-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(metrics.currentGeneration / config.generations) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Pareto Front Size</span>
                  <span className="text-cyan-400">{metrics.paretoFrontSize}</span>
                </div>
                <div className="text-xs text-gray-500">Optimal architectures found</div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Best Accuracy</span>
                  <span className="text-green-400">{metrics.bestAccuracy.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, metrics.bestAccuracy * 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Best Efficiency</span>
                  <span className="text-blue-400">{metrics.bestEfficiency.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, metrics.bestEfficiency * 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Architecture Diversity</span>
                  <span className="text-purple-400">{metrics.architectureDiversity.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-purple-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.architectureDiversity * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Convergence Score</span>
                  <span className="text-yellow-400">{metrics.convergenceScore.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.convergenceScore}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Architecture Visualization */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Layers className="w-6 h-6 text-teal-400" />
            Architecture Search
          </h2>

          <div className="space-y-4">
            {config.vectorCompositionEnabled && (
              <div className="bg-teal-600/20 border border-teal-500/30 rounded-lg p-3">
                <div className="text-sm font-semibold text-teal-400 mb-2 flex items-center gap-1">
                  <Zap className="w-4 h-4" />
                  Vector Composition Active
                </div>
                <div className="text-xs text-gray-400">
                  Scale: {config.compositionScale.toFixed(2)}
                </div>
                <div className="mt-2 space-y-1">
                  {vectorOperations.slice(0, config.techniquePoolSize).map((op, i) => (
                    <div key={i} className="text-xs text-teal-300">• {op}</div>
                  ))}
                </div>
              </div>
            )}

            {config.nsgaIIEnabled && (
              <div className="bg-cyan-600/20 border border-cyan-500/30 rounded-lg p-3">
                <div className="text-sm font-semibold text-cyan-400 mb-2">
                  NSGA-II Optimization
                </div>
                <div className="text-xs text-gray-400 mb-2">
                  Active Objectives: {config.objectives.length}
                </div>
                <div className="space-y-1">
                  {config.objectives.map(obj => (
                    <div key={obj} className="text-xs text-cyan-300">
                      • {availableObjectives.find(o => o.id === obj)?.name}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="border-t border-white/10 pt-4">
              <div className="text-sm font-semibold text-gray-400 mb-2">Search Pipeline</div>
              <div className="space-y-2">
                <div className="p-2 bg-teal-600/20 rounded-lg border border-teal-500/30">
                  <div className="text-xs text-teal-400">1. Architecture Generation</div>
                </div>
                <div className="p-2 bg-blue-600/20 rounded-lg border border-blue-500/30">
                  <div className="text-xs text-blue-400">2. Vector Composition</div>
                </div>
                <div className="p-2 bg-cyan-600/20 rounded-lg border border-cyan-500/30">
                  <div className="text-xs text-cyan-400">3. Multi-Objective Evaluation</div>
                </div>
                <div className="p-2 bg-purple-600/20 rounded-lg border border-purple-500/30">
                  <div className="text-xs text-purple-400">4. Pareto Front Selection</div>
                </div>
                <div className="p-2 bg-green-600/20 rounded-lg border border-green-500/30">
                  <div className="text-xs text-green-400">5. Architecture Training</div>
                </div>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4">
              <div className="text-sm font-semibold text-gray-400 mb-2">Genetic Parameters</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-orange-600/20 rounded-lg p-2 border border-orange-500/30">
                  <div className="text-xs text-gray-400">Mutation</div>
                  <div className="text-sm font-bold text-orange-400">
                    {(config.mutationRate * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="bg-pink-600/20 rounded-lg p-2 border border-pink-500/30">
                  <div className="text-xs text-gray-400">Crossover</div>
                  <div className="text-sm font-bold text-pink-400">
                    {(config.crossoverRate * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-teal-600/10 border border-teal-500/30 rounded-lg p-3">
              <div className="text-xs text-teal-400 font-semibold mb-1">Sandboxed Execution</div>
              <div className="text-xs text-gray-400">
                Secure technique evaluation with timeout protection
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}