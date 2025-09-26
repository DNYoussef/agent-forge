'use client';

import { useState } from 'react';
import { ArrowLeft, Zap, TrendingUp, Settings, Binary, Activity } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';

interface BitNetConfig {
  quantizationBits: number;
  preserveCriticalLayers: boolean;
  criticalLayerThreshold: number;
  compressionRatio: number;
  grokfastEnabled: boolean;
  grokfastAlpha: number;
  grokfastLambda: number;
  learningRate: number;
  fineTuneEpochs: number;
  memoryOptimizationLevel: string;
}

export default function BitNetPage() {
  const [config, setConfig] = useState<BitNetConfig>({
    quantizationBits: 1.58,
    preserveCriticalLayers: true,
    criticalLayerThreshold: 0.8,
    compressionRatio: 8.0,
    grokfastEnabled: true,
    grokfastAlpha: 0.98,
    grokfastLambda: 0.05,
    learningRate: 0.0001,
    fineTuneEpochs: 5,
    memoryOptimizationLevel: 'aggressive'
  });

  const [metrics, setMetrics] = useState({
    compressionProgress: 0,
    memoryReduction: 0,
    performanceRetention: 100,
    quantizedLayers: 0,
    totalLayers: 0,
    modelSizeMB: 0
  });

  const optimizationLevels = [
    { id: 'conservative', name: 'Conservative', desc: 'Preserve all precision' },
    { id: 'balanced', name: 'Balanced', desc: 'Balance compression & quality' },
    { id: 'aggressive', name: 'Aggressive', desc: 'Maximum compression' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-4">
          <Binary className="w-12 h-12 text-blue-400" />
          Phase 4: BitNet 1.58-bit Compression
        </h1>
        <p className="text-xl text-gray-400">
          Memory-efficient quantization with {-1, 0, +1} weights
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Settings className="w-6 h-6 text-blue-400" />
            Configuration
          </h2>

          <div className="space-y-4">
            {/* Quantization Bits */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Quantization: {config.quantizationBits.toFixed(2)} bits
              </label>
              <div className="text-xs text-gray-500 mb-2">
                BitNet 1.58: Three values {-1, 0, +1}
              </div>
            </div>

            {/* Critical Layer Preservation */}
            <div className="border-t border-white/10 pt-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.preserveCriticalLayers}
                  onChange={(e) => setConfig({...config, preserveCriticalLayers: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Preserve Critical Layers</label>
              </div>

              {config.preserveCriticalLayers && (
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">
                    Sensitivity Threshold: {config.criticalLayerThreshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    value={config.criticalLayerThreshold}
                    onChange={(e) => setConfig({...config, criticalLayerThreshold: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
              )}
            </div>

            {/* Compression Ratio */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Target Compression: {config.compressionRatio.toFixed(1)}x
              </label>
              <input
                type="range"
                min="4"
                max="16"
                step="0.5"
                value={config.compressionRatio}
                onChange={(e) => setConfig({...config, compressionRatio: parseFloat(e.target.value)})}
                className="w-full"
              />
            </div>

            {/* Memory Optimization */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Memory Optimization</label>
              <select
                value={config.memoryOptimizationLevel}
                onChange={(e) => setConfig({...config, memoryOptimizationLevel: e.target.value})}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white"
              >
                {optimizationLevels.map(level => (
                  <option key={level.id} value={level.id}>
                    {level.name} - {level.desc}
                  </option>
                ))}
              </select>
            </div>

            {/* Fine-tuning */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Fine-tune Epochs: {config.fineTuneEpochs}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={config.fineTuneEpochs}
                onChange={(e) => setConfig({...config, fineTuneEpochs: parseInt(e.target.value)})}
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
            phaseName="BitNet"
            phaseId={4}
            apiEndpoint="/api/phases/bitnet"
          />

          {/* Real-time Metrics */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-cyan-400" />
              Compression Metrics
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Compression Progress</span>
                  <span className="text-blue-400">{metrics.compressionProgress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.compressionProgress}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Memory Reduction</span>
                  <span className="text-green-400">{metrics.memoryReduction.toFixed(1)}x</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(metrics.memoryReduction / config.compressionRatio) * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Performance Retention</span>
                  <span className="text-yellow-400">{metrics.performanceRetention.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-yellow-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.performanceRetention}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Quantized Layers</span>
                  <span className="text-cyan-400">{metrics.quantizedLayers} / {metrics.totalLayers}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-cyan-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.totalLayers > 0 ? (metrics.quantizedLayers / metrics.totalLayers) * 100 : 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Weight Distribution Visualization */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-400" />
            Weight Distribution
          </h2>

          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">1.58-bit</div>
              <div className="text-sm text-gray-400">BitNet Quantization</div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="bg-red-600/20 rounded-lg p-3 border border-red-500/30 text-center">
                <div className="text-2xl font-bold text-red-400">-1</div>
                <div className="text-xs text-gray-400">Negative</div>
              </div>

              <div className="bg-gray-600/20 rounded-lg p-3 border border-gray-500/30 text-center">
                <div className="text-2xl font-bold text-gray-400">0</div>
                <div className="text-xs text-gray-400">Zero</div>
              </div>

              <div className="bg-green-600/20 rounded-lg p-3 border border-green-500/30 text-center">
                <div className="text-2xl font-bold text-green-400">+1</div>
                <div className="text-xs text-gray-400">Positive</div>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4">
              <div className="text-sm text-gray-400 mb-2">Model Size</div>
              <div className="text-2xl font-bold text-cyan-400">
                {metrics.modelSizeMB.toFixed(1)} MB
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs text-gray-500">
                <span>Original Size</span>
                <span>{(metrics.modelSizeMB * config.compressionRatio).toFixed(1)} MB</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>Compression Ratio</span>
                <span>{config.compressionRatio.toFixed(1)}x</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>Space Saved</span>
                <span>{((1 - 1/config.compressionRatio) * 100).toFixed(1)}%</span>
              </div>
            </div>

            {config.preserveCriticalLayers && (
              <div className="bg-yellow-600/10 border border-yellow-500/30 rounded-lg p-3">
                <div className="text-xs text-yellow-400 font-semibold mb-1">Critical Layers Protected</div>
                <div className="text-xs text-gray-400">
                  High-sensitivity layers preserved at full precision
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}