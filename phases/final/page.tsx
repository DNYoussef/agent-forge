'use client';

import { useState } from 'react';
import { ArrowLeft, Package, TrendingUp, Settings, Database, Sparkles } from 'lucide-react';
import Link from 'next/link';
import PhaseController from '@/components/shared/PhaseController';

interface FinalCompressionConfig {
  seedLMEnabled: boolean;
  seedLMBits: number;
  seedLMCandidates: number;
  vptqEnabled: boolean;
  vptqBits: number;
  vptqCodebookSize: number;
  hypercompressionEnabled: boolean;
  trajectorySteps: number;
  ergodicity: number;
  grokfastEnabled: boolean;
  grokfastAlpha: number;
  grokfastLambda: number;
  targetCompressionRatio: number;
}

export default function FinalCompressionPage() {
  const [config, setConfig] = useState<FinalCompressionConfig>({
    seedLMEnabled: true,
    seedLMBits: 4,
    seedLMCandidates: 16,
    vptqEnabled: true,
    vptqBits: 2,
    vptqCodebookSize: 256,
    hypercompressionEnabled: true,
    trajectorySteps: 100,
    ergodicity: 0.8,
    grokfastEnabled: true,
    grokfastAlpha: 0.98,
    grokfastLambda: 0.05,
    targetCompressionRatio: 32.0
  });

  const [metrics, setMetrics] = useState({
    overallProgress: 0,
    seedLMCompressionRatio: 0,
    vptqCompressionRatio: 0,
    hyperCompressionRatio: 0,
    totalCompressionRatio: 0,
    modelSizeMB: 0,
    performanceRetention: 100,
    perplexityDelta: 0
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-violet-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-violet-400 hover:text-violet-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent flex items-center gap-4">
          <Package className="w-12 h-12 text-violet-400" />
          Phase 8: Final Hypercompression
        </h1>
        <p className="text-xl text-gray-400">
          SeedLM + VPTQ + Ergodic trajectory compression stack
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Settings className="w-6 h-6 text-violet-400" />
            Configuration
          </h2>

          <div className="space-y-4">
            {/* SeedLM */}
            <div className="border-b border-white/10 pb-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.seedLMEnabled}
                  onChange={(e) => setConfig({...config, seedLMEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">SeedLM (Pseudo-Random Projection)</label>
              </div>

              {config.seedLMEnabled && (
                <>
                  <div className="mb-2">
                    <label className="text-xs text-gray-500 mb-1 block">
                      Bits per Weight: {config.seedLMBits}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="8"
                      value={config.seedLMBits}
                      onChange={(e) => setConfig({...config, seedLMBits: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">
                      Seed Candidates: {config.seedLMCandidates}
                    </label>
                    <input
                      type="range"
                      min="8"
                      max="32"
                      step="8"
                      value={config.seedLMCandidates}
                      onChange={(e) => setConfig({...config, seedLMCandidates: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </>
              )}
            </div>

            {/* VPTQ */}
            <div className="border-b border-white/10 pb-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.vptqEnabled}
                  onChange={(e) => setConfig({...config, vptqEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">VPTQ (Vector Post-Training Quantization)</label>
              </div>

              {config.vptqEnabled && (
                <>
                  <div className="mb-2">
                    <label className="text-xs text-gray-500 mb-1 block">
                      Quantization Bits: {config.vptqBits}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="4"
                      value={config.vptqBits}
                      onChange={(e) => setConfig({...config, vptqBits: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">
                      Codebook Size: {config.vptqCodebookSize}
                    </label>
                    <input
                      type="range"
                      min="128"
                      max="512"
                      step="128"
                      value={config.vptqCodebookSize}
                      onChange={(e) => setConfig({...config, vptqCodebookSize: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </>
              )}
            </div>

            {/* Hypercompression */}
            <div className="border-b border-white/10 pb-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={config.hypercompressionEnabled}
                  onChange={(e) => setConfig({...config, hypercompressionEnabled: e.target.checked})}
                  className="w-4 h-4"
                />
                <label className="text-sm text-gray-400">Hypercompression (Ergodic Trajectory)</label>
              </div>

              {config.hypercompressionEnabled && (
                <>
                  <div className="mb-2">
                    <label className="text-xs text-gray-500 mb-1 block">
                      Trajectory Steps: {config.trajectorySteps}
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="200"
                      step="50"
                      value={config.trajectorySteps}
                      onChange={(e) => setConfig({...config, trajectorySteps: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="text-xs text-gray-500 mb-1 block">
                      Ergodicity: {config.ergodicity.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.5"
                      max="0.95"
                      step="0.05"
                      value={config.ergodicity}
                      onChange={(e) => setConfig({...config, ergodicity: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </>
              )}
            </div>

            {/* Target Compression */}
            <div>
              <label className="text-sm text-gray-400 mb-1 block">
                Target Compression: {config.targetCompressionRatio.toFixed(1)}x
              </label>
              <input
                type="range"
                min="16"
                max="64"
                step="8"
                value={config.targetCompressionRatio}
                onChange={(e) => setConfig({...config, targetCompressionRatio: parseFloat(e.target.value)})}
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
                <label className="text-sm text-gray-400">Grokfast Optimization</label>
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
            phaseName="Final Compression"
            phaseId={8}
            apiEndpoint="/api/phases/final"
          />

          {/* Real-time Metrics */}
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-violet-400" />
              Compression Metrics
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Overall Progress</span>
                  <span className="text-violet-400">{metrics.overallProgress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-violet-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.overallProgress}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Total Compression Ratio</span>
                  <span className="text-fuchsia-400">{metrics.totalCompressionRatio.toFixed(1)}x</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-fuchsia-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(metrics.totalCompressionRatio / config.targetCompressionRatio) * 100}%` }}
                  />
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Target: {config.targetCompressionRatio.toFixed(1)}x
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Performance Retention</span>
                  <span className="text-green-400">{metrics.performanceRetention.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.performanceRetention}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Model Size</span>
                  <span className="text-blue-400">{metrics.modelSizeMB.toFixed(1)} MB</span>
                </div>
                <div className="text-xs text-gray-500">
                  Space saved: {((1 - 1/metrics.totalCompressionRatio) * 100).toFixed(1)}%
                </div>
              </div>

              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-1">
                  <span>Perplexity Delta</span>
                  <span className={metrics.perplexityDelta <= 0.1 ? 'text-green-400' : 'text-yellow-400'}>
                    +{metrics.perplexityDelta.toFixed(3)}
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  Lower is better (quality preservation)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Compression Stack Visualization */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Database className="w-6 h-6 text-violet-400" />
            Compression Stack
          </h2>

          <div className="space-y-4">
            {config.seedLMEnabled && (
              <div className="bg-violet-600/20 border border-violet-500/30 rounded-lg p-3">
                <div className="text-sm font-semibold text-violet-400 mb-1 flex items-center gap-1">
                  <Sparkles className="w-4 h-4" />
                  SeedLM Layer
                </div>
                <div className="text-xs text-gray-400 mb-1">
                  Pseudo-random projection compression
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Bits: {config.seedLMBits}</span>
                  <span className="text-violet-400">Ratio: {metrics.seedLMCompressionRatio.toFixed(1)}x</span>
                </div>
              </div>
            )}

            {config.vptqEnabled && (
              <div className="bg-fuchsia-600/20 border border-fuchsia-500/30 rounded-lg p-3">
                <div className="text-sm font-semibold text-fuchsia-400 mb-1 flex items-center gap-1">
                  <Database className="w-4 h-4" />
                  VPTQ Layer
                </div>
                <div className="text-xs text-gray-400 mb-1">
                  Vector quantization with learned codebook
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Codebook: {config.vptqCodebookSize}</span>
                  <span className="text-fuchsia-400">Ratio: {metrics.vptqCompressionRatio.toFixed(1)}x</span>
                </div>
              </div>
            )}

            {config.hypercompressionEnabled && (
              <div className="bg-purple-600/20 border border-purple-500/30 rounded-lg p-3">
                <div className="text-sm font-semibold text-purple-400 mb-1">
                  Hypercompression Layer
                </div>
                <div className="text-xs text-gray-400 mb-1">
                  Ergodic trajectory-based hyper-function
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Steps: {config.trajectorySteps}</span>
                  <span className="text-purple-400">Ratio: {metrics.hyperCompressionRatio.toFixed(1)}x</span>
                </div>
              </div>
            )}

            <div className="border-t border-white/10 pt-4">
              <div className="text-sm font-semibold text-gray-400 mb-2">Compression Pipeline</div>
              <div className="space-y-2">
                <div className="p-2 bg-violet-600/20 rounded-lg border border-violet-500/30">
                  <div className="text-xs text-violet-400">1. SeedLM: Pseudo-random projection</div>
                </div>
                <div className="p-2 bg-fuchsia-600/20 rounded-lg border border-fuchsia-500/30">
                  <div className="text-xs text-fuchsia-400">2. VPTQ: Vector quantization</div>
                </div>
                <div className="p-2 bg-purple-600/20 rounded-lg border border-purple-500/30">
                  <div className="text-xs text-purple-400">3. Hypercompression: Ergodic trajectory</div>
                </div>
                <div className="p-2 bg-green-600/20 rounded-lg border border-green-500/30">
                  <div className="text-xs text-green-400">4. Validation & metrics</div>
                </div>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4">
              <div className="text-center">
                <div className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent mb-2">
                  {metrics.totalCompressionRatio.toFixed(1)}x
                </div>
                <div className="text-sm text-gray-400">Combined Compression</div>
              </div>
            </div>

            <div className="bg-green-600/10 border border-green-500/30 rounded-lg p-3">
              <div className="text-xs text-green-400 font-semibold mb-1">Quality Preservation</div>
              <div className="text-xs text-gray-400">
                Perplexity delta: +{metrics.perplexityDelta.toFixed(3)}
              </div>
              <div className="text-xs text-gray-400">
                Performance: {metrics.performanceRetention.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}