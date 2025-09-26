'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stats } from '@react-three/drei';
import { CompressionPipeline3D } from './components/CompressionPipeline3D';
import { CompressionMetrics3D } from './components/CompressionMetrics3D';
import { CompressionWebSocket } from './components/CompressionWebSocket';
import {
  Package,
  Cpu,
  Zap,
  Activity,
  Settings,
  Play,
  Pause,
  RotateCcw,
  Download,
  ChevronRight
} from 'lucide-react';

interface CompressionState {
  stage: 'idle' | 'seedlm' | 'vptq' | 'hypercompression' | 'complete';
  progress: number;
  metrics: {
    seedlmRatio: number;
    vptqRatio: number;
    hyperRatio: number;
    totalRatio: number;
    accuracy: number;
    speed: number;
  };
  config: {
    seedlm: {
      bitsPerWeight: number;
      maxCandidates: number;
      blockSize: number;
    };
    vptq: {
      bits: number;
      vectorDim: number;
      codebookSize: number;
      iterations: number;
    };
    hypercompression: {
      numClusters: number;
      trajectoryType: string;
      ergodicity: number;
      grokfastAlpha: number;
    };
  };
}

export default function CompressionDashboard() {
  const [state, setState] = useState<CompressionState>({
    stage: 'idle',
    progress: 0,
    metrics: {
      seedlmRatio: 1.0,
      vptqRatio: 1.0,
      hyperRatio: 1.0,
      totalRatio: 1.0,
      accuracy: 100,
      speed: 0
    },
    config: {
      seedlm: {
        bitsPerWeight: 4,
        maxCandidates: 16,
        blockSize: 8
      },
      vptq: {
        bits: 2,
        vectorDim: 4,
        codebookSize: 256,
        iterations: 10
      },
      hypercompression: {
        numClusters: 16,
        trajectoryType: 'auto',
        ergodicity: 0.8,
        grokfastAlpha: 0.98
      }
    }
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'compression_update') {
      setState(prev => ({
        ...prev,
        stage: data.stage,
        progress: data.progress,
        metrics: { ...prev.metrics, ...data.metrics }
      }));
    }
  };

  const startCompression = () => {
    setIsPlaying(true);
    // Send start command through WebSocket
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'start_compression',
        config: state.config
      }));
    }
  };

  const pauseCompression = () => {
    setIsPlaying(false);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'pause_compression' }));
    }
  };

  const resetCompression = () => {
    setIsPlaying(false);
    setState(prev => ({
      ...prev,
      stage: 'idle',
      progress: 0,
      metrics: {
        seedlmRatio: 1.0,
        vptqRatio: 1.0,
        hyperRatio: 1.0,
        totalRatio: 1.0,
        accuracy: 100,
        speed: 0
      }
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-blue-950 text-white">
      <CompressionWebSocket
        onMessage={handleWebSocketMessage}
        wsRef={wsRef}
      />

      {/* Header */}
      <div className="bg-black/30 backdrop-blur-lg border-b border-white/10 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Package className="w-8 h-8 text-purple-400" />
            <div>
              <h1 className="text-2xl font-bold">Phase 8: 3-Stage Compression Pipeline</h1>
              <p className="text-sm text-gray-400">SeedLM → VPTQ → Hypercompression</p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {!isPlaying ? (
              <button
                onClick={startCompression}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Start
              </button>
            ) : (
              <button
                onClick={pauseCompression}
                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg flex items-center gap-2 transition-colors"
              >
                <Pause className="w-4 h-4" />
                Pause
              </button>
            )}
            <button
              onClick={resetCompression}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
            <button
              onClick={() => setShowStats(!showStats)}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center gap-2 transition-colors"
            >
              <Activity className="w-4 h-4" />
              Stats
            </button>
          </div>
        </div>

        {/* Pipeline Progress */}
        <div className="mt-4 flex items-center gap-2">
          <div className={`flex-1 text-center p-2 rounded-lg transition-all ${
            state.stage === 'seedlm' ? 'bg-purple-600' : state.stage === 'idle' ? 'bg-gray-700' : 'bg-green-600'
          }`}>
            <div className="text-xs font-semibold">SeedLM</div>
            <div className="text-lg">{state.metrics.seedlmRatio.toFixed(2)}x</div>
          </div>
          <ChevronRight className="w-4 h-4 text-gray-500" />
          <div className={`flex-1 text-center p-2 rounded-lg transition-all ${
            state.stage === 'vptq' ? 'bg-purple-600' :
            ['seedlm', 'idle'].includes(state.stage) ? 'bg-gray-700' : 'bg-green-600'
          }`}>
            <div className="text-xs font-semibold">VPTQ</div>
            <div className="text-lg">{state.metrics.vptqRatio.toFixed(2)}x</div>
          </div>
          <ChevronRight className="w-4 h-4 text-gray-500" />
          <div className={`flex-1 text-center p-2 rounded-lg transition-all ${
            state.stage === 'hypercompression' ? 'bg-purple-600' :
            ['complete'].includes(state.stage) ? 'bg-green-600' : 'bg-gray-700'
          }`}>
            <div className="text-xs font-semibold">Hypercompression</div>
            <div className="text-lg">{state.metrics.hyperRatio.toFixed(2)}x</div>
          </div>
        </div>
      </div>

      {/* Main 3D Canvas */}
      <div className="relative h-[calc(100vh-200px)]">
        <Canvas shadows>
          <PerspectiveCamera makeDefault position={[0, 5, 15]} fov={60} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            maxPolarAngle={Math.PI * 0.9}
            minDistance={5}
            maxDistance={30}
          />

          {/* Lighting */}
          <ambientLight intensity={0.2} />
          <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
          <pointLight position={[-10, -10, -5]} intensity={0.5} color="#8b5cf6" />
          <pointLight position={[10, -10, 5]} intensity={0.5} color="#3b82f6" />

          {/* Fog for depth */}
          <fog attach="fog" args={['#0a0a1a', 10, 50]} />

          {/* Main Pipeline Visualization */}
          <CompressionPipeline3D
            stage={state.stage}
            progress={state.progress}
            config={state.config}
            isPlaying={isPlaying}
          />

          {/* 3D Metrics Display */}
          <CompressionMetrics3D
            metrics={state.metrics}
            position={[0, 8, 0]}
          />

          {/* Stats */}
          {showStats && <Stats />}
        </Canvas>

        {/* Stage Info Overlay */}
        <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-lg rounded-lg p-4 max-w-md">
          <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            Current Stage: {state.stage.charAt(0).toUpperCase() + state.stage.slice(1)}
          </h3>
          {state.stage === 'seedlm' && (
            <p className="text-sm text-gray-300">
              Applying pseudo-random projection compression using Linear Feedback Shift Register (LFSR)
              to generate reproducible projection matrices. Compressing to {state.config.seedlm.bitsPerWeight} bits per weight.
            </p>
          )}
          {state.stage === 'vptq' && (
            <p className="text-sm text-gray-300">
              Vector Post-Training Quantization in progress. Learning optimal codebook with {state.config.vptq.codebookSize} entries
              using K-means++ initialization. Quantizing to {state.config.vptq.bits} bits.
            </p>
          )}
          {state.stage === 'hypercompression' && (
            <p className="text-sm text-gray-300">
              Applying ergodic trajectory-based hypercompression. Fitting weight clusters to {state.config.hypercompression.trajectoryType}
              trajectories in phase space with Grokfast acceleration (α={state.config.hypercompression.grokfastAlpha}).
            </p>
          )}
        </div>

        {/* Metrics Overlay */}
        <div className="absolute bottom-4 right-4 bg-black/70 backdrop-blur-lg rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Compression Metrics
          </h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-8">
              <span className="text-gray-400">Total Compression:</span>
              <span className="text-green-400 font-mono">{state.metrics.totalRatio.toFixed(2)}x</span>
            </div>
            <div className="flex justify-between gap-8">
              <span className="text-gray-400">Accuracy Retained:</span>
              <span className="text-blue-400 font-mono">{state.metrics.accuracy.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between gap-8">
              <span className="text-gray-400">Processing Speed:</span>
              <span className="text-purple-400 font-mono">{state.metrics.speed.toFixed(1)} MB/s</span>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-black/30 backdrop-blur-lg border-t border-white/10 p-4">
        <div className="grid grid-cols-3 gap-4">
          {/* SeedLM Config */}
          <div className="bg-white/5 rounded-lg p-3">
            <h4 className="font-semibold mb-2 text-purple-400">SeedLM Configuration</h4>
            <div className="space-y-2 text-sm">
              <div>
                <label className="text-gray-400">Bits per Weight: {state.config.seedlm.bitsPerWeight}</label>
                <input
                  type="range"
                  min="2"
                  max="4"
                  value={state.config.seedlm.bitsPerWeight}
                  onChange={(e) => setState(prev => ({
                    ...prev,
                    config: {
                      ...prev.config,
                      seedlm: { ...prev.config.seedlm, bitsPerWeight: parseInt(e.target.value) }
                    }
                  }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* VPTQ Config */}
          <div className="bg-white/5 rounded-lg p-3">
            <h4 className="font-semibold mb-2 text-green-400">VPTQ Configuration</h4>
            <div className="space-y-2 text-sm">
              <div>
                <label className="text-gray-400">Quantization Bits: {state.config.vptq.bits}</label>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={state.config.vptq.bits}
                  onChange={(e) => setState(prev => ({
                    ...prev,
                    config: {
                      ...prev.config,
                      vptq: { ...prev.config.vptq, bits: parseInt(e.target.value) }
                    }
                  }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* Hypercompression Config */}
          <div className="bg-white/5 rounded-lg p-3">
            <h4 className="font-semibold mb-2 text-blue-400">Hypercompression Configuration</h4>
            <div className="space-y-2 text-sm">
              <div>
                <label className="text-gray-400">Clusters: {state.config.hypercompression.numClusters}</label>
                <input
                  type="range"
                  min="8"
                  max="32"
                  value={state.config.hypercompression.numClusters}
                  onChange={(e) => setState(prev => ({
                    ...prev,
                    config: {
                      ...prev.config,
                      hypercompression: { ...prev.config.hypercompression, numClusters: parseInt(e.target.value) }
                    }
                  }))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}