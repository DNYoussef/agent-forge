'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Play, Square } from 'lucide-react';
import Link from 'next/link';
import SimpleOrb from '@/components/phases/SimpleOrb';

interface BitNetMetrics {
  compressionProgress: number;
  compressionRatio: number;
  performanceRetention: number;
  currentPhase: 'initializing' | 'calibration' | 'quantization' | 'fine_tuning' | 'completed';
  quantizedLayers: number;
  totalLayers: number;
  modelSizeMB: number;
}

export default function SimpleBitNetPage() {
  const [metrics, setMetrics] = useState<BitNetMetrics>({
    compressionProgress: 0,
    compressionRatio: 1.0,
    performanceRetention: 100,
    currentPhase: 'initializing',
    quantizedLayers: 0,
    totalLayers: 64,
    modelSizeMB: 6400
  });
  const [sessionId, setSessionId] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);

  // Fetch metrics from API
  const fetchMetrics = async () => {
    try {
      const response = await fetch(`/api/phases/bitnet?sessionId=${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  // Start BitNet compression
  const startCompression = async () => {
    try {
      const response = await fetch('/api/phases/bitnet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start' })
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.sessionId);
        setIsRunning(true);
      }
    } catch (error) {
      console.error('Failed to start compression:', error);
    }
  };

  // Stop compression
  const stopCompression = () => {
    setIsRunning(false);
  };

  // Fetch metrics every second when running
  useEffect(() => {
    if (isRunning && sessionId) {
      const interval = setInterval(fetchMetrics, 1000);
      return () => clearInterval(interval);
    }
  }, [isRunning, sessionId]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <Link href="/" className="text-blue-400 hover:text-blue-300">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <h1 className="text-2xl font-bold">BitNet 1.58-bit Compression</h1>
        </div>

        <div className="flex space-x-3">
          {!isRunning ? (
            <button
              onClick={startCompression}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center space-x-2"
            >
              <Play className="w-4 h-4" />
              <span>Start Compression</span>
            </button>
          ) : (
            <button
              onClick={stopCompression}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center space-x-2"
            >
              <Square className="w-4 h-4" />
              <span>Stop</span>
            </button>
          )}
        </div>
      </div>

      <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 3D Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">3D Model Visualization</h2>
            <SimpleOrb
              compressionProgress={metrics.compressionProgress}
              compressionRatio={metrics.compressionRatio}
              currentPhase={metrics.currentPhase}
            />
            <div className="mt-4 text-center text-gray-300">
              <p className="text-sm">
                Phase: <span className="font-semibold capitalize">{metrics.currentPhase}</span>
              </p>
              <p className="text-sm">
                Progress: {metrics.compressionProgress.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Panel */}
        <div className="space-y-6">
          {/* Compression Stats */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Compression Metrics</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{metrics.compressionProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${metrics.compressionProgress}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-300">Compression Ratio:</span>
                <span className="font-semibold">{metrics.compressionRatio.toFixed(1)}x</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-300">Performance:</span>
                <span className="font-semibold">{metrics.performanceRetention.toFixed(1)}%</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-300">Model Size:</span>
                <span className="font-semibold">{(metrics.modelSizeMB / 1000).toFixed(1)}GB</span>
              </div>
            </div>
          </div>

          {/* Layer Progress */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Layer Progress</h3>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">
                {metrics.quantizedLayers}/{metrics.totalLayers}
              </div>
              <div className="text-gray-300 text-sm">Layers Quantized</div>
            </div>
          </div>

          {/* Weight Distribution Legend */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Weight Values</h3>
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                <span className="text-sm">-1 values</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 bg-gray-500 rounded-full"></div>
                <span className="text-sm">0 values</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                <span className="text-sm">+1 values</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}