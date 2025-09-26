/**
 * AttentionWeightVisualizer Component
 *
 * Visualizes the attention weights and thought injection process from the
 * Quiet-STaR ThoughtInjectionSystem with multi-head attention fusion.
 */

'use client';

import { useState, useEffect } from 'react';
import { Network, Layers, Eye, Settings, Play, Pause } from 'lucide-react';

interface AttentionWeight {
  from_position: number;
  to_position: number;
  weight: number;
  head: number;
  layer: string;
}

interface InjectionData {
  base_sequence_length: number;
  thought_count: number;
  attention_heads: number;
  injection_points: number[];
  attention_weights: AttentionWeight[];
  fusion_weights: number[];
  gate_weights: number[];
}

interface Props {
  isActive: boolean;
  injectionData: InjectionData | null;
  onVisualizationUpdate: (data: any) => void;
}

export default function AttentionWeightVisualizer({
  isActive,
  injectionData,
  onVisualizationUpdate
}: Props) {
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState<string>('thought_fusion');
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [showGating, setShowGating] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Generate mock injection data if none provided
  const mockInjectionData: InjectionData = {
    base_sequence_length: 16,
    thought_count: 4,
    attention_heads: 8,
    injection_points: [4, 8, 12],
    attention_weights: [],
    fusion_weights: [0.85, 0.76, 0.92, 0.68],
    gate_weights: [0.72, 0.89, 0.54, 0.91, 0.67, 0.83, 0.95, 0.71, 0.78, 0.86, 0.59, 0.94, 0.82, 0.77, 0.88, 0.63]
  };

  const data = injectionData || mockInjectionData;

  // Generate attention weights if not provided
  useEffect(() => {
    if (data.attention_weights.length === 0) {
      const weights: AttentionWeight[] = [];
      for (let head = 0; head < data.attention_heads; head++) {
        for (let from = 0; from < data.base_sequence_length; from++) {
          for (let to = 0; to < data.thought_count; to++) {
            weights.push({
              from_position: from,
              to_position: to,
              weight: Math.random() * 0.6 + 0.2, // 0.2 to 0.8
              head,
              layer: 'thought_fusion'
            });
          }
        }
      }
      data.attention_weights = weights;
    }
  }, [data]);

  // Animation loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnimating && isActive) {
      interval = setInterval(() => {
        setCurrentStep(prev => (prev + 1) % 100);
      }, 1000 / animationSpeed);
    }
    return () => clearInterval(interval);
  }, [isAnimating, isActive, animationSpeed]);

  const getAttentionWeightsForHead = (head: number, layer: string) => {
    return data.attention_weights.filter(w => w.head === head && w.layer === layer);
  };

  const getWeightColor = (weight: number): string => {
    if (weight > 0.7) return 'bg-green-400';
    if (weight > 0.5) return 'bg-yellow-400';
    if (weight > 0.3) return 'bg-orange-400';
    return 'bg-red-400';
  };

  const getWeightOpacity = (weight: number): string => {
    return `opacity-${Math.floor(weight * 100)}`;
  };

  const renderAttentionMatrix = () => {
    const weights = getAttentionWeightsForHead(selectedHead, selectedLayer);
    const matrix: number[][] = Array(data.base_sequence_length).fill(0).map(() => Array(data.thought_count).fill(0));

    weights.forEach(w => {
      if (w.from_position < data.base_sequence_length && w.to_position < data.thought_count) {
        matrix[w.from_position][w.to_position] = w.weight;
      }
    });

    return (
      <div className="bg-black/40 p-4 rounded-lg">
        <div className="text-sm font-semibold text-gray-300 mb-3">
          Attention Matrix: Head {selectedHead + 1} → {selectedLayer}
        </div>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${data.thought_count}, 1fr)` }}>
          {/* Headers */}
          {Array.from({ length: data.thought_count }, (_, i) => (
            <div key={`header-${i}`} className="text-xs text-center text-gray-400 p-1">
              T{i + 1}
            </div>
          ))}

          {/* Matrix cells */}
          {matrix.map((row, rowIndex) =>
            row.map((weight, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`w-8 h-8 rounded flex items-center justify-center text-xs font-semibold transition-all duration-300 ${getWeightColor(weight)}`}
                style={{ opacity: weight }}
                title={`Pos ${rowIndex} → Thought ${colIndex}: ${weight.toFixed(3)}`}
              >
                {weight > 0.5 ? weight.toFixed(1).replace('0.', '') : ''}
              </div>
            ))
          )}
        </div>

        {/* Position labels */}
        <div className="mt-2 flex justify-between text-xs text-gray-400">
          <span>Position 0</span>
          <span>Position {data.base_sequence_length - 1}</span>
        </div>
      </div>
    );
  };

  const renderFusionProcess = () => {
    return (
      <div className="bg-black/40 p-4 rounded-lg">
        <div className="text-sm font-semibold text-gray-300 mb-3">
          Multi-Head Attention Fusion
        </div>

        <div className="space-y-3">
          {/* Base Input */}
          <div className="flex items-center gap-2">
            <div className="w-16 text-xs text-gray-400">Base Input</div>
            <div className="flex gap-1">
              {Array.from({ length: data.base_sequence_length }, (_, i) => (
                <div
                  key={`base-${i}`}
                  className={`w-6 h-6 rounded ${
                    data.injection_points.includes(i) ? 'bg-blue-500' : 'bg-gray-600'
                  } flex items-center justify-center text-xs`}
                >
                  {i}
                </div>
              ))}
            </div>
          </div>

          {/* Thought Representations */}
          <div className="flex items-center gap-2">
            <div className="w-16 text-xs text-gray-400">Thoughts</div>
            <div className="flex gap-1">
              {Array.from({ length: data.thought_count }, (_, i) => (
                <div
                  key={`thought-${i}`}
                  className="w-8 h-8 rounded bg-purple-500 flex items-center justify-center text-xs font-semibold"
                  style={{ opacity: data.fusion_weights[i] }}
                >
                  T{i + 1}
                </div>
              ))}
            </div>
          </div>

          {/* Fusion Arrow */}
          <div className="text-center text-gray-400 text-xl">↓</div>

          {/* Fused Output */}
          <div className="flex items-center gap-2">
            <div className="w-16 text-xs text-gray-400">Fused</div>
            <div className="flex gap-1">
              {data.gate_weights.map((gateWeight, i) => (
                <div
                  key={`fused-${i}`}
                  className="w-6 h-6 rounded bg-green-500 flex items-center justify-center text-xs"
                  style={{ opacity: gateWeight }}
                  title={`Gate weight: ${gateWeight.toFixed(3)}`}
                >
                  {i < data.base_sequence_length ? i : ''}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderGatingMechanism = () => {
    if (!showGating) return null;

    return (
      <div className="bg-black/40 p-4 rounded-lg">
        <div className="text-sm font-semibold text-gray-300 mb-3">
          Gating Mechanism (σ(Linear([base; fused])))
        </div>

        <div className="space-y-2">
          {data.gate_weights.map((weight, i) => (
            <div key={`gate-${i}`} className="flex items-center gap-3">
              <div className="w-12 text-xs text-gray-400">Pos {i}</div>
              <div className="flex-1 bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-purple-400 to-pink-400 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${weight * 100}%` }}
                />
              </div>
              <div className="w-16 text-xs text-right">
                <span className={weight > 0.7 ? 'text-green-400' : weight > 0.4 ? 'text-yellow-400' : 'text-red-400'}>
                  {weight.toFixed(3)}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-3 text-xs text-gray-500">
          Gate weight determines how much of the fused thought vs base representation to use
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-bold flex items-center gap-2">
          <Network className="w-6 h-6 text-blue-400" />
          Attention Weight Visualization
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsAnimating(!isAnimating)}
            className={`px-3 py-1 rounded-lg text-sm ${
              isAnimating ? 'bg-green-600/20 text-green-400' : 'bg-white/10 text-gray-400'
            }`}
          >
            {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="text-sm text-gray-400 mb-1 block">Attention Head</label>
          <select
            value={selectedHead}
            onChange={(e) => setSelectedHead(parseInt(e.target.value))}
            className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm"
          >
            {Array.from({ length: data.attention_heads }, (_, i) => (
              <option key={i} value={i}>Head {i + 1}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-sm text-gray-400 mb-1 block">Layer</label>
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(e.target.value)}
            className="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm"
          >
            <option value="thought_fusion">Thought Fusion</option>
            <option value="attention_mixer">Attention Mixer</option>
          </select>
        </div>

        <div>
          <label className="text-sm text-gray-400 mb-1 block">Animation Speed</label>
          <input
            type="range"
            min="0.1"
            max="3.0"
            step="0.1"
            value={animationSpeed}
            onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-gray-400 mt-1">{animationSpeed.toFixed(1)}x</div>
        </div>

        <div className="flex items-end gap-2">
          <button
            onClick={() => setShowGating(!showGating)}
            className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm ${
              showGating ? 'bg-purple-600/20 text-purple-400' : 'bg-white/10 text-gray-400'
            }`}
          >
            <Eye className="w-4 h-4" />
            Gating
          </button>
        </div>
      </div>

      {/* Visualization Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Attention Matrix */}
        <div>
          {renderAttentionMatrix()}
        </div>

        {/* Fusion Process */}
        <div>
          {renderFusionProcess()}
        </div>
      </div>

      {/* Gating Mechanism */}
      {showGating && (
        <div className="mt-6">
          {renderGatingMechanism()}
        </div>
      )}

      {/* Statistics */}
      <div className="mt-6 p-4 bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-lg border border-blue-500/30">
        <div className="text-sm font-semibold text-blue-400 mb-2">Injection Statistics</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div>
            <div className="text-gray-400">Attention Heads:</div>
            <div className="text-blue-400 font-semibold">{data.attention_heads}</div>
          </div>
          <div>
            <div className="text-gray-400">Injection Points:</div>
            <div className="text-green-400 font-semibold">{data.injection_points.length}</div>
          </div>
          <div>
            <div className="text-gray-400">Avg Gate Weight:</div>
            <div className="text-purple-400 font-semibold">
              {(data.gate_weights.reduce((a, b) => a + b, 0) / data.gate_weights.length).toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-gray-400">Avg Attention:</div>
            <div className="text-yellow-400 font-semibold">
              {data.attention_weights.length > 0 ?
                (data.attention_weights.reduce((a, b) => a + b.weight, 0) / data.attention_weights.length).toFixed(3) :
                '0.000'
              }
            </div>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-white/10 text-xs text-gray-400">
          <span className="font-semibold text-blue-400">Multi-head Attention:</span> Thoughts serve as keys/values,
          base sequence as queries. <span className="font-semibold text-purple-400">Gating:</span> Controls
          thought vs base representation mixing at each position.
        </div>
      </div>
    </div>
  );
}