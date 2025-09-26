'use client';

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import { GitBranch, Layers, Activity, Zap } from 'lucide-react';

interface TransformersSquaredVisualizerProps {
  data?: any;
  activeConfiguration?: any;
}

export const TransformersSquaredVisualizer: React.FC<TransformersSquaredVisualizerProps> = ({
  data,
  activeConfiguration
}) => {
  // Generate expert vector positions for 3D visualization
  const expertVectorPositions = useMemo(() => {
    const positions: [number, number, number][] = [];
    const numVectors = activeConfiguration?.config_data?.t2_format?.expert_vectors?.length || 8;

    for (let i = 0; i < numVectors; i++) {
      const angle = (i / numVectors) * Math.PI * 2;
      const radius = 3;
      positions.push([
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        0
      ]);
    }
    return positions;
  }, [activeConfiguration]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-6 border border-blue-500">
        <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <GitBranch className="w-6 h-6" />
          Transformers² System Visualization
        </h2>
        <p className="text-gray-300">
          Two-pass architecture with SVD-based weight introspection and expert vector adaptation
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Left: Architecture Diagram */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" />
            Two-Pass Architecture
          </h3>

          <div className="space-y-4">
            {/* Pass 1: Task Dispatch */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="font-semibold text-purple-400 mb-2">Pass 1: Task Dispatch</div>
              <div className="text-sm text-gray-300 space-y-1">
                <div>• Input: Task description + context</div>
                <div>• Process: Transformer attention over task tokens</div>
                <div>• Output: Expert weight distribution</div>
              </div>
              <div className="mt-2 bg-black/50 rounded p-2 font-mono text-xs text-green-400">
                dispatch_weights = softmax(Q @ K^T / sqrt(d))
              </div>
            </div>

            {/* Pass 2: Expert Application */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="font-semibold text-blue-400 mb-2">Pass 2: Expert Application</div>
              <div className="text-sm text-gray-300 space-y-1">
                <div>• Input: Expert weights + model weights</div>
                <div>• Process: SVF (Singular Value Fine-tuning)</div>
                <div>• Output: Adapted model parameters</div>
              </div>
              <div className="mt-2 bg-black/50 rounded p-2 font-mono text-xs text-green-400">
                W' = U @ (S + ΔS_expert) @ V^T
              </div>
            </div>

            {/* SVD Components */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="font-semibold text-yellow-400 mb-2">SVD Decomposition</div>
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div className="text-center">
                  <div className="text-gray-400">U Matrix</div>
                  <div className="text-xl font-bold">Left Singular</div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400">S Matrix</div>
                  <div className="text-xl font-bold">Values</div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400">V^T Matrix</div>
                  <div className="text-xl font-bold">Right Singular</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: 3D Visualization */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-400" />
            Expert Vector Space
          </h3>

          <div className="h-96 bg-black rounded-lg">
            <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
              <ambientLight intensity={0.3} />
              <pointLight position={[10, 10, 10]} />

              <OrbitControls enablePan={false} />

              {/* Central Model Node */}
              <Sphere args={[0.8, 32, 32]} position={[0, 0, 0]}>
                <meshStandardMaterial color="#8b5cf6" emissive="#8b5cf6" emissiveIntensity={0.2} />
              </Sphere>
              <Text position={[0, -1.2, 0]} fontSize={0.3} color="white">
                Target Model
              </Text>

              {/* Expert Vector Nodes */}
              {expertVectorPositions.map((pos, idx) => {
                const expertData = activeConfiguration?.config_data?.t2_format?.expert_vectors?.[idx];
                const isActive = expertData != null;

                return (
                  <group key={idx}>
                    {/* Connection Line */}
                    <Line
                      points={[[0, 0, 0], pos]}
                      color={isActive ? "#60a5fa" : "#374151"}
                      lineWidth={isActive ? 2 : 1}
                      opacity={isActive ? 0.8 : 0.3}
                    />

                    {/* Expert Node */}
                    <Box args={[0.4, 0.4, 0.4]} position={pos}>
                      <meshStandardMaterial
                        color={isActive ? "#3b82f6" : "#4b5563"}
                        emissive={isActive ? "#3b82f6" : "#000000"}
                        emissiveIntensity={isActive ? 0.3 : 0}
                      />
                    </Box>

                    {/* Expert Label */}
                    <Text
                      position={[pos[0], pos[1] - 0.6, pos[2]]}
                      fontSize={0.2}
                      color={isActive ? "#60a5fa" : "#6b7280"}
                    >
                      {expertData?.specialization || `Expert ${idx + 1}`}
                    </Text>
                  </group>
                );
              })}
            </Canvas>
          </div>
        </div>
      </div>

      {/* Expert Vector Details */}
      {activeConfiguration?.config_data?.t2_format && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Active Expert Vectors
          </h3>

          <div className="grid grid-cols-3 gap-4">
            {activeConfiguration.config_data.t2_format.expert_vectors.map((vector: any, idx: number) => (
              <div key={idx} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-sm">{vector.specialization}</span>
                  <span className="text-xs text-gray-400">#{idx + 1}</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Layer:</span>
                    <span className="font-mono">{vector.layer}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">SVD Components:</span>
                    <span>{vector.svd_components?.length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Weight Norm:</span>
                    <span>{Math.sqrt(vector.weights?.reduce((a: number, b: number) => a + b * b, 0) || 0).toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System Metrics */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Effective Rank', value: data?.effective_rank || 'N/A', color: 'text-purple-400' },
          { label: 'Spectral Norm', value: data?.spectral_norm?.toFixed(3) || 'N/A', color: 'text-blue-400' },
          { label: 'Adaptation Strength', value: data?.adaptation_strength || '1.0', color: 'text-green-400' },
          { label: 'Expert Vectors', value: expertVectorPositions.length, color: 'text-yellow-400' }
        ].map((metric, idx) => (
          <div key={idx} className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <div className="text-sm text-gray-400">{metric.label}</div>
            <div className={`text-2xl font-bold ${metric.color}`}>{metric.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};