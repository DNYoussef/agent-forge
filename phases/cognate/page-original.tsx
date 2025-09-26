'use client';

import { useEffect, useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial } from '@react-three/drei';
import { ArrowLeft, Brain, Layers, Zap } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function ModelSphere({ position, color }: { position: [number, number, number], color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime) * 0.3;
      meshRef.current.rotation.y = Math.cos(state.clock.elapsedTime) * 0.3;
    }
  });

  return (
    <Sphere ref={meshRef} args={[1, 64, 64]} position={position}>
      <MeshDistortMaterial color={color} speed={2} distort={0.3} />
    </Sphere>
  );
}

export default function CognatePage() {
  const models = [
    { id: 1, name: 'GPT-4', params: '1.8T', status: 'ready', color: '#3b82f6' },
    { id: 2, name: 'Claude-3', params: '1.2T', status: 'ready', color: '#a855f7' },
    { id: 3, name: 'Gemini', params: '540B', status: 'ready', color: '#06b6d4' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-4">
          <Brain className="w-12 h-12 text-blue-400" />
          Phase 1: Cognate Model Creator
        </h1>
        <p className="text-xl text-gray-400">
          Select and combine base models for agent creation
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Layers className="w-6 h-6 text-blue-400" />
            3D Model Visualization
          </h2>
          <Canvas camera={{ position: [0, 0, 8] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <ModelSphere position={[-2, 0, 0]} color="#3b82f6" />
            <ModelSphere position={[2, 0, 0]} color="#a855f7" />
            <ModelSphere position={[0, 2, 0]} color="#06b6d4" />
            <OrbitControls enableZoom={false} />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            Available Base Models
          </h2>
          {models.map((model) => {
            const isReady = model.status === 'ready';
            return (
              <div
                key={model.id}
                className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 hover:border-white/30 transition-all cursor-pointer"
                style={{ borderColor: model.color + '40' }}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold" style={{ color: model.color }}>
                      {model.name}
                    </h3>
                    <p className="text-gray-400">{model.params} parameters</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={'w-3 h-3 rounded-full ' + (isReady ? 'bg-green-400' : 'bg-yellow-400')} />
                    <span className="text-sm text-gray-400">{model.status}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 rounded-2xl p-6 border border-blue-500/30">
          <h3 className="text-xl font-bold mb-2">Fusion Mode</h3>
          <p className="text-gray-400">Combine multiple models with weighted averaging</p>
        </div>
        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30">
          <h3 className="text-xl font-bold mb-2">Fine-tune</h3>
          <p className="text-gray-400">Customize model behavior with domain data</p>
        </div>
        <div className="bg-gradient-to-r from-green-600/20 to-emerald-600/20 rounded-2xl p-6 border border-green-500/30">
          <h3 className="text-xl font-bold mb-2">Deploy</h3>
          <p className="text-gray-400">Export to EvoMerge for evolution</p>
        </div>
      </div>
    </div>
  );
}
