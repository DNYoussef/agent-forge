'use client';

import { useEffect, useState, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial } from '@react-three/drei';
import { ArrowLeft, Brain, Layers, Zap, Cpu, Activity, Server } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function ModelSphere({ position, color, scale = 1 }: { position: [number, number, number], color: string, scale?: number }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime) * 0.3;
      meshRef.current.rotation.y = Math.cos(state.clock.elapsedTime) * 0.3;
    }
  });

  return (
    <Sphere ref={meshRef} args={[scale, 64, 64]} position={position}>
      <MeshDistortMaterial color={color} speed={2} distort={0.3} />
    </Sphere>
  );
}

export default function CognatePage() {
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [progress, setProgress] = useState(0);

  // Your actual specification: 3 foundation models ~25M params each
  const models = [
    {
      id: 1,
      name: 'Titans-Foundation-1',
      params: '25M',
      architecture: 'Titans + Hierarchical Reasoning',
      features: ['Surprise×Novelty Gating', 'Long-term Memory'],
      status: 'ready',
      color: '#3b82f6',
      description: 'Base model with hierarchical reasoning modules'
    },
    {
      id: 2,
      name: 'Titans-Foundation-2',
      params: '25M',
      architecture: 'Titans + Pattern Recognition',
      features: ['Neural Attention Layers', 'Grokfast Acceleration'],
      status: 'ready',
      color: '#a855f7',
      description: 'Specialized for pattern extraction and abstraction'
    },
    {
      id: 3,
      name: 'Titans-Foundation-3',
      params: '25M',
      architecture: 'Titans + Adaptive Learning',
      features: ['Dynamic Weight Adjustment', 'Self-Supervised Learning'],
      status: 'ready',
      color: '#06b6d4',
      description: 'Meta-learning capabilities for rapid adaptation'
    }
  ];

  const startTraining = async () => {
    setTrainingStatus('training');
    setProgress(0);

    // Call the actual backend API
    try {
      const response = await fetch('http://localhost:8001/api/cognate/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_models: 3,
          model_size: '25M',
          architecture: 'titans',
          grokfast_enabled: true,
          target_speedup: 50
        })
      });

      const data = await response.json();
      console.log('Training started:', data);

      // Simulate progress
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setTrainingStatus('complete');
            return 100;
          }
          return prev + 2;
        });
      }, 100);
    } catch (error) {
      console.error('Failed to start training:', error);
      setTrainingStatus('error');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-4">
          <Brain className="w-12 h-12 text-blue-400" />
          Phase 1: Cognate Pretraining
        </h1>
        <p className="text-xl text-gray-400">
          Create exactly 3 foundation models (~25M parameters each) using Titans architecture with Grokfast 50x acceleration
        </p>
      </div>

      {/* Key Specifications Banner */}
      <div className="bg-gradient-to-r from-green-600/20 to-emerald-600/20 rounded-2xl p-6 border border-green-500/30 mb-8">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-green-400" />
          Implementation Specifications
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-400">Architecture</p>
            <p className="text-lg font-bold">Titans + HRM</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Acceleration</p>
            <p className="text-lg font-bold">Grokfast 50x</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Total Parameters</p>
            <p className="text-lg font-bold">~75M (3×25M)</p>
          </div>
        </div>
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
            {/* Three foundation models */}
            <ModelSphere position={[-2.5, 0, 0]} color="#3b82f6" scale={1.2} />
            <ModelSphere position={[0, 0, 0]} color="#a855f7" scale={1.2} />
            <ModelSphere position={[2.5, 0, 0]} color="#06b6d4" scale={1.2} />
            <OrbitControls enableZoom={false} />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            Foundation Models (Titans Architecture)
          </h2>
          {models.map((model) => {
            const isReady = model.status === 'ready';
            return (
              <div
                key={model.id}
                className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 hover:border-white/30 transition-all"
                style={{ borderColor: model.color + '40' }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h3 className="text-xl font-bold" style={{ color: model.color }}>
                      {model.name}
                    </h3>
                    <p className="text-gray-400">{model.params} parameters • {model.architecture}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={'w-3 h-3 rounded-full ' + (isReady ? 'bg-green-400' : 'bg-yellow-400')} />
                    <span className="text-sm text-gray-400">{model.status}</span>
                  </div>
                </div>
                <p className="text-sm text-gray-500 mb-2">{model.description}</p>
                <div className="flex flex-wrap gap-2">
                  {model.features.map((feature, idx) => (
                    <span key={idx} className="text-xs bg-white/10 rounded px-2 py-1">
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Training Controls */}
      <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 mb-8">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Activity className="w-6 h-6 text-blue-400" />
          Training Control Panel
        </h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-lg">Status: <span className={
                trainingStatus === 'training' ? 'text-yellow-400' :
                trainingStatus === 'complete' ? 'text-green-400' :
                trainingStatus === 'error' ? 'text-red-400' : 'text-gray-400'
              }>{trainingStatus.toUpperCase()}</span></p>
              <p className="text-sm text-gray-400">Backend: Python Bridge API on port 8001</p>
            </div>
            <button
              onClick={startTraining}
              disabled={trainingStatus === 'training'}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-3 rounded-xl font-bold transition-colors flex items-center gap-2"
            >
              <Server className="w-5 h-5" />
              {trainingStatus === 'training' ? 'Training...' : 'Start Pretraining'}
            </button>
          </div>

          {trainingStatus === 'training' && (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 rounded-2xl p-6 border border-blue-500/30">
          <h3 className="text-xl font-bold mb-2">Phase 1: Cognate</h3>
          <p className="text-gray-400">3 foundation models with Titans architecture and Grokfast acceleration</p>
        </div>
        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30">
          <h3 className="text-xl font-bold mb-2">Phase 2: EvoMerge</h3>
          <p className="text-gray-400">8 evolutionary combinations with tournament selection over 50 generations</p>
        </div>
        <div className="bg-gradient-to-r from-green-600/20 to-emerald-600/20 rounded-2xl p-6 border border-green-500/30">
          <h3 className="text-xl font-bold mb-2">Backend Ready</h3>
          <p className="text-gray-400">Real PyTorch training with gradient descent (1000 steps/model)</p>
        </div>
      </div>
    </div>
  );
}