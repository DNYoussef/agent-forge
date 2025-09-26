'use client';

import { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import { ArrowLeft, Dna, GitBranch, TrendingUp } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function EvolutionParticles() {
  const particlesRef = useRef<THREE.Points>(null);
  const particleCount = 1000;
  
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 10;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
  }

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.05} color="#a855f7" />
    </points>
  );
}

export default function EvoMergePage() {
  const generations = [
    { id: 1, gen: 'Gen 1', fitness: 0.45, models: 100, color: '#a855f7' },
    { id: 2, gen: 'Gen 5', fitness: 0.67, models: 100, color: '#c084fc' },
    { id: 3, gen: 'Gen 10', fitness: 0.82, models: 100, color: '#d8b4fe' },
    { id: 4, gen: 'Gen 15', fitness: 0.93, models: 50, color: '#e9d5ff' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent flex items-center gap-4">
          <Dna className="w-12 h-12 text-purple-400" />
          Phase 2: EvoMerge Evolution
        </h1>
        <p className="text-xl text-gray-400">
          Genetic algorithm-based model evolution and merging
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-purple-400" />
            Evolution Visualization
          </h2>
          <Canvas camera={{ position: [0, 0, 5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <EvolutionParticles />
            <OrbitControls enableZoom={true} />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-green-400" />
            Generation Progress
          </h2>
          {generations.map((gen) => (
            <div
              key={gen.id}
              className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10"
              style={{ borderColor: gen.color + '40' }}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xl font-bold" style={{ color: gen.color }}>
                  {gen.gen}
                </h3>
                <span className="text-sm text-gray-400">{gen.models} models</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3 mb-2">
                <div
                  className="h-3 rounded-full transition-all"
                  style={{ 
                    width: gen.fitness * 100 + '%',
                    background: 'linear-gradient(to right, ' + gen.color + ', #ec4899)'
                  }}
                />
              </div>
              <p className="text-sm text-gray-400">Fitness: {(gen.fitness * 100).toFixed(0)}%</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30">
          <h3 className="text-xl font-bold mb-2">Crossover</h3>
          <p className="text-gray-400">Combine traits from top-performing models</p>
        </div>
        <div className="bg-gradient-to-r from-pink-600/20 to-rose-600/20 rounded-2xl p-6 border border-pink-500/30">
          <h3 className="text-xl font-bold mb-2">Mutation</h3>
          <p className="text-gray-400">Introduce random variations for diversity</p>
        </div>
        <div className="bg-gradient-to-r from-violet-600/20 to-purple-600/20 rounded-2xl p-6 border border-violet-500/30">
          <h3 className="text-xl font-bold mb-2">Selection</h3>
          <p className="text-gray-400">Keep best performers for next generation</p>
        </div>
      </div>
    </div>
  );
}
