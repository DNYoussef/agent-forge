'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Sphere } from '@react-three/drei';
import { ArrowLeft, Brain, Sparkles, Network } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function ThinkingNodes() {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      <Sphere args={[0.5, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#06b6d4" emissive="#06b6d4" emissiveIntensity={0.5} />
      </Sphere>
      {[...Array(6)].map((_, i) => {
        const angle = (i / 6) * Math.PI * 2;
        return (
          <Sphere key={i} args={[0.3, 32, 32]} position={[Math.cos(angle) * 2, Math.sin(angle) * 2, 0]}>
            <meshStandardMaterial color="#67e8f9" />
          </Sphere>
        );
      })}
    </group>
  );
}

export default function QuietStarPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-cyan-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent flex items-center gap-4">
          <Sparkles className="w-12 h-12 text-cyan-400" />
          Phase 3: Quiet-STaR Reasoning
        </h1>
        <p className="text-xl text-gray-400">
          Chain-of-thought reasoning and self-taught rationales
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Network className="w-6 h-6 text-cyan-400" />
            Reasoning Network
          </h2>
          <Canvas camera={{ position: [0, 0, 6] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <ThinkingNodes />
            <OrbitControls />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Brain className="w-6 h-6 text-cyan-400" />
            Reasoning Metrics
          </h2>
          {[
            { label: 'Token Efficiency', value: 89, color: '#06b6d4' },
            { label: 'Rationale Quality', value: 94, color: '#22d3ee' },
            { label: 'Chain Coherence', value: 87, color: '#67e8f9' }
          ].map((metric, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-300">{metric.label}</span>
                <span className="text-lg font-bold" style={{ color: metric.color }}>{metric.value}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full"
                  style={{ width: metric.value + '%', backgroundColor: metric.color }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
