'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Torus } from '@react-three/drei';
import { ArrowLeft, Flame, TrendingUp, Gauge } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function TrainingRing() {
  const torusRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (torusRef.current) {
      torusRef.current.rotation.x = state.clock.elapsedTime * 0.3;
      torusRef.current.rotation.y = state.clock.elapsedTime * 0.2;
    }
  });

  return (
    <Torus ref={torusRef} args={[1.5, 0.5, 16, 100]}>
      <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.3} />
    </Torus>
  );
}

export default function ForgePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-red-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-red-400 hover:text-red-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent flex items-center gap-4">
          <Flame className="w-12 h-12 text-red-400" />
          Phase 5: Forge Training
        </h1>
        <p className="text-xl text-gray-400">
          High-performance training and fine-tuning
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Gauge className="w-6 h-6 text-red-400" />
            Training Progress
          </h2>
          <Canvas camera={{ position: [0, 0, 5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <TrainingRing />
            <OrbitControls />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-green-400" />
            Training Metrics
          </h2>
          {[
            { label: 'Loss', value: 0.032, color: '#ef4444', trend: 'down' },
            { label: 'Accuracy', value: 96.8, color: '#10b981', trend: 'up' },
            { label: 'Learning Rate', value: 0.0001, color: '#f59e0b', trend: 'stable' }
          ].map((metric, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-300">{metric.label}</span>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold" style={{ color: metric.color }}>
                    {metric.label === 'Accuracy' ? metric.value + '%' : metric.value}
                  </span>
                  <span className={'text-xs ' + (metric.trend === 'up' ? 'text-green-400' : metric.trend === 'down' ? 'text-blue-400' : 'text-gray-400')}>
                    {metric.trend}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
