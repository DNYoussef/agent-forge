'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Cone, Cylinder } from '@react-three/drei';
import { ArrowLeft, Microscope, Network, Cpu } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function ArchitectureStructure() {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.15;
    }
  });

  return (
    <group ref={groupRef}>
      <Cylinder args={[0.5, 0.5, 3, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#ec4899" />
      </Cylinder>
      {[...Array(4)].map((_, i) => (
        <Cone key={i} args={[0.3, 1, 32]} position={[Math.cos(i * Math.PI / 2) * 2, 0, Math.sin(i * Math.PI / 2) * 2]}>
          <meshStandardMaterial color="#f472b6" />
        </Cone>
      ))}
    </group>
  );
}

export default function AdasPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-pink-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-pink-400 hover:text-pink-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-4">
          <Microscope className="w-12 h-12 text-pink-400" />
          Phase 7: ADAS Architecture Search
        </h1>
        <p className="text-xl text-gray-400">
          Neural architecture search and optimization
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Network className="w-6 h-6 text-pink-400" />
            Architecture Visualization
          </h2>
          <Canvas camera={{ position: [5, 3, 5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <ArchitectureStructure />
            <OrbitControls />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Cpu className="w-6 h-6 text-purple-400" />
            Search Results
          </h2>
          {[
            { name: 'Transformer++', params: '2.1B', score: 94.2, color: '#ec4899' },
            { name: 'Hybrid CNN-RNN', params: '1.8B', score: 91.5, color: '#f472b6' },
            { name: 'Custom Attention', params: '1.5B', score: 89.7, color: '#f9a8d4' }
          ].map((arch, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-bold" style={{ color: arch.color }}>{arch.name}</h3>
                <span className="text-sm text-gray-400">{arch.params}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Score</span>
                <span className="text-xl font-bold" style={{ color: arch.color }}>{arch.score}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
