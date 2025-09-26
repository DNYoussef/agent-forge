'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Html } from '@react-three/drei';
import { ArrowLeft, Sparkles, Rocket, CheckCircle } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function FinalSphere() {
  const sphereRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (sphereRef.current) {
      sphereRef.current.rotation.y = state.clock.elapsedTime * 0.5;
      sphereRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.3;
    }
  });

  return (
    <Sphere ref={sphereRef} args={[1.5, 64, 64]}>
      <meshStandardMaterial color="#eab308" emissive="#eab308" emissiveIntensity={0.5} metalness={0.8} roughness={0.2} />
    </Sphere>
  );
}

export default function FinalPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-yellow-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-yellow-400 hover:text-yellow-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text text-transparent flex items-center gap-4">
          <Sparkles className="w-12 h-12 text-yellow-400" />
          Phase 8: Final Production
        </h1>
        <p className="text-xl text-gray-400">
          Deploy optimized agent to production
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Rocket className="w-6 h-6 text-yellow-400" />
            Production Agent
          </h2>
          <Canvas camera={{ position: [0, 0, 5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <FinalSphere />
            <OrbitControls />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-400" />
            Deployment Checklist
          </h2>
          {[
            { task: 'Model Optimization', status: 'complete', color: '#10b981' },
            { task: 'Quality Assurance', status: 'complete', color: '#10b981' },
            { task: 'Performance Testing', status: 'complete', color: '#10b981' },
            { task: 'Production Deployment', status: 'ready', color: '#eab308' }
          ].map((item, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10 flex items-center justify-between">
              <span className="text-gray-300">{item.task}</span>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-sm capitalize" style={{ color: item.color }}>{item.status}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
