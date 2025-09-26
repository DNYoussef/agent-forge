'use client';

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box } from '@react-three/drei';
import { ArrowLeft, Package, Zap, HardDrive } from 'lucide-react';
import Link from 'next/link';
import * as THREE from 'three';

function CompressionCubes() {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.x = state.clock.elapsedTime * 0.2;
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {[...Array(27)].map((_, i) => {
        const x = (i % 3) - 1;
        const y = Math.floor((i / 3) % 3) - 1;
        const z = Math.floor(i / 9) - 1;
        return (
          <Box key={i} args={[0.3, 0.3, 0.3]} position={[x, y, z]}>
            <meshStandardMaterial color="#f97316" wireframe />
          </Box>
        );
      })}
    </group>
  );
}

export default function BitNetPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-orange-950 to-slate-950 text-white p-8">
      <Link href="/" className="flex items-center gap-2 text-orange-400 hover:text-orange-300 mb-8">
        <ArrowLeft className="w-5 h-5" />
        Back to Dashboard
      </Link>

      <div className="mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent flex items-center gap-4">
          <Package className="w-12 h-12 text-orange-400" />
          Phase 4: BitNet Compression
        </h1>
        <p className="text-xl text-gray-400">
          1-bit quantization and extreme model compression
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 border border-white/10 h-[500px]">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <HardDrive className="w-6 h-6 text-orange-400" />
            Compression Layers
          </h2>
          <Canvas camera={{ position: [3, 3, 3] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <CompressionCubes />
            <OrbitControls />
          </Canvas>
        </div>

        <div className="space-y-4">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            Compression Stats
          </h2>
          {[
            { label: 'Model Size', before: '7.2GB', after: '890MB', ratio: '8.1x' },
            { label: 'Inference Speed', before: '45ms', after: '8ms', ratio: '5.6x' },
            { label: 'Accuracy Loss', before: '94.2%', after: '93.8%', ratio: '0.4%' }
          ].map((stat, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-lg rounded-xl p-4 border border-white/10">
              <h3 className="text-lg font-bold text-orange-400 mb-2">{stat.label}</h3>
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Before: {stat.before}</span>
                <span className="text-gray-400">After: {stat.after}</span>
              </div>
              <div className="mt-2 text-right text-green-400 font-bold">{stat.ratio}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
