'use client';

import { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface SimpleOrbProps {
  compressionProgress: number;
  compressionRatio: number;
  currentPhase: 'initializing' | 'calibration' | 'quantization' | 'fine_tuning' | 'completed';
}

// Simple rotating orb
function Orb({ compressionProgress, compressionRatio, currentPhase }: SimpleOrbProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Color based on phase
  const getColor = () => {
    switch (currentPhase) {
      case 'initializing': return '#3b82f6'; // Blue
      case 'calibration': return '#8b5cf6';  // Purple
      case 'quantization': return '#06b6d4'; // Cyan
      case 'fine_tuning': return '#10b981';  // Green
      case 'completed': return '#22c55e';    // Bright Green
      default: return '#3b82f6';
    }
  };

  // Size based on compression
  const scale = Math.max(0.5, 1.0 / (compressionRatio * 0.1 + 0.9));

  useFrame((state) => {
    if (meshRef.current) {
      // Rotate the orb
      meshRef.current.rotation.y += 0.01;
      meshRef.current.rotation.x += 0.005;

      // Pulsing effect during quantization
      if (currentPhase === 'quantization') {
        const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.1 + 1;
        meshRef.current.scale.setScalar(scale * pulse);
      } else {
        meshRef.current.scale.setScalar(scale);
      }
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, 0]}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshStandardMaterial
        color={getColor()}
        metalness={0.7}
        roughness={0.3}
        emissive={getColor()}
        emissiveIntensity={0.2}
      />
    </mesh>
  );
}

// Particle system for weight visualization
function WeightParticles() {
  const particlesRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.002;
    }
  });

  // Create particles for -1, 0, +1 weights
  const particles = [];
  const colors = [0xef4444, 0x6b7280, 0x22c55e]; // Red, Gray, Green

  for (let i = 0; i < 50; i++) {
    const color = colors[i % 3];
    const radius = 2 + Math.random() * 2;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    particles.push(
      <mesh key={i} position={[x, y, z]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>
    );
  }

  return (
    <group ref={particlesRef}>
      {particles}
    </group>
  );
}

// Simple lighting setup
function Lighting() {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#06b6d4" />
    </>
  );
}

// Main component
export default function SimpleOrb(props: SimpleOrbProps) {
  return (
    <div className="w-full h-[400px] bg-black rounded-lg">
      <Canvas camera={{ position: [3, 3, 3], fov: 60 }}>
        <Lighting />
        <Orb {...props} />
        <WeightParticles />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={8}
        />
      </Canvas>
    </div>
  );
}