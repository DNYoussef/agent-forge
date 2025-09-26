'use client';

import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Sphere, Text, OrbitControls, Float } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';

interface BitNetOrbProps {
  compressionProgress: number;
  compressionRatio: number;
  performanceRetention: number;
  weightsDistribution: {
    negative: number;
    zero: number;
    positive: number;
  };
  currentPhase: 'initializing' | 'calibration' | 'quantization' | 'fine_tuning' | 'completed';
  quantizedLayers: number;
  totalLayers: number;
}

// Main orb component
function ModelOrb({
  compressionProgress,
  compressionRatio,
  performanceRetention,
  currentPhase
}: BitNetOrbProps) {
  const orbRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);

  // Calculate orb size based on compression ratio
  // Original size: 2, compressed size scales down
  const orbSize = useMemo(() => {
    const baseSize = 2.0;
    const scaleFactor = 1.0 / Math.max(1, compressionRatio * 0.1 + 0.9);
    return baseSize * scaleFactor;
  }, [compressionRatio]);

  // Calculate color based on phase and progress
  const orbColor = useMemo(() => {
    const progress = compressionProgress / 100;

    switch (currentPhase) {
      case 'initializing':
        return new THREE.Color(0x3b82f6); // Blue
      case 'calibration':
        return new THREE.Color().lerpColors(
          new THREE.Color(0x3b82f6), // Blue
          new THREE.Color(0x8b5cf6), // Purple
          progress
        );
      case 'quantization':
        return new THREE.Color().lerpColors(
          new THREE.Color(0x8b5cf6), // Purple
          new THREE.Color(0x06b6d4), // Cyan
          progress
        );
      case 'fine_tuning':
        return new THREE.Color().lerpColors(
          new THREE.Color(0x06b6d4), // Cyan
          new THREE.Color(0x10b981), // Green
          progress
        );
      case 'completed':
        return new THREE.Color(0x10b981); // Green
      default:
        return new THREE.Color(0x3b82f6);
    }
  }, [currentPhase, compressionProgress]);

  // Animate orb rotation and pulsing
  useFrame((state) => {
    if (orbRef.current) {
      orbRef.current.rotation.y += 0.005;

      // Pulsing effect during compression
      if (currentPhase === 'quantization') {
        const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.05 + 1;
        orbRef.current.scale.setScalar(pulse);
      } else {
        orbRef.current.scale.setScalar(1);
      }
    }

    // Update material color
    if (materialRef.current) {
      materialRef.current.color.lerp(orbColor, 0.1);

      // Adjust opacity based on performance retention
      const targetOpacity = Math.max(0.4, performanceRetention / 100);
      materialRef.current.opacity = THREE.MathUtils.lerp(
        materialRef.current.opacity,
        targetOpacity,
        0.1
      );
    }
  });

  return (
    <Float speed={1} rotationIntensity={0.1} floatIntensity={0.2}>
      <mesh ref={orbRef} position={[0, 0, 0]}>
        <sphereGeometry args={[orbSize, 64, 64]} />
        <meshStandardMaterial
          ref={materialRef}
          color={orbColor}
          metalness={0.7}
          roughness={0.3}
          transparent={true}
          opacity={performanceRetention / 100}
        />
      </mesh>
    </Float>
  );
}

// Weight particles showing -1, 0, +1 values
function WeightParticles({ weightsDistribution, compressionProgress }: Pick<BitNetOrbProps, 'weightsDistribution' | 'compressionProgress'>) {
  const particlesRef = useRef<THREE.Group>(null);

  // Create particles based on weight distribution
  const particles = useMemo(() => {
    const particleCount = 200;
    const result = [];

    const totalWeights = weightsDistribution.negative + weightsDistribution.zero + weightsDistribution.positive;

    if (totalWeights === 0) return result;

    const negativeCount = Math.floor(particleCount * weightsDistribution.negative);
    const zeroCount = Math.floor(particleCount * weightsDistribution.zero);
    const positiveCount = Math.floor(particleCount * weightsDistribution.positive);

    // Create negative weight particles (red)
    for (let i = 0; i < negativeCount; i++) {
      result.push({
        value: -1,
        color: 0xef4444,
        position: [
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8
        ]
      });
    }

    // Create zero weight particles (gray)
    for (let i = 0; i < zeroCount; i++) {
      result.push({
        value: 0,
        color: 0x6b7280,
        position: [
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8
        ]
      });
    }

    // Create positive weight particles (green)
    for (let i = 0; i < positiveCount; i++) {
      result.push({
        value: 1,
        color: 0x22c55e,
        position: [
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 8
        ]
      });
    }

    return result;
  }, [weightsDistribution]);

  // Animate particles
  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.001;

      // Make particles orbit around the center orb
      particlesRef.current.children.forEach((child, index) => {
        const particle = child as THREE.Mesh;
        const time = state.clock.elapsedTime;
        const offset = index * 0.1;

        particle.position.x += Math.sin(time + offset) * 0.01;
        particle.position.z += Math.cos(time + offset) * 0.01;
      });
    }
  });

  return (
    <group ref={particlesRef}>
      {particles.map((particle, index) => (
        <mesh key={index} position={particle.position as [number, number, number]}>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshBasicMaterial color={particle.color} transparent opacity={0.7} />
        </mesh>
      ))}
    </group>
  );
}

// Compression ring effect
function CompressionRing({ compressionProgress, compressionRatio }: Pick<BitNetOrbProps, 'compressionProgress' | 'compressionRatio'>) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z += 0.01;

      // Scale ring based on compression
      const scale = 1 + (compressionRatio * 0.1);
      ringRef.current.scale.setScalar(scale);
    }
  });

  return (
    <mesh ref={ringRef} position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[3, 0.1, 16, 100]} />
      <meshBasicMaterial
        color={0x06b6d4}
        transparent
        opacity={compressionProgress / 100 * 0.3}
      />
    </mesh>
  );
}

// Phase label
function PhaseLabel({ currentPhase, quantizedLayers, totalLayers }: Pick<BitNetOrbProps, 'currentPhase' | 'quantizedLayers' | 'totalLayers'>) {
  const phaseNames = {
    initializing: 'Initializing',
    calibration: 'Calibrating',
    quantization: 'Quantizing',
    fine_tuning: 'Fine-tuning',
    completed: 'Completed'
  };

  return (
    <Text
      position={[0, 3.5, 0]}
      fontSize={0.4}
      color="#ffffff"
      anchorX="center"
      anchorY="middle"
      font="https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ.woff"
    >
      {phaseNames[currentPhase]}
      {currentPhase === 'quantization' && (
        `\n${quantizedLayers}/${totalLayers} layers`
      )}
    </Text>
  );
}

// Scene lighting
function SceneLighting() {
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#06b6d4" />
    </>
  );
}

// Main BitNet Orb component
export default function BitNetOrb(props: BitNetOrbProps) {
  return (
    <div className="w-full h-full min-h-[400px] relative">
      {/* Canvas with 3D scene */}
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <SceneLighting />

        <ModelOrb {...props} />
        <WeightParticles
          weightsDistribution={props.weightsDistribution}
          compressionProgress={props.compressionProgress}
        />
        <CompressionRing
          compressionProgress={props.compressionProgress}
          compressionRatio={props.compressionRatio}
        />
        <PhaseLabel
          currentPhase={props.currentPhase}
          quantizedLayers={props.quantizedLayers}
          totalLayers={props.totalLayers}
        />

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={false}
          minDistance={2}
          maxDistance={15}
        />
      </Canvas>

      {/* Overlay information */}
      <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg p-3 text-sm text-white">
        <div className="mb-2 font-semibold">BitNet 1.58-bit Quantization</div>
        <div className="space-y-1">
          <div>Progress: {props.compressionProgress.toFixed(1)}%</div>
          <div>Compression: {props.compressionRatio.toFixed(1)}x</div>
          <div>Performance: {props.performanceRetention.toFixed(1)}%</div>
        </div>
      </div>

      {/* Weight legend */}
      <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-3 text-sm text-white">
        <div className="mb-2 font-semibold">Weight Values</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span>-1 ({(props.weightsDistribution.negative * 100).toFixed(1)}%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
            <span>0 ({(props.weightsDistribution.zero * 100).toFixed(1)}%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>+1 ({(props.weightsDistribution.positive * 100).toFixed(1)}%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

