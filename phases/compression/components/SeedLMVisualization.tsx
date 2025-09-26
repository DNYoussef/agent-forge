import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Sphere, Cylinder, Ring, Text } from '@react-three/drei';
import * as THREE from 'three';

interface SeedLMVisualizationProps {
  active: boolean;
  config: {
    bitsPerWeight: number;
    maxCandidates: number;
    blockSize: number;
  };
  progress: number;
}

export const SeedLMVisualization: React.FC<SeedLMVisualizationProps> = ({
  active,
  config,
  progress
}) => {
  const lfsrRef = useRef<THREE.Group>(null);
  const projectionRef = useRef<THREE.Group>(null);
  const blocksRef = useRef<THREE.InstancedMesh>(null);

  // LFSR cipher wheel rotation
  useFrame((state, delta) => {
    if (lfsrRef.current && active) {
      lfsrRef.current.rotation.z += delta * 0.5;
      lfsrRef.current.rotation.y = Math.sin(state.clock.elapsedTime) * 0.1;
    }

    if (projectionRef.current && active) {
      projectionRef.current.rotation.x += delta * 0.2;
      projectionRef.current.rotation.y += delta * 0.1;
    }

    // Animate weight blocks compression
    if (blocksRef.current && active) {
      const time = state.clock.elapsedTime;
      const dummy = new THREE.Object3D();

      for (let i = 0; i < config.blockSize; i++) {
        const angle = (i / config.blockSize) * Math.PI * 2;
        const radius = 2 - (progress / 100) * 0.8; // Compress radius based on progress

        dummy.position.set(
          Math.cos(angle + time) * radius,
          Math.sin(i * 0.5 + time * 2) * 0.3,
          Math.sin(angle + time) * radius
        );

        // Scale blocks based on compression
        const scale = 1 - (progress / 100) * (1 - 1 / config.bitsPerWeight);
        dummy.scale.set(scale, scale, scale);

        dummy.updateMatrix();
        blocksRef.current.setMatrixAt(i, dummy.matrix);
      }
      blocksRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  // Create LFSR cipher wheel segments
  const lfsrSegments = useMemo(() => {
    const segments = [];
    const segmentCount = 16; // 16-bit LFSR

    for (let i = 0; i < segmentCount; i++) {
      const angle = (i / segmentCount) * Math.PI * 2;
      const color = i % 2 === 0 ? '#8b5cf6' : '#6366f1';

      segments.push(
        <mesh
          key={i}
          position={[
            Math.cos(angle) * 1.5,
            0,
            Math.sin(angle) * 1.5
          ]}
          rotation={[0, angle, 0]}
        >
          <boxGeometry args={[0.3, 0.2, 0.1]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={active ? 0.5 : 0.1}
            metalness={0.7}
            roughness={0.3}
          />
        </mesh>
      );
    }

    return segments;
  }, [active]);

  // Projection matrix visualization
  const projectionMatrix = useMemo(() => {
    const points = [];
    const matrixSize = 8;

    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        const x = (i - matrixSize / 2) * 0.2;
        const y = (j - matrixSize / 2) * 0.2;
        const z = Math.sin(i * j) * 0.1;

        points.push(
          <Sphere
            key={`${i}-${j}`}
            position={[x, y, z]}
            args={[0.05, 8, 8]}
          >
            <meshStandardMaterial
              color="#a78bfa"
              emissive="#8b5cf6"
              emissiveIntensity={active ? 0.3 : 0}
            />
          </Sphere>
        );
      }
    }

    return points;
  }, [active]);

  // Color for weight blocks based on compression
  const blockColor = useMemo(() => {
    const compressionLevel = progress / 100;
    const r = 0.5 + (1 - compressionLevel) * 0.5;
    const g = 0.3 + compressionLevel * 0.5;
    const b = 0.8;
    return new THREE.Color(r, g, b);
  }, [progress]);

  return (
    <group>
      {/* LFSR Cipher Wheel */}
      <group ref={lfsrRef} position={[0, 2, 0]}>
        {/* Central hub */}
        <Cylinder args={[0.5, 0.5, 0.3, 16]}>
          <meshStandardMaterial
            color="#8b5cf6"
            metalness={0.9}
            roughness={0.1}
          />
        </Cylinder>

        {/* Rotating segments */}
        {lfsrSegments}

        {/* Outer ring */}
        <Ring args={[1.8, 2, 32]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial
            color="#6366f1"
            metalness={0.8}
            roughness={0.2}
            side={THREE.DoubleSide}
          />
        </Ring>
      </group>

      {/* Projection Matrix */}
      <group ref={projectionRef} position={[0, 0, 0]}>
        {projectionMatrix}
      </group>

      {/* Weight Blocks */}
      <instancedMesh
        ref={blocksRef}
        args={[undefined, undefined, config.blockSize]}
        position={[0, -1, 0]}
      >
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial
          color={blockColor}
          metalness={0.6}
          roughness={0.4}
          emissive={blockColor}
          emissiveIntensity={active ? 0.2 : 0}
        />
      </instancedMesh>

      {/* Compression Rays */}
      {active && (
        <group>
          {Array.from({ length: 8 }).map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            return (
              <mesh
                key={i}
                position={[
                  Math.cos(angle) * 1,
                  0,
                  Math.sin(angle) * 1
                ]}
                rotation={[0, angle, Math.PI / 2]}
              >
                <cylinderGeometry args={[0.02, 0.02, 3, 8]} />
                <meshBasicMaterial
                  color="#8b5cf6"
                  transparent
                  opacity={0.5 + (progress / 100) * 0.5}
                />
              </mesh>
            );
          })}
        </group>
      )}

      {/* Info Display */}
      <group position={[0, 3.5, 0]}>
        <Text
          fontSize={0.3}
          color={active ? "#e9d5ff" : "#9ca3af"}
          anchorX="center"
          anchorY="middle"
        >
          SeedLM
        </Text>
        <Text
          position={[0, -0.5, 0]}
          fontSize={0.2}
          color={active ? "#c084fc" : "#6b7280"}
          anchorX="center"
          anchorY="middle"
        >
          {config.bitsPerWeight} bits/weight
        </Text>
      </group>

      {/* Progress Indicator */}
      <mesh position={[0, -2.5, 0]} rotation={[0, 0, 0]}>
        <ringGeometry args={[1.8, 2, 32, 1, 0, (progress / 100) * Math.PI * 2]} />
        <meshBasicMaterial
          color="#8b5cf6"
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
};