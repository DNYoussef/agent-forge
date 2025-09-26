import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Sphere, Line, Text, Icosahedron } from '@react-three/drei';
import * as THREE from 'three';

interface VPTQVisualizationProps {
  active: boolean;
  config: {
    bits: number;
    vectorDim: number;
    codebookSize: number;
    iterations: number;
  };
  progress: number;
}

export const VPTQVisualization: React.FC<VPTQVisualizationProps> = ({
  active,
  config,
  progress
}) => {
  const codebookRef = useRef<THREE.Group>(null);
  const vectorsRef = useRef<THREE.InstancedMesh>(null);
  const kmeansRef = useRef<THREE.Group>(null);
  const time = useRef(0);

  // Generate codebook entries as constellation
  const codebookEntries = useMemo(() => {
    const entries = [];
    const numEntries = Math.min(config.codebookSize, 32); // Limit for visualization
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    for (let i = 0; i < numEntries; i++) {
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.acos(1 - 2 * i / numEntries);

      const x = Math.sin(phi) * Math.cos(theta) * 2;
      const y = Math.sin(phi) * Math.sin(theta) * 2;
      const z = Math.cos(phi) * 2;

      // Color based on cluster usage
      const hue = (i / numEntries) * 360;
      const color = new THREE.Color().setHSL(hue / 360, 0.8, 0.6);

      entries.push({
        position: [x, y, z] as [number, number, number],
        color,
        id: i
      });
    }

    return entries;
  }, [config.codebookSize]);

  // Animate codebook constellation
  useFrame((state, delta) => {
    time.current += delta;

    if (codebookRef.current && active) {
      codebookRef.current.rotation.y += delta * 0.1;

      // Pulsate codebook entries based on K-means iterations
      const iterationProgress = (progress / 100) * config.iterations;
      const scale = 1 + Math.sin(iterationProgress * Math.PI) * 0.2;
      codebookRef.current.scale.setScalar(scale);
    }

    // Animate weight vectors snapping to codebook entries
    if (vectorsRef.current && active) {
      const dummy = new THREE.Object3D();
      const numVectors = 50; // Visualization sample

      for (let i = 0; i < numVectors; i++) {
        const t = progress / 100;
        const angle = (i / numVectors) * Math.PI * 2 + time.current * 0.2;

        // Initial position (scattered)
        const startX = Math.cos(angle) * 3 * (1 - t);
        const startY = Math.sin(angle) * 3 * (1 - t);
        const startZ = Math.sin(i * 0.5) * 2 * (1 - t);

        // Target codebook entry
        const targetEntry = codebookEntries[i % codebookEntries.length];

        // Interpolate to codebook entry
        dummy.position.set(
          startX + targetEntry.position[0] * t,
          startY + targetEntry.position[1] * t,
          startZ + targetEntry.position[2] * t
        );

        // Scale based on quantization
        const quantizationScale = 1 - (1 - 1 / (2 ** config.bits)) * t;
        dummy.scale.setScalar(quantizationScale * 0.5);

        dummy.updateMatrix();
        vectorsRef.current.setMatrixAt(i, dummy.matrix);
      }

      vectorsRef.current.instanceMatrix.needsUpdate = true;
    }

    // K-means centroid updates visualization
    if (kmeansRef.current && active) {
      kmeansRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Mesh) {
          const offset = Math.sin(time.current * 2 + i) * 0.1;
          child.position.y = codebookEntries[i % codebookEntries.length].position[1] + offset;
        }
      });
    }
  });

  // Quantization shells (discrete levels)
  const quantizationShells = useMemo(() => {
    const shells = [];
    const levels = 2 ** config.bits;

    for (let i = 0; i < Math.min(levels, 8); i++) {
      const radius = 0.5 + (i / levels) * 2.5;
      const opacity = 0.1 + (i / levels) * 0.2;

      shells.push(
        <mesh key={i}>
          <sphereGeometry args={[radius, 32, 16]} />
          <meshBasicMaterial
            color="#10b981"
            transparent
            opacity={opacity}
            side={THREE.BackSide}
            wireframe
          />
        </mesh>
      );
    }

    return shells;
  }, [config.bits]);

  // Connection lines between vectors and codebook entries
  const ConnectionLines = () => {
    if (!active || progress < 20) return null;

    return (
      <group>
        {codebookEntries.slice(0, 8).map((entry, i) => {
          const opacity = (progress - 20) / 80;
          return (
            <Line
              key={i}
              points={[
                [0, 0, 0],
                entry.position
              ]}
              color="#10b981"
              lineWidth={1}
              transparent
              opacity={opacity * 0.3}
              dashed
              dashSize={0.1}
              gapSize={0.05}
            />
          );
        })}
      </group>
    );
  };

  return (
    <group>
      {/* Quantization Shells */}
      <group>
        {quantizationShells}
      </group>

      {/* Codebook Constellation */}
      <group ref={codebookRef}>
        {codebookEntries.map((entry) => (
          <group key={entry.id} position={entry.position}>
            {/* Codebook entry */}
            <Icosahedron args={[0.15, 0]}>
              <meshStandardMaterial
                color={entry.color}
                emissive={entry.color}
                emissiveIntensity={active ? 0.5 : 0.1}
                metalness={0.8}
                roughness={0.2}
              />
            </Icosahedron>

            {/* Entry glow */}
            <Sphere args={[0.25, 16, 16]}>
              <meshBasicMaterial
                color={entry.color}
                transparent
                opacity={active ? 0.3 : 0.1}
              />
            </Sphere>
          </group>
        ))}
      </group>

      {/* Weight Vectors */}
      <instancedMesh
        ref={vectorsRef}
        args={[undefined, undefined, 50]}
      >
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshStandardMaterial
          color="#34d399"
          emissive="#10b981"
          emissiveIntensity={active ? 0.3 : 0}
          metalness={0.6}
          roughness={0.4}
        />
      </instancedMesh>

      {/* K-means Centroids */}
      <group ref={kmeansRef}>
        {codebookEntries.slice(0, 8).map((entry, i) => (
          <Box
            key={i}
            position={entry.position}
            args={[0.1, 0.1, 0.1]}
          >
            <meshBasicMaterial
              color="#fbbf24"
              transparent
              opacity={active ? 0.8 : 0.2}
            />
          </Box>
        ))}
      </group>

      {/* Connection Lines */}
      <ConnectionLines />

      {/* Vector Dimension Indicator */}
      <group position={[0, -2, 0]}>
        {Array.from({ length: config.vectorDim }).map((_, i) => {
          const angle = (i / config.vectorDim) * Math.PI * 2;
          const radius = 1.5;
          return (
            <mesh
              key={i}
              position={[
                Math.cos(angle) * radius,
                0,
                Math.sin(angle) * radius
              ]}
            >
              <cylinderGeometry args={[0.05, 0.05, 0.5, 8]} />
              <meshStandardMaterial
                color="#10b981"
                emissive="#10b981"
                emissiveIntensity={active ? 0.3 : 0}
              />
            </mesh>
          );
        })}
      </group>

      {/* Info Display */}
      <group position={[0, 3.5, 0]}>
        <Text
          fontSize={0.3}
          color={active ? "#d1fae5" : "#9ca3af"}
          anchorX="center"
          anchorY="middle"
        >
          VPTQ
        </Text>
        <Text
          position={[0, -0.5, 0]}
          fontSize={0.2}
          color={active ? "#34d399" : "#6b7280"}
          anchorX="center"
          anchorY="middle"
        >
          {config.bits} bits | {config.codebookSize} codes
        </Text>
      </group>

      {/* Progress Ring */}
      <mesh position={[0, -2.5, 0]} rotation={[0, 0, 0]}>
        <ringGeometry args={[1.8, 2, 32, 1, 0, (progress / 100) * Math.PI * 2]} />
        <meshBasicMaterial
          color="#10b981"
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* K-means Iteration Indicator */}
      {active && (
        <group position={[0, 2.5, 0]}>
          <Text
            fontSize={0.15}
            color="#fbbf24"
            anchorX="center"
            anchorY="middle"
          >
            Iteration: {Math.floor((progress / 100) * config.iterations)}/{config.iterations}
          </Text>
        </group>
      )}
    </group>
  );
};