import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Sphere, Torus } from '@react-three/drei';
import * as THREE from 'three';
import { SeedLMVisualization } from './SeedLMVisualization';
import { VPTQVisualization } from './VPTQVisualization';
import { HyperCompressionVisualization } from './HyperCompressionVisualization';
import { WeightMatrixFlow } from './WeightMatrixFlow';
import { GrokfastAccelerator } from './GrokfastAccelerator';

interface CompressionPipeline3DProps {
  stage: string;
  progress: number;
  config: any;
  isPlaying: boolean;
}

export const CompressionPipeline3D: React.FC<CompressionPipeline3DProps> = ({
  stage,
  progress,
  config,
  isPlaying
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const time = useRef(0);

  useFrame((state, delta) => {
    if (isPlaying) {
      time.current += delta;

      // Gentle rotation for the entire pipeline
      if (groupRef.current) {
        groupRef.current.rotation.y = Math.sin(time.current * 0.1) * 0.05;
      }
    }
  });

  // Pipeline chamber positions
  const chamberPositions = {
    seedlm: new THREE.Vector3(-8, 0, 0),
    vptq: new THREE.Vector3(0, 0, 0),
    hypercompression: new THREE.Vector3(8, 0, 0)
  };

  // Connection pipes between chambers
  const ConnectionPipe = ({ start, end, active }: { start: THREE.Vector3; end: THREE.Vector3; active: boolean }) => {
    const midPoint = new THREE.Vector3().lerpVectors(start, end, 0.5);
    const distance = start.distanceTo(end);

    return (
      <group position={midPoint}>
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.3, 0.3, distance - 3, 16]} />
          <meshStandardMaterial
            color={active ? "#8b5cf6" : "#374151"}
            emissive={active ? "#8b5cf6" : "#000000"}
            emissiveIntensity={active ? 0.3 : 0}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>
      </group>
    );
  };

  // Chamber glow effect
  const ChamberGlow = ({ position, active, color }: { position: THREE.Vector3; active: boolean; color: string }) => {
    const glowRef = useRef<THREE.Mesh>(null);

    useFrame((state, delta) => {
      if (glowRef.current && active) {
        glowRef.current.scale.x = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
        glowRef.current.scale.y = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
        glowRef.current.scale.z = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      }
    });

    return (
      <Sphere ref={glowRef} position={position} args={[3.5, 32, 32]}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={active ? 0.2 : 0.05}
          side={THREE.BackSide}
        />
      </Sphere>
    );
  };

  return (
    <group ref={groupRef}>
      {/* Base Platform */}
      <Box position={[0, -3, 0]} args={[30, 0.5, 15]}>
        <meshStandardMaterial
          color="#1f2937"
          metalness={0.9}
          roughness={0.1}
        />
      </Box>

      {/* Grid Lines on Platform */}
      <gridHelper
        args={[30, 30, "#4b5563", "#374151"]}
        position={[0, -2.7, 0]}
        rotation={[0, 0, 0]}
      />

      {/* SeedLM Chamber */}
      <group position={chamberPositions.seedlm}>
        <SeedLMVisualization
          active={stage === 'seedlm'}
          config={config.seedlm}
          progress={stage === 'seedlm' ? progress : stage === 'idle' ? 0 : 100}
        />
        <ChamberGlow
          position={new THREE.Vector3(0, 0, 0)}
          active={stage === 'seedlm'}
          color="#8b5cf6"
        />
      </group>

      {/* VPTQ Chamber */}
      <group position={chamberPositions.vptq}>
        <VPTQVisualization
          active={stage === 'vptq'}
          config={config.vptq}
          progress={stage === 'vptq' ? progress : ['seedlm', 'idle'].includes(stage) ? 0 : 100}
        />
        <ChamberGlow
          position={new THREE.Vector3(0, 0, 0)}
          active={stage === 'vptq'}
          color="#10b981"
        />
      </group>

      {/* Hypercompression Chamber */}
      <group position={chamberPositions.hypercompression}>
        <HyperCompressionVisualization
          active={stage === 'hypercompression'}
          config={config.hypercompression}
          progress={stage === 'hypercompression' ? progress : ['complete'].includes(stage) ? 100 : 0}
        />
        <ChamberGlow
          position={new THREE.Vector3(0, 0, 0)}
          active={stage === 'hypercompression'}
          color="#3b82f6"
        />
      </group>

      {/* Connection Pipes */}
      <ConnectionPipe
        start={chamberPositions.seedlm}
        end={chamberPositions.vptq}
        active={['vptq', 'hypercompression', 'complete'].includes(stage)}
      />
      <ConnectionPipe
        start={chamberPositions.vptq}
        end={chamberPositions.hypercompression}
        active={['hypercompression', 'complete'].includes(stage)}
      />

      {/* Weight Flow Animation */}
      <WeightMatrixFlow
        stage={stage}
        chamberPositions={chamberPositions}
        isPlaying={isPlaying}
      />

      {/* Grokfast Accelerator */}
      {stage === 'hypercompression' && (
        <GrokfastAccelerator
          position={chamberPositions.hypercompression}
          intensity={config.hypercompression.grokfastAlpha}
          active={true}
        />
      )}

      {/* Stage Labels */}
      <group>
        <mesh position={[chamberPositions.seedlm.x, -2, 5]}>
          <planeGeometry args={[3, 0.8]} />
          <meshBasicMaterial color="#1f2937" transparent opacity={0.8} />
        </mesh>
        <mesh position={[chamberPositions.vptq.x, -2, 5]}>
          <planeGeometry args={[3, 0.8]} />
          <meshBasicMaterial color="#1f2937" transparent opacity={0.8} />
        </mesh>
        <mesh position={[chamberPositions.hypercompression.x, -2, 5]}>
          <planeGeometry args={[4, 0.8]} />
          <meshBasicMaterial color="#1f2937" transparent opacity={0.8} />
        </mesh>
      </group>
    </group>
  );
};