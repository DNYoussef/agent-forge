import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Sphere, Line, Text, Torus, TorusKnot } from '@react-three/drei';
import * as THREE from 'three';

interface HyperCompressionVisualizationProps {
  active: boolean;
  config: {
    numClusters: number;
    trajectoryType: string;
    ergodicity: number;
    grokfastAlpha: number;
  };
  progress: number;
}

export const HyperCompressionVisualization: React.FC<HyperCompressionVisualizationProps> = ({
  active,
  config,
  progress
}) => {
  const phaseSpaceRef = useRef<THREE.Group>(null);
  const trajectoriesRef = useRef<THREE.Group>(null);
  const clustersRef = useRef<THREE.InstancedMesh>(null);
  const time = useRef(0);

  // Generate trajectory paths based on type
  const generateTrajectoryPath = (type: string, t: number, clusterId: number) => {
    const scale = 2;
    let x, y, z;

    switch (type) {
      case 'sinusoidal':
        x = Math.sin(t * 2 + clusterId) * scale;
        y = Math.cos(t * 3 + clusterId * 0.5) * scale * 0.8;
        z = Math.sin(t + clusterId * 0.3) * scale * 0.6;
        break;

      case 'spiral':
        const spiralRadius = scale * (1 - t / (Math.PI * 4));
        x = Math.cos(t * 2 + clusterId) * spiralRadius;
        y = (t / (Math.PI * 4) - 0.5) * scale * 2;
        z = Math.sin(t * 2 + clusterId) * spiralRadius;
        break;

      case 'chaotic':
        // Lorenz attractor-inspired
        const sigma = 10;
        const rho = 28;
        const beta = 8/3;
        const dt = 0.01;
        x = Math.sin(t * sigma + clusterId) * scale;
        y = Math.cos(t * rho + clusterId * 0.7) * scale;
        z = Math.sin(t * beta + clusterId * 0.3) * scale * 0.8;
        break;

      default: // auto
        // Mix of trajectories
        const mixT = t + clusterId * Math.PI / config.numClusters;
        x = Math.sin(mixT * 2) * Math.cos(mixT) * scale;
        y = Math.sin(mixT * 3) * scale * 0.8;
        z = Math.cos(mixT * 2) * Math.sin(mixT * 0.5) * scale * 0.6;
    }

    return new THREE.Vector3(x, y, z);
  };

  // Create trajectory curves
  const trajectoryPaths = useMemo(() => {
    const paths = [];
    const numPaths = Math.min(config.numClusters, 8); // Limit for performance

    for (let i = 0; i < numPaths; i++) {
      const points = [];
      const segments = 100;

      for (let j = 0; j <= segments; j++) {
        const t = (j / segments) * Math.PI * 4;
        const point = generateTrajectoryPath(
          config.trajectoryType === 'auto' ? ['sinusoidal', 'spiral', 'chaotic'][i % 3] : config.trajectoryType,
          t,
          i
        );
        points.push(point);
      }

      const curve = new THREE.CatmullRomCurve3(points, true);
      const curvePoints = curve.getPoints(200);

      // Color based on trajectory type
      const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];
      const color = colors[i % colors.length];

      paths.push({
        points: curvePoints,
        color,
        id: i
      });
    }

    return paths;
  }, [config.numClusters, config.trajectoryType]);

  useFrame((state, delta) => {
    time.current += delta;

    // Rotate phase space
    if (phaseSpaceRef.current && active) {
      phaseSpaceRef.current.rotation.y += delta * 0.05;
      phaseSpaceRef.current.rotation.x = Math.sin(time.current * 0.2) * 0.1;
    }

    // Animate trajectories
    if (trajectoriesRef.current && active) {
      trajectoriesRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Line) {
          // Pulsate trajectory based on ergodicity
          const scale = 1 + Math.sin(time.current * 2 + i) * config.ergodicity * 0.2;
          child.scale.setScalar(scale);
        }
      });
    }

    // Animate weight clusters following trajectories
    if (clustersRef.current && active) {
      const dummy = new THREE.Object3D();

      for (let i = 0; i < config.numClusters; i++) {
        const t = (time.current + i * 0.5) % (Math.PI * 4);
        const position = generateTrajectoryPath(
          config.trajectoryType === 'auto' ? ['sinusoidal', 'spiral', 'chaotic'][i % 3] : config.trajectoryType,
          t,
          i
        );

        // Apply compression based on progress
        const compressionScale = 1 - (progress / 100) * 0.5;
        position.multiplyScalar(compressionScale);

        dummy.position.copy(position);

        // Scale clusters based on compression
        const clusterScale = 0.3 * (1 - (progress / 100) * 0.6);
        dummy.scale.setScalar(clusterScale);

        dummy.updateMatrix();
        clustersRef.current.setMatrixAt(i, dummy.matrix);
      }

      clustersRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  // Phase space grid
  const PhaseSpaceGrid = () => {
    const gridSize = 4;
    const spacing = 1;

    return (
      <group>
        {/* XY plane */}
        {Array.from({ length: gridSize }).map((_, i) => {
          const pos = (i - gridSize / 2) * spacing;
          return (
            <group key={`xy-${i}`}>
              <Line
                points={[
                  [-gridSize * spacing / 2, pos, 0],
                  [gridSize * spacing / 2, pos, 0]
                ]}
                color="#1e40af"
                lineWidth={0.5}
                transparent
                opacity={0.3}
              />
              <Line
                points={[
                  [pos, -gridSize * spacing / 2, 0],
                  [pos, gridSize * spacing / 2, 0]
                ]}
                color="#1e40af"
                lineWidth={0.5}
                transparent
                opacity={0.3}
              />
            </group>
          );
        })}
      </group>
    );
  };

  // Ergodic field effect
  const ErgodicField = () => {
    const fieldRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
      if (fieldRef.current && active) {
        fieldRef.current.rotation.x += 0.001;
        fieldRef.current.rotation.y += 0.002;
        (fieldRef.current.material as THREE.ShaderMaterial).uniforms.time.value = state.clock.elapsedTime;
      }
    });

    const shaderMaterial = useMemo(
      () => new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          ergodicity: { value: config.ergodicity }
        },
        vertexShader: `
          varying vec3 vPosition;
          void main() {
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform float time;
          uniform float ergodicity;
          varying vec3 vPosition;

          void main() {
            float noise = sin(vPosition.x * 5.0 + time) *
                         cos(vPosition.y * 5.0 + time * 0.7) *
                         sin(vPosition.z * 5.0 + time * 1.3);
            float alpha = (noise * 0.5 + 0.5) * ergodicity * 0.2;
            gl_FragColor = vec4(0.2, 0.5, 1.0, alpha);
          }
        `,
        transparent: true,
        side: THREE.DoubleSide
      }),
      [config.ergodicity]
    );

    return (
      <mesh ref={fieldRef} material={shaderMaterial}>
        <sphereGeometry args={[3, 32, 32]} />
      </mesh>
    );
  };

  return (
    <group>
      {/* Phase Space Container */}
      <group ref={phaseSpaceRef}>
        {/* Phase space grid */}
        <PhaseSpaceGrid />

        {/* Ergodic field */}
        <ErgodicField />

        {/* Trajectory paths */}
        <group ref={trajectoriesRef}>
          {trajectoryPaths.map((path) => (
            <Line
              key={path.id}
              points={path.points}
              color={path.color}
              lineWidth={2}
              transparent
              opacity={0.6 + (progress / 100) * 0.4}
            />
          ))}
        </group>

        {/* Phase space boundary */}
        <mesh>
          <sphereGeometry args={[3, 32, 16]} />
          <meshBasicMaterial
            color="#3b82f6"
            wireframe
            transparent
            opacity={0.2}
          />
        </mesh>
      </group>

      {/* Weight Clusters */}
      <instancedMesh
        ref={clustersRef}
        args={[undefined, undefined, config.numClusters]}
      >
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial
          color="#60a5fa"
          emissive="#3b82f6"
          emissiveIntensity={active ? 0.3 : 0}
          metalness={0.7}
          roughness={0.3}
        />
      </instancedMesh>

      {/* Trajectory Type Indicators */}
      <group position={[0, -2.5, 0]}>
        {config.trajectoryType === 'auto' ? (
          <>
            <TorusKnot args={[0.3, 0.1, 64, 8, 2, 3]} position={[-1, 0, 0]}>
              <meshStandardMaterial color="#8b5cf6" emissive="#8b5cf6" emissiveIntensity={0.3} />
            </TorusKnot>
            <Torus args={[0.3, 0.1, 8, 16]} position={[0, 0, 0]}>
              <meshStandardMaterial color="#ec4899" emissive="#ec4899" emissiveIntensity={0.3} />
            </Torus>
            <Box args={[0.4, 0.4, 0.4]} position={[1, 0, 0]} rotation={[0.5, 0.5, 0]}>
              <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.3} />
            </Box>
          </>
        ) : (
          <TorusKnot args={[0.5, 0.15, 64, 8, 3, 4]}>
            <meshStandardMaterial
              color="#3b82f6"
              emissive="#3b82f6"
              emissiveIntensity={active ? 0.5 : 0.1}
            />
          </TorusKnot>
        )}
      </group>

      {/* Info Display */}
      <group position={[0, 3.5, 0]}>
        <Text
          fontSize={0.3}
          color={active ? "#dbeafe" : "#9ca3af"}
          anchorX="center"
          anchorY="middle"
        >
          Hypercompression
        </Text>
        <Text
          position={[0, -0.5, 0]}
          fontSize={0.2}
          color={active ? "#60a5fa" : "#6b7280"}
          anchorX="center"
          anchorY="middle"
        >
          {config.numClusters} clusters | {config.trajectoryType}
        </Text>
      </group>

      {/* Ergodicity Indicator */}
      <group position={[2.5, 0, 0]}>
        <Text
          fontSize={0.15}
          color="#fbbf24"
          anchorX="center"
          anchorY="middle"
          rotation={[0, -Math.PI / 4, 0]}
        >
          Ergodicity: {(config.ergodicity * 100).toFixed(0)}%
        </Text>
      </group>

      {/* Progress Ring */}
      <mesh position={[0, -2.5, 0]} rotation={[0, 0, 0]}>
        <ringGeometry args={[1.8, 2, 32, 1, 0, (progress / 100) * Math.PI * 2]} />
        <meshBasicMaterial
          color="#3b82f6"
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Grokfast Indicator */}
      {active && config.grokfastAlpha > 0 && (
        <group position={[0, 2.5, 0]}>
          <Text
            fontSize={0.15}
            color="#f59e0b"
            anchorX="center"
            anchorY="middle"
          >
            Grokfast: α={config.grokfastAlpha.toFixed(2)}
          </Text>
        </group>
      )}
    </group>
  );
};