import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Text, Line } from '@react-three/drei';
import * as THREE from 'three';

interface CompressionMetrics3DProps {
  metrics: {
    seedlmRatio: number;
    vptqRatio: number;
    hyperRatio: number;
    totalRatio: number;
    accuracy: number;
    speed: number;
  };
  position: [number, number, number];
}

export const CompressionMetrics3D: React.FC<CompressionMetrics3DProps> = ({
  metrics,
  position
}) => {
  const graphRef = useRef<THREE.Group>(null);
  const barsRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (graphRef.current) {
      // Gentle floating animation
      graphRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      graphRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.05;
    }

    if (barsRef.current) {
      // Animate bar heights based on metrics
      barsRef.current.children.forEach((child, index) => {
        if (child instanceof THREE.Mesh) {
          const targetScale = getBarScale(index);
          child.scale.y = THREE.MathUtils.lerp(child.scale.y, targetScale, 0.1);
        }
      });
    }
  });

  const getBarScale = (index: number): number => {
    switch (index) {
      case 0: return metrics.seedlmRatio / 10;
      case 1: return metrics.vptqRatio / 10;
      case 2: return metrics.hyperRatio / 10;
      case 3: return metrics.totalRatio / 20;
      case 4: return metrics.accuracy / 100;
      case 5: return metrics.speed / 100;
      default: return 1;
    }
  };

  const getBarColor = (index: number): string => {
    switch (index) {
      case 0: return '#8b5cf6'; // SeedLM - Purple
      case 1: return '#10b981'; // VPTQ - Green
      case 2: return '#3b82f6'; // Hypercompression - Blue
      case 3: return '#f59e0b'; // Total - Yellow
      case 4: return '#ec4899'; // Accuracy - Pink
      case 5: return '#06b6d4'; // Speed - Cyan
      default: return '#ffffff';
    }
  };

  const barLabels = [
    'SeedLM',
    'VPTQ',
    'Hyper',
    'Total',
    'Accuracy',
    'Speed'
  ];

  // Create holographic panel effect
  const HolographicPanel = ({ width = 8, height = 4, depth = 0.1 }) => {
    const shaderMaterial = useMemo(
      () => new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 }
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform float time;
          varying vec2 vUv;

          void main() {
            vec3 color = vec3(0.1, 0.2, 0.4);
            float scanline = sin(vUv.y * 50.0 + time * 2.0) * 0.04;
            float edge = smoothstep(0.0, 0.02, vUv.x) * smoothstep(1.0, 0.98, vUv.x) *
                        smoothstep(0.0, 0.02, vUv.y) * smoothstep(1.0, 0.98, vUv.y);

            color += vec3(scanline);
            float alpha = 0.3 * edge;

            gl_FragColor = vec4(color, alpha);
          }
        `,
        transparent: true,
        side: THREE.DoubleSide
      }),
      []
    );

    useFrame((state) => {
      shaderMaterial.uniforms.time.value = state.clock.elapsedTime;
    });

    return (
      <mesh material={shaderMaterial}>
        <boxGeometry args={[width, height, depth]} />
      </mesh>
    );
  };

  // Create 3D line graph
  const LineGraph = () => {
    const points = useMemo(() => {
      const pts = [];
      const numPoints = 20;

      for (let i = 0; i < numPoints; i++) {
        const x = (i / numPoints) * 4 - 2;
        const y = Math.sin(i * 0.5) * 0.5 + metrics.totalRatio / 10 - 1;
        const z = 0.1;
        pts.push(new THREE.Vector3(x, y, z));
      }

      return pts;
    }, [metrics.totalRatio]);

    return (
      <Line
        points={points}
        color="#f59e0b"
        lineWidth={2}
        dashed={false}
      />
    );
  };

  return (
    <group ref={graphRef} position={position}>
      {/* Holographic Background Panel */}
      <HolographicPanel />

      {/* Title */}
      <Text
        position={[0, 1.5, 0.1]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        Compression Metrics
      </Text>

      {/* Bar Chart */}
      <group ref={barsRef} position={[-2.5, -0.5, 0.1]}>
        {barLabels.map((label, i) => {
          const xPos = i * 0.8;
          const barHeight = getBarScale(i);
          const color = getBarColor(i);

          return (
            <group key={i} position={[xPos, 0, 0]}>
              {/* Bar */}
              <Box args={[0.6, barHeight, 0.2]} position={[0, barHeight / 2, 0]}>
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={0.3}
                  metalness={0.7}
                  roughness={0.3}
                />
              </Box>

              {/* Label */}
              <Text
                position={[0, -0.3, 0]}
                fontSize={0.12}
                color={color}
                anchorX="center"
                anchorY="middle"
                rotation={[0, 0, -Math.PI / 6]}
              >
                {label}
              </Text>

              {/* Value */}
              <Text
                position={[0, barHeight + 0.2, 0]}
                fontSize={0.1}
                color="#ffffff"
                anchorX="center"
                anchorY="middle"
              >
                {i === 4 ? `${metrics.accuracy.toFixed(0)}%` :
                 i === 5 ? `${metrics.speed.toFixed(0)}MB/s` :
                 i === 3 ? `${metrics.totalRatio.toFixed(1)}x` :
                 i === 0 ? `${metrics.seedlmRatio.toFixed(1)}x` :
                 i === 1 ? `${metrics.vptqRatio.toFixed(1)}x` :
                 `${metrics.hyperRatio.toFixed(1)}x`}
              </Text>
            </group>
          );
        })}
      </group>

      {/* Grid Lines */}
      <group position={[0, 0, 0]}>
        {Array.from({ length: 5 }).map((_, i) => {
          const y = (i / 4) * 2 - 1;
          return (
            <Line
              key={i}
              points={[
                [-3, y, 0],
                [3, y, 0]
              ]}
              color="#374151"
              lineWidth={0.5}
              transparent
              opacity={0.3}
            />
          );
        })}
      </group>

      {/* Line Graph */}
      <group position={[0, 0, 0.2]}>
        <LineGraph />
      </group>

      {/* Efficiency Indicator */}
      <group position={[2.5, 0, 0.1]}>
        <mesh>
          <ringGeometry args={[0.4, 0.5, 32, 1, 0, (metrics.totalRatio / 50) * Math.PI * 2]} />
          <meshBasicMaterial color="#f59e0b" side={THREE.DoubleSide} />
        </mesh>
        <Text
          position={[0, 0, 0.1]}
          fontSize={0.15}
          color="#f59e0b"
          anchorX="center"
          anchorY="middle"
        >
          {metrics.totalRatio.toFixed(1)}x
        </Text>
        <Text
          position={[0, -0.7, 0]}
          fontSize={0.1}
          color="#9ca3af"
          anchorX="center"
          anchorY="middle"
        >
          Total
        </Text>
      </group>

      {/* Accuracy Gauge */}
      <group position={[2.5, -1.5, 0.1]}>
        <mesh>
          <ringGeometry args={[0.3, 0.4, 32, 1, 0, (metrics.accuracy / 100) * Math.PI * 2]} />
          <meshBasicMaterial color="#10b981" side={THREE.DoubleSide} />
        </mesh>
        <Text
          position={[0, 0, 0.1]}
          fontSize={0.12}
          color="#10b981"
          anchorX="center"
          anchorY="middle"
        >
          {metrics.accuracy.toFixed(0)}%
        </Text>
        <Text
          position={[0, -0.6, 0]}
          fontSize={0.08}
          color="#9ca3af"
          anchorX="center"
          anchorY="middle"
        >
          Accuracy
        </Text>
      </group>
    </group>
  );
};