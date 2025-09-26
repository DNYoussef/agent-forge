import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Points, Line, Text, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface SVDWeightData {
  singular_values: number[];
  principal_components: number[][];
  weight_regions: {
    id: string;
    center: [number, number, number];
    radius: number;
    color: string;
    label: string;
    importance: number;
  }[];
  task_vectors: {
    id: string;
    direction: [number, number, number];
    magnitude: number;
    task_type: string;
  }[];
}

interface SVDWeightSpaceVisualizationProps {
  data?: SVDWeightData;
  compact?: boolean;
}

export const SVDWeightSpaceVisualization: React.FC<SVDWeightSpaceVisualizationProps> = ({
  data,
  compact = false
}) => {
  const groupRef = useRef<THREE.Group>(null);

  // Mock data if none provided
  const mockData: SVDWeightData = useMemo(() => ({
    singular_values: [15.2, 12.8, 9.4, 6.1, 3.2, 1.8, 0.9],
    principal_components: [
      [0.8, 0.6, 0.0],
      [0.0, 0.8, 0.6],
      [-0.6, 0.0, 0.8]
    ],
    weight_regions: [
      {
        id: 'attention_heads',
        center: [2, 1, 0],
        radius: 1.5,
        color: '#ff6b6b',
        label: 'Attention Heads',
        importance: 0.95
      },
      {
        id: 'mlp_layers',
        center: [-1.5, 2, 1],
        radius: 1.2,
        color: '#4ecdc4',
        label: 'MLP Layers',
        importance: 0.87
      },
      {
        id: 'embedding_space',
        center: [0, -2, -1],
        radius: 0.8,
        color: '#45b7d1',
        label: 'Embeddings',
        importance: 0.78
      },
      {
        id: 'layer_norm',
        center: [1.5, -1, 2],
        radius: 0.6,
        color: '#f9ca24',
        label: 'LayerNorm',
        importance: 0.62
      }
    ],
    task_vectors: [
      {
        id: 'reasoning_vector',
        direction: [1, 0.5, 0.2],
        magnitude: 2.5,
        task_type: 'reasoning'
      },
      {
        id: 'generation_vector',
        direction: [-0.3, 1, 0.8],
        magnitude: 2.1,
        task_type: 'generation'
      },
      {
        id: 'comprehension_vector',
        direction: [0.6, -0.8, 1],
        magnitude: 1.8,
        task_type: 'comprehension'
      }
    ]
  }), []);

  const activeData = data || mockData;

  // Generate points for principal component space
  const pcaPoints = useMemo(() => {
    const points = [];
    const geometry = new THREE.BufferGeometry();

    for (let i = 0; i < 1000; i++) {
      // Generate points in reduced dimensional space
      const x = (Math.random() - 0.5) * 8;
      const y = (Math.random() - 0.5) * 8;
      const z = (Math.random() - 0.5) * 8;

      // Weight by singular values for realistic distribution
      const sv1 = activeData.singular_values[0] || 1;
      const sv2 = activeData.singular_values[1] || 1;
      const sv3 = activeData.singular_values[2] || 1;

      points.push(x * sv1/15, y * sv2/15, z * sv3/15);
    }

    geometry.setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    return geometry;
  }, [activeData.singular_values]);

  // Animation frame
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Ambient lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />

      {/* Principal component axes */}
      {activeData.principal_components.slice(0, 3).map((pc, index) => (
        <group key={`pc-${index}`}>
          <Line
            points={[[0, 0, 0], [pc[0] * 4, pc[1] * 4, pc[2] * 4]]}
            color={['#ff4757', '#2ed573', '#3742fa'][index]}
            lineWidth={3}
          />
          {!compact && (
            <Text
              position={[pc[0] * 4.5, pc[1] * 4.5, pc[2] * 4.5]}
              fontSize={0.3}
              color={['#ff4757', '#2ed573', '#3742fa'][index]}
            >
              PC{index + 1}
            </Text>
          )}
        </group>
      ))}

      {/* Weight space point cloud */}
      <Points>
        <bufferGeometry attach="geometry" {...pcaPoints} />
        <pointsMaterial
          attach="material"
          color="#ffffff"
          size={0.02}
          transparent
          opacity={0.6}
        />
      </Points>

      {/* Weight regions as colored spheres */}
      {activeData.weight_regions.map((region) => (
        <group key={region.id}>
          <Sphere
            args={[region.radius]}
            position={region.center}
          >
            <meshPhongMaterial
              color={region.color}
              transparent
              opacity={0.3 + region.importance * 0.4}
              wireframe={false}
            />
          </Sphere>

          {/* Region labels */}
          {!compact && (
            <Text
              position={[
                region.center[0],
                region.center[1] + region.radius + 0.5,
                region.center[2]
              ]}
              fontSize={0.2}
              color={region.color}
              anchorX="center"
              anchorY="middle"
            >
              {region.label}
              {'\n'}
              {`${(region.importance * 100).toFixed(1)}%`}
            </Text>
          )}
        </group>
      ))}

      {/* Task vectors as arrows */}
      {activeData.task_vectors.map((vector) => (
        <group key={vector.id}>
          <Line
            points={[[0, 0, 0], [
              vector.direction[0] * vector.magnitude,
              vector.direction[1] * vector.magnitude,
              vector.direction[2] * vector.magnitude
            ]]}
            color="#ffd700"
            lineWidth={4}
          />

          {/* Arrow head */}
          <Sphere
            args={[0.1]}
            position={[
              vector.direction[0] * vector.magnitude,
              vector.direction[1] * vector.magnitude,
              vector.direction[2] * vector.magnitude
            ]}
          >
            <meshPhongMaterial color="#ffd700" />
          </Sphere>

          {!compact && (
            <Text
              position={[
                vector.direction[0] * vector.magnitude * 1.2,
                vector.direction[1] * vector.magnitude * 1.2,
                vector.direction[2] * vector.magnitude * 1.2
              ]}
              fontSize={0.2}
              color="#ffd700"
              anchorX="center"
            >
              {vector.task_type.toUpperCase()}
            </Text>
          )}
        </group>
      ))}

      {/* Coordinate grid */}
      <gridHelper args={[10, 10]} position={[0, -4, 0]} />

      {/* Orbit controls for interaction */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxDistance={15}
        minDistance={3}
      />
    </group>
  );
};

/* AGENT FOOTER BEGIN: DO NOT EDIT ABOVE THIS LINE */
## Version & Run Log
| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-09-25T20:31:15-04:00 | agent@Sonnet | Created SVD Weight Space 3D visualization with PCA axes, weight regions, and task vectors | SVDWeightSpaceVisualization.tsx | OK | Interactive Three.js visualization with mock data | 0.00 | a94f7d2 |

### Receipt
- status: OK
- reason_if_blocked: --
- run_id: phase7-agentic-svd-001
- inputs: ["weight_introspection_backend", "three_js_requirements"]
- tools_used: ["Write", "React", "Three.js", "@react-three/fiber", "@react-three/drei"]
- versions: {"model":"claude-sonnet-4","prompt":"phase7-frontend-v1"}
<!-- AGENT FOOTER END: DO NOT EDIT BELOW THIS LINE -->