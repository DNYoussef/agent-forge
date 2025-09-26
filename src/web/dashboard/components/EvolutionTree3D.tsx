'use client';

import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface TreeNode {
  id: string;
  generation: number;
  model_index: number;
  fitness: number;
  type: 'winner' | 'loser' | 'middle' | 'original';
  color: string;
  position: { x: number; y: number; z: number };
  parent?: string;
  parents?: string[];
  breeding_type?: 'mutation' | 'chaos_merge' | 'crossover';
}

interface GenerationData {
  generation: number;
  nodes: TreeNode[];
}

interface EvolutionTree3DProps {
  treeData: GenerationData[];
  currentGeneration: number;
  animateBreeding?: boolean;
  onNodeClick?: (node: TreeNode) => void;
}

// Individual node component
function ModelNode({ node, onClick, isHighlighted }: { node: TreeNode; onClick?: () => void; isHighlighted: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation for winners
      if (node.type === 'winner') {
        meshRef.current.position.y = node.position.y + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      }
      // Pulse effect for highlighted nodes
      if (isHighlighted) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
        meshRef.current.scale.setScalar(scale);
      }
    }
  });

  // Node size based on fitness
  const nodeSize = 0.3 + (node.fitness * 0.3);

  // Node shape based on type
  const geometry = useMemo(() => {
    switch (node.type) {
      case 'winner':
        return new THREE.OctahedronGeometry(nodeSize);
      case 'loser':
        return new THREE.TetrahedronGeometry(nodeSize);
      case 'original':
        return new THREE.BoxGeometry(nodeSize, nodeSize, nodeSize);
      default:
        return new THREE.SphereGeometry(nodeSize, 16, 16);
    }
  }, [node.type, nodeSize]);

  return (
    <group position={[node.position.x, node.position.y, node.position.z]}>
      <mesh
        ref={meshRef}
        geometry={geometry}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={node.color}
          emissive={node.color}
          emissiveIntensity={hovered ? 0.5 : 0.2}
          metalness={0.3}
          roughness={0.4}
        />
      </mesh>

      {/* Fitness label */}
      <Text
        position={[0, nodeSize + 0.2, 0]}
        fontSize={0.15}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {node.fitness.toFixed(3)}
      </Text>

      {/* Hover info */}
      {hovered && (
        <Html position={[0, nodeSize + 0.5, 0]} center>
          <div className="bg-gray-900/90 text-white p-2 rounded-lg text-xs whitespace-nowrap">
            <div className="font-bold">{node.id}</div>
            <div>Type: {node.type}</div>
            <div>Fitness: {node.fitness.toFixed(4)}</div>
            {node.breeding_type && <div>Breeding: {node.breeding_type}</div>}
          </div>
        </Html>
      )}
    </group>
  );
}

// Breeding connection line
function BreedingConnection({
  start,
  end,
  type,
  animated
}: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  type: string;
  animated: boolean;
}) {
  const lineRef = useRef<THREE.Line>(null);

  useFrame((state) => {
    if (animated && lineRef.current) {
      // Animate the line opacity for breeding visualization
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
  });

  const color = type === 'mutation' ? '#10b981' : type === 'chaos_merge' ? '#f97316' : '#6366f1';

  return (
    <Line
      ref={lineRef}
      points={[start, end]}
      color={color}
      lineWidth={2}
      opacity={0.3}
      transparent
      dashed={type === 'chaos_merge'}
    />
  );
}

// Generation plane for visual separation
function GenerationPlane({ generation, width }: { generation: number; width: number }) {
  const y = generation * 3; // Spacing between generations

  return (
    <>
      <mesh position={[0, y, -1]} rotation={[0, 0, 0]}>
        <planeGeometry args={[width, 0.01]} />
        <meshBasicMaterial color="#ffffff" opacity={0.05} transparent />
      </mesh>
      <Text
        position={[-width / 2 - 1, y, 0]}
        fontSize={0.3}
        color="#666666"
        anchorX="right"
        anchorY="middle"
      >
        Gen {generation}
      </Text>
    </>
  );
}

// Main scene component
function EvolutionScene({ treeData, currentGeneration, animateBreeding, onNodeClick }: EvolutionTree3DProps) {
  const { camera } = useThree();
  const [highlightedNode, setHighlightedNode] = useState<string | null>(null);

  // Position camera to see the tree
  useEffect(() => {
    if (camera && currentGeneration > 0) {
      const targetY = currentGeneration * 1.5;
      camera.position.lerp(new THREE.Vector3(10, targetY + 5, 10), 0.1);
    }
  }, [currentGeneration, camera]);

  // Create node map for quick lookup
  const nodeMap = useMemo(() => {
    const map = new Map<string, TreeNode>();
    treeData.forEach(gen => {
      gen.nodes.forEach(node => {
        map.set(node.id, node);
      });
    });
    return map;
  }, [treeData]);

  // Generate connections
  const connections = useMemo(() => {
    const conns: Array<{
      start: THREE.Vector3;
      end: THREE.Vector3;
      type: string;
    }> = [];

    treeData.forEach(gen => {
      gen.nodes.forEach(node => {
        if (node.parent) {
          const parentNode = nodeMap.get(node.parent);
          if (parentNode) {
            conns.push({
              start: new THREE.Vector3(parentNode.position.x, parentNode.position.y, parentNode.position.z),
              end: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
              type: node.breeding_type || 'crossover'
            });
          }
        } else if (node.parents) {
          // Multiple parents (chaos merge)
          node.parents.forEach(parentId => {
            const parentNode = nodeMap.get(parentId);
            if (parentNode) {
              conns.push({
                start: new THREE.Vector3(parentNode.position.x, parentNode.position.y, parentNode.position.z),
                end: new THREE.Vector3(node.position.x, node.position.y, node.position.z),
                type: 'chaos_merge'
              });
            }
          });
        }
      });
    });

    return conns;
  }, [treeData, nodeMap]);

  return (
    <>
      {/* Ambient and directional lighting */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Generation planes */}
      {treeData.map((gen, idx) => (
        idx <= currentGeneration && (
          <GenerationPlane key={gen.generation} generation={gen.generation} width={12} />
        )
      ))}

      {/* Breeding connections */}
      {connections.map((conn, idx) => (
        <BreedingConnection
          key={idx}
          start={conn.start}
          end={conn.end}
          type={conn.type}
          animated={animateBreeding || false}
        />
      ))}

      {/* Model nodes */}
      {treeData.map((gen) => (
        gen.generation <= currentGeneration &&
        gen.nodes.map(node => (
          <ModelNode
            key={node.id}
            node={node}
            isHighlighted={highlightedNode === node.id}
            onClick={() => {
              setHighlightedNode(node.id);
              onNodeClick?.(node);
            }}
          />
        ))
      ))}

      {/* Stats overlay */}
      <Html position={[6, currentGeneration * 3 + 2, 0]} center>
        <div className="bg-gray-900/80 text-white p-3 rounded-lg">
          <div className="text-sm font-bold mb-1">Generation {currentGeneration}</div>
          <div className="text-xs space-y-1">
            <div className="flex gap-2">
              <span className="text-green-400">● Winners: 2</span>
            </div>
            <div className="flex gap-2">
              <span className="text-orange-400">● Chaos Pool: 6</span>
            </div>
            <div className="flex gap-2">
              <span className="text-purple-400">● Population: 8</span>
            </div>
          </div>
        </div>
      </Html>
    </>
  );
}

// Main component
export function EvolutionTree3D({ treeData, currentGeneration, animateBreeding = true, onNodeClick }: EvolutionTree3DProps) {
  return (
    <div className="w-full h-[600px] bg-gradient-to-b from-gray-900 to-black rounded-xl overflow-hidden">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[10, 5, 10]} fov={60} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={50}
          maxPolarAngle={Math.PI / 2}
        />
        <fog attach="fog" args={['#000000', 10, 50]} />

        <EvolutionScene
          treeData={treeData}
          currentGeneration={currentGeneration}
          animateBreeding={animateBreeding}
          onNodeClick={onNodeClick}
        />
      </Canvas>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-900/80 text-white p-3 rounded-lg text-xs">
        <div className="font-bold mb-2">Node Types</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Winners (Top 2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span>Chaos Pool (Bottom 6)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-indigo-500 rounded-full"></div>
            <span>Middle Performers</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500"></div>
            <span>Original Models</span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 bg-gray-900/80 text-white p-2 rounded-lg text-xs">
        <div className="font-bold mb-1">Controls</div>
        <div>Left Click + Drag: Rotate</div>
        <div>Right Click + Drag: Pan</div>
        <div>Scroll: Zoom</div>
      </div>
    </div>
  );
}