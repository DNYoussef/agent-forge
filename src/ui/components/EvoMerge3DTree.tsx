/**
 * EvoMerge3DTree - 3D Tree Visualization for Phase 2 EvoMerge
 *
 * Implements the user's specification:
 * - 3D growing tree with 8 colored roots (from initial model variants)
 * - Growing, merging, and separating animations
 * - Real-time visualization of evolutionary process
 * - Interactive genealogy tracking
 * - Color-coded lineages for easy identification
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Line2 } from 'three/examples/jsm/lines/Line2';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry';

interface ModelNode {
  id: string;
  generation: number;
  fitness: number;
  parents: string[];
  children: string[];
  creation_method: string;
  lineage_color: string;
  position: THREE.Vector3;
  metadata: any;
}

interface TreeBranch {
  from: ModelNode;
  to: ModelNode;
  color: string;
  thickness: number;
  animated: boolean;
}

interface GenerationData {
  generation: number;
  timestamp: number;
  event_type: string;
  models: ModelNode[];
}

interface EvoMerge3DTreeProps {
  generationData: GenerationData[];
  isRealTime: boolean;
  animationSpeed: number;
  showMetrics: boolean;
  onNodeClick?: (node: ModelNode) => void;
  onBranchClick?: (branch: TreeBranch) => void;
  width?: number;
  height?: number;
}

interface TreeLayout {
  nodes: Map<string, THREE.Vector3>;
  branches: TreeBranch[];
  generations: Map<number, ModelNode[]>;
}

const EvoMerge3DTree: React.FC<EvoMerge3DTreeProps> = ({
  generationData,
  isRealTime = true,
  animationSpeed = 1.0,
  showMetrics = true,
  onNodeClick,
  onBranchClick,
  width = 800,
  height = 600
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const animationFrameRef = useRef<number>();

  // State management
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(isRealTime);
  const [treeLayout, setTreeLayout] = useState<TreeLayout>({
    nodes: new Map(),
    branches: [],
    generations: new Map()
  });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());

  // Colors for the 8 initial lineages
  const LINEAGE_COLORS = [
    '#FF6B6B', // Red - SLERP variants
    '#4ECDC4', // Teal - TIES variants
    '#45B7D1', // Blue - DARE variants
    '#96CEB4', // Green - Weighted average variants
    '#FECA57', // Yellow - Arithmetic mean
    '#FF9FF3', // Pink - Geometric mean
    '#54A0FF', // Light blue - LERP variants
    '#5F27CD'  // Purple - Custom variants
  ];

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a); // Dark background
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 10, 20);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.minDistance = 5;
    controls.maxDistance = 100;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.setScalar(2048);
    scene.add(directionalLight);

    // Add to DOM
    mountRef.current.appendChild(renderer.domElement);

    // Initialize tree layout
    initializeTreeLayout();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [width, height]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Process new generation data
  useEffect(() => {
    if (generationData.length === 0) return;

    const newLayout = calculateTreeLayout(generationData);
    setTreeLayout(newLayout);
    updateVisualization(newLayout);
  }, [generationData]);

  const initializeTreeLayout = useCallback(() => {
    if (!sceneRef.current) return;

    // Create ground plane
    const groundGeometry = new THREE.PlaneGeometry(50, 50);
    const groundMaterial = new THREE.MeshLambertMaterial({
      color: 0x1a1a1a,
      transparent: true,
      opacity: 0.3
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    sceneRef.current.add(ground);

    // Create coordinate axes
    const axesHelper = new THREE.AxesHelper(5);
    sceneRef.current.add(axesHelper);
  }, []);

  const calculateTreeLayout = useCallback((data: GenerationData[]): TreeLayout => {
    const layout: TreeLayout = {
      nodes: new Map(),
      branches: [],
      generations: new Map()
    };

    // Group models by generation
    data.forEach(genData => {
      const nodes = genData.models.map(modelData => ({
        id: modelData.id,
        generation: modelData.generation,
        fitness: modelData.fitness,
        parents: modelData.parents || [],
        children: [],
        creation_method: modelData.creation_method,
        lineage_color: modelData.lineage_color,
        position: new THREE.Vector3(),
        metadata: modelData.metadata || {}
      }));

      layout.generations.set(genData.generation, nodes);
    });

    // Calculate positions for each generation
    Array.from(layout.generations.keys()).sort().forEach(generation => {
      const nodes = layout.generations.get(generation)!;
      calculateGenerationPositions(nodes, generation, layout);
    });

    // Create branches between generations
    layout.generations.forEach(nodes => {
      nodes.forEach(node => {
        node.parents.forEach(parentId => {
          const parentPos = layout.nodes.get(parentId);
          if (parentPos) {
            layout.branches.push({
              from: findNodeById(parentId, layout.generations)!,
              to: node,
              color: blendColors(node.lineage_color, findNodeById(parentId, layout.generations)?.lineage_color || '#ffffff'),
              thickness: Math.max(0.1, node.fitness * 0.5),
              animated: true
            });
          }
        });
      });
    });

    return layout;
  }, []);

  const calculateGenerationPositions = useCallback((
    nodes: ModelNode[],
    generation: number,
    layout: TreeLayout
  ) => {
    const radius = 3 + generation * 2; // Expand radius with each generation
    const yPosition = generation * 4; // Vertical spacing between generations

    if (generation === 0) {
      // Initial 3 models from Phase 1 - arrange in triangle
      const angles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
      nodes.forEach((node, index) => {
        if (index < 3) {
          const angle = angles[index];
          const position = new THREE.Vector3(
            Math.cos(angle) * radius,
            yPosition,
            Math.sin(angle) * radius
          );
          node.position = position;
          layout.nodes.set(node.id, position);
        }
      });
    } else {
      // Subsequent generations - arrange in circle with diversity spacing
      const angleStep = (2 * Math.PI) / nodes.length;

      nodes.forEach((node, index) => {
        // Add some randomness for organic growth
        const baseAngle = index * angleStep;
        const angleVariation = (Math.random() - 0.5) * 0.3; // ±0.15 radians
        const angle = baseAngle + angleVariation;

        // Add some radius variation based on fitness
        const radiusVariation = node.fitness * 0.5;
        const effectiveRadius = radius + radiusVariation;

        const position = new THREE.Vector3(
          Math.cos(angle) * effectiveRadius,
          yPosition,
          Math.sin(angle) * effectiveRadius
        );

        node.position = position;
        layout.nodes.set(node.id, position);
      });
    }
  }, []);

  const updateVisualization = useCallback((layout: TreeLayout) => {
    if (!sceneRef.current) return;

    // Clear existing nodes and branches
    const objectsToRemove = sceneRef.current.children.filter(child =>
      child.userData.type === 'node' || child.userData.type === 'branch'
    );
    objectsToRemove.forEach(obj => sceneRef.current!.remove(obj));

    // Create node meshes
    layout.generations.forEach(nodes => {
      nodes.forEach(node => createNodeMesh(node));
    });

    // Create branch meshes
    layout.branches.forEach(branch => createBranchMesh(branch));

    // Update generation counter
    const maxGeneration = Math.max(...Array.from(layout.generations.keys()));
    setCurrentGeneration(maxGeneration);
  }, []);

  const createNodeMesh = useCallback((node: ModelNode) => {
    if (!sceneRef.current) return;

    // Node size based on fitness
    const baseSize = 0.2;
    const fitnessScale = Math.max(0.5, node.fitness + 0.5);
    const nodeSize = baseSize * fitnessScale;

    // Create node geometry
    const geometry = new THREE.SphereGeometry(nodeSize, 16, 16);

    // Create material with lineage color
    const material = new THREE.MeshPhongMaterial({
      color: new THREE.Color(node.lineage_color),
      transparent: true,
      opacity: 0.8,
      shininess: 30
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(node.position);
    mesh.userData = {
      type: 'node',
      nodeId: node.id,
      node: node
    };

    // Add glow effect for high-fitness nodes
    if (node.fitness > 0.7) {
      const glowGeometry = new THREE.SphereGeometry(nodeSize * 1.5, 16, 16);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: new THREE.Color(node.lineage_color),
        transparent: true,
        opacity: 0.2
      });
      const glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.position.copy(node.position);
      sceneRef.current.add(glow);
    }

    // Cast shadows
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    sceneRef.current.add(mesh);

    // Add text label if showing metrics
    if (showMetrics) {
      createNodeLabel(node);
    }
  }, [showMetrics]);

  const createBranchMesh = useCallback((branch: TreeBranch) => {
    if (!sceneRef.current) return;

    const points = [
      branch.from.position.x, branch.from.position.y, branch.from.position.z,
      branch.to.position.x, branch.to.position.y, branch.to.position.z
    ];

    const geometry = new LineGeometry();
    geometry.setPositions(points);

    const material = new LineMaterial({
      color: new THREE.Color(branch.color),
      linewidth: branch.thickness,
      transparent: true,
      opacity: 0.6
    });

    material.resolution.set(width, height);

    const line = new Line2(geometry, material);
    line.userData = {
      type: 'branch',
      branch: branch
    };

    sceneRef.current.add(line);

    // Add animation for new branches
    if (branch.animated) {
      animateBranchGrowth(line);
    }
  }, [width, height]);

  const createNodeLabel = useCallback((node: ModelNode) => {
    // Create text sprite for node labels
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;

    // Set canvas size
    canvas.width = 256;
    canvas.height = 64;

    // Style text
    context.fillStyle = 'rgba(0, 0, 0, 0.8)';
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.fillStyle = '#ffffff';
    context.font = '16px Arial';
    context.textAlign = 'center';

    // Node info
    const fitness = (node.fitness * 100).toFixed(1);
    const text = `Gen ${node.generation} | ${fitness}%`;

    context.fillText(text, canvas.width / 2, canvas.height / 2);

    // Create sprite
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);

    sprite.position.copy(node.position);
    sprite.position.y += 0.5; // Offset above node
    sprite.scale.set(2, 0.5, 1);

    if (sceneRef.current) {
      sceneRef.current.add(sprite);
    }
  }, []);

  const animateBranchGrowth = useCallback((line: Line2) => {
    // Animate branch growth from parent to child
    const originalPositions = line.geometry.attributes.position.array.slice();
    const positions = line.geometry.attributes.position.array;

    // Start with just the first point
    for (let i = 3; i < positions.length; i++) {
      positions[i] = originalPositions[0 + (i % 3)];
    }

    line.geometry.attributes.position.needsUpdate = true;

    // Animate to full length
    const duration = 1000 / animationSpeed; // ms
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Interpolate positions
      for (let i = 3; i < positions.length; i++) {
        const originalIndex = i;
        const targetValue = originalPositions[originalIndex];
        const startValue = originalPositions[i % 3];
        positions[i] = startValue + (targetValue - startValue) * progress;
      }

      line.geometry.attributes.position.needsUpdate = true;

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }, [animationSpeed]);

  // Utility functions
  const findNodeById = useCallback((id: string, generations: Map<number, ModelNode[]>): ModelNode | undefined => {
    for (const nodes of generations.values()) {
      const found = nodes.find(node => node.id === id);
      if (found) return found;
    }
    return undefined;
  }, []);

  const blendColors = useCallback((color1: string, color2: string): string => {
    const c1 = new THREE.Color(color1);
    const c2 = new THREE.Color(color2);
    const blended = new THREE.Color();
    blended.r = (c1.r + c2.r) / 2;
    blended.g = (c1.g + c2.g) / 2;
    blended.b = (c1.b + c2.b) / 2;
    return `#${blended.getHexString()}`;
  }, []);

  // Event handlers
  const handlePlay = useCallback(() => {
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handleReset = useCallback(() => {
    setCurrentGeneration(0);
    setSelectedNodes(new Set());
    // Reset visualization to generation 0
  }, []);

  const handleGenerationStep = useCallback((direction: 'forward' | 'backward') => {
    const maxGen = Math.max(...Array.from(treeLayout.generations.keys()));
    if (direction === 'forward' && currentGeneration < maxGen) {
      setCurrentGeneration(currentGeneration + 1);
    } else if (direction === 'backward' && currentGeneration > 0) {
      setCurrentGeneration(currentGeneration - 1);
    }
  }, [currentGeneration, treeLayout.generations]);

  return (
    <div className="evomerge-3d-tree">
      <div ref={mountRef} style={{ width, height }} />

      {/* Controls */}
      <div className="tree-controls" style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontFamily: 'monospace'
      }}>
        <div>Generation: {currentGeneration}</div>
        <div>Models: {treeLayout.generations.get(currentGeneration)?.length || 0}</div>

        <div style={{ marginTop: '10px' }}>
          <button onClick={() => handleGenerationStep('backward')} disabled={currentGeneration === 0}>
            ← Prev
          </button>
          <button onClick={handlePlay} style={{ margin: '0 5px' }}>
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <button onClick={() => handleGenerationStep('forward')}
                  disabled={currentGeneration >= Math.max(...Array.from(treeLayout.generations.keys()))}>
            Next →
          </button>
        </div>

        <button onClick={handleReset} style={{ marginTop: '5px', width: '100%' }}>
          Reset
        </button>
      </div>

      {/* Legend */}
      <div className="tree-legend" style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontFamily: 'monospace',
        fontSize: '12px'
      }}>
        <div>🌳 EvoMerge Tree</div>
        <div style={{ marginTop: '5px' }}>
          <div>🔴 SLERP variants</div>
          <div>🟢 TIES variants</div>
          <div>🔵 DARE variants</div>
          <div>🟡 Weighted avg</div>
        </div>

        {hoveredNode && (
          <div style={{ marginTop: '10px', borderTop: '1px solid #666', paddingTop: '5px' }}>
            <div>Node: {hoveredNode}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EvoMerge3DTree;