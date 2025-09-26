'use client';

import { useState, useEffect, useRef } from 'react';
import { Eye, Maximize2, Minimize2, RotateCcw, Play, Pause, Settings } from 'lucide-react';

interface WeightSpaceVisualizationProps {
  config: any;
  metrics: any;
  mode: '2D' | '3D';
  showGrokTransitions: boolean;
}

interface WeightNode {
  id: string;
  x: number;
  y: number;
  z?: number;
  magnitude: number;
  gradient: number;
  layer: string;
  grokking: boolean;
}

export default function WeightSpaceVisualization({ config, metrics, mode, showGrokTransitions }: WeightSpaceVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [weightNodes, setWeightNodes] = useState<WeightNode[]>([]);
  const [rotationX, setRotationX] = useState(0);
  const [rotationY, setRotationY] = useState(0);
  const [zoom, setZoom] = useState(1);

  // Initialize weight nodes
  useEffect(() => {
    const nodes: WeightNode[] = [];
    const layers = ['input', 'hidden1', 'hidden2', 'output'];

    for (let i = 0; i < 200; i++) {
      nodes.push({
        id: `node_${i}`,
        x: (Math.random() - 0.5) * 400,
        y: (Math.random() - 0.5) * 400,
        z: mode === '3D' ? (Math.random() - 0.5) * 400 : 0,
        magnitude: Math.random(),
        gradient: (Math.random() - 0.5) * 0.1,
        layer: layers[Math.floor(Math.random() * layers.length)],
        grokking: Math.random() > 0.95
      });
    }
    setWeightNodes(nodes);
  }, [mode]);

  // Animation loop
  useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setWeightNodes(prev => prev.map(node => ({
        ...node,
        magnitude: Math.max(0, Math.min(1, node.magnitude + node.gradient + (Math.random() - 0.5) * 0.05)),
        gradient: node.gradient + (Math.random() - 0.5) * 0.02,
        grokking: showGrokTransitions ? (Math.random() > 0.98 ? !node.grokking : node.grokking) : false
      })));

      if (mode === '3D') {
        setRotationX(prev => prev + 0.5);
        setRotationY(prev => prev + 0.3);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [isAnimating, mode, showGrokTransitions]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw weight nodes
    weightNodes.forEach(node => {
      let screenX = node.x * zoom + centerX;
      let screenY = node.y * zoom + centerY;

      // 3D projection if in 3D mode
      if (mode === '3D' && node.z !== undefined) {
        const rotXRad = (rotationX * Math.PI) / 180;
        const rotYRad = (rotationY * Math.PI) / 180;

        // Rotate around X axis
        const tempY = node.y * Math.cos(rotXRad) - node.z * Math.sin(rotXRad);
        const tempZ = node.y * Math.sin(rotXRad) + node.z * Math.cos(rotXRad);

        // Rotate around Y axis
        const finalX = node.x * Math.cos(rotYRad) + tempZ * Math.sin(rotYRad);
        const finalZ = -node.x * Math.sin(rotYRad) + tempZ * Math.cos(rotYRad);

        // Perspective projection
        const perspective = 500;
        const projectedX = (finalX * perspective) / (perspective + finalZ + 200);
        const projectedY = (tempY * perspective) / (perspective + finalZ + 200);

        screenX = projectedX * zoom + centerX;
        screenY = projectedY * zoom + centerY;
      }

      // Skip if outside canvas
      if (screenX < 0 || screenX > rect.width || screenY < 0 || screenY > rect.height) return;

      // Node color based on layer and magnitude
      const layerColors: {[key: string]: string} = {
        'input': '#3B82F6',
        'hidden1': '#8B5CF6',
        'hidden2': '#EC4899',
        'output': '#F59E0B'
      };

      const baseColor = layerColors[node.layer] || '#6B7280';
      const alpha = node.magnitude;

      // Grokking effect
      if (node.grokking && showGrokTransitions) {
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(screenX, screenY, 8, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Draw node
      ctx.fillStyle = baseColor.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
      ctx.beginPath();
      ctx.arc(screenX, screenY, Math.max(1, node.magnitude * 4), 0, 2 * Math.PI);
      ctx.fill();

      // Gradient flow visualization
      if (Math.abs(node.gradient) > 0.05) {
        ctx.strokeStyle = node.gradient > 0 ? '#10B981' : '#EF4444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(screenX, screenY);
        ctx.lineTo(screenX + node.gradient * 50, screenY);
        ctx.stroke();
      }
    });

    // Draw grid if 2D
    if (mode === '2D') {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      for (let x = 0; x < rect.width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, rect.height);
        ctx.stroke();
      }
      for (let y = 0; y < rect.height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(rect.width, y);
        ctx.stroke();
      }
    }

  }, [weightNodes, mode, rotationX, rotationY, zoom, showGrokTransitions]);

  const renderControls = () => (
    <div className="absolute top-2 right-2 flex gap-2">
      <button
        onClick={() => setIsAnimating(!isAnimating)}
        className="p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-all"
        title={isAnimating ? 'Pause' : 'Play'}
      >
        {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
      </button>

      <button
        onClick={() => {
          setRotationX(0);
          setRotationY(0);
          setZoom(1);
        }}
        className="p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-all"
        title="Reset View"
      >
        <RotateCcw className="w-4 h-4" />
      </button>

      <button
        onClick={() => setIsFullscreen(!isFullscreen)}
        className="p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-all"
        title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
      >
        {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
      </button>
    </div>
  );

  const renderLegend = () => (
    <div className="absolute bottom-2 left-2 bg-black/70 rounded-lg p-3 text-xs">
      <div className="text-white font-semibold mb-2">Weight Space Legend</div>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-400"></div>
          <span className="text-gray-300">Input Layer</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-400"></div>
          <span className="text-gray-300">Hidden Layers</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
          <span className="text-gray-300">Output Layer</span>
        </div>
        {showGrokTransitions && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full border-2 border-green-400"></div>
            <span className="text-gray-300">Grokking</span>
          </div>
        )}
      </div>
    </div>
  );

  const renderStats = () => (
    <div className="absolute top-2 left-2 bg-black/70 rounded-lg p-3 text-xs">
      <div className="text-white font-semibold mb-2">Proprioception Stats</div>
      <div className="space-y-1 text-gray-300">
        <div>Nodes: {weightNodes.length}</div>
        <div>Mode: {mode}</div>
        <div>Zoom: {zoom.toFixed(1)}x</div>
        {mode === '3D' && (
          <>
            <div>Rotation X: {rotationX.toFixed(1)}°</div>
            <div>Rotation Y: {rotationY.toFixed(1)}°</div>
          </>
        )}
        <div>Grokking: {weightNodes.filter(n => n.grokking).length}</div>
      </div>
    </div>
  );

  const handleMouseWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.1, Math.min(3, prev - e.deltaY * 0.001)));
  };

  return (
    <div className={`bg-white/5 backdrop-blur-lg rounded-xl border border-white/10 ${
      isFullscreen ? 'fixed inset-4 z-50' : 'h-64'
    } relative overflow-hidden`}>
      <div className="absolute top-2 left-1/2 transform -translate-x-1/2 z-10">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <Eye className="w-4 h-4 text-cyan-400" />
          Weight Space Proprioception ({mode})
        </h3>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-move"
        onWheel={handleMouseWheel}
        onMouseMove={(e) => {
          if (e.buttons === 1 && mode === '3D') {
            setRotationX(prev => prev + e.movementY * 0.5);
            setRotationY(prev => prev + e.movementX * 0.5);
          }
        }}
      />

      {renderControls()}
      {renderLegend()}
      {renderStats()}

      {/* Zoom controls */}
      <div className="absolute bottom-2 right-2 flex flex-col gap-1">
        <button
          onClick={() => setZoom(prev => Math.min(3, prev + 0.2))}
          className="w-8 h-8 bg-black/50 hover:bg-black/70 rounded text-white text-lg font-bold transition-all"
        >
          +
        </button>
        <button
          onClick={() => setZoom(prev => Math.max(0.1, prev - 0.2))}
          className="w-8 h-8 bg-black/50 hover:bg-black/70 rounded text-white text-lg font-bold transition-all"
        >
          -
        </button>
      </div>
    </div>
  );
}