/**
 * ParallelThoughtsViewer Component
 * 3D visualization of 4-stream parallel reasoning generation
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ThoughtStream } from '../types/phases';

interface ParallelThoughtsViewerProps {
  streams: ThoughtStream[];
  activeStreams: number;
  visualizationMode: '2d' | '3d';
}

interface StreamNode {
  x: number;
  y: number;
  z: number;
  velocity: { x: number; y: number; z: number };
  color: string;
  size: number;
  stream: ThoughtStream;
  pulsePhase: number;
}

export const ParallelThoughtsViewer: React.FC<ParallelThoughtsViewerProps> = ({
  streams,
  activeStreams,
  visualizationMode
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [nodes, setNodes] = useState<StreamNode[]>([]);
  const [cameraAngle, setCameraAngle] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [showConnections, setShowConnections] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  // Initialize stream nodes
  useEffect(() => {
    const newNodes: StreamNode[] = streams.map((stream, index) => {
      const angle = (index / streams.length) * Math.PI * 2;
      const radius = 150;

      return {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        z: (Math.random() - 0.5) * 100,
        velocity: {
          x: (Math.random() - 0.5) * 2,
          y: (Math.random() - 0.5) * 2,
          z: (Math.random() - 0.5) * 2
        },
        color: `hsl(${(index * 90) % 360}, 70%, 60%)`,
        size: 10 + stream.tokens.length * 0.1,
        stream,
        pulsePhase: Math.random() * Math.PI * 2
      };
    });

    setNodes(newNodes);
  }, [streams]);

  const project3D = useCallback((x: number, y: number, z: number, width: number, height: number) => {
    const distance = 500;
    const perspective = distance / (distance + z);

    // Apply camera rotation
    const cos_x = Math.cos(cameraAngle.x);
    const sin_x = Math.sin(cameraAngle.x);
    const cos_y = Math.cos(cameraAngle.y);
    const sin_y = Math.sin(cameraAngle.y);

    // Rotate around X axis
    const y1 = y * cos_x - z * sin_x;
    const z1 = y * sin_x + z * cos_x;

    // Rotate around Y axis
    const x2 = x * cos_y + z1 * sin_y;
    const z2 = -x * sin_y + z1 * cos_y;

    return {
      x: (x2 * perspective * zoom + width / 2),
      y: (y1 * perspective * zoom + height / 2),
      scale: perspective * zoom
    };
  }, [cameraAngle, zoom]);

  const calculateStreamInteraction = useCallback((stream1: ThoughtStream, stream2: ThoughtStream): number => {
    // Calculate interaction strength based on convergence scores and token similarity
    const convergenceDiff = Math.abs(stream1.convergenceScore - stream2.convergenceScore);
    const temperatureDiff = Math.abs(stream1.temperature - stream2.temperature);
    const lengthRatio = Math.min(stream1.tokens.length, stream2.tokens.length) /
                       Math.max(stream1.tokens.length, stream2.tokens.length || 1);

    return (1 - convergenceDiff) * (1 - temperatureDiff * 0.5) * lengthRatio;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      if (visualizationMode === '2d') {
        draw2D(ctx, width, height);
      } else {
        draw3D(ctx, width, height);
      }
    };

    const draw2D = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const centerX = width / 2;
      const centerY = height / 2;
      const time = Date.now() * 0.001;

      // Draw stream flows in 2D circular arrangement
      nodes.forEach((node, index) => {
        const angle = (index / nodes.length) * Math.PI * 2 + time * 0.1;
        const radius = 100 + Math.sin(time + index) * 20;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;

        // Draw stream path
        ctx.strokeStyle = node.color + '40';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();

        // Draw stream node
        const pulseSize = node.size + Math.sin(time * 3 + node.pulsePhase) * 3;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseSize);
        gradient.addColorStop(0, node.color);
        gradient.addColorStop(1, node.color + '00');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, pulseSize, 0, Math.PI * 2);
        ctx.fill();

        // Draw stream border
        ctx.strokeStyle = node.stream.isActive ? 'white' : 'rgba(255,255,255,0.5)';
        ctx.lineWidth = node.stream.isActive ? 2 : 1;
        ctx.beginPath();
        ctx.arc(x, y, node.size, 0, Math.PI * 2);
        ctx.stroke();

        // Draw stream info
        if (showMetrics) {
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`S${index}`, x, y - node.size - 10);
          ctx.fillText(`${node.stream.tokens.length}`, x, y + 3);
          ctx.fillText(`${(node.stream.convergenceScore * 100).toFixed(0)}%`, x, y + node.size + 15);
        }

        // Draw thought tokens as flowing particles
        node.stream.tokens.slice(-20).forEach((token, tokenIndex) => {
          const tokenAngle = angle + (tokenIndex / 20) * Math.PI * 0.5;
          const tokenRadius = radius - 30 + (tokenIndex / 20) * 60;
          const tokenX = centerX + Math.cos(tokenAngle) * tokenRadius;
          const tokenY = centerY + Math.sin(tokenAngle) * tokenRadius;

          ctx.fillStyle = node.color + Math.floor((tokenIndex / 20) * 255).toString(16).padStart(2, '0');
          ctx.beginPath();
          ctx.arc(tokenX, tokenY, 2, 0, Math.PI * 2);
          ctx.fill();
        });
      });

      // Draw stream connections
      if (showConnections) {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const interaction = calculateStreamInteraction(nodes[i].stream, nodes[j].stream);
            if (interaction > 0.3) {
              const angle1 = (i / nodes.length) * Math.PI * 2 + time * 0.1;
              const angle2 = (j / nodes.length) * Math.PI * 2 + time * 0.1;
              const radius1 = 100 + Math.sin(time + i) * 20;
              const radius2 = 100 + Math.sin(time + j) * 20;

              const x1 = centerX + Math.cos(angle1) * radius1;
              const y1 = centerY + Math.sin(angle1) * radius1;
              const x2 = centerX + Math.cos(angle2) * radius2;
              const y2 = centerY + Math.sin(angle2) * radius2;

              ctx.strokeStyle = `rgba(255, 255, 255, ${interaction * 0.5})`;
              ctx.lineWidth = interaction * 3;
              ctx.setLineDash([5, 5]);
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2, y2);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      }
    };

    const draw3D = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const time = Date.now() * 0.001;

      // Update node positions
      setNodes(prevNodes => prevNodes.map(node => {
        const stream = node.stream;

        // Apply physics-based movement
        const newVelocity = {
          x: node.velocity.x + (Math.random() - 0.5) * 0.1,
          y: node.velocity.y + (Math.random() - 0.5) * 0.1,
          z: node.velocity.z + (Math.random() - 0.5) * 0.1
        };

        // Apply convergence attraction (streams converge when they have similar patterns)
        let attractionX = 0, attractionY = 0, attractionZ = 0;
        prevNodes.forEach(otherNode => {
          if (otherNode !== node) {
            const interaction = calculateStreamInteraction(stream, otherNode.stream);
            if (interaction > 0.5) {
              const dx = otherNode.x - node.x;
              const dy = otherNode.y - node.y;
              const dz = otherNode.z - node.z;
              const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

              if (distance > 0) {
                const force = interaction * 0.1 / distance;
                attractionX += dx * force;
                attractionY += dy * force;
                attractionZ += dz * force;
              }
            }
          }
        });

        newVelocity.x += attractionX;
        newVelocity.y += attractionY;
        newVelocity.z += attractionZ;

        // Apply damping
        newVelocity.x *= 0.98;
        newVelocity.y *= 0.98;
        newVelocity.z *= 0.98;

        // Update position
        let newX = node.x + newVelocity.x;
        let newY = node.y + newVelocity.y;
        let newZ = node.z + newVelocity.z;

        // Boundary constraints
        const boundary = 200;
        if (Math.abs(newX) > boundary) newVelocity.x *= -0.8;
        if (Math.abs(newY) > boundary) newVelocity.y *= -0.8;
        if (Math.abs(newZ) > boundary) newVelocity.z *= -0.8;

        newX = Math.max(-boundary, Math.min(boundary, newX));
        newY = Math.max(-boundary, Math.min(boundary, newY));
        newZ = Math.max(-boundary, Math.min(boundary, newZ));

        return {
          ...node,
          x: newX,
          y: newY,
          z: newZ,
          velocity: newVelocity,
          size: 10 + stream.tokens.length * 0.1 + Math.sin(time * 2 + node.pulsePhase) * 2
        };
      }));

      // Sort nodes by z-depth for proper rendering
      const sortedNodes = [...nodes].sort((a, b) => b.z - a.z);

      // Draw connections first (behind nodes)
      if (showConnections) {
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const interaction = calculateStreamInteraction(nodes[i].stream, nodes[j].stream);
            if (interaction > 0.3) {
              const proj1 = project3D(nodes[i].x, nodes[i].y, nodes[i].z, width, height);
              const proj2 = project3D(nodes[j].x, nodes[j].y, nodes[j].z, width, height);

              ctx.strokeStyle = `rgba(255, 255, 255, ${interaction * proj1.scale * proj2.scale * 0.3})`;
              ctx.lineWidth = interaction * Math.min(proj1.scale, proj2.scale) * 3;
              ctx.setLineDash([5, 10]);
              ctx.beginPath();
              ctx.moveTo(proj1.x, proj1.y);
              ctx.lineTo(proj2.x, proj2.y);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      }

      // Draw nodes
      sortedNodes.forEach((node, index) => {
        const projected = project3D(node.x, node.y, node.z, width, height);
        const size = node.size * projected.scale;

        // Skip if behind camera or too small
        if (projected.scale <= 0 || size < 1) return;

        // Draw node shadow/depth effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.beginPath();
        ctx.arc(projected.x + 2, projected.y + 2, size, 0, Math.PI * 2);
        ctx.fill();

        // Draw node body
        const gradient = ctx.createRadialGradient(
          projected.x, projected.y, 0,
          projected.x, projected.y, size
        );
        gradient.addColorStop(0, node.color);
        gradient.addColorStop(0.7, node.color);
        gradient.addColorStop(1, node.color + '40');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
        ctx.fill();

        // Draw node border
        ctx.strokeStyle = node.stream.isActive ? 'white' : 'rgba(255,255,255,0.5)';
        ctx.lineWidth = (node.stream.isActive ? 2 : 1) * projected.scale;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
        ctx.stroke();

        // Draw selection highlight
        if (selectedNode === node.stream.id) {
          ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
          ctx.lineWidth = 3 * projected.scale;
          ctx.beginPath();
          ctx.arc(projected.x, projected.y, size + 5, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Draw metrics if enabled and node is large enough
        if (showMetrics && size > 8) {
          ctx.fillStyle = 'white';
          ctx.font = `${Math.floor(10 * projected.scale)}px monospace`;
          ctx.textAlign = 'center';
          ctx.fillText(`S${node.stream.id}`, projected.x, projected.y - size - 10);
          ctx.fillText(`${node.stream.tokens.length}`, projected.x, projected.y + 3);

          if (size > 12) {
            ctx.fillText(`${(node.stream.convergenceScore * 100).toFixed(0)}%`,
                        projected.x, projected.y + size + 15);
          }
        }

        // Draw thought particles orbiting the node
        const particleCount = Math.min(node.stream.tokens.length, 20);
        for (let p = 0; p < particleCount; p++) {
          const particleAngle = (p / particleCount) * Math.PI * 2 + time * 2;
          const particleRadius = size + 10 + Math.sin(time * 3 + p) * 5;
          const px = projected.x + Math.cos(particleAngle) * particleRadius;
          const py = projected.y + Math.sin(particleAngle) * particleRadius;

          ctx.fillStyle = node.color + Math.floor((1 - p/particleCount) * 255).toString(16).padStart(2, '0');
          ctx.beginPath();
          ctx.arc(px, py, 1.5 * projected.scale, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Draw 3D grid reference
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      for (let i = -200; i <= 200; i += 50) {
        const p1 = project3D(i, -200, 0, width, height);
        const p2 = project3D(i, 200, 0, width, height);
        const p3 = project3D(-200, i, 0, width, height);
        const p4 = project3D(200, i, 0, width, height);

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.moveTo(p3.x, p3.y);
        ctx.lineTo(p4.x, p4.y);
        ctx.stroke();
      }
    };

    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, visualizationMode, project3D, calculateStreamInteraction, showConnections, showMetrics, selectedNode, cameraAngle, zoom]);

  const handleMouseDown = (event: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = event.clientX - lastMousePos.x;
    const deltaY = event.clientY - lastMousePos.y;

    setCameraAngle(prev => ({
      x: prev.x + deltaY * 0.01,
      y: prev.y + deltaX * 0.01
    }));

    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.2, Math.min(3, prev * delta)));
  };

  const handleCanvasClick = (event: React.MouseEvent) => {
    if (isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Find clicked node
    let clickedNode = null;
    let minDistance = Infinity;

    nodes.forEach(node => {
      const projected = project3D(node.x, node.y, node.z, canvas.width, canvas.height);
      const distance = Math.sqrt((clickX - projected.x) ** 2 + (clickY - projected.y) ** 2);
      const nodeSize = node.size * projected.scale;

      if (distance <= nodeSize && distance < minDistance) {
        minDistance = distance;
        clickedNode = node.stream.id;
      }
    });

    setSelectedNode(clickedNode);
  };

  return (
    <div className="parallel-thoughts-viewer" data-testid="parallel-thoughts-viewer">
      <div className="viewer-header">
        <h3>Parallel Thought Streams ({activeStreams} active)</h3>
        <div className="header-controls">
          <label>
            <input
              type="checkbox"
              checked={showConnections}
              onChange={(e) => setShowConnections(e.target.checked)}
            />
            Show Connections
          </label>

          <label>
            <input
              type="checkbox"
              checked={showMetrics}
              onChange={(e) => setShowMetrics(e.target.checked)}
            />
            Show Metrics
          </label>

          <button
            className="reset-view"
            onClick={() => {
              setCameraAngle({ x: 0, y: 0 });
              setZoom(1);
              setSelectedNode(null);
            }}
          >
            Reset View
          </button>
        </div>
      </div>

      <div className="viewer-content">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          onClick={handleCanvasClick}
          className="thoughts-canvas"
        />

        {visualizationMode === '3d' && (
          <div className="controls-overlay">
            <div className="control-hint">
              🖱️ Drag to rotate • 🔍 Scroll to zoom • 👆 Click nodes to select
            </div>
            <div className="camera-info">
              Camera: ({cameraAngle.x.toFixed(2)}, {cameraAngle.y.toFixed(2)}) | Zoom: {zoom.toFixed(2)}x
            </div>
          </div>
        )}
      </div>

      {selectedNode !== null && (
        <div className="selected-stream-info" data-testid="selected-stream-info">
          {(() => {
            const stream = streams.find(s => s.id === selectedNode);
            if (!stream) return null;

            return (
              <div className="stream-details">
                <h4>Stream {selectedNode} Details</h4>
                <div className="detail-grid">
                  <div><strong>Status:</strong> {stream.isActive ? 'Active' : 'Inactive'}</div>
                  <div><strong>Tokens:</strong> {stream.tokens.length}</div>
                  <div><strong>Temperature:</strong> {stream.temperature.toFixed(3)}</div>
                  <div><strong>Convergence:</strong> {(stream.convergenceScore * 100).toFixed(1)}%</div>
                  <div><strong>Length:</strong> {stream.length}</div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      <style>{`
        .parallel-thoughts-viewer {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 1.5rem;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .viewer-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .viewer-header h3 {
          margin: 0;
          font-size: 1.5rem;
          font-weight: 500;
          color: white;
        }

        .header-controls {
          display: flex;
          align-items: center;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .header-controls label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
        }

        .reset-view {
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(5px);
        }

        .reset-view:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .viewer-content {
          position: relative;
        }

        .thoughts-canvas {
          background: rgba(0, 0, 0, 0.3);
          border-radius: 8px;
          cursor: grab;
          width: 100%;
          height: 500px;
        }

        .thoughts-canvas:active {
          cursor: grabbing;
        }

        .controls-overlay {
          position: absolute;
          bottom: 1rem;
          left: 1rem;
          right: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: rgba(0, 0, 0, 0.7);
          border-radius: 6px;
          padding: 0.5rem 1rem;
          color: white;
          font-size: 0.8rem;
          backdrop-filter: blur(10px);
        }

        .control-hint {
          opacity: 0.8;
        }

        .camera-info {
          font-family: monospace;
          opacity: 0.6;
        }

        .selected-stream-info {
          margin-top: 1.5rem;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .stream-details h4 {
          margin: 0 0 1rem 0;
          color: white;
          font-size: 1.1rem;
        }

        .detail-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 0.5rem;
          color: white;
          font-size: 0.9rem;
        }

        @media (max-width: 768px) {
          .viewer-header {
            flex-direction: column;
            align-items: stretch;
          }

          .header-controls {
            justify-content: space-between;
          }

          .controls-overlay {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
          }

          .detail-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};