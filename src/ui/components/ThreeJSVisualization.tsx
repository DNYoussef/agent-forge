/**
 * ThreeJSVisualization Component
 * Advanced 3D visualization for thought flows using Three.js
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ThoughtStream, ReasoningToken } from '../types/phases';

// Three.js types (simplified - would normally import from 'three')
interface Vector3 {
  x: number;
  y: number;
  z: number;
}

interface ThreeJSScene {
  scene: any;
  camera: any;
  renderer: any;
  controls: any;
}

interface ThoughtParticle {
  position: Vector3;
  velocity: Vector3;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  token: ReasoningToken;
}

interface StreamFlow {
  id: number;
  particles: ThoughtParticle[];
  path: Vector3[];
  color: string;
  intensity: number;
}

interface ThreeJSVisualizationProps {
  streams: ThoughtStream[];
  width?: number;
  height?: number;
  interactive?: boolean;
  showPaths?: boolean;
  particleCount?: number;
  animationSpeed?: number;
}

export const ThreeJSVisualization: React.FC<ThreeJSVisualizationProps> = ({
  streams,
  width = 800,
  height = 600,
  interactive = true,
  showPaths = true,
  particleCount = 1000,
  animationSpeed = 1.0
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ThreeJSScene | null>(null);
  const animationRef = useRef<number>();
  const [isLoaded, setIsLoaded] = useState(false);
  const [streamFlows, setStreamFlows] = useState<StreamFlow[]>([]);
  const [cameraMode, setCameraMode] = useState<'orbit' | 'follow' | 'free'>('orbit');
  const [selectedStream, setSelectedStream] = useState<number | null>(null);

  // Initialize Three.js scene (using Canvas API as fallback)
  useEffect(() => {
    if (!mountRef.current) return;

    // For this demo, we'll use Canvas API to simulate Three.js functionality
    // In a real implementation, you would use actual Three.js
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvas.style.cursor = interactive ? 'grab' : 'default';

    mountRef.current.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Mock Three.js scene setup
    sceneRef.current = {
      scene: { children: [] },
      camera: { position: { x: 0, y: 0, z: 500 }, rotation: { x: 0, y: 0, z: 0 } },
      renderer: { domElement: canvas },
      controls: null
    };

    setIsLoaded(true);

    // Setup mouse controls for interaction
    if (interactive) {
      setupMouseControls(canvas);
    }

    return () => {
      if (mountRef.current && canvas) {
        mountRef.current.removeChild(canvas);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [width, height, interactive]);

  // Initialize stream flows
  useEffect(() => {
    if (!isLoaded) return;

    const flows: StreamFlow[] = streams.map((stream, index) => {
      const angle = (index / streams.length) * Math.PI * 2;
      const radius = 200;

      return {
        id: stream.id,
        particles: [],
        path: generateStreamPath(angle, radius, stream.tokens.length),
        color: `hsl(${(index * 60) % 360}, 70%, 60%)`,
        intensity: stream.isActive ? 1.0 : 0.3
      };
    });

    setStreamFlows(flows);
  }, [streams, isLoaded]);

  // Animation loop
  useEffect(() => {
    if (!isLoaded || !sceneRef.current) return;

    const animate = () => {
      updateParticles();
      render();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isLoaded, streamFlows, animationSpeed]);

  const generateStreamPath = (angle: number, radius: number, tokenCount: number): Vector3[] => {
    const path: Vector3[] = [];
    const steps = Math.max(20, tokenCount);

    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const spiralAngle = angle + t * Math.PI * 4;
      const spiralRadius = radius * (0.5 + t * 0.5);
      const height = Math.sin(t * Math.PI * 2) * 100;

      path.push({
        x: Math.cos(spiralAngle) * spiralRadius,
        y: Math.sin(spiralAngle) * spiralRadius,
        z: height + t * 200 - 100
      });
    }

    return path;
  };

  const createParticleFromToken = (token: ReasoningToken, streamFlow: StreamFlow): ThoughtParticle => {
    const pathIndex = Math.floor((token.position / 100) * streamFlow.path.length);
    const pathPoint = streamFlow.path[pathIndex] || streamFlow.path[0];

    const sizeMultiplier = {
      thought_start: 2.0,
      thought_end: 2.0,
      thought_content: 1.5,
      regular: 1.0
    };

    const colorModifier = {
      thought_start: '#00ff00',
      thought_end: '#ff0000',
      thought_content: '#0066ff',
      regular: streamFlow.color
    };

    return {
      position: {
        x: pathPoint.x + (Math.random() - 0.5) * 20,
        y: pathPoint.y + (Math.random() - 0.5) * 20,
        z: pathPoint.z + (Math.random() - 0.5) * 20
      },
      velocity: {
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 2,
        z: Math.random() * 3
      },
      life: 0,
      maxLife: 300 + Math.random() * 200,
      color: colorModifier[token.type],
      size: 3 * sizeMultiplier[token.type] * token.confidence,
      token
    };
  };

  const updateParticles = useCallback(() => {
    setStreamFlows(prev => prev.map(flow => {
      // Add new particles from recent tokens
      const stream = streams.find(s => s.id === flow.id);
      if (stream && stream.tokens.length > flow.particles.length) {
        const newTokens = stream.tokens.slice(flow.particles.length);
        const newParticles = newTokens.map(token => createParticleFromToken(token, flow));
        flow.particles.push(...newParticles);
      }

      // Update existing particles
      flow.particles = flow.particles
        .map(particle => ({
          ...particle,
          position: {
            x: particle.position.x + particle.velocity.x * animationSpeed,
            y: particle.position.y + particle.velocity.y * animationSpeed,
            z: particle.position.z + particle.velocity.z * animationSpeed
          },
          velocity: {
            x: particle.velocity.x * 0.99,
            y: particle.velocity.y * 0.99,
            z: particle.velocity.z * 0.98
          },
          life: particle.life + animationSpeed
        }))
        .filter(particle => particle.life < particle.maxLife);

      // Limit particle count
      if (flow.particles.length > particleCount / streams.length) {
        flow.particles = flow.particles.slice(-Math.floor(particleCount / streams.length));
      }

      return flow;
    }));
  }, [streams, animationSpeed, particleCount]);

  const project3D = (point: Vector3, camera: any, canvas: HTMLCanvasElement): { x: number; y: number; scale: number } => {
    // Simple 3D to 2D projection
    const distance = 500;
    const perspective = distance / (distance + point.z - camera.position.z);

    return {
      x: (point.x - camera.position.x) * perspective + canvas.width / 2,
      y: (point.y - camera.position.y) * perspective + canvas.height / 2,
      scale: perspective
    };
  };

  const render = useCallback(() => {
    if (!sceneRef.current) return;

    const canvas = sceneRef.current.renderer.domElement as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const camera = sceneRef.current.camera;
    const time = Date.now() * 0.001;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background gradient
    const gradient = ctx.createRadialGradient(
      canvas.width / 2, canvas.height / 2, 0,
      canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 2
    );
    gradient.addColorStop(0, 'rgba(20, 20, 40, 1)');
    gradient.addColorStop(1, 'rgba(5, 5, 15, 1)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Update camera for orbit mode
    if (cameraMode === 'orbit') {
      camera.position.x = Math.cos(time * 0.1) * 300;
      camera.position.z = Math.sin(time * 0.1) * 300 + 200;
      camera.rotation.y = time * 0.1;
    }

    // Draw stream paths
    if (showPaths) {
      streamFlows.forEach(flow => {
        if (flow.path.length < 2) return;

        ctx.strokeStyle = flow.color + '40';
        ctx.lineWidth = 2 * flow.intensity;
        ctx.beginPath();

        const firstPoint = project3D(flow.path[0], camera, canvas);
        ctx.moveTo(firstPoint.x, firstPoint.y);

        for (let i = 1; i < flow.path.length; i++) {
          const point = project3D(flow.path[i], camera, canvas);
          ctx.lineTo(point.x, point.y);
        }

        ctx.stroke();
      });
    }

    // Draw particles
    const allParticles: { particle: ThoughtParticle; flow: StreamFlow; projected: { x: number; y: number; scale: number } }[] = [];

    streamFlows.forEach(flow => {
      flow.particles.forEach(particle => {
        const projected = project3D(particle.position, camera, canvas);
        allParticles.push({ particle, flow, projected });
      });
    });

    // Sort by depth for proper rendering
    allParticles.sort((a, b) => b.particle.position.z - a.particle.position.z);

    // Render particles
    allParticles.forEach(({ particle, flow, projected }) => {
      if (projected.scale <= 0) return;

      const size = particle.size * projected.scale;
      if (size < 0.5) return;

      const alpha = Math.max(0, (particle.maxLife - particle.life) / particle.maxLife) * flow.intensity;
      const color = particle.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');

      // Particle glow effect
      const gradient = ctx.createRadialGradient(
        projected.x, projected.y, 0,
        projected.x, projected.y, size * 2
      );
      gradient.addColorStop(0, color);
      gradient.addColorStop(0.5, color + '80');
      gradient.addColorStop(1, color + '00');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, size * 2, 0, Math.PI * 2);
      ctx.fill();

      // Particle core
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
      ctx.fill();

      // Special highlighting for thought tokens
      if (particle.token.type === 'thought_start' || particle.token.type === 'thought_end') {
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size + 2, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    // Draw stream info
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    streamFlows.forEach((flow, index) => {
      const y = 20 + index * 60;
      const stream = streams.find(s => s.id === flow.id);

      if (stream) {
        ctx.fillStyle = flow.color;
        ctx.fillRect(10, y - 15, 20, 20);

        ctx.fillStyle = 'white';
        ctx.fillText(`Stream ${flow.id}`, 40, y);
        ctx.fillText(`Tokens: ${stream.tokens.length}`, 40, y + 15);
        ctx.fillText(`Active: ${stream.isActive}`, 40, y + 30);
        ctx.fillText(`Particles: ${flow.particles.length}`, 40, y + 45);
      }
    });

    // Camera info
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '10px monospace';
    ctx.fillText(`Camera: ${cameraMode}`, canvas.width - 120, 20);
    ctx.fillText(`Pos: (${camera.position.x.toFixed(0)}, ${camera.position.y.toFixed(0)}, ${camera.position.z.toFixed(0)})`, canvas.width - 120, 35);
    ctx.fillText(`Particles: ${allParticles.length}`, canvas.width - 120, 50);
  }, [streamFlows, streams, showPaths, cameraMode]);

  const setupMouseControls = (canvas: HTMLCanvasElement) => {
    let isDragging = false;
    let lastMousePos = { x: 0, y: 0 };

    canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      lastMousePos = { x: e.clientX, y: e.clientY };
      canvas.style.cursor = 'grabbing';
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!isDragging || !sceneRef.current) return;

      const deltaX = e.clientX - lastMousePos.x;
      const deltaY = e.clientY - lastMousePos.y;

      if (cameraMode === 'free') {
        sceneRef.current.camera.rotation.y += deltaX * 0.01;
        sceneRef.current.camera.rotation.x += deltaY * 0.01;
      }

      lastMousePos = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mouseup', () => {
      isDragging = false;
      canvas.style.cursor = 'grab';
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (!sceneRef.current) return;

      const delta = e.deltaY > 0 ? 1.1 : 0.9;
      sceneRef.current.camera.position.z *= delta;
      sceneRef.current.camera.position.z = Math.max(100, Math.min(1000, sceneRef.current.camera.position.z));
    });

    canvas.addEventListener('click', (e) => {
      // Ray casting simulation for particle selection
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Find closest particle to click
      let closestDistance = Infinity;
      let closestStream = null;

      streamFlows.forEach(flow => {
        flow.particles.forEach(particle => {
          if (!sceneRef.current) return;
          const projected = project3D(particle.position, sceneRef.current.camera, canvas);
          const distance = Math.sqrt((mouseX - projected.x) ** 2 + (mouseY - projected.y) ** 2);

          if (distance < 20 && distance < closestDistance) {
            closestDistance = distance;
            closestStream = flow.id;
          }
        });
      });

      setSelectedStream(closestStream);
    });
  };

  return (
    <div className="threejs-visualization" data-testid="threejs-visualization">
      <div className="visualization-controls">
        <div className="camera-controls">
          <label>Camera Mode:</label>
          <select value={cameraMode} onChange={(e) => setCameraMode(e.target.value as any)}>
            <option value="orbit">Orbit</option>
            <option value="follow">Follow</option>
            <option value="free">Free</option>
          </select>
        </div>

        <label>
          <input
            type="checkbox"
            checked={showPaths}
            onChange={(e) => setShowPaths(e.target.checked)}
          />
          Show Paths
        </label>

        <div className="particle-count">
          <label>Particles:</label>
          <input
            type="range"
            min="100"
            max="2000"
            value={particleCount}
            onChange={(e) => setParticleCount(parseInt(e.target.value))}
          />
          <span>{particleCount}</span>
        </div>

        <div className="animation-speed">
          <label>Speed:</label>
          <input
            type="range"
            min="0.1"
            max="3.0"
            step="0.1"
            value={animationSpeed}
            onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
          />
          <span>{animationSpeed}x</span>
        </div>
      </div>

      <div
        ref={mountRef}
        className="threejs-container"
        style={{ width, height }}
      />

      {selectedStream !== null && (
        <div className="selection-info">
          <h4>Selected Stream {selectedStream}</h4>
          <div className="stream-details">
            {(() => {
              const stream = streams.find(s => s.id === selectedStream);
              const flow = streamFlows.find(f => f.id === selectedStream);

              if (!stream || !flow) return null;

              return (
                <>
                  <div>Tokens: {stream.tokens.length}</div>
                  <div>Particles: {flow.particles.length}</div>
                  <div>Active: {stream.isActive ? 'Yes' : 'No'}</div>
                  <div>Temperature: {stream.temperature.toFixed(3)}</div>
                  <div>Convergence: {(stream.convergenceScore * 100).toFixed(1)}%</div>
                </>
              );
            })()}
          </div>
        </div>
      )}

      <style>{`
        .threejs-visualization {
          position: relative;
          background: #000;
          border-radius: 8px;
          overflow: hidden;
        }

        .visualization-controls {
          position: absolute;
          top: 10px;
          left: 10px;
          z-index: 10;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          background: rgba(0, 0, 0, 0.7);
          padding: 1rem;
          border-radius: 6px;
          color: white;
          font-size: 0.8rem;
        }

        .camera-controls {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .camera-controls select {
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 4px;
          color: white;
          padding: 0.25rem;
        }

        .visualization-controls label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
        }

        .particle-count, .animation-speed {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .particle-count input, .animation-speed input {
          width: 80px;
        }

        .threejs-container {
          position: relative;
          overflow: hidden;
        }

        .threejs-container canvas {
          display: block;
          width: 100%;
          height: 100%;
        }

        .selection-info {
          position: absolute;
          bottom: 10px;
          right: 10px;
          background: rgba(0, 0, 0, 0.8);
          padding: 1rem;
          border-radius: 6px;
          color: white;
          min-width: 200px;
        }

        .selection-info h4 {
          margin: 0 0 0.5rem 0;
          color: #3498db;
        }

        .stream-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          font-size: 0.9rem;
        }

        @media (max-width: 768px) {
          .visualization-controls {
            position: relative;
            flex-direction: row;
            flex-wrap: wrap;
            top: 0;
            left: 0;
            margin-bottom: 1rem;
          }

          .selection-info {
            position: relative;
            bottom: auto;
            right: auto;
            margin-top: 1rem;
          }
        }
      `}</style>
    </div>
  );
};

export default ThreeJSVisualization;