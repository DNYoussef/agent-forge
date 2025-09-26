'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Eye, Maximize2, Minimize2, Play, Pause, RotateCcw, Layers } from 'lucide-react';

interface WeightSpaceSphereProps {
  config: any;
  currentLevel: number;
}

export default function WeightSpaceSphere({ config, currentLevel }: WeightSpaceSphereProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const sphereRef = useRef<THREE.Mesh>();
  const particlesRef = useRef<THREE.Points>();
  const controlsRef = useRef<OrbitControls>();
  const animationIdRef = useRef<number>();

  const [isAnimating, setIsAnimating] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showParticles, setShowParticles] = useState(true);
  const [grokIntensity, setGrokIntensity] = useState(0);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.FogExp2(0x0a0a0a, 0.001);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 150);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 50;
    controls.maxDistance = 300;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 0.5).normalize();
    scene.add(directionalLight);

    // Create rippling sphere with custom shader
    const sphereGeometry = new THREE.SphereGeometry(40, 64, 64);

    const shaderMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        level: { value: currentLevel || 1 },
        grokking: { value: 0 },
        edgeChaos: { value: 0.75 },
        dreamMode: { value: 0 }
      },
      vertexShader: `
        uniform float time;
        uniform float level;
        uniform float grokking;
        uniform float edgeChaos;

        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vDistortion;

        void main() {
          vUv = uv;
          vNormal = normal;
          vPosition = position;

          vec3 pos = position;

          // Multi-frequency ripples based on training level
          float frequency1 = 2.0 + level * 0.5;
          float frequency2 = 3.5 + level * 0.3;
          float frequency3 = 5.0 + level * 0.2;

          // Complex wave interference for edge-of-chaos effect
          float wave1 = sin(frequency1 * position.x + time * 2.0) * cos(frequency1 * position.y + time);
          float wave2 = sin(frequency2 * position.y + time * 1.5) * cos(frequency2 * position.z + time * 0.8);
          float wave3 = sin(frequency3 * position.z + time) * cos(frequency3 * position.x + time * 1.2);

          // Combine waves with edge-of-chaos modulation
          float amplitude = 0.1 * (1.0 + grokking * 2.0) * edgeChaos;
          float distortion = (wave1 + wave2 * 0.7 + wave3 * 0.5) * amplitude;

          // Apply distortion along normal
          pos += normal * distortion;

          // Pulsing effect based on grokking
          float pulse = 1.0 + grokking * 0.1 * sin(time * 4.0);
          pos *= pulse;

          vDistortion = distortion;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float level;
        uniform float grokking;
        uniform float dreamMode;

        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vDistortion;

        void main() {
          // Dynamic color based on training state
          vec3 baseColor = vec3(0.2, 0.6, 1.0); // Blue base
          vec3 grokColor = vec3(0.4, 1.0, 0.6); // Green for grokking
          vec3 chaosColor = vec3(1.0, 0.4, 0.8); // Pink for edge-of-chaos
          vec3 dreamColor = vec3(0.8, 0.6, 1.0); // Purple for dream mode

          // Mix colors based on state
          vec3 color = baseColor;
          color = mix(color, grokColor, grokking);
          color = mix(color, chaosColor, abs(vDistortion) * 2.0);
          color = mix(color, dreamColor, dreamMode);

          // Add shimmer based on normal
          float shimmer = dot(normalize(vNormal), vec3(0.0, 0.0, 1.0));
          shimmer = pow(shimmer, 2.0);
          color += shimmer * 0.3;

          // Pulsing glow
          float glow = sin(time * 2.0 + vUv.x * 10.0) * 0.1 + 0.9;
          color *= glow;

          // Level-based intensity
          float intensity = 0.5 + (float(level) / 10.0) * 0.5;
          color *= intensity;

          gl_FragColor = vec4(color, 0.9);
        }
      `,
      transparent: true,
      wireframe: false,
      side: THREE.DoubleSide
    });

    const sphere = new THREE.Mesh(sphereGeometry, shaderMaterial);
    scene.add(sphere);
    sphereRef.current = sphere;

    // Add particle system for weight connections
    const particleCount = 1000;
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const radius = 45 + Math.random() * 60;

      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);

      // Color based on layer (input=blue, hidden=purple, output=orange)
      const layerType = Math.random();
      if (layerType < 0.33) {
        colors[i * 3] = 0.2;
        colors[i * 3 + 1] = 0.6;
        colors[i * 3 + 2] = 1.0;
      } else if (layerType < 0.66) {
        colors[i * 3] = 0.6;
        colors[i * 3 + 1] = 0.4;
        colors[i * 3 + 2] = 0.9;
      } else {
        colors[i * 3] = 1.0;
        colors[i * 3 + 1] = 0.6;
        colors[i * 3 + 2] = 0.2;
      }
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
      size: 2,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);
    particlesRef.current = particles;

    // Add connection lines
    const lineCount = 200;
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(lineCount * 6);

    for (let i = 0; i < lineCount; i++) {
      const idx1 = Math.floor(Math.random() * particleCount);
      const idx2 = Math.floor(Math.random() * particleCount);

      linePositions[i * 6] = positions[idx1 * 3];
      linePositions[i * 6 + 1] = positions[idx1 * 3 + 1];
      linePositions[i * 6 + 2] = positions[idx1 * 3 + 2];
      linePositions[i * 6 + 3] = positions[idx2 * 3];
      linePositions[i * 6 + 4] = positions[idx2 * 3 + 1];
      linePositions[i * 6 + 5] = positions[idx2 * 3 + 2];
    }

    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x4466ff,
      transparent: true,
      opacity: 0.1,
      blending: THREE.AdditiveBlending
    });

    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      const time = Date.now() * 0.001;

      // Update shader uniforms
      if (sphereRef.current) {
        const material = sphereRef.current.material as THREE.ShaderMaterial;
        material.uniforms.time.value = time;
        material.uniforms.level.value = currentLevel;
        material.uniforms.grokking.value = grokIntensity;
        material.uniforms.edgeChaos.value = 0.55 + Math.sin(time * 0.5) * 0.2;
        material.uniforms.dreamMode.value = config?.sleepMode ? 1.0 : 0.0;
      }

      // Rotate particles
      if (particlesRef.current && showParticles) {
        particlesRef.current.rotation.y += 0.001;
        particlesRef.current.rotation.x += 0.0005;
      }

      // Update controls
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      // Render
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    if (isAnimating) {
      animate();
    }

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

      cameraRef.current.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && mountRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
  }, [isAnimating, showParticles, currentLevel, config?.sleepMode]);

  // Update grok intensity based on metrics
  useEffect(() => {
    if (config?.metrics?.grokfast_active) {
      setGrokIntensity(Math.min(1, config.metrics.grokfast_lambda || 0));
    }
  }, [config?.metrics]);

  return (
    <div className={`bg-gradient-to-br from-gray-900 via-purple-900/20 to-black rounded-xl border border-purple-500/20 ${
      isFullscreen ? 'fixed inset-4 z-50' : 'h-96'
    } relative overflow-hidden`}>
      <div className="absolute top-4 left-4 z-10">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Eye className="w-5 h-5 text-cyan-400" />
          Weight Space Proprioception
        </h3>
        <p className="text-xs text-gray-400 mt-1">Level {currentLevel}/10 • Edge of Chaos: {(0.55 + Math.sin(Date.now() * 0.0005) * 0.2).toFixed(2)}</p>
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-2 z-10">
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className="p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-all"
          title={isAnimating ? 'Pause' : 'Play'}
        >
          {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>

        <button
          onClick={() => setShowParticles(!showParticles)}
          className="p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-all"
          title="Toggle Particles"
        >
          <Layers className="w-4 h-4" />
        </button>

        <button
          onClick={() => {
            if (controlsRef.current) {
              controlsRef.current.reset();
            }
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

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-xs z-10">
        <div className="text-white font-semibold mb-2">Neural Weight States</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-400"></div>
            <span className="text-gray-300">Base Weights</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-400"></div>
            <span className="text-gray-300">Grokking Active</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-pink-400"></div>
            <span className="text-gray-300">Edge of Chaos</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-400"></div>
            <span className="text-gray-300">Dream Mode</span>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="absolute bottom-4 right-4 bg-black/70 rounded-lg p-3 text-xs z-10">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-300">
          <div>Particles:</div>
          <div className="text-white">{showParticles ? '1000' : 'Hidden'}</div>
          <div>Connections:</div>
          <div className="text-white">200</div>
          <div>Grok Factor:</div>
          <div className="text-white">{(grokIntensity * 100).toFixed(0)}%</div>
          <div>Animation:</div>
          <div className="text-white">{isAnimating ? 'Active' : 'Paused'}</div>
        </div>
      </div>

      {/* Three.js mount point */}
      <div ref={mountRef} className="absolute inset-0" />
    </div>
  );
}