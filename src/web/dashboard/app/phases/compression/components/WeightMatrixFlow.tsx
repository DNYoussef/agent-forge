import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface WeightMatrixFlowProps {
  stage: string;
  chamberPositions: {
    seedlm: THREE.Vector3;
    vptq: THREE.Vector3;
    hypercompression: THREE.Vector3;
  };
  isPlaying: boolean;
}

interface Particle {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  color: THREE.Color;
  size: number;
  stage: string;
  targetPosition: THREE.Vector3;
  progress: number;
  id: number;
}

export const WeightMatrixFlow: React.FC<WeightMatrixFlowProps> = ({
  stage,
  chamberPositions,
  isPlaying
}) => {
  const particlesRef = useRef<THREE.Points>(null);
  const particles = useRef<Particle[]>([]);
  const time = useRef(0);

  // Initialize particles
  useEffect(() => {
    const numParticles = 200;
    const newParticles: Particle[] = [];

    for (let i = 0; i < numParticles; i++) {
      const startX = (Math.random() - 0.5) * 4;
      const startY = (Math.random() - 0.5) * 4;
      const startZ = (Math.random() - 0.5) * 4;

      newParticles.push({
        position: new THREE.Vector3(
          chamberPositions.seedlm.x + startX - 4,
          chamberPositions.seedlm.y + startY,
          chamberPositions.seedlm.z + startZ
        ),
        velocity: new THREE.Vector3(
          Math.random() * 0.02 - 0.01,
          Math.random() * 0.02 - 0.01,
          Math.random() * 0.02 - 0.01
        ),
        color: new THREE.Color(0.5, 0.5, 1.0), // Blue for original weights
        size: Math.random() * 0.5 + 0.5,
        stage: 'idle',
        targetPosition: chamberPositions.seedlm.clone(),
        progress: 0,
        id: i
      });
    }

    particles.current = newParticles;
  }, [chamberPositions]);

  useFrame((state, delta) => {
    if (!particlesRef.current || !isPlaying) return;

    time.current += delta;

    const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
    const colors = particlesRef.current.geometry.attributes.color.array as Float32Array;
    const sizes = particlesRef.current.geometry.attributes.size.array as Float32Array;

    particles.current.forEach((particle, i) => {
      // Update particle stage based on global stage
      if (stage === 'seedlm' && particle.stage === 'idle') {
        particle.stage = 'seedlm';
        particle.targetPosition = chamberPositions.seedlm.clone();
        particle.targetPosition.x += (Math.random() - 0.5) * 2;
        particle.targetPosition.y += (Math.random() - 0.5) * 2;
        particle.targetPosition.z += (Math.random() - 0.5) * 2;
        particle.color.set(0.5, 0.3, 0.8); // Purple after SeedLM
      } else if (stage === 'vptq' && particle.stage === 'seedlm') {
        particle.stage = 'transitioning_vptq';
        particle.targetPosition = chamberPositions.vptq.clone();
        particle.targetPosition.x += (Math.random() - 0.5) * 2;
        particle.targetPosition.y += (Math.random() - 0.5) * 2;
        particle.targetPosition.z += (Math.random() - 0.5) * 2;
        particle.progress = 0;
      } else if (stage === 'hypercompression' && particle.stage === 'vptq') {
        particle.stage = 'transitioning_hyper';
        particle.targetPosition = chamberPositions.hypercompression.clone();
        particle.targetPosition.x += (Math.random() - 0.5) * 2;
        particle.targetPosition.y += (Math.random() - 0.5) * 2;
        particle.targetPosition.z += (Math.random() - 0.5) * 2;
        particle.progress = 0;
      }

      // Handle transitions
      if (particle.stage.startsWith('transitioning')) {
        particle.progress += delta * 0.5; // Transition speed

        if (particle.progress >= 1) {
          particle.progress = 1;

          if (particle.stage === 'transitioning_vptq') {
            particle.stage = 'vptq';
            particle.color.set(0.2, 0.8, 0.4); // Green after VPTQ
            particle.size *= 0.7; // Smaller after quantization
          } else if (particle.stage === 'transitioning_hyper') {
            particle.stage = 'hypercompression';
            particle.color.set(0.8, 0.7, 0.2); // Gold after Hypercompression
            particle.size *= 0.5; // Even smaller after hypercompression
          }
        }

        // Smooth transition along path
        const t = particle.progress;
        const smoothT = t * t * (3 - 2 * t); // Smoothstep

        particle.position.lerp(particle.targetPosition, smoothT * delta * 2);
      } else {
        // Normal movement within chamber
        particle.position.x += particle.velocity.x;
        particle.position.y += particle.velocity.y;
        particle.position.z += particle.velocity.z;

        // Add some noise movement
        particle.position.x += Math.sin(time.current + particle.id) * 0.01;
        particle.position.y += Math.cos(time.current * 0.7 + particle.id) * 0.01;
        particle.position.z += Math.sin(time.current * 1.3 + particle.id) * 0.01;

        // Attract to target position
        const direction = new THREE.Vector3().subVectors(particle.targetPosition, particle.position);
        direction.multiplyScalar(0.01);
        particle.velocity.add(direction);

        // Damping
        particle.velocity.multiplyScalar(0.98);
      }

      // Update buffers
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;

      colors[i * 3] = particle.color.r;
      colors[i * 3 + 1] = particle.color.g;
      colors[i * 3 + 2] = particle.color.b;

      sizes[i] = particle.size * (1 + Math.sin(time.current * 2 + particle.id) * 0.2);
    });

    particlesRef.current.geometry.attributes.position.needsUpdate = true;
    particlesRef.current.geometry.attributes.color.needsUpdate = true;
    particlesRef.current.geometry.attributes.size.needsUpdate = true;
  });

  // Particle geometry and material
  const [geometry, material] = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const numParticles = 200;

    // Position attribute
    const positions = new Float32Array(numParticles * 3);
    const colors = new Float32Array(numParticles * 3);
    const sizes = new Float32Array(numParticles);

    particles.current.forEach((particle, i) => {
      positions[i * 3] = particle.position.x;
      positions[i * 3 + 1] = particle.position.y;
      positions[i * 3 + 2] = particle.position.z;

      colors[i * 3] = particle.color.r;
      colors[i * 3 + 1] = particle.color.g;
      colors[i * 3 + 2] = particle.color.b;

      sizes[i] = particle.size;
    });

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // Shader material for particles
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        varying float vSize;

        void main() {
          vColor = color;
          vSize = size;

          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vSize;

        void main() {
          vec2 xy = gl_PointCoord.xy - vec2(0.5);
          float r = length(xy);

          if (r > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.0, 0.5, r);
          alpha *= 0.8;

          // Glow effect
          vec3 glowColor = vColor * (1.0 + (1.0 - r) * 0.5);

          gl_FragColor = vec4(glowColor, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      vertexColors: true
    });

    return [geo, mat];
  }, []);

  return (
    <points ref={particlesRef} geometry={geometry} material={material} />
  );
};