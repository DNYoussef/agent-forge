import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Ring } from '@react-three/drei';
import * as THREE from 'three';

interface GrokfastAcceleratorProps {
  position: THREE.Vector3;
  intensity: number;
  active: boolean;
}

export const GrokfastAccelerator: React.FC<GrokfastAcceleratorProps> = ({
  position,
  intensity,
  active
}) => {
  const pulseRef = useRef<THREE.Group>(null);
  const waveRef = useRef<THREE.Group>(null);
  const particlesRef = useRef<THREE.Points>(null);

  // Animate Grokfast acceleration waves
  useFrame((state, delta) => {
    if (!active) return;

    const time = state.clock.elapsedTime;

    // Pulse effect
    if (pulseRef.current) {
      const scale = 1 + Math.sin(time * 3) * intensity * 0.3;
      pulseRef.current.scale.setScalar(scale);
      pulseRef.current.rotation.y += delta * intensity;
    }

    // Wave expansion
    if (waveRef.current) {
      waveRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Mesh) {
          const scale = 1 + (Math.sin(time * 2 - i * 0.5) * 0.5 + 0.5) * 2;
          child.scale.setScalar(scale);

          const material = child.material as THREE.MeshBasicMaterial;
          material.opacity = Math.max(0, 1 - scale / 3) * intensity;
        }
      });
    }

    // Particle acceleration
    if (particlesRef.current) {
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
      const numParticles = positions.length / 3;

      for (let i = 0; i < numParticles; i++) {
        const angle = (i / numParticles) * Math.PI * 2 + time * intensity;
        const radius = 0.5 + Math.sin(time * 3 + i) * 0.3;
        const height = Math.sin(time * 2 + i * 0.5) * 0.5;

        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = height;
        positions[i * 3 + 2] = Math.sin(angle) * radius;
      }

      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  // Create acceleration particles
  const particleGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const numParticles = 50;
    const positions = new Float32Array(numParticles * 3);
    const colors = new Float32Array(numParticles * 3);

    for (let i = 0; i < numParticles; i++) {
      const angle = (i / numParticles) * Math.PI * 2;
      positions[i * 3] = Math.cos(angle) * 0.5;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = Math.sin(angle) * 0.5;

      // Orange-yellow gradient for energy
      colors[i * 3] = 1.0;
      colors[i * 3 + 1] = 0.5 + (i / numParticles) * 0.3;
      colors[i * 3 + 2] = 0.1;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    return geometry;
  }, []);

  const particleMaterial = useMemo(
    () => new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: intensity,
      blending: THREE.AdditiveBlending
    }),
    [intensity]
  );

  return (
    <group position={position}>
      {/* Energy Core */}
      <group ref={pulseRef}>
        <Sphere args={[0.3, 16, 16]}>
          <meshBasicMaterial
            color="#f59e0b"
            transparent
            opacity={0.8 * intensity}
          />
        </Sphere>
        <Sphere args={[0.4, 16, 16]}>
          <meshBasicMaterial
            color="#fbbf24"
            wireframe
            transparent
            opacity={0.5 * intensity}
          />
        </Sphere>
      </group>

      {/* Acceleration Waves */}
      <group ref={waveRef}>
        {Array.from({ length: 3 }).map((_, i) => (
          <Ring
            key={i}
            args={[0.1, 0.5 + i * 0.3, 32]}
            rotation={[Math.PI / 2, 0, 0]}
          >
            <meshBasicMaterial
              color="#f59e0b"
              transparent
              opacity={0.5}
              side={THREE.DoubleSide}
            />
          </Ring>
        ))}
      </group>

      {/* Acceleration Particles */}
      <points
        ref={particlesRef}
        geometry={particleGeometry}
        material={particleMaterial}
      />

      {/* Energy Field Lines */}
      {active && (
        <group>
          {Array.from({ length: 8 }).map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            return (
              <mesh
                key={i}
                position={[
                  Math.cos(angle) * 0.8,
                  0,
                  Math.sin(angle) * 0.8
                ]}
                rotation={[0, angle, 0]}
              >
                <cylinderGeometry args={[0.01, 0.01, 0.5, 4]} />
                <meshBasicMaterial
                  color="#fbbf24"
                  transparent
                  opacity={0.3 * intensity}
                />
              </mesh>
            );
          })}
        </group>
      )}

      {/* Gradient Flow Visualization */}
      <mesh>
        <torusGeometry args={[1, 0.02, 8, 32]} />
        <meshBasicMaterial
          color="#f97316"
          transparent
          opacity={0.4 * intensity}
        />
      </mesh>
    </group>
  );
};