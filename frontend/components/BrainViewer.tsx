"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere } from "@react-three/drei";
import * as THREE from "three";
import { ELECTRODE_POSITIONS, riskColor } from "@/lib/constants";

interface ElectrodeNodeProps {
  position: [number, number, number];
  risk: number;
  label: string;
}

function ElectrodeNode({ position, risk, label }: ElectrodeNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const color   = riskColor(risk);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      const pulse = 1 + Math.sin(clock.elapsedTime * 2 + risk * 10) * 0.12 * risk;
      meshRef.current.scale.setScalar(pulse);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.055, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={risk * 2.5}
        transparent
        opacity={0.85 + risk * 0.15}
        roughness={0.2}
        metalness={0.6}
      />
    </mesh>
  );
}

function BrainSphere() {
  return (
    <Sphere args={[1, 64, 64]}>
      <meshStandardMaterial
        color="#0c1a3a"
        wireframe={false}
        transparent
        opacity={0.45}
        roughness={0.85}
        metalness={0.2}
      />
    </Sphere>
  );
}

function WireframeSphere() {
  return (
    <Sphere args={[1.01, 24, 24]}>
      <meshStandardMaterial
        color="#1e40af"
        wireframe
        transparent
        opacity={0.10}
      />
    </Sphere>
  );
}

interface BrainViewerProps {
  channelRisk: Record<string, number>;
}

export default function BrainViewer({ channelRisk }: BrainViewerProps) {
  const nodes = useMemo(() =>
    Object.entries(ELECTRODE_POSITIONS).map(([ch, pos]) => ({
      ch,
      pos: pos as [number, number, number],
      risk: channelRisk[ch] ?? 0,
    })),
    [channelRisk]
  );

  return (
    <div style={{ width: "100%", height: "100%", background: "#030712", borderRadius: 8 }}>
      <Canvas
        camera={{ position: [0, 0, 2.8], fov: 45 }}
        style={{ borderRadius: 8 }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <pointLight position={[3, 3, 3]} intensity={0.8} color="#4fc3f7" />
        <pointLight position={[-3, -3, -3]} intensity={0.4} color="#ef4444" />

        <BrainSphere />
        <WireframeSphere />

        {nodes.map(({ ch, pos, risk }) => (
          <ElectrodeNode key={ch} position={pos} risk={risk} label={ch} />
        ))}

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          autoRotate
          autoRotateSpeed={0.6}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={(3 * Math.PI) / 4}
        />
      </Canvas>
    </div>
  );
}
