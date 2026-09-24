"use client";

import { useEffect, useRef } from "react";
import { BANDS, BAND_COLORS } from "@/lib/constants";

interface EEGCanvasProps {
  bandPower: Record<string, number>;
}

const HISTORY_SIZE = 200;

export default function EEGCanvas({ bandPower }: EEGCanvasProps) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<Record<string, number[]>>({});

  // Initialise history buffers
  BANDS.forEach(b => {
    if (!historyRef.current[b]) historyRef.current[b] = new Array(HISTORY_SIZE).fill(0);
  });

  useEffect(() => {
    if (!bandPower) return;
    // Push new values into each band history
    BANDS.forEach(band => {
      const h = historyRef.current[band];
      h.push(bandPower[band] ?? 0);
      if (h.length > HISTORY_SIZE) h.shift();
    });

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Background
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * H;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    const step = W / HISTORY_SIZE;

    BANDS.forEach(band => {
      const data   = historyRef.current[band];
      const maxVal = Math.max(...data, 0.01);
      const color  = BAND_COLORS[band] ?? "#ffffff";

      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur  = 6;
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.globalAlpha = 0.85;

      ctx.beginPath();
      data.forEach((val, i) => {
        const x = i * step;
        const y = H - (val / maxVal) * (H - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.shadowBlur  = 0;
      ctx.globalAlpha = 1;
    });

    // Band legend
    BANDS.forEach((band, i) => {
      const color = BAND_COLORS[band];
      ctx.fillStyle = color;
      ctx.font      = "10px 'JetBrains Mono', monospace";
      ctx.fillText(band, 8 + i * 62, H - 8);
    });
  }, [bandPower]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <canvas
        ref={canvasRef}
        width={800}
        height={240}
        style={{
          width: "100%",
          height: "100%",
          borderRadius: "8px",
          display: "block",
        }}
      />
    </div>
  );
}
