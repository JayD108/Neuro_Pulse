"use client";

import { CHANNELS, ELECTRODES_1020 } from "@/lib/constants";

interface ElectrodeMapProps {
  /** Per-channel deviation from the training baseline, 0–1 */
  channelRisk: Record<string, number>;
}

const R = 100;          // head radius in SVG units
const SCALE = 0.86 * R; // electrode ring inside the head outline

function pos(name: string): [number, number] {
  const [x, y] = ELECTRODES_1020[name] ?? [0, 0];
  return [x * SCALE, -y * SCALE];
}

/** 0 -> slate, 1 -> amber: a single-hue intensity scale, not a risk colour */
function intensity(v: number): string {
  const t = Math.min(Math.max(v, 0), 1);
  const from = [58, 70, 86];   // #3a4656
  const to   = [232, 168, 64]; // #e8a840
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/**
 * Top view of the 10-20 montage. Each of the 18 bipolar channels is drawn
 * between its two electrodes and shaded by how far its features are from
 * the training baseline in this window.
 */
export default function ElectrodeMap({ channelRisk }: ElectrodeMapProps) {
  const ranked = Object.entries(channelRisk).sort((a, b) => b[1] - a[1]).slice(0, 3);

  return (
    <div style={{ display: "flex", gap: 16, height: "100%", alignItems: "center" }}>
      <svg viewBox="-125 -125 250 250" style={{ flex: 1, height: "100%", maxHeight: 300 }} role="img"
           aria-label="10-20 electrode map shaded by channel deviation">
        {/* Head outline, nose and ears */}
        <circle cx={0} cy={0} r={R} fill="var(--bg-inset)" stroke="var(--border-strong)" strokeWidth={1.5} />
        <path d={`M -10 ${-R + 1} L 0 ${-R - 12} L 10 ${-R + 1}`} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} />
        <path d={`M ${-R} -14 q -9 14 0 28`} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} />
        <path d={`M ${R} -14 q 9 14 0 28`} fill="none" stroke="var(--border-strong)" strokeWidth={1.5} />

        {/* Bipolar channels */}
        {CHANNELS.map(ch => {
          const [a, b] = ch.split("-");
          const [x1, y1] = pos(a);
          const [x2, y2] = pos(b);
          const v = channelRisk[ch] ?? 0;
          return (
            <line key={ch} x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke={intensity(v)} strokeWidth={3 + v * 4} strokeLinecap="round">
              <title>{`${ch}: ${(v * 100).toFixed(0)}% of max deviation`}</title>
            </line>
          );
        })}

        {/* Electrodes */}
        {Object.keys(ELECTRODES_1020).map(name => {
          const [x, y] = pos(name);
          return (
            <g key={name}>
              <circle cx={x} cy={y} r={7.5} fill="var(--bg-panel)" stroke="var(--text-muted)" strokeWidth={1} />
              <text x={x} y={y + 2.4} textAnchor="middle" fontSize={6.2} fill="var(--text-secondary)"
                    fontFamily="var(--font-mono)">
                {name.charAt(0) + name.slice(1).toLowerCase()}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ width: 150, flexShrink: 0 }}>
        <div className="section-label" style={{ marginBottom: 8 }}>Most deviant</div>
        {ranked.length === 0 && <div className="panel-note">No data yet</div>}
        {ranked.map(([ch, v]) => (
          <div key={ch} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "3px 0" }}>
            <span className="mono" style={{ color: "var(--text-primary)" }}>{ch}</span>
            <span className="mono" style={{ color: "var(--text-muted)" }}>{(v * 100).toFixed(0)}%</span>
          </div>
        ))}
        <div style={{ marginTop: 16 }}>
          <div style={{ height: 6, borderRadius: 3, background: `linear-gradient(90deg, ${intensity(0)}, ${intensity(1)})` }} />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }} className="panel-note">
            <span>baseline</span><span>max</span>
          </div>
        </div>
      </div>
    </div>
  );
}
