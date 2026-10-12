"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { TelemetryFrame } from "@/hooks/usePatientStream";
import { BANDS, BAND_INFO, formatClock } from "@/lib/constants";

interface BandPowerChartProps {
  history: TelemetryFrame[];
}

const AXIS_TICK = { fill: "#6b7684", fontSize: 11, fontFamily: "IBM Plex Mono, monospace" };

/** Relative power of the five clinical bands over time (stacked to 100%). */
export default function BandPowerChart({ history }: BandPowerChartProps) {
  const data = history.map(f => {
    const bp = f.band_power ?? {};
    const total = BANDS.reduce((s, b) => s + (bp[b] ?? 0), 0) || 1;
    const row: Record<string, number> = { t: f.window_start };
    BANDS.forEach(b => { row[b] = +(((bp[b] ?? 0) / total) * 100).toFixed(1); });
    return row;
  });
  const latest = data[data.length - 1];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, minHeight: 0 }}>
        {data.length === 0 ? (
          <div className="panel-note" style={{ display: "grid", placeItems: "center", height: "100%" }}>Waiting for data…</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke="#1f2731" vertical={false} />
              <XAxis dataKey="t" tick={AXIS_TICK} tickFormatter={formatClock} axisLine={{ stroke: "#2b3542" }}
                     tickLine={false} minTickGap={60} />
              <YAxis domain={[0, 100]} ticks={[0, 50, 100]} tick={AXIS_TICK} axisLine={false} tickLine={false} unit="%" />
              <Tooltip
                contentStyle={{ background: "#161c24", border: "1px solid #2b3542", borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: "#a3adbb" }}
                labelFormatter={v => `t ${formatClock(Number(v))}`}
                formatter={(val, name) => [`${val}%`, String(name)]}
              />
              {BANDS.map(b => (
                <Area key={b} type="monotone" dataKey={b} stackId="1" stroke={BAND_INFO[b].color}
                      fill={BAND_INFO[b].color} fillOpacity={0.55} strokeWidth={1} isAnimationActive={false} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, marginTop: 12 }}>
        {BANDS.map(b => (
          <div key={b} style={{ borderLeft: `3px solid ${BAND_INFO[b].color}`, paddingLeft: 8 }}>
            <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 500 }}>{b}</div>
            <div className="panel-note">{BAND_INFO[b].range}</div>
            <div className="mono" style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              {latest ? `${latest[b].toFixed(1)}%` : "—"}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
