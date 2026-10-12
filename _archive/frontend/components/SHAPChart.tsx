"use client";

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";

interface SHAPFeature {
  feature: string;
  shap_value: number;
}

interface SHAPChartProps {
  features: SHAPFeature[];
}

function shortName(name: string): string {
  return name.replace(/^([A-Z0-9]+-[A-Z0-9]+)_/, "$1 ").slice(0, 28);
}

export default function SHAPChart({ features }: SHAPChartProps) {
  if (!features || features.length === 0) {
    return (
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center",
        height: "100%", color: "var(--text-muted)", fontSize: 13,
      }}>
        Waiting for SHAP data…
      </div>
    );
  }

  const data = features
    .slice(0, 10)
    .map(f => ({
      name: shortName(f.feature),
      value: parseFloat(f.shap_value.toFixed(3)),
      abs: Math.abs(f.shap_value),
    }))
    .sort((a, b) => b.abs - a.abs);

  const maxAbs = Math.max(...data.map(d => d.abs), 0.01);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
      >
        <XAxis
          type="number"
          domain={[-maxAbs * 1.1, maxAbs * 1.1]}
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={160}
          tick={{ fill: "var(--text-secondary)", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          contentStyle={{
            background: "var(--bg-panel-solid)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 8,
            color: "var(--text-primary)",
            fontSize: 12,
          }}
          formatter={(val: any) => [Number(val).toFixed(4), "SHAP"]}
        />
        <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry, idx) => (
            <Cell
              key={idx}
              fill={entry.value >= 0 ? "#ef4444" : "#00d4c8"}
              opacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
