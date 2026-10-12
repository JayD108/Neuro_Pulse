"use client";

import {
  XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid,
  ComposedChart, Line, Area,
} from "recharts";
import { TelemetryFrame } from "@/hooks/usePatientStream";
import { ALERT_THRESHOLD, WATCH_THRESHOLD, formatClock } from "@/lib/constants";

interface RiskHorizonProps {
  history: TelemetryFrame[];
}

const AXIS_TICK = { fill: "#6b7684", fontSize: 11, fontFamily: "IBM Plex Mono, monospace" };

/**
 * Smoothed and raw pre-ictal probability over the recent windows, with the
 * dataset's annotated pre-ictal periods shaded for reference.
 */
export default function RiskHorizon({ history }: RiskHorizonProps) {
  const data = history.map(f => ({
    t: f.window_start,
    risk: +(f.risk_score * 100).toFixed(1),
    raw: +((f.risk_score_raw ?? f.risk_score) * 100).toFixed(1),
    preictal: f.label_true === 1 ? 100 : 0,
  }));

  if (data.length === 0) {
    return <div className="panel-note" style={{ display: "grid", placeItems: "center", height: "100%" }}>Waiting for data…</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke="#1f2731" vertical={false} />
        <XAxis dataKey="t" tick={AXIS_TICK} tickFormatter={formatClock} axisLine={{ stroke: "#2b3542" }}
               tickLine={false} minTickGap={60} />
        <YAxis domain={[0, 100]} ticks={[0, 30, 60, 100]} tick={AXIS_TICK} axisLine={false} tickLine={false} unit="%" />
        <Tooltip
          contentStyle={{ background: "#161c24", border: "1px solid #2b3542", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "#a3adbb" }}
          labelFormatter={v => `t ${formatClock(Number(v))}`}
          formatter={(val, name) => {
            if (name === "preictal") return [Number(val) > 0 ? "yes" : "no", "Annotated pre-ictal"];
            return [`${val}%`, name === "risk" ? "Risk (smoothed)" : "Window score"];
          }}
        />
        <Area type="stepAfter" dataKey="preictal" fill="#e5534b" fillOpacity={0.08} stroke="none" isAnimationActive={false} />
        <ReferenceLine y={ALERT_THRESHOLD * 100} stroke="#e5534b" strokeDasharray="4 4" strokeOpacity={0.7} />
        <ReferenceLine y={WATCH_THRESHOLD * 100} stroke="#d9a441" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Line type="linear" dataKey="raw" stroke="#6b7684" strokeWidth={1} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="risk" stroke="#5b9cf0" strokeWidth={2} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
