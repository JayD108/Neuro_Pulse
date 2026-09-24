"use client";

import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
  Area, AreaChart,
} from "recharts";
import { TelemetryFrame } from "@/hooks/usePatientStream";
import { ALERT_THRESHOLD, riskColor } from "@/lib/constants";

interface RiskHorizonProps {
  history: TelemetryFrame[];
}

interface CustomDotProps {
  cx?: number;
  cy?: number;
  payload?: { alert: boolean };
}

function CustomDot({ cx, cy, payload }: CustomDotProps) {
  if (!payload?.alert || !cx || !cy) return null;
  return (
    <circle
      cx={cx} cy={cy} r={4}
      fill="#ef4444"
      stroke="#ff6666"
      strokeWidth={2}
      style={{ filter: "drop-shadow(0 0 6px #ef4444)" }}
    />
  );
}

export default function RiskHorizon({ history }: RiskHorizonProps) {
  const data = history.slice(-300).map((f, i) => ({
    i,
    risk: parseFloat((f.risk_score * 100).toFixed(1)),
    alert: f.alert,
  }));

  const latest   = data[data.length - 1]?.risk ?? 0;
  const color    = riskColor(latest / 100);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Live value overlay */}
      <div style={{
        position: "absolute", top: 8, right: 12,
        fontWeight: 700, fontSize: "22px",
        color,
        textShadow: `0 0 12px ${color}`,
        zIndex: 10,
        fontFamily: "JetBrains Mono, monospace",
      }}>
        {latest.toFixed(1)}%
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 16, right: 16, left: -8, bottom: 4 }}>
          <defs>
            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={color} stopOpacity={0.35} />
              <stop offset="95%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.05)"
            vertical={false}
          />
          <XAxis
            dataKey="i"
            hide
          />
          <YAxis
            domain={[0, 100]}
            ticks={[0, 20, 40, 60, 80, 100]}
            tick={{ fill: "var(--text-muted)", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              background: "var(--bg-panel-solid)",
              border: "1px solid var(--border-subtle)",
              borderRadius: 8,
              color: "var(--text-primary)",
              fontSize: 12,
            }}
            formatter={(val: any) => [`${val}%`, "Seizure Risk"]}
            labelFormatter={() => ""}
          />
          {/* 60% threshold line */}
          <ReferenceLine
            y={ALERT_THRESHOLD * 100}
            stroke="rgba(239,68,68,0.55)"
            strokeDasharray="6 3"
            label={{
              value: "Alert 60%",
              position: "insideTopRight",
              fill: "#ef4444",
              fontSize: 10,
            }}
          />
          <Area
            type="monotoneX"
            dataKey="risk"
            stroke={color}
            strokeWidth={2}
            fill="url(#riskGrad)"
            dot={<CustomDot />}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
