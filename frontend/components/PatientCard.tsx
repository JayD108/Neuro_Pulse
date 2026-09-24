"use client";

import Link from "next/link";
import { usePatientStream } from "@/hooks/usePatientStream";
import { riskColor, ALERT_THRESHOLD } from "@/lib/constants";

interface PatientCardProps {
  patientId: string;
}

export default function PatientCard({ patientId }: PatientCardProps) {
  const { frame, history, connected } = usePatientStream(patientId);

  const risk        = frame?.risk_score ?? 0;
  const isAlert     = risk >= ALERT_THRESHOLD;
  const fillColor   = riskColor(risk);
  const riskPct     = Math.round(risk * 100);

  // Mini sparkline — last 20 risk_score values
  const sparkData   = history.slice(-20).map(f => f.risk_score);
  const sparkMax    = Math.max(...sparkData, 0.01);

  return (
    <Link href={`/patient/${patientId}`} style={{ textDecoration: "none" }}>
      <div
        className={`glass-panel hover-lift ${isAlert ? "card-alert-pulse" : ""}`}
        style={{
          padding: "20px",
          cursor: "pointer",
          border: isAlert
            ? "1px solid rgba(255, 51, 102, 0.5)"
            : "1px solid var(--border-subtle)",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Header row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div>
            <div className="section-label" style={{ marginBottom: 4 }}>Patient</div>
            <div className="display-font gradient-text" style={{ fontWeight: 800, fontSize: "22px", letterSpacing: "0.04em" }}>
              {patientId.toUpperCase()}
            </div>
          </div>

          {/* Connection indicator */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
            <span style={{
              width: 10, height: 10, borderRadius: "50%",
              background: connected ? "var(--accent-teal)" : "#475569",
              boxShadow: connected ? "0 0 12px var(--accent-teal)" : "none",
              display: "block",
            }} />
            {isAlert && (
              <span className="risk-badge alert">Alert</span>
            )}
          </div>
        </div>

        {/* Risk Score */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <span className="panel-title">Seizure Risk</span>
            <span className="display-font" style={{ fontWeight: 800, fontSize: "18px", color: fillColor, textShadow: `0 0 10px ${fillColor}50` }}>
              {riskPct}%
            </span>
          </div>
          <div className="risk-bar-track">
            <div
              className="risk-bar-fill"
              style={{ width: `${riskPct}%`, background: fillColor }}
            />
          </div>
        </div>

        {/* Mini Sparkline */}
        <div style={{ position: "relative", height: "36px", margin: "0 -8px" }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${sparkData.length} 1`} preserveAspectRatio="none" style={{ position: "absolute", bottom: 0 }}>
            <defs>
              <linearGradient id={`gradient-${patientId}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={fillColor} stopOpacity="0.3" />
                <stop offset="100%" stopColor={fillColor} stopOpacity="0.0" />
              </linearGradient>
            </defs>
            <polygon
              points={`0,1 ${sparkData.map((v, i) => `${i},${1 - v / sparkMax}`).join(" ")} ${sparkData.length - 1},1`}
              fill={`url(#gradient-${patientId})`}
            />
            <polyline
              points={sparkData.map((v, i) => `${i},${1 - v / sparkMax}`).join(" ")}
              fill="none"
              stroke={fillColor}
              strokeWidth="0.06"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.9"
            />
          </svg>
        </div>

        {/* Window index */}
        {frame && (
          <div className="section-label mono" style={{ marginTop: 6 }}>
            Window #{frame.window_idx}
          </div>
        )}
      </div>
    </Link>
  );
}
