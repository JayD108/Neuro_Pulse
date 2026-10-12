"use client";

import Link from "next/link";
import { usePatientStream } from "@/hooks/usePatientStream";
import { riskLevel, RISK_LEVEL_INFO, formatClock } from "@/lib/constants";
import Sparkline from "@/components/Sparkline";

interface PatientCardProps {
  patientId: string;
}

export default function PatientCard({ patientId }: PatientCardProps) {
  const { frame, history, connected } = usePatientStream(patientId, { maxHistory: 60 });

  const risk    = frame?.risk_score ?? 0;
  const level   = riskLevel(risk);
  const info    = RISK_LEVEL_INFO[level];
  const isAlert = frame?.alert ?? false;

  return (
    <Link href={`/patient/${patientId}`} className="card-link">
      <div className={`panel ${isAlert ? "card-alert" : ""}`} style={{ padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              className="status-dot"
              title={connected ? "Receiving data" : "Not connected"}
              style={{ background: connected ? "var(--status-normal)" : "var(--text-muted)" }}
            />
            <span className="mono" style={{ fontWeight: 600, fontSize: 15, letterSpacing: "0.04em" }}>
              {patientId.toUpperCase()}
            </span>
          </div>
          <span className="status-tag" style={{ color: frame ? info.color : "var(--text-muted)" }}>
            {frame ? info.label : "No data"}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 6 }}>
          <span className="panel-note">Pre-ictal risk</span>
          <span className="mono" style={{ fontSize: 22, fontWeight: 600, color: frame ? info.color : "var(--text-muted)" }}>
            {frame ? `${Math.round(risk * 100)}%` : "—"}
          </span>
        </div>
        <div className="risk-bar-track" style={{ marginBottom: 12 }}>
          <div className="risk-bar-fill" style={{ width: `${risk * 100}%`, background: info.color }} />
        </div>

        <Sparkline values={history.map(f => f.risk_score)} color={info.color} height={32} />

        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
          <span className="panel-note mono">
            {frame ? `t ${formatClock(frame.window_start)}` : "waiting…"}
          </span>
          <span className="panel-note mono">
            {frame ? `win ${frame.window_idx}` : ""}
          </span>
        </div>
      </div>
    </Link>
  );
}
