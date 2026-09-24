"use client";

import { use, useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { usePatientStream } from "@/hooks/usePatientStream";
import EEGCanvas from "@/components/EEGCanvas";
import SHAPChart from "@/components/SHAPChart";
import RiskHorizon from "@/components/RiskHorizon";
import PatientSummaryBanner from "@/components/PatientSummaryBanner";
import { riskColor, ALERT_THRESHOLD, API_BASE_URL } from "@/lib/constants";

// 3D Brain Viewer — SSR-incompatible, loaded only on client
const BrainViewer = dynamic(() => import("@/components/BrainViewer"), {
  ssr: false,
  loading: () => (
    <div style={{
      width: "100%", height: "100%",
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "var(--text-muted)", fontSize: 13,
    }}>
      Loading brain map…
    </div>
  ),
});

interface PatientMeta {
  n_windows: number;
  n_preictal: number;
  n_interictal: number;
  n_features: number;
  model_roc_auc: number;
  model_sensitivity: number;
  model_specificity: number;
}

const PANEL_STYLE = {
  // Using the glass-panel class for most styles, just adding layout specific ones here
  padding: "20px",
  display: "flex" as const,
  flexDirection: "column" as const,
  overflow: "hidden" as const,
};

// Panel title with plain-English subtitle
function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div style={{ marginBottom: 16, flexShrink: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="display-font" style={{ fontWeight: 700, fontSize: 16, color: "var(--text-secondary)", letterSpacing: "0.03em" }}>{title}</span>
      </div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, lineHeight: 1.5 }}>
        {subtitle}
      </div>
    </div>
  );
}

export default function PatientPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { frame, history, connected } = usePatientStream(id);
  const [meta, setMeta] = useState<PatientMeta | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/patient/${id}/info`)
      .then(r => r.json())
      .then(setMeta)
      .catch(() => null);
  }, [id]);

  const risk    = frame?.risk_score ?? 0;
  const isAlert = risk >= ALERT_THRESHOLD;
  const color   = riskColor(risk);

  return (
    <div style={{ padding: "24px 24px 32px", minHeight: "calc(100vh - 52px)", display: "flex", flexDirection: "column", gap: 14 }}>

      {/* ── Navigation + Identity strip ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
        <Link href="/" style={{
          color: "var(--text-muted)", fontSize: 12, textDecoration: "none",
          border: "1px solid var(--border-subtle)", padding: "4px 12px",
          borderRadius: 6,
        }}>
          ← Back to All Patients
        </Link>
        <div>
          <div className="section-label" style={{ marginBottom: 4 }}>
            Monitoring Patient
          </div>
          <div className="display-font gradient-text" style={{ fontSize: 26, fontWeight: 800 }}>
            {id.toUpperCase()}
          </div>
        </div>

        {/* Live connection pill */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 16px",
          background: connected ? "rgba(0,240,255,0.08)" : "rgba(100,100,100,0.1)",
          border: `1px solid ${connected ? "rgba(0,240,255,0.30)" : "rgba(100,100,100,0.2)"}`,
          borderRadius: "99px", fontSize: 13, fontWeight: 600,
          color: connected ? "var(--accent-teal)" : "#888",
          boxShadow: connected ? "0 0 15px rgba(0,240,255,0.15)" : "none",
        }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: connected ? "var(--accent-teal)" : "#555",
            boxShadow: connected ? "0 0 10px var(--accent-teal)" : "none",
            display: "inline-block",
          }} />
          {connected ? "Live Monitoring Active" : "Reconnecting…"}
        </div>

        {/* Quick model trust stats — in plain language */}
        {meta && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 12 }}>
            {[
              { l: "AI Accuracy",     v: `${(meta.model_roc_auc * 100).toFixed(1)}%`,      tip: "How often the AI is right" },
              { l: "Seizure Detection", v: `${(meta.model_sensitivity * 100).toFixed(1)}%`, tip: "How often it catches real events" },
              { l: "False Alarms",    v: `${((1-meta.model_specificity)*100).toFixed(1)}%`, tip: "How often it alerts unnecessarily" },
            ].map(({ l, v, tip }) => (
              <div key={l} title={tip} className="glass-panel hover-lift" style={{
                borderRadius: 12, padding: "10px 20px", textAlign: "center", cursor: "default"
              }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4, fontFamily: "var(--font-display)", fontWeight: 600 }}>{l}</div>
                <div className="display-font" style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-teal)", textShadow: "0 0 15px rgba(0,240,255,0.3)" }}>{v}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Patient-Friendly Summary Banner ── */}
      <PatientSummaryBanner frame={frame} patientId={id} />

      {/* ── 4-Quadrant Technical Grid (for medical staff) ── */}
      <div style={{
        fontSize: 10, fontWeight: 600, letterSpacing: "0.15em",
        textTransform: "uppercase", color: "var(--text-muted)",
        paddingBottom: 4, borderBottom: "1px solid rgba(255,255,255,0.05)",
      }}>
        Detailed Brainwave Monitoring — For Medical Staff
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gridTemplateRows: "1fr 1fr",
        gap: 14,
        flex: 1,
        minHeight: 480,
      }}>

        {/* Top-Left: 3D Brain */}
        <div className="glass-panel hover-lift" style={{ ...PANEL_STYLE }}>
          <PanelHeader
            title="Brain Activity Map"
            subtitle="Each dot is an electrode on the scalp. Brighter red = more unusual activity in that region"
          />
          <div style={{ flex: 1, minHeight: 0 }}>
            <BrainViewer channelRisk={frame?.channel_risk ?? {}} />
          </div>
        </div>

        {/* Top-Right: EEG Canvas */}
        <div className="glass-panel hover-lift" style={{ ...PANEL_STYLE }}>
          <PanelHeader
            title="Live Brainwave Activity"
            subtitle="Five types of brainwaves tracked every second — each color is a different frequency of brain signal"
          />
          <div style={{ flex: 1, minHeight: 0 }}>
            <EEGCanvas bandPower={frame?.band_power ?? {}} />
          </div>
          {/* Band legend with plain names */}
          <div style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap", background: "rgba(0,0,0,0.2)", padding: "8px 12px", borderRadius: "8px" }}>
            {[
              { name: "Delta", desc: "Deep/slow", color: "#8b5cf6" },
              { name: "Theta", desc: "Drowsy",    color: "#00f0ff" },
              { name: "Alpha", desc: "Calm",      color: "#10b981" },
              { name: "Beta",  desc: "Thinking",  color: "#fbbf24" },
              { name: "Gamma", desc: "Alert",     color: "#ff3366" },
            ].map(b => (
              <div key={b.name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: b.color, display: "inline-block", boxShadow: `0 0 8px ${b.color}` }} />
                <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>
                  {b.name} <span style={{ color: "var(--text-muted)" }}>({b.desc})</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom-Left: SHAP */}
        <div className="glass-panel hover-lift" style={{ ...PANEL_STYLE }}>
          <PanelHeader
            title="What the AI Is Focusing On"
            subtitle="Red bars = signals pushing toward a seizure warning. Teal bars = signals keeping you in the clear"
          />
          <div style={{ flex: 1, minHeight: 0 }}>
            <SHAPChart features={frame?.top_shap ?? []} />
          </div>
        </div>

        {/* Bottom-Right: Risk Horizon */}
        <div className="glass-panel hover-lift" style={{ ...PANEL_STYLE }}>
          <PanelHeader
            title="Seizure Risk Over Time"
            subtitle={`Showing the last ${Math.max(1, Math.round(history.length / 6))} minutes. If the line reaches the red dotted line at 60%, an alert fires.`}
          />
          <div style={{ flex: 1, minHeight: 0 }}>
            <RiskHorizon history={history} />
          </div>
        </div>

      </div>
    </div>
  );
}
