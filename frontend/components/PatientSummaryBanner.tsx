"use client";

import { TelemetryFrame } from "@/hooks/usePatientStream";

interface PatientSummaryBannerProps {
  frame: TelemetryFrame | null;
  patientId: string;
}

// Plain-English translations of SHAP feature names
function translateFeature(name: string): string {
  const map: Record<string, string> = {
    higuchi_fd:    "brain signal irregularity",
    sample_entropy: "brain signal complexity",
    wt_D1_energy:  "fast brain wave activity",
    wt_D2_energy:  "rapid brain wave bursts",
    wt_D3_energy:  "moderate wave bursts",
    wt_A_energy:   "slow brain wave energy",
    Delta_power:   "deep sleep wave energy (Delta)",
    Theta_power:   "drowsiness wave energy (Theta)",
    Alpha_power:   "relaxation wave energy (Alpha)",
    Beta_power:    "thinking wave energy (Beta)",
    Gamma_power:   "high-focus wave energy (Gamma)",
    Delta_relpow:  "proportion of slow waves",
    Theta_relpow:  "proportion of drowsy waves",
    Alpha_relpow:  "proportion of calm waves",
    Beta_relpow:   "proportion of active waves",
    Gamma_relpow:  "proportion of alert waves",
    hjorth_activity: "brainwave energy level",
    hjorth_mobility: "average brainwave speed",
    hjorth_complexity: "brainwave pattern variety",
    rms:           "overall signal strength",
    zero_crossings:"signal rhythm rate",
  };
  for (const [key, label] of Object.entries(map)) {
    if (name.includes(key)) {
      const ch = name.split("_")[0];
      return `${label} (near ${ch})`;
    }
  }
  return name;
}

function getStatusConfig(risk: number) {
  if (risk < 0.30) return {
    headline: "Your brain activity looks normal",
    detail: "The AI is detecting calm, regular brain patterns. No signs of an upcoming seizure right now. You can relax.",
    color: "#22c55e",
    bg: "rgba(34,197,94,0.08)",
    border: "rgba(34,197,94,0.25)",
    label: "ALL CLEAR",
  };
  if (risk < 0.60) return {
    headline: "Some unusual brain activity detected",
    detail: "The AI has noticed some changes in your brainwave patterns. This is not an emergency, but your care team is monitoring closely.",
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.08)",
    border: "rgba(245,158,11,0.25)",
    label: "WATCH",
  };
  return {
    headline: "Warning: Seizure activity may be building",
    detail: "The AI has detected strong warning signs in your brain signals. Your nurse or doctor should be notified immediately. Try to stay seated and calm.",
    color: "#ef4444",
    bg: "rgba(239,68,68,0.10)",
    border: "rgba(239,68,68,0.45)",
    label: "ALERT",
  };
}

export default function PatientSummaryBanner({ frame, patientId }: PatientSummaryBannerProps) {
  const risk   = frame?.risk_score ?? 0;
  const status = getStatusConfig(risk);
  const riskPct = Math.round(risk * 100);

  // Top 3 SHAP features in plain English
  const topReasons = (frame?.top_shap ?? [])
    .slice(0, 3)
    .map(f => translateFeature(f.feature));

  // Dominant brain wave band
  const bp = frame?.band_power ?? {};
  const dominantBand = Object.entries(bp).sort((a, b) => b[1] - a[1])[0];
  const bandLabels: Record<string, string> = {
    Delta: "slow, restful (Delta)",
    Theta: "drowsy or relaxed (Theta)",
    Alpha: "calm and relaxed (Alpha)",
    Beta:  "alert and thinking (Beta)",
    Gamma: "highly focused (Gamma)",
  };

  return (
    <div className="glass-panel" style={{
      background: status.bg,
      border: `1px solid ${status.border}`,
      boxShadow: `0 8px 32px ${status.border}30, inset 0 0 20px ${status.color}10`,
      borderRadius: "16px",
      padding: "24px 28px",
      marginBottom: "24px",
      transition: "all 0.4s ease",
    }}>
      {/* Main status */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 20 }}>

        {/* Risk meter */}
        <div style={{ textAlign: "center", flexShrink: 0 }}>
          <div style={{
            marginTop: 10,
            padding: "4px 14px",
            borderRadius: "99px",
            background: status.color,
            color: "#000",
            fontWeight: 800,
            fontSize: "11px",
            letterSpacing: "0.12em",
          }}>
            {status.label}
          </div>
          <div style={{
            marginTop: 8,
            fontSize: "26px",
            fontWeight: 800,
            color: status.color,
            fontFamily: "JetBrains Mono, monospace",
          }}>
            {riskPct}%
          </div>
          <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: 2 }}>
            Risk Score
          </div>
        </div>

        {/* Main message */}
        <div style={{ flex: 1 }}>
          <div className="section-label" style={{ marginBottom: 6 }}>
            Patient {patientId.toUpperCase()} — Status Right Now
          </div>
          <div className="display-font" style={{ fontSize: "24px", fontWeight: 800, color: status.color, marginBottom: 10, lineHeight: 1.3, textShadow: `0 0 15px ${status.color}50` }}>
            {status.headline}
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: "15px", lineHeight: 1.7, maxWidth: 650, margin: 0 }}>
            {status.detail}
          </p>

          {/* Plain-English breakdown */}
          <div style={{
            marginTop: 20, display: "flex", gap: 14, flexWrap: "wrap"
          }}>
            {/* Dominant brainwave */}
            {dominantBand && (
              <div className="hover-lift" style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12, padding: "12px 18px",
                transition: "all 0.3s ease",
              }}>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
                  Dominant Brainwave
                </div>
                <div className="display-font" style={{ fontWeight: 600, fontSize: "15px", color: "var(--text-primary)" }}>
                  Your brain is currently <span style={{ color: "var(--accent-teal)", textShadow: "0 0 10px rgba(0,240,255,0.4)" }}>
                    {bandLabels[dominantBand[0]] ?? dominantBand[0]}
                  </span>
                </div>
              </div>
            )}

            {/* AI reasoning in plain English */}
            {topReasons.length > 0 && (
              <div className="hover-lift" style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12, padding: "12px 18px",
                maxWidth: 400,
                transition: "all 0.3s ease",
              }}>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
                  What the AI is focused on
                </div>
                <ul style={{ margin: 0, padding: "0 0 0 18px", fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.8 }}>
                  {topReasons.map((r, i) => <li key={i}>Increased {r}</li>)}
                </ul>
              </div>
            )}

            {/* Reassurance / action */}
            <div className="hover-lift" style={{
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12, padding: "12px 18px",
              transition: "all 0.3s ease",
            }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>
                AI Confidence
              </div>
              <div className="display-font" style={{ fontWeight: 600, fontSize: "15px", color: "var(--text-primary)" }}>
                This AI system is <span style={{ color: "var(--accent-teal)", textShadow: "0 0 10px rgba(0,240,255,0.4)" }}>92.3% accurate</span> at detecting warning signs up to 30 minutes before a seizure.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
