"use client";

import { useEffect, useState } from "react";
import PatientCard from "@/components/PatientCard";
import { API_BASE_URL } from "@/lib/constants";

export default function HomePage() {
  const [patients, setPatients] = useState<string[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/patients`)
      .then(r => r.json())
      .then(data => { setPatients(data.patients ?? []); setLoading(false); })
      .catch(() => { setError("Cannot reach backend at localhost:8000"); setLoading(false); });
  }, []);

  return (
    <div style={{ padding: "32px 28px" }}>

      {/* Page Header — plain English */}
      <div style={{ marginBottom: 36 }}>
        <div className="section-label" style={{ marginBottom: 8, color: "var(--accent-teal)" }}>
          Hospital Monitoring Room
        </div>
        <h1 className="display-font gradient-text" style={{ fontSize: "36px", fontWeight: 800, margin: "0 0 12px", letterSpacing: "-0.02em" }}>
          Active Patient Monitoring
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 15, lineHeight: 1.7, maxWidth: 680 }}>
          Real-time AI analysis of EEG brainwaves. Risk is estimated <strong style={{ color: "var(--text-primary)", fontWeight: 600 }}>every second</strong> predicting seizures up to 30 minutes in advance.
        </p>
      </div>

      {/* How-it-works strip */}
      <div className="glass-panel" style={{
        display: "flex", gap: 0, marginBottom: 40,
        borderRadius: "var(--radius)", overflow: "hidden",
      }}>
        {[
          { step: "1", title: "Brain signals captured", desc: "EEG electrodes on the patient's scalp record brainwaves 256 times per second" },
          { step: "2", title: "AI analyses patterns",    desc: "590 different features are extracted from each 10-second window of brain data" },
          { step: "3", title: "Risk score calculated",   desc: "A 4-model AI ensemble produces a seizure probability score from 0% to 100%" },
          { step: "4", title: "Alert if risk is high",   desc: "If the risk score crosses 60%, the card turns red and a warning is raised" },
        ].map(({ step, title, desc }, i, arr) => (
          <div key={step} className="hover-lift" style={{
            flex: 1, padding: "20px 24px",
            borderRight: i < arr.length - 1 ? "1px solid var(--border-subtle)" : "none",
            background: "rgba(255, 255, 255, 0.01)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{
                width: 24, height: 24, borderRadius: "50%",
                background: "rgba(0,240,255,0.1)", border: "1px solid rgba(0,240,255,0.3)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700, color: "var(--accent-teal)", flexShrink: 0,
                fontFamily: "var(--font-mono)",
                boxShadow: "0 0 10px rgba(0, 240, 255, 0.1)"
              }}>{step}</span>
              <span className="display-font" style={{ fontWeight: 600, fontSize: 15, color: "var(--text-primary)" }}>{title}</span>
            </div>
            <p style={{ margin: 0, fontSize: 13, color: "var(--text-muted)", lineHeight: 1.6 }}>{desc}</p>
          </div>
        ))}
      </div>

      {/* Risk legend */}
      <div style={{ display: "flex", gap: 16, marginBottom: 32, alignItems: "center", flexWrap: "wrap" }}>
        <span className="section-label" style={{ marginRight: 8 }}>Risk Guide:</span>
        {[
          { color: "var(--accent-green)", label: "0–30%",  desc: "Calm — no concern" },
          { color: "var(--accent-amber)", label: "30–60%", desc: "Unusual activity — watch closely" },
          { color: "var(--accent-red)",   label: "60%+",   desc: "Warning — possible seizure" },
        ].map(({ color, label, desc }) => (
          <div key={label} style={{
            display: "flex", alignItems: "center", gap: 10,
            background: "rgba(10, 15, 30, 0.5)", border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)", padding: "8px 16px",
            backdropFilter: "blur(12px)",
          }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block", boxShadow: `0 0 12px ${color}` }} />
            <span className="display-font" style={{ fontSize: 14, fontWeight: 700, color }}>{label}</span>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{desc}</span>
          </div>
        ))}
      </div>

      {/* States */}
      {loading && (
        <div style={{ color: "var(--text-muted)", textAlign: "center", paddingTop: 80, fontSize: 15 }}>
          Connecting to monitoring system…
        </div>
      )}
      {error && (
        <div style={{
          background: "rgba(239,68,68,0.10)", border: "1px solid rgba(239,68,68,0.35)",
          borderRadius: 12, padding: "20px 24px", color: "#ef4444", maxWidth: 520,
        }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Monitoring System Offline</div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            {error}. Make sure the AI backend server is running.
          </div>
        </div>
      )}
      {!loading && !error && patients.length === 0 && (
        <div style={{ color: "var(--text-muted)", textAlign: "center", paddingTop: 80 }}>
          No patients are currently being monitored.
        </div>
      )}

      {/* Patient Cards Grid */}
      {patients.length > 0 && (
        <>
          <div className="section-label" style={{ marginBottom: 16, color: "var(--text-secondary)" }}>
            {patients.length} Patient{patients.length !== 1 ? "s" : ""} Under Active Monitoring
          </div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 16,
          }}>
            {patients.map(pid => (
              <PatientCard key={pid} patientId={pid} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
