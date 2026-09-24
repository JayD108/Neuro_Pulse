"use client";

import { useEffect, useState } from "react";
import { API_BASE_URL } from "@/lib/constants";

interface PatientStat {
  id: string;
  n_windows: number;
  n_preictal: number;
  n_interictal: number;
  n_features: number;
  seizure_count: number;
}

const SEIZURE_COUNTS: Record<string, number> = {
  chb01: 7,  chb02: 3,  chb03: 7,  chb04: 4,  chb05: 5,
  chb06: 10, chb07: 3,  chb08: 5,  chb09: 4,  chb10: 7,
  chb11: 3,  chb12: 40, chb13: 12, chb14: 8,  chb15: 20,
  chb16: 10, chb17: 3,  chb18: 6,  chb19: 3,  chb20: 8,
  chb21: 4,  chb22: 3,  chb23: 7,  chb24: 16,
};

export default function DatasetPage() {
  const [patients, setPatients] = useState<PatientStat[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/patients`)
      .then(r => r.json())
      .then(async data => {
        const ids: string[] = data.patients ?? [];
        const stats = await Promise.all(
          ids.map(id =>
            fetch(`${API_BASE_URL}/patient/${id}/info`)
              .then(r => r.json())
              .then(d => ({
                id,
                n_windows:    d.n_windows    ?? 0,
                n_preictal:   d.n_preictal   ?? 0,
                n_interictal: d.n_interictal ?? 0,
                n_features:   d.n_features   ?? 0,
                seizure_count: SEIZURE_COUNTS[id] ?? 0,
              }))
              .catch(() => ({
                id,
                n_windows: 0, n_preictal: 0, n_interictal: 0, n_features: 0,
                seizure_count: SEIZURE_COUNTS[id] ?? 0,
              }))
          )
        );
        setPatients(stats);
        setLoading(false);
      })
      .catch(() => { setError("Cannot reach backend."); setLoading(false); });
  }, []);

  const totalWindows   = patients.reduce((s, p) => s + p.n_windows, 0);
  const totalPreictal  = patients.reduce((s, p) => s + p.n_preictal, 0);
  const totalSeizures  = patients.reduce((s, p) => s + p.seizure_count, 0);

  return (
    <div style={{ padding: "40px 28px 80px", maxWidth: 1200, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: 36 }}>
        <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
          Research Dataset
        </div>
        <h1 style={{ fontSize: "clamp(24px, 4vw, 36px)", fontWeight: 800, margin: "0 0 14px", letterSpacing: "-0.02em" }}>
          CHB-MIT Scalp EEG Database
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.75, maxWidth: 720 }}>
          Collected at <strong style={{ color: "var(--text-primary)" }}>Boston Children's Hospital</strong> in collaboration with MIT. 24 pediatric patients with epilepsy, continuously monitored in the Epilepsy Monitoring Unit (EMU). Each recording session lasted multiple days, capturing both routine brain activity and spontaneous seizure events.
        </p>
      </div>

      {/* Summary Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14, marginBottom: 42 }}>
        {[
          { label: "Patients",        value: "24",                     sub: "Ages 1.5 – 22 years",        color: "var(--accent-teal)" },
          { label: "EDF Recordings",  value: "686",                    sub: "~1 hour each",               color: "#60a5fa" },
          { label: "Seizure Events",  value: `${totalSeizures}`,        sub: "Spontaneous clinical events", color: "#ef4444" },
          { label: "Feature Windows", value: totalWindows.toLocaleString(), sub: "10-second windows",      color: "#a78bfa" },
          { label: "Preictal Windows",value: totalPreictal.toLocaleString(), sub: "Danger-zone labels",    color: "#fb923c" },
          { label: "Features/Window", value: "595",                    sub: "Mathematical features",       color: "#34d399" },
        ].map(({ label, value, sub, color }) => (
          <div key={label} style={{ background: "var(--bg-panel)", border: "1px solid var(--border-subtle)", borderTop: `2px solid ${color}`, borderRadius: 10, padding: "18px 16px" }}>
            <div style={{ fontSize: 24, fontWeight: 800, color, fontFamily: "var(--font-mono)", marginBottom: 4 }}>{value}</div>
            <div style={{ fontSize: 12, fontWeight: 600 }}>{label}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>
          </div>
        ))}
      </div>

      {/* Patient Table */}
      <div style={{ background: "var(--bg-panel)", border: "1px solid var(--border-subtle)", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ fontWeight: 700, fontSize: 14 }}>Per-Patient Breakdown</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>All 24 patients — complete dataset</div>
        </div>

        {/* Table Header */}
        <div style={{ display: "grid", gridTemplateColumns: "100px 1fr 1fr 1fr 1fr 160px", gap: 0, padding: "10px 22px", fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)", background: "rgba(255,255,255,0.02)" }}>
          <span>Patient</span>
          <span style={{ textAlign: "right" }}>Seizures</span>
          <span style={{ textAlign: "right" }}>Total Windows</span>
          <span style={{ textAlign: "right" }}>Interictal</span>
          <span style={{ textAlign: "right" }}>Preictal</span>
          <span style={{ textAlign: "right", paddingRight: 4 }}>Imbalance Ratio</span>
        </div>

        {loading && (
          <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)", fontSize: 14 }}>
            Loading patient statistics from backend…
          </div>
        )}
        {error && (
          <div style={{ padding: "24px 22px", color: "#ef4444", fontSize: 13 }}>
            {error} Make sure the backend is running at localhost:8000.
          </div>
        )}

        {patients.map((p, i) => {
          const ratio = p.n_preictal > 0 ? (p.n_interictal / p.n_preictal).toFixed(1) : "—";
          const pctPreictal = p.n_windows > 0 ? (p.n_preictal / p.n_windows) * 100 : 0;
          return (
            <div key={p.id} style={{
              display: "grid",
              gridTemplateColumns: "100px 1fr 1fr 1fr 1fr 160px",
              gap: 0,
              padding: "13px 22px",
              fontSize: 13,
              borderBottom: "1px solid rgba(255,255,255,0.04)",
              background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
              alignItems: "center",
            }}>
              {/* Patient ID */}
              <a href={`/patient/${p.id}`} style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: "var(--accent-teal)", textDecoration: "none", fontSize: 13 }}>
                {p.id.toUpperCase()}
              </a>

              {/* Seizures */}
              <div style={{ textAlign: "right" }}>
                <span style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.25)", color: "#ef4444", borderRadius: 6, padding: "2px 10px", fontSize: 12, fontWeight: 700 }}>
                  ⚡ {p.seizure_count}
                </span>
              </div>

              {/* Total Windows */}
              <div style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {p.n_windows.toLocaleString()}
              </div>

              {/* Interictal */}
              <div style={{ textAlign: "right", color: "#22c55e", fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {p.n_interictal.toLocaleString()}
              </div>

              {/* Preictal */}
              <div style={{ textAlign: "right", color: "#fb923c", fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {p.n_preictal.toLocaleString()}
              </div>

              {/* Imbalance bar */}
              <div style={{ textAlign: "right", paddingRight: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "flex-end" }}>
                  <div style={{ flex: 1, height: 4, borderRadius: 4, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
                    <div style={{ width: `${Math.min(pctPreictal * 10, 100)}%`, height: "100%", background: "#fb923c", borderRadius: 4 }} />
                  </div>
                  <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)", minWidth: 40, textAlign: "right" }}>{ratio}:1</span>
                </div>
              </div>
            </div>
          );
        })}

        {/* Footer note */}
        <div style={{ padding: "14px 22px", fontSize: 11, color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)" }}>
          Imbalance Ratio = Interictal windows ÷ Preictal windows. A higher ratio means the patient had relatively fewer seizure warning periods captured. All class imbalance is handled in training with <code style={{ fontSize: 10 }}>scale_pos_weight</code> (XGBoost) and <code style={{ fontSize: 10 }}>class_weight="balanced"</code> (LightGBM / Random Forest).
        </div>
      </div>

      {/* Feature breakdown */}
      <div style={{ marginTop: 42, background: "var(--bg-panel)", border: "1px solid var(--border-subtle)", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border-subtle)", fontWeight: 700, fontSize: 14 }}>Feature Engineering Pipeline</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 0 }}>
          {[
            { group: "Welch PSD (Frequency)", count: "10 × 18 = 180", color: "#a78bfa", desc: "Absolute + relative power in 5 frequency bands (δ θ α β γ) for each of 18 channels" },
            { group: "Hjorth Parameters", count: "3 × 18 = 54", color: "#60a5fa", desc: "Activity, Mobility, Complexity — pure time-domain complexity descriptors" },
            { group: "Signal Stats", count: "2 × 18 = 36", color: "#34d399", desc: "RMS amplitude and Zero-Crossing Rate per channel" },
            { group: "Wavelet Energy", count: "6 × 18 = 108", color: "#fb923c", desc: "db4 discrete wavelet transform energy at 6 decomposition levels per channel" },
            { group: "Sample Entropy", count: "1 × 18 = 18", color: "#f472b6", desc: "Numba JIT compiled — measures signal predictability (collapses before seizures)" },
            { group: "Higuchi Fractal Dim.", count: "1 × 18 = 18", color: "#facc15", desc: "Self-similarity index — higher = more complex and healthy brain activity" },
            { group: "Cross-Channel Coherence", count: "8 pairs × 4 bands = 32", color: "#e879f9", desc: "Spectral coherence between 8 hemisphere pairs in δ θ α β bands" },
            { group: "Band Power Ratios", count: "8 ratios × 18 = 144", color: "#38bdf8", desc: "Clinical ratios like δ/α and (δ+θ)/(α+β) that shift dramatically preictally" },
          ].map(({ group, count, color, desc }, i) => (
            <div key={group} style={{ padding: "18px 20px", borderRight: "1px solid var(--border-subtle)", borderBottom: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 4 }}>{group}</div>
              <div style={{ fontSize: 18, fontWeight: 800, color, fontFamily: "var(--font-mono)", marginBottom: 6 }}>{count}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.6 }}>{desc}</div>
            </div>
          ))}
        </div>
        <div style={{ padding: "14px 22px", borderTop: "1px solid var(--border-subtle)", fontSize: 12, color: "var(--text-muted)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>Total: 180 + 54 + 36 + 108 + 18 + 18 + 32 + 144 = <strong style={{ color: "var(--text-primary)" }}>590 base features</strong> (up to 1,365 with extended montage patients)</span>
          <span style={{ fontSize: 11, color: "var(--accent-teal)" }}>Window: 10 sec • 256 Hz • No overlap</span>
        </div>
      </div>

    </div>
  );
}
