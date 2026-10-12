"use client";

import Link from "next/link";
import { useEffect, useState, useCallback } from "react";
import { API_BASE_URL } from "@/lib/constants";

interface PatientRow {
  patient_id: string;
  n_windows?: number;
  n_preictal?: number;
  n_interictal?: number;
  n_ictal?: number;
  lopo_auc?: number | null;
  error?: string;
}

interface DatasetStats {
  n_patients: number;
  total_windows: number;
  total_preictal: number;
  total_interictal: number;
  imbalance_ratio: number;
  patients: PatientRow[];
  error?: string;
}

// Annotated seizures per patient (CHB-MIT summary files); 198 in total.
const SEIZURE_COUNTS: Record<string, number> = {
  chb01: 7,  chb02: 3,  chb03: 7,  chb04: 4,  chb05: 5,
  chb06: 10, chb07: 3,  chb08: 5,  chb09: 4,  chb10: 7,
  chb11: 3,  chb12: 40, chb13: 12, chb14: 8,  chb15: 20,
  chb16: 10, chb17: 3,  chb18: 6,  chb19: 3,  chb20: 8,
  chb21: 4,  chb22: 3,  chb23: 7,  chb24: 16,
};

const FEATURE_GROUPS: Array<[string, string, number, string]> = [
  ["Welch PSD",            "10 × 18 channels", 180, "Absolute and relative power in δ θ α β γ"],
  ["Hjorth parameters",    "3 × 18",            54, "Activity, mobility, complexity"],
  ["Time-domain",          "2 × 18",            36, "RMS amplitude, zero-crossing rate"],
  ["Wavelet energy",       "6 × 18",           108, "db4 DWT, levels A5 and D1–D5"],
  ["Sample entropy",       "1 × 18",            18, "Signal regularity (Numba-compiled)"],
  ["Higuchi fractal dim.", "1 × 18",            18, "Waveform complexity"],
  ["Coherence",            "8 pairs × 4 bands", 32, "Magnitude-squared coherence, incl. inter-hemispheric pairs"],
  ["Band-power ratios",    "8 × 18",           144, "θ/α, (δ+θ)/(α+β), β/α, relative powers"],
];

function Tile({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="panel" style={{ padding: "14px 16px" }}>
      <div className="mono" style={{ fontSize: 22, fontWeight: 600 }}>{value}</div>
      <div style={{ fontSize: 13, fontWeight: 500, marginTop: 2 }}>{label}</div>
      <div className="panel-note">{sub}</div>
    </div>
  );
}

export default function DatasetPage() {
  const [stats, setStats]     = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const fetchStats = useCallback(() => {
    fetch(`${API_BASE_URL}/dataset/stats`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: DatasetStats) => { setStats(data); setError(null); })
      .catch(() => setError(`Cannot reach the monitoring server at ${API_BASE_URL}`))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  const rows = stats?.patients ?? [];

  return (
    <div className="page" style={{ maxWidth: 1200 }}>
      <div className="section-label" style={{ marginBottom: 6 }}>Dataset</div>
      <h1 className="page-title" style={{ marginBottom: 8 }}>CHB-MIT Scalp EEG Database</h1>
      <p className="lead" style={{ marginBottom: 24 }}>
        Long-term scalp EEG from 24 paediatric patients with intractable seizures, recorded at Boston Children&apos;s
        Hospital and published by MIT on PhysioNet. Seizure onsets and offsets were annotated by clinical experts.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 12, marginBottom: 28 }}>
        <Tile value="24"      label="Patients"        sub="ages 1.5–22" />
        <Tile value="686"     label="EDF recordings"  sub="~686 h, 256 Hz, 16-bit" />
        <Tile value="198"     label="Seizures"        sub="expert-annotated" />
        <Tile value="347,075" label="Windows"         sub="10 s, non-overlapping" />
        <Tile value="22,312"  label="Pre-ictal windows" sub="14.6 : 1 imbalance" />
        <Tile value="590"     label="Features / window" sub="18-channel montage (up to 1,365)" />
      </div>

      <section className="panel" style={{ marginBottom: 28, overflow: "hidden" }}>
        <div className="panel-header">
          <span className="panel-title">Per-patient windows</span>
          <span className="panel-note">
            {stats && !stats.error ? `${stats.n_patients} patient files on this server` : "from the processed feature files"}
          </span>
        </div>

        {loading && <div className="panel-body panel-note">Loading…</div>}
        {error && (
          <div className="panel-body" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16 }}>
            <span className="panel-note">{error}.</span>
            <button className="btn" onClick={() => { setLoading(true); setError(null); fetchStats(); }}>Retry</button>
          </div>
        )}
        {!loading && !error && rows.length === 0 && (
          <div className="panel-body panel-note">
            No processed feature files found in <span className="mono">V1_Classical_ML/ml/data/processed/</span>.
          </div>
        )}

        {rows.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th className="r">Seizures</th>
                <th className="r">Windows</th>
                <th className="r">Interictal</th>
                <th className="r">Pre-ictal</th>
                <th className="r">Seizure / post-ictal</th>
                <th className="r">Imbalance</th>
                <th className="r" title="ROC-AUC when this patient was left out of training">Held-out AUC</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(p => (
                <tr key={p.patient_id}>
                  <td>
                    <Link href={`/patient/${p.patient_id}`} className="mono" style={{ textDecoration: "none" }}>
                      {p.patient_id.toUpperCase()}
                    </Link>
                  </td>
                  <td className="r">{SEIZURE_COUNTS[p.patient_id] ?? "—"}</td>
                  <td className="r">{p.n_windows?.toLocaleString() ?? "—"}</td>
                  <td className="r">{p.n_interictal?.toLocaleString() ?? "—"}</td>
                  <td className="r">{p.n_preictal?.toLocaleString() ?? "—"}</td>
                  <td className="r">{p.n_ictal?.toLocaleString() ?? "—"}</td>
                  <td className="r">
                    {p.n_preictal ? `${((p.n_interictal ?? 0) / p.n_preictal).toFixed(1)} : 1` : "—"}
                  </td>
                  <td className="r mono" style={{ color: p.lopo_auc != null && p.lopo_auc < 0.5 ? "var(--status-watch)" : undefined }}>
                    {p.lopo_auc != null ? p.lopo_auc.toFixed(2) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="panel-note" style={{ padding: "10px 16px", borderTop: "1px solid var(--border-subtle)" }}>
          Imbalance = interictal ÷ pre-ictal windows. Seizure and post-ictal windows are excluded from training and replay.
          Held-out AUC = ROC-AUC on the patient when it was left out of training (leave-one-patient-out); below 0.5 is worse than chance.
        </div>
      </section>

      <section className="panel" style={{ overflow: "hidden" }}>
        <div className="panel-header">
          <span className="panel-title">Features per 10-second window</span>
          <span className="panel-note">bandpass 0.5–50 Hz · notch 60 Hz · 2,560 samples per channel</span>
        </div>
        <table className="data-table">
          <thead>
            <tr><th>Group</th><th>Count</th><th className="r">Features</th><th>Description</th></tr>
          </thead>
          <tbody>
            {FEATURE_GROUPS.map(([group, count, n, desc]) => (
              <tr key={group}>
                <td style={{ color: "var(--text-primary)" }}>{group}</td>
                <td className="mono">{count}</td>
                <td className="r mono">{n}</td>
                <td>{desc}</td>
              </tr>
            ))}
            <tr>
              <td style={{ color: "var(--text-primary)", fontWeight: 600 }}>Total</td>
              <td />
              <td className="r mono" style={{ color: "var(--text-primary)", fontWeight: 600 }}>590</td>
              <td>for an 18-channel recording; up to 1,365 with extended montages</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
