"use client";

import { use, useState, useEffect } from "react";
import Link from "next/link";
import { usePatientStream } from "@/hooks/usePatientStream";
import RiskHorizon from "@/components/RiskHorizon";
import BandPowerChart from "@/components/BandPowerChart";
import ElectrodeMap from "@/components/ElectrodeMap";
import FeatureContributions from "@/components/FeatureContributions";
import { API_BASE_URL, ALERT_THRESHOLD, riskLevel, RISK_LEVEL_INFO, formatClock } from "@/lib/constants";

interface PatientMeta {
  n_windows: number;
  n_preictal: number;
  n_interictal: number;
  model_loaded: boolean;
  model_type: string;
  lopo_auc?: number | null;
}

function Panel({ title, note, children, style }: {
  title: string; note?: string; children: React.ReactNode; style?: React.CSSProperties;
}) {
  return (
    <section className="panel" style={{ display: "flex", flexDirection: "column", minWidth: 0, ...style }}>
      <div className="panel-header">
        <span className="panel-title">{title}</span>
        {note && <span className="panel-note">{note}</span>}
      </div>
      <div className="panel-body" style={{ flex: 1, minHeight: 0 }}>{children}</div>
    </section>
  );
}

function Stat({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div className="section-label" style={{ marginBottom: 4 }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 600, color: color ?? "var(--text-primary)" }}>{value}</div>
      {sub && <div className="panel-note">{sub}</div>}
    </div>
  );
}

const STATUS_TEXT = {
  normal: "No pre-ictal pattern detected in recent windows.",
  watch:  "Risk elevated above the watch level. Review the trend and contributing channels.",
  alert:  `Smoothed risk has stayed at or above ${ALERT_THRESHOLD * 100}% for several windows. Pre-ictal pattern suspected.`,
};

export default function PatientPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { frame, history, connected, error } = usePatientStream(id);
  const [meta, setMeta] = useState<PatientMeta | null>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/patient/${id}/info`)
      .then(r => (r.ok ? r.json() : null))
      .then(setMeta)
      .catch(() => setMeta(null));
  }, [id]);

  const risk  = frame?.risk_score ?? 0;
  const level = frame?.alert ? "alert" : riskLevel(risk);
  const info  = RISK_LEVEL_INFO[level];
  const minutes = Math.max(1, Math.round((history.length * 10) / 60));

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Identity strip */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <Link href="/" className="btn" style={{ fontSize: 12 }}>← All patients</Link>
        <h1 className="mono" style={{ fontSize: 22, fontWeight: 600, letterSpacing: "0.04em" }}>{id.toUpperCase()}</h1>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-secondary)" }}>
          <span className="status-dot" style={{ background: connected ? "var(--status-normal)" : "var(--text-muted)" }} />
          {connected ? "Receiving data" : "Reconnecting…"}
        </span>
        {meta && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 20, fontSize: 12, color: "var(--text-muted)" }}>
            <span>Windows <span className="mono" style={{ color: "var(--text-secondary)" }}>{meta.n_windows.toLocaleString()}</span></span>
            <span>Annotated pre-ictal <span className="mono" style={{ color: "var(--text-secondary)" }}>{meta.n_preictal.toLocaleString()}</span></span>
            <span>Model <span style={{ color: "var(--text-secondary)" }}>{meta.model_type}</span></span>
            {meta.lopo_auc != null && (
              <span title="ROC-AUC on this patient when it was left out of training (leave-one-patient-out)">
                Held-out AUC <span className="mono" style={{ color: "var(--text-secondary)" }}>{meta.lopo_auc.toFixed(2)}</span>
              </span>
            )}
          </div>
        )}
      </div>

      {error && <div className="notice error">{error}</div>}
      {meta && !meta.model_loaded && (
        <div className="notice warn">
          The model is not loaded on the server, so risk scores are 0. Download the model files into
          <span className="mono"> V1_Classical_ML/models/</span> (see README).
        </div>
      )}

      {/* Current status */}
      <section className="panel" style={{ borderLeft: `3px solid ${frame ? info.color : "var(--border-strong)"}` }}>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 1.4fr) repeat(4, minmax(120px, 1fr))", gap: 24, padding: "16px 20px", alignItems: "center" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span className="status-tag" style={{ color: frame ? info.color : "var(--text-muted)" }}>
                {frame ? info.label : "No data"}
              </span>
              <span className="mono" style={{ fontSize: 28, fontWeight: 600, color: frame ? info.color : "var(--text-muted)" }}>
                {frame ? `${Math.round(risk * 100)}%` : "—"}
              </span>
            </div>
            <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              {frame ? STATUS_TEXT[level] : "Waiting for the first window…"}
            </div>
          </div>
          <Stat label="Window score" value={frame ? `${Math.round((frame.risk_score_raw ?? 0) * 100)}%` : "—"} sub="unsmoothed model output" />
          <Stat label="Recording time" value={frame ? formatClock(frame.window_start) : "—"} sub={frame ? `window ${frame.window_idx}` : undefined} />
          <Stat
            label="Annotated state"
            value={frame?.label_true == null ? "—" : frame.label_true === 1 ? "Pre-ictal" : "Interictal"}
            color={frame?.label_true === 1 ? "var(--status-alert)" : undefined}
            sub="from dataset labels"
          />
          <Stat label="Model decision" value={frame ? (frame.label_pred === 1 ? "Pre-ictal" : "Interictal") : "—"} sub="window score ≥ 50%" />
        </div>
      </section>

      {/* Monitoring grid */}
      <Panel title="Pre-ictal risk" note={`last ${minutes} min of recording · smoothed (blue), window score (grey), annotated pre-ictal (shaded)`}>
        <div style={{ height: 220 }}><RiskHorizon history={history} /></div>
      </Panel>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", gap: 12 }}>
        <Panel title="Channel deviation" note="10-20 bipolar montage">
          <div style={{ height: 280 }}><ElectrodeMap channelRisk={frame?.channel_risk ?? {}} /></div>
        </Panel>
        <Panel title="Relative band power" note="mean over channels">
          <div style={{ height: 280 }}><BandPowerChart history={history} /></div>
        </Panel>
        <Panel title="Contributing features" note="this window · ↑ towards pre-ictal">
          <FeatureContributions features={frame?.top_features ?? []} />
        </Panel>
      </div>

      <p className="panel-note" style={{ marginTop: 4 }}>
        Replay of recorded data for research. The model has seen this recording in training, so live scores are
        optimistic; the held-out AUC above shows how it did when this patient was left out (mean 0.56 across
        patients, see Methods). Not for clinical decisions.
      </p>
    </div>
  );
}
