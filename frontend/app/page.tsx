"use client";

import { useEffect, useState, useCallback } from "react";
import PatientCard from "@/components/PatientCard";
import { API_BASE_URL, RISK_LEVEL_INFO, ALERT_THRESHOLD, WATCH_THRESHOLD } from "@/lib/constants";

export default function HomePage() {
  const [patients, setPatients] = useState<string[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);

  const fetchPatients = useCallback((quiet = false) => {
    fetch(`${API_BASE_URL}/patients`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        setPatients(data.patients ?? []);
        setError(null);
        setLoading(false);
      })
      .catch(() => {
        if (!quiet) setError(`Cannot reach the monitoring server at ${API_BASE_URL}`);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    fetchPatients();
    const interval = setInterval(() => fetchPatients(true), 5000);
    return () => clearInterval(interval);
  }, [fetchPatients]);

  const legend = [
    { ...RISK_LEVEL_INFO.normal, range: `< ${WATCH_THRESHOLD * 100}%` },
    { ...RISK_LEVEL_INFO.watch,  range: `${WATCH_THRESHOLD * 100}–${ALERT_THRESHOLD * 100}%` },
    { ...RISK_LEVEL_INFO.alert,  range: `≥ ${ALERT_THRESHOLD * 100}%, sustained 3 windows` },
  ];

  return (
    <div className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 24, flexWrap: "wrap", marginBottom: 20 }}>
        <div>
          <div className="section-label" style={{ marginBottom: 6 }}>Epilepsy Monitoring Unit · Replay</div>
          <h1 className="page-title" style={{ marginBottom: 8 }}>Patient Monitor</h1>
          <p className="lead">
            Recorded scalp EEG from the CHB-MIT database is replayed window by window (10 s each) through the
            V1 model. Each card shows the probability that the patient is in the pre-ictal state, up to 30 minutes
            before a seizure.
          </p>
        </div>

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          {legend.map(({ label, color, range }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <span className="status-dot" style={{ background: color }} />
              <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{label}</span>
              <span style={{ color: "var(--text-muted)" }}>{range}</span>
            </div>
          ))}
        </div>
      </div>

      {loading && <div className="panel-note" style={{ padding: "48px 0", textAlign: "center" }}>Connecting to the monitoring server…</div>}

      {error && (
        <div className="notice error" style={{ maxWidth: 640, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16 }}>
          <div>
            <div style={{ color: "var(--text-primary)", fontWeight: 600, marginBottom: 2 }}>Monitoring server offline</div>
            {error}. Start the backend (see README) and retry.
          </div>
          <button className="btn" onClick={() => fetchPatients()}>Retry</button>
        </div>
      )}

      {!loading && !error && patients.length === 0 && (
        <div className="notice warn" style={{ maxWidth: 640 }}>
          No patient data found. Download the processed feature files into
          <span className="mono"> V1_Classical_ML/ml/data/processed/</span> (see README).
        </div>
      )}

      {patients.length > 0 && (
        <>
          <div className="section-label" style={{ margin: "8px 0 12px" }}>
            {patients.length} patient{patients.length !== 1 ? "s" : ""}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: 12 }}>
            {patients.map(pid => <PatientCard key={pid} patientId={pid} />)}
          </div>
        </>
      )}
    </div>
  );
}
