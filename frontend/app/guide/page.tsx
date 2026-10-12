import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Methods — APALA",
  description:
    "How APALA V1 labels, processes and scores scalp EEG: data, pre-ictal labelling, features, model, alerting and results on the CHB-MIT database.",
};

const SECTIONS = [
  ["problem",  "Problem"],
  ["data",     "Data"],
  ["labels",   "Labelling"],
  ["features", "Signal processing and features"],
  ["model",    "Model"],
  ["alerting", "Smoothing and alerting"],
  ["results",  "Results"],
  ["limits",   "Limitations and next steps"],
] as const;

export default function MethodsPage() {
  return (
    <div className="page-narrow">
      <div className="section-label" style={{ marginBottom: 6 }}>Methods</div>
      <h1 className="page-title" style={{ fontSize: 28, marginBottom: 10 }}>How APALA V1 works</h1>
      <p className="lead" style={{ marginBottom: 28 }}>
        A technical summary of the pipeline behind this dashboard, from raw scalp EEG to the risk score on each
        patient card. Full details are in <span className="mono">Docs/report.txt</span>.
      </p>

      <nav className="panel" style={{ padding: "12px 16px", marginBottom: 40 }}>
        <ol style={{ listStyle: "none", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 24px" }}>
          {SECTIONS.map(([id, title], i) => (
            <li key={id} style={{ fontSize: 13 }}>
              <span className="mono" style={{ color: "var(--text-muted)", marginRight: 8 }}>{i + 1}</span>
              <a href={`#${id}`} style={{ textDecoration: "none" }}>{title}</a>
            </li>
          ))}
        </ol>
      </nav>

      <Section id="problem" n={1} title="Problem">
        <p>
          Seizure <em>detection</em> recognises a seizure once it has started. Seizure <em>forecasting</em>, the
          goal here, recognises the <strong>pre-ictal</strong> state: the period before onset in which the EEG
          changes but the patient shows no clinical signs. A reliable warning would leave time to call staff,
          secure the patient or give medication.
        </p>
      </Section>

      <Section id="data" n={2} title="Data">
        <p>
          CHB-MIT Scalp EEG Database (Boston Children&apos;s Hospital and MIT, PhysioNet): 24 patients aged 1.5–22,
          686 EDF recordings (about 686 hours) and 198 expert-annotated seizures.
        </p>
        <KeyValue rows={[
          ["Sampling rate", "256 Hz, 16-bit"],
          ["Montage", "International 10-20 system, bipolar pairs"],
          ["Channels", "18–23 per recording; 18 used"],
          ["Ground truth", "Seizure onset and offset times from each patient's summary file"],
        ]} />
      </Section>

      <Section id="labels" n={3} title="Labelling">
        <p>Each 10-second window receives one label relative to the nearest annotated seizure:</p>
        <table className="data-table" style={{ margin: "12px 0" }}>
          <thead><tr><th>Label</th><th>Definition</th><th>Use</th></tr></thead>
          <tbody>
            <tr><td style={{ color: "var(--text-primary)" }}>Interictal</td><td>At least 30 min from any seizure</td><td>Class 0</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>Pre-ictal</td><td>Within 30 min before a seizure onset</td><td>Class 1 (target)</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>Ictal</td><td>Overlaps a seizure</td><td>Excluded</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>Post-ictal</td><td>Within 5 min after a seizure ends</td><td>Excluded</td></tr>
          </tbody>
        </table>
        <p>
          The result is 347,075 windows: 324,763 interictal and 22,312 pre-ictal, an imbalance of 14.6 : 1.
        </p>
      </Section>

      <Section id="features" n={4} title="Signal processing and features">
        <p>
          Each recording is band-pass filtered at 0.5–50 Hz (4th-order Butterworth, second-order sections) with a
          60 Hz notch, then cut into non-overlapping 10-second windows (2,560 samples per channel). For an
          18-channel recording, 590 features are computed per window:
        </p>
        <KeyValue rows={[
          ["Spectral", "Welch band power, absolute and relative, in δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–50 Hz"],
          ["Time-domain", "Hjorth activity, mobility and complexity; RMS amplitude; zero-crossing rate"],
          ["Multi-scale", "db4 discrete wavelet energy at levels A5 and D1–D5"],
          ["Non-linear", "Sample entropy (Numba-compiled, 158× faster) and Higuchi fractal dimension"],
          ["Connectivity", "Magnitude-squared coherence for 8 channel pairs in 4 bands, incl. inter-hemispheric"],
          ["Ratios", "θ/α, (δ+θ)/(α+β), β/α and relative powers per channel"],
        ]} />
      </Section>

      <Section id="model" n={5} title="Model">
        <p>
          <strong>Dashboard model.</strong> The live dashboard uses an XGBoost classifier (500 trees, depth 6) trained
          on all 24 patients and all 590 features, saved in XGBoost&apos;s native format. Features are used unscaled;
          channels missing from a recording are left as missing values, which XGBoost handles natively. It outputs
          the probability that a window is pre-ictal. If this model is absent, the original 180-feature V1 model
          (trained on chb01) is used instead.
        </p>
        <p>
          <strong>Research ensemble.</strong> V1 also trained a two-level stacking ensemble on the full feature
          set: XGBoost (500 trees, GPU), LightGBM (500 trees) and Random Forest (300 trees), combined by a
          logistic-regression meta-learner trained on out-of-fold predictions. Class imbalance is handled with
          <span className="mono"> scale_pos_weight</span> and balanced class weights.
        </p>
        <p>
          <strong>Explanations.</strong> For each window the dashboard lists the features with the largest
          contribution to the score (XGBoost&apos;s exact per-feature contributions, with direction) and shades each
          channel by its share of the total. They explain the model&apos;s output, not the cause of a seizure.
        </p>
      </Section>

      <Section id="alerting" n={6} title="Smoothing and alerting">
        <KeyValue rows={[
          ["Smoothing", "Mean of the 8 highest window scores among the last 10 windows"],
          ["Watch", "Smoothed risk ≥ 30%"],
          ["Alert", "Smoothed risk ≥ 60%, sustained for 3 consecutive windows"],
        ]} />
      </Section>

      <Section id="results" n={7} title="Results">
        <p>V1 performs very differently depending on whether the test patients were seen in training:</p>
        <table className="data-table" style={{ margin: "12px 0" }}>
          <thead>
            <tr><th>Evaluation</th><th className="r">ROC-AUC</th><th className="r">Sensitivity</th><th className="r">Specificity</th></tr>
          </thead>
          <tbody>
            <tr><td style={{ color: "var(--text-primary)" }}>XGBoost · one known patient (chb01)</td><td className="r mono">0.998</td><td className="r mono">94.3%</td><td className="r mono">99.5%</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>Ensemble · held-out windows (3,585)</td><td className="r mono">0.993</td><td className="r mono">92.3%</td><td className="r mono">97.6%</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>Ensemble · unseen patients (train chb01–10, test chb11–24)</td><td className="r mono">0.595</td><td className="r mono">5.7%</td><td className="r mono">98.1%</td></tr>
            <tr><td style={{ color: "var(--text-primary)" }}>XGBoost · leave-one-patient-out, 24 patients (mean)</td><td className="r mono">0.560</td><td className="r mono">27.9%</td><td className="r mono">77.7%</td></tr>
          </tbody>
        </table>
        <div className="notice warn">
          On patients it has never seen, V1 is only slightly better than chance: leave-one-patient-out ROC-AUC ranges
          from 0.26 (chb01) to 0.71 (chb08), median 0.59, and a fixed 0.5 threshold gives 0–86% sensitivity depending on
          the patient. Each patient&apos;s held-out score is shown on its page and on the Dataset page. Generalising to
          new patients is the central open problem and the reason for the APALA rebuild.
        </div>
      </Section>

      <Section id="limits" n={8} title="Limitations and next steps">
        <ul style={{ paddingLeft: 18, listStyle: "disc" }}>
          <li>The dashboard replays recorded data. Scores on patients the model was trained on are optimistic.</li>
          <li>Window-level metrics do not equal clinical metrics such as false alarms per hour or warning time.</li>
          <li>CHB-MIT is a paediatric cohort from a single centre.</li>
          <li>This is a research prototype and must not be used for clinical decisions.</li>
        </ul>
        <p style={{ marginTop: 12 }}>
          The APALA rebuild evaluates leave-one-patient-out across all 24 cases, keeps every preprocessing step
          inside the training folds to prevent leakage, and tests a multi-scale CNN with squeeze-and-excitation
          blocks and a KAN classification head against strong baselines.
        </p>
      </Section>
    </div>
  );
}

function Section({ id, n, title, children }: { id: string; n: number; title: string; children: React.ReactNode }) {
  return (
    <section id={id} style={{ marginBottom: 40, scrollMarginTop: 72 }}>
      <h2 style={{ fontSize: 18, marginBottom: 12, display: "flex", gap: 12, alignItems: "baseline" }}>
        <span className="mono" style={{ color: "var(--text-muted)", fontSize: 14 }}>{n}</span>
        {title}
      </h2>
      <div style={{ color: "var(--text-secondary)", fontSize: 14, lineHeight: 1.75, display: "grid", gap: 10 }}>
        {children}
      </div>
    </section>
  );
}

function KeyValue({ rows }: { rows: Array<[string, string]> }) {
  return (
    <table className="data-table" style={{ margin: "4px 0" }}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ width: 160, color: "var(--text-primary)", fontWeight: 500, verticalAlign: "top" }}>{k}</td>
            <td>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
