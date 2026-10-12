// APALA — shared constants

export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Smoothed risk at or above this value raises an alert. */
export const ALERT_THRESHOLD = 0.60;
/** Smoothed risk at or above this value is shown as "Watch". */
export const WATCH_THRESHOLD = 0.30;

export const CHANNELS = [
  "FP1-F7","F7-T7","T7-P7","P7-O1",
  "FP1-F3","F3-C3","C3-P3","P3-O1",
  "FP2-F4","F4-C4","C4-P4","P4-O2",
  "FP2-F8","F8-T8","T8-P8","P8-O2",
  "FZ-CZ","CZ-PZ",
] as const;

export const BANDS = ["Delta","Theta","Alpha","Beta","Gamma"] as const;

export const BAND_INFO: Record<string, { range: string; color: string }> = {
  Delta: { range: "0.5–4 Hz", color: "#5b7fd6" },
  Theta: { range: "4–8 Hz",   color: "#4ea6a6" },
  Alpha: { range: "8–13 Hz",  color: "#7fae5a" },
  Beta:  { range: "13–30 Hz", color: "#c9a14a" },
  Gamma: { range: "30–50 Hz", color: "#c46b5c" },
};

/**
 * Electrode positions of the international 10-20 system, top view
 * (nose up), on a unit circle. Used by the 2D electrode map.
 */
export const ELECTRODES_1020: Record<string, [number, number]> = {
  FP1: [-0.31,  0.95], FP2: [ 0.31,  0.95],
  F7:  [-0.81,  0.59], F3:  [-0.40,  0.52], FZ: [0,  0.47], F4: [0.40,  0.52], F8: [0.81,  0.59],
  T7:  [-0.98,  0.00], C3:  [-0.49,  0.00], CZ: [0,  0.00], C4: [0.49,  0.00], T8: [0.98,  0.00],
  P7:  [-0.81, -0.59], P3:  [-0.40, -0.52], PZ: [0, -0.47], P4: [0.40, -0.52], P8: [0.81, -0.59],
  O1:  [-0.31, -0.95], O2:  [ 0.31, -0.95],
};

export type RiskLevel = "normal" | "watch" | "alert";

export function riskLevel(score: number): RiskLevel {
  if (score >= ALERT_THRESHOLD) return "alert";
  if (score >= WATCH_THRESHOLD) return "watch";
  return "normal";
}

export const RISK_LEVEL_INFO: Record<RiskLevel, { label: string; color: string }> = {
  normal: { label: "Normal",   color: "#3fb27f" },
  watch:  { label: "Watch",    color: "#d9a441" },
  alert:  { label: "Alert",    color: "#e5534b" },
};

export function riskColor(score: number): string {
  return RISK_LEVEL_INFO[riskLevel(score)].color;
}

/** Seconds -> "h:mm:ss" (recording time of a window). */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

const FEATURE_LABELS: Array<[RegExp, string]> = [
  [/^(Delta|Theta|Alpha|Beta|Gamma)_relpow$/, "$1 relative power"],
  [/^(Delta|Theta|Alpha|Beta|Gamma)(_power)?$/, "$1 power"],
  [/^hjorth_(activity|mobility|complexity)$/, "Hjorth $1"],
  [/^rms$/,               "RMS amplitude"],
  [/^zero_crossings$/,    "Zero-crossing rate"],
  [/^sample_entropy$/,    "Sample entropy"],
  [/^higuchi_fd$/,        "Higuchi fractal dimension"],
  [/^wt_([AD]\d)_energy$/, "Wavelet $1 energy"],
  [/^theta_alpha_ratio$/, "Theta/alpha ratio"],
  [/^slow_fast_ratio$/,   "Slow/fast ratio"],
  [/^beta_alpha_ratio$/,  "Beta/alpha ratio"],
];

/** "FP1-F7_hjorth_mobility" -> { channel: "FP1-F7", label: "Hjorth mobility" } */
export function describeFeature(name: string): { channel: string; label: string } {
  // Channel names appear as "FP1-F7" and, in the ratio columns, as "FP1_F7".
  const coh = name.match(/^coh_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z0-9]+)_([A-Z0-9]+)_(delta|theta|alpha|beta)$/);
  if (coh) {
    const band = coh[5].charAt(0).toUpperCase() + coh[5].slice(1);
    return { channel: `${coh[1]}-${coh[2]} ↔ ${coh[3]}-${coh[4]}`, label: `${band} coherence` };
  }
  const m = name.match(/^([A-Z0-9]+[-_][A-Z0-9]+)(?:[-_]\d+)?_(.+)$/);
  if (!m) return { channel: "", label: name.replace(/_/g, " ") };
  const channel = m[1].replace("_", "-");
  const suffix = m[2];
  for (const [re, label] of FEATURE_LABELS) {
    if (re.test(suffix)) return { channel, label: suffix.replace(re, label) };
  }
  return { channel, label: suffix.replace(/_/g, " ") };
}
