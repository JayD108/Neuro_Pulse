// NeuroPulse — shared constants

export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const ALERT_THRESHOLD = 0.60;

export const CHANNELS = [
  "FP1-F7","F7-T7","T7-P7","P7-O1",
  "FP1-F3","F3-C3","C3-P3","P3-O1",
  "FP2-F4","F4-C4","C4-P4","P4-O2",
  "FP2-F8","F8-T8","T8-P8","P8-O2",
  "FZ-CZ","CZ-PZ",
] as const;

export const BANDS = ["Delta","Theta","Alpha","Beta","Gamma"] as const;

export const BAND_COLORS: Record<string, string> = {
  Delta: "#6366f1",
  Theta: "#00d4c8",
  Alpha: "#22c55e",
  Beta:  "#f59e0b",
  Gamma: "#ef4444",
};

// Maps each channel name to approximate XYZ position on a unit sphere
// for the 3D brain electrode viewer
export const ELECTRODE_POSITIONS: Record<string, [number,number,number]> = {
  "FP1-F7":  [-0.55, 0.75,  0.35],
  "F7-T7":   [-0.85, 0.30,  0.10],
  "T7-P7":   [-0.90,-0.20, -0.10],
  "P7-O1":   [-0.55,-0.70, -0.45],
  "FP1-F3":  [-0.30, 0.85,  0.40],
  "F3-C3":   [-0.50, 0.50,  0.60],
  "C3-P3":   [-0.55, 0.10,  0.75],
  "P3-O1":   [-0.40,-0.55,  0.60],
  "FP2-F4":  [ 0.30, 0.85,  0.40],
  "F4-C4":   [ 0.50, 0.50,  0.60],
  "C4-P4":   [ 0.55, 0.10,  0.75],
  "P4-O2":   [ 0.40,-0.55,  0.60],
  "FP2-F8":  [ 0.55, 0.75,  0.35],
  "F8-T8":   [ 0.85, 0.30,  0.10],
  "T8-P8":   [ 0.90,-0.20, -0.10],
  "P8-O2":   [ 0.55,-0.70, -0.45],
  "FZ-CZ":   [ 0.00, 0.40,  0.85],
  "CZ-PZ":   [ 0.00,-0.10,  0.95],
};

export function riskColor(score: number): string {
  if (score < 0.3)  return "#22c55e";   // green
  if (score < 0.6)  return "#f59e0b";   // amber
  return "#ef4444";                      // red
}
