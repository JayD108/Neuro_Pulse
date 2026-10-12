"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { WS_BASE_URL, riskLevel, type RiskLevel } from "@/lib/constants";

export interface TopFeature {
  name: string;
  value: number;
  /** Size of this feature's contribution to the window's score */
  score?: number;
  /** "up" pushes towards pre-ictal, "down" away from it */
  direction?: "up" | "down";
}

export interface TelemetryFrame {
  timestamp: number;
  patient_id: string;
  window_idx: number;
  window_start: number;
  window_end: number;
  /** Smoothed risk (top-K mean over recent windows) */
  risk_score: number;
  /** Unsmoothed model output for this window */
  risk_score_raw: number;
  soft_score: number;
  label_pred: number;
  /** Annotated state from the dataset: 0 = interictal, 1 = pre-ictal */
  label_true: number | null;
  /** True once the smoothed risk has stayed in the alert band for several windows */
  alert: boolean;
  /** Relative power per band, averaged over channels (fractions of total power) */
  band_power: Record<string, number>;
  top_features: TopFeature[];
  /** Per-channel deviation from the training baseline, 0–1 */
  channel_risk: Record<string, number>;
  error?: string;
}

interface UsePatientStreamOptions {
  maxHistory?: number;
  onAlert?: (frame: TelemetryFrame) => void;
  /** Number of consecutive windows required to change the alert state. Default 3 */
  persistenceWindows?: number;
}

interface UsePatientStreamReturn {
  frame: TelemetryFrame | null;
  history: TelemetryFrame[];
  connected: boolean;
  error: string | null;
  reconnect: () => void;
}

// Top-K soft fusion: mean of the K highest raw scores in the last N windows.
const FUSION_WINDOW = 10;
const FUSION_TOP_K  = 8;

export function usePatientStream(
  patientId: string | null,
  options: UsePatientStreamOptions = {}
): UsePatientStreamReturn {
  const { maxHistory = 300, onAlert, persistenceWindows = 3 } = options;

  const [frame, setFrame]         = useState<TelemetryFrame | null>(null);
  const [history, setHistory]     = useState<TelemetryFrame[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const wsRef        = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef   = useRef(true);
  const connectRef   = useRef<() => void>(() => {});

  // Smoothing state (refs: no re-render cost)
  const bufferRef       = useRef<number[]>([]);
  const levelBufRef     = useRef<RiskLevel[]>([]);
  const confirmedRef    = useRef<RiskLevel>("normal");

  const connect = useCallback(() => {
    if (!patientId || !mountedRef.current) return;
    if (reconnectRef.current) clearTimeout(reconnectRef.current);
    if (wsRef.current) wsRef.current.close();

    bufferRef.current   = [];
    levelBufRef.current = [];

    const ws = new WebSocket(`${WS_BASE_URL}/ws/${patientId}`);
    wsRef.current = ws;

    // Events from a socket that has since been replaced are ignored; otherwise
    // an old socket's close would trigger a reconnect that kills the new one.
    const isCurrent = () => mountedRef.current && wsRef.current === ws;

    ws.onopen = () => {
      if (!isCurrent()) return;
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!isCurrent()) return;
      try {
        const raw = JSON.parse(event.data as string);
        if (raw.error) setError(raw.error);
        const rawScore: number = raw.risk_score ?? 0;

        const buffer = bufferRef.current;
        buffer.push(rawScore);
        if (buffer.length > FUSION_WINDOW) buffer.shift();

        let smoothed = rawScore;
        if (buffer.length >= FUSION_TOP_K) {
          const topK = [...buffer].sort((a, b) => b - a).slice(0, FUSION_TOP_K);
          smoothed = topK.reduce((a, b) => a + b, 0) / topK.length;
        }

        // Persistence filter: the level only changes when all recent windows agree
        const level = riskLevel(smoothed);
        levelBufRef.current.push(level);
        if (levelBufRef.current.length > persistenceWindows) levelBufRef.current.shift();
        if (levelBufRef.current.every(l => l === level)) confirmedRef.current = level;

        const data: TelemetryFrame = {
          ...raw,
          risk_score:     smoothed,
          risk_score_raw: rawScore,
          alert:          confirmedRef.current === "alert",
        };

        setFrame(data);
        setHistory(prev => {
          const next = [...prev, data];
          return next.length > maxHistory ? next.slice(next.length - maxHistory) : next;
        });
        if (data.alert && onAlert) onAlert(data);
      } catch {
        // malformed frame — ignore
      }
    };

    ws.onerror = () => {
      if (!isCurrent()) return;
      setError("Connection to the monitoring server failed");
      setConnected(false);
    };

    ws.onclose = () => {
      if (!isCurrent()) return;
      setConnected(false);
      reconnectRef.current = setTimeout(() => {
        if (mountedRef.current) connectRef.current();
      }, 3000);
    };
  }, [patientId, maxHistory, onAlert, persistenceWindows]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) ws.close();
    };
  }, [connect]);

  return { frame, history, connected, error, reconnect: connect };
}
