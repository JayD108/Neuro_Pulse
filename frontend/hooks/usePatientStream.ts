/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { WS_BASE_URL } from "@/lib/constants";

export interface TelemetryFrame {
  timestamp: number;
  patient_id: string;
  window_idx: number;
  risk_score: number;
  risk_score_raw: number;   // unsmoothed, for charts
  label_pred: number;
  label_true: number | null;
  alert: boolean;
  band_power: Record<string, number>;
  top_shap: Array<{ feature: string; shap_value: number }>;
  channel_risk: Record<string, number>;
}

interface UsePatientStreamOptions {
  maxHistory?: number;
  onAlert?: (frame: TelemetryFrame) => void;
  /** EMA smoothing factor 0–1. Lower = smoother. Default 0.25 */
  emaAlpha?: number;
  /** Number of consecutive windows required to change alert state. Default 3 */
  persistenceWindows?: number;
}

interface UsePatientStreamReturn {
  frame: TelemetryFrame | null;
  history: TelemetryFrame[];
  connected: boolean;
  error: string | null;
  reconnect: () => void;
}

function getRiskCategory(score: number): "clear" | "watch" | "alert" {
  if (score >= 0.60) return "alert";
  if (score >= 0.30) return "watch";
  return "clear";
}

export function usePatientStream(
  patientId: string | null,
  options: UsePatientStreamOptions = {}
): UsePatientStreamReturn {
  const {
    maxHistory = 300,
    onAlert,
    emaAlpha = 0.25,
    persistenceWindows = 3,
  } = options;

  const [frame, setFrame]         = useState<TelemetryFrame | null>(null);
  const [history, setHistory]     = useState<TelemetryFrame[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const wsRef        = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef   = useRef(true);

  // Smoothing state (not React state — no re-render cost)
  const emaRef          = useRef<number | null>(null);
  const categoryBuf     = useRef<Array<"clear" | "watch" | "alert">>([]);
  const confirmedCatRef = useRef<"clear" | "watch" | "alert">("clear");

  const connect = useCallback(() => {
    if (!patientId || !mountedRef.current) return;
    if (wsRef.current) wsRef.current.close();

    // Reset smoothing on reconnect
    emaRef.current      = null;
    categoryBuf.current = [];

    const url = `${WS_BASE_URL}/ws/${patientId}`;
    const ws  = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return;
      try {
        const raw = JSON.parse(event.data as string);
        const rawScore: number = raw.risk_score;

        // --- EMA smoothing ---
        if (emaRef.current === null) {
          emaRef.current = rawScore;
        } else {
          emaRef.current = emaAlpha * rawScore + (1 - emaAlpha) * emaRef.current;
        }
        const smoothed = emaRef.current;

        // --- Persistence filter on alert category ---
        const newCat = getRiskCategory(smoothed);
        categoryBuf.current.push(newCat);
        if (categoryBuf.current.length > persistenceWindows) {
          categoryBuf.current.shift();
        }
        // Only change confirmed category if all buffered windows agree
        const allAgree = categoryBuf.current.every(c => c === newCat);
        if (allAgree) {
          confirmedCatRef.current = newCat;
        }

        // Build the enriched frame the UI sees
        const data: TelemetryFrame = {
          ...raw,
          risk_score:     smoothed,
          risk_score_raw: rawScore,
          alert:          confirmedCatRef.current === "alert",
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
      if (!mountedRef.current) return;
      setError("WebSocket connection error");
      setConnected(false);
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      reconnectRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, 3000);
    };
  }, [patientId, maxHistory, onAlert, emaAlpha, persistenceWindows]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { frame, history, connected, error, reconnect: connect };
}
