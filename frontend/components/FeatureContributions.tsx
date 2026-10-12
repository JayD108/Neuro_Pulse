"use client";

import { TopFeature } from "@/hooks/usePatientStream";
import { describeFeature } from "@/lib/constants";

interface FeatureContributionsProps {
  features: TopFeature[];
}

/**
 * The features driving the current window's score: model feature
 * importance multiplied by how far the value is from the training mean.
 */
export default function FeatureContributions({ features }: FeatureContributionsProps) {
  if (!features || features.length === 0) {
    return <div className="panel-note" style={{ display: "grid", placeItems: "center", height: "100%" }}>Waiting for data…</div>;
  }

  const rows = features.slice(0, 8);
  const max = Math.max(...rows.map(f => f.score ?? 0), 1e-9);

  return (
    <table className="data-table" style={{ fontSize: 12 }}>
      <thead>
        <tr>
          <th style={{ padding: "6px 8px", background: "transparent" }}>Channel</th>
          <th style={{ padding: "6px 8px", background: "transparent" }}>Feature</th>
          <th style={{ padding: "6px 8px", background: "transparent", width: "38%" }}>Contribution</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(f => {
          const { channel, label } = describeFeature(f.name);
          const w = ((f.score ?? 0) / max) * 100;
          return (
            <tr key={f.name} title={`${f.name} = ${f.value.toPrecision(4)}`}>
              <td className="mono" style={{ padding: "6px 8px", color: "var(--text-primary)" }}>{channel || "—"}</td>
              <td style={{ padding: "6px 8px" }}>{label}</td>
              <td style={{ padding: "6px 8px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ flex: 1, height: 6, background: "var(--border-subtle)", borderRadius: 3 }}>
                    <div style={{ width: `${w}%`, height: "100%", background: "var(--accent)", borderRadius: 3 }} />
                  </div>
                  <span className="mono" style={{ width: 34, textAlign: "right", color: "var(--text-muted)" }}>{w.toFixed(0)}</span>
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
