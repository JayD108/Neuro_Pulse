interface SparklineProps {
  values: number[];
  color: string;
  height?: number;
}

/** Risk sparkline on a fixed 0–1 scale, so cards are comparable. */
export default function Sparkline({ values, color, height = 32 }: SparklineProps) {
  const n = Math.max(values.length, 2);
  const points = values.map((v, i) => `${(i / (n - 1)) * 100},${(1 - Math.min(Math.max(v, 0), 1)) * 100}`).join(" ");

  return (
    <svg width="100%" height={height} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" x2="100" y1="40" y2="40" stroke="var(--border-strong)" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
      {values.length > 1 && (
        <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      )}
    </svg>
  );
}
