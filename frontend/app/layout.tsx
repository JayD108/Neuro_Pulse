import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroPulse — Live EEG Seizure Monitoring",
  description:
    "Real-time AI-powered EEG seizure prediction dashboard for Epilepsy Monitoring Units. Built on CHB-MIT dataset with classical ML ensemble achieving ROC-AUC 0.993.",
  keywords: ["EEG", "seizure prediction", "epilepsy", "brain monitoring", "AI", "XAI"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {/* Global nav bar */}
        <header style={{
          position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
          height: "60px",
          background: "rgba(3, 5, 12, 0.6)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "0 28px",
          boxShadow: "0 4px 30px rgba(0, 0, 0, 0.5)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            {/* Logo pulse dot */}
            <span style={{
              width: 12, height: 12, borderRadius: "50%",
              background: "var(--accent-teal)",
              boxShadow: "0 0 12px var(--accent-teal), 0 0 24px var(--accent-teal)",
              animation: "pulse-badge-red 2.5s ease-in-out infinite",
              display: "inline-block",
            }} />
            <span className="display-font" style={{ fontWeight: 800, fontSize: "20px", letterSpacing: "0.02em" }}>
              Neuro<span className="neon-teal">Pulse</span>
            </span>
            <span className="section-label" style={{ marginLeft: 12, marginTop: 4 }}>
              Live EMU Dashboard
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
            <a href="/" className="hover-lift" style={{
              fontSize: "13px", color: "var(--text-secondary)", textDecoration: "none",
              padding: "6px 16px", borderRadius: "8px",
              border: "1px solid var(--border-subtle)",
              background: "rgba(255, 255, 255, 0.02)",
              transition: "all 0.3s ease",
            }}>
              Live Monitor
            </a>
            <a href="/dataset" className="hover-lift" style={{
              fontSize: "13px", color: "var(--text-secondary)", textDecoration: "none",
              padding: "6px 16px", borderRadius: "8px",
              border: "1px solid var(--border-subtle)",
              background: "rgba(255, 255, 255, 0.02)",
              transition: "all 0.3s ease",
            }}>
              Dataset Info
            </a>
            <a href="/guide" className="hover-lift" style={{
              fontSize: "13px", textDecoration: "none",
              padding: "6px 16px", borderRadius: "8px",
              border: "1px solid rgba(0,240,255,0.4)",
              color: "var(--accent-teal)",
              background: "rgba(0, 240, 255, 0.05)",
              boxShadow: "0 0 10px rgba(0,240,255,0.1)",
              transition: "all 0.3s ease",
            }}>
              How It Works
            </a>
            <span style={{
              padding: "5px 14px",
              background: "rgba(0,240,255,0.10)",
              border: "1px solid rgba(0,240,255,0.30)",
              borderRadius: "99px",
              fontSize: "12px", fontWeight: 700, color: "var(--accent-teal)",
              fontFamily: "var(--font-mono)",
              boxShadow: "0 0 15px rgba(0, 240, 255, 0.15)",
            }}>
              MODEL ROC-AUC 0.993
            </span>
          </div>
        </header>

        <main style={{ paddingTop: "60px", minHeight: "100vh" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
