"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/",        label: "Patient Monitor" },
  { href: "/dataset", label: "Dataset" },
  { href: "/guide",   label: "Methods" },
];

export default function NavBar() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" || pathname.startsWith("/patient") : pathname.startsWith(href);

  return (
    <header style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      height: 56,
      background: "var(--bg-panel)",
      borderBottom: "1px solid var(--border-subtle)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 24px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
        <Link href="/" style={{ display: "flex", alignItems: "baseline", gap: 10, textDecoration: "none" }}>
          <span style={{ fontWeight: 700, fontSize: 17, letterSpacing: "0.12em", color: "var(--text-primary)" }}>
            APALA
          </span>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Pre-ictal EEG Monitoring
          </span>
        </Link>

        <nav style={{ display: "flex", alignItems: "stretch", height: 56 }}>
          {LINKS.map(({ href, label }) => {
            const active = isActive(href);
            return (
              <Link key={href} href={href} style={{
                display: "flex", alignItems: "center",
                padding: "0 14px",
                fontSize: 13, fontWeight: 500,
                textDecoration: "none",
                color: active ? "var(--text-primary)" : "var(--text-secondary)",
                borderBottom: `2px solid ${active ? "var(--accent)" : "transparent"}`,
              }}>
                {label}
              </Link>
            );
          })}
        </nav>
      </div>

      <span style={{
        fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase",
        color: "var(--text-muted)",
        border: "1px solid var(--border-strong)", borderRadius: 4, padding: "3px 8px",
      }}>
        Research prototype · Not for clinical use
      </span>
    </header>
  );
}
