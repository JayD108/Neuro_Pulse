import type { Metadata } from "next";
import "./globals.css";
import NavBar from "@/components/NavBar";

export const metadata: Metadata = {
  title: "APALA — Pre-ictal EEG Monitoring",
  description:
    "Research dashboard for pre-ictal EEG detection on the CHB-MIT Scalp EEG Database. Streams recorded EEG features through the V1 classical model and shows seizure risk per patient.",
  keywords: ["EEG", "seizure prediction", "pre-ictal", "epilepsy", "CHB-MIT"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        <main style={{ paddingTop: 56, minHeight: "100vh" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
