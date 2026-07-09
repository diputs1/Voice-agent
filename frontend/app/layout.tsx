import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vin Agent",
  description: "Voice Q&A for Vinpearl Safari Phu Quoc",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}

