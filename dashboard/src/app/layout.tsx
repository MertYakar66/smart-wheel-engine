import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YAKAR TERMINAL",
  description:
    "Bloomberg-style financial terminal combining news aggregation, options engine, and AI agent",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
