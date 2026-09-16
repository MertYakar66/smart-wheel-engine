import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YAKAR TERMINAL",
  description:
    "Smart Wheel Engine dashboard — decision cockpit, live portfolio viewer, and options terminal",
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
