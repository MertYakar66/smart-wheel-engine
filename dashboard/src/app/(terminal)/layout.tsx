export default function TerminalGroupLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <div className="terminal-body">{children}</div>;
}
