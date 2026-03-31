import { Nav } from "@/components/nav";

export default function MainLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
    </>
  );
}
