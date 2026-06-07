import { NextResponse } from "next/server";

/**
 * Proxy bridge to the smart-wheel-engine read-only performance-viewer API
 * (design D26). Mirrors the /api/engine proxy pattern: forwards to the
 * Python engine on :8787 and never caches.
 *
 * GET /api/portfolio/{summary|positions|returns|income|risk|history}
 *   → ENGINE_API/api/portfolio/<sub>
 *
 * Read-only + observational: the engine endpoints behind this report the
 * held book + realized history. No order routing, no EV authority.
 */

const ENGINE_API = process.env.ENGINE_API_URL || "http://localhost:8787";

const ALLOWED = new Set([
  "summary",
  "positions",
  "returns",
  "income",
  "risk",
  "history",
]);

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ sub: string }> }
) {
  const { sub } = await params;

  if (!ALLOWED.has(sub)) {
    return NextResponse.json(
      { error: `Unknown portfolio view: ${sub}` },
      { status: 400 }
    );
  }

  try {
    const res = await fetch(`${ENGINE_API}/api/portfolio/${sub}`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });
    if (!res.ok) {
      // Surface the engine's status (e.g. 503 when the snapshot fixture is
      // absent) so the client can fall back to its typed mock.
      const detail = await res.text().catch(() => "");
      return NextResponse.json(
        { error: `Engine returned ${res.status}`, detail },
        { status: res.status }
      );
    }
    return NextResponse.json(await res.json());
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        error: "Engine unavailable",
        detail: message,
        hint: "Start the Python API server: python engine_api.py",
      },
      { status: 503 }
    );
  }
}
