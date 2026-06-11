// ─── Tradable-Universe Cache ──────────────────────────────────────────
// Server-side cache of the engine's S&P 500 universe UNION the held-book
// symbols (off-universe holds like CLS must stay linkable). Used to
// validate extracted ticker entities so regex junk ("TRUMP", "NATO")
// never becomes a /ticker link or an exposure-ranking match.

const ENGINE_API = process.env.ENGINE_API_URL || "http://localhost:8787";
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // universe changes ~quarterly

let cache: { symbols: Set<string>; fetchedAt: number } | null = null;

/**
 * Returns the validation set (engine universe + held-book symbols), or
 * null when the engine is unreachable AND no prior cache exists. Callers
 * must treat null as "cannot validate" — never as "nothing is valid".
 */
export async function getValidationUniverse(): Promise<Set<string> | null> {
  if (cache && Date.now() - cache.fetchedAt < CACHE_TTL_MS) {
    return cache.symbols;
  }

  try {
    const [uniRes, posRes] = await Promise.allSettled([
      fetch(`${ENGINE_API}/api/universe`, {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      }),
      fetch(`${ENGINE_API}/api/portfolio/positions`, {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      }),
    ]);

    let tickers: string[] = [];
    if (uniRes.status === "fulfilled" && uniRes.value.ok) {
      const json = await uniRes.value.json();
      if (Array.isArray(json?.tickers)) tickers = json.tickers;
    }
    // No universe -> serve the stale cache if we have one, else null.
    if (tickers.length === 0) return cache?.symbols ?? null;

    const symbols = new Set<string>(
      tickers.map((t) => String(t).toUpperCase())
    );

    if (posRes.status === "fulfilled" && posRes.value.ok) {
      const json = await posRes.value.json();
      const holdings: { sym?: string }[] = Array.isArray(json?.holdings)
        ? json.holdings
        : [];
      for (const h of holdings) {
        if (h.sym) symbols.add(String(h.sym).toUpperCase());
      }
    }

    cache = { symbols, fetchedAt: Date.now() };
    return symbols;
  } catch {
    return cache?.symbols ?? null;
  }
}
