// ─── Tradable-Universe Cache ──────────────────────────────────────────
// Server-side cache of the engine's S&P 500 universe UNION the held-book
// symbols (off-universe holds like CLS must stay linkable). Used to
// validate extracted ticker entities so regex junk ("TRUMP", "NATO")
// never becomes a /ticker link or an exposure-ranking match.
//
// Completeness contract:
//   complete=true  — BOTH the universe leg AND the positions leg succeeded.
//                    This set is authoritative for demotion: anything absent
//                    from it is genuinely not tradeable and not held.
//   complete=false — Only the universe leg succeeded; the positions leg
//                    failed (e.g. engine 503 on missing IBKR snapshot, which
//                    is the normal state on a fresh deploy). The set is safe
//                    for promoting new ticker entities (membership = good),
//                    but MUST NOT be used to demote existing ones — off-
//                    universe held names like CLS would be silently destroyed.
//   null           — Engine entirely unreachable and no prior cache exists.
//                    Cannot validate; callers must not touch existing rows.
//
// The 6 h TTL is only applied to complete sets. Partial sets are served for
// the immediate call (so new-story promotion works) but never written to the
// module-level cache, keeping the cache slot free for the next full fetch.
//
// Three-state convergence:
//   1. Complete set available:  new tickers promoted, bad tickers demoted,
//      past bad-demotion rows re-promoted by backfillTickerEntities.
//   2. Partial set (no positions): new tickers promoted (universe-safe),
//      demotion suppressed (cannot prove a held ticker is off-universe).
//   3. No engine:  new tickers held as 'topic'; no demotion; backfill skips.
//      Re-promotion fires on next boot or cache refresh with a complete set.

const ENGINE_API = process.env.ENGINE_API_URL || "http://localhost:8787";
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // universe changes ~quarterly

export interface ValidationUniverse {
  /** Uppercased symbols: engine universe ∪ held-book (when complete=true). */
  set: Set<string>;
  /**
   * true  — both the universe leg and the positions leg succeeded; safe for
   *         demotion (absence = genuinely not tradeable and not held).
   * false — universe leg succeeded but positions leg failed; use for
   *         promotion only; never demote on a partial set.
   */
  complete: boolean;
}

// Only complete sets are cached; partial sets are not promoted to the
// module-level cache so a later full fetch is not blocked by them.
let cache: { universe: ValidationUniverse; fetchedAt: number } | null = null;

/**
 * Returns a ValidationUniverse, or null when the engine is entirely
 * unreachable and no prior complete cache exists.
 *
 * Callers MUST check `complete` before using the set for demotion.
 * Never treat null as "nothing is valid".
 */
export async function getValidationUniverse(): Promise<ValidationUniverse | null> {
  if (cache && Date.now() - cache.fetchedAt < CACHE_TTL_MS) {
    return cache.universe;
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
    // No universe -> serve the stale complete cache if we have one, else null.
    // A stale complete set is better than nothing for both promotion and
    // demotion; partial (positions-missing) sets are never written to cache.
    if (tickers.length === 0) return cache?.universe ?? null;

    const set = new Set<string>(tickers.map((t) => String(t).toUpperCase()));

    const positionsOk =
      posRes.status === "fulfilled" && posRes.value.ok;

    if (positionsOk) {
      const json = await posRes.value.json();
      const holdings: { sym?: string }[] = Array.isArray(json?.holdings)
        ? json.holdings
        : [];
      for (const h of holdings) {
        if (h.sym) set.add(String(h.sym).toUpperCase());
      }
    }

    const universe: ValidationUniverse = { set, complete: positionsOk };

    // Cache only complete sets (both legs succeeded). A partial set is
    // returned for the immediate call but not persisted, so the next fetch
    // can still resolve into a complete set without waiting 6 h.
    if (positionsOk) {
      cache = { universe, fetchedAt: Date.now() };
    }

    return universe;
  } catch {
    return cache?.universe ?? null;
  }
}
