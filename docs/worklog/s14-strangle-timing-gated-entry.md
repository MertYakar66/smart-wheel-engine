---
id: S14
title: Strangle timing-gated entry
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the strangle timing engine
(`engine/strangle_timing.py`) — the one timing-gated strategy in
CLAUDE.md's NEVER list,
strategy, unexercised by every prior usage test — and answer the §2
question: does the strangle path ever produce a tradeable candidate
that bypasses `EVEngine.evaluate`, or is it purely a timing signal?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`. A 20-name diversified watchlist (10 GICS sectors,
mega-caps through large-caps), scored three ways:
`StrangleTimingEngine.score_entry` (Layer 1 — technical proxies),
`StrangleTimingWithIV.score_entry_with_iv` (Layer 2 — IV overlay), and
`WheelRunner.analyze_ticker`'s integrated `strangle_*` fields, plus
targeted near-earnings / high-IV / low-IV probes. One genuine bug fixed.

**Path.** `WheelRunner.strangle_engine` → `StrangleTimingWithIV` →
`score_entry_with_iv` / `scan_universe_with_iv` → a `StrangleEntryScore`
(0–100 `total_score`; `recommendation` ∈ {`strong_entry` ≥80,
`conditional` ≥60, `avoid`}; a `VolatilityPhase`). `analyze_ticker`
surfaces `strangle_score` / `strangle_phase` / `strangle_recommendation`.
`strangle_timing.py` holds **zero** references to `EVEngine` / `evaluate`.

**Status.** Done. §2 holds — *vacuously*: the strangle path produces a
timing score, never a strikes-and-premium candidate, so nothing reaches
a tradeable verdict to bypass the EV authority. But the strategy sits
**entirely outside the EV decision authority** — there is no strangle
analogue of `rank_candidates_by_ev`. One genuine bug fixed (the dead
Layer-2 IV overlay); the rest logged.

**Findings:**

- **§2 — the strangle path is a standalone timing surface with no EV
  layer.** `score_entry` / `score_entry_with_iv` / `scan_universe_with_iv`
  return a `StrangleEntryScore` — a 0–100 timing score, a phase, a
  `recommendation` string — and nothing more. Nothing constructs a
  tradeable strangle (strikes, premium, sizing); nothing EV-ranks one.
  §2 ("no tradeable candidate bypasses `EVEngine.evaluate`") is not
  *violated*, but only because the path stops at "is now a good moment",
  short of a candidate. A trader acting on a `strong_entry` then builds
  the strangle fully unranked. The same structural gap S8 logged for the
  covered-call leg: an in-scope strategy (per CLAUDE.md's NEVER list) with no EV
  authority beneath it. **Fixed in `#126`** —
  `WheelRunner.rank_strangles_by_ev` EV-ranks short-strangle candidates
  (strikes + premium): the put leg and the call leg are each scored
  through `EVEngine.evaluate`, and the composed strangle EV is the sum
  of the two. The strangle strategy is now under the EV authority.

- **The Layer-2 IV overlay was dead code — `score_entry_with_iv` crashed
  on every call. Fixed.** It called `connector.get_ohlcv(ticker,
  as_of=..., lookback=200)` — the real `MarketDataConnector.get_ohlcv`
  accepts neither kwarg → `TypeError` on the first line — and depended on
  four methods the connector never exposed (`get_realized_vol`,
  `get_current_iv`, `get_vix_level`, `get_vix_contango`).
  `scan_universe_with_iv` therefore returned an empty frame for every
  input, and `analyze_ticker`'s strangle block caught the `TypeError` in
  a bare `except` and silently fell back to Layer-1-only scoring. The IV
  overlay — IV rank, vol risk premium, VIX context — never ran in any
  live path. **Fixed** in this PR: rewired to the connector's real API
  (`get_ohlcv(end_date=...)`, `get_iv_rank`, `get_vol_risk_premium`,
  `get_vix_regime`); the Layer-2 connector tests in
  `tests/test_strangle_timing.py` were reworked onto the real interface
  and the strict xfail that pinned this bug replaced with a passing
  regression test.

- **`get_iv_rank` unit mismatch — a bug inside the bug.** `get_iv_rank`
  returns a 0–1 fraction (AAPL @ 2026-03-20 → 0.947);
  `_compute_iv_multiplier` expects a 0–100 rank (`> 70`, `< 20`). The
  overlay passed the fraction straight through — so even had it run, a
  0.95 rich-IV name would have tripped the `< 20` *low-IV penalty*,
  exactly inverted. **Fixed** with the above (scaled ×100).

- **The live recommendation ignored IV rank entirely.** Because the
  overlay was dead, every strangle score a trader saw — via
  `analyze_ticker` or a direct `score_entry` — was Layer-1 only:
  Bollinger / ATR / RSI / trend / range proxies. A short strangle is a
  premium-selling trade; whether IV is rich or cheap is *the* edge. In
  the Layer-1 sweep JNJ (IV rank 0.30, cheap) scored 63.5 and CAT (IV
  rank 1.00, rich) 61.0 — the cheap-premium name *higher*. With the fix
  the overlay differentiates: GE 69.5 → 90.3 (×1.30), AAPL 61.6 → 77.0
  (×1.25), JNJ 63.5 → 69.9 (×1.10). Resolved by the fix above.

- **The strangle engine has no earnings awareness.** `classify_regime`
  and `score_entry` use only price/vol proxies; the IV overlay adds
  IV/VIX. Nothing reads the earnings calendar. At `as_of=2026-03-20` the
  top Layer-1 name BAC (70.5) had earnings in 26 days; XOM (66.1,
  "conditional") in 18; JPM and JNJ in 25 — all inside a 35-DTE
  strangle's life, where an earnings gap + IV crush is the dominant
  risk. The put-wheel path is earnings-gated by `EventGate`; the
  strangle path has no equivalent and surfaces no flag. **Logged.**

- **`recommendation` is decoupled from phase and confidence.** The
  recommendation is a pure `total_score` cut (≥80 / ≥60).
  `_classify_phase` computes a `VolatilityPhase` and a confidence
  (UNKNOWN → 0.30), but neither gates the recommendation and the
  confidence is never surfaced on `StrangleEntryScore`. Post-fix, MSFT
  scored 81.6 → **"strong_entry"** while its phase was **`unknown`** — a
  top-tier entry recommendation on a name whose volatility lifecycle the
  engine cannot classify, against the model's stated "enter in
  POST_EXPANSION" premise. **Logged.**

- **`VolatilityPhase.TREND` is documented "AVOID" but has no
  recommendation override.** `score_entry` hard-overrides the
  recommendation to "avoid" on `compression_warning` and (conditionally)
  `expansion_active` — but not on a TREND phase or `strong_trend_warning`;
  only the trend *component* score drops. The `TREND` enum comment reads
  "Persistent direction → AVOID symmetric", yet a strong-trend name
  scoring ≥60 still reads "conditional". The warning flag is set but
  inert. **Logged.**

- **Silent failure modes hid the dead overlay.** `scan_universe_with_iv`
  wraps each ticker in `except Exception: continue`; `analyze_ticker`'s
  strangle block in `except Exception: pass`. So `score_entry_with_iv`
  crashing on *every* ticker surfaced only as an empty scan and a silent
  Layer-1 fallback — no warning, no log. The bare excepts are why a
  fully-dead feature went unnoticed. **Logged** (the swallowing left in
  place — pre-existing defensive pattern, beyond a usage test's remit).

- **`score_strangle_entry` doc-drift.** `wheel_runner.py`'s module
  docstring advertises `runner.score_strangle_entry("AAPL")` as a usage
  example; no such method exists on `WheelRunner`. The real entry points
  are the `strangle_engine` property and `analyze_ticker`. **Logged.**

**Follow-up.** A strangle EV layer — strike selection plus an
EV-ranked strangle candidate, the §4 timing-gated parallel of
`rank_candidates_by_ev` — **shipped in `#126`** (`rank_strangles_by_ev`),
which also carries an `EventGate`, so the EV-ranked strangle path is
earnings-gated. This brings the one timing-gated strategy under the
same EV authority as the wheel legs. The bare timing engine
(`strangle_timing.py` — `score_entry`) is unchanged and still has no
earnings awareness (finding above, **Logged**).

**Validation re-run (2026-05-21).** Confirm-fixed pass on real
Bloomberg data, `as_of=2026-03-20`.

- *Strangle EV layer.* `rank_strangles_by_ev("AAPL")` returns **16
  EV-ranked short-strangle candidates** — each a concrete (`put_strike`,
  `call_strike`, `total_premium`) with a composed `ev_dollars`
  (put-leg EV + call-leg EV, both through `EVEngine.evaluate`); AAPL
  composed EV −660 … −106, default floor → 0 tradeable. *Before:* the
  strangle path stopped at a 0–100 timing score; a trader acting on a
  `strong_entry` then built the strangle fully unranked.
- *Earnings gate.* S14 named JPM as a name with earnings inside a
  strangle's life that the old path could not see. At the same
  `as_of`, `rank_strangles_by_ev("JPM")` returns **0 candidates** —
  all 16 dropped at the `event` gate (`earnings@2026-04-14`, inside
  every 21–63 DTE window). The new EV-ranked strangle path carries an
  `EventGate`; a trader routing a strangle through the ranker is now
  earnings-gated. (The bare timing engine is still earnings-blind —
  the "no earnings awareness" finding above stays **Logged**.)

No new bug surfaced. The `recommendation` / phase findings remain
**Logged** (#118 P5).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `rank_strangles_by_ev` still routes both legs through `EVEngine.evaluate`. JPM strangle at `as_of=2026-03-20` (earnings 2026-04-14 inside every probed DTE window) returns 0 rows with 4 `event` drops on `.attrs["drops"]` — the EventGate on the strangle ranker correctly blocks. AAPL strangle at the same as_of also returns 0 rows (negative composed EV at the default floor, just like the original entry's `-660...-106` range).
  - Qualitative verdict: match — `StrangleTimingWithIV` constructor signature is now `(data_connector=..., weights=..., **kwargs)` (originally `connector=...` per the entry — pure-API rename; behavior unchanged). `WheelRunner.rank_strangles_by_ev` exists with `EventGate`. Layer-2 IV overlay computes correctly: AAPL Layer-2 total_score = 76.97 / CAT = 79.30 / JPM = 79.02 / JNJ = 69.88 — all non-zero scores (the Layer-2 overlay is alive, confirming PR #118 / commit `210463d` fix). Layer-1 `score_entry` signature also changed slightly — `connector` kwarg now required positionally; pure-API note.
  - Numerical drift > 5%:
    - metric `Layer2_score[AAPL]`: orig `77.0` → new `76.97` (`-0.04%`) — within rounding.
    - metric `Layer2_score[CAT]`: orig not directly quoted at `as_of=2026-03-20` (the entry quoted GE 90.3 / AAPL 77.0 / JNJ 69.9); CAT was Layer-1 only in the published table.
  - Notes: `WheelTracker.open_strangle` still does not exist — the strangle "tradeable-strategy-with-no-tracker-integration" finding from S14 / S24 is **still open**.

---
