---
id: S9
title: Adversarial / gate stress
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Attack each engine gate with inputs that should be
rejected and confirm it fails closed — drops or flags the candidate,
never emits a tradeable result. Gates: history, event, chain-quality,
stress-residual, survivorship.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `as_of=2026-03-20`, 35-DTE /
25-delta. Per-gate probes through `WheelRunner.rank_candidates_by_ev`
(each gate's `enforce_*` / `use_*` flag toggled on vs off), plus direct
probes of `DataQualityFramework._check_options_consistency` and
`StressTester.greeks_stress_ladder`. No code changes.

**Status.** Done. No gate fails open. Four gates (history, event,
stress-residual, survivorship) were exercised live and each fails
closed; the chain-quality gate is structurally inactive on the
Bloomberg provider — no option chain to police — and its check logic
fails closed when probed directly. No bug, no fix; all findings logged.

**Findings:**

- **History gate — fails closed.** `enforce_history_gate` /
  `min_history_days=504` (`wheel_runner.py:614`; in-code, the
  survivorship-bias protection). Probe: GEV (497 OHLCV bars) and SOLV
  (498) — real 2024 spin-offs, both under 504 bars at `as_of` — with
  AAPL (2065) as control. Gate on → ranks `[AAPL]`; gate off →
  `[AAPL, GEV, SOLV]`. The short-history names are correctly blocked.
  **Logged.**

- **Event gate — fails closed.** `use_event_gate` → `EVEngine.evaluate`
  event lockout (`ev_engine.py:262`) → the ranker drops the row on
  `event_lockout_reason` (`wheel_runner.py:1056`). Probe: XOM, JPM,
  UNH, JNJ, GE, NFLX — all with earnings inside the 35-DTE window at
  `as_of=2026-03-20` — plus AAPL control. Gate on → `[AAPL]`; gate off
  → all 7. All six earnings-window names blocked. `EVEngine` does
  compute `event_lockout_reason`, but the ranker discards it on
  `continue` — see the silent-rejection finding below. **Logged.**

- **Chain-quality gate — logic fails closed, but dormant on the
  Bloomberg provider.** `enforce_chain_quality_gate`
  (`wheel_runner.py:843`) runs `_check_options_consistency` on the raw
  chain and `continue`s the ticker on any ERROR/CRITICAL issue. Logic
  probe: a degenerate chain (negative volume, IV 9.5, crossed
  bid > ask) → 3 ERROR issues; a clean chain → 0 — the gate would
  block. **But** `MarketDataConnector` exposes neither `get_options`
  nor `get_option_chain`, so on Bloomberg `chain_df` is always `None`
  and the gate at `:843` never executes. It is reachable only with a
  live-chain provider (Theta — S6); on the default provider it is a
  no-op, and the premium it would police is synthetic BSM anyway.
  **Logged.**

- **Stress-residual gate — fails closed; advisory, off the EV path.**
  The Greeks-decomposition residual gate lives in
  `engine/stress_testing.py` (`:639`), not in `rank_candidates_by_ev`.
  Probe: an extreme `greeks_stress_ladder` (spot ±35 %, `iv_shock=0.80`)
  → 8 of 9 rows tagged `reliable=False`,
  `attrs["residual_gate_passed"]=False`, `max_residual_pct ≈ 3.12`, and
  a `warnings.warn` fires; a mild ±1 % ladder → all reliable. The gate
  correctly flags Greeks the Taylor decomposition cannot attribute. It
  is **advisory** — it tags rows, never drops them — and never blocks
  an EV candidate. **Logged.**

- **Survivorship gate — fails closed; it is the history gate, not a
  membership check.** Probe: `[ZZZZ, NOTAREALTICKER, FIX, AAPL]` →
  ranks `[AAPL, FIX]`; the bogus tickers have no OHLCV and are dropped
  at the data-fetch step (`wheel_runner.py:593`).
  `rank_candidates_by_ev` runs **no constituent-membership check** —
  `get_universe()` (`data_connector.py:654`) is the union of tickers in
  the OHLCV / fundamentals / vol_iv CSVs, not
  `data_raw/sp500_constituents_current.csv`. Benign in practice:
  index-removed names (IPG, K, LKQ, MHK) have no OHLCV at all and
  cannot be ranked, while FIX — a genuine member missing from the
  stale constituents CSV — ranks correctly. Survivorship protection
  thus reduces to no-data-drop plus the 504-bar history gate; there is
  no data-freshness gate, though the Bloomberg data carries no
  stale-but-long delisted name to exploit one. **Logged.**

- **Rejections are silent — recurring S1 / S2 finding.** History,
  event and survivorship rejections are indistinguishable in the
  output: the candidate is simply absent, with no reason, count, or
  diagnostic — a caller cannot tell "gated out" from "never a
  candidate." Of the five gates, only stress-residual surfaces its
  verdict (a `warnings.warn` + `.attrs`); the chain-quality gate at
  least emits a `logger.warning` when it blocks. The three live ranker
  gates are fully silent. Same gap S1 and S2 logged. **Fixed in #121.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every gate exercised below fails closed; no gate failure produces a tradeable candidate.
  - Qualitative verdict: match — all five gates still behave as documented:
    - History gate ON → `[AAPL]` only (GEV, SOLV dropped); OFF → `[AAPL, GEV, SOLV]` ✓
    - Event gate ON @ `as_of=2026-03-20` for the 7-name probe → `[AAPL]` only; six earnings-window names dropped with structured `{ticker:"XOM", gate:"event", reason:"event_lockout:earnings@2026-04-07 (±5d buffer)"}` (the `±` Unicode literal is now in the reason string verbatim — unchanged S1 cp1252 footgun).
    - Survivorship — bogus tickers (ZZZZ, NOTAREALTICKER) silently dropped at data-fetch; FIX/AAPL survive ✓
    - Stress-residual gate / chain-quality gate — structurally inactive on Bloomberg provider (no chain), as documented.
  - Numerical drift > 5%: not applicable — original entry counts gate behavior boolean, not magnitudes.
  - Notes: drops schema confirmed structured `{ticker, gate, reason}` across all observed drops in this run (PR #121 holds).
