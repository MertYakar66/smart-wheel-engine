---
id: S1
title: Single-snapshot trader session
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the morning-scan → dossier → sizing path as a
retail wheel trader would, top-down across the SP500.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, offline charts
(`FilesystemChartProvider`), 40-name diversified watchlist,
`as_of=2026-03-20`, $150k account, 35-DTE / 25-delta puts.

**Status.** Done. One critical bug fixed
(`#102` — dividend-yield normalization). Other findings logged.

**Findings:**

- **Dividend-yield normalization bug** (`wheel_runner.py` ~ line 655).
  Sub-1% yields skipped the `> 1.0` guard and reached BSM as a
  whole-number decimal (`0.87` used as 87% q). Corrupted the
  delta → strike solve and the synthetic premium across ~92 of 410
  priced names — MSFT, COST, AAPL surfaced as positive-EV when truly
  negative. Fixed in **`#102`** (merged `afee837`).
- **`Δ` (U+0394) Unicode crash** in `candidate_dossier.py`'s R3
  review note — crashes Windows cp1252 console on print / log.
  **Logged.**
- **Silent drops** — `rank_candidates_by_ev` returns only
  survivors; no diagnostic when a name is gated out (earnings,
  history, chain quality). **Logged.**
- **`as_of` footgun** — defaults to today; pairs stale Bloomberg
  prices with current-date event timing. **Logged.**
- **R4 reviewer rule effectively dead** in the standard ranker
  path — needs a `phase` field the ranker never emits. **Logged.**
- **Committee delta silent default** — `_build_advisor_input`
  falls back to `delta=-0.30` (`integration.py:165`) because the
  ranker emits no delta column. The 45-DTE figure in the original
  S1 note is an omission-only fallback (`integration.py:164`),
  **not** a live mismatch: the ranker emits `dte`, so the committee
  sees the correct 35. Corrected by S7. **Logged.**
- **No `ev_raw` exposed** in the ranker output despite being a
  core EV-engine field. **Logged.**
- **No return-on-capital column / no account-size input** — the
  ranker optimizes absolute EV/day, structurally biased to
  expensive names. **Logged.** Addressed in part by S4 (see S4).
- **Regime (HMM) multiplier unlabeled** — silently cuts EV
  50–80 % on some names with no surfaced regime. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — 28 ranked rows on the 40-name watchlist at `as_of=2026-03-20`, all `ev_dollars` finite (range −481.38…+713.73); 12 drops emitted with structured `{ticker, gate, reason}` per PR #121.
  - Qualitative verdict: match — dividend-yield bug stays fixed (no premium > strike anomaly). Seven of nine logged findings are now mechanically closed: dividend (#102), drops surface (#121), `ev_raw` column, `roc` + `collateral` (#109), `hmm_regime` label (#208), `sector` column (#210), `news_n_articles` (#119). Two remain logged-by-design: Unicode cp1252 mangle on `Δ` / `±` (still present in drop reason strings — observed verbatim during this run), and R4 dormancy (still no phase-aware chart provider on `main`).
  - Numerical drift > 5%: no orig figures cited in the original entry — it was a bug-narrative entry, no row counts or EV magnitudes were quoted to compare against.
  - Notes: 12 dropped at `as_of=2026-03-20` are Q1-earnings-season lockouts (consistent with the pattern S16 documented at the same as_of); current top-5 by EV are LLY (+713.73), CAT (+444.99), BLK (+407.09), DE (+318.53), TMO (+227.27). Diagnostic columns present: `ev_raw`, `roc`, `collateral`, `hmm_regime`, `sector`, `news_n_articles` — all of S1's "missing column" findings are now visible.
