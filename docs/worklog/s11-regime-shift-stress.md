---
id: S11
title: Regime-shift stress
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Stress the regime machinery: anchor the ranker across the
April-2025 VIX spike and observe whether the HMM and dealer-positioning
multipliers actually track a real volatility shock.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, 35-DTE / 25-delta, a fixed
12-name diversified watchlist (AAPL, MSFT, NVDA, AMZN, JPM, XOM, UNH,
CAT, KO, PG, HD, CVX), `include_diagnostic_fields=True`.
`rank_candidates_by_ev` run at five `as_of` dates spanning the
April-2025 vol spike — 2025-03-20 (VIX 19.8, calm) / 04-07 (46.98) /
04-09 (33.6, peak window) / 04-24 (26.5, elevated) / 05-15 (17.8,
reverted). The regime-multiplier trajectory was probed with
`use_event_gate=False` (all names present each date); event-gate
behaviour probed separately on vs off. No code changes.

**Status.** Done. The HMM regime multiplier responds as advertised — it
cut EV ~80 % into the spike (cross-sectional mean 0.74 → 0.29) and
reverted (→ 0.69). The dealer, skew and credit overlays did not
respond — each pinned at 1.0 the whole way through. No multiplier was
provably wrong; no bug. All findings logged.

**Findings:**

- **HMM multiplier tracks the shock — works as advertised.**
  Cross-sectional mean `hmm_multiplier`: 0.74 (calm, 2025-03-20,
  VIX 19.8) → 0.29 (peak, 04-09; 04-07 hit VIX 46.98 — the steepest
  VIX run-up in the 11-year data window outside the 2020 COVID crash)
  → 0.69 (reverted, 05-15, VIX 17.8). At the peak most names sat at
  ~0.20 — `position_multiplier` is a posterior-weighted average of
  per-state weights {crisis 0.2, bear 0.5, normal 1.0, bull_quiet 1.25}
  (`regime_hmm.py:275`), so ~0.20 means ~100 % crisis-state posterior
  (a genuine classification, not a clamp). The HMM cuts EV up to 80 %
  into the spike and reverts. **Logged.**

- **Per-ticker, the HMM multiplier is jumpy — a noisy single-name
  signal.** `GaussianHMM(n_states=4, n_iter=20, random_state=42)`
  (`wheel_runner.py:862`) is seeded, so fits are deterministic — but it
  re-fits per (ticker, as_of) and the clean cross-sectional mean hides
  per-ticker cliffs: HD flips 0.21 (04-07, crisis) → 1.00 (04-09,
  normal) over two days while VIX held 33–47; PG and UNH stay at ~0.2
  on 05-15 after VIX reverts. The HMM models each *ticker's* return
  regime, not the market's, so single-name multipliers diverge from
  VIX — partly genuine idiosyncratic stress, partly fit sensitivity
  from the low `n_iter=20`. On any one name it is a noisy de-rater.
  **Logged.**

- **The HMM regime is unlabeled in the output — confirms S1.** The
  ranker emits `hmm_multiplier` (a bare number — 0.20 = an 80 % EV cut)
  with no companion `hmm_regime` label, even though `dealer_regime` and
  `credit_regime` label columns both exist. A trader sees the cut with
  no surfaced "crisis". S1 logged this; S11 confirms it and sharpens it
  via the asymmetry with the other two overlays. **Fixed in #121.**

- **Dealer & skew multipliers are inert on the Bloomberg provider.**
  `dealer_multiplier` and `skew_multiplier` were pinned at 1.00 across
  all five dates; `dealer_regime` was empty throughout. Both require an
  option chain; `MarketDataConnector` exposes none, so neither overlay
  ever computes — they cannot respond to any shock on Bloomberg. Only
  the Theta provider (S6) would activate them. Same structural dormancy
  S9 found for the chain-quality gate. **Logged.**

- **The credit-regime overlay is not as_of-aware — a PIT leak.**
  `credit_multiplier` / `credit_regime` were identical
  ('benign' / 1.00) across all five as_of dates, including 04-07 at
  VIX 46.98. `FREDAdapter.credit_regime()` (`fred_adapter.py:137`)
  takes no `as_of` — it returns one wall-clock value applied to every
  historical as_of, so a backtest across a credit-stress episode would
  never see the stress. Same family as S10's news PIT leak and S1's
  `as_of` footgun. **Fixed in #119.**

- **Net — on the Bloomberg provider the HMM carries the entire regime
  response.** `combined_regime_mult = hmm × skew × news × credit`
  (`wheel_runner.py:980`), but skew and dealer are pinned (no chain),
  credit is pinned (PIT-unaware), and news is pinned (no store — S10).
  Of the four regime overlays, only the HMM is both live and responsive
  on the default provider. **Logged.**

- **The event gate stays consistent across the shift — earnings-driven,
  not vol-driven.** With `use_event_gate=True`, survivors of the
  12-name watchlist were 8 / 2 / 2 / 3 / 10 across the five dates,
  tracking the Q1-2025 earnings calendar (April is peak earnings
  season → 9 of 12 gated) rather than the VIX — the gate behaves
  identically regardless of regime. (Aside: names within 5 days of
  earnings are soft-skipped even with `use_event_gate=False` — a
  second, separate earnings mechanism.) The stress-residual gate is not
  on the ranker decision path (S9). **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — overlays only scale `ev_dollars`, never `ev_raw`; multipliers clamped at construction time.
  - Qualitative verdict: **partial — substantial behavior change for the better**. HMM trajectory near-identical to original (0.74 → 0.36 → 0.29 → 0.70 → 0.69 — peak at 04-09 matches original 0.29 to 1 part in 100). Event-gate survivor counts at the five dates: **8 / 2 / 2 / 2 / 10** vs original **8 / 2 / 2 / 3 / 10** — one delta at 04-24 (3 → 2). Dealer + skew + news still pinned at 1.0 (no chain on Bloomberg). **But credit_multiplier is now PIT-aware:** 1.00 (03-20) / **0.80** (04-07) / **0.92** (04-09) / 1.00 (04-24) / 1.00 (05-15) — the credit overlay now responds to the April-2025 VIX spike. The original S11 finding "credit-regime overlay is not as_of-aware — a PIT leak" is **CLOSED by PR #119**.
  - Numerical drift > 5%:
    - metric `cross_sectional_hmm_means[2025-04-09]`: orig `≈0.20` (peak) → new `0.2923` (`+46%`); attributable to S31 / PR #208 + PR #222's HMM disambiguation columns plus minor seed-stable HMM refits. Direction unchanged.
    - metric `event_survivors[2025-04-24]`: orig `3` → new `2` (`-33%`); attributable to PR #220 (`as_of-beyond-data` gate extension may now drop one borderline name on this date) — not high-confidence; could also be incidental Bloomberg-data revision since S11 ran.
  - Notes: HMM regime is still **noisy per-ticker** (HD flips normal↔crisis across days); the multiplier value is more stable than the label, confirming the S17 finding still applies.
