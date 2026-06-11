# Supervised-Block Worklist — consolidated queue for the next operator sessions

**Compiled 2026-06-11** (post brain-audit fix wave: #405/#407/#408/#410 merged).
This is the single routing document for everything that deliberately waits for
the operator. Two distinct blocks, **A before B** — Block A moves the data the
Block-B baselines re-pin against; doing B first pays the ~4h re-baseline tax
twice (the same coupling argument as the 2026-06-08 batch decision that
produced `docs/bloomberg_refresh_runbook.md`).

Nothing here is actionable autonomously: every item is Terminal-gated,
EV-moving (re-baseline event), reserved by prior operator decision, or needs
an explicit operator choice.

---

## Block A — Terminal / data session (~4–6 h; needs Theta Terminal + Bloomberg)

Plan-of-record: `docs/bloomberg_refresh_runbook.md` (PR #365). Branch the work
off **live `origin/main`** at session start (main moves between sessions).

| # | Item | Source | Notes |
|---|---|---|---|
| A1 | #339 — BK↔BNY re-ticker continuity, recent-only universe names, dividends re-pull (CTRA/BK/LW/PAYC source regression), S34 re-baseline | data queue | Only TRUE Bloomberg blocker: CASY 4 files + the 10 blue-chip OHLCV backfills; the rest is git-reconstructable per the runbook. |
| A2 | #355 — 11 long-history names truncated in `sp500_ohlcv.csv` (504-day-gate backfill defect) | data queue | Same pull session as A1. |
| A3 | #354 — dated PIT fundamentals pull so `get_fundamentals`/`get_credit_risk` can honor `as_of` | data queue | The tracking xfail is **shape-only** — when fixing, pin *behavior* (as_of selects a historical row), not signature. |
| A4 | #357 — treasury rate band note + dividend float-noise clamp | data queue, low-pri | Ride-along. |
| A5 | #369 — (E) #363 IV gate doesn't clean the fundamentals-fallback IV path (`wheel_runner.py` fallback block) | reserved (E) | Trio file → supervised. |
| A6 | #372 — (E) R9 sector cap uses `DEFAULT_SECTOR_MAP` (132/511 names; 379 collapse to "Unknown") instead of pulled GICS | reserved (E) | Pairs naturally with A1's universe work. |
| A7 | #378 — (E) IV-staleness gate (`_resolve_pit_atm_iv`) + rate-fallback divergence under deep-history/staggered refresh | reserved (E) | Note: the 2026-06-11 M3 fix (#408) deliberately did NOT touch this path. |
| A8 | NFLX OHLCV ~10× price mis-scale (strike 1075 vs "spot" 110) | IBKR Phase-3 calibration finding | Inflated a Pearson to 0.9966 before the moneyness gate; also why the calibration truth table excludes NFLX/BKNG/CVNA. Fix at the pull, then lift the exclusions. |
| A9 | *(enabler, optional)* Per-contract option **volume** capture in the Theta pull — or formally sanction the OI / stock-ADV proxy — so the Almgren-Chriss `adv_contracts` input exists | brain-audit M4 scoping (2026-06-11) | Zero EV surface by itself; unblocks B5. Stock-side ADV already exists (`sp500_liquidity.csv` `avg_vol_30d`). |
| A10 | Post-refresh hygiene: bump `EXPECTED_FRONTIER` in the preflight env-guard test; re-baseline cmd `python -m backtests.regression.<id> --update-snapshot` ×4 | DATA_POLICY §5 / runbook | Mandatory after any OHLCV refresh. |

## Block B — coordinated decision-layer re-baseline (one pass, one harness run)

Anchor document: `docs/REBASELINE_D19_D21_RECAL_SCOPE.md` (planning-only
draft, tracked on main; its entanglement analysis is the controlling logic).
Everything EV-moving rides the SAME commit + backtest re-run + baseline
refresh:

| # | Item | Source | Why it must ride this pass |
|---|---|---|---|
| B1 | D21 — calendar→trading-bar horizon (~46% over-long → ~21% over-dispersion; fix authored, not applied) | operator draft | Moves `prob_profit`/`prob_assignment` everywhere. |
| B2 | D19 — net exit-leg cost into `ev_raw` (fix authored, not applied; insertion note at `ev_engine.py` exit-cost block) | operator draft | Moves EV everywhere. |
| B3 | Probability recalibration (top-bin over-confidence; prior W2-C map failed LOCO **and was fit on D21-deflated probs**) | operator draft + IBKR Phase-3 | Must be re-measured/re-fit on post-D21 probabilities; guardrail: do NOT widen R11's VIX>25 gate to chase the calm-regime miss. |
| B4 | Brain-audit **M2** — RV30/RV252 widening applied only on the put-entry ranker; CC/strangle/roll EVs run unwidened (≤15% σ) | brain-audit 2026-06-11 | Moves CC/strangle/roll EVs → same re-baseline. |
| B5 | Brain-audit **M4** — wire Almgren-Chriss size impact (`num_contracts`/`adv_contracts`) into the `ev_engine.py` slippage call | brain-audit M4 scoping | Ungated form is EV-moving even at 1 contract; the wiring site is the SAME cost block as B2 (D19) — one trio touch, not two. Needs A9 first (or the sanctioned proxy) + `k` calibration vs IBKR fills. |
| B6 | #402 — snapshot drift re-pin (s27/s32/s34 `ev_mean`/`spearman_p` outside tolerance since #338; prime suspect #363 IV gate; s27 drift independently reproduced from a 3rd env 2026-06-11) | drift issue | Revert-isolation verdict first, then re-pin — which happens here anyway since B1/B2/B4/B5 re-pin everything. Tag the new pins with the causal item, not a bare number. |
| B7 | Re-pin discipline: fingerprint currently pins OHLCV only (IV/treasury unpinned — latent gap); close it while re-pinning | slow-lane drift validation 2026-06-02 | Cheap to fix inside the same pass. |

**Sequencing inside B:** D21 → re-measure calibration → D19 + M2 + M4 →
single re-baseline + S27 ρ gate check (≥ +0.15 hard gate for any widening
change) → recalibration fit last, on corrected probabilities.

## Standing operator decisions still open (not session work)

- **Morning portfolio refresh**: Task Scheduler registration for
  `scripts/ibkr_gateway_pull.py` is ready but needs the operator to confirm the
  trigger time (classifier-blocked as unauthorized persistence otherwise);
  optional IBC install for zero-touch 2FA. IBKR Phase 5 (flat-to-flat cycle
  aggregation) parked by design.
- **Primary working tree**: the `dashboard/` deletion in the operator's
  uncommitted tree looks accidental (`git checkout -- dashboard/` restores);
  the premium-correction pilot files there are untracked WIP.
- **Quant-test-audit round 2**: PR-4..7 waves remain (probe-then-pin cadence,
  tests-only PRs — these are autonomous-eligible, listed here only for
  completeness of the queue).

---

*Maintenance: strike items as they merge; move this file to `archive/` when
both blocks have landed.*
