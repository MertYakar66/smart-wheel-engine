---
id: paper-trading-sim
title: Forward paper-trading book — simulated wheel loop the real engine drives
kind: feature
status: in-flight
terminal: builder
pr:
decisions: []
date: 2026-07-06
headline: A simulated wheel book the real engine ranks + manages day-by-day, accumulating a zero-risk live equity curve + calibration accumulator + SIM API slices, honestly split into backfill (in-sample-ish) vs live-forward (true OOS).
surface:
  - engine/paper_book.py
  - scripts/run_paper_book.py
  - tests/test_paper_book.py
  - engine_api.py
  - docs/PAPER_TRADING_RUNBOOK.md
  - docs/PAPER_TRADING_PANEL_HANDOFF.md
---

## Goal

Stand up the "simulation world" the validation arc was building toward: a
FORWARD paper-trading loop — a simulated wheel book that the **real** engine
ranks and manages day-by-day, accumulating a live equity curve with **zero money
at risk**, plus a continuous calibration accumulator and dashboard API slices, so
the operator can *watch the engine operate* before any real capital is committed.
The MC machinery (#483) and the OOS gate (#484/#485) are done; this is the live,
forward-accumulating book those foundations feed.

Branch `claude/paper-trading-sim`, stacked on `claude/mc-forward-sim` (#483) so
`engine/sim_portfolio.py` is available to reuse for the MC bands — the same
house pattern as #485-on-#484.

## What we tried

**Architecture — the §2 split (mirrors #483 / #484-#485).** Invariant 1 wants an
AST guard that the paper module never imports the decision-layer trio, but the
loop must call the ranker. Resolution, mirroring `engine.sim_portfolio` (trio-free
library) + `scripts.run_forward_sim` (driver that drives the engine):

- `engine/paper_book.py` = **trio-free** reporting/state library: SIM-namespace
  I/O + guard, phase-labelled history, forecast ledger + held-to-expiry outcome,
  calibration (wilson/reliability), MC bands (soft-imports `sim_portfolio`),
  idempotent append. AST-guarded by `test_paper_book.py`.
- `scripts/run_paper_book.py` = the driver: imports `wheel_runner` +
  `backtests.regression._common` to drive the engine, then hands outputs to
  `engine.paper_book`.

**Backfill caps-off vs forward caps-armed.** `make_live_book_tracker`'s R9/R10
caps size against `_compute_live_nav`, which marks NAV at `date.today()` +
latest close. For a genuine forward step that is correct (today is today). For a
*historical* backfill it would **leak future prices** into cap decisions. So:
- `seed` reuses caps-off `run_backtest` (PIT-correct; never calls
  `_compute_live_nav`) — the task's explicit "reuse `_common.run_backtest`" path,
  labelled `backfill` (in-sample-ish).
- `forward` loads the continuation state, **re-arms** R9/R10 (the `enforce_*`
  flags are not serialised by `to_dict`, so re-arming is load-bearing — pinned by
  a test), and opens EV>0 with the armed caps against the correct forward NAV.

**Calibration outcome = `realized_pnl > 0`** — matches the engine's
`prob_profit = mean(pnls>0)` definition and the locked convention in
`test_parameter_oos.py` (`(realized_pnl > 0).astype(float)`). Backfill forecasts
are recovered by joining each opened position to its `(entry_date, ticker)` rank
row; forward forecasts are recorded at open and settled at expiry.

## What worked

End-to-end validation with the real Bloomberg connector (provider logged
`MarketDataConnector`):

- **Seed** (12 names, 2025-09-02 → 2026-05-15, $1M, friction=full): 182 backfill
  points, final NAV $1,012,916 (+1.3%), 44 forecast entries (38 settled),
  calibration pooled Brier 0.082 / ECE 0.094, MC reconciled (gap 0.04%). ~5 min.
- **Forward-append** (as_of 2026-06-01 / -06-15 / -07-01): opened NVDA/XOM/JNJ,
  then CAT/UNH/MSFT (caps armed, fit under R9/R10 at $1M), settled due forecasts,
  one labelled `forward` point each. Re-running the same as_of = idempotent no-op.
  Default `as_of=today` (2026-07-06) **clamped to the data frontier 2026-07-02**
  (the engine returns no candidates past the latest healthy bar) and appended.
- **Caps proven firing**: a $100k smoke book showed `caps_refused=2/3` — R10
  refuses a single high-priced contract that exceeds 10% of a small NAV. Correct
  gated behaviour; the $1M scale is where opens fit (the S34 scale).
- **API slices**: `/api/portfolio/{papertrade,montecarlo,calibration}` served the
  SIM data (`source:"simulated"`, n_backfill=182/n_forward=4); the real-data
  `history` slice still HTTP 200 (untouched); `bogus` still 404.
- **Tests**: 20/20 fast tests pass (§2 AST guard, SIM isolation incl. real
  `data_processed/ibkr/` fingerprint-unchanged, wilson/reliability byte-match vs
  `scripts/ibkr_ev_calibration`, MC determinism seed 42, caps-armed refusal after
  a save/load round-trip, idempotency, held-to-expiry outcome, driver join
  helpers). One slow-lane engine-driven integration marked `backtest_regression`.

## What didn't

- **`as_of=today` returns 0 candidates** — today (2026-07-06) is past the
  healthy-ranking frontier; even `as_of=frontier` (2026-07-02) gives ~1 row (thin
  front-month). Fixed by clamping `as_of` to the frontier and choosing healthy
  demo dates; documented so the operator's daily cron isn't surprised.
- **Small-NAV books refuse every forward open** — R10 (10% single-name) vs a
  $200 strike × 100 = $20k notional needs NAV ≥ $200k. Not a bug; documented the
  capital-sizing guidance (≈$1M for mega-cap strikes).

## How we fixed it

Shipped as described above. The honesty boundary (backfill vs forward) is a
per-point `phase` field + `meta.backfill_end_date`; synthetic-fill /
model-vs-measured / W3 top-bin-overconfidence caveats are carried on every report
and surfaced on the API + panel-handoff.

## Adversarial review + fixes (self-run, 12-agent 5-dimension workflow)

A §2 / SIM-isolation / OMS / correctness / honesty review (5 dimensions → verify)
returned **7 confirmed findings, 0 refuted — and the §2 and OMS reviewers found
NOTHING** (both `findings:[]`). So the hard invariants (no trio import, no EV
mutation, no order surface, no isolation *bypass*) are clean. The 7 were quality
issues in my own files, all fixed:

1. **[LOW]** `paper_book.py` `ibkr` guard matched ANY ancestor dir named `ibkr`
   (over-broad availability false-positive, not a bypass) → scoped it to the tail
   BELOW the SIM root via `_rel_parts`; added `test_resolve_book_dir_allows_ibkr_ancestor`.
2. **[MEDIUM]** `run_paper_book.py` flat-book equity mark used `{}` AFTER opens →
   booked just-credited premium as NAV without the offsetting liability (~0.9%
   sawtooth + corrupted MC reconciliation) → moved the mark BEFORE opens with PIT
   spots (`step 2.5`); pinned by `test_pre_open_mark_excludes_new_premium`.
3+4. **[MEDIUM]** forward point + forecast phase hardcoded `"forward"` → use
   `pb._phase_for(as_of, backfill_end)` so an `as_of ≤ backfill_end` is correctly
   `backfill`, not fake-OOS.
5. **[LOW]** papertrade payload omitted `caps_detail` → added it (so a bare
   `caps_armed=true` can't imply the caps-off backfill was gated).
6. **[LOW]** the armed caps' NAV (`_compute_live_nav`, in `wheel_tracker.py`
   which I must not modify) uses `date.today()`, so a BACKDATED forward run sizes
   caps against the frontier not `as_of` (caps still fire; no bypass) → documented
   as the `caps_backdated` caveat carried on every report + meta.
7. **[LOW]** montecarlo caveats fell back to `{}` if `meta.json` absent → fall
   back to the authoritative `pb.PAPER_CAVEATS`.

## Evidence

```
# provider + smoke
[paper_book] provider=MarketDataConnector | frontier ~2026-07-02
# seed
history points: 182 (backfill=182)  final NAV $1,012,916  calib n=38 Brier=0.082 ECE=0.094  MC reconciled gap 0.04%
# forward
FORWARD as_of=2026-06-01 port=$1,025,940 opened=['NVDA','XOM','JNJ'] settled=4 caps_refused=0
FORWARD as_of=2026-06-15 port=$1,034,521 opened=['CAT','UNH','MSFT'] settled=2
as_of 2026-07-01 already appended — idempotent no-op
as_of 2026-07-06 clamped to data frontier 2026-07-02
# caps (100k smoke): caps_refused=2/3  (R10 single-name fires)
# api: /papertrade source=simulated n_backfill=182 n_forward=4 ; /history HTTP=200 (untouched) ; /bogus HTTP=404
# pytest tests/test_paper_book.py -m "not backtest_regression": 20 passed
```

## Unresolved / handoff

- **Frontend** (`(terminal)/paper/page.tsx`) is deliberately NOT built — the
  `dashboard/` React pipeline is owned by the Dashboard terminal (§6). The engine
  side + shaped data are ready; the view is handed off via
  `docs/PAPER_TRADING_PANEL_HANDOFF.md`.
- **Genuine OOS accrues in wall-clock**: the forward segment only becomes a real
  out-of-sample record as the operator runs the daily append going forward. The
  forward-calibration block is empty until forward-opened positions expire.
- **Stacked on #483** — review-only; hold for operator merge in order
  (#483 → this). Do not self-merge.
