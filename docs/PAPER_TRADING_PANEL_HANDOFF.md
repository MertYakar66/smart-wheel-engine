# Paper-trading panel — frontend handoff to the Dashboard terminal

> **Ownership.** The Next.js dashboard (`dashboard/`) and the `/api/portfolio/*`
> React consumers are owned by the **Dashboard terminal** (CLAUDE.md §6). The
> engine side of the paper-trading panel — the API slices + shaped SIM data — is
> built and merged separately (branch `claude/paper-trading-sim`). **This doc is
> the handoff**: everything the Dashboard terminal needs to build
> `(terminal)/paper/page.tsx` without the paper-book builder reshaping the
> real-portfolio pipeline.

---

## What already exists (engine side — ready to consume)

Three **SIM-only** endpoints under `/api/portfolio/*` (served by
`engine_api._handle_paper_view`, reading the SIM namespace via
`engine.paper_book`; the real-data slices are untouched). Book selected by
`$SWE_PAPER_BOOK_NAME` (default `live_forward`). Every payload is tagged
`source: "simulated"`.

### `GET /api/portfolio/papertrade`
```jsonc
{
  "book": "live_forward",
  "as_of": "2026-07-02",
  "initial_capital": 1000000,
  "final_nav": 1033874,
  "backfill_end_date": "2026-05-15",     // the honest boundary
  "points": [ { "label": "Sep 2", "date": "2025-09-02",
                "port": 1000000, "spy": 1000000, "premium": null,
                "phase": "backfill" }, ... ],   // phase ∈ {backfill, forward}
  "n_backfill": 182, "n_forward": 4,
  "caps_armed": true,
  "caveats": { "synthetic_fills": "...", "backfill_vs_forward": "...", ... },
  "provenance": { "backfill_vs_forward": "...", "synthetic_fills": "...",
                  "model_vs_measured": "history=engine-measured; MC bands=model" },
  "source": "simulated"
}
```

### `GET /api/portfolio/montecarlo`
```jsonc
{
  "status": "ok",
  "bands": { "kind": "model", "band_days": [0,1,...],
             "equity_bands": { "p5":[...], "p25":[...], "p50":[...],
                               "p75":[...], "p95":[...] },
             "quantiles": [5,25,50,75,95] },
  "terminal_return_quantiles": { "p5": -0.016, "p50": 0.013, "p95": 0.050 },
  "max_drawdown_quantiles": { ... },
  "reconciliation": { "reconciled": true, "median_abs_pct_gap": 0.04 },
  "correlation_tail": { "stress_vs_empirical": { "t_cvar_multiple": 1.8, ... } },
  "source": "simulated"
}
```

### `GET /api/portfolio/calibration`
```jsonc
{
  "n_settled_total": 44, "n_pending_total": 6,
  "pooled":   { "n": 44, "brier": 0.09, "ece": 0.09, "bins": [ {bin,n,mean_pred,obs,ci_lo,ci_hi}, ... ] },
  "backfill": { ... }, "forward": { ... },      // same shape, phase-split
  "top_bin_flag": { "bin": "(0.90, 1.0]", "known_overconfident": true, "note": "..." },
  "source": "simulated"
}
```

---

## Suggested panel (`(terminal)/paper/page.tsx`) — reusing existing components

| Section | Reuse | Notes |
|---|---|---|
| KPI header | `PfCard` | Final NAV, total return, #backfill/#forward points, `caps_armed` badge |
| Equity fan | `equity-curve.tsx` | Plot `points.port` + the MC `bands` fan (p5–p95). **Shade backfill vs forward** at `backfill_end_date` — the honest boundary is the whole point |
| Distribution | `distribution-bar.tsx` | `terminal_return_quantiles` + `max_drawdown_quantiles` |
| Calibration | `calibrated-prob.tsx` | reliability `bins` with Wilson CI whiskers; render the `top_bin_flag` as an explicit "known over-confident" chip |
| Provenance | `ProvenanceBadges` | **backfill vs live-forward** + **model (MC) vs measured (history)** + a prominent **"SIMULATED — synthetic fills, no money at risk"** banner |

### Non-negotiable honesty rendering (from `caveats`)
1. A persistent **"SIMULATED / paper — synthetic BSM fills, optimistic vs real"**
   banner. This is not a live book and not a real-fill test.
2. The **backfill/forward boundary** must be visually obvious (shading + a marker
   at `backfill_end_date`). Only the forward segment is true OOS.
3. The **MC fan is a model**; the `points` curve is the measured track — badge them
   differently.
4. The calibration **top bin is flagged over-confident** — never render it as truth.

---

## Guardrails for the Dashboard terminal

- These slices read the **SIM namespace only** (`$SWE_SIM_DATA_DIR` /
  `data_processed/sim/`). Do **not** point the panel at `data_processed/ibkr/`.
- The real-portfolio slices (`summary/positions/returns/income/risk/history`) are
  **unchanged** — the paper slices were added as a separate intercept branch.
- Data refreshes need no engine restart (the engine re-reads files per request);
  a new forward-append just lands new JSON the panel picks up on next fetch.
- To point at a different book, set `$SWE_PAPER_BOOK_NAME` before starting the
  engine (defaults to `live_forward`).
