# Forward paper-trading book — operator runbook

> **What this is.** A *simulated* wheel book that the **real** engine ranks and
> manages day-by-day, accumulating a live equity curve with **zero money at
> risk**. It is the "simulation world" the validation arc was building toward:
> the MC distribution machinery (PR #483) and the parameter-OOS gate
> (#484/#485) feed this live, forward-accumulating book so you can **watch the
> engine operate before any real capital is committed**.
>
> **It never places an order.** It simulates fills; it does not route them
> (CLAUDE.md §3). It routes every candidate through the real ranker and never
> mutates `ev_dollars` / `ev_raw` / `prob_profit` / a verdict (CLAUDE.md §2).

---

## 0. TL;DR

```bash
# one-time: seed a backfill (in-sample-ish) from a recent point-in-time run
python scripts/run_paper_book.py seed --book live_forward \
  --tickers AAPL MSFT NVDA JPM GS XOM UNH JNJ HD PG KO CAT \
  --start 2025-09-02 --end 2026-05-15 --capital 1000000

# daily (schedule this): one idempotent forward-append at today's market state
python scripts/run_paper_book.py forward --book live_forward
```

Everything lands in the **SIM namespace** — `$SWE_SIM_DATA_DIR` or the gitignored
`data_processed/sim/live_forward/`. Nothing commits; regenerate with the driver.
The book **never** reads or writes `data_processed/ibkr/` (owned by the Dashboard
terminal, CLAUDE.md §6).

---

## 1. The honest boundary — backfill vs live-forward

True forward paper-trading accumulates in **wall-clock** over days and weeks. You
cannot manufacture weeks of forward data in one session, and this tool does not
fake it. So the book has two clearly-labelled segments:

| Segment | How it's produced | Honesty label |
|---|---|---|
| **backfill** | `seed` reuses `backtests.regression._common.run_backtest` to drive a WheelTracker over the last N months in one batch | **in-sample-ish** — the engine's parameters (HMM weights, POT-GPD, dealer clamp) were hand-set with full-history visibility (caveat **E5**) |
| **live-forward** | each daily `forward` append re-ranks `as_of` and opens/settles incrementally | **genuinely out-of-sample** — the only true paper record; it accrues as you run the daily append going forward |

Every equity point carries a `phase` field (`"backfill"` / `"forward"`); the
boundary date is `meta.backfill_end_date`. The dashboard panel shades the two
segments so the boundary is visible.

**Only the forward segment is a true OOS paper record.** Do not read the backfill
curve as validation — it is a seed so the panel has an immediate curve.

---

## 2. Why the seed is caps-off and the forward path is caps-armed

- **Forward path** builds the tracker via `engine.wheel_runner.make_live_book_tracker`
  → **R9 sector (25% NAV) + R10 single-name (10% NAV)** caps ARMED, so the paper
  book faithfully reproduces the *gated* live behaviour (task invariant 5). Those
  caps size against **live NAV** (`_compute_live_nav` marks at `date.today()`),
  which is correct for a genuine forward step (today is today).
- **Backfill seed** uses the caps-**off** `run_backtest` tracker on purpose: it is
  point-in-time-correct (it never calls `_compute_live_nav`). Arming the caps on a
  *historical* seed would mark NAV at `date.today()`'s prices and **leak future
  prices** into cap decisions for positions opened months earlier. The seed is a
  curve, not a gated-behaviour claim; the forward path is where gating is faithful.

This split is recorded in `meta.caps_detail`. A consequence: positions carried
from the backfill are grandfathered into the armed forward book (the caps only
refuse *new* over-concentrated opens, exactly as production would).

> **Backdated forward runs.** `_compute_live_nav` has no `as_of` — it marks at
> `date.today()` + the latest close. A **genuine** forward run (`as_of=today`) is
> point-in-time correct. A **backdated** run (`--as-of` < the data frontier, e.g.
> the demo) sizes the R9/R10 cap denominator against the *latest* data, not
> `as_of` — the caps still fire (no bypass), but the denominator isn't PIT on a
> backdated replay. This is the `caps_backdated` caveat carried on every report.

---

## 3. Commands

### 3.1 `seed` — one-time backfill

```bash
python scripts/run_paper_book.py seed --book live_forward \
  --tickers AAPL MSFT NVDA JPM GS XOM UNH JNJ HD PG KO CAT \
  --start 2025-09-02 --end 2026-05-15 --capital 1000000 \
  [--friction full] [--dte-target 35] [--delta-target 0.25] \
  [--max-new-per-day 3] [--seed 42]
```

Runs `run_backtest` over `[start, end]`, then writes the SIM artifacts (§4).
Deterministic (seed 42). Choose `end` a few weeks **inside** the data frontier so
options/forward-distribution inputs are healthy (the very last bar is thin).

> **Capital sizing.** With 1-contract positions and the R10 10%-NAV single-name
> cap, a single high-priced underlying's notional (`strike × 100`) can exceed 10%
> of a small NAV — so a small book refuses most forward opens (correct gated
> behaviour, but a dull demo). Use a capital scale where a contract fits under the
> caps (≈ $1M for mega-cap strikes, matching the S34 scale).

### 3.2 `forward` — the daily idempotent append

```bash
python scripts/run_paper_book.py forward --book live_forward [--as-of YYYY-MM-DD]
```

- `--as-of` defaults to **today**. It is **clamped to the data frontier**: once
  `today` drifts past the latest healthy bar the engine returns no candidates, so
  the append marks against the latest available market state and logs the clamp.
- **Idempotent**: re-running the same `as_of` is a no-op (the curve never grows
  twice for one day). Running an `as_of` *before* the last point is refused
  (forward-append is strictly monotonic).
- Each run: marks-to-market the carried book, settles expirations + wheels into
  covered calls, settles any due forecast outcomes, ranks `as_of` and opens up to
  N EV>0 positions (caps armed, full friction), appends **exactly one** labelled
  `forward` point, and recomputes calibration + MC.

### 3.3 `report` — recompute without driving the engine

```bash
python scripts/run_paper_book.py report --book live_forward
```

Recomputes calibration + MC bands from the current persisted state (no new rank).

---

## 4. Artifacts (SIM namespace)

Under `$SWE_SIM_DATA_DIR/<book>/` (default `data_processed/sim/<book>/`):

| File | Contents |
|---|---|
| `sim_portfolio_history.json` | panel curve — `points:[{label, date, port, spy, premium, phase}]` (mirrors `portfolio_history.json` + `phase`) |
| `forecast_ledger.json` | per-open `prob_profit` forecast + held-to-expiry outcome (the calibration source) |
| `sim_closed_trades.json` | realized wheel-cycle ledger (the tracker's closed positions) |
| `calibration.json` | rolling Brier/ECE + Wilson-CI reliability bins, split `pooled`/`backfill`/`forward`, with the W3 top-bin flag |
| `mc_forward_sim.json` | full #483 distributional report (fan + terminal/drawdown distributions + copula corr-to-1 tail + reconciliation) |
| `mc_equity_bands.json` | bands-only slice for the chart fan |
| `tracker_state.json` | the continuation state (WheelTracker JSON) — the forward path loads + re-arms this |
| `meta.json` | provenance, config, data frontier, caps detail, and the full caveat block |
| `_backfill_raw/` | the raw `run_backtest` output the seed was built from |

---

## 5. Scheduling the daily forward-append

**Do not** install a cron/scheduled task blindly — pick the cadence the operator
wants. The append is safe to run repeatedly (idempotent per day). Two options:

**Windows Task Scheduler** (mirrors the "SWE IBKR Morning Pull" schtask pattern):

```powershell
$py = 'C:\Users\merty\AppData\Local\Programs\Python\Python312\python.exe'
$repo = 'C:\Users\merty\Desktop\smart-wheel-engine'
$action  = New-ScheduledTaskAction -Execute $py `
  -Argument 'scripts\run_paper_book.py forward --book live_forward' -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At 6:30pm    # after the US close
Register-ScheduledTask -TaskName 'SWE Paper Book Forward' -Action $action -Trigger $trigger
```

**cron** (Linux/mac): `30 18 * * 1-5  cd /path/to/repo && python scripts/run_paper_book.py forward --book live_forward`

Run it on trading weekdays after the close (or after the morning data refresh, so
`as_of` resolves to fresh data). Because `as_of` is clamped to the data frontier,
running it before the day's bar has landed simply re-marks the latest available
point (still idempotent).

---

## 6. Dashboard API slices (engine side)

`engine_api._handle_portfolio_view` serves three **SIM-only** sub-views (they read
the SIM namespace via `engine.paper_book`; the real-data `/api/portfolio/*` slices
are untouched). Book selected by `$SWE_PAPER_BOOK_NAME` (default `live_forward`):

| Endpoint | Serves |
|---|---|
| `GET /api/portfolio/papertrade` | the phase-labelled curve + backfill/forward counts + caveats |
| `GET /api/portfolio/montecarlo` | the MC fan bands + terminal/drawdown quantiles + reconciliation + copula tail |
| `GET /api/portfolio/calibration` | the rolling Brier/ECE + Wilson-CI reliability bins |

Every payload is tagged `source: "simulated"` — a paper book can never masquerade
as a live IBKR pull on the viewer's provenance badge.

**Frontend (`dashboard/`) is owned by the Dashboard terminal (CLAUDE.md §6)** and
is intentionally NOT built here. The engine side + shaped data are ready; the React
view is handed off — see `docs/PAPER_TRADING_PANEL_HANDOFF.md`.

---

## 7. Honesty caveats (stated on the panel + in `meta.caveats`)

- **Synthetic fills.** Paper P&L uses **synthetic BSM premiums** (the engine's
  `premium` column net of the S32 friction model), **not real option fills** — so
  paper edge is **optimistic** vs reality. Real fill/slippage realism is the first
  tiny-live-size test, not this book.
- **Backfill is in-sample-ish; only the forward segment is true OOS.** The boundary
  is labelled.
- **The MC bands are a model; the realized paper curve is the honest track.** The
  copula corr-to-1 tail is reporting/stress-overlay only (never feeds EV, §2).
- **prob_profit's (0.90, 1.0] top bin is known over-confident** by 10-18pp at
  calm/elevated entry (finding W3 / `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`).
  The calibration report flags it; do not read it as truth. Per-bin `n` is small
  early — read the Wilson CI, don't label a bin SUPPORTED prematurely.
- The structural engine caveats still stand: **E1** (~92% of seed NAV gain was
  equity-beta, not put alpha), **E3** (single-name-dominated P&L), **E5**
  (parameters in-sample), **D19** (exit-leg cost omitted), **D21** (horizon
  over-dispersion).
```
