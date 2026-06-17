# Day-bot Bloomberg intraday pull — manifest (lab session 2026-06-17)

Branch `claude/daybot-bloomberg-pull`. **Separate concern from the wheel** — intraday + event data
the day-trading engine needs. Bloomberg intraday history is capped at ~140 calendar days (perishable)
→ pulling the full window now. Lab PC is ephemeral; persisted to GitHub. Reproducible producers
committed (the spec calls out connector files that lacked a producer — not repeating that).

## Pipeline validated ✓
`bdib` (1-min bars) and `bdtick` (NBBO+trades) both work and are **entitled**. Both return **correct
UTC** (no conversion); bars are **start-labelled**. Canonical, explicit headers enforced (the spec's
#1 rule, given `sp500_ohlcv.csv`'s rotated-column history). Validated on `SPY_ticks_2026-06-10`:
8,094,823 ticks, cols `ts,type,value,size`, UTC open `13:30:00.001Z`→close `20:00:00.927Z`, types
BID 3.94M / ASK 3.75M / TRADE 0.40M, NBBO sane (BID 731.00 < ASK 731.15 → measurable spread).

## The three pulls
| # | producer | output | scope | est |
|---|---|---|---|---|
| **1 — ticks** (highest) | `pull_ticks.py SPY,QQQ <start> <end>` | `ticks/<SYM>_ticks_<date>.csv.gz` (per symbol-day) | SPY+QQQ × ~101 trading days (2026-01-27→06-17) | ~8–16 h, ~6 GB |
| **2 — bars** | `pull_bars.py <start> <end>` | `bars_1m/<SYM>_1m.csv` (one per symbol) | 24 ETFs/equities + SPX Index, 1-min RTH | ~1.5 h, ~75 MB |
| **3 — events** | `pull_events.py` | `events/macro_calendar.csv` | 12 macro events (+PPI), date+time-ET+survey/actual/prior | ~5 min |

Run order = priority (ticks → bars → events), one terminal request stream (no concurrency).

## Hygiene
- **UTC** timestamps, explicit (`ts` ISO with `+0000`); bars start-labelled. Events store ET time +
  an explicit `tz_label` column.
- **Canonical headers** written explicitly, not Bloomberg field order.
- RTH = 09:30–16:00 America/New_York → UTC per date (handles the EST↔EDT boundary).
- `pull_ticks.py` is **resumable** (skips completed `.csv.gz`) + **atomic** (`.tmp`→rename, so an
  interruption never leaves a corrupt file) — safe for an unattended multi-hour run.

## Storage decision
Lab PC ephemeral + GitHub preferred → committed to this branch. Google Drive via MCP isn't viable for
multi-GB binaries. Ticks are per-symbol-day gz (each ~27–44 MB, under GitHub's 100 MB file limit);
committed incrementally as they land. **Recommendation:** the full ~6 GB tick set ideally migrates to
a standalone day-bot repo (or Git LFS) rather than permanently growing the wheel remote — flagged for
follow-up.

## NOT pulled (per spec — already have / wrong tool)
VIX term structure / futures / vol indices (wheel), 503-name EOD panels (wheel), intraday option
chain + tape (Theta, not Bloomberg).

## Run status
Facilitated run with ~30-min health checks; incremental commits per batch. See git log of this branch.
