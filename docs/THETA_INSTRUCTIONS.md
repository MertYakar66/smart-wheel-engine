# Theta — pull-the-data instructions

Quick-reference for refreshing every Theta-sourced dataset the engine
consumes. For the long-form rationale and the full table of what's covered
vs. not, see `docs/THETA_USAGE.md`.

---

## 1. Bring the Terminal up

Theta's HTTP API is served by the **Theta Terminal** tray app on
`http://127.0.0.1:25503`. Nothing below works until it's running.

```
# start Theta Terminal (the desktop app), then verify:
python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 25503)); print('UP')"
```

If that prints `UP`, you're good. Otherwise open the Terminal and retry.

---

## 2. Probe what your subscription unlocks

Run this first — and again any time you change tiers. It hits every
endpoint and tags each one `✓ OK / - EMPTY / × BLOCKED / ? MISSING / ! ERROR`.

```
python scripts/probe_theta_capabilities.py
```

Writes `data_processed/theta_capabilities.json` for the audit trail.

---

## 3. The one command you'll use 95% of the time

```
python scripts/pull_all.py
```

Runs all pullers in dependency order (vol indices → treasury → fundamentals
→ earnings → Theta indices → VIX futures → corp actions → IV surface →
options flow → news → feature backfill). Steps that need Theta are skipped
automatically when the Terminal is down; the news step is skipped when no
API key is in the environment.

Useful flags:

```
python scripts/pull_all.py --dry-run                        # print the plan
python scripts/pull_all.py --only theta                     # only Theta steps
python scripts/pull_all.py --skip theta_vix_futures news    # skip specific
python scripts/pull_all.py --years 2 --workers 8            # tune horizon / parallelism
```

Safe to schedule daily (cron / Task Scheduler).

---

## 4. Individual Theta pulls (if you need granular control)

Run any of these directly. Each is incremental by default and idempotent.

| Dataset | Command | Output |
|---|---|---|
| VIX-family index history | `python scripts/pull_theta_indices_history.py --years 5 --incremental` | `data_processed/vol_indices.parquet` (Theta rows win on duplicates) |
| VIX futures curve UX1–UX8 | `python scripts/pull_theta_vix_futures.py --years 5 --months 8 --incremental` | `vix_futures.parquet`, `vix_futures_wide.parquet` |
| IV surface per ticker | `python scripts/pull_theta_iv_surface_history.py --universe sp500 --days 7 --workers 4` | `data_processed/theta/iv_surface_history/ticker=<X>/year=<Y>/date=<YYYY-MM-DD>.parquet` |
| Daily options flow | `python scripts/pull_theta_options_flow.py --universe sp500 --days 30 --workers 4` | `data_processed/theta/options_flow/<TICKER>.parquet` |
| Corp actions (splits + divs) | `python scripts/pull_theta_corp_actions.py --universe sp500 --years 10 --workers 4` | `data_processed/corporate_actions/{splits,dividends}.parquet` |
| Intraday option tape (heavy) | `python scripts/pull_theta_option_tape.py --tickers AAPL --days 5 --atm-only` | `data_processed/theta/option_tape/ticker=<X>/date=<Y>/{trades,quotes}.parquet` |

**First-time bulk loads:** swap `--days 7` / `--incremental` for an explicit
`--start YYYY-MM-DD` (e.g. `--start 2024-04-01`) to backfill from scratch.

**Workers:** keep at `--workers 4` or below — Theta rate-limits aggressive
clients.

---

## 5. Catch-all backfill (alternative to per-pull scripts)

`scripts/theta_backfill.py` exposes the same data via a single CLI with
subcommands. Use it when you want one process driving everything:

```
python -m scripts.theta_backfill all                  # everything
python -m scripts.theta_backfill all --tickers AAPL,MSFT --start 2024-01-01
python -m scripts.theta_backfill all --workers 3 --overwrite
```

Writes a manifest at `data_processed/theta/_manifest.json`.

---

## 6. Verify

```
python scripts/feature_smoke_test.py --section theta --verbose
```

Sections 15 / 22 / 26 cover the Theta connector, history pulls, and
on-disk outputs. Anything still `SKIP` after a fresh pull means that
endpoint isn't on your tier — re-check `probe_theta_capabilities.py`.

---

## 7. Daily routine

```
python scripts/pull_all.py                    # ~5–10 min, +~15 min if features rebuild
python scripts/feature_smoke_test.py --fast   # verify
```

That's it. For deeper context (per-endpoint coverage, optional pulls,
caveats) see `docs/THETA_USAGE.md`.
