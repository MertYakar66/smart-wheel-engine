# Theta Data Pull — Session Notes

This document captures what we worked through to bring up a clean Theta data
pull on the laptop and the gotchas to remember next time. It complements
`LAPTOP_SETUP.md` (machine bring-up) and `docs/THETA_USAGE.md` (per-endpoint
reference) — read those first; this one is the operational checklist.

## Goal

Run a full daily Theta refresh on the laptop and verify the v3 path is
healthy end-to-end. Result: `pull_all.py` ready to execute, IV surface
puller verified against AAPL, dry-run plan clean.

---

## Step 1 — Verify the Terminal is up

The Theta Terminal serves a local HTTP API on port 25503. Before doing
anything else, confirm the socket is open.

```bash
python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 25503)); print('UP')"
```

Expected output: `UP`. If you get `Connection refused`, the Terminal isn't
running — launch it from Spotlight or directly:

```bash
open "/Users/mertyakar/Library/CloudStorage/GoogleDrive-mertyakar.my@gmail.com/My Drive/Theta/ThetaTerminalv3.jar"
```

Wait roughly 10 seconds after launching for it to connect upstream before
re-checking.

---

## Step 2 — Make sure exactly ONE Terminal instance is running

Multiple Terminal processes step on each other's session and every API call
starts returning HTTP 478 ("Invalid session ID"). This is the most common
cause of cryptic auth-style failures. Check:

```bash
ps aux | grep -i theta | grep -v grep
```

You should see a small number of entries: typically the Theta Terminal
launcher and its JAR. If you see two distinct sets of `java -jar
ThetaTerminalv3.jar` processes (different PIDs, different start times), or
any leftover `pull_theta_*.py` scripts from yesterday's session, kill them
all:

```bash
kill <pid1> <pid2> <pid3> ...
sleep 2
ps aux | grep -i theta | grep -v grep    # should be empty
```

Use `kill -9 <pid>` only if a graceful kill doesn't work. Then relaunch one
Terminal cleanly and wait for it to connect.

---

## Step 3 — Skip the v2-style probe

`scripts/probe_theta_capabilities.py` sends about 30 small requests across
stock, option, index, and futures endpoints. It's useful in principle but
has two practical issues you should know about:

First, it hits the Terminal sequentially with a 10-second per-request
timeout. The four endpoints `option/history_eod`,
`option/history_open_interest`, `option/history_greeks_iv`, and
`option/history_greeks_first` are heavy chain-aggregate queries (no strike
filter). They legitimately take longer than 10s on first hit and the probe
falsely reports them as `! ERROR  HTTPConnectionPool(...)`. The connector
itself uses `timeout=30` and chunked retries, so those endpoints actually
work in practice.

Second, the v3 capability matrix you actually need is already pinned in
`docs/DATA_POLICY.md` §2 (capability matrix) and `docs/THETA_USAGE.md` — fundamentals
and corp actions are not in v3, VIX futures are tier-blocked, snapshots are
blocked, EOD index history works.

**Recommendation:** do not gate your daily runs on the probe. Instead,
validate the v3 path with a real puller (next step). If you do want to run
the probe, raise the timeout from `10.0` to `30.0` in `run_probe()` (around
line 188).

---

## Step 4 — Smoke-test the v3 path with one real puller

Pick a small ticker and a short window. AAPL is a good target because it's
the in-git sample and existing partitions will be picked up via resume
mode, so a no-op success means the connector is happy.

```bash
python scripts/pull_theta_iv_surface_history.py --tickers AAPL --days 7 --workers 1
```

Expected output looks like:

```
Skipping 1 same-day/future date(s) (... ) — Theta history endpoints require explicit time bounds for current-day requests
Pulling IV surface history  tickers=1  dates=7  jobs=7  workers=1  strict=True
Done in 0.0s  |  7 OK  |  0 FAIL
Written under: .../data_processed/theta/iv_surface_history
```

Verify the output partitions:

```bash
ls -la "data_processed/theta/iv_surface_history/ticker=AAPL/" | head -20
```

You should see Hive-style partitions (`year=2024`, `year=2025`,
`year=2026`). If this all comes back clean, the v3 path is healthy and you
can move on to the full refresh.

---

## Step 5 — Dry-run the orchestrator

Before kicking off the full refresh, see exactly what `pull_all.py` plans
to do.

```bash
python scripts/pull_all.py --dry-run
```

This prints the 11-step plan with one line per step. Steps that need Theta
will note the Terminal is up; the `news` step will SKIP if no
`POLYGON_API_KEY` / `FINNHUB_API_KEY` / `BENZINGA_API_KEY` is in your
environment (this is normal — news isn't required for the EV path).

Expect two steps to either skip or write very little:

- `theta_vix_futures` — tier-blocked, will return no data
- `theta_corp_actions` — not in v3, will return no data

Both are non-fatal.

---

## Step 6 — Run the full refresh

Make sure `caffeinate` is still alive so the laptop doesn't sleep mid-run.
If you're starting a fresh session:

```bash
caffeinate -dimsu -t 3600 &
```

Then:

```bash
python scripts/pull_all.py
```

Total wall-clock is typically 15–25 minutes — most of the second half is
the feature store backfill. The script prints per-step status as it goes; a
FAIL row is what to look for, not silence.

Optional flags worth knowing:

```bash
python scripts/pull_all.py --skip theta_vix_futures news    # skip known-empty steps
python scripts/pull_all.py --only theta                     # Theta steps only
python scripts/pull_all.py --years 2 --workers 8            # tune scope and parallelism
```

---

## Step 7 — Verify

```bash
python scripts/feature_smoke_test.py --section theta --verbose
```

Sections 15 (connector), 22 (history pulls), and 26 (outputs) should all
report PASS. Persistent SKIP rows are tier-blocked endpoints (VIX futures,
snapshots, etc.) and are expected per `docs/DATA_POLICY.md` §2. What you **don't**
want to see: any FAIL.

For a faster sanity check, you can run:

```bash
python scripts/feature_smoke_test.py --fast
```

---

## Things to remember

The codebase is fully on Theta API v3. The connector at
`engine/theta_connector.py` and every `scripts/pull_theta_*.py` puller use
`/v3/...` paths with `symbol` and `expiration` params. There is no v2
anywhere in the live code — if you ever see an HTTP 410 "We have upgraded
to API v3" response, you typed a `/v2/...` URL by accident in a curl test
(which is exactly what tripped us up earlier).

Drive mounts are eventually-consistent mirrors and deny `unlink` on tracked
files, so `git pull` against the Drive worktree fetches refs but can't
update the working tree. To read a newer revision from Drive without
checking it out, use `git show origin/<branch>:<path>`. Treat
`~/Desktop/smart-wheel-engine` as the source-of-truth checkout on this
laptop.

The `.claude/SessionStart` hook validates dataset presence, Theta manifest
recency, and connector class on every fresh Claude session. If a future
session shows an unexpected provider class or stale manifest warning, that
hook is where to look first.

---

## Quick-reference: the canonical bring-up

If you only need to remember one block of commands, this is it.

```bash
cd ~/Desktop/smart-wheel-engine
caffeinate -dimsu -t 3600 &
python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 25503)); print('UP')"
ps aux | grep -i theta | grep -v grep
python scripts/pull_theta_iv_surface_history.py --tickers AAPL --days 7 --workers 1
python scripts/pull_all.py --dry-run
python scripts/pull_all.py
python scripts/feature_smoke_test.py --section theta --verbose
```
