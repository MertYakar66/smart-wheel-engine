---
id: Phase-0B
title: Phase 0B — broad-pull connector loaders (additive, dormant, §2-safe)
kind: feature
status: held
terminal: autonomous session 2026-06-22
pr:
decisions: [D10]
date: 2026-06-22
headline: 27 net-new broad-pull datasets integrated under data/bloomberg/broad_pull/ with a dormant BroadPullLoader + 53 tests; nothing consumes it (EV-moving wiring is Phase 1-3, supervised)
surface: [data/broad_pull_loaders.py, tests/test_broad_pull_loaders.py, data/bloomberg/broad_pull/, FILE_MANIFEST.md, CHANGELOG.md, pyproject.toml]
---

## Goal

Execute **Phase 0B** of `docs/WIRING_CAMPAIGN.md` (PR #416) — the §2-SAFE plumbing
slice: make the banked broad-pull data *loadable* without wiring it into any
consumer. Stop at the EV-moving boundary. Concretely: integrate the ~25 net-new
staged datasets to a connector-read location, add a `load_*` accessor per file
(roadmap §10 idiom), winsorize the manifest's outlier-flagged columns at load
(not silently), and write a test per loader. Everything held for review; nothing
that moves `EVEngine.evaluate` runs unattended.

## What we tried

- **Studied the existing idiom first.** `data/consolidated_loader.py`
  (`ConsolidatedBloombergLoader`: `_load_csv` → lowercase → optional col_map →
  `_index_by_ticker`; `get_*` accessors; `load_all`; singleton) is the roadmap
  §10 target. CI gates that bind: `ruff check` + `ruff format --check` over
  `data/`+`tests/`; `manifest-coverage` (FILE_MANIFEST covers every tracked
  file; worklog INDEX current; doc-currency); `pytest --cov-fail-under=80` with
  every `data/*` loader **omitted** from coverage (D10 — research-tier ETL off
  the decision path). mypy is non-blocking and `src/`-only; the decision-layer
  lane-claim only fires if the trio changes (it doesn't here).
- **Data placement.** Branch is off `main` (per the brief), so the staged bytes
  (on `claude/bloomberg-broad-pull-2026-06-17`) had to be brought in. Extracted
  the 27 net-new files **byte-identically** via `git show <ref>:staging/... >
  data/bloomberg/broad_pull/...`; blob hashes match the source → git dedupes the
  objects (already on origin from the broad-pull branch), so the push is cheap
  despite the ~352 MB working tree.
- **Loader shape.** A single isolated module `data/broad_pull_loaders.py` with a
  spec-driven `BroadPullLoader` (one `DatasetSpec` per dataset; generic
  `load`/`panel`/`series`/`category_series`/`snapshot_row` + a thin named
  `load_<name>()` per file). Per-ticker access is **lazy** (filter the cached
  panel) rather than the existing loader's eager dict-of-~510-copies, which would
  double the memory of the multi-million-row panels alongside the live day-bot pull.

## What worked

- **Dormant by construction.** The module is imported by **nothing** on the
  decision path (verified repo-wide + a structural test
  `test_module_not_consumed_by_production`), and it is **not** registered in
  `ConsolidatedBloombergLoader.load_all`. So it is loadable but unconsumed —
  exactly the Phase-0B contract. Wiring it in is the EV-moving Phase 1-3 work.
- **Winsorization, logged.** The manifest's outlier-flagged columns
  (`vol_term_rv` IV/RV, `beta_shares.beta_raw`, `valuation_m` multiples,
  `options_sentiment` pc-ratios/news_sent) are clipped to their 0.1%/99.9%
  quantiles; every clamp emits a WARNING (never silent); row counts are
  preserved, so the byte-pinned counts still hold. Verified against the raw
  bytes in `test_real_winsorization_applied`.
- **gz + dtype + dates.** `.csv.gz` (`iv_surface`, `vol_term_rv`) read via
  pandas `compression='infer'`; floats downcast to float32; date columns parsed
  to datetime. 53 tests pass (synthetic + real-data) in ~62 s; ruff clean.

## What didn't

- **First real-data run: `series("returns_micro","AAPL")` was empty.** The
  per-name files store **Bloomberg-style tickers** (`A UN`, `AAPL UW`), not plain
  symbols, so exact-match lookup found nothing. Fixed by reusing the canonical
  `normalize_ticker` from `consolidated_loader` — a `ticker_normalized` column is
  added at load (raw `ticker` preserved so its byte-pinned distinct count is
  unchanged), and `series`/`snapshot_row` match on it. This also makes the
  loader's universe align with the connector's, which the future wiring needs.
- **The data-commit trade-off.** Committing ~352 MB onto a branch off `main`
  duplicates bytes the broad-pull branch already holds. Accepted because (a) the
  brief lists integration as a Do and wants real-data tests, (b) git dedup makes
  the push cheap, and (c) tests are gated `HAS_BROAD_PULL_DATA` so the code PR is
  valid even if a reviewer strips the data. At the supervised merge, dedupe the
  bytes to one location (`staging/` vs `data/bloomberg/broad_pull/`).

## How we fixed it

Shipped: `data/broad_pull_loaders.py` (dormant `BroadPullLoader`), the 27
integrated files under `data/bloomberg/broad_pull/`, and
`tests/test_broad_pull_loaders.py` (53 tests). Supporting: FILE_MANIFEST rows
(module, `data/bloomberg/broad_pull/` glob, test), CHANGELOG `## 2026-06-22`,
`pyproject.toml` coverage-omit for the new loader (D10), this worklog +
regenerated `docs/worklog/INDEX.md`. Held for review; no consumer wiring, no
decision-trio/risk-gate edits, no Phase 0A frontier overwrite, no re-baseline.

## Evidence

- Extraction integrity: `git hash-object` of each extracted file == the source
  blob (`git rev-parse <ref>:staging/...`) — byte-identical, 27/27.
- Tests: `pytest tests/test_broad_pull_loaders.py` → **53 passed** (~62 s),
  pinning per-dataset rows/date-range/ticker-count/schema to the byte census in
  `docs/DATA_INVENTORY.md` §6 (e.g. `iv_surface` 1,944,699 rows / 509 names /
  2010-01-04→2026-06-17; `options_sentiment` 1,998,083 / 511).
- `ruff check` + `ruff format --check` clean; repo-wide grep confirms no module
  imports `broad_pull_loaders`.

## Unresolved / handoff

- **Acceptance criteria for the supervised Phase 1-3 wiring** (write the
  behaviour test as each lands, then wire — these are NOT done here, they are
  EV-moving):
  - **#372 (R9→GICS):** R9 groups a served name by its real
    `gics_sector_name` (from main `fundamentals`, **not** the 2026-06-18
    `snapshot_bdp` — lookahead), with a counted `'Unknown'` fallback.
  - **#369 / #378:** fundamentals-fallback IV is cleaned by the #363 gate; the
    `_resolve_pit_atm_iv` staleness gate fires; land #378 **before** the 0A
    frontier bump (it opens the IV↔spot gap).
  - **Phase 2 skew:** `option_pricer` IV reflects the moneyness curve from
    `iv_surface` (put 90%MNY IV > ATM for a put-skewed name); `edge_vs_fair`
    stays 0 (C4 — skew does not revive VRP).
  - **Phase 3:** macro_calendar → `event_gate` (PCE/the 11 events lockout);
    `dividend_pit` → BSM `q` PIT (#354); ratings → a downgrade-only credit
    reviewer (0.85×); short_interest → a new R10 soft-warn.
- A companion **xfail(strict) scaffold** for these may be added test-only; until
  then the structural §2 guard tracks the boundary.
- At merge: reconcile the duplicated broad-pull bytes to a single location.
