---
id: earnings-lockout-restore
title: "Earnings-lockout restoration: PIT-gated snapshot overlay + de-silenced gate registration (D3-1 + D6-1)"
kind: fix
status: shipped
terminal: X
pr: 464
decisions: []
date: 2026-07-02
headline: "Live earnings lockout restored from ~8% to 100% forward coverage via a PIT-gated broad_pull snapshot overlay; event-gate registration failures now logged per stage instead of one silent blanket swallow"
surface:
  - engine/data_connector.py
  - engine/wheel_runner.py
  - tests/test_earnings_calendar_overlay.py
  - tests/test_preflight_environment.py
---

## Goal

Remediation-campaign item 2 (adversarial review 2026-07-01, findings
D3-1 critical + D6-1 high). Two independent failure modes of the same
control:

- **D3-1:** `get_next_earnings` read only `sp500_earnings.csv`, whose
  *forward* coverage is 39/511 names (110 rows, banks that pre-schedule
  to 2028) — at live `as_of=None` the earnings hard-lockout silently
  no-op'ed for 92.4 % of the universe going into the July/August
  earnings season. Verified live: a rank on 2026-07-02 recommended
  35-DTE short puts on AAPL/MSFT/XOM/UNH, all of which report inside
  the holding window (only JPM, one of the 39, locked).
- **D6-1:** each ranker wrapped its whole event-exclusion block
  (forward + back-buffer + corp-action registration + the
  `use_event_gate=False` soft skip) in one bare
  `except Exception: days_to_earn = None` — a single raising stage
  silently un-armed the entire lockout for that ticker,
  indistinguishable from "no earnings scheduled".

## What we tried

Source census across the three candidates on disk (4-agent workflow +
first-hand pandas probes, 2026-07-02):

| source | fwd coverage | knowledge stamp | notes |
|---|---|---|---|
| `sp500_earnings.csv` (consumed) | 39/511 (7.6 %) | n/a (historical record) | the defect |
| `sp500_earnings_yf.csv` | 357/511 (69.9 %) | **none** | May-06 pull: 140 names' dates already passed; 14 roots absent; −1d BMO/AMC skew vs bdp; 25.5 % exact-day agreement |
| `broad_pull/per_name/sp500_snapshot_bdp.csv::next_earnings_dt` | 511/511 (100 %) | `asof=2026-06-18` column | freshest; single date per name; decays ~1 name/day in season |

## What worked

The **snapshot overlay**: union the base file with the snapshot's
`next_earnings_dt`, gated point-in-time on the snapshot's own `asof`
column — the overlay participates only when `as_of >= asof`.

## What didn't

- **A yf union.** 879/12,242 yf rows have no exact Bloomberg date
  match; 70 fall inside the S27/S32/S34 regression windows (incl.
  UNIVERSE_24 members COST and CVX — CVX's yf 2023-07-23 is a Sunday),
  so a naive union **rewrites history** and forces the exact-count
  slow-lane re-baseline this campaign is deliberately deferring
  (item 6). yf also has no knowledge-date stamp to PIT-gate on.
- **A wall-clock preflight test.** `tests/test_preflight_environment.py`'s
  design contract explicitly forbids `date.today()` (deterministic,
  never flaky) — and a test that goes red by clock in September would
  break every unrelated PR. Split into three layers instead (below).

## How we fixed it

- `engine/data_connector.py`: `_load_snapshot_bdp_panel` (lazy
  `BroadPullLoader`, **scoped to the connector's `data_dir`** so tmp-dir
  test connectors stay hermetic — several tests pin
  `get_next_earnings(...) is None` on empty dirs) +
  `_snapshot_bdp_next_earnings` (PIT gate on `asof`; refuses unstamped
  or unparseable rows). `get_next_earnings` = earlier-future-date-wins
  union (tie → richer base row); `get_recent_earnings` = symmetric
  back-buffer union (a snapshot date that just PASSED is the
  IV-crush case). Both add a `source` key
  (`"earnings_csv"`/`"snapshot_bdp"`). Consuming the overlay only ADDS
  gate events — remove-only w.r.t. the ranked book (§2-safe direction);
  structural guard `test_broad_pull_loaders.py` permits broad-pull
  consumption only inside `data_connector`, which this respects.
- `engine/wheel_runner.py` (decision trio, lane-claimed): the blanket
  try/except in all three rankers replaced with per-stage guards —
  module helpers `_fetch_next_earnings` / `_fetch_recent_earnings`
  (hasattr-gated like the existing `get_recent_earnings` treatment,
  `logger.warning` on failure) + `_earnings_event_date` (malformed
  dates logged, not swallowed). Soft-skip control flow moved OUTSIDE
  any try. `_register_corp_action_events`' silent `except: return` now
  warns. Fail-open kept deliberately (an error is not evidence of an
  event — R6-R11 missing-data semantics); the fix is that it is no
  longer fail-*silent*, and a raising forward lookup no longer kills
  the back-buffer/corp-action stages.
- **Staleness (fails OPEN, inverted polarity vs #463's refuse-safe
  rail):** three layers — (1) deterministic
  `EXPECTED_EARNINGS_CALENDAR_ASOF` pin in the preflight (bump on every
  snapshot refresh, catches stale tree/clone); (2) once-per-connector
  runtime `logger.warning` when the overlay is > **45** days older than
  the query date (panel-corrected from 90: per-name forward-lockout
  decay becomes material at snapshot age ~51d — quarterly cadence 91d
  minus the 40d gate lookahead — so a 90d alarm would sit silent through
  ~40 days of un-armed names); (3) opt-in `SWE_LIVE_PREFLIGHT=1`
  wall-clock age check for live bring-up.
- **Refuter-panel round (PANEL ceremony per the WIRING_CAMPAIGN row;
  verdicts §2 CONCERN / regression SAFE / ops CONCERN, zero blockers)**
  — three should-fixes, all addressed in the hardening commit:
  (1) `_load_snapshot_bdp_panel`'s bare except silently un-armed the
  lockout on import/loader failure (the D6-1 class one seam higher;
  proven live via a tree missing `src/`) → now logs
  "earnings-calendar overlay unavailable"; (2) **BRK/B + BF/B fell
  through** a normalize mismatch (connector keeps `/`, loader's
  `ticker_normalized` uses `.`) — BRK/B ranked TRADEABLE at as_of=None
  despite 2026-08-03 earnings in-window → both sides now compared in
  dot-form (the sibling `_pit_dividend_yield` has the same latent
  mismatch; left to its own lane — fixing it moves served dividend
  yields, i.e. EV-moving); (3) the 90d threshold (above). Plus two
  hardenings from panel notes: multi-asof panels serve the NEWEST
  snapshot knowable at ref (a future appending refresh would otherwise
  silently serve the decayed calendar under a green preflight), and
  `_earnings_event_date` now degrades-with-log on truthy non-dict
  returns (DataFrame/list) instead of crashing the run.

## Evidence

- Live A/B (the fix working): `rank_candidates_by_ev` at `as_of=None`
  on 2026-07-02, 5-ticker smoke → **0 rows, 5 event drops** with
  structured reasons (AAPL `earnings@2026-07-31`, MSFT `07-30`, JPM
  `07-14`, XOM `07-31`, UNH `07-16`). On main: 4 rows + JPM only.
  NOTE: CLAUDE.md §4's "five rows = healthy" bring-up oracle is
  therefore seasonally wrong at `as_of=None` — flagged for the item-4
  deployment-truth doc pass (CLAUDE.md is operator-maintained).
- Dated byte-identity: `get_next_earnings("AAPL", as_of="2026-06-04")`
  (the data frontier every dated pin runs at) → `None`, unchanged;
  W15/W16/W30 + the @slow full-universe 480/31 pin at 2026-06-04
  green (frontier < snapshot asof 2026-06-18 ⇒ overlay off).
- Mutation check (worktree at origin/main + new test file): **14/23
  fail on main** — every value-adding pin (overlay serve, merge
  precedence, e2e lock, all three D6-1 behavioral pins) is red there;
  the 9 passing on both are deliberate invariance pins (PIT refusal,
  hermeticity, quiet hasattr-gate).
- Real-data pins (dated, deterministic): AAPL @ 2026-06-20 →
  2026-07-31 `snapshot_bdp`; JPM @ 2026-06-20 → 2026-07-14
  `earnings_csv` (base still wins where it has data); TSLA
  back-buffer @ 2026-07-03 → 2026-07-02.
- Refuter-panel deep A/B (§2 lens, clean same-tree archive method):
  ranked books at as_of=2026-06-10/-17 `DataFrame.equals == True` vs
  merge-base (byte-identical pre-asof); at 2026-06-25 strict SUBSET
  (18→1, zero added, survivor EV byte-equal) — remove-only proven at
  book level, with determinism controls. Ops lens 40-name live run at
  as_of=None: 15 rows / 23 event-locks / 2 pre-existing drops; all 8
  just-reported names >5d out CLEARED (no back-buffer over-blocking),
  Sep reporters produced rows — per-name behavior exactly per spec,
  NOT a #462-style blackout. Regression lens: the @slow full-universe
  480/31 pin passed (81s), fingerprint tests 8/8, ~800 targeted tests
  green across the three lenses.
- Full fast suite + launch blockers green (see the PR body for
  counts); fingerprint guards untouched (`_FILES` unchanged; the
  snapshot is read outside it — same precedent as `dividend_pit`,
  with the pinning gap folded into campaign item 3).

## Unresolved / handoff

- **Item 3 must add broad-pull consumed files to
  `connector_data_sha256`** (`dividend_pit` + now `snapshot_bdp`) — the
  overlay is currently an unpinned read, the exact blind spot item 3
  (regression-lock honesty) exists to close. *(CLOSED 2026-07-02 by
  #465 — both files now pinned as `broad_pull_dividend_pit` /
  `broad_pull_snapshot_bdp`.)*
- The snapshot decays ~1 name/day in earnings season; 15/511 dates had
  already passed at wiring (those names now serve via the back-buffer
  instead). Next broad-pull re-pull refreshes; the three staleness
  layers alarm if it doesn't happen by ~mid-September 2026.
- `engine_api._handle_calendar` (`/api/calendar`) bypasses the
  connector (reads `sp500_earnings.csv` via `data_integration`) — the
  dashboard calendar still under-shows forward earnings. Advisory
  surface, not decision-layer; left for the interface lane.
- `analyze_ticker`'s events block (advisory path) still has its own
  broad `except: pass` — same D6-1 pattern, non-§2 surface, left
  untouched to keep this PR's trio diff minimal.
- Deep-historical earnings backfill (2026-Q2 has rows for only 75/503
  names even though those announcements HAPPENED) remains open —
  EDGAR 8-K Item-2.02 (`scripts/pull_edgar_earnings.py`) is the
  planned PIT-correct source.
