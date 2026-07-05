---
id: branch-merge-down-2026-07-04
title: "Branch merge-down: 14 unmerged remote branches → 6 PRs of extracted value + 2 carriers kept"
kind: fix
status: shipped
terminal: X
pr: 482
decisions: []
date: 2026-07-04
headline: "Operator-directed merge-down of the 14-branch backlog: every branch's unlanded value extracted onto main (campaign docs, producer scripts, lab tooling, backfill fragments, R11b evidence, 3 mined engine items), 12 branches deleted with SHAs recorded, 2 data carriers kept"
surface:
  - engine/wheel_runner.py
  - engine/ev_engine.py
  - engine/dealer_positioning.py
  - backtests/regression/_common.py
---

## Goal

Operator: "now lets focus on the branches. try to merge them down." After the
PR backlog clear (0 open PRs at main `0be7f6c`), 14 unmerged remote branches
remained. Method: per-branch supersession scan (ahead/behind, cherry
patch-equivalence, content-vs-main diffs), extract every unlanded artifact via
fresh PRs rebuilt on current main, keep irreplaceable-data carriers, delete
the rest with tip SHAs recorded here (restorable via
`git push origin <sha>:refs/heads/<name>` while objects persist).

## What landed (6 PRs)

- **#475 mag7** (`087c66d`): MAG7 reliability campaign doc + driver + analyzer.
- **#476 data-acq plan** (`de00a57`): DATA_ACQUISITION_PLAN_2026-06-14 as dated record.
- **#479 sim200k** (`6a229e1`): SIM-200K campaign + `_common.py` opt-in
  `enforce_single_name_cap` harness-layer R10 emulation (default False,
  canonical path byte-identical; PIT NAV, not `_compute_live_nav` lookahead).
- **#477 salvage** (`f0a6595`): r11b_skew_edge_impact artifacts (#437's close
  comment cited them as "retained" — the citation was dangling until now);
  THETA_PULL_AUDIT_2026-06-15; the 10 refresh-branch-only xbbg producer
  scripts (several are the ONLY in-repo producer of a tracked connector CSV);
  the broad-pull `staging/` lab tooling incl. the macro_calendar re-pull
  script the #462 gate will need when its 2025→2027 calendar ages out;
  4 missing worklogs.
- **#478 phase1a fragments** (`2844328`): 6.9 MB carrier-only landing of the
  Phase-1A backfill fragments + phase1b integration recipe. **The defect they
  fix is still live on main**: batch-1 blue chips truncated under a wrong UW
  exchange code (WMT 141 rows, KMB 274, CPB 450, DPZ 375, PLTR 399) and the
  2026-03-23 reconstitution entrants at ~52-71 rows. Integration is
  re-baseline-coupled → stays with the operator-gated data batch.
- **#480 rescue-mined engine items** (`951469c`): the supersession analysis of
  the rescue WIP (`c1fdbd4`) found 3 unlanded items, re-implemented fresh:
  (1) `hmm_converged` on the diagnostic row (main docs already claimed it
  existed — doc-vs-code drift closed; with n_iter=20 fits are known
  non-converged, and that honest False IS the audit value); (2)
  `distribution_source` engine-truth relabel on all 3 rankers (cascade "none"
  no longer masks `lognormal_fallback`, which feeds the EV-authority token
  hash); (3) `DEALER_MULT_FLOOR/CEIL` named constants + unconditional source
  clamp + ev_engine apply-site re-assert (behavior-neutral §2 hardening).
  Full fast suite 3468/0. Consciously NOT ported: per-row `spot_date`
  (universe-level staleness landed via #470), `rel=1e-9` regime-linearity
  tightening (determinism premise never CI-verified), R11 pins in
  test_dossier_invariant (covered by test_r11_elevated_vol).

## Carriers kept (2)

- `deep-history/bloomberg-raw` @ `68a48b245c` — sole carrier of the deep
  1994+ OHLCV/vol_iv/liquidity/IV-surface slices + delisted panels (~150 MB;
  main's `data/bloomberg/deep/` is empty by design, SWE_DEEP_HISTORY-gated).
  Extended this session with the 2026-06-04 deep-history session transcript
  pair rescued off the refresh branch before its deletion.
- `claude/daybot-bloomberg-pull` @ `2abf850d76` — 468 MB of SPY/QQQ tick
  data (Bloomberg intraday lookback is finite; deleting destroys it) + the
  day-bot pullers. Out of engine scope (§3: no tick-level features) — never
  merge into main; pure data carrier.

## Bonus fix: #481 — calendar time-bomb defused

#480's first CI run failed both suites on
`test_asof_none_staleness.py::TestFutureCorruptRow` — a main defect, not an
F regression: the test pinned FRESH's bars at the module FRONTIER
(2026-06-04) while its connector clamps the corrupt 2099 frontier to
`date.today()`, so it could only pass while today−FRONTIER ≤ 30d. From
2026-07-05 UTC it failed on EVERY CI run on any branch. Fixed test-only in
#481 (`1423fea`): bars built fresh-to-today so the assertion tests the
intended property (2099 row doesn't inflate the frontier). 4th instance of
the frontier/wall-clock ticking class this cycle — this one a loud red, the
better failure mode.

## Delete-ready (12, tip SHAs for restoration — deletion awaits operator confirmation)

| Branch | Tip | Value disposition |
|---|---|---|
| `claude/mag7-reliability` | `cf9fd63441` | landed via #475 |
| `claude/sim200k-reliability` | `46b8deb615` | landed via #479 |
| `claude/data-acquisition-expansion` | `172fb17d25` | landed via #476 |
| `claude/bloomberg-broad-pull-2026-06-17` | `c0b5945ea4` | data already on main (`broad_pull/`); tooling via #477; 06-05→06-18 fragments superseded by #472 |
| `claude/mac-439-ohlcv-split-scale-fix` | `53663eba54` | fully superseded (#455 closed; script+doc on main; splice re-applied by #472) |
| `claude/theta-top20-deep365` | `4ca7c66788` | docs on main (#415 closed superseded) |
| `claude/theta-iv-surface-puller-EXPERIMENT` | `3851064324` | timeout/workers tune patch-equivalent on main; DTE-knob prototype self-labeled NOT FOR MERGE (finding: capped buckets drop crisis front-weeklies — recorded in memory) |
| `claude/r11-skew-edge-gate` | `2a67dd39cb` | evidence via #477; R11b feature REJECTED per #437 close (its DECISIONS/CLAUDE.md edits pin R11b as shipped = false) |
| `claude/rescue-2026-06-15-fixes` | `c1fdbd46ea` | docs on main via #422 era; THETA_PULL_AUDIT via #477; WIP items 1-3 via #480; skipped items recorded above |
| `claude/phase1a-casy-bloomberg-pull` | `f117fe0ca0` | fragments via #478 |
| `claude/phase1b-fragment-integration` | `3e363dcde7` | recipe + worklog via #477/#478; monolith payload written against pre-#472 data (unusable as-is) |
| `data/bloomberg-refresh-2026-06-02` | `6bb3399b57` | 16-file refresh landed via #338 + superseded by #472; extras superseded by broad_pull; delisted lives on deep-history; 10 producer scripts via #477; 06-04 transcript rescued to the deep-history carrier; **engine-REVERT hazard branch — never merge** |

Deletion of the 12 was attempted at campaign end and blocked by the
permission classifier (mass remote-branch deletion requires explicit
operator consent — "merge them down" was read as extraction, not deletion).
Every branch above is fully extracted; deleting them loses nothing. The
operator's one-word go-ahead executes the batch.

## Traps / notes

- The refresh branch was NOT an ancestor of deep-history (siblings) — the
  06-04 session transcript existed only on refresh and would have been lost;
  rescued to the carrier as `68a48b2`.
- CI lints an explicit path list: `staging/` and `docs/` are outside it, so
  lab tooling landed verbatim; the 10 `scripts/` producers needed
  `ruff format` + mechanical E702/C408 fixes.
- The manifest coverage gate accepts directory tokens — one row per
  `staging/` bucket keeps 28 fragment files covered without 28 rows.
- Lane claim posted for the #480 trio touch:
  issues/113#issuecomment-4884084736.

## Unresolved / handoff

- **Phase-1A integration** (fixes the truncated-names defect) — operator-gated
  data-batch session; recipe = `staging/integrate_phase1b.py`, re-validate
  against the fresh #472 monolith first; re-baseline-coupled.
- Truncated-name defect list is worth a check in the batch session: WMT/KMB/
  CPB/DPZ under wrong `UW` code, plus CASY/VEEV/COHR/LITE/SATS/VRT
  reconstitution entrants.
- `hmm_converged` will read False broadly until the item-6 HELD n_iter bump.
