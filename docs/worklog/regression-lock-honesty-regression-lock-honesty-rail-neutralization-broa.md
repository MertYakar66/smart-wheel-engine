---
id: regression-lock-honesty
title: "Regression-lock honesty: rail neutralization, broad_pull fingerprint, corp-action payload guard (D4-2 + D5a-1)"
kind: fix
status: shipped
terminal: X
pr:
decisions: []
date: 2026-07-02
headline: "Replays and exact-EV test pins are now rail-independent (local == CI == committed baselines by construction); the two broad_pull connector inputs are fingerprint-pinned; a truthy non-DataFrame corp-action payload no longer kills the whole ranking run"
surface:
  - backtests/regression/_common.py
  - conftest.py
  - engine/data_connector.py
  - engine/wheel_runner.py
---

## Goal

Remediation-campaign item 3 (adversarial review 2026-07-01, findings
D4-2 + D5a-1), folding in the #463 verdict's cosmetic notes n2/n3 and
the #464 verdict's nit v1:

- **D4-2:** the regression lane and the fast-suite exact-EV pins were
  rail-*dependent*: a box with a produced (gitignored)
  `data_processed/option_premium/` larder ran replays and tests against
  real market mids while CI and the committed baselines (locked
  rail-off at b3aa236, pre-#435) ran synthetic-BSM. Live instance: the
  AAPL $5.35 F4 control red locally, green in CI, for weeks.
- **D5a-1:** the connector consumes two broad_pull files
  (`dividend_pit` panel → BSM carry-q; `snapshot_bdp` → the #464
  earnings-calendar overlay) **outside** `_FILES` — both were unpinned
  reads the `connector_data_sha256` drift guard could not see.
- **#464 v1:** `_register_corp_action_events` probed
  `len(ca)`/`ca.iterrows()` outside any guard — the one payload class
  the old blanket except caught that the #464 per-stage guards didn't.
- **#463 n2:** `list_option_expirations` hardcoded the wall-clock
  staleness 7 where `get_option_premium_chain` parameterizes it.

## What we tried

4-lens recon workflow (regression lane + fingerprint architecture /
rail-test class / accessor parameterization / corp-action guard), every
load-bearing claim verified first-hand before design.

## What worked

- **Neutralize at both consumption layers.** A session-scoped autouse
  fixture in the root `conftest.py` (empty tmp dir) for the whole
  pytest suite, plus a scoped env pin around connector construction in
  both `_common.py` replay drivers (nonexistent dir, save/restore).
- **Fingerprint via an explicit extras map** (`_BROAD_PULL_PINNED`),
  NOT by adding the files to `MarketDataConnector._FILES`.

## What didn't

- **Pinning the env var to empty string.** The connector treats
  unset/EMPTY as "use the repo default dir" — the exact vector: the
  operator shell had the var *unset* and the rail was still active via
  the default-path fallback. The pin must be a nonexistent/empty
  directory path.
- **A function-scoped autouse fixture.** Module/class-scoped runner
  fixtures (`test_w2_output_realism.ranked`, `TestF4CasesRanker.runner`)
  construct the connector during fixture setup, *before* any
  function-scoped autouse fixture runs — session scope is load-bearing.
- **Extending `_FILES` instead of an extras map.** `_FILES` feeds
  `_load()`'s generic CSV normalization and the data audits' core-CSV
  loops; the broad_pull files have their own loaders (spec'd dtypes,
  hermetic scoping) and must not be dragged through either.
- **Wrapping the whole corp-action loop in the new guard.** That would
  swallow genuine `EventGate.add_event` bugs, contradicting the #464
  fail-loud design note; only the payload-shape probe is guarded.

## How we fixed it

- `conftest.py`: `_neutralize_option_premium_rail` (session, autouse) —
  `pytest.MonkeyPatch().setenv` to an empty tmp dir. Rail tests keep
  opting in per-test with their own `monkeypatch.setenv` (function
  scope overrides the session baseline and restores on teardown).
- `backtests/regression/_common.py`: `_option_premium_rail_pinned_off`
  contextmanager (nonexistent dir; restores prior shell state) wrapped
  around `WheelRunner()`/connector construction in `run_backtest` and
  `run_backtest_multi_friction` — the env is read once per connector
  construction and every rail read flows through that single `conn`.
  Belt-and-braces `_assert_rail_neutralized(conn)` fails loud if a
  refactor ever moves construction outside the pin. Both fingerprints
  record `option_premium_rail: "pinned_off"`.
  `connector_data_sha256()` hashes `_FILES ∪ _BROAD_PULL_PINNED`; the
  four snapshot JSONs got the two sha entries hand-inserted (the slow
  test compares metric leaves only — engine output is untouched, so NO
  slow-lane re-run; `generated_at` intentionally left describing the
  original b3aa236 runs).
- `engine/wheel_runner.py` (trio, lane-claimed):
  `_register_corp_action_events` probes `len(ca)` + `ca.iterrows()`
  inside a `(TypeError, ValueError, AttributeError)` guard mirroring
  `_earnings_event_date` — logged fail-open ("unusable
  get_corporate_actions payload"), `add_event` left outside the
  swallow; inner date-parse except widened to include AttributeError;
  `row.get` hasattr-shielded.
- `engine/data_connector.py`: `list_option_expirations` gains
  keyword-only `max_staleness_days: int = 7` (default byte-identical);
  the explicit-as_of branch deliberately keeps NO staleness bound (a
  PIT backtest may see any snapshot ≤ as_of) — pinned as an asymmetry
  test, not "fixed", because bounding it would change dated backtest
  behavior and force a re-baseline.
- `TESTING.md`: data-drift-guards block extended — broad_pull pins +
  rail-neutralization polarity note + the post-#463 date-coherence
  story (#463 n3).

## Evidence

- **Replay 3-way A/B** (AAPL+MSFT, 2023-06-12→16, this box): main code
  + rail `ev_mean = −180.0275`; main code rail-off `−212.3225`; branch
  code with env DELIBERATELY pointed at the populated larder
  `−212.3225` — exact match to rail-off, and the contextmanager
  restored the hostile env value afterwards. The rail was live in
  main's replay lane; the committed baselines are now reproduced by
  construction on any box.
- **F4 file 21/21 green with the rail present** (was 20/21 locally;
  the $5.35 AAPL pin was the D4-2 live instance).
- **Mutation checks** (origin/main worktree + new test files): the 3
  corp-payload params crash main's ranker (`AttributeError`/`TypeError`
  escape everything); `test_list_expirations_staleness_parameterized`
  TypeErrors on main; `test_suite_runs_rail_neutralized` fails under
  main's conftest with the rail exposed. The well-formed-DataFrame
  control passes on BOTH sides (deliberate invariance pin).
- Targeted gate: 123 passed / 1 skipped across the six touched test
  files; fingerprint guards (`test_snapshot_data_fingerprint_matches_current`,
  `test_fingerprint_pins_every_connector_file`) green with the new
  keys; full fast suite + panel round in the PR body.

## Unresolved / handoff

- The regression snapshots' `generated_at`/`engine_sha_at_snapshot_lock`
  still describe the b3aa236 runs — correct (metrics are untouched)
  but worth knowing when reading fingerprint provenance: the two
  broad_pull sha entries were added 2026-07-02 without a re-run.
- The dividend_pit loader inside the connector is CWD-relative
  (`BroadPullLoader()` default) while the snapshot_bdp loader is
  data_dir-scoped — latent asymmetry if a driver ever runs from a
  non-repo-root CWD; flagged, not fixed (fixing moves served dividend
  yields = EV-moving, own lane).
- Rail-ON replay behavior now has zero coverage in the regression lane
  by design; the synthetic-parquet tests in
  `tests/test_real_premium_wiring.py` remain the rail-ON coverage. A
  future deliberate rail-ON regression lane needs its own baselines.
- **Verification-artifact drivers stay env-dependent** (panel residual):
  `docs/verification_artifacts/r10_strict_driver.py` and the R11
  dollar-impact driver build their own `WheelRunner` outside any pin,
  so their re-runs on a rail-bearing box still follow the shell env.
  (The 2026-06-28 skew study is fine — it hard-refuses when the env
  var is unset, rail-ON by explicit contract.)
- The four committed snapshots do NOT carry the new
  `option_premium_rail: "pinned_off"` fingerprint key (they predate
  the pin; retro-writing provenance would be dishonest — the rail was
  off at lock time because #435 postdates b3aa236, not because of a
  pin). Nothing compares the key; the next re-baseline adds it
  naturally.
- If the HELD #462 macro-calendar branch ever merges, its
  `macro_calendar` broad_pull read becomes a NEW unpinned connector
  input — extend `_BROAD_PULL_PINNED` in the same PR.
- #464 verdict nit v2 (overlay-served names show `estimate_eps=None`
  on advisory surfaces) remains backlog — cosmetic, memo-side.
