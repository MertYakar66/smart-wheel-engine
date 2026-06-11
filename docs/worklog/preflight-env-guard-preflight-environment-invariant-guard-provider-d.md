---
id: preflight-env-guard
title: Preflight environment-invariant guard (provider + data frontier)
kind: feature
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: New tests/test_preflight_environment.py — automates the two CLAUDE.md §4 session-start checks (provider is MarketDataConnector + logged; bundled OHLCV reaches the pinned frontier) so a stale tree / wrong clone / silent provider fails loud & diagnostically instead of poisoning a whole task
surface: [tests/test_preflight_environment.py]
---

## Goal

Turn this session's most expensive recurring friction into an automated guard:
the **stale-tree / wrong-provenance** class — reading `_common.py` /
`data_connector.py` / data from the stale primary clone instead of `main`,
which produced the W3 fingerprint false-positive and the "OHLCV 79 days stale"
premise — plus **silent provider selection** (CLAUDE.md §4.1). Codify the env
invariants I kept checking by hand into one fast, deterministic, self-skipping
preflight so every future task starts from verified ground.

## What we tried / what worked

`tests/test_preflight_environment.py`, two checks:
1. `test_default_provider_is_market_data_connector` — `type(WheelRunner().
   connector).__name__ == "MarketDataConnector"` for the default/bloomberg
   provider, and PRINTS the resolved provider (so it's in the run log either
   way). Skips when `SWE_DATA_PROVIDER=theta` (intentional non-default).
2. `test_bundled_ohlcv_reaches_expected_frontier` — the bundled OHLCV's max
   date `>= EXPECTED_FRONTIER` (pinned `2026-06-04`, bumped on each refresh).
   On a stale tree it fails LOUD + actionable: "OHLCV ends YYYY-MM-DD, expected
   >= … you may be on a STALE tree / wrong clone … verify main … or bump
   EXPECTED_FRONTIER". Skips when bundled data is absent.

Fast (reads only the OHLCV date column), deterministic (pinned constant, not
`today()`), self-skipping. Messages diagnose, not just fail.

## What didn't / deliberately excluded

- **Fingerprint/_FILES completeness** — already guarded by
  `test_data_integrity_bloomberg::test_fingerprint_pins_every_connector_file`;
  duplicating it would add noise, not value.
- **Trio-vs-origin/main provenance diff** — needs network and would false-fail
  on any legitimate decision-layer branch. The operator's own rule applies: a
  flaky preflight hinders, so it's out. (No trio touch; nothing risky → no
  stop-and-leave-a-note needed.)

## How we fixed it / Evidence

Additive only — one new test file, a FILE_MANIFEST row, this worklog. No trio
touch. `pytest tests/test_preflight_environment.py`: 2 passed in ~1s (provider
logged `MarketDataConnector`; frontier 2026-06-04 reached). ruff clean.

## Unresolved / handoff

`EXPECTED_FRONTIER` must be bumped in the same commit as each data refresh
(docs/DATA_POLICY.md §5) — that's the one maintenance touch. Held for operator
review + sign-off; no auto-merge.
