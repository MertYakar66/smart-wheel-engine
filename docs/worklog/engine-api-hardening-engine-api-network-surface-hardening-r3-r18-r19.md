---
id: engine-api-hardening
title: Engine API network-surface hardening (R3/R18/R19/R20/R21/R27/R43)
kind: fix
status: in-flight
terminal: swe-api
pr:
decisions: []
date: 2026-06-02
headline: Loopback bind by default, 400 on bad params, 404 on unknown tickers, no exception leak, and verdict-label parity — all NON-§2.
surface: [engine_api.py]
---

## Goal
Harden the network-facing HTTP layer (`engine_api.py`) without touching any
decision/EV semantics. The server bound `0.0.0.0` while the webhook auth
rationale assumed loopback and accepted all alerts when no secret was set, so
`POST /api/tv/webhook` and `POST /api/news/ingest` were unauthenticated writes
reachable from any LAN host. Several GET handlers leaked exception text,
returned 200 on error bodies, coerced params with bare `int()`/`float()`, and
fabricated a $100 spot for unknown tickers.

## What worked
Audit items landed against e1d7453:

- **R3** — new `_resolve_host` (env `SWE_API_HOST`, default `127.0.0.1`);
  `main()` binds the resolved host instead of hard-coded `0.0.0.0`. Webhook
  loopback comment reconciled to the new default. CORS `*` replaced with a
  localhost/loopback allow-list (`_resolve_cors_origin`) plus optional
  `SWE_API_CORS_ORIGIN`; the server-side dashboard proxy sends no Origin so
  it is unaffected.
- **R18** — `_parse_param` + `BadParam`; malformed int/float query params now
  return 400 (caught in `do_GET`) instead of falling through to the catch-all
  500. Applied to chart/iv_history/payoff/expected_move/strikes/tv_alerts/
  tv_ranked/tv_dossier/tv_dealer_positioning/tv_enrich/news.
- **R19** — `_send_internal_error` emits `{"error":"internal server error",
  "error_id":<uuid4 hex[:8]>}`, logs the full traceback + context server-side.
  Wired into both `do_GET`/`do_POST` catch-alls and the four handler 500s
  (candidates, dealer_positioning chain-fetch, dossier, tv_ranked).
- **R20** — payoff/expected_move/strikes now 404 on an unknown ticker
  (empty OHLCV) instead of computing against a fake $100 spot.
- **R21** — chart no-data → 404, iv_history no-IV → 404, fundamentals
  not-found → 404, strangle insufficient-data → 422. Bodies keep their
  existing `{error,...}` shape.
- **R27 (§2-adjacent, label-only)** — `_enrich_alert` webhook ladder now emits
  verdict `"blocked"` (was `"skip"`) for negative + non-finite EV, matching
  the dossier reviewer R1/R1a. `verdict_reason` strings unchanged
  (`negative_ev` / `ev_non_finite`). No behaviour/EV-math change — both labels
  are non-tradeable and neither rescues a candidate, so §2 holds.
- **R43** — startup banner appends the previously-omitted ranked/dossier/
  dealer_positioning/news/news_ingest endpoints.

## How we fixed it
All changes are confined to `engine_api.py` (robustness/security/error
handling + one verdict label). The decision trio (`ev_engine.py`,
`wheel_runner.py`, `candidate_dossier.py`) was not touched. The dealer
clamp `[0.70,1.05]` and the downgrade-only reviewer contract are untouched.

## Evidence
- `pytest tests/test_tv_api.py tests/test_engine_api_hardening.py -q` → 42 passed.
- `pytest tests/ -k "api or webhook or engine_api or tv" -q` → 308 passed, 0 failed.
- `ruff check` + `ruff format --check` on `engine_api.py` + both test files → clean.
- `scripts/check_manifest_coverage.py` → OK (0 uncovered).
- New tests: `tests/test_engine_api_hardening.py` (R3/R18/R19/R20/R21/R27 +
  unknown-path/oversized-body/invalid-JSON routing).

## Unresolved / handoff
- R27 is the only §2-adjacent change — flagged for review. It is a pure
  audit-trail label reconciliation; verify the `blocked` label is what the
  dashboard alert view expects (it already renders dossier `blocked`).
- An operator who sets `SWE_API_HOST=0.0.0.0` MUST also set a webhook secret;
  the webhook docstring now says so but this is not enforced in code.
