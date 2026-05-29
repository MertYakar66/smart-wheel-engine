---
id: S12
title: TradingView webhook ingest
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the Pine-signal entry path end-to-end —
`POST /api/tv/webhook` → ring buffer → read endpoints — and answer the
§2 question: can a webhook alert produce a tradeable verdict that
bypasses `EVEngine.evaluate`?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`. `engine_api.py` started on
`:8787` (stdlib `ThreadingHTTPServer`); a probe script drove the
webhook over HTTP with synthetic Pine payloads — valid, malformed,
bad-ticker, replay, and a 210-alert overflow burst — plus the read
endpoints, run once with no webhook secret and once with
`TV_WEBHOOK_SECRET` set. No code changes.

**Path.** `POST /api/tv/webhook` → `TVAlert.parse` + `is_valid()` →
optional HMAC / shared-secret auth → timestamp-freshness + replay
guard → `_enrich_alert` → append to the in-memory ring buffer
`_TV_ALERT_LOG`. `_enrich_alert` computes the alert's verdict by
running `WheelRunner.rank_candidates_by_ev([ticker])` →
`EVEngine.evaluate` (`engine_api.py:2055`) — the verdict is
EV-authoritative (`authority="ev_ranked"`).

**Status.** Done. §2 holds — a webhook alert cannot reach a tradeable
verdict without `EVEngine.evaluate`, and a Pine signal can only
downgrade. No §2 violation; no bug. All findings logged.

**Findings:**

- **§2 holds — a Pine signal cannot rescue a non-tradeable
  candidate.** A webhook alert for AAPL (`ev_dollars = −95.47`) with a
  bullish `wheel_put_zone` signal returned `verdict="skip"`,
  `reason="negative_ev"` — the bullish Pine signal did not flip it
  tradeable. The Pine signal is **downgrade-only**: `pine_agrees` (the
  engine's own recomputed TA agreeing with the claimed signal) is
  *required* for `"proceed"`, so MSFT (ev +13.2, prob 0.78) and XOM
  (ev +171, prob 0.96) — both clearing the EV bar — were downgraded to
  `"review"` (`chart_disagrees`) because the engine's TA did not
  agree. A signal moves proceed→review, never the reverse. **Logged.**

- **A prior §2 leak existed here and is closed.** `engine_api.py:2028`
  carries an audit-fix comment (2026-04-14): the webhook verdict
  previously used a `wheel_score >= 60` heuristic, producing
  `"proceed"` verdicts in the ring buffer "never validated against
  EV … a silent authority leak." It now runs the EV ranker and uses
  `ev_dollars` / `prob_profit` as the authority; `wheel_score` is
  supplementary-only. S12 confirms the fix holds — every probed
  verdict carried `authority="ev_ranked"`, and EV-unreachable falls
  back to `"review"`, never `"proceed"`. **Logged.**

- **Ingest and the EV-ranked read paths are decoupled.**
  `/api/tv/alerts` serves the ring buffer; `/api/tv/ranked`
  (`rank_candidates_by_ev`) and `/api/tv/dossier`
  (`build_candidate_dossiers`) are independent EV-ranking endpoints
  that never read `_TV_ALERT_LOG`. A webhook alert influences only its
  own stored enriched verdict — it does not enter, reorder, or bias
  the EV ranking. **Logged.**

- **Ring buffer — capacity 200, FIFO, in-memory.** `_TV_ALERT_LOG`,
  `_TV_ALERT_LOG_MAX = 200`; on overflow `del _TV_ALERT_LOG[0:len−MAX]`
  (`engine_api.py:1683`) drops the oldest. Probed: 210 distinct alerts
  POSTed → `/api/tv/alerts` returned exactly 200, the first 10
  (`ZZ0001`–`ZZ0010`) evicted, newest-first ordering. The buffer is
  in-memory only — rebuilt empty on every server restart, no
  persistence. **Logged.**

- **Validation is solid; a bad ticker is soft-rejected.** Missing
  ticker or signal → 400; invalid JSON → 400; non-object JSON → 400;
  body > 16 KB → 413; unknown POST path → 404; a duplicate body within
  300 s → 409 (replay guard, confirmed). But an unknown ticker
  (`ZZZZ`) returns **HTTP 200** with `enriched.accepted = false`,
  `reason = "ticker_not_in_universe"` — soft-rejected: acked and
  *stored in the ring buffer* with an `accepted:false` flag rather
  than an HTTP error. A Pine caller cannot tell "ingested + enriched"
  from "ingested but un-enrichable" by status code alone. **Logged.**

- **The webhook is unauthenticated by default — auth is opt-in.** With
  neither `TV_WEBHOOK_HMAC_SECRET` nor `TV_WEBHOOK_SECRET` set the
  handler accepts every POST (intended for a loopback-only deployment,
  per the handler docstring). With `TV_WEBHOOK_SECRET` set the in-body
  secret is enforced by constant-time compare — probed: no secret →
  401, wrong secret → 401, correct → 200. Safe on loopback; were the
  API ever bound beyond localhost without a secret set, the webhook
  would accept arbitrary alerts. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — see S20 re-verify for the full network-surface §2 re-test. Inline verdict-producing logic in `_enrich_alert` (`engine_api.py:2009+`) still routes the EV through `runner.rank_candidates_by_ev(...).iloc[0]["ev_dollars"]`; payload `ev_dollars` is ignored.
  - Qualitative verdict: match — `_enrich_alert` is a method on the API handler (originally documented as a top-level function in S12 — that's a doc nuance, not a behavior change). Ring buffer `_TV_ALERT_LOG_MAX = 200`, `_tv_verify_hmac` constant-time compare, `_TV_SEEN_NONCES` OrderedDict — all present unchanged.
  - Numerical drift > 5%: not applicable.
  - Notes: full webhook concurrency / ring-trim / nonce-replay / HMAC-under-load re-verified in S20 below — all 5 race vectors clean.
