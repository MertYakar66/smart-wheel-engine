---
id: data-sufficiency-review
title: "Data-sufficiency review: pull vs wire vs blocked"
kind: research
status: done
terminal:
pr:
decisions: []
date: 2026-07-21
headline: "Source-verified answer to 'do we need more data?': mostly no — refresh + backfill 18 truncated names, produce the premium parquets, wire what's already on disk."
surface: [docs/DATA_SUFFICIENCY_REVIEW_2026-07-21.md]
---

## Goal

Answer the operator question "is the data we feed the engine enough — do we
need more data / more history / more datasets / wider ranges?" rigorously from
source (the docs drift), prioritized: what to pull and its ROI, what is a
wiring problem rather than a data problem, and the honest ceiling that pulling
cannot fix.

## What we tried

Multi-agent verification over source: 6 parallel tracers (connector accessor→
consumer map; ranker end-to-end EV-input trace; quant-module history-adequacy
constants; built-but-starved capability audit; dormant-dataset consumer sweep
over all 22 monoliths + 26 broad_pull panels; docs-drift audit), then 3
adversarial refuters attacking the "single per-strike-chain root cause"
headline from independent lenses, a completeness critic, and 6 gap-fillers on
the critic's open questions (tier-feasibility of a Bloomberg chain pull; who
produced the 2026-07-02 frontier; cause of the truncated-name histories;
R7-R10 liveness; FRED credit-multiplier reachability; the VIX-futures
tier-block contradiction).

## What worked

Reading everything from source with file:line receipts and adversarially
re-testing the inherited claim map. Several inherited claims flipped:
- "ranker premium wiring still needed" → wiring landed (#435/#463/#465);
  the gap is purely producing `data_processed/option_premium/` parquets.
- "more history might help the tail" → EVT is scenario-count-bound
  (gate 200 at ev_engine.py:505, effective convergence floor ~300); the 5y
  window is a hardcoded, non-overridable constant; even 32y ≈ 229 scenarios.
- "NS term structure needs a tenor pull" → `vol_term_rv.csv.gz` already
  carries atm_iv 30/60/90/180/365/730d, unconsumed.
- "total-return series is a missing pull" → refuted; raw price returns are
  the correct physical measure for terminal-price option P&L.
- "runbook §1 is the refresh procedure" → stale; #472 (14b21cb) shipped
  `_bbg_panel.py` env knobs (`SWE_PULL_MODE=forward SWE_PULL_END=…`) and a
  vol_iv producer; merge-only semantics, except `pull_treasury_yields.py`
  which full-overwrites.

## What didn't

- The inherited premise of a git→Drive migration keyed on
  `data/data_manifest.json` + `scripts/fetch_data.py`: neither exists anywhere
  in the repo or docs; the deep archive's durable store is the git buffer
  branch `deep-history/bloomberg-raw` (Drive = partial 2-of-12 mirror).
- Trusting `DATA_POLICY.md` §2 as the entitlement authority: the VIX-futures
  "tier-blocked on both providers" row is provably wrong (pulled twice via
  plain BDH; committed puller `staging/macro_vol/pull_vix_futures.py`).
  The tested census in `staging/BROAD_PULL_MANIFEST.md` is the authority.

## How we fixed it

Shipped `docs/DATA_SUFFICIENCY_REVIEW_2026-07-21.md`: the one-paragraph
verdict, per-method history adequacy (incl. the EVT overlapping-tier
inversion and the 18-truncated-names pull bug with WMT silently dropped),
EV-critical vs dormant classification of every on-disk dataset,
built-but-starved capabilities with their three distinct unblockers,
the ROI-ordered Terminal pull list, the tier-blocked ceiling, the
have-data-nothing-consumes-it table, the code-only ceiling (5y window, D21,
D19, R7 unfireable, credit-multiplier fragility, D18 news stub), and an
8-item priority queue.

## Evidence

- Workflow: 16 agents, ~1.87M subagent tokens, 526 tool uses; per-claim
  file:line evidence embedded throughout the review doc.
- Spot-verified in the main loop: EVT gate `ev_engine.py:505`;
  `_resolve_real_premium` wiring `wheel_runner.py:236-364,1920-1934`;
  `chain_df` probe `wheel_runner.py:2040-2047`; `skew_mult` block
  `wheel_runner.py:2115-2168`; monolith frontier 2026-07-02 via `tail` on
  `sp500_ohlcv.csv` / `sp500_vol_iv_full.csv`; `WMT UW` = 141 rows starting
  2025-12-09; `SWE_PULL_MODE`/`SWE_PULL_END` in `scripts/_bbg_panel.py`;
  frontier-bump commit 14b21cb (#472).

## Unresolved / handoff

- Live defects surfaced but NOT fixed here (each deserves its own card):
  (a) BK→BNY rename split makes the company invisible under both tickers;
  (b) R8-R10 held-book dict key mismatch ("ticker" vs "symbol"/"is_short")
  zeroes the held book's contribution on the cockpit flow; (c) R7 VaR gate
  structurally unfireable (no builder computes returns/correlation from
  on-disk OHLCV); (d) FRED `credit_regime()` TypeError on dated as_of when
  offline, masked to 1.0; regression locks are FRED-environment-dependent;
  (e) the EVT overlapping-tier inversion penalizes exactly the short-history
  names (GEV/SOLV/LIN/DASH/VLTO/COO band).
- The priority queue in the review doc §8 is the recommended execution order;
  items 4-7 are ceremony/re-baseline-tier code work, not pulls.
