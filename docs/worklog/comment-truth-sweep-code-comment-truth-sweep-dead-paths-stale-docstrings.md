---
id: comment-truth-sweep
title: "Code-comment truth sweep: dead doc paths + stale docstrings (item 5a)"
kind: docs
status: shipped
terminal: X
pr: 467
decisions: []
date: 2026-07-02
headline: "Nine dead END_TO_END_REVIEW paths repointed to archive/; four falsified docstrings (get_fundamentals 'no consumer threads as_of', produce_option_premiums 'premium is synthetic today', pull_theta_corp_actions 'fills the empty 2-byte CSV', bql_pulls '2 bytes') brought to truth; walk_forward wording airtightened"
surface:
  - engine/wheel_runner.py
  - engine/candidate_dossier.py
  - engine_api.py
  - engine/data_connector.py
---

## Goal

Campaign item 5a — the comment-only truth sweep deferred by the
2026-06-29 doc audit (its item 4) and item 4's docs pass (which stayed
out of code files by design). Comments lie to the next reader exactly
like docs do; these were all *confirmed-false* claims, not style.

## What we tried / worked

Recon re-enumerated every site at current line numbers (the audit's
had drifted). All edits comment/docstring-only — zero behavior change
by construction (no doctests in the repo; no test parses these
docstrings; `produce_option_premiums.__doc__` feeds argparse
description — rendering verified).

## What didn't

Nothing — mechanical once the sites were re-verified. One wording trap
avoided: only the PUTS ranker threads `as_of` into `get_fundamentals`
(#428); CC/strangle still call the snapshot form — the docstring fix
says exactly that instead of "the rankers thread as_of".

## How we fixed it

- 9 × `docs/END_TO_END_REVIEW_2026_05_25.md` →
  `archive/2026-05/END_TO_END_REVIEW_2026_05_25.md` (engine_api ×2,
  wheel_runner ×2, candidate_dossier ×2, three test module docstrings).
- `get_fundamentals` docstring: "no consumer threads ``as_of`` yet /
  not EV-moving until…" → the #428/#429 landed truth (puts ranker
  threads it; CC/strangle don't; fallback semantics kept).
- `produce_option_premiums.py`: "premium is synthetic today / ranker
  wiring intentionally NOT part of this rail" → past tense + #435
  wiring + #463 coherence + #465 neutralization; "pure additive rail
  with zero re-baseline" qualified to the current fingerprint story.
- `pull_theta_corp_actions.py`: "fills the empty 2-byte CSV" → the
  truth (script never writes that CSV; the 2.6 MB/52,442-row file came
  from the operator's BQL pull; Theta endpoints 404 → script dormant);
  `sp500_dividends_theta.csv` correctly described as consumer-less.
- `scripts/bloomberg_bql_pulls.md` §2 "2 bytes" → populated 2026-07.
- PRODUCTION_READINESS walk_forward phrasing (#466 verdict nit):
  "zero callers" → "never invoked — its only reference is the
  `backtests/__init__.py` re-export, which has no runner".

## Evidence

- Post-sweep grep: zero `docs/`-prefixed END_TO_END_REVIEW references
  outside worklog point-in-time records and archive/.
- ruff format/check clean on all 9 touched .py files; ast-parse +
  `--help` render verified for both scripts.
- Full fast suite green (counts in the PR body) — comment-only diff,
  trio touched → full-suite rule applied anyway.

## Unresolved / handoff

- `sp500_dividends_theta.csv` having zero consumers is now *documented*
  but the file still sits in `data/bloomberg/` — candidate for removal
  in a future data-hygiene pass (operator call; it's 'a manual-merge
  convenience').
