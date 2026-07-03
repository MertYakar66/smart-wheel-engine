---
id: doc-truth-pass
title: "Deployment-truth doc pass: honest headline, Blocker-2 recast, R-number collision, rail/data-doc currency"
kind: docs
status: shipped
terminal: X
pr: 466
decisions: []
date: 2026-07-02
headline: "PRODUCTION_READINESS now tells the honest evidence story (overlap-inflated rho caveat, I1 -0.002, no-true-OOS) and Blocker-2 no longer contradicts itself; the news-redesign R9 collision, the rail docs' missing #463 frontier-coherence caveat, and the deferred data-doc count deltas are all reconciled"
surface:
  - docs/PRODUCTION_READINESS.md
  - docs/NEWS_REDESIGN_CAMPAIGN.md
  - docs/WIRING_CAMPAIGN.md
  - docs/PHASE2_SKEW_EXECUTION_SPEC.md
  - PROJECT_STATE.md
  - docs/DATA_POLICY.md
---

## Goal

Remediation-campaign item 4 — the deployment-truth doc pass. The
adversarial review's doc-class findings plus drift accumulated across
items 1-3 and the 2026-06-29 doc audit's deferred deltas. Docs-only by
design: **no trio files, no code** (the code-comment truth sweep —
dead END_TO_END_REVIEW paths in trio comments, the stale
`get_fundamentals`/`produce_option_premiums`/`pull_theta_corp_actions`
docstrings — needs a lane-claim and is deliberately left to its own
non-docs PR, per the 2026-06-29 audit's own design note).

## What we tried

4-lens recon workflow (PRODUCTION_READINESS / rail docs / data docs /
CLAUDE.md oracle), every claimed defect verified against main @
3e251fb with verbatim quotes before any edit; the CLAUDE.md smoke was
re-run live as evidence.

## What worked

Every campaign-claimed defect CONFIRMED on main, with two corrections
to the claims themselves (see below). All fixes applied.

## What didn't

- **The deferred audit's "10/4" producer count is wrong.** The "4th
  producer" (`scripts/pull_theta_corp_actions.py`) never writes
  `sp500_corporate_actions.csv` — it writes parquet side-files +
  `sp500_dividends_theta.csv`, and Theta's corp-actions endpoints 404
  at this tier. The real file came from the operator's manual BQL pull
  (`scripts/bloomberg_bql_pulls.md` §2). Honest count: **10 files / 3
  reproducible producers / 7 without** — shipped that, not 10/4.
- **Rewriting D9's "28/503" in place.** It was TRUE at decision time
  (exactly 28 names carry the 2026-04-23 pilot snapshot; the 05-24 /
  06-01 waves grew the dir to ~502/503). D-entries are historical
  records — shipped a dated coverage annotation instead.
- **The I1 contradiction is CROSS-doc, not intra-doc** (the review's
  phrasing implied the −0.002 lived in PRODUCTION_READINESS; it lives
  in HEAVY_VERIFY_2026-05-31_I1_CALIBRATION.md, which the headline
  simply never acknowledged).

## How we fixed it

- **PRODUCTION_READINESS.md:** headline "EV-correct outputs" cell
  rewritten — selector-vs-dollar-forecaster split, overlap-inflation
  caveat on every pooled p-value, I1 −0.002 quoted, I6/W7 named as the
  honest evidence base with their in-sample status, no-true-OOS
  disclosure (`walk_forward.py` zero callers, C1 never built). §3
  Blocker-2 recast to SHIPPED (PR #233 + #255) with the gap text
  preserved as historical record — previously the header + body said
  "not wired / hookup is open" while §1, the §3 intro, and §6 all said
  shipped (code says shipped: `_build_portfolio_context_from_params`
  et al.). §3 intro B1 rollback/shipped contradiction smoothed;
  PR #256 → #262. Stale test-count row refreshed.
- **NEWS_REDESIGN_CAMPAIGN.md:** the quality reviewer no longer
  hard-claims **R9** (taken by the D17 sector cap; R10/R11 also gone)
  — now "next free R-number at implementation time, R12 as of
  2026-07-02" with an explicit don't-hard-claim note; all 8 R9
  mentions renamed; the pause-point's "yfinance snapshot" event-gate
  claim corrected (base file ∪ #464 snapshot_bdp overlay); PR 3.5's
  incumbent-changed note added. `HEAVY_NEWS_CALIBRATION_REVERIFY.md`
  (frozen record) got bracketed editor's notes instead of rewording.
- **WIRING_CAMPAIGN.md + PHASE2_SKEW_EXECUTION_SPEC.md:** "live where
  the rail is present" now carries the #463 frontier-coherence caveat
  (exact spot-bar date match, DTE ±10d, as_of=None → spot bar,
  no-anchor refusal, D1-1 4-18× story); "gitignored ⇒ CI/regression
  never see it" corrected to the #465 active-neutralization truth
  (gitignored-alone was NOT sufficient — D4-2); the pre-#463 "90-100%
  usage" validation figures qualified (per-covered-name, ~25%
  universe-wide per the 2026-06-27 rail audit); 3G dividend_pit row
  updated (carry-q half LANDED #426/#428/#429, fingerprint-pinned
  #465; W2 xfail correctly still stands for the broader dated-selection
  contract); PHASE2's drifted line-number references into
  WIRING_CAMPAIGN de-line-numbered.
- **Data docs:** DECISIONS.md D9 dated coverage annotation;
  PROJECT_STATE.md §1 currency (frontier 2026-06-04 + the
  puller-would-REGRESS warning; 10/3/7 counts) + fingerprint note
  extended for #465 + Last-updated bump; DATA_POLICY.md §5 counts
  (10/3/7 + corporate_actions near-miss parenthetical + broad_pull
  panels); bloomberg_refresh_runbook.md got a currency-note header
  (dated snapshot — not silently rewritten);
  BLOOMBERG_TERMINAL_NEXT_SESSION.md's "remaining = ranker wiring"
  superseded-note (#435/#463/#465). MODULE_INDEX.md data_connector row
  now names the overlay + rail accessors. EDGAR_EARNINGS.md's wrong
  source-file claim fixed + #464 update note.
- **Worklog loop-closing:** #464 fragment's item-3 handoff marked
  CLOSED; #463 fragment's D4-2/D5a-1 handoff marked shipped-as-#465
  and its doc-reconciliation item discharged by this pass.

## Evidence

- CLAUDE.md §4 smoke re-run live 2026-07-02 at `as_of=None`: 0 rows,
  `drops_summary = {'total_dropped': 5, 'by_gate': {'event': 5}}`
  (AAPL 07-31 / MSFT 07-30 / JPM 07-14 / XOM 07-31 / UNH 07-16) — the
  documented "five rows = healthy" oracle reads as failure on a
  correctly-behaving engine.
- iv_surface dir disk-verified: 558 files, 502 distinct tickers, 28
  carry the 2026-04-23 pilot snapshot.
- OHLCV frontier verified: last row 2026-06-04 (= `EXPECTED_FRONTIER`).
- `walk_forward.py` zero callers re-verified by repo-wide grep.
- Every OLD text block quoted verbatim by recon and matched exactly at
  edit time (Edit-tool exact-match is the mechanical check).

## Unresolved / handoff

- **CLAUDE.md §4 oracle — PROPOSAL for the operator** (CLAUDE.md is
  operator-maintained; NOT applied). Replace the two-line oracle
  sentence after the §4 step-2 command block ("Runs in ~2 s. Five rows
  with non-null `ev_dollars`, `iv`, and `premium` means the Bloomberg
  CSVs + connector + EV engine path is healthy.") with:

  > Runs in ~2 s. Healthy = rows + structured drops account for all
  > five tickers: every returned row has non-null `ev_dollars`, `iv`,
  > and `premium`, and every missing ticker appears in
  > `df.attrs["drops"]` with a real gate reason — check
  > `df.attrs["drops_summary"]`, e.g.
  > `{'total_dropped': 5, 'by_gate': {'event': 5}}`. Around earnings
  > season 0 rows + 5 `event` drops
  > (`event_lockout:earnings@... (±5d buffer)`) is the CORRECT answer
  > (#464 restored the live lockout); 0 rows with an empty `drops`
  > is what signals a broken data path.

- Code-comment truth sweep (own PR, needs trio lane-claim): dead
  `docs/END_TO_END_REVIEW_2026_05_25.md` paths in
  candidate_dossier/wheel_runner/engine_api + 3 test docstrings;
  `get_fundamentals` docstring's false "no consumer threads as_of yet";
  `produce_option_premiums.py` "premium is synthetic today" +
  `pull_theta_corp_actions.py` "fills the empty CSV" docstrings.
- PROJECT_STATE §1's contested producer-count wording now says 3-of-10
  with the corporate_actions footnote; if the operator recovers the
  original BQL queries the count changes — DATA_POLICY §5 is the
  canonical place to update first.
