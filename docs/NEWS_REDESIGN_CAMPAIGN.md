# News-architecture redesign campaign

Tracking doc for the 9-PR campaign that severs verbal news from the EV
decision path and replaces it with structured quantitative layers
(earnings calendar, fundamentals, macro). Branch prefix:
`claude/lucid-davinci-pm15H`. Coordination on board #113.

This is a **temporal** doc — update the status table as each PR lands.
The structural decisions taken inside the campaign live in
`DECISIONS.md` (D18 onwards — note D18 lands with PR #249 and is not yet
merged into `DECISIONS.md` on `main`).

---

## 1. Campaign goal

Today, three "news" sources flow into the engine:

1. `engine/news_sentiment.py` — VADER + 50-word lexicon over headlines
   → multiplier in `[0.88, 1.05]` stacked into `combined_regime_mult`
   in `wheel_runner.py`. **On the EV path.**
2. `news_pipeline/` — 29-file browser-agent editorial pipeline. Feeds
   the dashboard. Not on the EV path.
3. `financial_news/` — 33-file macro/event platform. Not on the EV
   path.

The campaign reframes this around **verbal vs numbered** news:

- **Verbal news** (qualitative narrative — "China blocks Nvidia
  chips") → operator layer only; engine ignores. This is the
  severance.
- **Numbered news**
  - **Price / IV** — already wired (Bloomberg / Theta).
  - **Fundamentals** — NEW: continuous quality score `[0, 1]` per
    ticker, feeds a NEW reviewer rule (R9, since R7+R8 are taken by
    D17). Threshold is regime-aware (tightens in stress, relaxes in
    constructive).
  - **Macro** — NEW: FRED rewrites the existing `credit_mult`.

Plus operator-side UX improvements (EV percentile spread instead of
point estimate; portfolio pane; "no-signal" pane in morning brief).

The §2 invariant (CLAUDE.md `EVEngine.evaluate` authority) holds across
every PR — new inputs land as downgrade-only reviewers or as inputs
that feed `EVEngine` itself, never as paths that rescue a negative-EV
candidate.

---

## 2. PR sequence (status table)

Order optimised for risk: lowest blast-radius warm-up first, §2-surface
PRs surfaced for Session peer review per `docs/PARALLEL_SESSIONS.md`
rule 4, infrastructure (EDGAR + FRED) before the consumer (R9).

| # | Scope | Branch | §2? | Status | PR | Decision |
|---|---|---|---|---|---|---|
| 1 | EV percentile spread (p25/p50/p75) on `EVResult`, ranker row, `/api/candidates` | `claude/lucid-davinci-pm15H-pct-spread` | yes (additive on `EVResult`) | **open** | [#248](https://github.com/MertYakar66/smart-wheel-engine/pull/248) | — |
| 2 | Sever verbal news from EV path; stub `sentiment_multiplier → 1.0`; D18 | `claude/lucid-davinci-pm15H-sever-news` | yes (drops `news_mult` from `combined_regime_mult`) | **open** | [#249](https://github.com/MertYakar66/smart-wheel-engine/pull/249) | D18 |
| 3 | EDGAR earnings history + projection (data layer; integration deferred) | `claude/lucid-davinci-pm15H-edgar` | no (data layer only) | **open** | [#251](https://github.com/MertYakar66/smart-wheel-engine/pull/251) | — |
| 3.5 | Wire `EDGARAdapter.project_next_earnings` into `MarketDataConnector.get_next_earnings` | `claude/lucid-davinci-pm15H-edgar-wire` | likely yes (touches `wheel_runner.py` consumption) | not started | — | TBD |
| 4 | Quality score computation (sector-relative z-scores from EDGAR XBRL) | `claude/lucid-davinci-pm15H-quality` | no | not started | — | — |
| 5 | R9 reviewer rule (regime-aware threshold, abstain on missing) | `claude/lucid-davinci-pm15H-r9` | **yes** (new reviewer in `EnginePhaseReviewer`) | not started | — | TBD |
| 6 | FRED → `credit_mult` rewrite | `claude/lucid-davinci-pm15H-fred` | partial (changes a multiplier input source) | not started | — | TBD |
| 7 | Backtest re-baseline + new `S<N>` ledger entry | `claude/lucid-davinci-pm15H-s-rebase` | no | not started | — | — |
| 8 | Dashboard: candidates pane + portfolio pane + "no-signal" pane | `claude/lucid-davinci-pm15H-dashboard` | no | not started | — | — |
| 9 | Override-accuracy log (schema + first month's data) | `claude/lucid-davinci-pm15H-override-log` | no | not started | — | — |
| meta | This tracking doc | `claude/lucid-davinci-pm15H-tracker` | no | **open** | [#250](https://github.com/MertYakar66/smart-wheel-engine/pull/250) | — |
| meta | Descriptive-doc alignment sweep (`MODULE_INDEX`, `PROJECT_STATE`, `README`, `AGENTS`, `FILE_MANIFEST`, `ROADMAP`, `CHANGELOG`, `pull_news_sentiment.py` docstring) for D18 + EDGAR | `claude/lucid-davinci-pm15H-docs-align` | no | **open** | [#252](https://github.com/MertYakar66/smart-wheel-engine/pull/252) | — |

PR 3.5 was inserted after PR 3 was implemented as a data-layer-only PR. The integration step (wiring EDGAR into the existing `conn.get_next_earnings` consumption pattern) deserves its own design decision (replace yfinance / preferred-with-fallback / surface-both) — see `docs/EDGAR_EARNINGS.md` §6 for the three integration shapes.

---

## 3. Non-obvious facts a future agent needs

### R-number reservation
The earlier design review on board #113 said the quality reviewer would
be **R7**. That was stale: D17 (merged 2026-05-26) already took **R7
and R8** for portfolio-risk soft-warn reviewers. The quality reviewer
in PR 5 must be **R9**. See `engine/candidate_dossier.py` for the
current R1–R8 surface; `tests/test_dossier_invariant.py::TestD17DossierSoftWarns`
pins R7+R8.

### `combined_regime_mult` is a misnamed product
`engine/wheel_runner.py` line 1180 computes
`combined_regime_mult = hmm_regime_mult × skew_mult × news_mult × credit_mult`
then passes it as `trade.regime_multiplier` into `EVEngine.evaluate`.
The dealer multiplier scales separately inside `EVEngine`. That's
**5 multipliers**, not 3. After PR 2 (D18), `news_mult` is always 1.0
so the product is meaningfully a regime × skew × credit composition;
the variable name still says "regime" but is a small naming lie.

### FMP rejected
The user opted out of paid Financial Modeling Prep. Fundamentals come
from **SEC EDGAR (10-Q XBRL + 8-K)** and **FRED** only. The architecture
supports a `FundamentalsProvider` protocol so a paid vendor could be
plugged in later; nothing in v1 depends on one.

### Quality score N/A handling
For tickers where fundamentals are unavailable (recent IPOs, SPACs,
foreign ADRs, data gaps), the quality score is **N/A** — not a default.
R9 **abstains** when the score is N/A (no downgrade), symmetric with
how R6 (dealer) abstains when `market_structure=None`. Anything else
creates a hidden gate that selects against new/rotated names without an
audit trail.

The user's stated commitment: "I will make sure to fill all in" — the
operator is the source of truth for missing fundamentals; the engine
surfaces the gap, the operator closes it.

### D3 SUPERSEDED marker is missing
The DECISIONS.md format says to mark earlier entries with
`**SUPERSEDED by D<N>**` when overridden. PR 2 (D18) supersedes D3's
"news_sentiment is the only news module on the EV path" clause, but
the auto-mode classifier blocked editing D3's existing block (it's an
append-only doc and the entry was originally authored by another
terminal). D18 carries an inline back-reference to D3 in lieu of the
marker. **A user-driven manual edit may want to add the SUPERSEDED tag
to D3 to match the documented convention.**

### Backtest re-baseline is mandatory
Severing news + adding R9 + extending the event gate via EDGAR
compounds the existing IV-PIT backtest invalidation (memory
`ranker-iv-was-not-pit`, fix `d26a8d6`). Every prior `S<N>` scenario
result becomes obsolete after this campaign. PR 7 is the re-baseline.
**Do not infer "engine performance" from any pre-campaign scenario
once PRs 1–6 have landed.**

### `gh pr create` cwd gotcha
The Claude Code harness keeps resetting cwd to the primary checkout
between bash calls. `gh pr create` without `--head` infers the branch
from the current shell's git state — which is the primary's HEAD, not
the worktree's. PR #246 was opened with the wrong branch attached for
exactly this reason (its title/body described the pct-spread work; the
actual diff was the prior TradingView Windows setup docs from the
primary's branch). PR #246's metadata was retroactively corrected to
match its real diff. **For all future PRs from a terminal worktree,
either `cd <worktree>` immediately before `gh pr create` AND pass
`--head <branch>` explicitly, or run `gh -C <worktree> pr create` —
not all gh subcommands support `-C` so the safer pattern is `cd` +
`--head`.**

### Quartile fields are pre-multiplier (PR 1)
`pnl_p25 / pnl_p50 / pnl_p75` on `EVResult` are raw distribution
percentiles, *not* scaled by `regime_multiplier` or `dealer_multiplier`.
A consumer who wants the scaled spread can multiply explicitly.
`test_percentiles_are_pre_multiplier` in
`tests/test_ev_engine_percentiles.py` pins this invariant.

---

## 4. PR review flow

§2-surface PRs (PR 1, 2, 5 currently; possibly 6) flag for **Session
peer review by a Session other than D's paired Session** per
`docs/PARALLEL_SESSIONS.md` rule 4. The board #113 claim comment for
each §2-surface PR explicitly names this requirement so Sessions know
to pick it up before the user merges.

Non-§2-surface PRs (3, 4, 7, 8, 9, this tracking doc) follow the
standard one-review path.

---

## 5. What changes if the campaign needs to pause

If PR 1 / 2 ship and the rest pauses (e.g. another priority lands on
the user's plate), the engine is in a **safe intermediate state**:

- Verbal news has no EV influence (D18). ✓
- Operator transparency preserved (sentiment surfaces in row dict). ✓
- Event gate continues to use the existing `conn.get_next_earnings`
  (yfinance snapshot via Bloomberg connector). EDGAR upgrade deferred.
- `credit_mult` continues to use the existing logic in
  `wheel_runner.py` (Bloomberg-based). FRED rewrite deferred.
- Quality score is unused (PR 4/5 deferred). No quality data, no R9.
- Backtest results from PRs 1+2 reflect "engine without verbal news";
  PR 7 re-baseline produces the headline performance number for the
  fully-redesigned engine.

The campaign **can** pause cleanly after PR 2. It **cannot** pause
between PR 4 (quality score computation) and PR 5 (R9 consumer) without
leaving a dead-data artifact — those two should land in close
sequence.

---

## 6. References

- **CLAUDE.md** §2 — the no-bypass-`EVEngine.evaluate` invariant
- **DECISIONS.md** D1, D3, D17, D18 — directly relevant
- **MODULE_INDEX.md** — `engine/news_sentiment.py`, `engine/event_gate.py`,
  `engine/candidate_dossier.py`
- **docs/PARALLEL_SESSIONS.md** rule 4 — §2-surface peer review
- **docs/LAUNCH_READINESS.md** — launch-blocker test floor
- **board #113** — live coordination + claims log
