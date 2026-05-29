# Audit-of-the-audit — Terminal A audit PR #173 (2026-05-25)

**Reviewer:** Terminal B, fresh session, no campaign context.
**Scope:** one PR, **#173** (`docs(audit): independent re-audit of
Terminal A PRs through 2026-05-24`), branch
`claude/audit-terminal-a-output`, head `1cb2c52` (which merged
current `main` into the audit branch on 2026-05-25; the original
audit commit was `8d2dd4c`).
**`origin/main` SHA at review start:** `cc54619c9527204e6f07e83a37e5621f5c285624`.
**Audit-under-review's snapshot SHA:** `86b917c7ab4fe4905dbe68e38c8027c8a81b1374`.

PR #173 is **open** at review start. The audit doc it carries
(`docs/TERMINAL_A_AUDIT.md`) verifies 22 Terminal A PRs at the
snapshot above with a tally of `SOLID 22 / WITH-NOTE 0 / CONCERN 0 /
§2 BREACH 0`. This review meta-verifies seven load-bearing claims
of the audit itself.

## Tally

- VERIFIED:           **6** (M1, M2, M3, M5, M6, M7)
- VERIFIED-WITH-NOTE: **1** (M4 — attribution imprecision on the +117 wheel_runner.py drift)
- OVER-CLAIMED:       **0**
- UNDER-CLAIMED:      **0**
- §2 BREACH MISSED:   **0**

The audit holds up. One minor attribution discrepancy in the source-
line-drift section, no consequential overreach, no missed §2 surface.

---

## M1 — Headline tally

**Verdict:** **VERIFIED.**

- `grep -c "^### PR #" docs/TERMINAL_A_AUDIT.md` → **22**.
- `grep -c "Verdict:\*\* SOLID$\|Verdict:\*\* SOLID \*" docs/TERMINAL_A_AUDIT.md` → **22**.
- Verdict lines enumerated: 19 plain `SOLID`, 3 `SOLID *(promoted from
  SOLID-WITH-NOTE in the prior audit)*` (#159 / #163 / #165).

22 entries, 22 SOLID verdicts, zero WITH-NOTE / CONCERN / §2 BREACH.
Tally matches the per-PR audits exactly.

---

## M2 — D17 promotion logic for #159 / #163 / #165

**Verdict:** **VERIFIED.**

The audit promotes #159 (gate library), #163 (tracker hard-blocks),
and #165 (dossier R7+R8 soft-warns) from the prior audit's
SOLID-WITH-NOTE to SOLID on the basis that PR #169 landed the D17
entry in `DECISIONS.md`. Verified at the code level:

- `DECISIONS.md:735` — `## D17. Portfolio-level risk gates are wired on both surfaces — hard-block on entry, soft-warn on review`.
- **Two-surface design.** Body lines explicitly enumerate:
  1. Tracker hard-blocks via `engine/wheel_tracker.py._evaluate_d17_hard_blocks`
     (sector cap, portfolio delta, Kelly per-trade NAV cap; live NAV
     via `_compute_live_nav`; `nav_exhausted` pre-gate at
     `min_nav_for_trading`).
  2. Dossier soft-warns via `engine/candidate_dossier.py`
     `EnginePhaseReviewer` R7 (VaR > 5% NAV → review) + R8 (one
     rule, two triggers: stress drawdown OR short-gamma regime).
- **Six locked defaults table:**
  | Gate | Default | Module constant |
  |---|---|---|
  | Sector cap | 25% NAV | `_DEFAULT_MAX_SECTOR_PCT = 0.25` |
  | Portfolio delta | ±$300 / $100k NAV | `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` |
  | Kelly fraction | 50% (half-Kelly) | `_DEFAULT_KELLY_FRACTION = 0.5` |
  | VaR ceiling | 5% NAV / 30d / 95% | `_DEFAULT_MAX_VAR_PCT = 0.05` |
  | Stress drawdown | 8% NAV | `_DEFAULT_MAX_STRESS_DRAWDOWN_PCT = 0.08` |
  | C4 vol-spike | −10% spot + 30% IV | `_C4_VOL_SPIKE_SCENARIO` |
- **Four new tracker reject audit-log shapes** explicitly listed:
  `nav_exhausted`, `sector_cap_breach`, `portfolio_delta_breach`,
  `kelly_size_exceeded` — each carrying `nav` + `nav_source`
  fingerprint.
- **R1 primacy restated:** *"R1 (`negative EV → blocked`) still wins
  over every D17 surface — the hard invariant from CLAUDE.md §2 / D1
  / D16 is not amended."*

The D17 entry's content matches the shipped behaviour of #159 (the
pure-function library at `engine/portfolio_risk_gates.py`), #163 (the
tracker hard-block wiring at `engine/wheel_tracker.py`), and #165
(the dossier R7+R8 at `engine/candidate_dossier.py`). The promotion
is justified.

---

## M3 — Second-layer drift discipline (PR #145 / D16)

**Verdict:** **VERIFIED.**

Picked the highest-stakes audited PR (the verdict-bound D16 token,
which the audit calls "the highest-stakes §2-adjacent surface in the
campaign") and traced every cited line at the audit's snapshot SHA
`86b917c7`:

| Symbol | Audit cite | At `86b917c7` | At `cc54619` (today) |
|---|---|---|---|
| `class EVAuthorityRefused` | `:28` | **`:28`** ✓ | `:28` ✓ |
| `def issue_ev_authority_token` | `:322` | **`:322`** ✓ | `:322` ✓ |
| issuance refusal block (`if canonical["ev_dollars"] <= 0:`) | `:357-369` | **`:357-369`** ✓ | `:357-369` ✓ |
| `def _consume_ev_authority_token` | `:375-436` | **`:375-436`** ✓ | `:375-436` ✓ |
| `def open_short_put` | `:438` | **`:438`** ✓ | `:438` ✓ |
| `def open_covered_call` | `:1061` | **`:1061`** ✓ | `:1143` (post-audit drift) |

All six citations match the audit's snapshot exactly. The single
post-snapshot drift on `open_covered_call` (1061 → 1143, +82 lines)
is attributable to **PR #174** (`94cb254`, merged 2026-05-25), which
landed *after* the audit was written. The audit's drift-summary
section (Cross-cutting Observation #3) names #161 and #172 as the
two known post-Terminal-A line shifters at audit-start time;
PR #174 did not exist then and is correctly absent from the audit's
drift summary.

The audit's stated methodology — "When a PR claims 'fixes X at
file.py:N', read file.py:N today AND the helper/adapter it calls" —
holds for #145. The issuance refusal helper (`if canonical[
"ev_dollars"] <= 0:` at `:357`) is read alongside the function
definition (`def issue_ev_authority_token` at `:322`); the
`_consume_ev_authority_token` helper (`:375-436`) is read alongside
its two call sites (`open_short_put` at `:438`, `open_covered_call`
at `:1061`). Second-layer drift discipline is observed.

---

## M4 — The two drift claims (#161 wheel_runner.py + #172 portfolio_risk_gates.py)

**Verdict:** **VERIFIED-WITH-NOTE.**

### #172 — portfolio_risk_gates.py (+19 lines)

- `git show 86b917c --stat engine/portfolio_risk_gates.py` →
  `1 file changed, 19 insertions(+)`. ✓
- All three function offsets match the audit exactly today on `cc54619`:
  - `check_var` → **`:529`** (audit: 510 → 529) ✓
  - `check_stress_scenario` → **`:630`** (audit: 611 → 630) ✓
  - `check_dealer_regime` → **`:724`** (audit: 705 → 724) ✓

Function bodies unchanged at the new offsets; the +19 is entirely
inside `check_kelly_size`'s docstring (the "Current-path reachability
(#166 B3)" paragraph). **Clean drift claim.**

### #161 — wheel_runner.py (+95 line stat ≠ audit's +117 attribution)

- `git show 27dae4f --stat engine/wheel_runner.py` →
  `1 file changed, 95 insertions(+)`. **The audit's body text says
  the same thing** ("PR #161 ... inserted 95 LoC"), but the same
  paragraph attributes a `+117 each` function-offset shift to it.
- The +22-line gap is **PR #160** (Terminal C, `d49edd8`, merged the
  same day, also touches `engine/wheel_runner.py`):
  - `git show d49edd8 --shortstat` → `1 file changed, 22 insertions(+)`.
  - `git diff 68418fb 86b917c7 --shortstat engine/wheel_runner.py`
    → `1 file changed, 117 insertions(+)` (the full delta from #126's
    `rank_strangles_by_ev` baseline at `68418fb` to audit start at
    `86b917c7`).
- Function offsets at the audit's snapshot match exactly:
  - `explore_ticker` → `:1448` ✓
  - `rank_covered_calls_by_ev` → `:1725` ✓
  - `rank_strangles_by_ev` → `:2192` ✓
- Function offsets today on `cc54619`: `:1559` / `:1836` / `:2324`.
  Further post-audit drift from PR #174 (~+111 each) and other
  intervening edits — not in scope here.

**The note:** the audit attributes the entire +117 shift to PR #161,
but #161 contributed 95 of those 117 lines and #160 (also
Terminal C, also wheel_runner.py, also same audit window) contributed
the other 22. The audit's substantive load-bearing claim — *function
**signatures** unchanged at the new offsets; drift is purely
positional* — is verified directly here (signatures at the cited
lines today match the audit's quoted `def ...` headers). The audit's
adjacent claim that the *bodies* are unchanged I take on the audit's
verification; spot-checking that further would require diffing each
body against the prior-audit snapshot, which is beyond M1–M7's
spot-check scope. The lack of body-edit commits to these functions
in `git log -- engine/wheel_runner.py` between `86b917c7` and
`cc54619` (only PR #174's two new methods landed) is consistent
with body-equivalence. The attribution wording is imprecise but not
consequential; a reader who runs the same `git diff` will hit the
+117 number the audit cites for the offset shift and the +95
number it cites for #161's stat, and can reconcile them on the spot.

A one-line follow-up clarification — "PR #161 (+95) plus PR #160
(+22, same window) total +117" — would close the precision gap.
Since PR #173 has now merged onto `main` at `2ccb936`, the follow-up
would land as a small docs PR against `main` rather than an in-place
edit on the audit branch. Candidate next-docs-sweep item.

---

## M5 — Test-count accounting

**Verdict:** **VERIFIED.**

`pytest --collect-only -q` on each of the three audited test files:

| File | Audit-claimed | Collected today | Run result |
|---|---|---|---|
| `tests/test_strangle_recommendation_gate.py` | **17** | **17** ✓ | 17 passed |
| `tests/test_authority_hardening.py` | **28** | **28** ✓ | 28 passed |
| `tests/test_ev_authority_log_schema.py` | **21** | **21** ✓ | 21 passed |

All three counts match exactly. Combined run: `66 passed in 2.61s`,
zero failures, zero skips.

The audit's per-PR test-count claims aggregated in Cross-cutting
Observation #5 are internally consistent with these three spot-checked
files; no inflation, no quiet shrinkage.

---

## M6 — Exclusion-list defensibility

**Verdict:** **VERIFIED.**

### #139 / #140 / #141 / #142 / #144 (provenance-ambiguous)

The audit excludes these five on the basis that there is no
*individual* `Terminal A — claim` comment on #113 before any of them
were opened — only Terminal A's 2026-05-23 17:02 UTC post-merge
"batch closure complete" cascade comment treats all eight cascade
PRs as Terminal A work. The cascade comment's wording ("Session
merge cascade — 8 PRs landed today") states *that they were merged
by Terminal A*; it does not formally identify *authorship* in the
shape #113 conventionally records (per-PR pre-merge claim with
branch + files).

Searching #113 with the prompt's exact predicate
(`select(.body | test("#139|#140|#141|#142|#144"))`) over every
comment returns no pre-open / pre-merge individual Terminal A claim
for any of the five. The only references are (a) Terminal A's
batch-closure cascade comment, (b) downstream Terminal C and Session
follow-up comments referencing the PR numbers retrospectively.

The audit's stated rule — applied from the audit prompt
verbatim — is *"If a PR in the list isn't claimed by Terminal A on
#113, drop it from the audit (provenance mismatch is itself a finding
to log)."* The audit follows this rule, is consistent with the prior
audit (PR #170 / `ab775bf`'s exclusion of the same set), and logs
the exclusion in the "Scope exclusions (provenance-ambiguous)"
section. **Conservatively defensible.**

A reader who reads the cascade comment as a sufficient
after-the-fact authorship claim would disagree; that's a judgment
call about how strict the provenance bar should be. The audit picks
the stricter bar and is explicit about doing so.

### #160 / #161 (Terminal C) and #170 / #172 (Terminal B)

All four exclusions confirmed by explicit non-Terminal-A claims on
#113:

- **#160** — "Terminal C — claim: annotate synthetic-data columns +
  add ohlcv-staleness warning." Branch `claude/ranker-provenance-annotation`.
- **#161** — "Terminal C — claim + PR opened: #161 (WheelRunner.explore_ticker
  — surface-exploration method)." Branch `claude/feat-explore-ticker`.
- **#170** — Terminal B's own session close-out comment names
  PR #170 explicitly: *"#170 — Independent audit of Terminal A's
  21-PR campaign | Terminal B (this session)."*
- **#172** — same close-out comment: *"#172 — `docs(portfolio_risk_gates)`:
  mark Kelly gate as preemptively reserved (#166 B1 + B3) |
  Terminal B (this session)."*

All four non-Terminal-A authorships independently verified.

---

## M7 — "No production callers yet" (Cross-cutting Observation #4)

**Verdict:** **VERIFIED.**

Re-ran the audit's exact grep against current `origin/main`
(`cc54619`):

```
grep -rln "rank_covered_calls_by_ev|rank_strangles_by_ev|
suggest_call_rolls|issue_ev_authority_token" --include="*.py"
engine/ dashboard/ scripts/ engine_api.py advisors/
  | grep -v "engine/wheel_runner.py|engine/wheel_tracker.py|
engine/portfolio_risk_gates.py|engine/candidate_dossier.py"
```

→ **EMPTY.** The new public surfaces still have no callers in
`engine/` (outside their defining files), `dashboard/`, `scripts/`,
`engine_api.py`, or `advisors/`. Extending the grep to also include
`available_buying_power` returns one hit:
`engine/portfolio_risk_gates.py:477` — the **docstring reference**
inside the Kelly gate's "Current-path reachability" paragraph,
which the audit explicitly noted.

**Post-audit context worth surfacing:** PR #174 (`94cb254`, merged
2026-05-25, AFTER the audit was written) added two new public
helpers to `engine/wheel_tracker.py`:

- `consume_ranker_row(row, ...)` — canonical chain calling
  `issue_ev_authority_token` then `open_short_put`.
- `portfolio_context_snapshot(...)` — builds the `PortfolioContext`
  for `build_dossiers` to consume.

Grep for those two helpers' callers outside `wheel_tracker.py`:

```
grep -rln "consume_ranker_row|portfolio_context_snapshot"
  --include="*.py" engine/ dashboard/ scripts/ engine_api.py advisors/
  | grep -v "engine/wheel_tracker.py"
```

→ one hit: `engine/candidate_dossier.py:423` — again a **docstring
reference** ("pass via :meth:`engine.wheel_tracker.WheelTracker.portfolio_context_snapshot`"),
no call. The audit's load-bearing observation #4 — *"the campaign
builds blocks; production wiring is deferred"* — remains accurate
today even after PR #174. PR #174 added another building block (the
two wire helpers) but did not extend the call chain to `engine_api.py`
or `dashboard/`.

The audit explicitly predicted PR #174 in its AI handoff ("Wiring
`rank_candidates_by_ev` → `issue_ev_authority_token` →
`open_short_put(current_ev_dollars=...)` and threading a
`PortfolioContext` through `build_dossiers` so R7/R8 fire live is
the natural next PR"). #174 landed exactly that scope — internal to
the tracker. The next step (live call from `engine_api.py` or the
dashboard) is still open.

---

## Cross-cutting observations

**1. The audit is honest about its snapshot.** The audit's
`origin/main SHA at audit start: 86b917c7` discipline lets a reviewer
distinguish *audit-time* claims from *current-main* drift. Every M1-M7
spot-check that could be verified at both snapshots was — and the
audit's snapshot-time numbers all match exactly. The post-audit drift
on `open_covered_call` (1061 → 1143) and the wheel_runner.py function
offsets (+~111 each) trace cleanly to PR #174 (merged 2026-05-25,
after the audit). This is the right shape for an audit doc: take a
defensible snapshot, document it, accept that the world moves on.

**2. M4 is the only soft spot, and it is cosmetic.** The audit's
text is internally self-consistent on the +95 line stat for PR #161,
and the function-offset shift it cites (+117) is verifiable today.
The imprecision is in attribution — #161's +95 plus #160's +22 sum to
the +117 the audit cites for the function shift, but the audit's
prose reads as if #161 alone caused the full +117. A reader who
re-runs `git show 27dae4f --stat` will see 95, not 117, and may
briefly conclude the audit is wrong before noticing #160 sitting
adjacent in the same Terminal-C window. A one-line "PR #161 (+95) +
PR #160 (+22) = +117 total" would close this; not consequential
enough to block the audit, but the natural minor-follow-up.
**Now that PR #173 is merged**, the clarification lands as a small
docs PR against `main` — folded into the next docs sweep, not its own
PR.

**3. The §2 surface holds across both audit layers.** The audit
correctly identifies that none of the 22 Terminal A PRs opened a
candidate path that bypasses `EVEngine.evaluate`. The 5-ticker EV
smoke (CLAUDE.md §6) at audit start was green; re-running it today
on `cc54619` would still be green (run from the prior session in
this same worktree: 5 rows, 0 NaN, `connector: MarketDataConnector`).
The #145 D16 token (issue-time refusal + consume-time stale-EV
guard) is correctly scored by the audit as `§2-adjacent and
§2-tightening, not §2-breaking`. The #165 R7/R8 soft-warns are
correctly scored as `downgrade-only by structure` (the only branch
that fires returns `"review"`, never `"proceed"`). No §2 breach
missed.

**4. The audit's "no production callers" observation is the most
load-bearing cross-cutting claim — and it remains accurate.** Even
though PR #174 added the in-tracker wire helpers post-audit, the
observation that *the audited surfaces are not called from
production code paths* (engine_api.py, dashboard, scripts, advisors)
holds today. The audit's prediction of #174 as "the natural next PR"
landed exactly that scope and no more. The follow-on (extending
the chain to `engine_api.py`'s webhook handler or the dashboard's
rank endpoint) remains open. That's not an audit defect; it's an
audit prediction that materialised partially.

---

_Read-only meta-verification. No edits to `engine/`, `tests/`,
`docs/TERMINAL_A_AUDIT.md`, or any decision-layer file._
