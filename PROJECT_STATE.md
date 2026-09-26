# Project State

**Last updated:** 2026-09-26 (the Operator's rulings on card 1's three findings,
§0 B, and `main` at `3b1203c` after #539, the close after #538. Earlier the same day:
close after #538, `main` at `610f9c8`. D33 card 1 ran on the
desktop: the morning pull retired, `main` merged into the desktop branch under guard, the
Drive census taken and the plan written; nothing deleted, Drive unchanged. The census found
a much fuller Theta copy on Drive than the records said, and a credential-shaped file on
Drive; both wait on the Operator (§0 B).
2026-09-25: close after #536, `main` at `f26e785`, the SmartWheelData area's
Drive parent id recorded in `docs/DATA_INVENTORY.md` §C.3 (N-7).
2026-09-24: close after #534, `main` at `7bfa626`, the D33 consolidation
tool merged. Earlier the same day: close after #533, `main` at `249f913`, D33 in force.
Before it: D33: the MacBook is outside the data estate, Drive
becomes a complete second copy, and the four data branches go once the history bundle
restores from Drive; §0 A and B updated, reviewed twice by Codex.
2026-09-23: close after #531, `main` at `8cf6389`, the v4 structure in force.
Before it: working structure v4, D32: §0 now opens with the direction (A),
the handoff (B) and the Branches line session-open measures drift from; the restart record is
§0 C. Earlier the same day, D31 step 5: git tracks no market data; the desktop root
holds all 144 manifest files, `check` 144/0/0. 2026-09-18: data-home ruling D31 recorded
in §1: the data lives under `SWE_DATA_ROOT` on the Operator's desktop, git holds the
manifest; #507 closed by the ruling. 2026-09-17: §3 reduced to the open items; the dated
May–July narrative moved verbatim to `archive/2026-09/`; §1 news sentence
updated for D29. 2026-09-11: §0 added below with the verified state at restart
and a pointer to the restart brief. Prior: 2026-07-02 deployment-truth doc pass —
data-currency blockquote refreshed to the 2026-06-04 frontier + 10-file/3-producer
counts; fingerprint note extended for the #465 broad_pull pins.)

> **Live sources of truth — don't duplicate them here, they decay.** The
> exact test count is in the latest CI run; in-flight work is on the campaign
> issue named in each Execution Prompt (the old board, issue #113, is closed —
> Operating Model §9.5); per-PR history is in `CHANGELOG.md`; the canonical
> verification index is `docs/VERIFICATION_INDEX_2026-05-28.md`. **One commit
> is pinned on purpose:** the Branches line in §0 records the `main` commit this
> file describes. Session-open counts the commits `main` has gained since it
> (`python scripts/session_open.py`), and session-close moves it (D32). A
> non-zero count is the signal that this file is behind, not a defect to hide.

This file records *temporal* state — what is authoritative now, what is
in progress, what is deprecated. It is the half-life partner of
`OPERATING_MODEL.md` (the *structural* contract; `CLAUDE.md` is its
checklist). Update this file
when you finish a meaningful unit of work or discover that something
described here is no longer accurate.

> ⚠️ **Real-money deployment gate — read `docs/PRODUCTION_READINESS.md`
> before any decision to operate this engine against a real brokerage
> account.** Predictive signal is verified (Spearman ρ ∈ [0.19, 0.55]
> across 14+ window×year cells; never negative; window- and capital-
> invariant within ~0.05). All three historical deployment blockers
> are now closed at the code level: **B1** (F4 tail-risk) shipped as
> a defense-in-depth bundle via PR #260 (realized-vol-ratio widening
> — the frequency guard) + PR #262 (R10 single-name notional cap —
> the magnitude guard); **B2** (D17 live-wire) shipped via PR #233 +
> #255; **B3** (capacity) is structurally closed via S34. The
> *structural* finding from S38 + S40 + S44 remains: the engine
> systematically underperforms passive in bull-dominated multi-year
> windows due to limited deployment (15-23% NAV) — this is not
> fixable engine-side. `docs/LAUNCH_READINESS.md` covers code-
> quality merge gates; `docs/PRODUCTION_READINESS.md` covers
> commercial deployment gates. They are complementary, not
> substitutes.

> **What the engine is / isn't (defensive-sleeve framing — heavy-verify
> 2026-05-31, Category C).** The engine is a **defensive premium sleeve, not a
> bull-market growth substitute**: it earns its keep in down/sideways/high-rate
> tape (+27pp vs passive in the 2022 bear, +10pp in the 2020 crash, at ~0.4–0.6×
> the index drawdown) and structurally lags strong bulls (−19 to −26pp) — size it
> as a complement to long equity beta, not a replacement. And **`ev_dollars` is a
> tail-risk-adjusted *ranking* score, NOT a dollar-profit forecast** — it has ≈0
> rank-correlation with realized $ P&L (I1); its value is in *selection* (top-K
> beats random, I6-B). Use `prob_profit` / `ev_roc` for ranking and read
> `ev_dollars` only as a tail-aware score. Basis:
> `docs/HEAVY_VERIFY_2026-05-31_INDEX.md` (Category C of
> `docs/HEAVY_VERIFY_2026-05-31_REMEDIATION.md`).

---

## 0. Direction, handoff, and the restart record — read this first

**Branches:** `main` is at `3b1203c` (2026-09-26). Others on `origin`:
- the four data branches that D31 step 6 deletes: `deep-history/bloomberg-raw`,
  `claude/daybot-bloomberg-pull`, `backup/drive-tier-c-2026-07-22` and
  `data/drive-migration`;
- `claude/project-restart-ai-agents-kot5jr`, the pen's branch, while a close is open.

Merging a pull request deletes its head branch on GitHub (a repository setting). So
`claude/data-home-desktop-round3` went when #538 merged, on 2026-09-26. All its commits
are in `main`, and the PR page can restore it. The desktop's checkout still has it
locally.

### 0 A. Business direction

- **What the engine is.** A probabilistic EV decision engine for the wheel on S&P
  500 names. It is a defensive premium sleeve, not a bull-market growth
  substitute (header above).
- **Where it is going** (D29, 2026-09-16; plan `docs/RESTART_PLAN_2026-09-16.md`):
  - tradeable expiries of 7, 14, 21 and at most 28 days;
  - trading close to events, behind a validated event-conditioned distribution
    (Track B);
  - an advisory exit evaluator (Track C);
  - a strategist commentary layer (Track D).
- **Data** (D29, D31, D33):
  - The Bloomberg Terminal is gone, and the frozen CSVs end 2026-07-02.
    Bloomberg data is kept "at all costs" (D33).
  - New data subscriptions are deferred (Track A parked, 2026-09-17). The Theta
    subscription is no longer active. Theta will be collected again from the
    beginning, from a source chosen once the repository structure is settled.
  - Every dataset lives on the Operator's desktop under `SWE_DATA_ROOT`, and git
    holds none. Google Drive becomes the complete second copy, one folder
    `swe-data/` laid out like the root (D33). The MacBook is outside the data
    estate.
- **Current focus.** Repository structure and efficiency (Track F), the data home
  (D31), and the working structure (D32).
- **Brokerage.** Read-only everywhere. There is no order path.

### 0 B. Handoff — updated at every close

- **Done.**
  - D31 steps 1–5. The manifest exists; the desktop root is filled and proved at
    144/0/0; git tracks no market data (#530).
  - The desktop's round 3 (2026-09-23), landed by #538:
    - a full-history bundle, built and proved by readback;
    - `data_archive` plus the bundle on Drive: 0 differences, 33 matching;
    - every child of Drive's `swe-local-only` pulled home and checked, its theta
      included (17,188 files).
  - Working structure v4 adopted (D32) and merged (#531, `8cf6389`). Codex, as the
    second opinion, found two defects in `scripts/session_open.py` and
    `scripts/check_working_structure.py`; both were fixed before the merge.
  - D33 (2026-09-24). The Operator dropped the MacBook and asked for Drive to hold
    all the data, with no duplicates. Codex reviewed the consolidation plan twice
    ("agree with changes" both times), and every finding was accepted. The
    Operator approved the final plan and D33's wording ("yes"). Merged in #533
    (`249f913`). Codex's automatic review of #533 found that a git object is not a
    byte copy of the bundle, so the old `.git` upload comes home like any other
    file (§C.3).
  - The D33 consolidation tool, `scripts/drive_consolidate.py`, merged in #534
    (`7bfa626`). It takes the Drive census, writes the plan and the ledger, copies
    Drive-only files home by Drive id without overwriting, verifies, and writes
    `SHA256SUMS`. No command deletes, moves or uploads. 202 tests; all 144
    deliberate breaks of its safety rules fail them. Seven independent reviews
    (the last in five rounds) and twenty-one Codex reviews; the last of each found
    nothing. One recorded limit: another name for the root that `realpath` keeps
    (a bind mount, a loopback share) passes its checks, so the cards spell every
    output path the ordinary way.
  - The SmartWheelData area's parent folder id, OptionsEngine_Project
    (`1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4`), recorded in `docs/DATA_INVENTORY.md`
    §C.3 after the pen read it from Drive. Merged in #536 (`f26e785`). The areas
    test now ties each of the tool's areas to its own §C.3 row: id, folder name,
    and the parent in the row's "Folder (id)" cell. Codex and then the
    independent check each found a gap in the test, and both were closed before
    the merge.
  - Desktop card 1, prove and plan, merged in #538 (`610f9c8`). Nothing was
    deleted from the root or from Drive, and Drive was only read. Its outputs are in
    the root's `_logs/d33-card1/`:
    - the morning-pull task exported (UTF-16) and unregistered; the root's
      `portfolio_snapshot.json` unchanged (5,279 B, 2026-07-18);
    - the tool's tests on Windows: 190 passed, 12 skipped (the expected skips);
    - `main` merged into the desktop branch under guard (`e13220f`): it removed only
      the 87 checkout copies (566,845,655 B), each first proved identical to the
      root's manifest copy;
    - the 20 `staging/` CSVs copied into the root (all 61 swe-ops data files were
      already there);
    - the desktop's copy of the bundle restored into an empty repository, with
      `fsck` clean and the four branch tips present;
    - the census: 7 areas, 332,772 objects, each equal to `rclone size`;
    - the plan: 120,139 files (10,716,801,880 B) to copy home, 0 needing a byte
      check, and 1 unresolved row, the credential-shaped file below.

    The pen's review, by a workflow in a separate context with each finding
    checked by a skeptic, found the numbers sound and the record partly wrong.
    The close corrects it; see the corrections in the two worklog fragments.
- **Card 1's three findings, and the Operator's rulings (2026-09-26).** The census
  found three things the records did not know:
  1. **Theta is very likely on Drive in full.** `SmartWheelData/data_processed/theta`
     (uploaded 2026-07-12/13) holds all 14 Theta trees. The pen confirmed on Drive
     that it holds `ticker=XOM` and AAPL expirations from 2016-01-08 to 2026-08-21.
     The review found that the `swe-local-only` upload stops at tickers A to N and
     at AAPL 2017-08-25. The records say the MacBook's ~132,862-file corpus was
     "not recovered" because only `swe-local-only` (17,188 files) was ever measured. The plan counts 118,367
     SmartWheelData files the root lacks. Card 1b confirms the Theta share from the
     plan.
     **Ruling: "yes".** Theta, which cannot be pulled again while the subscription
     is lapsed, gets Bloomberg's rule: nothing that holds Theta is deleted until
     the desktop copy and the Drive copy are both proven. The `DECISIONS.md` text,
     with the correction of D33 points 1–2, is drafted once card 1b gives the
     count, for the Operator's yes.
  2. **A credential-shaped file is on Drive:**
     `SmartWheelData/data_processed/ibkr/flex_credentials.json` (674 B, on Drive
     since 2026-07-12, last modified 2026-06-09). It has never been opened. The
     tool leaves it in place and lists it. CLAUDE.md §7 says that file never
     leaves the desktop. The pen checked its sharing settings, not its contents:
     only the owner has access.
     **Ruling: "yes, will do both".** The Operator rotates the IBKR Flex token and
     deletes this Drive copy by hand. No tool or agent opens, copies or deletes it.
  3. **The day-bot's `data_raw` is the day-bot's own data** (yahoo,
     ibkr_primary, swe_tests, evaluation files), not a copy of this project's.
     The plan would copy 1,725 of its files (34,816,381 B) into this project's
     archive.
     **Ruling: "yes, leave them out".** Card 2 copies none of them. They stay
     untouched in the day-bot's folder, which D33 only reads.
- **Remains.**
  1. **Card 1b, the plan check** (next, read-only). Written; it goes out after a
     second independent check. A short run on the desktop over card 1's saved
     plan:
     - what the 120,139 files are, by folder;
     - the Theta count;
     - the duplicates by reason;
     - `copy --dry-run`, which contacts no Drive and writes nothing, to prove 0
       conflicts and enough space;
     - the census's real duration.
  2. **Desktop card 2, copy and prove.** Nothing is deleted. The card:
     - copies home everything of this project's that exists only on Drive. By
       card 1's plan that is 118,414 files (10.7 GB) once the day-bot's
       `data_raw` is left out, into `data_archive/drive-legacy/<area>/…`, never
       into a live tree.
       One `copy` run takes two full censuses and all the downloads, an estimated
       12–20 hours by the tool's design, and it publishes nothing until the end;
     - builds `swe-data/` on Drive and checks it both ways, plus the checksum
       list;
     - runs the restore tests from Drive: the 144 manifest files into an empty
       root, and the bundle into an empty repository.
  3. **D31 step 6, under D33.** It runs after card 2. The four data branches are
     deleted in one atomic push with a lease on each, and only with the Operator's
     yes. #507 has been closed since 2026-09-23.
  4. **Desktop card 3, clean up.** It needs the Operator's yes, and Codex reviews
     the card first. Proven duplicates in the old Drive areas go to the trash,
     from a named list. The card also deletes the stray `ibkr$p` (a
     byte-identical copy of `portfolio_history.json`), on the desktop and on
     Drive.
  5. **A D31 gap:** git still tracks 20 data fragments under `staging/` (7,152,880
     B). Card 1 copied them into the root, so card 2 takes them to Drive. Untracking
     them remains proposed.
- **Next action.** The Operator rotates the Flex token and deletes the Drive copy
  by hand, and runs card 1b on the desktop (read-only) when the pen sends it. The
  pen then drafts the `DECISIONS.md` entry for the Operator's yes, and writes
  card 2.
- **Authorized.**
  - The D33 plan and wording ("yes", 2026-09-24). That covers step 6 under D33's
    conditions, still with a yes at the push, and card 3 with a yes at its gate.
  - Retiring the morning-pull task ("delete the morning pull"): done 2026-09-25 in
    card 1, after its gate.
  - Recording the SmartWheelData parent id ("yes on the Drive-id record",
    2026-09-25): done in #536.
  - v4 ("change it right away").
  - The rulings on card 1's findings (2026-09-26): "1. yes 2. yes, will do both
    3. yes, leave them out". Theta gets Bloomberg's rule; the Operator rotates the
    Flex token and deletes the Drive copy; the day-bot's `data_raw` is left out.
- **Proposed, not authorized.**
  - Purging the data from git history; the fixture subset; untracking the
    `staging/` data.
  - Optional: a stricter areas test, which would require the parent to be the
    exact last id in the "Folder (id)" cell and the folder name to sit next to its
    id. The independent check of #536 found seven mismatches the test still
    misses, such as a truncated parent id. The census catches each against Drive,
    so none can reach the data.
  - A `DECISIONS.md` entry, drafted after card 1b. It records the Theta rule and
    the `data_raw` ruling (both decided 2026-09-26), and corrects D33 points 1–2
    with card 1b's Theta count. The Theta lines in `docs/DATA_INVENTORY.md` and
    `docs/deadlines.md` follow it. Writing it needs the Operator's yes for the text.
  - Before anyone runs `scripts/data_manifest.py build` after card 2: decide whether
    `data_archive/drive-legacy/` belongs in the manifest. It would add about 120,000
    rows.
- **Dropped.** The MacBook transfer (step 2b) and Theta on Drive as tar chunks
  (D33).

### 0 C. Restart 2026-09-11 — the record

Work stopped on 2026-07-29 (UTC) with PR #522 merged (`ec1c5c5`, the
OPERATING_MODEL.md consolidation) and PR #523 (docs restructure, CI green)
left open. Nothing merged for six weeks. The full re-onboarding analysis is
`docs/RESTART_BRIEF_2026-09-11.md` (Part 1 product, Part 2 working schema);
this section is the durable summary.

**Engine health at restart (sandbox, `SWE_DATA_PROVIDER=bloomberg`, run
2026-09-11):** provider resolved to `MarketDataConnector`; the §9.4 5-ticker
EV smoke returned 4 rows in 11.8 s with JPM correctly dropped by the event
gate (`event_lockout:earnings@2026-10-13`) — the "five rows means healthy"
wording is date-dependent; launch-blocker subset 118 passed / 0 failed;
`pytest --collect-only` 3,720 tests; `check_manifest_coverage.py` 0/0;
`gen_worklog_index.py --check` OK; `check_lane_claim.py` OK.

**Data is stale.** Committed OHLCV frontier is **2026-07-02**
(`EXPECTED_FRONTIER`, set by #472; the §1 "point-in-time as of 2026-06-04"
blockquote below predates that bump). The earnings-calendar overlay snapshot is
2026-07-03. The connector warns on every live (`as_of=None`) run and
`SWE_REFUSE_STALE_LIVE=1` hard-refuses. Refresh needs a logged-in Bloomberg
Terminal (`docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md` §1, then bump
`EXPECTED_FRONTIER` / `EXPECTED_EARNINGS_CALENDAR_ASOF` and re-baseline per
`docs/DATA_POLICY.md` §5). **Data home ruled 2026-09-18 (D31):** the data lives
under `SWE_DATA_ROOT` on the Operator's desktop, never in git; `data/DATA_MANIFEST.json`
is the checksum ledger and `scripts/data_manifest.py` proves / fills a root. Draft
PR #507 (Google Drive as the store) is closed by that ruling; the untracking of the
served CSVs is held until the desktop `check` passes (`docs/DATA_INVENTORY.md` §A).

**Memory decay found and partly repaired.** `check_doc_currency.py` was FAILing
(this file 71 d, CHANGELOG 62 d; fail threshold 45 d), which blocked CI on any
new PR — cleared by this pass (CHANGELOG reconciled with the 27 merges of
2026-07-12 → 07-28). Still open: 60 worklog fragments marked `in-flight` whose
PRs merged; board #113 body frozen at 2026-05-30 and silent since 2026-07-04;
181 live-file references to "CLAUDE.md §" that now resolve to OPERATING_MODEL.md
§9.2/§7; README still cites CLAUDE.md for the four-layer model and NEVER list.

**Coordination reality.** July work ran on three ad-hoc Operator-away channels
(#493 machine mailbox, #494 Control Tower, #517 SANDBOX↔MACBOOK), none described
in OPERATING_MODEL.md v2; merge authority is stated three incompatible ways
(§2 Leg 3 / §7 vs §9.5 vs #494); `main` has no branch protection. Proposals
P1–P8 and 15 Operator decisions are in the brief, Part 2.

**Decision-layer items carried across the break.** F1 (roll_put/roll_call
bypass the D16 token + D17 caps), F3 (HMM `bull_quiet` positional label), F4
(IV-fallback look-ahead at historical `as_of`) remain `xfail(strict)` on main
(#516) — tracked, not fixed. The D19 / D21 / recalibration re-baseline block
(`docs/SUPERVISED_BLOCK_WORKLIST.md` Block B) is still operator-gated.

**Rulings 2026-09-16 (`DECISIONS.md` D29; plan `docs/RESTART_PLAN_2026-09-16.md`).**
`OPERATING_MODEL.md` v3 is the one protocol (two equal Strategists, Claude Code
and Codex; main Executor Claude Code in VS Code; Operator-away mode §3.1; prompts
per `docs/PROMPTING_STANDARD.md`). All news code is removed. Tradeable expiries
become 7/14/21/28 days with an event-aware policy to be validated (Track B); an
exit evaluator (Track C) and a strategist commentary layer (Track D) are to be
built. **Bloomberg Terminal access is gone for good**: `data/bloomberg/` is a
frozen 2018-01-02 → 2026-07-02 history and the live data path is rebuilt from
online sources (Track A). Model-name commit trailers are allowed again.

**2026-09-17.** Local-AI integrations removed (D29 R12): `local_agent/`,
`engine/trade_memo.py` + `/api/memo`, `/api/summary`, `/api/ollama_status`,
and the dashboard's Ollama chat panel, AI status indicator and chat tables.
Operator answers: data subscriptions deferred (Track A parked; the current
focus is repository structure and efficiency — Track F, the Operator picks the
rows); delta target deferred and noted; exit evaluator advisory; the strategist
brief's prose comes from an API model.

**2026-09-17, later (PR #524).** Track F executed under the ruling "all the cuts
except wheel_runner; no protection for main": `advisors/`, `ml/` + `models/`,
`studies/`, the `src/` scaffold (features promoted to `engine/features/`, schemas
to `data/`), five dormant engine modules, the TradingView MCP path and analyst
workspace are gone; R2 (chart context) is a note, not a stop (D30); the dated
docs are archived by vintage (#523 merged) and the pre-restart §3 narrative lives
in `archive/2026-09/`. `main` is deliberately unprotected (D29 ruling 13). The
`wheel_runner` ranker/ladder unification stays open as its own campaign.

---

## 1. Authoritative — do not bypass

| Module | Public entry | Locked by |
|---|---|---|
| `engine/ev_engine.py` | `EVEngine.evaluate` | `tests/test_audit_invariants.py`, `tests/test_audit_viii_*`, `tests/test_evengine_event_lockout.py`, `tests/test_dealer_multiplier_evengine_integration.py` |
| `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` | `tests/test_authority_hardening.py`, `tests/test_audit_viii_real_data_smoke.py`, `tests/test_f4_tail_risk_gap.py`, `tests/test_consume_ranker_row_anchor.py` |
| `engine/candidate_dossier.py` | `EnginePhaseReviewer`, rules **R1–R11** (R1 ev-non-finite / negative-EV; R2 chart-missing (a note since D30, no longer a stop); R3 spot-mismatch; R4 phase-contradiction; R5 ev-threshold; R6 short-gamma / dealer-flip; R7 portfolio VaR; R8 stress + dealer-regime; **R9 sector cap**; **R10 single-name cap**; **R11 elevated-vol top-bin**) | `tests/test_dossier_invariant.py`, `tests/test_portfolio_risk_gates.py`, `tests/test_dossier_r9_r10_audit.py`, `tests/test_r11_elevated_vol.py` |
| `engine_api.py` | HTTP API on `SWE_API_PORT` (default `:8787`; per-terminal in worktrees per D15); endpoint header in the file | `tests/test_tv_api.py`, `tests/test_tv_dossier.py`, `tests/test_engine_api_port.py` |

These four routes are the only sanctioned paths from raw inputs to a
tradeable verdict. Reviewers (chart provider, dealer positioning, **R7-R10
portfolio-context gates**, R11) can
downgrade outputs — never upgrade. News sentiment was severed from the
EV path on 2026-05-26 (D18) and the news stacks were removed on
2026-09-16 (D29): there is no news input anywhere on the decision path.
R1 (negative or non-finite EV →
blocked) is the hard CLAUDE.md §2 invariant; R7-R10 are conditional
soft-warns that fire only when a `PortfolioContext` is attached.
**The token gate (D16) re-checks R1 at fire time** — see `DECISIONS.md` D16.

> **Data currency (point-in-time).** The Bloomberg CSVs (under the data root — `SWE_DATA_ROOT`, D31; untracked from git 2026-09-23) are
> point-in-time as of **2026-06-04** (the R1 refresh cut, #338 —
> pinned by `EXPECTED_FRONTIER` in `tests/test_preflight_environment.py`;
> the legacy `pull_ohlcv.py` / `pull_liquidity.py` hardcode
> `end_date="2026-03-20"`, so re-running those two unedited would
> *regress* the frontier). Post the #477 xbbg-puller salvage (census
> refreshed 2026-07-08, D28 close-out), **9 of the 10** connector CSVs
> (`engine/data_connector.py::_FILES`) have a runnable in-repo
> producer (xbbg ones need a logged-in Bloomberg Terminal; the salvaged
> `pull_vol_iv.py` pins the current 2026-06-04 frontier). The one
> remaining gap is `sp500_earnings.csv` (BDS backfill — deferred).
> Current census: `docs/DATA_POLICY.md` §5; pre-salvage history:
> `archive/2026-06/bloomberg_refresh_runbook.md`.

## 2. Recent decision-layer audits

Each row links to the commit that shipped the change. Use
`git log --grep "^audit-<N>"` for the full diff and rationale.

| Audit | What shipped | Tests added |
|---|---|---|
| `audit` (`8ca561c`) | PIT bug fixes, TV webhook hardening, institutional EV engine | — |
| `audit-ii` (`3be3f2a`) | EV engine wired into runner; forward distributions; empirical surface; early-assignment-div; survivorship audit; calibration gate; sqrt impact | — |
| `audit-iii` (`81a42b1`) | POT-GPD CVaR; 4-state Gaussian HMM; Nelson-Siegel skew dynamics; Student-t copula; event gate | — |
| `audit-iv` (`2440891`) | TradingView visual-layer bridge + candidate dossier (Mode B) | `test_tv_dossier.py` |
| `audit-v` (`4afe7ea`, `48fe29b`) | Market-level dealer positioning (GEX, walls, flip, regime); P0/P1 unify decision authority; survivorship + chain quality + stress residual gates | `test_dealer_positioning.py` |
| `audit-vi` (`7e1bda7`) | Closed authority leaks across `tv` webhook, analyze, strangle, strikes, wheel_tracker EV gate | `test_authority_hardening.py` |
| `audit-vii` (`506b348`) | Unified orchestrator; HMM regime wiring; Grok/X agent; news API; ML guard | — |
| `audit-viii` (`e4c30e1`) | EV-path unit bugs (IV / risk-free rate percent↔decimal); roll/close P&L double-count; committee authority leak | `test_audit_viii_unit_invariants.py`, `test_audit_viii_e2e.py`, `test_audit_viii_real_data_smoke.py` (20 new tests) |

After audit-VIII the suite reported 1087 passed / 0 failed and 287
deprecation warnings (down from 1067+1 / 578).

**Post-audit-viii shipped work (2026-05).** Tracked as PR-level
entries in `CHANGELOG.md` rather than the `audit-<N>` series:

| Bundle | Shipped via | Tests added |
|---|---|---|
| **D16** EV-authority token verdict-binding (issuance + consume both re-check `ev_dollars > 0`) | PR #128 + audit-viii follow-on | `tests/test_authority_hardening.py` (D16 block), `tests/test_audit_viii_e2e.py`, `tests/test_wheel_tracker_persistence.py::test_persisted_token_consume_round_trip_d16` |
| **D17** portfolio-risk gates (sector / single-name / delta / Kelly / VaR / stress / dealer-regime) — tracker hard-blocks + R7-R10 dossier soft-warns; live-wired on `/api/tv/dossier` and `/api/tv/enrich` | PR #154 + #163 + #233 + #255 + #262 | `tests/test_portfolio_risk_gates.py`, `tests/test_authority_hardening.py::TestD17HardBlocks`, `tests/test_dossier_r9_r10_audit.py`, `tests/test_ev_authority_log_schema.py` |
| **F4 tail-risk widening v2** — realized-vol-ratio (RV30/RV252) regime-conditioned widening; replaces rolled-back HMM v1 from #253 | PR #260 | `tests/test_f4_rv_widening.py` |
| **Backtest regression harness** — S27/S32/S34/S35 + S43 rolling multi-window pinned against current engine | PR #196 / #220 / #270 | `tests/test_backtest_regression.py` (gated by `@pytest.mark.backtest_regression`) |
| **Verification battery wrap-up** — live R1-R10 check + real-data anchor checks + master canonical index | PR #268 / #270 / #271 / #273 | (docs + drivers; observable-test pattern, not pytest) |

The suite is **~2,500 tests** — run `pytest --collect-only -q` for the
live count — up from 1087 at audit-viii. A small number of Windows-local
Theta-tier tests skip/flake off the laptop; they are not engine defects.

## 3. Work in progress

**As of 2026-09-17.** The order of work is ruled in
`docs/RESTART_PLAN_2026-09-16.md` §1 (Track F structure pass, then Track E
protocol adoption, then B / C / D; Track A data refresh parked until a data
subscription is chosen) and mirrored in `ROADMAP.md`'s open-work table. The
run in flight is PR #524 (the restart run: protocol v3, the news and local-AI
removals, the Track F cuts).

Still-open findings carried forward from the pre-restart narrative:

- **prob_profit calibration — open structural finding.** The top bin
  (0.95, 1.0] of `prob_profit` is over-optimistic by 10–18pp across all ten
  backtest configurations measured (S22 … S40 W3); structural to the
  empirical-distribution method in `engine.forward_distribution`, not
  S38-specific; the F4 fix (#260) does not improve it. R10 (#262) is the
  load-bearing magnitude guard for exactly this reason. Doc:
  `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`. Open research direction:
  wire the POT-GPD tail extension (`engine/tail_risk.py`) into the
  `prob_profit` computation path.
- **D17 caps are armed only on the production entry path.** The bare
  `WheelTracker(...)` constructor keeps `enforce_sector_cap` /
  `enforce_single_name_cap` OFF by design (D22); `make_live_book_tracker()`
  — reached through `WheelRunner.consume_into_live_book` (#343) — arms both.
  Dossier soft-warns R7–R10 fire only when a populated `PortfolioContext` is
  attached (`/api/tv/dossier` and `/api/tv/enrich` when the caller supplies
  `nav`). The inline `wheel_tracker.py` comment that calls the R9 gate
  "default on" is stale (true only inside the factory) — flagged, not fixed;
  the original note deferred it to an operator-greenlit decision-layer touch.

The pre-restart §3 narrative (2026-05-04 → 2026-07-08: the D28 efficiency
audit, the D27 restructure, the 2026-06 campaign wave, the D17 closure and
cap-adoption status, F4 tail-risk widening v2, the S34–S47 backtest and
verification campaign, the TradingView MCP integration, the iv_surface
decision (resolved, D9), the SessionStart hook, the 2026-05-04 Theta refresh,
`pull_all.py` streaming, the Foundation pass, the coverage push, the D14
reorg) is preserved verbatim in
`archive/2026-09/PROJECT_STATE_WIP_2026-05_to_2026-07.md`.

## 4. Deprecated / phantom — do not extend

- `src/` — **collapsed 2026-09-17** (Track F, `DECISIONS.md` D2 update):
  `src/features/` → `engine/features/`, `src/data/schemas.py` → `data/schemas.py`,
  `src/backtest/wheel_backtest.py` (heuristic, §2-non-compliant) deleted with
  its test. Nothing imports `src` any more; `pyproject.toml` no longer names it.
- `ml/` and `models/` — **removed 2026-09-17** (Track F, D29 direction):
  the research ML models (`wheel_model.py`, `earnings_model.py`,
  `model_governance.py`) had no engine consumer; the orchestrator's
  calibration stage and the `SWE_MODELS_DIR` convention went with them.
- `dashboard/quant_dashboard.py` — legacy Python CLI dashboard. The
  primary dashboard is the Next.js app under `dashboard/src/`.
  README.md still describes the legacy CLI as the main entry point.
- News stacks — **removed 2026-09-16** (Operator ruling, `DECISIONS.md` D29):
  `financial_news/`, `news_pipeline/`, `morning_run.py`,
  `engine/news_sentiment.py`, `scripts/pull_news_sentiment.py` and their tests
  are deleted; `rank_candidates_by_ev` has no news multiplier and the
  engine API has no news endpoints. A news layer will be redesigned later;
  `archive/2026-09/NEWS_REDESIGN_CAMPAIGN.md` is retained as history.

## 5. Documentation drift to repair

These are stale relative to `CLAUDE.md` and the live code, and have
not been fixed in this review pass:

_All previously-listed drift entries here are now closed — see the
closer paragraph below for the route to each fix._

The entries that previously lived here for `README.md`,
`docs/CONTRIBUTING.md`, and `dashboard/README.md` were closed by
the entry-doc repair pass — see `ROADMAP.md` Track B (B1, B2, B4).
The `pyproject.toml` drift entry (broken `wheel = "src.cli:app"`
script + wrong `packages = ["src"]`) was closed by ROADMAP Track B5
— see `CHANGELOG.md` 2026-05. The `engine/__init__.py`
modern-decision-layer re-export entry was closed by `ROADMAP.md`
Track A3 — also see `CHANGELOG.md` 2026-05.

**This 2026-05-28 PR** (`claude/docs-consolidate-verification`)
closes the latest drift batch: it archives 12 superseded review
docs to `archive/2026-05/`, marks `docs/VERIFICATION_INDEX_2026-05-28.md`
as the single CANONICAL living verification index (with an
"Archived snapshots" map + "Deferred (locked by open PRs)" table),
and refreshes §1-§6 of this file to reflect the 2026-05 late
campaign (D17 closures, F4 v2, R9-R10, S34→S46, verification
wrap-up). No code changes; the engine SHA and the §2 invariant are
unchanged.

## 6. Branch + workflow policy

- Default branch: `main`. Don't edit `main` directly.
- Feature work happens on `claude/<short-slug>` branches. The branch
  that introduced these docs (`claude/handoff-docs`) was a
  documentation-only foundation review.
- CI runs on push to `main` / `develop` and on PRs (`.github/workflows/ci.yml`).
- The `.pre-commit-config.yaml` is committed; install with
  `pre-commit install`.
- `.claude/settings.json` SessionStart hook validates dataset presence,
  Theta manifest recency, and connector class on every fresh session.
- **Multiple executors (Operating Model v3, 2026-09-16; v4 2026-09-23 makes
  Codex a read-only second opinion, D32).** Every executor
  works in its own worktree or clone; one campaign issue per piece of work
  replaces the retired board (#113), task cards and allocator; when several
  executors share one machine, isolate them with
  `scripts/setup-terminal.{sh,ps1}` (`SWE_API_PORT`, `COVERAGE_FILE`,
  `PYTEST_CACHE_DIR`). See `OPERATING_MODEL.md` §2.4, §3.1 and §9.5.
