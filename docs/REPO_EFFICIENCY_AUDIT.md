# Repo Structure & Reading-Efficiency Audit + Executable Proposal

> **Status:** Phase-1 proposal (read-only audit). Authored 2026-05-31 against
> `main @ 1f72dd5`. This document recommends; it does not refactor. Every
> file move/delete/merge below is **Phase 2 — human-gated, deferred until the
> in-flight heavy-verify cycle (HT-A..HT-D) closes**. Hand this to the Major
> Session to decompose into gated cards. Nothing here has been actioned.
>
> _Housekeeping note for whoever lands this file: it is a new tracked doc, so it
> needs one row in `FILE_MANIFEST.md` (the `check_manifest_coverage` CI gate).
> That one-line edit was intentionally NOT made here to honour the Phase-1
> "create exactly one doc, edit nothing else" scope rule._

---

## Executive summary

**The premise is half right.** The task asked for a token-cost audit *and* a
test-dedup hunt "especially in tests/". The evidence (110 of 111 test files
opened+classified across a 15-agent fan-out, then the headline clusters
re-verified by hand) says:

- **Test redundancy is essentially a non-finding.** Zero whole-file DUPLICATEs.
  The suite is intentional, granular, and heavily invariant-pinning: **42 of 110
  files are §2/decision-layer/audit INVARIANT-PINs**, 63 are DISTINCT, 4 are
  self-documented COVERAGE-ONLY companions, 1 is dead weight. The "redundant
  tests, especially in tests/" framing is **refuted on evidence** — see §2.
- **The real "where/what/authoritative" token cost is navigation-doc
  fragmentation** — the same five orientation facts restated across 3–5 docs,
  several already **drifted out of sync** (the reviewer rule count is `R1-R10`
  in 2 docs and stale `R1-R6` in 3). An agent answering one bootstrap question
  reads ~1,700 lines across 4 docs and *still* hits a contradiction.
- **A second, large cost surfaced that the brief did not name: test-fixture
  duplication.** With no `tests/conftest.py`, the synthetic-OHLCV generator is
  re-inlined in **24 files / 43 occurrences**, a fake-connector stub in **23
  files / 39 occurrences**, plus `_ev_row` (6×), `_trade` (4×),
  `_proceeding_dossier`/`ChartContext` (dozens). The *root* `conftest.py` exists
  but its 5 fixtures are dead-or-shadowed.

### The three highest-leverage changes (value ÷ blast-radius)

| # | Change | Est. token reduction | Blast radius |
|---|---|---|---|
| **1** | **Fix the live drift** (DOC-ONLY): `R1-R6`→`R1-R10` in MODULE_INDEX/README/TESTING; strike the already-removed `wheel = "src.cli:app"` claim from 4 docs; stop pinning a test count. | Removes the *wrong-answer* tax entirely; ~0 cost to make. | Zero code. CLAUDE.md change is **proposed to the user only**. |
| **2** | **One canonical where/what index** (`docs/REPO_MAP.md` ~180L, or fold a router into the top of the CI-guarded `FILE_MANIFEST.md`). Single-sources the four drifting facts; every other doc becomes a one-line pointer. | "Where does X live + what's authoritative" bootstrap **~1,712L → ~420L (~75%)**, drift class structurally eliminated. | One new doc + one-line pointer edits. No file moves. |
| **3** | **`tests/conftest.py` shared-fixture consolidation** (factories: `make_synthetic_ohlcv`, `fake_connector`, `ev_row`, `chart_context`, `bloomberg_csv_dir`); delete the 3 never-consumed root fixtures. | −30–50 setup lines × ~24 files; removes a maintenance hazard (a connector-interface change today edits ~23 stubs). | Structural, truth-neutral PR; fixtures consumed by INVARIANT-PIN tests are HUMAN-APPROVAL. |

**Bias, stated up front:** prefer **indexing over moving**. A good index buys
most of the token savings at a fraction of the import/CI/git-history/collision
risk. No `engine/` file moves; no test moved or rewritten; the §2 firewall
(`engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`)
is untouched by every recommendation.

---

## §1 — Token-cost map of the navigation surface

### 1.1 The surface (current `main`, measured)

| Doc | Lines | Tier / role |
|---|---|---|
| CLAUDE.md | 183 | T1 entry contract (user-owned) |
| AGENTS.md | 103 | T1 agent read-order |
| README.md | 237 | T1 |
| PROJECT_STATE.md | 539 | T2 temporal state |
| MODULE_INDEX.md | 243 | T2 per-module index |
| FILE_MANIFEST.md | 746 | T2 exhaustive per-file index (CI-guarded) |
| TESTING.md | 234 | T2 test taxonomy |
| docs/worklog/INDEX.md | 117 | T2 per-task history (generated) |
| DECISIONS.md | 1,224 | T2 "why" (single-sourced — **healthy**) |
| CHANGELOG.md | 768 | T2 "shipped" |
| ROADMAP.md | 237 | T2 "next" |

The seven "where/what" docs total **~2,165 lines**. The layout is *well-governed*
(D14 tiering; `check_manifest_coverage.py` + `check_doc_currency.py` CI gates).
**The cost is not disorder — it is restatement and drift.**

### 1.2 The cost: same fact, many docs (cited, with drift)

- **Dossier reviewer rule count — the sharpest drift.** Current is `R1-R10`
  (CLAUDE.md §2 L58-109; PROJECT_STATE.md L46). **Stale `R1-R6`** in MODULE_INDEX.md
  L49, README.md L30, TESTING.md L52. *One fact, 5 docs, 3 wrong* — a fresh agent
  reading MODULE_INDEX/README under-counts the live reviewer rules by four. This
  is the single strongest argument for one canonical index.
- **`wheel = "src.cli:app"` stale-script claim.** PROJECT_STATE.md L303/L335,
  MODULE_INDEX.md L38, TESTING.md L175, ROADMAP.md L96 all flag it as a live
  problem — but `pyproject.toml` has **no `[project.scripts]` table at all** (it
  was already removed; `src/cli.py` never existed). *One falsehood, 4 docs.*
- **§2 hard invariant** restated in 5 docs (CLAUDE §2, AGENTS, README, DECISIONS
  D1, PROJECT_STATE §1); only DECISIONS D1 carries the rationale.
- **Four-layer mental model** in 3 (CLAUDE §1, README, PROJECT_STATE §1).
- **Module listing** split MODULE_INDEX (Role/Status columns) vs FILE_MANIFEST
  (exhaustive) — engine/ purpose-lines overlap heavily.
- **Test count drift:** TESTING.md "~2,300" (L3) / "~2,300+" (L14) /
  PROJECT_STATE.md "~2,500" (L91) — and the docs themselves warn not to trust
  pinned counts, yet pin them in three places.
- **Launch-blocker subset** enumerated in TESTING.md + README + PROJECT_STATE +
  LAUNCH_READINESS + the launch-blockers skill (3–4 copies of one list).
- **5-ticker smoke** verbatim in CLAUDE §4 + README + `/ev-smoke` (3 copies).
- **Doc-routing tables** ("where to look next") independently maintained in
  CLAUDE §5, AGENTS, README (3 pointer indexes into the same set).

### 1.3 Question → docs-to-open (the bootstrap tax)

| Question | Docs to open today |
|---|---|
| Q1 Where does module/file X live? | 2–3 (MODULE_INDEX + FILE_MANIFEST [+README tree]) |
| Q2 What tests cover area Y? | 1 (TESTING) for taxonomy; 2–3 to trace invariant→test→module |
| Q3 What is authoritative for decision Z? | 2–4 (CLAUDE §2 + PROJECT_STATE §1 + MODULE_INDEX + DECISIONS) **and hits the R-rule contradiction** |
| Q4 Current state / WIP? | 2–4 (PROJECT_STATE + ROADMAP + worklog INDEX + CHANGELOG) |
| Q5 Why was choice W made? | **1 (DECISIONS) — healthy, leave alone** |

### 1.4 Proposal — one canonical index (INDEX, don't move)

Add **`docs/REPO_MAP.md` (~150–200L)** — or fold the same router into the top
~40 lines of the already-CI-guarded `FILE_MANIFEST.md` (lighter; no new tracked
file; rides the existing manifest gate). Contents, all checkable so they cannot
drift:

1. **WHERE table:** top-level dir → one-line purpose → owning detail doc.
2. **AUTHORITY block:** the 4 routes (ev_engine / wheel_runner /
   candidate_dossier / engine_api) → public entry → the **single** canonical
   R1-R10 list → the pinning tests. CLAUDE.md §2 keeps the rule *text* (user-owned,
   current); everywhere else points here instead of re-listing.
3. **QUESTION router:** the 5 questions → the ONE doc to open for each.
4. **Single-source the volatile facts:** stop pinning a test count anywhere
   (reference `pytest --collect-only -q`); keep the launch-blocker file list only
   in TESTING.md with the others pointing to it.

**Before/after (measured):** answering "where does X live + what's authoritative
for it" today = scan CLAUDE (183) + MODULE_INDEX (243) + FILE_MANIFEST (746) +
PROJECT_STATE (539) ≈ **1,712 lines**, and still hit the R1-R6/R1-R10
contradiction with no signal which is right. After: REPO_MAP (~180) + one
targeted doc (e.g. MODULE_INDEX 243) ≈ **420 lines**, zero contradiction.
**~75% reduction**, and the drift class is eliminated structurally, not re-fixed
by hand. **Confidence: high** (line counts measured; overlaps cited).

---

## §2 — tests/ inventory + dedup verdict per cluster

**Method:** 111 `test_*.py` files; 110 classified by a 15-agent fan-out (each
opened its files and reported test-fn names + assertions + imports + fixtures +
invariant-pin signal); the headline clusters re-opened by hand for this author.
**Only `tests/fixtures/` is a subdir; there is no `tests/conftest.py` (the
`conftest.py` is at repo root).**

### 2.1 Classification result

| Verdict | Count | Meaning |
|---|---|---|
| DISTINCT | 63 | real, non-overlapping coverage — keep |
| **INVARIANT-PIN** | **42** | pins §2 / decision-layer / audit / authority / D11 / D16 / D17 / PIT / Greeks-unit contract — **DO NOT TOUCH** |
| COVERAGE-ONLY | 4 | self-documented coverage-lift companions — additive |
| UNCERTAIN | 1 | `test_wheel_cycle.py` (see 2.4) |
| **DUPLICATE** | **0** | none found, on evidence |

The §2/decision-layer INVARIANT-PIN set (the protected firewall — any
move/merge is **HUMAN-APPROVAL-REQUIRED**): `test_check_lane_claim`,
`test_dossier_invariant`, `test_dossier_r9_r10_audit`, `test_decision_layer_wiring`,
`test_consume_ranker_row_anchor`, `test_ranker_tracker_wire`, `test_ranker_transparency`,
`test_wheel_runner_select_book`, `test_covered_call_ranker`, `test_strangle_ev_ranker`,
`test_strangle_recommendation_gate`, `test_ev_non_finite_defense`,
`test_evengine_event_lockout`, `test_dealer_multiplier_evengine_integration`,
`test_authority_hardening`, `test_ev_authority_log_schema`, `test_audit_invariants`,
`test_audit_viii_{unit_invariants,e2e,real_data_smoke}`, `test_launch_blockers`,
`test_pit_leaks`, `test_point_in_time`, `test_greeks_unit_invariants`,
`test_iv_surface_failloud` (D9), `test_news_severance` (D18) … (42 total).

### 2.2 The named suspected clusters — verdicts (evidence-cited)

**theta — `test_theta_connector{,_coverage,_v3}` → NOT redundant (3 distinct tiers).**
- `test_theta_connector.py` (12) = **DISTINCT** live-Terminal integration smoke
  (module-skips in CI; this is the source of the 2 known env-only failures).
- `test_theta_connector_v3.py` (86) = **INVARIANT-PIN** — the mocked unit suite +
  the **D11 PerEndpointFailure** contract (issue #71).
- `test_theta_connector_coverage.py` (31) = **COVERAGE-ONLY**; its own docstring
  names `_v3` as the 87-test base it extends and lists the exact uncovered source
  lines. *Cited non-overlap:* `_v3 test_happy_path` asserts `mid==1.525` (bid+ask
  present); `_coverage test_mid_uses_bid_when_no_ask` asserts `mid==1.50` (ask
  absent) — explicitly opposite branches. The D11 contract is pinned on
  **disjoint** methods across the two (v3: get_ohlcv/get_fundamentals; _coverage:
  get_iv_rank/get_vix_regime/get_vol_risk_premium). **Verdict: keep all three.**
- `_coverage` *could* be folded into `_v3` (both mock the same module) as low-value
  tidy — but `_v3` is INVARIANT-PIN ⇒ **HUMAN-APPROVAL**. Not recommended.

**`test_wheel_runner_coverage.py` → COVERAGE-ONLY, §2-adjacent.** Docstring: it
tests only the *shallow* surface (TickerAnalysis, lazy properties, provider
selection, `_compute_wheel_score`, empty `screen_candidates`); the EV-path is
covered by the real-data smoke. **No overlap with the ranker tests.** It imports
a §2-trio module (`wheel_runner`) ⇒ any move is **HUMAN-APPROVAL**.

**`test_portfolio_copula_coverage.py` → COVERAGE-ONLY (hypothesis refuted).** It
is *not* a mis-named sole test: the happy paths live in
`test_quant_upgrades.py::TestPortfolioCopula`; this file covers the edge branches
(PSD repair, Cholesky fallback, empty marginals). Zero overlapping test names.
**Keep both.**

**`test_check_manifest_coverage.py` → DISTINCT (false positive).** Despite the
`_coverage` suffix it tests the *manifest-coverage script*, not a coverage-lift.

**recovery / external_data trios → DISTINCT.** `test_recovery_{checkpoints,fallbacks,health}`
target different `news_pipeline/recovery/` modules; the 4 `test_external_data_*`
hit different adapters. No shared test names.

### 2.3 Genuine overlaps found (all between SACRED files → flag, don't merge)

- **R1 "pristine chart cannot rescue negative EV"** pinned in **3** files
  (`test_tv_dossier`, `test_dossier_invariant`, `test_dossier_cp1252`) — **intentional
  §2 firewall triplication** (different threat each: perfect chart / novel provider
  / note-encoding). Do **not** merge.
- **R9 sector-cap happy path** — closest thing to a true duplicate: identical
  setup + assertions in `test_dossier_invariant` vs `test_dossier_r9_r10_audit`
  (audit version is the richer superset, adds note-text). **Both INVARIANT-PIN.**
  If ever consolidated, fold the bare check into the audit version **only with §2
  owner sign-off**; default recommendation is **leave** (firewall redundancy).
- **`test_news_sentiment::TestSentimentMultiplier`** overlaps `test_news_severance`
  (D18) — additive; `TestSentimentMultiplier` could shrink to a one-band smoke
  pointing at `test_news_severance` (the authoritative D18 pin). Low value.
- **`test_option_pricer` basic BSM cases ⊂ `test_quant_fixtures`** (parity,
  expired, zero-vol — same constants; quant_fixtures tol `1e-10` vs `1e-3`).
  Neither is INVARIANT-PIN. `test_option_pricer` also has unique coverage
  (vectorized pricing, `estimate_option_price`, dividend monotonicity) ⇒
  **COVERAGE-ONLY overlap, not a whole-file dup.** Folding the basic cases into
  quant_fixtures is a small, separate content PR. Low value.

### 2.4 The one genuine cleanup (verified by hand)

**`test_wheel_cycle.py` (48 L) — 0 `assert`, 0 `def test_`, 9 `print()`.** It is a
module-level **print-script**: pytest collects nothing from it, so it provides
*zero regression value*. The same lifecycle (open_short_put → assignment →
covered_call → expiration) is covered with **57 assertions across 13 tests** in
`test_wheel_lifecycle.py` (and again in `test_wheel_tracker_persistence.py`).
**Verdict: delete (or convert to assertions).** Off the §2 path (tracker
bookkeeping). Lowest-risk action in this document.

---

## §3 — Structure / location proposal

### 3.1 tests/ subdirectories → **DO NOT MOVE** (index instead)

Strong evidence *against* layer subdirs (`tests/decision_layer/` etc.):
- `.github/workflows/ci.yml` runs **6 files by exact flat path** (L138-155:
  test_quant_fixtures, test_point_in_time, test_properties, test_advanced_quant,
  test_infrastructure, test_dashboard). Moving any breaks that CI step.
- TESTING.md launch-blocker subset (L146-152) and the per-module pre-merge table
  (L183-198) invoke **exact flat paths and node-IDs**; the launch-blockers skill
  and the in-flight heavy-verify cycle read these literals.
- `tests/` is a package (`tests/__init__.py`) with the single root conftest;
  subdirs would need per-dir `__init__.py` and remove the no-duplicate-basename
  safety margin.

**Risk = HIGH** (CI + documented-command breakage + collides with HT-A..HT-D).
**Recommendation: a layer→file *lookup table*** (in REPO_MAP or TESTING.md). It
captures the entire mental-model navigation benefit at ~0 risk. Files stay flat.

### 3.2 `conftest.py` → **warranted, but reframe** (a root one already exists)

Premise correction: there *is* a root `conftest.py`; its 5 fixtures are
**dead-or-shadowed** — `sample_iv_series` and `sample_greeks` have **zero live
consumers** (grep); `sample_prices`/`sample_ohlcv`/`sample_portfolio` are
re-defined locally with *incompatible* shapes (e.g. `test_features` n=100 vs
conftest n=252). So the root fixtures cost read-tokens yet are never the version
actually used.

Meanwhile the **real** duplication (evidence, current main):
- synthetic-OHLCV GBM generator — **24 files / 43 occurrences**
- fake-connector `get_ohlcv` stub — **23 files / 39 occurrences**
- `_ev_row(...)` ranker-row builder — **6 files**
- `_trade(...)` ShortOptionTrade factory — **4 files** (+ mirrors `test_dealer_positioning`)
- `_proceeding_dossier` / `ChartContext` builders — defined 3× *within*
  `test_dossier_invariant` alone, plus across the dossier files
- `requests_mock.Mocker` fixture — **6 files** (4 external_data + 2 theta)

**Proposal:** a new **`tests/conftest.py`** housing shared factories —
`make_synthetic_ohlcv(n,seed,mu,vol)`, a parametrizable `fake_connector`,
`ev_row(**overrides)`, `chart_context(**overrides)` + `proceeding_dossier`,
`bloomberg_csv_dir(tmp_path)`, `requests_mocker` — and **delete the 3
never-consumed root fixtures** (reconcile `sample_prices`/`sample_ohlcv` with the
divergent local shapes first). Saves ~30–50 setup lines per affected file and
removes the maintenance hazard. **Import impact:** none on production; many test
files import the new fixtures (behaviour unchanged). **§2 impact:** fixtures
consumed by INVARIANT-PIN tests must be migrated with assertions byte-unchanged
⇒ **HUMAN-APPROVAL** for those. Ship as a **truth-neutral structural PR**
(single-concern rule); do it **after** HT-A..HT-D, which reads `tests/`.
**Risk: MEDIUM.**

### 3.3 `src/` mislocation → flag, do not action (D2)

Grounded importer analysis of the 19 `src/*.py`:

| src/ file | Importers | Status |
|---|---|---|
| `src/features/technical.py` | `engine/strangle_timing.py`, `engine/tv_signals.py`, `engine_api.py` + data ETL + scripts + tests | **LIVE (decision-adjacent)** — blocks deletion |
| `src/data/schemas.py` | `data/quality.py` → `engine/wheel_runner.py` (the §2 chain-quality gate) | **Transitively on EV path — keep** |
| `src/features/volatility.py` | research ETL + scripts + tests | not live engine |
| `src/features/{assignment,dynamics,events,labels,options,regime,vol_edge}.py` | `data/feature_pipeline.py` + tests | research/test only |
| `src/data/validators.py` | self only | **dead** (already coverage-omitted) |
| `src/backtest/wheel_backtest.py` | `tests/test_wheel_backtest.py` only | test-only |
| `src/{risk,models,execution}/` | none (empty `__init__` stubs) | **zero importers** |

**Reconcile decision (do not action), lowest-risk-first:**
1. **DOC FIX** — strike the dead `wheel = "src.cli:app"` claim from PROJECT_STATE
   L303/L335, MODULE_INDEX L38, TESTING L175, ROADMAP L96 (it's gone from
   pyproject; zero code blast radius). Also refresh D2's "pyproject still names
   src — known stale" note (the cli:app half is already fixed).
2. The empty `src/{risk,models,execution}` stubs + `src/data/validators.py` are
   removable today with near-zero blast radius — **but D2 says keep frozen ⇒
   flag, don't move.**
3. A true `src/features/technical.py` → `engine/` migration is the *only* thing
   that lets `src` leave pyproject `[tool.hatch] packages`, isort
   `known-first-party`, and coverage `source`. **Blast radius:** 2 live engine
   modules + engine_api + 2 data ETL modules + a benchmark script + ~6 tests + 3
   pyproject stanzas + the CI ruff target list. **Out of D2 budget; bias to
   index.** **§2 firewall:** `wheel_runner` imports `data/quality` which imports
   `src/data/schemas` — any schemas touch is HUMAN-APPROVAL.

---

## §4 — Phased, disjoint execution plan (for the Major Session)

Ordered by value ÷ blast-radius. **All deferred until HT-A..HT-D closes.** Each
card is disjoint (own file-set). Step 0 first registers this proposal doc.

| Step | Card | File-set | Import/CI impact | Touches §2 trio? |
|---|---|---|---|---|
| 0 | Register this doc | `FILE_MANIFEST.md` (+1 row) | manifest gate | No |
| 1 | **Drift fix (DOC-ONLY)** — `R1-R6`→`R1-R10` (MODULE_INDEX L49, README L30, TESTING L52); strike `src.cli:app` from 4 docs; replace pinned test counts with `--collect-only` reference | MODULE_INDEX, README, TESTING, PROJECT_STATE, ROADMAP | none | No (CLAUDE.md unchanged — **propose to user**) |
| 2 | **Canonical index** — add `docs/REPO_MAP.md` (WHERE / AUTHORITY / QUESTION-router / src-per-file truth / engine→test map / layer→test lookup / single launch-blocker list); convert duplicate sections elsewhere to one-line pointers | new doc + pointer edits in MODULE_INDEX/README/TESTING/PROJECT_STATE | manifest gate | No (additive routing) |
| 3 | **Delete `test_wheel_cycle.py`** (print-script, no asserts; covered by test_wheel_lifecycle) | `tests/test_wheel_cycle.py` | removes a non-collecting file | No |
| 4 | **`tests/conftest.py` consolidation** (shared factories; delete 3 dead root fixtures) — truth-neutral structural PR | new `tests/conftest.py` + import edits across ~24 test files | full suite must stay green | **Fixtures used by INVARIANT-PIN tests ⇒ HUMAN-APPROVAL**; assertions unchanged |
| 5 | **Small content folds** — `test_option_pricer` basic BSM cases → `test_quant_fixtures`; trim `test_news_sentiment::TestSentimentMultiplier` → pointer to `test_news_severance` | those 2 file pairs | none | No (neither file is INVARIANT-PIN) |
| 6 | **R9 sector-cap near-dup** | `test_dossier_invariant` / `test_dossier_r9_r10_audit` | none | **YES — both INVARIANT-PIN; HUMAN-APPROVAL; default LEAVE** |
| 7 | **`src/` reconcile** (flag-only this cycle) | pyproject + src/ + importers | large | **YES (schemas→wheel_runner); out of D2 budget** |

**Sequencing rationale:** Steps 1–3 are zero/near-zero risk and DOC/dead-weight
only — do them first and independently. Step 4 is the biggest *test-token* win
but is structural and §2-adjacent, so it waits for the heavy-verify cycle and
goes in as a single truth-neutral PR. Steps 5–7 are low-value / high-care and
explicitly gated. **The §2 firewall is never edited by steps 0–5.**

---

## Confidence & what would change the verdict

- **High** confidence: the classification counts, the cited drift, the fixture
  duplication counts (greps + opened files), the `test_wheel_cycle` deletion, the
  `src/` importer map, the subdir-is-risky finding (CI literals read directly).
- **Medium** confidence: that *no other* whole-file DUPLICATE exists among the
  ~35 files not opened line-by-line (15 agents opened the rest; the headline
  clusters were re-verified by hand). Anything not personally opened that this
  doc would *act* on is flagged HUMAN-APPROVAL or left as DISTINCT.
- This is a proposal. No PR opened, no file moved, no test changed, `engine/`
  untouched. Hand to the Major Session to decompose into the gated cards above.
