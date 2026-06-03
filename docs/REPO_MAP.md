# REPO_MAP — the one "where / what / authoritative" index

> **Read this first.** It routes every "where does X live / what tests cover Y /
> what is authoritative for Z" question to the **one** owning doc, so you don't
> open three and reconcile them. It is mostly *pointers* (to avoid becoming a new
> drift source) plus two net-new tables (`src/` per-file truth, launch-blocker
> subset). Rationale + the full audit: `docs/REPO_EFFICIENCY_AUDIT.md`.

## Question router — open ONE doc per question

| You're asking… | Open | Not |
|---|---|---|
| Where does **module X** live + its role? | `MODULE_INDEX.md` | — |
| Where does an **exact file** live (grep target)? | `FILE_MANIFEST.md` (CI-guarded, exhaustive) | — |
| What is **authoritative** for a trade decision? | the **Authority block** below → then `CLAUDE.md` §2 for the rule text | re-deriving from 4 docs |
| What **tests** cover area Y / what must I run? | `TESTING.md` (taxonomy + per-module "what to run") | — |
| What is the **current state / WIP**? | `PROJECT_STATE.md` (now) · `ROADMAP.md` (next) · `docs/worklog/INDEX.md` (per-task) | pinned counts (run `pytest --collect-only -q`) |
| **Why** was a choice made? | `DECISIONS.md` (single-sourced — D1…) | — |

## Authority block — the §2 firewall (do not bypass)

The four sanctioned routes from raw inputs to a tradeable verdict (full contract
+ rationale: `CLAUDE.md` §2, `DECISIONS.md` D1):

| Route | File | Public entry | Role |
|---|---|---|---|
| Ranker authority | `engine/ev_engine.py` | `EVEngine.evaluate` | **authority** — the only place a candidate becomes tradeable |
| Runner | `engine/wheel_runner.py` | `WheelRunner.rank_candidates_by_ev` | the one route every tradeable path takes |
| Reviewer | `engine/candidate_dossier.py` | `EnginePhaseReviewer` (R1–R11) | downgrade-only |
| Interface | `engine_api.py` | HTTP API on `:8787` | serves verdicts; never re-ranks |

**Invariant:** no tradeable candidate bypasses `EVEngine.evaluate`; reviewers can
**downgrade** (proceed→review→skip→blocked) but never upgrade; the dealer
multiplier is clamped `[0.70, 1.05]` and scales `ev_dollars` only, never `ev_raw`.

**Reviewer rules (the canonical count is R1–R11 — see D23 in `DECISIONS.md`; rule *text* lives in `CLAUDE.md`
§2):** R1 negative/non-finite EV→blocked (R1a non-finite guard) · R2 chart
missing/errored→review · R3 spot mismatch >2%→skip · R4 phase contradiction→skip
(*dormant*) · R5 EV ≥ `min_proceed_ev` (10.0)→proceed else review · R6 short-gamma
+ strike ≥ put wall / near gamma flip→review · **R7–R10 = D17 portfolio
soft-warns** (require an attached `PortfolioContext`): R7 VaR breach · R8
stress/dealer-regime · R9 sector-cap breach · R10 single-name-cap breach · R11
elevated-vol top-bin size-down (VIX level >25 + top-bin `prob_profit` >0.90). All
downgrade-only.

**Pinned by** (the INVARIANT-PIN test set — never move/merge without §2 owner
sign-off): `test_audit_invariants`, `test_audit_viii_{unit_invariants,e2e,real_data_smoke}`,
`test_dossier_invariant` (R1–R10), `test_r11_elevated_vol` (R11),
`test_dossier_r9_r10_audit` (R9/R10 structural),
`test_authority_hardening` + `test_ev_authority_log_schema` (D16 token gate),
`test_decision_layer_wiring` + `test_consume_ranker_row_anchor` + `test_ranker_tracker_wire`
(D16/D17 wire), `test_ev_non_finite_defense` (R1a), `test_evengine_event_lockout`
(no-rescue ordering), `test_dealer_multiplier_evengine_integration` (clamp),
`test_launch_blockers`, `test_check_lane_claim` (the PR lane-claim gate),
`test_pit_leaks` + `test_point_in_time` (no look-ahead), `test_greeks_unit_invariants`
(`docs/GREEKS_UNIT_CONTRACT.md`).

## WHERE — top-level layout → owning detail doc

| Path | Purpose | Detail in |
|---|---|---|
| `engine/` | quant + decision layer (the brain) | `MODULE_INDEX.md` |
| `engine_api.py` | HTTP API on `:8787` | `MODULE_INDEX.md` |
| `advisors/` | Buffett/Munger/Simons/Taleb committee (advisory only) | `MODULE_INDEX.md` |
| `data/`, `data_processed/`, `data_raw/` | market-data layer (tiers, providers) | `docs/DATA_POLICY.md` |
| `scripts/` | data pullers + diagnostics | `FILE_MANIFEST.md` |
| `financial_news/`, `news_pipeline/` | two off-EV-path news subsystems (D3) | `MODULE_INDEX.md`, `DECISIONS.md` D3 |
| `dashboard/` | Next.js cockpit + legacy CLI (D4) | `MODULE_INDEX.md` |
| `tradingview/` | Pine indicator + analyst workspace (D5) | `docs/TRADINGVIEW_INTEGRATION.md` |
| `ml/`, `backtests/` | research models + backtest harness | `FILE_MANIFEST.md` |
| `src/` | **deprecated phantom (D2)** — but partly live; see the table below | this doc + `DECISIONS.md` D2 |
| `utils/`, `config/`, `local_agent/` | helpers / config / experimental agent | `FILE_MANIFEST.md` |
| `tests/` | 111 flat `test_*.py` (+ `tests/fixtures/`); root `conftest.py` | `TESTING.md` |
| `docs/` | reference + design-contract docs | `FILE_MANIFEST.md` |

## `src/` per-file truth (kills the recurring "is src/ dead?" grep)

`src/` is frozen-deprecated (D2) **but not uniformly dead.** The
`wheel = "src.cli:app"` console-script is **gone** (no `[project.scripts]` in
pyproject); `src` remains in `[tool.hatch] packages` / isort / coverage by the D2
freeze. Per-file import reality (grounded by importer grep):

| `src/` file | Importers | Status |
|---|---|---|
| `features/technical.py` | `engine/strangle_timing.py`, `engine/tv_signals.py`, `engine_api.py` + data ETL + scripts + tests | **LIVE (decision-adjacent)** — blocks deletion |
| `data/schemas.py` | `data/quality.py` → `engine/wheel_runner.py` chain-quality gate | **transitively on the EV path — keep** |
| `features/volatility.py` | research ETL + scripts + tests | not live engine |
| `features/{assignment,dynamics,events,labels,options,regime,vol_edge}.py` | `data/feature_pipeline.py` + tests | research/test only |
| `data/validators.py` | self only | dead (coverage-omitted) |
| `backtest/wheel_backtest.py` | `tests/test_wheel_backtest.py` only | test-only |
| `risk/`, `models/`, `execution/` | none (empty `__init__` stubs) | zero importers |

## Tests — find them without globbing

`TESTING.md` is authoritative for the **taxonomy** and the **per-module "what to
run when you touch X"** table. Quick layer lookup (files stay flat — see the audit
for why subdirs are *not* recommended):

| Layer | Representative test files |
|---|---|
| Decision-layer / §2 | the INVARIANT-PIN set in the Authority block above |
| Ranker / dossier | `test_wheel_runner_select_book`, `test_covered_call_ranker`, `test_strangle_ev_ranker`, `test_ranker_*`, `test_tv_dossier*`, `test_dossier_*` |
| Quant / pricer | `test_option_pricer`, `test_binomial_tree`, `test_monte_carlo`, `test_tail_risk`, `test_realized_vol`, `test_quant_fixtures` (authoritative BSM), `test_greeks_unit_invariants`, `test_properties` |
| Data / connectors | `test_data_*`, `test_bloomberg_loader`, `test_theta_connector{,_coverage,_v3}`, `test_external_data_*`, `test_features` |
| Risk / portfolio | `test_risk_manager`, `test_portfolio_tracker`, `test_portfolio_copula_coverage`, `test_stress_testing`, `test_portfolio_risk_gates`, `test_dealer_positioning` |
| Wheel lifecycle | `test_wheel_lifecycle`, `test_wheel_tracker_*`, `test_suggest_rolls_drops`, `test_wheel_backtest` |
| News (off EV path) | `test_news_pipeline`, `test_news_processing`, `test_news_sentiment`, `test_news_severance` (D18), `test_adversarial_news`, `test_financial_news` |
| Interface / infra | `test_tv_*`, `test_mcp_client`, `test_dashboard`, `test_engine_api_port`, `test_infrastructure`, `test_recovery_*` |

### Launch-blocker subset (the §2 gate — single source)

Run before any decision-layer change (the runnable command + the canonical list
live in `TESTING.md`; `docs/LAUNCH_READINESS.md` and the `launch-blockers` skill
point here):

```
pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py \
       tests/test_r11_elevated_vol.py \
       tests/test_authority_hardening.py tests/test_audit_viii_unit_invariants.py \
       tests/test_audit_viii_e2e.py tests/test_audit_viii_real_data_smoke.py \
       tests/test_launch_blockers.py -v
```

---
*Maintenance:* keep this file pointer-only. If a fact would live in two places,
it belongs in its owner (above) and is *referenced* here, never copied.
