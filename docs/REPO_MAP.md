# REPO_MAP — the one "where / what / authoritative" index

> **Read this first.** It routes every "where does X live / what tests cover Y /
> what is authoritative for Z" question to the **one** owning doc, so you don't
> open three and reconcile them. It is mostly *pointers* (to avoid becoming a new
> drift source) plus two net-new tables (`engine/features/` origin note, launch-blocker
> subset). Rationale + the full audit: `archive/2026-05/REPO_EFFICIENCY_AUDIT.md`.

## Question router — open ONE doc per question

| You're asking… | Open | Not |
|---|---|---|
| How is this project **worked** — roles, handoffs, Run Summary format, concurrency, invariants? | `OPERATING_MODEL.md` (the single authoritative operating document) | — |
| Where does **module X** live + its role? | `MODULE_INDEX.md` | — |
| Where does an **exact file** live (grep target)? | `FILE_MANIFEST.md` (CI-guarded, exhaustive) | — |
| What is **authoritative** for a trade decision? | the **Authority block** below → then `OPERATING_MODEL.md` §9.2 and §7 for the rule text | re-deriving from 4 docs |
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
missing/errored→note only, R3/R4 skipped (D30) · R3 spot mismatch >2%→skip · R4 phase contradiction→skip
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
| `data/`, `data_processed/`, `data_raw/` | market-data layer (tiers, providers) | `docs/DATA_POLICY.md` |
| `scripts/` | data pullers + diagnostics | `FILE_MANIFEST.md` |
| `dashboard/` | Next.js cockpit + legacy CLI (D4) | `MODULE_INDEX.md` |
| `tradingview/` | Pine indicator + analyst workspace (D5) | `docs/TRADINGVIEW_INTEGRATION.md` |
| `backtests/` | research backtest harness + the regression reproducers | `FILE_MANIFEST.md` |
| `engine/features/` | feature-engineering library (moved from `src/features/` 2026-09-17): `technical.py` + `volatility.py` engine/data-live, seven research modules behind `data/feature_pipeline.py` | `MODULE_INDEX.md`, `DECISIONS.md` D2 |
| `utils/`, `config/` | helpers / config | `FILE_MANIFEST.md` |
| `tests/` | flat `test_*.py` files (+ `tests/fixtures/`); root `conftest.py`; live count via `ls tests/test_*.py \| wc -l` | `TESTING.md` |
| `docs/` | reference + design-contract docs | `FILE_MANIFEST.md` |

## `src/` — gone (2026-09-17)

The legacy scaffold was collapsed under Track F: `src/features/` →
`engine/features/`, `src/data/schemas.py` → `data/schemas.py` (the
`data/quality.py` → `engine/wheel_runner.py` chain-quality gate keeps its
import), `src/backtest/wheel_backtest.py` (heuristic, §2-non-compliant,
test-only) deleted with its test. `DECISIONS.md` D2 carries the update.

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
| Wheel lifecycle | `test_wheel_lifecycle`, `test_wheel_tracker_*`, `test_suggest_rolls_drops` |
| Interface / infra | `test_tv_*`, `test_dashboard`, `test_engine_api_port`, `test_infrastructure`, `test_recovery_*` |

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
