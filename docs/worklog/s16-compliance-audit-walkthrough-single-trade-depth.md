---
id: S16
title: Compliance / audit walkthrough (single-trade depth on the diagnostic surface)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** A compliance audience asks "show me why this trade was
authorized" (or refused, or never proposed). The campaign covered
portfolio-level observability in S15; this Sn drills the *other*
direction — does the diagnostic surface a single candidate emits
(ranker row + `.attrs["drops"]` entry + dossier verdict + R-rule
notes) reconstruct a defensible narrative **without re-running the
engine**? Per case, walk Inputs → Gates → EV computation → Regime
multipliers → Probabilities → Dossier verdict → Sizing, and grade
each row `Reconstructable` / `Partial` / `Silent`. The §2 question:
does any traced code path surface a tradeable verdict without an EV
computation upstream? (None observed — see verdict below.)

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, 20-ticker broad universe spanning seven GICS
sectors. `WheelRunner.rank_candidates_by_ev(..., top_n=50,
min_ev_dollars=-1e9, include_diagnostic_fields=True)`. Result: 12
survivors, 8 drops. Dossiers built via `engine.candidate_dossier.
build_dossiers(...)` with `FilesystemChartProvider(base_dir=
data_processed/screenshots)` (offline, no cached files — so the chart
is intentionally absent for the survivor cases, exercising R2). The
reviewer is the default `EnginePhaseReviewer(min_proceed_ev=10.0,
spot_tolerance_pct=0.02)`. Three concrete cases pulled from the run:

| Case | Ticker | Picked because |
|---|---|---|
| **A** — proceed/review | **CAT** | Highest survivor EV (`ev_dollars=290.26`, `ev_raw=639.64`). With no chart attached R2 fires → `review`; with a live current-spot chart it would be R5 → `proceed` (cf. S5/S6). |
| **B** — R1-blocked  | **NVDA** | Lowest survivor EV (`ev_dollars=−124.32`); R1 fires before chart is examined. |
| **C** — gate-dropped | **JPM**  | First entry in `.attrs["drops"]`; gate=`event`, reason=`event_lockout:earnings@2026-04-14 (±5d buffer)`. |

**Path.** Survivors carry the full diagnostic row
(`engine/wheel_runner.py` lines 1334–1411): `ticker / spot / strike /
premium / dte / iv / ev_dollars / ev_per_day / collateral / roc /
prob_profit / prob_assignment / days_to_earnings / distribution_source`
(core, always) plus 22 diagnostic fields when `include_diagnostic_fields=
True` — `ev_raw / cvar_5 / cvar_99_evt / tail_xi / heavy_tail /
omega_ratio / fair_value / edge_vs_fair / breakeven_move_pct /
total_transaction_cost / skew_pnl / dealer_regime / dealer_multiplier /
gex_total / gamma_flip_distance_pct / nearest_put_wall_strike /
nearest_call_wall_strike / skew_slope / put_skew / risk_reversal /
skew_multiplier / hmm_multiplier / hmm_regime / news_multiplier /
news_sentiment / news_n_articles / credit_multiplier / credit_regime /
strike_open_interest / chain_quality_warning`. Drops carry only
`{ticker, gate, reason}` where `reason` is free text. Dossier carries
`verdict ∈ {proceed, review, skip, blocked}`, `verdict_reason` (R-rule
ID string), and `review_notes: list[str]` (one entry per R-rule the
reviewer considered).

**Status.** Done. **Compliance verdict: PARTIAL.** For *survivor*
rows (A & B), the audit trail is defensible with one important caveat
(forward-distribution / HMM-posterior / GEX-distribution / news-articles
internals are silent — only the *summary number* and the *label* are
on the row). For *gate-dropped* rows (C), the audit trail is
**insufficient on its own**: the only artifact is a free-text reason
string, with no structured `{observed, threshold, units}` breakdown,
no EV (not computed — gate fires upstream), no multipliers, no
sizing. §2 holds — verified across the three traced cases, no
verdict path emits a tradeable outcome without an upstream
`EVEngine.evaluate` call (R1 enforces it as the first reviewer rule).

**Findings:**

- **Numbers from the run** (Bloomberg `as_of=2026-03-20`, 20 tickers,
  `include_diagnostic_fields=True`). 12 survivors, 8 drops. Survivor
  EV range `−124.32 … +290.26`. Drops: 7 × `event` (JPM/BAC/GS/XOM/UNH/
  JNJ/GE — all earnings within the ±5d lockout buffer at this `as_of`)
  + 1 × `history` (WMT, `history 70d < required 504d`). 0 × `chain_quality`,
  0 × `ev_threshold` (because `min_ev_dollars=-1e9` was set). At this
  Bloomberg `as_of` the dealer overlay is OFF on every survivor
  (`dealer_regime=None`, `dealer_multiplier=1.0`) — the provider doesn't
  supply chains, so dealer positioning isn't aggregated. News overlay
  OFF (`news_n_articles=0`, `news_multiplier=1.0`). Credit regime is
  `benign` (`credit_multiplier=1.0`). HMM is the **only active
  overlay** in this run, ranging `0.30 … 1.02` per ticker
  (`crisis`/`bear`/`normal`/`bull_quiet`).

- **Case A — CAT.** `ev_dollars=290.26`, `ev_raw=639.64`,
  `ev_dollars/ev_raw=0.4538`, exactly matching `hmm_multiplier=0.4538`
  (`hmm_regime=bear`); all other overlays at 1.0, so the composite
  multiplier identity `ev_dollars = ev_raw × Π(multipliers)` reconstructs
  cleanly. `prob_profit=0.8286`, `prob_assignment=0.1714`,
  `cvar_5=-4911.17`, `distribution_source=empirical_non_overlapping`,
  `fair_value=13.248`, `edge_vs_fair=0.0` (Bloomberg-synthetic premium is
  BSM-fair by construction — known issue per PROJECT_STATE §3.4, see also
  S1). Dossier: `verdict=review`, `verdict_reason=chart_context_missing`,
  one note (`"chart context unavailable: screenshot_not_found"`).
  Collateral `$62,550`, ROC `+0.464%` over 35 DTE.

- **Case B — NVDA.** `ev_dollars=-124.32`, `ev_raw=-139.58`. Composite
  multiplier 0.8907 = `hmm_multiplier=0.8907` (`hmm_regime=normal`); all
  others 1.0. Dossier: `verdict=blocked`, `verdict_reason=negative_ev`,
  one note (`"engine ev_dollars=-124.32 < 0 - chart cannot upgrade
  negative EV"`). R1 fires at `candidate_dossier.py:167` before the
  chart is examined — confirms S5's finding on a separate ticker.

- **Case C — JPM.** Drop entry:
  `{ticker: "JPM", gate: "event", reason: "event_lockout:earnings@
  2026-04-14 (±5d buffer)"}`. **No row in the ranker output.** No EV
  computed (event_gate short-circuits at `wheel_runner.py:1302–1310`,
  *before* the cost / regime / EV evaluation), so no multipliers, no
  probabilities, no sizing. The earnings date and the 5-day buffer are
  *in the reason string*, not in structured fields — a parser has to
  regex them out. **Logged** as the headline drop-schema gap below.

- **Per-case audit-trace grading.**

  | Layer | Case A (CAT, review) | Case B (NVDA, blocked) | Case C (JPM, gate-dropped) |
  |---|---|---|---|
  | Inputs (spot/strike/premium/dte/iv) | **Reconstructable** | **Reconstructable** | Silent (no row) |
  | Gates fired/passed | **Reconstructable** (absent from drops + `days_to_earnings=41`) | **Reconstructable** | **Partial** — `gate=event` is structured, `reason` is free text (earnings date + buffer embedded; not parsed out) |
  | EV computation (`ev_raw`, `ev_dollars`, identity) | **Reconstructable** (`ev_dollars = ev_raw × Π(multipliers)` verifies to within rounding) | **Reconstructable** | Silent (event gate short-circuits before EV) |
  | Regime multipliers (per-overlay + label) | **Reconstructable** | **Reconstructable** | Silent |
  | Probabilities (`prob_profit`, `prob_assignment`, `cvar_5`) | **Partial** — summary on row; forward distribution itself silent | **Partial** | Silent |
  | Dossier verdict (`verdict / verdict_reason / review_notes`) | **Reconstructable** (R-rule + a human-readable note per rule) | **Reconstructable** | N/A — no dossier (didn't enter ranking) |
  | Sizing (`collateral`, `roc`) | **Reconstructable** | **Reconstructable** | Silent |

- **Named silent surfaces** (the audit row has the summary number / a
  label; the *derivation* is not surfaced):

  1. **Forward-distribution posterior.** `prob_profit` /
     `prob_assignment` / `cvar_5` are on the row; the actual distribution
     (block bootstrap / HAR-RV / non-overlapping samples) and the input
     window used are not. Only the `distribution_source` *label* is
     surfaced.
  2. **HMM 4-state posterior.** `hmm_regime` is the argmax-state label
     (`crisis` / `bear` / `normal` / `bull_quiet`); the 4-vector posterior
     probabilities are not on the row, so the audit can't distinguish a
     "75/15/7/3" assignment from a "30/28/22/20" near-tie that picked the
     same label.
  3. **Dealer GEX distribution.** When the dealer overlay is on, the row
     carries `gex_total / gamma_flip_distance_pct / nearest_put_wall_strike
     / nearest_call_wall_strike`; the full per-strike GEX vector and all
     non-nearest walls are not. (Not exercised in this Bloomberg run —
     dealer overlay is off here; this is verified against PROJECT_STATE
     and a code read.)
  4. **News-sentiment article-level breakdown.** When the overlay is on,
     `news_sentiment` is a single number and `news_n_articles` is a count;
     the per-article scores, sources, and dates are not on the row.
  5. **Credit-regime mapping.** `credit_regime` is the label
     (`benign`/`stressed`/`crisis`); the credit-spread inputs and the
     policy mapping label → multiplier are not.
  6. **Volatility-surface curve.** `skew_slope` / `put_skew` /
     `risk_reversal` summarize three points; the full vol-surface curve
     is not on the row. (`distribution_source` and the SVI calibration
     surface live behind `volatility_surface.py`, dormant per DECISIONS
     D9.)

- **`.attrs["drops"]` schema is unstructured — the compliance gap.**
  Drops carry `{ticker: str, gate: str, reason: str}`. The reason string
  embeds the observed value and threshold (e.g. `"history 70d <
  required 504d"`, `"event_lockout:earnings@2026-04-14 (±5d buffer)"`,
  `"ev_dollars -39.13 < min_ev_dollars 10.00"`), but the schema does
  not pull those into discrete fields. A compliance officer asking
  "show me all candidates dropped on history within the last 30 days"
  has to regex 8 free-text strings instead of querying a structured
  log. **Logged.**

- **EV-authority identity holds on the survivor rows.** For both Case A
  (CAT, +290.26 / +639.64) and Case B (NVDA, −124.32 / −139.58), the
  composite ratio `ev_dollars / ev_raw` matches the only non-1.0
  multiplier (hmm) to 4 dp. This is exactly the auditable property §2
  exists to protect — the EV authority's input is the EV row; the row
  carries `ev_raw` pre-overlay; the multipliers and their labels are
  per-overlay; the composite identity reconstructs end-to-end. **Logged
  as a positive — the diagnostic surface DOES carry the EV-authority
  algebra in a verifiable way.**

- **R1 fires first, before the chart is even examined** — confirmed on
  Case B (NVDA, `chart_context.is_ok=False, error=screenshot_not_found`;
  dossier emitted `blocked / negative_ev` with the negative-EV note and
  *no* chart note, exactly the order described in CLAUDE.md §2 R1).
  Replicates S5's AAPL finding on a separate ticker.

- **Bloomberg `as_of=2026-03-20` is dense in `event` drops** because
  US Q1 earnings season runs early April; 7 of the 8 drops are earnings
  lockouts inside a 35-DTE window. **Logged** — choice of `as_of` shapes
  which gates exercise. A run at a non-earnings-season date would
  exercise different gates.

**§2 verdict.** Holds. Every traced path surfaces an EV-authority
verdict (R1–R6) only after `EVEngine.evaluate` ran (for survivors) or
emits a drop (for non-survivors) — no observed path emits a tradeable
verdict without upstream EV. R1 fires first in the reviewer, ahead of
chart inspection (Case B). The dossier reviewer is provably
downgrade-only: R1 → `blocked`; R2/R6 → `review`; R3/R4 → `skip`; R5
→ `proceed` *only when* `ev_dollars ≥ min_proceed_ev` AND no
downstream rule (R6) downgrades it. None of R1–R6 can upgrade. No §2
bug surfaced; no regression test added.

**AI handoff.**

- **Structured drops are the highest-leverage compliance fix.** Replace
  the free-text `reason` field with a structured record:
  `{ticker, gate, reason_code, observed: float|str, threshold: float|str,
  units: str, message: str}`. `reason_code` is the discriminator (e.g.
  `event_lockout_earnings`, `history_too_short`, `ev_below_min`,
  `chain_unavailable`); `observed`/`threshold`/`units` carry the values
  the current free text embeds; `message` keeps the human-readable line.
  Backwards-compatible — old consumers read `message`; new consumers
  query the structured fields. Not claimed.

- **HMM posterior on the diagnostic row.** Add `hmm_posterior_probs:
  list[float] | None` (length 4, summing to ~1) to the
  `include_diagnostic_fields=True` block. Small change; surfaces "label
  picked but the posterior was 0.30/0.28/0.22/0.20 — barely picked"
  which a 4dp `hmm_multiplier` can't reveal.

- **Optional — persist `EVResult` per evaluation for replay.** The
  current row is a flattened summary; the full `EVResult` (in
  `ev_engine.py`) carries fields that don't reach the row in normal mode
  (`expected_days_held`, `regime_multiplier` composed, `std_pnl`,
  `pinning_zones`, `metadata`). An opt-in `--audit-persist` mode that
  writes the per-candidate `EVResult` to `data_processed/audit/<as_of>/
  <ticker>.json` would close the "replay this trade" gap entirely.
  Out of scope here — flagged for a future Sn or human-scoped decision.

- **Ruled out per the prompt (don't litigate):** a trade-explainability
  API or compliance-export tool, multi-trade portfolio audit (S15),
  live chains (S6), persistent on-disk audit logging as a default mode,
  advisor committee output structure. The gap analysis above is the
  artifact; the build-out decisions belong to the user.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — case-by-case re-traced on the 20-ticker run at `as_of=2026-03-20`: CAT (highest EV +444.99 with `ev_raw=980.62`, `hmm_multiplier=0.4538`, `hmm_regime=bear`); NVDA (`ev_dollars=-70.76`, `ev_raw=-79.45`, `hmm_multiplier=0.8907`, `hmm_regime=normal`); JPM dropped with structured `{ticker:"JPM", gate:"event", reason:"event_lockout:earnings@2026-04-14 (±5d buffer)"}`. EV-authority identity holds: NVDA `ev_raw × hmm_multiplier = -79.45 × 0.8907 = -70.77` ≈ actual `-70.76` (rounding); CAT `980.62 × 0.4538 = 445.01` ≈ actual `444.99`.
  - Qualitative verdict: **partial — drops schema still free-text; CAT/NVDA EV magnitudes drifted post-IV-PIT-fix**. The original CAT EV was 290.26 (vs new 444.99 — +53% delta) and NVDA was -124.32 (vs new -70.76 — +43% magnitude reduction). Direction unchanged on both (CAT remains highest survivor, NVDA remains negative-EV blocked). All structured-drops findings still apply: `.attrs["drops"]` carries `{ticker, gate, reason}` but `reason` is free text.
  - Numerical drift > 5% (with attribution):
    - metric `CAT_ev_dollars[2026-03-20]`: orig `290.26` → new `444.99` (`+53.3%`); attributable to **PR #179** (`_resolve_pit_atm_iv` in `rank_candidates_by_ev`). PIT IV for CAT @ 2026-03-20 is higher than the snapshot `implied_vol_atm` per the IV history file, raising the synthetic premium and the forward-EV magnitude. HMM multiplier identical (0.4538 / bear in both runs).
    - metric `NVDA_ev_dollars[2026-03-20]`: orig `-124.32` → new `-70.76` (`-43.1%` magnitude); attributable to same PR #179 IV-PIT propagation — sign preserved; magnitude moves toward zero because the PIT IV propagation tightens the forward distribution on a name where the snapshot IV was elevated relative to recent realized.
  - Notes: the S16 audit-trace grading table (Reconstructable / Partial / Silent) still applies row-for-row on the diagnostic columns this run produced — confirmed.

---
