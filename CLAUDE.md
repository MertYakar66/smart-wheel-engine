# CLAUDE.md — Session orientation for Smart Wheel Engine

You are working on a probabilistic expected-value (EV) decision engine for
wheel strategies (short cash-secured puts → covered calls) on S&P 500 names.
This file is the structural contract: how the product thinks. Read it in full.

For machine/environment rehydration (cloning, Theta Terminal, regenerating
local data), see `docs/LAPTOP_SETUP.md`. For everything else, the pointer
block at the bottom routes you to the right on-demand document.

---

## 1. Mental model — four layers

1. **Data layer** (`data/`, `data_processed/`, `scripts/pull_*.py`) — OHLCV,
   option chains, IV, fundamentals, macro, news. Two providers, selected by
   `SWE_DATA_PROVIDER` (default `bloomberg`). Full capability matrix in
   `docs/DATA_POLICY.md` §2.
2. **Quant layer** (`engine/`) — Black-Scholes-Merton pricing + Greeks to
   3rd order, empirical forward distributions (non-overlapping → block
   bootstrap → HAR-RV cascade), POT-GPD tail risk, 4-state Gaussian HMM
   regime, Nelson-Siegel skew, Student-t copula CVaR, dealer GEX / walls /
   gamma-flip. SVI surface tooling exists but is dormant — see
   `DECISIONS.md` D9.
3. **Decision layer — the authoritative ranker.**
   - `engine/ev_engine.py` — `EVEngine.evaluate`. Runs event lockout →
     forward distribution → cost model → regime & dealer multipliers →
     returns `EVResult`.
   - `engine/wheel_runner.py` — `WheelRunner.rank_candidates_by_ev`. The
     one ranker every tradeable path routes through.
   - `engine/candidate_dossier.py` — `EnginePhaseReviewer` applies the
     downgrade rules R1–R9 (see §2).
4. **Interface layer** — `engine_api.py` (HTTP on `:8787`), `dashboard/`
   (Next.js), `engine/tradingview_bridge.py` (chart providers — sanity
   check, not a decider), `advisors/` (Buffett / Munger / Simons / Taleb
   committee, advisory only), local Ollama for memos.

---

## 2. The hard invariant — never break this

**No tradeable candidate bypasses `EVEngine.evaluate`.**

Chart providers, news sentiment, the advisor committee, the dealer
positioning overlay, and the TradingView bridge can all **downgrade** a
verdict (negative → blocked, proceed → review, proceed → skip). None of
them can **rescue** a negative-EV trade. The dealer multiplier
(`dealer_regime_multiplier` in `engine/dealer_positioning.py`) is clamped
to `[0.70, 1.05]` and only scales the final `ev_dollars` — it never
touches `ev_raw`.

If you add a new input (new data source, new advisor, a TradingView MCP
chart feed), wire it as a participant in a chained provider or as a
downgrade-only reviewer. Do not introduce a code path that converts a
non-tradeable candidate into a tradeable one without a fresh
`EVEngine.evaluate` call.

The `EnginePhaseReviewer` rules, for reference:

- R1: negative OR non-finite EV → blocked (hard stop; **R1a** at
  `engine/candidate_dossier.py` guards `+inf` / `-inf` / `NaN` via
  `math.isfinite` *before* the negative check, returning
  `verdict_reason="ev_non_finite"` distinct from `"negative_ev"` so the
  audit trail tells an unparseable engine value apart from an evaluated
  loss — see PR #204)
- R2: chart missing → review
- R3: spot mismatch > 2% → skip
- R4: phase contradiction → skip *(conditional/reserved — the rule is
  implemented and unit-tested but dormant in the production path: no
  current chart provider populates `visible_indicators['phase']` (empty
  through M1) and the ranker emits no `phase` on `ev_row`. It fires only
  when a phase-aware chart provider lands — see
  `docs/TRADINGVIEW_INTEGRATION.md`. Not a live downgrade today.)*
- R5: EV above threshold → proceed (below → review)
- R6: short-gamma regime + strike at/above put wall, or dealer regime
  near gamma flip → downgrade to review
- R7: *(D17 soft-warn — conditional on attached `PortfolioContext`)*
  portfolio VaR_95 (30-day horizon) above `max_var_pct × NAV` (default
  5%) → downgrade proceed → review. Skips silently when no context is
  attached OR when `check_var` lacks correlation/returns data — soft-warns
  don't fire on absent evidence.
- R8: *(D17 soft-warn — conditional on attached `PortfolioContext`,
  two trigger conditions mirroring R6.)* Either the C4 vol-spike stress
  drawdown > 8% NAV OR the candidate's underlying is in
  `short_gamma_amplifying` regime → downgrade proceed → review. Distinct
  `verdict_reason` per trigger.
- R9: *(D17 soft-warn — conditional on attached `PortfolioContext`,
  added in B2 closure.)* Sector concentration cap. If opening the
  candidate would push its GICS sector over `max_sector_pct × NAV`
  (default 25% per the D17 sector cap; same gate
  `engine.portfolio_risk_gates.check_sector_cap` the tracker applies
  as a HARD refusal at `open_short_put` time when
  `require_ev_authority=True`), downgrade proceed → review with
  `verdict_reason="sector_cap_breach"`. Skips silently when no
  context is attached OR `nav == 0` — soft-warns don't fire on
  absent evidence.

---

## 3. NEVER — out of scope by design (get explicit consent first)

- **Never bypass `EVEngine.evaluate`.** No code path converts a
  non-tradeable candidate into a tradeable one without a fresh evaluate
  call (§2).
- **Never weaken the downgrade-only reviewer contract.** Reviewers
  downgrade; they never upgrade.
- **Never alter the dealer multiplier clamp `[0.70, 1.05]`** — asymmetric
  by design.
- **No broker / OMS / order routing surface.** The engine produces ranked
  candidates and memos.
- **No tick-level order flow / sub-minute features.** Theta v3 does not
  expose realtime stock quotes at this tier.
- **No non-S&P-500 universe and no non-wheel strategies.** Short puts +
  covered calls + timing-gated strangles only.
- **Branch + PR for every change.** Never commit to `main` directly.

---

## 4. Fresh-session bring-up

1. Confirm the provider. Default is `bloomberg`; verify with:

   ```python
   from engine.wheel_runner import WheelRunner
   print(type(WheelRunner().connector).__name__)
   ```

   Expect `MarketDataConnector` in a Cowork sandbox. **Always log which
   provider was actually selected** — silent provider selection is a
   recurring bug source.

2. Sanity-check the data layer with the 5-ticker smoke test:

   ```python
   from engine.wheel_runner import WheelRunner
   df = WheelRunner().rank_candidates_by_ev(
       tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
       top_n=10, min_ev_dollars=-1e9,
       include_diagnostic_fields=True,
   )
   ```

   Runs in ~2 s. Five rows with non-null `ev_dollars`, `iv`, and `premium`
   means the Bloomberg CSVs + connector + EV engine path is healthy.

3. For any change touching the decision layer (`ev_engine.py`,
   `wheel_runner.py`, `candidate_dossier.py`): run the full test suite
   (`pytest tests/ -v`), not just the targeted file. Invariants are
   cross-cutting.

For sandbox-vs-laptop capability differences (what works in Cowork vs.
needs Theta Terminal), see `docs/DATA_POLICY.md` §7.

---

## 5. Where to look next

| For… | Read |
|---|---|
| Onboarding contract for any AI agent | `AGENTS.md` |
| What is authoritative / WIP / deprecated right now | `PROJECT_STATE.md` |
| Per-module purpose + decision-layer role | `MODULE_INDEX.md` |
| Test taxonomy + launch-blocker subset | `TESTING.md` |
| Every tracked file by purpose — grep, don't read | `FILE_MANIFEST.md` |
| Data layer (tiers, provider matrix, refresh, sandbox caveats) | `docs/DATA_POLICY.md` |
| Launch-blocker invariants before merging | `docs/LAUNCH_READINESS.md` |
| Greek units (canonical) | `docs/GREEKS_UNIT_CONTRACT.md` |
| TradingView wiring (engine bridge + analyst workspace) | `docs/TRADINGVIEW_INTEGRATION.md` |
| Commit / PR format | `COMMIT_GUIDE.md` |
| *Why* a structural choice was made | `DECISIONS.md` |
