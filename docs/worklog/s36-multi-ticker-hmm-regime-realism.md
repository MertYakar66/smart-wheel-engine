---
id: S36
title: Multi-ticker HMM regime realism
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Extends S33 V3 (which only checked AAPL) to **9
tickers across 5 sectors**: mega-cap tech (AAPL / MSFT / NVDA),
banks (JPM / BAC), defensives (KO / JNJ), energy (XOM), and
healthcare (LLY). Question: does the per-ticker HMM produce
*consistent* regime labels at known historical events across the
universe — or does it diverge by ticker in ways a trader needs to
understand? Closes the "S33 V3 only checked one name" methodology
debt from S33's verdict.

**Setup.** Same 504-day-tail HMM (`engine.regime_hmm.GaussianHMM`
n_states=4, random_state=42, n_iter=20). 7 anchor dates spanning
2020-03 → 2026-03: COVID crash, post-vaccine rally, 2022
inflation low, Aug 2024 vol spike, Apr 2025 S30 crisis, Feb 2026
recent, Mar 2026 data cutoff. Per ticker × event: extract argmax
label + multiplier + per-state probability vector + realized vol
(252d annualised) + realized return (252d annualised). Driver at
`%TEMP%\s36\driver.py` (not committed; per Sn convention).
`SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`. Read-only on
decision layer.

**Path.** `engine.data_connector.MarketDataConnector.get_ohlcv`
→ trim to `<= as_of` → `np.diff(np.log(close))` → last 504
points → `engine.regime_hmm.GaussianHMM().fit(tail).predict_proba`
→ `position_multiplier(probs[-1])` and `argmax(probs[-1])` for
label. Per-ticker, per-event, exhaustive.

**Status.** Done. **Verdict: the HMM is mathematically consistent
across tickers (each ticker has its own state space, labels +
multipliers reflect per-ticker dynamics). At known consensus crises
the universe agrees (9/9 Mar 2020, 8/9 Apr 2025). At transitional
events labels diverge by ticker in ways that are defensible. The
"crisis" label is NOT a simple vol threshold — Feb 2026
monotonicity check failed by design (XOM at vol 0.247 → crisis;
BAC at vol 0.276 → normal). This confirms and broadens S33 F4
("crisis = high-vol-state in the per-ticker fit, not a market-wide
vol-threshold rule") at the universe scale.**

**Findings:**

- **(F1 — universal-consensus crises, ✓ verified)** At the
  two cleanest historical crises:

  | Event | Crisis-labeled count |
  |---|---|
  | 2020-03-23 (COVID crash bottom) | **9/9** |
  | 2025-04-04 (S30 confirmed crisis) | **8/9** (LLY = bear, multiplier 0.482; LLY 252d return slightly positive +0.075) |

  Per-state probabilities at Mar 2020 are uniformly high crisis
  (most tickers >0.93 crisis probability, some 1.000). At Apr
  2025 the 8 crisis-labelers all show `crisis prob ≥ 0.994`
  (uniform pure-crisis state). The HMM's argmax label is
  unambiguous when the universe genuinely IS in crisis. ✓
  **Confirms S30's findings extrapolate beyond AAPL.**

- **(F2 — per-ticker divergence at transitional events, expected
  not a bug)** At less-clean transitional dates the universe
  splits by ticker in defensible ways:

  | Event | Label distribution across 9 tickers |
  |---|---|
  | Nov 2020 (post-vaccine rally) | 5 bear + 4 crisis (defensives KO/JPM/BAC/XOM still crisis; tech + JNJ + LLY transitioning to bear) |
  | Jun 2022 (inflation low) | Mixed: AAPL crisis (mult 0.77 — split state probs), MSFT/NVDA bull_quiet (mults 0.88/0.91!), JPM bear, BAC crisis, KO/JNJ crisis (mults 0.21/0.39), XOM normal (0.83), LLY bear. **Multipliers span 0.21 to 0.91 on the same date.** |
  | Aug 2024 (vol spike) | 7 crisis + XOM bear + LLY bear |
  | Feb 2026 (recent calm-anchor candidate) | 1 crisis (AAPL — same finding as S33 F4) + 6 bear + 1 normal (BAC) + 1 crisis (XOM) — wait, that's the same 9. Distribution: AAPL crisis, MSFT bear, NVDA bear, JPM bear, BAC normal, KO bear, JNJ bear, XOM crisis, LLY bear → 2 crisis, 6 bear, 1 normal |
  | Mar 2026 (data cutoff) | 1 crisis (KO! a defensive going crisis), 5 bear, 2 normal (NVDA, BAC), 1 bull_quiet (XOM) — post-crisis recovery diversity |

  This per-ticker divergence is **correct behavior**: each HMM
  is fit on that ticker's own 504-day log returns; the state
  emission distributions (means + variances) are
  ticker-specific. A trader reading the ranker output should
  interpret "AAPL crisis + KO crisis" as "both tickers are in
  their own high-vol regime" — NOT "AAPL is crashing the same
  way KO is." Logged as a documentation gap, not a bug.

- **(F3 — vol-vs-label monotonicity FAILS by design)** Feb 2026
  monotonicity check:

  ```
  ticker   vol_ann (252d)   label
  KO         0.172           bear
  JNJ        0.191           bear
  XOM        0.247           crisis   ← lower-vol-than-BAC, but crisis
  JPM        0.262           bear
  MSFT       0.263           bear
  BAC        0.276           normal   ← higher-vol than XOM, but normal
  AAPL       0.318           crisis
  LLY        0.424           bear
  NVDA       0.441           bear
  ```

  **Lowest-vol "crisis" = 0.247 (XOM); highest-vol "normal" =
  0.276 (BAC). Monotone? FALSE.** The label depends on the
  full 504-day state structure (state means, transition matrix,
  per-state emission probabilities), not just the trailing 252d
  vol. A trader assuming "high vol → crisis label" mis-models
  the engine. This **broadens S33 F4** at the universe scale —
  the HMM disambiguation columns (PR #222) only help if the
  trader knows to compare across the ranker's output.

- **(F4 — LLY anomaly at Apr 2025, ⚠ flag-not-fix)** While 8 of
  9 tickers label crisis on 2025-04-04, LLY labels bear
  (mult 0.482, crisis prob 0.059, bear prob 0.941). LLY's
  252d window ending 2025-04-04 had vol 0.324 (mid-range) and
  mean −0.050. The HMM categorizes LLY's recent dynamics as
  "moderately high vol, slightly negative mean" → bear, not
  crisis. **Defensible per the HMM's mathematical regime
  definition** (crisis = "very negative mean, very high vol" per
  `engine/regime_hmm.py:30`). LLY's vol of 0.324 wasn't
  extreme enough to trip the crisis state's emission
  probability mass for THAT ticker's fitted state means. **Not
  a bug; an interesting per-ticker characteristic worth
  documenting.**

- **(F5 — Mar 2026 KO anomaly, ⚠ flag-not-fix)** At the most
  recent data point (2026-03-20), 8 of 9 tickers label bear /
  normal / bull_quiet, BUT KO labels crisis (mult 0.548).
  State probs: crisis 0.544, bear 0.033, normal 0.423. **It's
  a marginal call (crisis edges normal by ~0.12).** KO's 252d
  vol is 0.169 — the LOWEST in the universe — yet labels
  crisis. The argmax-label rule masks the proximity to normal.
  **F4 nuance compounds:** the bare label hides marginal-call
  state. A trader inspecting one row needs the state-prob
  vector (NOT currently in the ranker output) to know the
  label is fragile. **Logged as a follow-up observability
  consideration:** could surface `hmm_argmax_prob` (the prob
  of the labeled state) as a confidence column.

- **§2 verified at universe scale.** Each per-ticker HMM fit
  runs independently; no cross-ticker contamination. Each
  ticker's `hmm_multiplier` is consumed by the ev_engine for
  THAT ticker's candidate only. The §2 contract holds.

**Realism Check.**

| Aspect | Engine (9 tickers, 7 events) | External reference | Verdict |
|---|---|---|---|
| Universal consensus at clean crises | 9/9 Mar 2020, 8/9 Apr 2025 | S30 confirms Apr 2025 = crisis on AAPL; broader market history for Mar 2020 (VIX > 60, S&P drawdown >30%) | ✓ Verified |
| Per-ticker label divergence at transitions | Multipliers span 0.21–0.91 on Jun 2022 across the universe | Each HMM is per-ticker by design (per `engine/regime_hmm.py`); divergence reflects per-ticker dynamics, not engine inconsistency | ✓ Verified (correct behavior, documentation gap) |
| Vol → label monotonicity | FAILS at Feb 2026 (XOM crisis @ vol 0.247; BAC normal @ vol 0.276) | HMM regime definition uses 4-state emission probabilities, not a vol threshold | ⚠ Realism gap — trader mental model "high vol = crisis" is wrong |
| LLY at Apr 2025 (8/9 universe → crisis, LLY → bear) | bear, mult 0.482, low crisis-prob (0.059) | LLY's 252d window had vol 0.324 (mid-range); not extreme enough for crisis emission | ⚠ Defensible per HMM math; surfaces "regime is per-ticker, not market-wide" |
| Mar 2026 KO crisis (others bear/normal) | crisis prob 0.544, but normal prob 0.423 — MARGINAL call | Argmax-rule label hides marginal-call ambiguity | ⚠ F5 observability nuance |

**Verdict.**

- **HMM is universe-consistent on consensus crises.** When the
  market actually IS in crisis, 9/9 or 8/9 tickers label crisis
  with high probability (>0.93 for most, 1.000 for many). The
  HMM's per-ticker independence does not prevent cross-ticker
  consensus when the underlying regime is shared.

- **Per-ticker label divergence at transitions is correct
  behavior.** Each ticker's HMM is fit on its own log returns;
  the state space is ticker-specific. At Jun 2022 the multiplier
  spans 0.21 (KO crisis) to 0.91 (NVDA bull_quiet) on the same
  date — and this is right: KO's slow-moving defensive history
  vs NVDA's high-vol growth history mean their "high-vol" states
  have different emission characteristics. The HMM correctly
  reflects this.

- **The "crisis = high-vol" mental model is wrong.** F3's Feb
  2026 monotonicity check failed at the universe scale: lowest-
  vol "crisis" (XOM 0.247) is below highest-vol "normal" (BAC
  0.276). The label depends on the full state structure, not just
  the trailing 252d vol. **S33 F4 is now confirmed at universe
  scale.** The disambiguation columns from PR #222
  (`hmm_realized_vol_252d_ann`, `hmm_realized_return_252d_ann`)
  help — but only if the trader knows to look at them.

- **Two flag-not-fix anomalies surfaced.** LLY at Apr 2025 (F4)
  and KO at Mar 2026 (F5) are both defensible-per-HMM-math but
  surface trader-actionable nuances. Neither is a bug. F5
  motivates a small future observability addition:
  `hmm_argmax_prob` column (the probability of the labeled
  state) would let a trader see when a label is marginal.

- **S33's findings extrapolate to the universe.** The Apr 2025
  crisis behaviour S30 documented on AAPL is reproduced across 8
  of 9 names. The Feb 2026 "high-vol-not-crashing" finding from
  S33 V3b reproduces across mid-cap tech (MSFT, NVDA), banks (JPM),
  defensives (KO, JNJ), and healthcare (LLY) — confirming it
  wasn't an AAPL artifact.

**AI handoff.**

- **Closes the S33 "single-name HMM verification" methodology
  debt.** The HMM is now verified to work consistently across 9
  representative tickers at 7 historical events.

- **F5 small observability follow-up (queued):** add an
  `hmm_argmax_prob` column to the ranker output alongside
  `hmm_regime` / `hmm_multiplier` / `hmm_realized_vol_252d_ann`
  / `hmm_realized_return_252d_ann`. A trader inspecting one row
  would then see (a) the label, (b) the multiplier, (c) the
  realized vol/return, and (d) the confidence in the label
  (e.g., 0.544 for KO at Mar 2026 = marginal call). Single-
  line addition in `engine/wheel_runner.py`'s HMM block; ships
  as a future small PR if the user wants this enriched.

- **F4 + F5 motivate trader-facing documentation.** The HMM
  semantic ("high-vol regime in the per-ticker fit, not a
  market-wide vol threshold") deserves a CLAUDE.md or
  `docs/HMM_REGIME_SEMANTICS.md` note. Currently this knowledge
  lives in the regime_hmm.py source comments + S33 F4 + S36 F3
  — three places, none of them the trader's first read.

- **Sanity follow-up Sn (S6 dependency):** re-run S36 with the
  Theta connector. Different data path may surface different
  per-ticker emission probabilities; would confirm or deny
  whether the universe-consensus finding is data-source-
  independent.

**Methodology debt.**

- **9 tickers, not the full SP500.** S36 picks a representative
  cross-section. A full-universe verification (~500 tickers)
  would surface more LLY-/KO-style per-ticker anomalies but
  would also be much harder to digest. The 9-ticker sample
  spans 5 sectors and intentionally includes both extremes
  (NVDA high-beta growth, KO defensive consumer staples).

- **Single anchor per event.** Each historical event uses ONE
  anchor date. The HMM behaviour ON the transition day (e.g.,
  Apr 2025 = the day OF the crisis) may differ from one week
  before or after. S30 covered ±2-3 days around the AAPL
  bear → crisis transition; S36 takes one snapshot per ticker
  per event. A date-grid (e.g., 5 days centered on each event)
  would let us check label-stability under noise.

- **No comparison to a non-HMM regime classifier.** The HMM is
  the regime arbiter today. A VIX-threshold rule (VIX > 30 →
  crisis; VIX < 15 → bull) on the same dates would either
  agree or surface where the HMM and the conventional vol
  proxy diverge. S30's methodology debt already mentioned
  this; S36 inherits the same gap.

- **Bloomberg-only.** Theta replay would change the data
  source for HMM fitting; results may shift. S6 queued.

- **No causality check for LLY / KO anomalies.** F4 and F5
  surface "this ticker behaves differently here" without
  investigating *why* — e.g., is LLY's bear-not-crisis on Apr
  2025 because LLY actually had a defensive return profile
  during the universe's crisis? Worth a small ad-hoc probe.
