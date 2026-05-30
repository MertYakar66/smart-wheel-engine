---
id: S29
title: Skew data absence on Bloomberg connector
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Realism counterpart to S28, but on the *negative-result*
side: prove that the engine's skew-dynamics machinery is structurally
**dormant** on the Bloomberg connector, despite emitting skew columns
in the ranker output. Wheel-trader pain point: a steep put skew is a
well-known risk-off signal — "sell less premium when the put skew
steepens." The engine has `engine/skew_dynamics.py` (Nelson-Siegel
term structure, `skew_slope`, `ivs_dislocation_score`) and surfaces
`skew_multiplier`, `skew_slope`, `put_skew`, `skew_pnl` in the
diagnostic columns of `rank_candidates_by_ev`. S29 asks: is any of
this actually computed on Bloomberg data, or is it cosmetic?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
five-name probe spanning high- and low-beta sectors at
`as_of=2026-03-20`: TSLA, NVDA, KO, AAPL, XOM. Four orthogonal
checks in one driver:

1. **Raw-file scan.** Re-read `sp500_vol_iv_full.csv` (81.2 MB,
   1,361,615 rows, 503 tickers, 2015-01-02 → 2026-03-20) and
   measure `hist_put_imp_vol - hist_call_imp_vol` over every
   row with both columns populated.
2. **`_resolve_pit_atm_iv` empirical check.** Call the helper
   (`engine/wheel_runner.py:147`, the Fix #1 IV-PIT resolver)
   on 10 names and compare `(put_iv + call_iv) / 2` to the
   returned value plus the put / call inputs.
3. **`skew_mult` runtime check.** Run `rank_candidates_by_ev`
   on the 5-name probe and inspect the diagnostic columns
   `skew_multiplier`, `skew_slope`, `put_skew`, `skew_pnl`.
4. **Caller audit on `engine/skew_dynamics.py`.** Static scan of
   `engine/*.py` for imports of `skew_slope`,
   `NelsonSiegelTermStructure`, `ivs_dislocation_score`.

Driver under `%TEMP%\s28\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]]. Read-only on decision
layer; no engine code changes.

**Path.** `_resolve_pit_atm_iv` at `engine/wheel_runner.py:147`
(consumes `conn.get_iv_history` → averages `hist_put_imp_vol` +
`hist_call_imp_vol`). The `skew_mult` block at
`engine/wheel_runner.py:1252-1285` (consumes `chain_df` from
`conn.get_options(ticker)` if present — otherwise stays at 1.0).
The skew machinery at `engine/skew_dynamics.py` (`skew_slope`,
`NelsonSiegelTermStructure`, `ivs_dislocation_score`).

**Status.** Done. **Verdict: skew is structurally dormant on the
Bloomberg connector for two independent reasons (data + chain
access), and the dormancy is invisible to a trader reading the
ranker output. The skew columns surface in the output but carry
either a constant (`skew_multiplier=1.0`) or are silently empty
(`skew_slope`, `put_skew`).**

**Findings:**

- **(F1 — data) The Bloomberg IV file has zero put/call
  asymmetry across its entire history.**

  ```
  total rows:                                      1,361,615
  rows with both put_iv & call_iv populated:       1,353,901
  rows where put_iv == call_iv EXACTLY:            1,353,901  (100.0%)
  gap (put - call):  mean = 0.000000   std = 0.000000   abs.max = 0.000000
  ```

  Across 503 tickers, 11 years of data. The two columns
  (`hist_put_imp_vol` / `hist_call_imp_vol` in
  `data/bloomberg/sp500_vol_iv_full.csv`) are mechanically
  identical — almost certainly the result of the Bloomberg
  extraction populating both columns from a single ATM IV
  field (likely `IVOL_MONEYNESS_DAYS_50_30` or equivalent),
  not from genuine ATM put-vs-call quotes. **The file claims
  to carry skew data; it does not. Logged.**

- **(F2 — degenerate Fix #1) `_resolve_pit_atm_iv` averaging
  is a provable no-op on this data.** Sample at
  `as_of=2026-03-20`:

  ```
  ticker    put_iv   call_iv     avg   resolved   matches?
  TSLA      46.99     46.99   0.4699   0.4699      YES
  NVDA      40.23     40.23   0.4023   0.4023      YES
  META      40.16     40.16   0.4016   0.4016      YES
  AMZN      40.25     40.25   0.4025   0.4025      YES
  KO        23.86     23.86   0.2386   0.2386      YES
  PG        25.90     25.90   0.2590   0.2590      YES
  JNJ       27.70     27.70   0.2770   0.2770      YES
  XOM       32.16     32.16   0.3216   0.3216      YES
  VZ        28.20     28.20   0.2820   0.2820      YES
  AAPL      30.79     30.79   0.3079   0.3079      YES
  ```

  The `(put_iv + call_iv) / 2` averaging in `_resolve_pit_atm_iv`
  is `X / X = X`. Fix #1 ([PR #179](https://github.com/MertYakar66/smart-wheel-engine/pull/179))
  chose the composite "average put and call" because that was the
  conservative call when the file *might* have asymmetry; on
  Bloomberg it provably doesn't, so the choice is harmless but
  also informationally vacuous. **Logged — Fix #1 is correct;
  the data layer is the binding constraint.**

- **(F3 — runtime) `skew_mult` is uniformly 1.0 in the ranker
  output because the Bloomberg connector lacks chain access.**
  Probe (5 names, 25-delta, 35-DTE puts at `as_of=2026-03-20`):

  ```
  hasattr(conn, 'get_options'):       False
  hasattr(conn, 'get_option_chain'):  False

  skew_multiplier:  unique values = [1.0]   (n_unique=1)
  skew_slope:       unique values = []      (n_unique=0)
  put_skew:         unique values = []      (n_unique=0)
  ```

  The `skew_mult` block at `engine/wheel_runner.py:1252-1285`
  is gated:

  ```python
  if use_skew_dynamics and chain_df is not None and len(chain_df) > 0:
      ...  # compute iv_25p, iv_atm, iv_25c from chain; skew_slope; skew_mult
  ```

  `chain_df` is sourced via `conn.get_options()` /
  `conn.get_option_chain()` at lines 1187-1193; the Bloomberg
  connector has neither method, so `chain_df = None`, the block
  is bypassed, and `skew_mult` stays at its initialised 1.0.
  The `skew_slope` and `put_skew` diagnostic columns are
  silently empty — not a "skew is neutral" signal but a "skew
  was not measured at all" non-signal. **Logged.**

- **(F4 — code museum) Skew-machinery callers in
  `engine/`.** Static caller scan of
  `engine/*.py` for the public symbols of
  `engine/skew_dynamics.py`:

  ```
  theta_connector.py        → imports skew_slope (1 occurrence)
  volatility_surface.py     → imports skew_slope (6 occurrences)
  wheel_runner.py           → imports skew_slope (6 occurrences, all inside chain_df branch)
  ```

  And the two larger surfaces:

  ```
  NelsonSiegelTermStructure → zero callers in engine/
  ivs_dislocation_score     → zero callers in engine/
  ```

  `wheel_runner.py`'s 6 occurrences all sit inside the
  `chain_df is not None` branch (F3 — dormant on Bloomberg).
  `theta_connector.py` and `volatility_surface.py` use it but
  per CLAUDE.md D9, `volatility_surface.py` is "dormant"
  (SVI tooling exists but is not wired into the live ranker
  on either connector). **The Nelson-Siegel term-structure
  fitter and the dislocation-score module are codified but
  unreachable from any live path.** Logged.

- **(F5 — observability, the trader-facing surface) The
  ranker output emits skew columns even when dormant.**
  Diagnostic columns `skew_multiplier=1.0`, `skew_slope=∅`,
  `put_skew=∅`, `skew_pnl=non-zero` (a separate concept;
  see below) appear in every output row of
  `rank_candidates_by_ev`. A trader inspecting the output
  would reasonably conclude the skew dynamics module is alive
  and contributing a neutral signal — the truth is the module
  never ran on Bloomberg. Same observability shape as S22 F1
  (`drops` missing on `suggest_rolls`) and S28 F2
  (`expected_dividend` populating regardless of the gate).
  **Logged.**

- **§2 verified.** `rank_candidates_by_ev` routes every
  candidate through `EVEngine.evaluate` regardless of skew
  status. The skew multiplier is a multiplicative scalar
  bounded above 0 — it cannot rescue a negative-EV trade
  even if it were active. The dormancy reduces the engine's
  signal richness on Bloomberg, but does not break the
  invariant. **Logged as a positive.**

- **(Observation, not a bug) `skew_pnl` is populated and
  is a separate concept.** Unique values in the probe were
  non-zero (e.g. `-3.843`, `-3.014`). This is *not*
  `skew_dynamics.skew_slope` — it is the empirical-forward
  cost-of-skew adjustment applied inside the EV
  computation, distinct from the multiplicative `skew_mult`.
  Not part of the dormancy finding. **Logged for clarity.**

**Realism Check.**

| Aspect | Engine (Bloomberg) | Real-market behaviour | Verdict |
|---|---|---|---|
| put_iv vs call_iv asymmetry in the IV file | identical on 100% of 1.35M rows across 503 tickers | Real markets carry meaningful put-call skew, especially on high-beta names (TSLA, NVDA) and risk-off sectors | ❌ Data missing entirely |
| `_resolve_pit_atm_iv` resolved IV (post Fix #1) | average of put + call = put_iv (since they're equal) | Average of put-IV and call-IV is meaningfully different when real skew exists | ⚠ Mechanically correct; informationally vacuous |
| `skew_multiplier` in `rank_candidates_by_ev` output | 1.0 across all rows on Bloomberg | Real put skew should pull `skew_mult` < 1.0 on risk-off names; mild call skew (rare) > 1.0 | ❌ Always 1.0 — block bypassed (no chain access) |
| `skew_slope`, `put_skew` diagnostic columns | empty (`n_unique=0`) | Should populate with the 25Δ-put-vs-25Δ-call slope when skew dynamics is alive | ⚠ Silently empty (looks like working machinery, isn't) |
| `engine/skew_dynamics.NelsonSiegelTermStructure` live callers in `engine/` | zero | Term-structure fitter would inform 1d/1w/1m IV regime classification | ⚠ Codified, unreachable |
| `engine/skew_dynamics.ivs_dislocation_score` live callers in `engine/` | zero | Dislocation score would flag VRP / gamma mispricing opportunities | ⚠ Codified, unreachable |

**Verdict.**

- **Skew is dormant on Bloomberg for two independent
  reasons, either of which would suffice to disable the
  signal.** (1) the IV file's apparent put/call columns are
  duplicates, so even a put-vs-call gap signal cannot be
  read from the data layer. (2) the `skew_mult` block needs
  a per-strike chain (delta + iv per leg) that the
  Bloomberg connector does not expose. Fixing one without
  the other still leaves the multiplier at 1.0.

- **The ranker output mis-implies activity.** Surfaced
  columns `skew_multiplier`, `skew_slope`, `put_skew` look
  like the live signals of a working skew module. They are
  not. The trader sees `skew_multiplier=1.0` and concludes
  "skew is neutral right now," when the honest reading is
  "skew was not measured at all." This is **not** a §2
  authority breach (the multiplier cannot rescue a
  negative-EV trade) but it is a **realism breach** —
  the engine claims more signal than it has.

- **The skew machinery itself is well-built and unused.**
  `engine/skew_dynamics.py` is a clean implementation of
  Nelson-Siegel term-structure fitting + 25Δ-put-vs-25Δ-call
  slope + dislocation scoring. None of it is reachable from
  the live ranker on Bloomberg, and the two largest
  surfaces (`NelsonSiegelTermStructure`,
  `ivs_dislocation_score`) have zero callers in
  `engine/*.py`. This is the same pattern as
  `volatility_surface.py` per CLAUDE.md D9 (SVI tooling
  "exists but is dormant").

**AI handoff.**

- **Fix #1 (smallest scope, observability):** when the
  `skew_mult` block does not execute, the diagnostic
  columns should reflect that. Either drop the columns
  from the output (cleaner but breaks downstream schema)
  or surface a `skew_source` column with values
  `"chain"` / `"unavailable"` so the trader can tell
  the difference between "measured neutral" and "not
  measured at all." Mirrors the [[realism-check-pattern]]
  precedent of provenance columns (S1B `oi_source`,
  `premium_source` from PR #160).

- **Fix #2 (data layer, harder):** when the connector
  carries a chain (Theta), the `skew_mult` block works
  as-built. When it doesn't (Bloomberg today), there is
  no readily-available fallback — the IV file's put/call
  columns are identical, and there is no per-strike data
  to compute a 25Δ slope. Two paths forward:
  - **Provider migration** — switch the live ranker to
    use Theta on demand for the skew computation, falling
    back to `skew_mult=1.0` when Theta is unavailable.
    Cleaner; matches the architecture per
    [[bloomberg-data-refresh-blocked]].
  - **Term-structure proxy** — use the existing IV-vintage
    columns (`hist_put_imp_vol`, `volatility_30d`,
    `volatility_60d`, `volatility_90d`, `volatility_260d`)
    to fit a Nelson-Siegel curve and back out an
    implied-skew proxy from the curvature parameter. Less
    direct than per-strike skew, but reachable from
    Bloomberg data. Would wire up the dormant
    `NelsonSiegelTermStructure` to a live consumer.

- **Fix #3 (re-extract the Bloomberg IV file):** if the
  Bloomberg side has actual put-skew and call-skew quotes
  for the names we cover (e.g. via the
  `IV_MONEYNESS_*_PUT` and `IV_MONEYNESS_*_CALL`
  field families), re-pulling them into the
  `sp500_vol_iv_full.csv` extraction would close F1. This
  is blocked the same way [[bloomberg-data-refresh-blocked]]
  blocks the other refreshes — needs the user's BQL
  queries.

- **Sanity follow-up Sn:** once Fix #1 (observability) is in,
  re-run S29 on a Theta replay (queued S6) to verify
  `skew_source="chain"` populates and the `skew_multiplier`
  actually moves below 1.0 on names with steep put skew
  (TSLA / NVDA expected; KO / PG should stay near 1.0).
  That would close the "is the skew code mechanically
  working?" question separately from "is the data
  available?"

- **Dealer positioning (S31 candidate, deferred this
  cycle):** the dealer-regime path has the *same*
  Bloomberg-chain-absence dependency as skew here (no
  per-strike gamma exposure data → synthetic `GEX`
  reconstruction). The user prompt flagged it as a
  contract-test only. Worth a small Sn after a Theta
  replay, paired with skew on the same data.

**Methodology debt.**

- **Single connector (Bloomberg).** The S29 verdict ("skew
  dormant") applies only to the Bloomberg connector. The
  Theta provider has chain access and presumably activates
  the `skew_mult` block. A Theta-replay Sn would let us
  confirm or deny that the *implementation* is correct,
  separately from the *data* gap covered here.

- **Did not test `use_skew_dynamics=False` vs True.** The
  driver defaulted to `use_skew_dynamics=True` (the
  ranker's default). Since `skew_mult` stays at 1.0
  regardless on Bloomberg, toggling the flag would not
  change the EV — but documenting the no-op would
  make the dormancy explicit in the kwarg surface.

- **No coverage-by-ticker breakdown.** We measured
  put_iv == call_iv at file scale (100%) but did not
  ask whether the "skew gap is zero" pattern is
  identical across all 503 tickers vs different ones at
  different vintages. A per-ticker time-series of the
  gap would confirm "this is a single extraction bug,
  not 503 individual ticker bugs," which matters if a
  fix is scoped per-ticker vs file-wide.
