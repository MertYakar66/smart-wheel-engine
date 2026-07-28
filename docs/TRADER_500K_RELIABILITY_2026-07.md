# TRADER-500K Reliability Campaign — Final Report (2026-07)

_A $500,000 simulated portfolio traded **only** on the engine's ranked output (fully mechanical: weekly rank, top-20 execution slice, EV>0, max 5 opens/week, 1 contract/name, R10 armed, hold-to-expiry, engine-selected covered calls after assignment), replayed over eleven overlapping 18-month windows (2020-01 → 2026-06) at three friction levels, plus two re-runs with real Theta market premiums. Coordination + full audit trail: issue #517. Bundles: `docs/verification_artifacts/trader500k/` (13 bundles; ~180k Phase-A/B forecast-outcome pairs analyzed, rail provenance separate). Method per `SIM_200K_RELIABILITY` / I1-calibration precedent; driver `backtests/regression/trader500k_campaign.py`._

## Headline verdicts

| Question | Verdict |
|---|---|
| Is `prob_profit` honest? | ⚠ **Compressed, regime-dependent**: pooled Brier 0.161 / ECE 4.3pp; low bins under-confident (+10 to +23pp), top bin over-confident (−6.8pp pooled). Worst NOT in labeled bears — see the transition finding. |
| Does higher `ev_dollars` mean better outcomes? | ⚠ **Only in high-dispersion regimes.** Pooled Spearman +0.02 (cluster-bootstrap 95% CI −0.02..+0.06 — statistically zero). Crisis rows +0.15; quiet-bull rows **−0.12**. `prob_profit` ordering stays positive everywhere (+0.15..+0.18). |
| Does the $500k book make money? | ✅ 9/11 windows positive at full friction (mean +15.4%/18mo, median +15.0%). |
| Does it beat the market? | ✗ Only 2/11 vs SPX (mean +23.0%): the 2022 mid-decline entry (W05, +12pp over) and W11. Structural bull lag, shallower drawdowns (mean maxDD 16.4% vs deeper index troughs). A defensive income profile, not a market-beater. |
| Do the risk gates work? | ✅ R10 fired 2–46×/window, peaking exactly under 2025 concentration pressure; event gate filtered ~46% of name-weeks as designed; zero §2 violations. |
| Is the sim trustworthy? | ✅ Deterministic (byte-identical re-run verified), PIT-guarded, SHA-fingerprinted; one NaN-mark bug found and guarded mid-campaign (counted carry-forward, 3 window-days affected). |

## 1. Economics (full friction)

| W | Span | Engine | SPX (ex-div) | EW passive | maxDD | Sharpe |
|---|---|---|---|---|---|---|
| W01 | 2020-01→21-06 | +29.0% | +31.9% | +38.3% | 30.8% | 0.63 |
| W02 | 2020-07→21-12 | +50.2% | +53.0% | +62.6% | 8.0% | 1.85 |
| W03 | 2021-01→22-06 | −0.7% | +2.3% | +6.3% | 23.1% | −0.12 |
| W04 | 2021-07→22-12 | −13.5% | −11.1% | −2.9% | 23.3% | −0.44 |
| W05 | 2022-01→23-06 | **+4.8%** | −7.2% | −2.9% | 18.1% | 0.08 |
| W06 | 2022-07→23-12 | +15.4% | +24.7% | +29.7% | 14.1% | 0.43 |
| W07 | 2023-01→24-06 | +12.1% | +42.8% | +38.1% | 10.1% | 0.49 |
| W08 | 2023-07→24-12 | +10.0% | +32.0% | +30.3% | 6.4% | 0.30 |
| W09 | 2024-01→25-06 | +15.0% | +30.8% | +25.5% | 20.0% | 0.40 |
| W10 | 2024-07→25-12 | +16.7% | +25.0% | +25.1% | 15.7% | 0.49 |
| W11 | 2025-01→26-06 | +30.9% | +29.2%† | +40.5% | 11.3% | 0.94 |

†coverage_truncated 2026-06-04. Benchmarks are price-return ex-dividend (add ~1.5%/yr for TR-approx). Windows overlap — not independent samples.

- **Bear economics are entry-timing-dependent, not regime-dependent**: W05 (entered mid-decline, harvested the recovery) beat SPX by +12pp; W04 (entered at the top, sold into the grind-down, ended at the bottom) is the worst window at −13.5%, underperforming both benchmarks. "Bears are good for premium selling" is false as stated; "entering after vol has repriced is good" is what the data supports.
- **Structural bull lag**: worst W07 (+12.1% vs SPX +42.8%) in the mega-cap-led tape — a diversified, capped-upside book cannot follow a concentrated rally.
- **Friction, measured cleanly** (1,212 shared entry×ticker×strike pairs): −$27/trade bid-ask, −$33/trade full (median −$22/−$26) — modest per-trade; book-level friction ordering is scrambled by path effects (documented in W05 pilot notes).

## 2. Calibration — the core reliability result (180,382 rows)

Pooled reliability (engine_exact attribution, Wilson 95% CIs):

| Forecast bin | n | Predicted | Observed | Gap |
|---|---|---|---|---|
| [0.5,0.6) | 726 | 0.564 | 0.792 | **+22.8pp** under-confident |
| [0.6,0.7) | 17,993 | 0.665 | 0.773 | +10.8pp |
| [0.7,0.8) | 66,122 | 0.750 | 0.790 | +3.9pp |
| [0.8,0.9) | 80,961 | 0.837 | 0.812 | −2.5pp |
| [0.9,1.0] | 14,558 | 0.922 | 0.854 | **−6.8pp** over-confident |

The engine's probabilities are **compressed**: realized outcomes cluster ~0.77–0.85 regardless of forecast. Ordering information is real; the *magnitudes* at both extremes are not to be taken literally.

**The transition finding (revises the window-level story).** Window-level top-bin gaps looked monotone in bearishness (pure bull +0.6pp → sustained bear −13.5/−14.2pp). Conditioning on the per-row HMM regime label flips it: rows the HMM labels `bear` are nearly calibrated at the top (−3.1pp; the vol-widened forward distribution does its job), while rows labeled `normal`/`bull_quiet` show −9.4/−10.9pp and `crisis` −9.4pp. The bear-window damage comes from rows where the per-ticker HMM **still said benign inside a deteriorating market**. The miscalibration concentrates at **regime transitions the detector hasn't caught yet** — which is precisely why the external VIX trigger (R11) exists, and why it should not be replaced by an HMM-internal signal.

## 3. Rank quality — what the ordering is worth

- Pooled Spearman(`ev_dollars`, realized): **+0.02**, cluster-bootstrap 95% CI **[−0.02, +0.06]** — zero overall.
- By regime: crisis **+0.15**, bear +0.03, normal −0.01, quiet-bull **−0.12**. Per-window: positive in every high-vol window (+0.11..+0.18), negative in every calm-bull window (−0.09..−0.23).
- Spearman(`prob_profit`, realized): **positive in every regime** (+0.15..+0.18).
- On *executed* trades (top-of-book), per-window rho is healthier (e.g. W05 +0.27) — the top of the book is where the ordering signal lives; the deep tail dilutes it.

**Actionable interpretation**: `ev_dollars` earns its rank authority when dispersion is high and *inverts* in calm bulls (its top names are structurally the high-IV names that underperform quiet tapes). `prob_profit` is the more robust ordering signal across regimes. The ranker's value is real but conditional — selection, not sizing, and regime-aware.

## 4. Premium provenance (Phase C: real Theta mids vs synthetic BSM)

- **Economics are provenance-sensitive, in both directions**: W05_rail roughly halved the bear-window return (+4.8% → +2.2%, still beating both benchmarks); W09_rail *raised* returns (+15.0% → +21.5%). Matched-pair premium deltas (real mid vs synthetic, market_mid rows): W05 mean +6.6% (IQR −2.4..+18.2%), W09 +2.3% (IQR −12.5..+14.7%) — name- and window-dependent with wide dispersion, so synthetic-BSM error does not cancel at book level and can flip either way.
- **Calibration is provenance-robust**: matched-pair top-bin realized 0.813 vs 0.803 (W05) and 0.762 vs 0.761 (W09); Brier identical to 3 decimals. **The probability miscalibration lives in the forward-distribution model, not the premium synthesis.**
- Coverage: 32–44% of executed trades / ~27% of captured rows priced at real mids (154-ticker larder).

## 5. Limitations (standing register)

1. Synthetic-BSM premiums in Phase A/B — economics carry window-dependent premium error (Phase C bounds it); calibration does not.
2. Survivorship: universe = 2026 monolith members (no PIT membership) — upward bias on absolute P&L; EW-passive shares the bias, SPX proxy does not.
3. Benchmarks ex-dividend (≈ +1.3–1.7%/yr understated); `spx_tr_approx` in bundles.
4. Hold-to-expiry, no profit-target/rolls — put-leg ledger attribution looks worse than full-cycle equity (documented split); profit-taking policies are untested follow-ups.
5. Overlapping windows (6-month step) — cross-window stats are not 11 independent draws; pooled CIs cluster by as_of date.
6. Three guarded carry-forward mark days (2020-11-06 ×3 books); W11 SPX benchmark truncated at 2026-06-04.

## 6. What this means for using the engine

1. **Trust the gate + the book, not the point estimates**: the EV>0 + R10 + event-lockout pipeline produced 9/11 profitable windows with drawdowns consistently shallower than the index. `ev_dollars` magnitudes and top-bin probabilities are not literal.
2. **Prefer `prob_profit` ordering (or blend) over raw `ev_dollars` ordering in calm regimes** — candidate for a follow-up study before any code change (§2: any change is ceremony-tier).
3. **R11 is validated in mechanism, not just outcome**: the overconfidence sits exactly where an HMM-internal signal lags and an external VIX trigger doesn't. Keep R11 external.
4. **Entry-timing dominates bear performance** — a deployment-pacing rule (scale-in after vol repricing rather than continuous full deployment) is the highest-value strategy-layer follow-up this data suggests.
5. **Produce the full-universe premium rail** before treating absolute EV magnitudes as economically meaningful — provenance moves book results double-digit percent in both directions.

_Method/audit: issue #517 (complete SANDBOX↔MACBOOK transcript), 13 fingerprinted bundles, driver at `backtests/regression/trader500k_campaign.py`, branch `claude/trader500k-campaign`. Analysis code: pooled reliability/Wilson/Brier/ECE + cluster bootstrap re-derived independently by SANDBOX from the committed bundles._
