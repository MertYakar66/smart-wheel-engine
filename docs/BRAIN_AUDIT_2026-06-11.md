# Engine brain audit — 2026-06-11 (autonomous overnight campaign)

**Mandate (operator, 2026-06-10):** "Run a thorough analysis on our smart wheel
engine whether its logic is safe and sound. It must produce reliable and
realistic outputs. … verify that our backend logic, the brain, is smart and
clear, start testing."

**Method:** 8-dimension multi-agent review panel against `origin/main @ 7f9dc10`
(post #397–#401), every claim backed by an executed probe in a pinned worktree;
adversarial verification armed for any NEW HIGH/CRITICAL finding; the three most
load-bearing probes re-executed first-hand by the session afterward
(`_qreview_probes/` in the campaign worktree; workflow `wf_b1f6f6aa-2c9`).
Cross-referenced against the documented evidence base (DECISIONS.md, TESTING.md,
PROJECT_STATE.md, the S-study ledger, the 2026-05/06 verification campaigns) so
known, documented limitations are reported as such, not re-flagged.

---

## 1. Verdict

**The brain is safe and sound.** All 8 dimensions returned SOUND_WITH_CAVEATS;
**zero CRITICAL and zero HIGH findings** — the adversarial-verification stage
never armed. 77 findings: 56 already-documented (the repo's self-knowledge is
accurate), 21 new (4 MEDIUM / 13 LOW / 4 INFO). The caveats are bounded,
disclosed below, and none blocks operation as a **supervised decision-aid +
crisis-refusal layer** (the documented defensive-sleeve framing stands).

The mathematically load-bearing results:

- **EV integral is exact.** Hand replication of a synthetic short put matched
  all 21 `EVResult` fields to <5e-9 — including a −30% crash path
  (`ev_dollars` $26.28 vs $246.35 premium-if-all-goes-well; `cvar_5` equals the
  crash-path P&L to the cent). Units verified (net dollars per position,
  contract-scaling exact). Re-run first-hand: `ALL_MATCH`.
- **Pricing is textbook.** Put-call parity to 8.5e-14 over 500 random nodes;
  Hull S=42/K=40 to 4e-7; all Greek signs/units match
  `docs/GREEKS_UNIT_CONTRACT.md` against finite differences. The IV
  percent→decimal contract is proven end-to-end (connector serves percent gated
  (3.0, 10000]; exactly five /100 conversion points; AAPL premium $3.69
  reconstructs from decimal vol — percent vol would give $233).
- **The §2 firewall holds, verified fresh.** Every tradeable-candidate path
  (3 rankers, explore/select, consume wires, roll suggesters, EV-emitting API
  endpoints) routes through `EVEngine.evaluate`; reviewer lattice R1–R11 is
  structurally downgrade-only; dealer multiplier proven bounded [0.70, 1.05],
  scales `ev_dollars` only, `ev_raw` byte-identical. 28 launch-blocker test
  files: **560 passed, 0 failed**.
- **Realism spot-checks pass.** ~25-delta 35d CSP premium yields 1.35–1.50%
  per 30d at IV 31–34% — inside the market-plausible band. Forward-distribution
  σ at the frontier sits within +3%/−9% of the IV-implied σ for AAPL/BIIB/XOM.
  Overlay envelopes verified on a 20-ticker live rank (HMM ∈ [0.28, 1.03],
  widening ∈ [1.00, 1.008], labels a sane mix — not 98% crisis).
- **The safety reflexes fire.** The 5-ticker smoke returns 4 rows today
  because JPM's 2026-07-14 earnings falls inside a ~35-DTE option's life —
  the event lockout refuses to sell through earnings (verified: ranks fine at
  as_of=2026-05-15 where expiry precedes earnings). UNH carries a negative EV
  under a hostile regime multiplier rather than being rescued.

---

## 2. Dimension verdicts

| # | Dimension | Verdict | Headline |
|---|---|---|---|
| 1 | EV core math | SOUND_WITH_CAVEATS | Integral exact to <5e-9; clamps/order verified; D19 exit-cost optimism known |
| 2 | Forward distribution | SOUND_WITH_CAVEATS | PIT clean (differential probes byte-identical); cascade honest; D21 over-dispersion ≈ offsets empirical-vs-implied gap |
| 3 | Pricing & Greeks | SOUND_WITH_CAVEATS | Textbook-exact; IV units proven; zero-skew bias is asymmetric (see §3) |
| 4 | Tail & regime overlays | SOUND_WITH_CAVEATS | All envelopes hold live; F4 doc state matches code exactly; GPD dormant on primary tier (known) |
| 5 | Costs & frictions | SOUND_WITH_CAVEATS | Round-trip 3.7–4.3% of premium — honest for liquid names; backtest friction 5.3× harsher than internal model |
| 6 | §2 invariant integrity | SOUND_WITH_CAVEATS | Firewall verified fresh; 560 launch-blocker tests green; D16 token-binding gap (see §3) |
| 7 | Calibration evidence | SOUND_WITH_CAVEATS | Evidence base reproduces byte-exactly; truth table in §4 |
| 8 | Data foundation | SOUND_WITH_CAVEATS | IV gate verified live; frontier actually 2026-06-04 (the "82d stale" warning is a stale-clone artifact); as_of=None bypass (see §3) |

---

## 3. New findings that deserve action (the four MEDIUMs)

All four are bounded (none flips an EV sign, none rescues a candidate, none is
reachable as a §2 breach on the canonical paths). Dispositions are
recommendations for the operator — none was actioned autonomously beyond
documentation.

1. **D16 EV-authority token is not bound to trade parameters at consume**
   (`wheel_tracker` consume path). Probe-confirmed end-to-end and re-verified
   first-hand: a token issued for an AAPL/180 row was consumed to open a
   never-ranked ZZZQ/50 in `require_ev_authority=True` strict mode. Single-use
   holds (replay refused); binding does not. Latent — the canonical
   `consume_ranker_row` wire passes consistent rows — but the chokepoint's
   "prevents manual/webhook bypass" claim is overstated. **Disposition:** bind
   ticker/strike/expiry (or the row hash already in the log) at consume;
   trio-adjacent → needs operator-greenlit decision-layer touch.
2. **RV30/RV252 tail widening protects only the put-entry ranker.**
   `realized_vol_widened_log_returns` is called only inside
   `rank_candidates_by_ev`; covered-call, strangle, and tracker roll EVs
   evaluate unwidened distributions in vol clusters (≤15% σ effect).
   **Disposition:** extend the widening call to the other three call sites —
   mechanical, but it moves CC/roll EVs ⇒ re-baseline event; schedule with the
   next supervised re-baseline block (alongside D19/D21).
3. **`as_of=None` ("today") scans bypass the per-ticker staleness gate.**
   Re-verified first-hand: CTRA (left the index at the 2026-03-23
   reconstitution; last bar 2026-03-20) ranks at +$40.43 EV on an 83-day-old
   spot under `as_of=None`, while an explicit `as_of` correctly drops it.
   **Disposition:** resolve `as_of=None` to the data frontier date before the
   staleness check (engine-side, small, not trio-exempt — needs the same
   review lane as any wheel_runner touch). Until then: operators should pass
   an explicit `as_of`.
4. **Size impact is modeled but never called.** The Almgren-Chriss sqrt
   size-impact term exists with zero production callers; multi-contract
   slippage scales linearly, understating cost as lot sizes grow at $1M scale.
   **Disposition:** wire `adv_contracts` through `calculate_slippage` when
   sizing exceeds a few contracts; until then treat $1M+ cost realism as
   "honest-but-untested-at-size" (matches the S32 framing).

**Worth a docs line (MEDIUM, known but under-documented):** the zero-skew IV
limitation is *asymmetric* — short-put EV is conservative (25Δ put premium
understated 13–41% vs a real smile) but **covered-call EV is optimistic
(~6–12% overstated)**. The docs treat zero-skew as uniformly conservative;
they shouldn't.

Notable new LOWs (full register in the workflow transcript): NaN dealer
confidence resolves to *full* confidence (min(1.0, NaN)=1.0 → grants the 1.05
boost on absent evidence; sandbox-unreachable, live-Theta relevant); negative-EV
magnitudes compress toward zero under sub-1 multipliers on operator displays;
the engine-internal lognormal fallback is mislabeled `'none'` by the ranker;
day-over-day NOS re-phasing jitters σ ±6% with no new information;
`/api/execute` on the dashboard is an arbitrary-code dev route to be aware of.

---

## 4. What a professional should trust today (calibration truth table)

Synthesized from the documented campaigns; every committed artifact the panel
re-derived reproduced its document exactly (Wilson CI [0.602, 0.775]; 2022
crisis realized 0.5771; Pearson −0.018).

| Output | Trust level |
|---|---|
| Ranking order (`ev_dollars` as a score) | **Trust.** ρ 0.19–0.36, window/capital/universe-invariant, concentration-robust |
| `ev_dollars` as a dollar forecast | **Do not.** ~0 correlation with realized $; it is a tail-aware ranking score (documented) |
| `prob_profit` 0.60–0.85 | **Trust ±5pp** (real-money fills run slightly under-confident here) |
| `prob_profit` (0.85, 0.95] | **Haircut ~6–11pp** |
| `prob_profit` (0.95, 1.0] | **Do not size off it.** Realized ~0.70–0.80 unconditional, ~0.58–0.67 in crisis; real IBKR fills: 0.936 predicted → 0.821 realized, with all 5 failures at calm VIX where R11 cannot see them |
| Covered-call leg probabilities | Under-confident (engine skips CCs that would win) — opposite-direction bias |
| Crisis refusal | **The engine's strongest property** (97.8% refusal through COVID onset; correct in aggregate) |
| Crash-onset EV | **Distrust for ~4 weeks at regime breaks** — procyclical until the forward window catches up (documented I10; no onset detector exists) |
| Per-name EV for NFLX / BKNG / CVNA | **Exclude** until the data-layer scale-corruption re-pull lands |
| "Today" scans | Pass an explicit `as_of`; data frontier is 2026-06-04 (~5 trading days back) |

---

## 5. Test evidence (Phase 2)

- Full suite (`-m "not backtest_regression"`, theta env file excluded):
  **3,126 passed / 0 failed** / 14 skipped / 15 xfailed, 4m03s, on
  `origin/main @ 7f9dc10`.
- 28 launch-blocker/§2 files (panel, individually): 560 passed / 0 failed.
- Canonical 5-ticker smoke: healthy (4 rows + JPM earnings-lockout refusal,
  explained above; sub-2s; all fields non-null).
- Backtest-regression lane: not re-run — executed yesterday (2026-06-10) as a
  full A/B (580/580 metric leaves byte-identical); the open snapshot drift is
  tracked in issue #402 and is pre-existing on main, `spearman_rho` unaffected.
- First-hand probe re-runs: EV hand-replication `ALL_MATCH`; D16 cross-ticker
  consume reproduced; CTRA staleness bypass reproduced.

## 6. Bottom line

The decision core is mathematically exact, the §2 authority firewall is intact
and freshly verified, costs and premia are realistic for the liquid S&P wheel
belt, and the system's documented self-knowledge is accurate — the engine knows
what it doesn't know, which is the most important property a capital-touching
system can have. The four new MEDIUMs are hardening items, not soundness
breaks: two belong in the next supervised decision-layer/re-baseline block
(D16 binding, widening coverage), one is an operator-discipline note until
fixed (explicit `as_of`), one is a cost-model wiring item for size growth.

*Generated by the 2026-06-11 autonomous campaign; workflow `wf_b1f6f6aa-2c9`
(8 reviewers, 1.07M tokens, 328 tool calls). Probe scripts preserved under the
campaign worktree's `_qreview_probes/`.*
