# Heavy-verify — Is the calm/elevated-entry top-bin over-confidence net-COSTLY? (Mac terminal)

**Campaign:** #446 (W6) · **Branch:** `claude/mac-w6-topbin-netcost` · **Started:** 2026-06-28
**Terminal:** Mac (fresh clone, no real-premium rail) · **Mode:** validation / **measurement-only** (measure & report; no engine-behaviour edits)
**Follow-up to:** W3/#442 (top-bin `prob_profit`>0.90 over-confidence is calm/elevated-**entry** ~−16pp, absent at crisis-entry) · **Precedent:** R11b/#437 (validated net-costly, **CLOSED**)
**Data path / provider:** `SWE_DATA_PROVIDER` unset → **bloomberg** (committed CSVs under `data/bloomberg/`); connector = `MarketDataConnector`.

> Every claim below carries its data path and, where statistical, a confidence interval.
> The machine-readable result is persisted to
> `docs/verification_artifacts/topbin_netcost_2026-06-28/w6_topbin_netcost.json` **before** any
> pretty-print, so a console-encoding crash never loses the ~40-min full-universe compute.
> `PYTHONIOENCODING=utf-8` set for all runs. The decision trio
> (`ev_engine` / `wheel_runner` / `candidate_dossier`) is **never edited** — this campaign only
> *calls* the authoritative ranker and forward-replays realized P&L.

---

## §1 The question (the whole point)

W3/#442 established — and the Windows verifier independently reproduced (calm −0.166, crisis +0.035) —
that the engine's highest-confidence bin (`prob_profit` > 0.90) is **over-confident specifically at
calm/elevated VIX entry** (realized win-rate ~16pp below forecast) and **well-calibrated at crisis
entry**. That is a *calibration* fact. It does **not** by itself say the trades lose money.

The R11b precedent (#437) is the cautionary tale: R11b gated VIX>25 top-bin real-premium picks on
exactly this kind of calibration argument, and a full-cycle A/B found it **net-costly** (W3 −14.85pp /
W4 −12.71pp) because it gated *profitable* VRP harvest. **Lesson: a calibration gap is not
automatically a reason to size down — the trades may still be net-positive premium.**

So W6 asks the dollar question that decides whether a rule is even worth building:

> Does the calm/elevated-entry top-bin over-confidence translate into **net dollar LOSSES**, or are
> those trades **still net-positive premium** despite missing their forecast ~16pp of the time?

- **Net-COSTLY** (mean realized < 0, or materially below the all-candidate baseline, in **both**
  windows) → a targeted calm/elevated top-bin size-down **may** be worth it → spec it for the Windows
  terminal (do not implement; trio is their lane).
- **Net-POSITIVE** → document the calibration caveat and **build nothing** — the R11b outcome.

---

## §2 Method (mirrors the R11b A/B discipline)

**Driver:** `scripts/audit_topbin_netcost.py` → JSON: `w6_topbin_netcost.json`. Rail-independent,
read-only on `engine/`; the trio is imported and *called*, never modified.

1. **Grid.** First business day of each month over the **union** of both windows (2020-01 … 2025-12,
   72 dates); each window is then sliced from the shared record set (Window A = 2020-2024, Window
   B = 2021-2025; the 2021-2024 overlap is computed once, not twice). Both windows **end well before
   the 2026-03-23 OHLCV splice** (W1/#439: BKNG ÷25, CVNA ÷4.7) — Window A ends 447 days prior,
   Window B 82 days prior — so that defect **cannot** contaminate any forward close.

2. **Entry regime.** At each as_of, VIX-at-entry from `connector.get_vix_regime(as_of)["vix"]` →
   bands per the #446 brief: **calm ≤15 / elevated 15-25 / crisis >25**. The calm/elevated
   population (VIX≤25) is the subject; crisis-entry is carried as a reference row. (Note these are the
   *W6* bands; W3/R11 used calm<20 / elevated 20-30 / crisis≥30 — the W6 brief redefines them to
   isolate the over-confident calm/elevated band and the R11 VIX>25 cut.)

3. **Population.** The **authoritative, rail-independent ranker**
   `rank_candidates_by_ev(as_of=…, universe_limit=None, top_n=10_000, min_ev_dollars=-1e9,
   include_diagnostic_fields=True)` — the full evaluated short-put population (matching the W3
   calibration study), not just the displayed top-N. **Top-bin** = `prob_profit` > 0.90.

4. **Realized P&L — the canonical helper.** For each candidate, expiry = as_of + `dte` (engine's
   `dte_target`=35); spot-at-expiry via **`_spot_on_or_after`** and held-to-expiry P&L via
   **`_forward_replay_realized_pnl`** — both imported verbatim from
   `backtests/regression/_common.py` (the helper the brief names), so the measurement cannot drift
   from the regression suite. P&L is computed at **three friction levels**:
   - `none` — gross modeled premium (the calibration-comparable number).
   - `bid_ask` — premium net of the bid/ask half-spread (`max(0.05, 0.08·premium)`).
   - **`full`** — bid/ask + $0.65 open commission + (10bp notional + $0.65) assignment slip **on ITM
     (assigned) puts only**. **This is the headline:** friction only ever makes a short put *less*
     profitable, so the honest net-cost verdict must be all-in.

5. **Buckets** (per window + pooled): for the calm/elevated (VIX≤25) population — **top-bin** vs
   **non-top-bin** vs **all-candidate baseline**; plus calm (≤15) and elevated (15-25) splits and a
   crisis (>25) reference. Each bucket reports n, mean realized $/contract at full friction with a
   **cluster (block) bootstrap 95% CI resampling as_of dates** (respects within-day correlation —
   honest where an i.i.d. bootstrap is not, since names recur and forward windows overlap), total $,
   Wilson 95% win-rate, mean forecast `prob_profit` vs realized win-rate (the **W3 calibration gap**),
   ITM rate, and mean premium. A `top_bin_would_trade` sub-bucket (ev_dollars>0) gives the
   "trades actually taken" lens.

6. **Verdict** (`make_verdict`): on the **pooled** calm/elevated top-bin, requiring the **two windows
   to agree on sign**. INSUFFICIENT at n<30 (no SUPPORTED below ~30); NET-COSTLY if the full-friction
   mean CI excludes 0 below; DRAG if positive but the CI sits materially below the baseline mean;
   NET-POSITIVE if the CI excludes 0 above; else INCONCLUSIVE.

**Honesty caveats (carried in the JSON `meta.data_caveats`):**
- Both windows end before the 2026-03-23 splice — no #439 contamination.
- `premium_source` is mostly `synthetic_bsm` on the Bloomberg-CSV path, so `edge_vs_fair`≈0 — this
  measures the **engine's own modeled premium vs realized moves**, not a market bid/ask edge. (This
  is the same surface R11b's A/B used; the comparison to that precedent is therefore apples-to-apples.)
- Wilson and cluster-bootstrap CIs are **optimistically tight** — candidates are not i.i.d. (recurring
  names, overlapping forward windows). Sign + cross-window agreement carry the verdict, not CI width.

---

## §3 Results

Full-universe run: **15,642** candidate-outcomes over **71** monthly as_of dates (one date,
2020-01-01, is a market holiday and produced 0 candidates). Band mix by entry VIX: elevated
8,993 / crisis 3,849 / calm 2,800. Headline friction = **full**. Per-bucket detail in
`w6_topbin_netcost.json`.

### §3.1 The headline — top-bin economics by independent cut (calm/elevated, VIX≤25)

| Cut (calm/elevated top-bin) | n | mean $/contract (full) | 95% CI (cluster-boot) | win-rate | calib gap | baseline (all / non-top) |
|---|---:|---:|---|---:|---:|---:|
| **pooled 2020-2025** | 488 | **+$132.79** | [15.83, 233.17] ✅excl 0 | 0.834 | −8.07pp | −$38.20 / −$45.58 |
| overlap 2020-2024 | 376 | +$115.34 | [−13.41, 222.94] | 0.840 | −7.19pp | −$25.67 / −$31.78 |
| overlap 2021-2025 | 443 | +$147.11 | [27.56, 256.31] ✅excl 0 | 0.840 | −7.32pp | −$38.29 / −$45.99 |
| **disjoint 2020-2022** (COVID+bear) | 294 | **+$66.91** | [−77.86, 149.35] | 0.840 | −7.27pp | −$46.59 / −$56.73 |
| **disjoint 2023-2025** (low-vol) | 194 | **+$232.62** | [0.47, 449.37] ✅excl 0 | 0.825 | −9.28pp | −$34.53 / −$41.00 |
| calm only (≤15) | 28 | +$266.02 | [50.35, 525.79] ✅excl 0 | 0.893 | −2.72pp | −$124.91 |
| elevated only (15-25) | 460 | +$124.68 | [−8.95, 232.79] | 0.830 | −8.39pp | −$11.20 |
| would-trade (ev>0) | 318 | +$173.20 | [16.79, 312.05] ✅excl 0 | 0.840 | −7.94pp | — |

**Every cut's point estimate is net-POSITIVE — minimum +$66.91/contract** (the COVID + 2022-bear
disjoint half). Four of the eight cuts have cluster-bootstrap CIs that exclude 0 above; the other
four straddle 0 (wide, date-clustered) but are positive point estimates. **No cut is net-costly.**

### §3.2 The over-confidence is real — and is a *level* miscalibration, not a rank inversion

The top bin **reproduces W3's over-confidence in sign**: forecast `prob_profit` ≈ 0.921 vs realized
(gross) win-rate ≈ 0.840 → **calibration gap ≈ −8pp** pooled (−7.2/−7.3pp per window; the same
negative direction as #442, smaller magnitude under W6's narrower calm/elevated bands + monthly
grid + breakeven realized rule). It misses its forecast ~8pp of the time — **yet it is the single
most profitable bucket in the book.** The all-candidate baseline at calm/elevated entry is
**−$38.20/contract** and the non-top-bin baseline is **−$45.58**; the top bin is **+$170.99 /
+$178.37 above** them. So `prob_profit` is over-confident in *absolute level* but strongly
**rank-discriminating for dollars** — the top bin is exactly where the money is. A size-down on it
would remove the book's best trades.

### §3.3 Friction does not flip the sign

Pooled top-bin mean: gross **+$162.14** → bid_ask **+$136.61** → full **+$132.79**. All-in friction
costs ~$29/contract against a ~$162 gross edge — nowhere near enough to turn the bucket costly.
(Friction reconciles arithmetically: ~$25.5 half-spread at mean premium $3.17, +$0.65 open, +
assignment slip on the ~20% ITM tail.)

### §3.4 Independence — the disjoint halves are the real robustness test

The brief's named windows overlap ~80% (2021-2024 shared), so their agreement is **not** independent
validation. The two **disjoint** halves are genuinely independent and span opposite regimes —
2020-2022 (COVID crash + 2022 bear) and 2023-2025 (low-vol recovery). The top bin is net-positive in
**both** (+$66.91 and +$232.62), with the over-confidence present in **both** (−7.27pp, −9.28pp) and
the large margin over baseline in **both** (+$113, +$267). The conclusion does not hinge on any one
regime.

### §3.5 Honesty / robustness checks

- **Survivorship ≈ 0.** The forward-spot skip rate (`_spot_on_or_after` → None) over the **full
  grid** is **1 in 15,643 = 0.006%** (`collection_diag` in the artifact; the one skip was a single
  elevated-entry candidate). The ranker's own 504-day-history + chain-quality gates pre-filter to
  established names that have forward data, so the silent-skip surface the review flagged has no
  practical effect here. (The ranker's own PIT gates dropped 21,149 candidate-dates upstream — its
  intended point-in-time filtering, not a measurement artifact.)
- **Bootstrap not a handful-of-dates artifact.** The 488 top-bin outcomes spread over **49 distinct
  as_of dates** (of 54 calm/elevated dates); max single-date share 16.2%, top-3 dates 29.5%.
- **calm-only sub-bucket is under-powered AND temporally concentrated** (n=28 < 30; calm entry
  occurred *only* in 2023-07…2024-12). Its favorable CI is reported but **not** labeled SUPPORTED.
- **elevated-only (n=460) CI straddles 0** [−8.95, 232.79]; the strict CI-exclusion comes from
  pooling + the disjoint-2023-2025 + would-trade cuts, not from elevated alone. The verdict is
  framed on point-estimate positivity across independent regimes + the baseline margin, not on a
  single CI.

### §3.6 Crisis-entry reference (NOT the W6 subject)

At crisis entry (VIX>25) the top bin is directionally **negative** — pooled −$157.88 (n=704) — but
its CI [−571.29, 187.88] **straddles 0** (inconclusive), and it is dominated by 2020 crash-onset
entries (Window 2021-2025 crisis top-bin is only −$11.18, n=327). This **differs from W3/#442's
"crisis well-calibrated"** for two methodology reasons, not a contradiction: (a) W6 defines crisis as
VIX>25 whereas W3 used VIX≥30 — W6's crisis bucket includes the 25-30 band W3 called *elevated*; and
(b) W6's monthly grid includes the 2020-01…05 COVID-crash entries that W3's grid (starting 2020-06-15)
excluded. Crisis is carried only as a reference; it is not part of the W6 verdict, and the existing
**R11 VIX>25 gate already targets this region** (its dollar impact was studied separately in
#306/#307). A dedicated crisis-entry net-cost study is a reasonable future item (see §4).

---

## §4 Verdict

> **NOT NET-COSTLY → build no calm/elevated top-bin size-down rule. Document the calibration caveat.
> This is the R11b precedent (#437) repeating.**

The calm/elevated-entry top-bin over-confidence (W3/#442) is **real but not costly**: the bucket is
net-**positive on every independent cut** (min +$66.91/contract; pooled +$132.79, CI [15.83,
233.17]), is net-positive in **both disjoint regime halves**, and sits **+$171/contract above the
−$38 all-candidate baseline**. The over-confidence is a *level* miscalibration, not a rank inversion —
`prob_profit` still rank-orders dollars strongly, and the top bin is the book's most profitable
slice. Sizing it down would gate the engine's best premium — **the exact R11b mistake**, amplified
(R11b at least gated a regime where the trades were marginal; here the targeted trades are the most
profitable in the book).

**Decision (per the #446 rule):** net-POSITIVE → **document, build nothing.** No spec is filed for
the Windows terminal — the correct action is *no rule*. The lesson is reaffirmed: **a calibration
gap is not a reason to size down; the dollar economics are.**

**What would change this verdict:** a cut showing CI-conclusive net-cost (mean < 0 with the CI
excluding 0) at calm/elevated entry. None exists across pooled, both overlap windows, both disjoint
halves, calm, elevated, or the would-trade subset.

**Honest scope:** the strict "CI excludes 0" holds for the pooled, 2021-2025, disjoint-2023-2025,
calm-only, and would-trade cuts; it straddles 0 (positive point estimate, wide CI) for 2020-2024,
the COVID-era disjoint half, and elevated-only. The "no rule" conclusion does **not** depend on
CI-conclusive positivity — it requires only the **absence of net-cost** (unambiguous here) plus the
top bin beating baseline (by +$171).

**For the Windows terminal (separate from the verdict):** the **crisis-entry** (VIX>25) top bin is
directionally negative but **inconclusive** (CI straddles 0; §3.6) and diverges from W3 for
band-definition + grid reasons. It is already covered by the existing R11 VIX>25 gate. If a fresh
look is wanted, a crisis-entry net-cost A/B (mirroring this driver, restricted to VIX>25, with the
2020 onset isolated) would settle it — but that is **not** an output of W6 and **no trio change is
implied by this campaign**.

---

## §5 Reproduction

```bash
# Full-universe (~40 min single-process; or parallel collect→merge):
PYTHONIOENCODING=utf-8 python scripts/audit_topbin_netcost.py        # inline single process
# parallel: 4 collectors over date-chunks, then merge —
PYTHONIOENCODING=utf-8 python scripts/audit_topbin_netcost.py --dates "<chunk>" --collect-to recs0.jsonl
PYTHONIOENCODING=utf-8 python scripts/audit_topbin_netcost.py --analyze-from "recs*.jsonl"
# smoke:
PYTHONIOENCODING=utf-8 python scripts/audit_topbin_netcost.py --quick --limit 40

# Methodology pins (fast; no grid re-run):
pytest tests/test_w6_topbin_netcost.py -v
```

Artifact: `docs/verification_artifacts/topbin_netcost_2026-06-28/w6_topbin_netcost.json`.
