# Theta Data — Entitlement Re-test & Next-Pull Decision (2026-06-17)

**Author:** Opus 4.8. **Mandate:** the larder pull is complete; free subscription
days remain; storage is unconstrained; maximize engine output. Re-probe the live
ThetaData v3 terminal (`127.0.0.1:25503`) to confirm *what is actually entitled*,
reconcile against the new Bloomberg acquisition plan
(`docs/DATA_ACQUISITION_PLAN_2026-06-14.md`), and **decide what to pull next from
Theta.** All probes below are read-only `GET`s.

> **Repo note (added on merge to `main`).** The companion `DATA_ACQUISITION_PLAN_2026-06-14.md`
> and `THETA_PULL_AUDIT_2026-06-15.md` referenced below live on the campaign/broad-pull branches,
> **not on `main`**. The canonical on-disk inventory is `docs/DATA_INVENTORY.md`.

---

## 1. Verdict in one line

The single highest-value Theta pull available at this entitlement is the
**survivor-bias delisted-name option history** (acquired/bankrupt S&P names) —
confirmed pullable, attacks a *documented* engine weakness, and is the one dataset
**only Theta can supply**. The audit's prior "#1 net-new" (per-strike IV/greeks
time series) is **DEAD — not entitled** (every greeks/IV route 404s). Pull,
ranked: **(1) delisted survivor-bias set · (2) index GEX reference SPX/NDX/RUT/VIX/XSP
· (3) universe 150→~200 · (4) merge BRKB (PR #413).** Do **not** pull tick
trades (out of scope per CLAUDE.md §3) or re-attempt an IV/greeks pull.

---

## 2. Live entitlement map (probed 2026-06-17)

| Endpoint | Result | Net-new? | Decision |
|---|---|---|---|
| `/v3/option/history/eod` (all-strikes) | ✅ 200 | the workhorse | KEEP — proven |
| `/v3/option/history/open_interest` | ✅ 200 | already pulled | KEEP |
| `/v3/option/history/trade` (tick prints) | ✅ 200 (~5,400 rows / strike / 4d) | yes | ❌ **out of scope** — CLAUDE.md §3 forbids tick/sub-minute; EOD `volume`+`count` already feed the ADV need (E-13) |
| `/v3/option/history/quote` (intraday NBBO) | ⚠️ 400 bare / **410 with `ivl`** | — | ❌ deprecated/not entitled this build |
| `/v3/option/history/ohlc` (intraday bars) | **410 with `ivl`** | — | ❌ not entitled |
| `/v3/option/history/greeks` | ❌ **404** | — | ❌ **NOT SERVED** |
| `/v3/option/history/implied_volatility` + every variant (`eod_greeks`, `all_greeks`, `at_time/greeks`, `bulk_hist/option/greeks`, `option/greeks`) | ❌ **404** | — | ❌ **NOT SERVED at this tier** |
| `/v2/*` (all) | ❌ **410 Gone** | — | v2 fully retired in this terminal |

**EOD schema (what we get):** `symbol,expiration,strike,right,created,last_trade,
open,high,low,close,volume,count,bid_size,bid_exchange,bid,bid_condition,ask_size,
ask_exchange,ask,ask_condition` — **no greek/IV columns.** IV stays an in-house
back-solve (consistent with the existing engine design).

### 2.1 The greeks/IV finding (corrects the prior audit)

`docs/THETA_PULL_AUDIT_2026-06-15.md §6` ranked *"per-strike IV / first-order
greeks time series (Theta option-greeks endpoint)"* as the #1 net-new opportunity.
**That endpoint does not exist at this subscription** — all greeks/IV routes 404
(route-absent, not 403/472 data-gaps). Corroborating on-disk evidence:
`data_processed/theta/iv_surface_history/` holds only **4 names** (A, AAPL, ABBV,
ABNB) — a stalled *back-solve* experiment (the `theta-iv-surface-puller-EXPERIMENT`
branch), not a vendor pull. **Conclusion: per-strike IV/greeks is infeasible from
Theta here; do not queue it.** If the engine ever needs a per-strike IV surface,
the only path is back-solving from the chains we already hold — an engine task, not
a pull.

---

## 3. What's already on disk (don't re-pull)

> **Update 2026-06-25 (superseded by the enrichment run this doc proposed).** The table below is the
> 06-17 snapshot. Since then: `option_history/` reached **154** dirs (BRKB landed, PR #413);
> `index_reference/` grew from SPY/QQQ to **6 roots** (NDX, QQQ, RUT, SPX, SPY, XSP — VIX still
> absent); and the staging trees **`option_history_deep365/`** (8 names, 17.5M rows) and
> **`option_history_delisted/`** (10 names, 9.7M rows) were added. Theta footprint is now **≈ 12 GB**.
> Current disk-verified counts live in **`docs/DATA_INVENTORY.md`**.

`data_processed/theta/` (as of 2026-06-17) ≈ **9.7 GB**:

| Dir | Coverage | Note |
|---|---|---|
| `option_history/` | **153 ticker dirs** (149 larder + 4 orphans AES/AFL/AMD/AOS), 2016→2026-06, all-strikes + OI | the just-finished Phase-1. **BRKB still ABSENT** (PR #413 pending) |
| `index_reference/option_history/` | SPY + QQQ, 853 partitions (Phase-2, done 06-17) | weekly index reference |
| `index_options_chains` / `index_options_surfaces` | 21 files each, snapshot-only (Jun 1) | tiny — NOT a history pull |
| `iv_history` | 497 names, ATM-IV daily | back-solved ATM series |
| `iv_surface` | 558 files, snapshot | 3-date surface |
| `iv_surface_history` | **4 names only** | stalled back-solve experiment (§2.1) |
| `chains`, `option_ohlc`, `stocks_eod`, `vix_family` | misc earlier pulls | small |

---

## 4. The decision — what to pull next (ranked)

All four are the **same proven `eod` + `open_interest` all-strikes config** as the
finished larder (4 workers, 2016 floor, 90-day lookback). No new endpoint risk.

### Pull 1 — Survivor-bias delisted set (HIGHEST VALUE)
**What:** option history for S&P names that were **acquired or went bankrupt**,
2016 → their delisting date. Probed and **confirmed serving** (all 200, full
depth): `SIVB, FRC, ATVI, TWTR, SGEN, PXD, SPLK, RE, ABMD, XLNX` — extend to the
full delisted roster (add e.g. `SBNY, CTXS, VIAC/PARA, DISH, FLT, ANTM→ELV seam,
WLTW, INFO, XEC, MXIM, ALXN, CERN, NLSN, PBCT, KSU, FBHS, DRE, CTLT-watch, FISV→FI`).
**Why:** the engine's **survivor bias is a documented weakness** — the
assignment-disaster tail looks artificially benign for a short-put book because
every name in the larder *survived*. Delisted chains are the only way to put real
blow-ups (SVB/FRC bank failures, buyout gaps) into the GPD tail + forward
distribution. **Bloomberg cannot mass-pull chains → this is Theta-exclusive.**
**§2 posture:** correctness-input to the tail/forward-dist fit; never rescues EV.
**Cost:** ~40–80 names × ~1–8 yrs each; hours, well within free days.

### Pull 2 — Index GEX reference: SPX · NDX · RUT · VIX · XSP
**What:** add the true cash-settled index roots to `index_reference/`, full-history
all-strikes EOD. All five roots probed ✅ (SPX 2016 EOD len=578k — deep strike
depth). Today we only have the **SPY/QQQ ETF** proxies.
**Why:** dealer gamma/GEX actually sits on **SPX** (and NDX/RUT), not the ETFs;
this is the reference surface for `DealerPositioningAnalyzer` and the macro/vol
overlays. Reference-only — **never enters the candidate ranker** (same gate as the
SPY/QQQ Phase-2).
**Cost:** small (5 roots).

### Pull 3 — Universe expansion 150 → ~200
**What:** the next ~50 liquid optionable S&P names not yet in the larder, same
config. The scope docs already call the 150→180 band a *"trivial resumable add."*
**Why:** broadens the wheel belt and the per-strike OI/ADV base (E-13/E-14 wiring).
**Cost:** ~50 names × full history; comparable to a third of the original pull.

### Pull 4 — BRKB (merge PR #413)
**What:** Berkshire — the one larder name dropped this run (symbol bug, fix already
in **PR #413**). `BRKB` confirmed serving (686 expirations back to 2012).
**Why:** recovers a top-10 S&P weight for the smallest possible effort. Merge #413,
then it's collected on the next larder pass automatically.

### Explicitly NOT pulling
- ❌ **Per-strike IV/greeks** — 404, not entitled (§2.1).
- ❌ **Tick trades / intraday quote/ohlc** — out of scope (CLAUDE.md §3); 410/not entitled anyway.
- ❌ **Anything Bloomberg already owns** — q-panel, fundamentals, calendar, etc. live on the Bloomberg track (`DATA_ACQUISITION_PLAN_2026-06-14.md`), not Theta.

---

## 5. Suggested execution order

1. **Merge PR #413** (BRKB) — unblocks the one known gap.
2. **Pull 1 (delisted)** — highest engine value; needs a finalized delisted roster
   (derive from PIT index add/drop history, off the Bloomberg track, or hand-list).
3. **Pull 2 (index roots)** — cheap, parallelizable, reference-only.
4. **Pull 3 (universe +50)** — last; broadens but doesn't fix a known defect.

All raw-vs-split-adjusted hygiene from the larder applies unchanged (Theta = RAW;
never mix with split-adjusted Bloomberg OHLCV). Branch + PR per the repo contract;
nothing is running now, so no live pull to protect.

---

---

## 6. Top-20 mega-cap "more depth" probe (2026-06-17) — the lookback finding

Goal: enrich the 20 largest S&P names (by `cur_mkt_cap`): `NVDA, AAPL, GOOGL, GOOG,
MSFT, AMZN, META, AVGO, TSLA, BRKB, WMT, LLY, JPM, XOM, V, JNJ, MU, MA, COST, ORCL`.
The only depth lever left (greeks/intraday/tick all unavailable, §2) is the
**per-contract lookback** — the larder captured each option's last **90 days**; the
EOD endpoint serves up to a **365-day span** (`--lookback-days 364`; 365 → window
366 → HTTP 400). Probed whether deepening it actually helps. **It largely does not.**

| Probe (NVDA) | Result | Implication |
|---|---|---|
| exp `20260501` EOD @ 90d | 49.8s, 632,141 B, 4,623 rows | even **90d is slow** for heavy mega-cap expirations |
| same @ 180d | 83.1s, **632,141 B, 4,623 rows — byte-identical** | a recently-listed contract has **no data** before ~90–120d; deeper lookback just **re-scans empty range** |
| same @ 364d | **timed out >180s** (read-timeout) | flat 364d re-pull **fails on exactly the heaviest names** at the larder's 180s timeout |
| LEAP exp `20250117` @ 90d | 8,344 rows | long-dated contracts **do** carry pre-90d history… |
| LEAP same @ 364d | so slow it didn't return in minutes | …but the call is operationally painful |

**Conclusion — do NOT do a flat 365-day re-pull of the top-20:**
1. The top-19 are **already near-complete at 90d** (every strike, both rights,
   OHLC+bid/ask+vol+OI, every expiration 2016→now). Very little entitled net-new
   depth exists for them.
2. Deeper lookback adds real data **only** for contracts that traded >90d before
   expiry (standard monthlies / LEAPS). That extra 90–365 DTE history is **low
   value for a 30–45 DTE wheel** — it feeds the *dormant* Nelson-Siegel / skew
   term-structure subsystem, not the live ranker.
3. The deep calls **time out at 180s on the heaviest names** → the flat re-pull is
   both wasteful (empty-window rescans) and operationally fragile.

**What actually enriches the top-20:**
- ✅ **BRKB (the missing #10)** — launched at the larder's exact 90d config to
  `option_history/ticker=BRKB` (net-new, no overwrite). This is the genuine top-20
  win: a top-10 S&P weight that the larder dropped on the symbol bug (PR #413).
- ⏸️ **Deep-DTE / term-structure pull** (364d, raised timeout ~300s, only where it
  adds data) — **deferred**: worth doing only if/when the NS/skew subsystem is
  activated; better ROI lies in the net-new pulls (§4: delisted survivor-bias,
  index GEX, universe expansion).

---

*Companion: `docs/THETA_PULL_AUDIT_2026-06-15.md` (prior audit — §6 superseded by
§2.1 here), `docs/DATA_ACQUISITION_PLAN_2026-06-14.md` (Bloomberg track).
Entitlement + depth state probed live 2026-06-17.*
