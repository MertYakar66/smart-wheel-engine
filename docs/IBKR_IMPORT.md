# IBKR PortfolioAnalyst PDF → viewer history (`scripts/ibkr_import.py`)

Phase 1 of integrating the operator's real IBKR account history into the
**read-only** portfolio viewer (`engine/ibkr_portfolio_adapter.py`, D24/D26).
Observational only (CLAUDE.md §2/§3): the importer and adapter never rank a
candidate, never call `EVEngine.evaluate`, never issue an EV-authority token.

## What it does

```
python scripts/ibkr_import.py "<PortfolioAnalyst.pdf>" --out data_processed/ibkr
```

Reads an IBKR **PortfolioAnalyst since-inception PDF** (via PyMuPDF / `fitz`)
and writes the three artifacts the adapter already consumes:

| Artifact | Built from (PDF section) |
|---|---|
| `portfolio_snapshot.json` | Open Position Summary (p6) + Account Overview (p3) |
| `portfolio_history.json`  | Monthly TWR returns (p5) + Account Overview (p3) |
| `wheel_ledger.json`       | Trade Summary (p66–85) + Performance by Symbol (p21–42) |

The output dir defaults to `data_processed/ibkr/` (gitignored — **real account
data never enters git**). Point the viewer at it with
`SWE_IBKR_DATA_DIR=data_processed/ibkr`.

## Extraction notes

* **Hidden duplicate layer.** The chart-bearing detail sections (positions,
  trades, perf, dividends) carry a 2× duplicate text layer; every parser
  dedups (perf by symbol; trades/positions by asset+ccy+symbol; div/dep by
  full row). Interest (p88–89) is not duplicated.
* Detail rows are clean 9-line column-order blocks in `get_text("text")`;
  `find_tables()` gets headers right but mis-splits data columns, so the
  parsers validate row shape (percent/number/Yes-No) rather than trust column
  geometry.

## Reconciliation (asserted at run time)

| Quantity | Imported | Report | Match |
|---|---|---|---|
| Ending NAV / net_liquidation | 143,115.00 | 143,115.00 | exact |
| Deposits + withdrawals + ACAT | 111,683.03 | 111,683.03 | exact |
| Dividends (net of payment-in-lieu) | 634.98 | 634.98 | exact |
| Cumulative return | +63.6% | +63.4% | +0.20pp* |
| FX CAD→USD | 0.71733 | (report-implied) | derived |

\* the +0.20pp drift comes from using the Jun-2026 **1-month** figure (p3) as
the partial-month return + rounding in the transcribed monthly series; the
history endpoint is anchored to the **exact** Ending NAV regardless.

## Honest limitations (→ future phases)

1. **No per-execution dates.** A PortfolioAnalyst statement aggregates each
   contract's buys/sells; it has no fill timestamps. Each closed contract is
   dated on its **exact option expiry** (from the OCC symbol) as `exit_date`;
   `entry_date` is set equal to expiry and flagged in `notes`. True per-fill
   open dates require the IBKR **Flex "Trades"** CSV export (a later phase).
2. **Monthly `port` is a TWR-consistent equity index**, not actual month-end
   NAV — the PDF charts the curve but does not tabulate dollar NAV. Period
   *returns* reproduce the report exactly; intra-series dollar *levels* ignore
   deposit timing (endpoints anchored to inception + Ending NAV).
3. **`premium` is gross credit collected** (sold-to-open proceeds), matching
   the demo-fixture convention — not net-of-buyback premium income.
4. **Balance-sheet margin fields** (available funds / excess liquidity /
   maintenance margin / day & week deltas) are not in a performance report →
   emitted as JSON `null` (viewer renders "—"). Only a live IBKR snapshot
   carries them.
5. **Scope tagging.** `in_universe` is set from
   `data_raw/sp500_constituents_current.csv` (503 names). Out-of-mandate names
   (e.g. TSM ADR, CLS, CCO, ENB, CNQ) are kept as **exposure-only** (they count
   toward NAV / sector / single-name / currency denominators — real risk) but
   are never placed in the rankable set, per the adapter's universe discipline.

## What the imported book reveals (2026-06-05)

NAV $143,115 on $111,683 deposited (+63.6% TWR since Mar 2025) but in a −35%
QTD drawdown; **CLS = 130% of NAV** (assigned stock, 13× the R10 single-name
cap), **Technology = 93% of NAV** (R9 sector cap is 25%), financed by a
~$165k USD margin debit. The viewer's concentration meters surface all of this
from real data.

## Adapter robustness fix (same change set)

`returns_view` and `risk_view` used `float(acct.get(key, 0.0))`, which only
defaults when a key is *absent* — a present-but-`null` value (which every real
import carries for the non-derivable fields) crashed `float(None)`. Both now
use the module's null-safe `_num()`. The demo fixtures had all fields
populated, so this latent bug only surfaces on a real (or PDF-imported) book.
Covered by `tests/test_ibkr_import.py`.

---

# Phase 4 — exact per-fill ledger (`scripts/ibkr_flex_ledger.py`)

Re-keys `wheel_ledger.json` from the **IBKR Flex "Trades"** export — the
authoritative per-execution feed the performance PDF lacks — and refreshes the
`premium` series in `portfolio_history.json`. Same read-only / §2-§3 discipline;
the decision-layer trio is untouched and real data stays gitignored.

```
python scripts/ibkr_flex_ledger.py "<A.csv>" "<B.csv>" --out data_processed/ibkr
```

## Inputs
Two contiguous Activity-Flex CSVs (IBKR caps a single run at 365 days). 14
columns: `AssetClass, Symbol, Strike, DateTime, Put/Call, Proceeds,
IBCommission, Open/CloseIndicator, Buy/Sell, CurrencyPrimary, Expiry, Quantity,
TradePrice, OrigTradeID`. **`OrigTradeID` is blank** in this export, so dedup is
by the file boundary: keep all of A, append B fills with `DateTime` strictly
after A's last timestamp (the 2026-03-10 boundary overlap drops out). Subtotal
rows (blank `DateTime`) are skipped.

## Reconstruction
- **FX** — CAD option/stock fills → USD at the trade-date `USD.CAD` rate carried
  by the forex (`CASH`) fills (nearest-prior). Forex conversions themselves are
  not trading P&L and are excluded from realized.
- **Stock** — the account trades **both long and short** (the `Open/Close`
  indicator proves it: `SELL O` opens a short, `BUY C` covers). Accounting uses
  separate long & short **average-cost** books driven by that indicator
  (order-robust); the long book is seeded with the ACAT-in transfer basis from
  the PDF. Replaying every fill reproduces the p6 book **exactly** (CLS 500;
  AMD/NVDA/TSM/WMT/CNQ/ENB 100 each).
- **Options** — realize per contract by net cash. IBKR books an expiry/assignment
  as a **$0 close**, so premium is kept; `exit_reason` distinguishes
  `csp_expired_otm` / `csp_assigned` / `csp_bought_to_close` (and the `cc_*`
  analogues), assignment detected by a stock fill at ~strike within 3 days of
  expiry.
- **Dates are now exact** — `entry_date` is the real sell-to-open fill, not the
  expiry (the Phase-1 limitation this phase removes). 957 closed positions
  (870 option contracts + 87 stock closes).

## Reconciliation (run-time)
| Quantity | Value | Source / check |
|---|---|---|
| Net deposits/withdrawals/ACAT | 111,683.03 | PDF — exact |
| Dividends | 634.98 | PDF — exact |
| Interest | −501.08 | PDF — exact |
| Ending NAV | 143,115.00 | PDF — exact |
| **MTM (trading) gain** | **+32,637.69** | PDF waterfall — exact |
| Option contracts | 871 | = PDF Trade Summary ✓ |
| Option net proceeds | 124,838.62 | vs PDF 124,829.96 (±$8.66 CAD FX) |
| Commissions | −1,209.27 | + other fees −61.71 = −1,270.98 ✓ |
| Ending stock book | CLS 500 + 6×100 | = p6 exactly ✓ |
| **Reconstructed MTM (from fills)** | realized 76,378 − unrealized 41,597 = **34,781** | vs 32,637.69 → **residual +$2,143** |

The +$2,143 residual on the **independent fills-based** MTM is the only
non-exact figure: it is the average-cost-basis methodology (commissions folded
into basis; trade-date vs IBKR-settlement FX on the CAD sleeve) versus IBKR's
own lot accounting. The statement's five printed truths reconcile to the cent.

## Return basis (both kept)
- **Money-weighted** = (143,115 − 111,683.03) / 111,683.03 = **+28.1%** on
  deposited capital.
- **Time-weighted (TWR)** = **+63.40%** since inception (the `portfolio_history`
  series; the viewer's All-time figure).

## Residual limitations
- `premium` is **gross** credit collected (sold-to-open proceeds), not
  net-of-buyback — matches the demo-fixture convention.
- Strict flat-to-flat *cycle* aggregation isn't emitted: heavily-traded names
  (CLS 560 option fills) hold many concurrent contracts, so the ledger is
  per-contract + per-stock-close with cycle-aware `exit_reason`s rather than one
  row per put→assignment→call episode.

