# DASHBOARD_TRADES.md — the Trades tab (full IBKR trade history)

The `/portfolio` **Trades** tab shows every IBKR execution — buys, sells,
**option expiries and assignments** — with per-ticker realized P&L broken out by
asset class. It answers "how much did I make or lose on <ticker> options?"
accurately, which the live-connector trade feed **cannot** (it omits option
strike/expiry and misses expiries entirely — the connector generates no row when
a short option expires worthless, which is how the wheel earns most of its
premium). The Trades tab therefore sources from **IBKR Flex**, which records
expiries (`Ep`) and assignments (`A`) with FIFO realized P&L.

Everything here is **read-only / observational** (CLAUDE.md §2/§3): it reports
what happened, never a forward score, never an order. Real trade data lives in
the gitignored `data_processed/ibkr/` and is **never committed**.

---

## 1. Pipeline (source → screen)

```
IBKR Flex Web Service (query 1537765, Trades)          ← operator regenerates token
  └─ flex_credentials.json  (gitignored secret: token + query_id + account)
       └─ SendRequest → GetStatement → flex_trades_raw.xml   (uploads/, gitignored)
            └─ scripts/ibkr_trades_ingest.py  →  trades.json (gitignored, schema_version 1)
                 └─ engine.ibkr_portfolio_adapter.load_trades / trades_view
                      └─ engine_api.py  GET /api/portfolio/trades   (source:"live")
                           └─ dashboard proxy /api/portfolio/[sub]  (allowlist)
                                └─ use-trades-data.ts → trades-panel.tsx (Trades tab)
```

## 2. Refresh flow (Dashboard terminal)

1. **Token.** Flex tokens are IP-locked and rotate. Operator regenerates one in
   IBKR Client Portal → Settings → Account Settings → **Flex Web Service**, and
   it is stored at `data_processed/ibkr/flex_credentials.json` (gitignored):
   ```json
   { "token": "<secret>", "query_id": "1537765", "account": "U17853958" }
   ```
   The token is a **secret** — never printed, echoed, or committed.
2. **Fetch.** `SendRequest` → `ReferenceCode`, then poll `GetStatement` (retry on
   `1019 in progress`) → save the raw XML to `data_processed/ibkr/uploads/`.
3. **Ingest.**
   ```bash
   python scripts/ibkr_trades_ingest.py \
     --xml data_processed/ibkr/uploads/flex_trades_raw.xml \
     --out data_processed/ibkr/trades.json --query "1537765 (Last 365d)"
   ```
   The engine re-reads `trades.json` on every request — no restart needed.

## 3. `trades.json` schema (`schema_version: 1`)

Top level: `source` (`ibkr_flex`), `generated_at`, `coverage`
(`{from, to, query, count}`), `trades[]`. Each trade:

| field | meaning |
|---|---|
| `id` | stable hash of the execution fields **+ occurrence ordinal** (this Flex query has no `tradeID`; IBKR fills identical legs of one order as byte-identical rows, so the ordinal keeps them distinct and dedupes cleanly across overlapping pulls) |
| `datetime` / `date` | ISO execution time / calendar date |
| `symbol` | **underlying** ticker (first token of the OCC symbol) |
| `occ_symbol` | full OCC contract string (options only) |
| `sec_type` | `STK` \| `OPT` \| `CASH` |
| `right` / `strike` / `expiry` | option contract detail |
| `side` / `qty` / `price` / `proceeds` / `commission` | execution economics (signed) |
| `realized_pnl` | IBKR `fifoPnlRealized`, in the trade's **native currency** |
| `currency` | native currency (from `ibCommissionCurrency`) |
| `open_close` / `code` | `O`/`C`; `Ep` (expired), `A` (assigned) |

## 4. P&L semantics (read this before trusting a number)

- **Realized, not mark-to-market.** `realized_pnl` is IBKR's FIFO realized P&L on
  each closing execution. Summed per ticker per asset class = realized gain/loss
  over the covered period. Unrealized P&L on still-open positions is **not** here
  (it's on the KPI cards / Holdings).
- **Net of commissions.** Verified: a short put opened for +$145.00 with −$1.50
  commission expires with `fifoPnlRealized = +$143.50`. The realized figure is the
  true bottom line, commissions included — do not subtract them again.
- **Basis is true, not window-truncated.** For a position opened *before* the
  365-day window and closed inside it, `fifoPnlRealized` uses the real historical
  cost basis (IBKR knows the full history), so the realized P&L is correct even
  though the opening trade isn't in the export.
- **Assignment attribution (important).** An *assigned* option realizes `$0` on
  the option row — IBKR folds its premium into the **stock** cost basis (puts) /
  proceeds (calls). So per-ticker **Options P&L captures expired/closed-option
  premium only; assigned-option premium lands in Stock P&L.** The per-ticker
  *total* is correct; only the option-vs-stock split follows this broker
  convention. (For a wheel whose puts get assigned, "options income" is therefore
  split across both columns — e.g. CLS's assigned-put premium sits inside its
  Stock figure.)
- **Currency.** Realized P&L is native; the per-ticker cards and totals report a
  **USD-equivalent** using the live snapshot's FX (`fx_rates`). This applies the
  *current* rate to historical P&L — an approximation the tab labels, not a claim
  of point-in-time FX. A dual-listed name (CLS: NYSE-USD + TSX-CAD) is flagged
  `mixed_currency` and its native `*_pnl` field mixes currencies — trust the
  `*_pnl_usd` figure there.
- **Premium collected** is the gross credit from opening short options
  (sell-to-open), informational — not net of buybacks.

## 5. Coverage / since-inception

Flex query `1537765` is **"Last 365 days."** If the account is older than a year
(closing trades on the window's first day with realized P&L prove earlier opens),
the history is **truncated**, not since-inception. To extend: widen the Flex
query's period (custom date range) or run additional dated pulls — the ingest is
**merge-safe** (`--merge-into`), so older pulls append without double-counting.

## 6. The screen (Trades tab)

Totals strip (Realized / Options / Stock USD + coverage) · **asset-class toggle**
(All / Stock / Options / Cash-FX) · **ticker filter** (dropdown, or click a row in
the per-ticker league table) · **date range** · a selected-ticker card (options
P&L, stock P&L, total, premium) · the filtered trade table (contract detail,
`expired`/`assigned` tags, realized P&L). Filtering is client-side; the P&L
aggregates come from the engine over the full set.

Provenance: the tab shows **Live** when served from a real Flex drop, **Mock**
when the engine is unreachable (typed empty fallback) — never fabricates trades.
