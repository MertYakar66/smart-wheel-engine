# EDGAR earnings calendar — PIT-correct earnings dates

PR3/9 of the news-architecture redesign campaign — see
`docs/NEWS_REDESIGN_CAMPAIGN.md` for the campaign context.

This doc covers the EDGAR earnings layer added in this PR:

- The connector extension (`EDGARAdapter.recent_8k_filings`,
  `earnings_history`, `project_next_earnings`).
- The puller script (`scripts/pull_edgar_earnings.py`).
- The PIT-correctness story and how it differs from the existing
  yfinance/Bloomberg path.
- **The integration boundary** — what this PR ships vs. what the next
  PR wires.

---

## 1. Why EDGAR for earnings dates

The current `MarketDataConnector.get_next_earnings` reads from
`data/bloomberg/sp500_earnings_yf.csv` — a yfinance snapshot of the
*current* next-earnings date. For live use it works fine; for
historical backtests it silently leaks lookahead.

**The leak:** yfinance only returns the most-recent-known schedule. If
you query "as of 2023-06-15, what's the next earnings date for AAPL?",
yfinance returns the date that was current when the CSV was pulled
(e.g. mid-2026), not what was known to a market participant on
2023-06-15.

**The fix:** EDGAR 8-K filings are immutable historical records. A
Form 8-K with Item 2.02 ("Results of Operations and Financial
Condition") IS the earnings release — the SEC-recorded `filing_date`
is the date the company actually announced. Wiring EDGAR replaces a
forward-looking calendar with a backward-looking record and a
projection heuristic.

This is the same approach professional data vendors (Zacks, Earnings
Whispers) take under the hood — they monitor SEC filings and project
forward.

---

## 2. What this PR ships

### 2.1 Connector extension

`engine/external_data/edgar_adapter.py` — `EDGARAdapter` gains:

| Method | Purpose |
|---|---|
| `recent_8k_filings(ticker, items_filter="2.02", since=None)` | Returns 8-K filings, optionally filtered by item code and lower-bound date. Sorted descending. |
| `earnings_history(ticker, since=None)` | Convenience wrapper around `recent_8k_filings` with the earnings-release item. Sorted **ascending** (oldest first) so callers can compute inter-filing deltas. |
| `project_next_earnings(ticker, as_of=None, min_history=3)` | Forward projection: takes the median inter-filing delta from history `<= as_of` and adds it to the most recent known filing. Returns a dict drop-in compatible with `MarketDataConnector.get_next_earnings`. |

The existing surface (`cik_for_ticker`, `recent_insider_trades`,
`insider_activity_signal`) is unchanged.

### 2.2 Puller

`scripts/pull_edgar_earnings.py` — CLI puller that writes
`data_processed/edgar/earnings_history.parquet`.

```bash
# Single-ticker smoke
python scripts/pull_edgar_earnings.py --tickers AAPL MSFT

# Full S&P 500 universe (~60 s wall-clock at the 10 req/sec SEC limit)
python scripts/pull_edgar_earnings.py --universe sp500

# Constrain depth
python scripts/pull_edgar_earnings.py --universe sp500 --since 2018-01-01

# Re-pull tickers already in the parquet
python scripts/pull_edgar_earnings.py --universe sp500 --refresh
```

Default behaviour is **append-only** — tickers already in the parquet
are skipped. The puller streams progress per ticker so it doesn't
look hung on a long run.

### 2.3 Storage contract

`data_processed/edgar/earnings_history.parquet` columns:

| Column | Type | Notes |
|---|---|---|
| `ticker` | str | upper-case |
| `filing_date` | `datetime[ns]` | SEC-recorded filing date (PIT-correct) |
| `accession` | str | unique SEC accession number; idempotency key |
| `items` | str | comma-separated 8-K item codes; will always contain `2.02` |
| `primary_document` | str | relative URL within the filing |
| `pulled_at` | `datetime[ns]` | UTC timestamp of this puller run |

`(ticker, accession)` is the de-dup key when appending.

### 2.4 Tests

`tests/test_external_data_edgar.py` — 22 new tests covering:

- The items-field parser (`_items_contains`) — whitespace, `Item `
  prefix, comma-separated lists, None safety.
- `recent_8k_filings` filtering — item code, since, network error, empty
  recent block.
- `earnings_history` — ascending sort, only 2.02 filings.
- `project_next_earnings` — **PIT correctness** (a historical `as_of`
  excludes future filings from the projection), forward-roll when the
  naive projection lands at or before `as_of`, min-history threshold,
  shape compatibility with `MarketDataConnector.get_next_earnings`.

All mocked via `requests_mock` — no actual SEC calls in CI.

---

## 3. The PIT correctness story

`project_next_earnings(ticker, as_of=...)` is the PIT-safe surface:

1. Fetches the historical Item 2.02 filings.
2. **Filters to `filing_date <= as_of`** — no future leak.
3. Computes the median inter-filing delta from the filtered set.
4. Adds the median delta to the most recent known filing.
5. If the resulting date is on or before `as_of` (cadence shorter than
   the gap from last filing to `as_of`), rolls forward in
   delta-sized steps until strictly after `as_of`. This ensures the
   EventGate treats the projected event as "upcoming".

Test: `tests/test_external_data_edgar.py::TestProjectNextEarnings::test_pit_uses_only_filings_before_as_of`.

---

## 4. What this PR does NOT do

**Integration with `wheel_runner.py` is deferred to a follow-up PR.**

This PR ships:

- The data layer (connector + puller + parquet schema).
- The PIT-correct projection contract.
- Tests pinning behaviour.

It does NOT:

- Modify `MarketDataConnector.get_next_earnings` to consume EDGAR.
- Modify `wheel_runner.py`'s `conn.get_next_earnings(ticker, as_of)`
  call sites.
- Add an EDGAR-vs-yfinance reconciliation report.

Reason: the integration decision deserves its own design review
(EDGAR replaces yfinance? EDGAR is preferred-with-fallback? Both
sources are surfaced and reconciled per-ticker?). Shipping the data
layer first lets the operator pull the parquet and inspect it before
any code path consumes it.

---

## 5. Operational notes

- **User-Agent:** SEC requires a `User-Agent` header identifying the
  caller. The adapter defaults to a generic one; override via
  `SWE_EDGAR_UA` env var.
- **Rate limit:** SEC permits up to 10 req/sec. The adapter sleeps
  120 ms between calls by default.
- **No API key needed.** EDGAR is free and SEC-mandated public.
- **Bulk pull duration:** S&P 500 (~503 tickers) × ~120 ms ≈ 60 s.
- **Storage size:** ~1 MB for the full universe, 10+ years of history.

---

## 6. What integration looks like (preview)

The next PR will wire `EDGARAdapter.project_next_earnings` into the
existing `conn.get_next_earnings` call sites in `engine/wheel_runner.py`.
Three plausible shapes; the design review on board #113 will pick one:

1. **EDGAR replaces yfinance.** Bloomberg connector's
   `get_next_earnings` reads from the EDGAR parquet instead of the
   yfinance CSV. Pros: simplest. Cons: a single new source.
2. **EDGAR preferred with yfinance fallback.** EDGAR projection is
   used when available (≥ min_history quarters); yfinance backfills
   tickers with insufficient EDGAR history. Pros: never less coverage
   than today. Cons: two sources to reason about.
3. **Both sources surfaced and reconciled.** Row dict carries both
   `next_earnings_edgar` and `next_earnings_yf` for transparency;
   operator decides which to trust. Pros: maximum transparency. Cons:
   doubles row width, defers the decision.

Option 2 is the campaign default unless the user redirects.

---

## 7. References

- Form 8-K Item 2.02 spec: SEC Item Code 2.02 ("Results of Operations
  and Financial Condition")
- SEC EDGAR fair-access policy: https://www.sec.gov/os/accessing-edgar-data
- Existing `EDGARAdapter`: `engine/external_data/edgar_adapter.py`
- Campaign context: `docs/NEWS_REDESIGN_CAMPAIGN.md`
