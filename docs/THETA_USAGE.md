# Theta Data — Usage Reference

A practical guide for using Theta Terminal v3 from Smart Wheel Engine.
This file pins what we consume from the Theta Data platform, how the
local Terminal works, our subscription tier behaviour, and the
reference codes we need to parse the wire format. The official Theta
Data documentation is the upstream source of truth; this file captures
the parts the engine cares about plus repo-specific integration notes.

---

## Table of contents

- §1 Theta Terminal — install and run
- §2 Subscriptions and tier access
- §3 Concurrent requests
- §4 Data availability — what to expect
- §5 SIPs — what feeds which
- §6 Index option symbology
- §7 Greeks model
- §8 OHLC + EOD calculation
- §9 Causes of perceived data issues — checklist
- §10 v2 → v3 migration
- §11 MCP server (Theta's own — distinct from the engine MCP)
- §12 Terminal control endpoints
- §13 HTTP error codes
- §14 Reference — exchange codes (77 entries)
- §15 Reference — quote conditions (74 entries)
- §16 Reference — trade conditions (149 entries)
- §17 REST API endpoints
  - §17.1 Conventions
  - §17.2 Endpoint index (master table, ~40 paths)
  - §17.3 Common parameters
  - §17.4 Common response fields
  - §17.5 Option list endpoints (5)
  - §17.6 Option snapshot endpoints (10)
  - §17.7 Option historical endpoints (6)
  - §17.8 Option historical Greeks endpoints (6)
  - §17.9 Option historical Trade Greeks endpoints (5)
  - §17.10 Option at-time endpoints (2)
  - §17.11 Stock REST endpoints (path shape)
  - §17.12 Index REST endpoints (path shape)
  - §17.13 Calendar endpoints (3)
  - §17.14 Quirks + gotchas
  - §17.15 Python library — `thetadata` (alternative to the Terminal)
  - §17.16 WebSocket streaming
  - §17.17 Client code patterns
- §18 Operational pipeline — what to pull and how
  - §18.1 Probe what your subscription unlocks
  - §18.2 Core pulls (run in this order)
  - §18.3 One command does it all
  - §18.4 Heavy / optional pulls
  - §18.5 Verify everything
  - §18.6 Coverage matrix — what's wired and what isn't
  - §18.7 Daily production routine
- §19 Smart Wheel Engine integration notes

Find sections by section number (e.g. `§17.5`) — every heading carries
its number verbatim, so Ctrl-F lands you precisely.

---

## Where Theta lives in this repo

| Where | What |
|---|---|
| `engine/theta_connector.py` | v3 connector class consumed when `SWE_DATA_PROVIDER=theta` |
| `scripts/probe_theta_capabilities.py` | Probes the running Terminal and writes `data_processed/theta_capabilities.json` |
| `scripts/theta_health_check.py` | Connectivity probe + Bloomberg fallback verification |
| `scripts/theta_backfill.py` | Tier-aware bulk backfill with circuit breakers |
| `scripts/pull_theta_*.py` | Per-dataset pullers (chains, IV surface, options flow, indices history, corp actions, vix futures) |
| `data_processed/theta/**` | Backfilled local cache (gitignored) |
| `data_processed/theta_capabilities.json` | Tier map cached from the last probe |

The local server runs at `http://127.0.0.1:25503`. Provider selection
happens in `WheelRunner.connector` (lazy-load property in
`engine/wheel_runner.py`). See `docs/DATA_POLICY.md` §2 for the
provider capability matrix and `LAPTOP_SETUP.md` for rehydration.

---

## Glossary

| Term | Definition |
|---|---|
| **Theta Terminal** | Local Java process that bridges your machine to Theta's servers. Hosts an HTTP server bound to `127.0.0.1:25503`; only the originating IP can hit it. |
| **FIC** (Financial Information Compression) | Theta's proprietary over-the-wire compression; up to 30× bandwidth reduction. |
| **FIT** (Financial Information Tick) | Lowest-level data row; arrays of signed `int32`. |
| **MDDS** (Market Data Distribution Server) | Upstream server the Terminal queries for snapshots and history. |
| **FPSS** (Feed Processing Stream Server) | Upstream server for streaming / continuous updates. |
| **Interp3** | Theta's matching engine that aligns two sets of tick-level data and tracks per-tick dividend yield + interest rate. Supersedes Interp2. |
| **NBBO** | National Best Bid and Offer (best across all exchanges via the SIP). |
| **SIP** | Securities Information Processor; consolidates trades + quotes across exchanges. Relevant SIPs: OPRA (options), CTA-A/B and UTP-C (equities). |
| **CGIF** | CBOE Global Indices Feed — covers SPX, VIX, etc. |

---

## 1. Theta Terminal — install and run

### Java prerequisite

Java **21+ required**. Check with `java -version`. If missing or
older:

- **Mac/Windows:** install from the Oracle website.
- **Ubuntu:** `apt install openjdk-21-jdk openjdk-21-jre`.

### Terminal install

The Terminal is an auto-updating JAR — no installer. Download
`ThetaTerminalv3.jar` and run:

```
java -jar ThetaTerminalv3.jar
```

The Terminal self-updates on **startup only** (never mid-run) and
falls back to the previous version if the new one fails. To pin a
version, run the JAR in `lib/` directly.

### macOS Gatekeeper

On first run macOS may block the JAR. Open
**System Settings → Privacy & Security → scroll to Security →
"Open Anyway"**. Optional sanity check: upload the JAR to
`https://www.virustotal.com/gui/home/upload` first.

### Credentials file

Create `creds.txt` in the same directory as `ThetaTerminalv3.jar`:

```
your_email@example.com
your_password
```

Email on line 1, password on line 2. Alternatively pass
`--creds-file=/path/to/creds.txt`.

### Config file

The Terminal writes a default config on first run. Most settings do
not need to change. The four tunables you might touch:

| Key | Meaning |
|---|---|
| `host` | Bind address for the HTTP server. Don't change unless you know why. |
| `port` | HTTP port (default `25503`). |
| `log_directory` | Where Terminal logs go. |
| `request_queue_length` | Concurrent-request queue depth (default 16, max 128). |

### Verify it's up

The startup banner shows your access level for each data type. From
this repo:

```python
import socket
s = socket.socket(); s.settimeout(1)
s.connect(("127.0.0.1", 25503))
print("UP")
```

Or run `python scripts/theta_health_check.py`.

---

## 2. Subscriptions and tier access

Free tier: 1 year of historical EOD for US stocks/options, 30 req/min
rate cap. Beyond that, tiers gate granularity, history depth, and
concurrency.

### Stock data

| Tier | Granularity | First date | Concurrent | Delay |
|---|---|---|---|---|
| FREE | EOD | 2023-06-01 | 30/min | 1 day |
| VALUE | 1 minute | 2021-01-01 | 2 | 15 min |
| STANDARD | 1 minute | 2016-01-01 | 4 | real-time |
| PRO | tick | 2012-06-01 | 8 | real-time |

UTP tape covers most history back to 2012-06-01. CTA-only symbols
(SPY, GE, etc.) start 2020-01-01.

**Stock historical endpoints:**

| Endpoint | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| EOD report | ✓ | ✓ | ✓ | ✓ |
| Quote |  | ✓ | ✓ | ✓ |
| OHLC |  | ✓ | ✓ | ✓ |
| Splits |  |  | ✓ | ✓ |
| Trades |  |  | ✓ | ✓ |
| Trade-Quote |  |  | ✓ | ✓ |

**Stock real-time endpoints:**

| Endpoint | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| Quote snapshot |  | 15-min | real-time | real-time |
| OHLC snapshot |  | 15-min | real-time | real-time |
| Trade snapshot |  |  | real-time | real-time |
| Bulk quote snapshot |  |  | real-time | real-time |

**Stock streaming (Nasdaq Basic feed):**

| Stream | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| Streamable quote contracts | 0 | 0 | 1,000 | 2,000 |
| Streamable trade contracts | 0 | 0 | 1,000 | 20,000 (full trade stream) |

### Options data

| Tier | Granularity | First date | Concurrent | Delay |
|---|---|---|---|---|
| FREE | EOD | 2023-06-01 | 30/min | 1 day |
| VALUE | 1 minute | 2020-01-01 | 2 | real-time |
| STANDARD | tick | 2016-01-01 | 4 | real-time |
| PRO | tick | 2012-06-01 | 8 | real-time |

**Options historical endpoints:**

| Endpoint | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| EOD | ✓ | ✓ | ✓ | ✓ |
| Quote |  | ✓ | ✓ | ✓ |
| Open Interest |  | ✓ | ✓ | ✓ |
| OHLC |  | ✓ | ✓ | ✓ |
| Trade |  |  | ✓ | ✓ |
| Trade-Quote |  |  | ✓ | ✓ |
| Implied Volatility |  |  | ✓ | ✓ |
| Greeks 1st Order |  |  | ✓ | ✓ |
| Greeks 2nd Order |  |  |  | ✓ |
| Greeks 3rd Order |  |  |  | ✓ |
| Trade Greeks 1st-3rd Order |  |  |  | ✓ |

**Options real-time endpoints:**

| Endpoint | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| Quote |  | ✓ | ✓ | ✓ |
| Open Interest |  | ✓ | ✓ | ✓ |
| OHLC |  | ✓ | ✓ | ✓ |
| Trade |  |  | ✓ | ✓ |

**Options streaming:**

| Stream | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| Streamable quote contracts | 0 | 0 | 10,000 | 15,000 |
| Streamable trade contracts | 0 | 0 | 15,000 | unlimited (full trade stream) |
| Full trade stream |  |  |  | ✓ |

### Index data

Index granularity is whatever the venue publishes (CBOE sends SPX
every second). **If a price didn't change, no new tick is emitted** —
you interpret a missing-tick gap as "no change since last tick."
Indices on the Nasdaq Indices Feed (including `NDX`) are not
supported.

| Tier | Granularity | First date | Delay | Concurrent |
|---|---|---|---|---|
| FREE | no access | — | — | — |
| VALUE | 15-minute | 2023-01-01 | 15-min | 2 |
| STANDARD | venue-lowest | 2022-01-01 | real-time | 4 |
| PRO | venue-lowest | 2017-01-01 | real-time | 8 |

Real-time / ongoing for all CGIF symbols (SPX, VIX, etc.). Historic
only for symbols like RUT and DJX between their first-access date and
2024-07-01. Theta plans synthetic indices data for ~99% accuracy on
NDX-family — not yet available.

**Index endpoints:**

| Endpoint | FREE | VALUE | STANDARD | PRO |
|---|---|---|---|---|
| EOD report |  |  |  |  |
| Price |  | ✓ | ✓ | ✓ |
| Price snapshot |  |  | ✓ | ✓ |
| OHLC snapshot |  |  | ✓ | ✓ |

---

## 3. Concurrent requests

These are **outstanding-request limits**, not request-rate limits.
You can queue beyond the limit (default queue 16, max 128); past the
queue the server returns HTTP 429.

| Tier | Outstanding |
|---|---|
| FREE | 1 |
| VALUE | 2 |
| STANDARD | 4 |
| PRO | 8 |

**Why concurrency matters:** the round-trip to MDDS dominates
per-request latency. Issuing N requests in parallel removes most of
it. Use a semaphore in your client to stay at-or-under the limit.
The repo handles this in `engine/theta_connector.py` and the
bulk-pull scripts.

---

## 4. Data availability — what to expect

### Equities historical depth

UTP-C covers most tickers back to 2012-06-01. CTA-only tickers (SPY,
GE, others) only go back to 2020-01-01 because they aren't on UTP.
There are no Greeks for stocks (Greeks are an options concept).

### Options historical depth

Trades, EOD, and OHLC go back to **2012-06-01**. Greeks for index
options go back to **2017-01-01**. Nasdaq-indices options (NDX) are
not covered. For NDX Greeks specifically, you can pass `under_price`
to a Greeks snapshot to force Theta to use a supplied underlying
price.

### Historical ETH OPRA gaps

Extended Trading Hours (ETH) trades + quotes:

- **2015 → 2018:** available
- **2019 → Dec 2022:** **not available** (vendor-side gap)
- **Jan 2022 onward:** full GTH/ETH coverage

ETH is only emitted for SPX, VIX, DJI, and RUT options. Equity
options are unaffected.

### Bulk date listing

Only quotes/trades support bulk date listing. Other request types
share the same dates as quotes/trades, so listing them isn't needed.

### Trade sequence overflow

The trade sequence is a signed 32-bit int. It overflows at
`2_147_483_647`. When the sequence reads `-1`, you're at
`4_294_967_294`. Once it reaches `0` for a second time (initial
value is also `0`), the upstream sequence has overflowed — handle
the wrap.

---

## 5. SIPs — what feeds which

A SIP (Securities Information Processor) consolidates trades and
quotes across exchanges so brokers can hit the NBBO.

### Options: OPRA

OPRA (Options Price Reporting Authority) — administered by CBOE —
provides nationally consolidated quote and trade data for US equity
and index options. Theta receives every NBBO quote and trade with
sub-3 ms average latency. Most OPRA vendors *filter* NBBO quotes
because they cannot keep up; Theta does not.

#### OPRA Global Trading Hours (GTH)

SPX, VIX, and XSP options trade outside RTH on the GTH session
(20:15 ET → 09:25 ET next day). The last GTH session of the week
starts Sunday 20:15 ET. As of 2024-08-26 the daily GTH end shifted
from 09:15 → 09:25 ET.

| Session | Start | End |
|---|---|---|
| Begin GTH order acceptance (SPX/VIX/XSP) | 20:00 | — |
| GTH (SPX/VIX/XSP) | 20:15 | 09:25 next day |
| Begin RTH order acceptance | 07:30 | — |
| RTH | 09:30 | 16:15 |
| Curb | 16:15 | 17:00 |

#### OPRA ETH (extended trading hours, until 16:15 ET)

The following symbols trade until 16:15 ET as part of OPRA ETH:

```
AUM, AUX, BACD, BPX, BRB, BSZ, BVZ, CDD, CITD, DBA, DBB, DBC, DBO,
DBS, DIA, DJX, EEM, EFA, EUI, EUU, GAZ, GBP, GSSD, IWM, IWN, IWO,
IWV, JJC, JPMD, KBE, KRE, MDY, MLPN, MNX, MOO, MRUT, MSTD, NDO, NDX,
NZD, OEF, OEX, OIL, PZO, QQQ, RUT, RVX, SFC, SKA, SLX, SPX,
SPX (PM expiration), SPY, SVXY, UNG, UUP, UVIX, UVXY, VIIX, VIX,
VIXM, VIXY, VXEEM, VXST, VXX, VXZ, XEO, XHB, XLB, XLE, XLF, XLI,
XLK, XLP, XLU, XLV, XLY, XME, XRT, XSP, XSP (AM expiration), YUK
```

### Equities: CTA + UTP

US equities run on two SIPs across three networks:

- CTA Network A (NYSE-administered)
- CTA Network B (NYSE-administered)
- UTP Network C (Nasdaq-administered)

Theta receives a 15-minute-delayed feed from all three. It receives a
real-time feed from **Nasdaq Basic**.

CTA-A trading hours:

| Session | Start | End |
|---|---|---|
| Pre-opening | 06:30 | 09:30 |
| Core open auction | 09:30 | 15:50 |
| 123(c) closing imbalance | 15:50 | 16:00 |

### Nasdaq Basic (real-time equities)

A BBO feed for all US exchange-listed securities, within 1% of NBBO
**99.22% of the time**. Also a real-time time-and-sales feed for
trades on the Nasdaq execution system + FINRA/Nasdaq TRF.

> **Quirk Theta filters out:** at session close, Nasdaq Basic emits
> a zero-ed BBO. Theta drops it before it reaches your snapshot, so
> you get the last legitimate value rather than zeros.

### CGIF (CBOE Global Indices Feed)

Real-time feed for SPX, VIX, and family. Theta receives this feed
directly.

---

## 6. Index option symbology

Settlement style splits some index option tickers across multiple
symbols:

| Index (AM-settled) | Weekly (PM) | Quarterly | PM-settled | Other |
|---|---|---|---|---|
| SPX | SPXW | SPXQ | SPXPM | |
| VIX | VIXW | | | |
| RUT | RUTW | RUTQ | | |
| NDX | | | NDXP | |
| DJIA | | | | |
| OEX | | | | |
| RUI | | | | |
| XSPA | XSP | | | XSPPM, XSPAM |

**Notes:**

- SPXPM is **deprecated as of 2018-12-21**.
- SPXQ is **deprecated as of 2014-07-02**.
- Prior to 2025-04-22, NDXP did not have 3rd-Friday standard monthly
  expirations.
- Prior to 2022-05-16, SPXW (SPX weekly) was quoted Mon/Wed/Fri only,
  not every weekday.

---

## 7. Greeks model

### Pricing model

**Black-Scholes-Merton (European)**, tick-by-tick. Theta computes
Greeks for each tick using the exact underlying tick at that moment.
Formula reference: standard BSM EU pricer.

### Implied volatility solver

Fast bisection. For deep ITM/OTM contracts where a clean solution
doesn't exist, the `iv_error` field will rise. This is a property of
the solver, not the data feed — other providers using the same model
exhibit the same behaviour.

### Unit conventions

**Rho and Vega must be divided by 100** to get conventional values.
Other Greeks are reported in their natural BSM units. (Compare with
`docs/GREEKS_UNIT_CONTRACT.md` for this repo's enforced units.)

### Dividends

Theta **ignores dividends by default** in the BSM calculation. Pass
`annual_div` (v2) / `annual_dividend` (v3) to specify an annual
dividend amount; Theta then computes the dividend yield per tick and
plugs it in.

### Risk-free rate

Default rate is **SOFR** (reported one day late). For current-day
Greeks, Theta uses the most recently reported SOFR. Override with
`rate` (v2) / `rate_value` (v3). v3 also has a `rate_type`
parameter that defaults when `rate_value` is omitted.

### DTE convention

- **Default (v2):** for contracts <7 DTE, DTE is computed from the
  quote timestamp. For contracts ≥7 DTE, a whole-number DTE is used.
- **Legacy (`version=1`):** same-day expiration uses DTE = 0.15;
  all other cases use whole-number DTE.

### When Greeks are unreliable

Greeks may not work as expected and are **unsupported outside
regular trading hours**. Option quotes and underlying prices stop and
start updating at different times across venues, which causes
unstable values during pre/post-market.

---

## 8. OHLC + EOD calculation

### OHLC mechanics

OHLC is computed from trade ticks. Some trade conditions disqualify a
tick from contributing — e.g. a late report at $90 when the stock is
trading at $100 should be ignored. Theta filters by trade condition
(see §15) to avoid that class of outlier.

### Three OHLC methods

1. **SIP-rule filtered (Theta v2 default).** Each trade's eligibility
   to update price-and-volume vs price-only is governed by its
   condition code. Less rigorous providers skip this filter and end
   up with inflated volume.
2. **Min/max over all SIP trades.** Simplest. Produces extreme highs
   and lows on bad ticks; *not recommended*.
3. **Hybrid.** SIP-rule filter, but admit a trade if its price is
   "close enough" to NBBO or last. Higher complexity; not used by
   Theta.

Theta v2 requests use **method 1**.

### EOD methods

The two equity SIPs (UTP, CTA) each generate EOD reports around 17:00
ET. OPRA's EOD reports are per-participant-exchange and unreliable.
Theta therefore generates a **normalised EOD report at 17:15 ET** for
all asset classes — same shape regardless of class, easier
cross-referencing.

1. Use the SIP-emitted EOD reports. Reliable for equities, not for
   options.
2. **Synthesise an EOD at 17:15 ET.** Theta v2 default for all asset
   classes.

### Notable behaviour

- "Missing" EOD or zero-ed OHLC bars on illiquid contracts =
  **no trades occurred during that bar**, not a data gap.
- You can compute your own OHLC from raw trades + condition codes if
  you need different filter logic.

---

## 9. Causes of perceived data issues — checklist

When data looks wrong, walk this list before opening a support
ticket.

### General

- Are you on V3? V1/V2 paths are deprecated; this repo's connector
  is v3-only.
- Was the market closed (holiday, weekend, halt)?
- Was trading halted on the symbol that day?
- Theta's previous-day data is **unavailable from 00:00 ET to 01:45
  ET** during the midnight reset.
- Was there a SIP / exchange issue? Check CBOE / CTA / UTP system
  notices for the date.
- For data outside RTH, set `start_time` / `end_time` explicitly.

### Equities

- Nasdaq Basic emits a zero-ed BBO at session close. Theta filters
  these — you should not see zeroes here.

### Options-specific

- Most option contracts list 4–12 weeks before expiration. A weekly
  expiration won't have multiple years of data — that's *expected*,
  not missing.
- **SPX is AM-settled**, so there's no data on the expiration date
  itself. SPXW is PM-settled.
- For index options, use the proper index option symbols OPRA
  reports (see §6).
- Strike prices in the wire format are in **10ths of a cent** —
  $140.00 = `140000`. Use this format when sending requests.
- 0DTE deep ITM/OTM strikes may not be quoted/traded.
- OPRA emits zero bid/ask quotes during premarket. This is normal.
- Test data with expiration year 1882 was historically sent by OPRA;
  drop it.
- Prior to 2022-05-16, SPXW was Mon/Wed/Fri only.
- For 0DTE index options, deep OTM/ITM strikes that aren't divisible
  by 2.5/5/10 stop being quoted/traded toward end of day.
- **Greeks outside RTH:** unsupported and unstable. Do not consume.
- Prior to 2025-04-22, NDXP didn't have 3rd-Friday monthly
  expirations.
- Prior to 2018-12-03, the lowest resolution for option quotes was 1 s.

---

## 10. v2 → v3 migration

v3 unified the schema across asset classes, simplified parameters,
expanded wildcard support, and added several new endpoints.
**This repo's `engine/theta_connector.py` is v3-only.**

### Deprecations from v2

| v2 endpoint / feature | v3 status | Notes |
|---|---|---|
| `/v2/hist/stock/split` | coming soon | TBD replacement |
| `/v2/hist/stock/dividend` | coming soon | TBD replacement |
| Bulk strikes/expirations | new behaviour | use `*` wildcard for `strike` and `expiration` |

### New in v3

- **Options:** Implied Volatility snapshot endpoint; full Option
  history; Trade IV history; wildcard support on `expiration` and
  `strike`.
- **Index:** index price at-time endpoint.
- **Market calendar:** on-date, current-day, year-holidays endpoints.

### Endpoint mapping — list

| v2 | v3 | Notes |
|---|---|---|
| `/v2/list/roots/{sec}` | `/v3/{sec}/list/symbols` | `sec` ∈ `stock`/`option`/`index` |
| `/v2/list/dates/{sec}/{req}` | `/v3/{sec}/list/dates/{req}` | `req` ∈ `trade`/`quote` |
| `/v2/list/expirations` | `/v3/option/list/expirations` | |
| `/v2/list/strikes` | `/v3/option/list/strikes` | |
| `/v2/list/contracts/option/{req}` | `/v3/option/list/contracts/{req}` | `req` ∈ `trade`/`quote` |

### Endpoint mapping — stock

| v2 | v3 | Notes |
|---|---|---|
| `/v2/hist/stock/eod` | `/v3/stock/history/eod` | |
| `/v2/hist/stock/{req}` | `/v3/stock/history/{req}` | `req` ∈ `trade`/`quote`/`ohlc`/`trade_quote`. **No more `start_date`/`end_date` — returns one day.** |
| `/v2/at_time/stock/{req}` | `/v3/stock/at_time/{req}` | `req` ∈ `trade`/`quote` |
| `/v2/snapshot/stock/{req}` | `/v3/stock/snapshot/{req}` | `req` ∈ `trade`/`quote`/`ohlc` |
| `/v2/bulk_snapshot/stock/{req}` | use v3 snapshot with `symbol=*` | |

### Endpoint mapping — options

| v2 | v3 | Notes |
|---|---|---|
| `/v2/hist/option/eod` | `/v3/option/history/eod` | |
| `/v2/snapshot/option/{req}` | `/v3/option/snapshot/{req}` | `strike`/`right` no longer required (returns bulk if omitted); `expiration=*` for bulk |
| `/v2/hist/option/{req}` | `/v3/option/history/{req}` | `req` ∈ `trade`/`quote`/`ohlc`/`open_interest`/`trade_quote` |
| `/v2/hist/option/greeks_{req}` | `/v3/option/history/greeks/{req}` | `req` ∈ `first_order`/`second_order`/`third_order` |
| `/v2/hist/option/trade_greeks_{req}` | `/v3/option/history/trade_greeks/{req}` | same |
| `/v2/at_time/option/{req}` | `/v3/option/at_time/{req}` | |

### Endpoint mapping — index

| v2 | v3 | Notes |
|---|---|---|
| `/v2/snapshot/index/{req}` | `/v3/index/snapshot/{req}` | `req` ∈ `ohlc`/`price` |
| `/v2/hist/index/{req}` | `/v3/index/history/{req}` | `req` ∈ `ohlc`/`eod` |
| `/v2/hist/index/price` | `/v3/index/history/price` | one day per request |

### Parameter mapping

| v2 param | v3 param | Notes |
|---|---|---|
| `pretty_time` | — | v3 uses pretty timestamps by default |
| `rth` | — | use `start_time` / `end_time` filters |
| `root` | `symbol` | direct symbol identifier |
| `exp` | `expiration` | accepts `YYYYMMDD`, `YYYY-MM-DD`, or `*` |
| `right` | `right` | values changed from `['C','P']` → `['call','put']` |
| `strike` | `strike` | now a float in dollars; supports `*` |
| `ivl` | `interval` | millisecond int → string (`1m`, `5m`, `1h`) |
| `use_csv` | `format` | options: `csv`, `ndjson`, `json`, `html` |
| `req` | `request_type` | now a path segment |
| `under_price` | `stock_price` | |
| `annual_div` | `annual_dividend` | |
| `rate` | `rate_value` | |
| — | `rate_type` | new; defaults when `rate_value` is omitted |

### OpenAPI source of truth

The `openapiv3.yaml` file from Theta is the canonical schema. You
can feed it to LLM CLIs (Gemini CLI: `@openapiv3.yaml`) for
natural-language access to the API surface.

---

## 11. MCP server (Theta's own — distinct from the engine MCP)

> **Don't confuse this with `docs/TRADINGVIEW_MCP_INTEGRATION.md`.**
> Theta's MCP is for ad-hoc data queries via natural language. **It
> is not part of the EV decision path** — using it cannot bypass
> `EVEngine.evaluate`. See `CLAUDE.md` §2.

Theta Terminal v3 exposes an MCP server at
`http://127.0.0.1:25503/mcp/sse` so you can ask LLM CLIs (Claude CLI,
Gemini CLI) for data in plain English. **A subscription is required.**

### Setup — Gemini CLI

1. `npm install -g @google/gemini-cli`
2. Run `gemini` once to authenticate.
3. Add to `~/.gemini/config.json`:

```json
{
  "mcpServers": {
    "Theta Data": {
      "url": "http://127.0.0.1:25503/mcp/sse",
      "timeout": 30000
    }
  }
}
```

### Setup — Claude CLI

1. `npm install -g @claude-ai/claude-cli`
2. Run `claude` once to authenticate.
3. `claude mcp add --transport sse ThetaData http://127.0.0.1:25503/mcp/sse`

### Verifying

Inside the LLM CLI, run `/mcp` to confirm the server is connected.

### Example prompts

> "Get the EOD greek for last week for AAPL strike 200.00 CALL and
> expiration 2025-08-01."

> "Put this in a table showing the delta change."

### Tips

- Be explicit: `symbol`, `expiration` as `YYYY-MM-DD`, `right` as
  `C`/`P`, `strike` as decimal (`200.00`), date ranges.
- For large results, narrow to specific dates / contracts.
- Add formatting instructions ("table", "CSV") to the prompt.

### Troubleshooting

- **Cannot connect / timeouts** — Terminal not running, or port
  25503 blocked by firewall/VPN.
- **CLI doesn't see the MCP server** — check the config-file path,
  restart the CLI after starting the Terminal.

---

## 12. Terminal control endpoints

These are not data endpoints — they manage the running Terminal.

### Shutdown

`GET http://127.0.0.1:25503/v3/terminal/shutdown` — kills the
Terminal process. Returns `OK`. **Use with caution.**

### MDDS status

`GET http://127.0.0.1:25503/v3/terminal/mdds/status`

| Response | Meaning |
|---|---|
| `CONNECTED` | Terminal is connected to MDDS. |
| `UNVERIFIED` | Connected but credentials failed authentication. |
| `DISCONNECTED` | Not connected. |
| `ERROR` | Other error — check the Terminal log. |

### FPSS status

`GET http://127.0.0.1:25503/v3/terminal/fpss/status` — same response
shape as MDDS, but for the streaming feed processor.

The repo's `scripts/theta_health_check.py` polls these endpoints.

---

## 13. HTTP error codes

These match `engine/theta_connector.py` retry/fallback logic. Code
**478 (`INVALID_SESSION_ID`) is the v3-specific addition** vs. v2.

| Code | Name | Meaning |
|---|---|---|
| 200 | `OKAY` | No error. |
| 404 | `NO_IMPL` | Endpoint not implemented; check Terminal version. |
| 429 | `OS_LIMIT` | OS-level rate throttling. Retry. |
| 470 | `GENERAL` | Generic error. |
| 471 | `PERMISSION` | Subscription tier insufficient. |
| 472 | `NO_DATA` | No data for the request. **Handle this — don't crash.** |
| 473 | `INVALID_PARAMS` | Bad parameters. Try updating the Terminal. |
| 474 | `DISCONNECTED` | Lost connection to MDDS. **Handle this — don't crash.** |
| 475 | `TERMINAL_PARSE` | Terminal failed to parse the request. |
| 476 | `WRONG_IP` | Don't switch between `127.0.0.1` and `localhost` mid-session. |
| 477 | `NO_PAGE_FOUND` | Page expired. |
| 478 | `INVALID_SESSION_ID` | Multiple Terminals running. |
| 570 | `LARGE_REQUEST` | Request too large; chunk it. |
| 571 | `SERVER_STARTING` | Terminal restarting. Retry after a delay. |
| 572 | `UNCAUGHT_ERROR` | Contact Theta support with the exact request. |

---

## 14. Reference — exchange codes

| Code | Symbol | Name |
|---|---|---|
| 1 | NQEX | Nasdaq Exchange |
| 2 | NQAD | Nasdaq Alternative Display Facility |
| 3 | NYSE | New York Stock Exchange |
| 4 | AMEX | American Stock Exchange |
| 5 | CBOE | Chicago Board Options Exchange |
| 6 | ISEX | International Securities Exchange |
| 7 | PACF | NYSE ARCA (Pacific) |
| 8 | CINC | National Stock Exchange (Cincinnati) |
| 9 | PHIL | Philadelphia Stock Exchange |
| 10 | OPRA | Options Pricing Reporting Authority |
| 11 | BOST | Boston Stock/Options Exchange |
| 12 | NQNM | Nasdaq Global+Select Market (NMS) |
| 13 | NQSC | Nasdaq Capital Market (SmallCap) |
| 14 | NQBB | Nasdaq Bulletin Board |
| 15 | NQPK | Nasdaq OTC |
| 16 | NQIX | Nasdaq Indexes (GIDS) |
| 17 | CHIC | Chicago Stock Exchange |
| 18 | TSE | Toronto Stock Exchange |
| 19 | CDNX | Canadian Venture Exchange |
| 20 | CME | Chicago Mercantile Exchange |
| 21 | NYBT | New York Board of Trade |
| 22 | MRCY | ISE Mercury |
| 23 | COMX | COMEX (division of NYMEX) |
| 24 | CBOT | Chicago Board of Trade |
| 25 | NYMX | New York Mercantile Exchange |
| 26 | KCBT | Kansas City Board of Trade |
| 27 | MGEX | Minneapolis Grain Exchange |
| 28 | NYBO | NYSE/ARCA Bonds |
| 29 | NQBS | Nasdaq Basic |
| 30 | DOWJ | Dow Jones Indices |
| 31 | GEMI | ISE Gemini |
| 32 | SIMX | Singapore International Monetary Exchange |
| 33 | FTSE | London Stock Exchange |
| 34 | EURX | Eurex |
| 35 | IMPL | Implied Price |
| 36 | DTN | Data Transmission Network |
| 37 | LMT | London Metals Exchange Matched Trades |
| 38 | LME | London Metals Exchange |
| 39 | IPEX | Intercontinental Exchange (IPE) |
| 40 | NQMF | Nasdaq Mutual Funds (MFDS) |
| 41 | fcec | COMEX Clearport |
| 42 | C2 | CBOE C2 Option Exchange |
| 43 | MIAX | Miami Exchange |
| 44 | CLRP | NYMEX Clearport |
| 45 | BARK | Barclays |
| 46 | EMLD | Miami Emerald Options Exchange |
| 47 | NQBX | NASDAQ Boston |
| 48 | HOTS | HotSpot Eurex US |
| 49 | EUUS | Eurex US |
| 50 | EUEU | Eurex EU |
| 51 | ENCM | Euronext Commodities |
| 52 | ENID | Euronext Index Derivatives |
| 53 | ENIR | Euronext Interest Rates |
| 54 | CFE | CBOE Futures Exchange |
| 55 | PBOT | Philadelphia Board of Trade |
| 56 | FCME | CME Floor |
| 57 | NQNX | FINRA/NASDAQ Trade Reporting Facility |
| 58 | BTRF | BSE Trade Reporting Facility |
| 59 | NTRF | NYSE Trade Reporting Facility |
| 60 | BATS | BATS Trading |
| 61 | FCBT | CBOT Floor |
| 62 | PINK | Pink Sheets |
| 63 | BATY | BATS Y Exchange |
| 64 | EDGE | Direct Edge A |
| 65 | EDGX | Direct Edge X |
| 66 | RUSL | Russell Indexes |
| 67 | CMEX | CME Indexes |
| 68 | IEX | Investors Exchange |
| 69 | PERL | Miami Pearl Options Exchange |
| 70 | LSE | London Stock Exchange |
| 71 | GIF | NYSE Global Index Feed |
| 72 | TSIX | TSX Indexes |
| 73 | MEMX | Members Exchange |
| 74 | EMPT | (empty) |
| 75 | LTSE | Long-Term Stock Exchange |
| 76 | EMPT | (empty) |
| 77 | 24X | 24X National Exchange |

---

## 15. Reference — quote conditions

`Firm` = a firm quote; `Halted` = associated with a trading halt.

| Code | Name | Firm | Halted |
|---|---|---|---|
| 0 | REGULAR | x | |
| 1 | BID_ASK_AUTO_EXEC | x | |
| 2 | ROTATION | | |
| 3 | SPECIALIST_ASK | x | |
| 4 | SPECIALIST_BID | x | |
| 5 | LOCKED | x | |
| 6 | FAST_MARKET | | |
| 7 | SPECIALIST_BID_ASK | x | |
| 8 | ONE_SIDE | x | |
| 9 | OPENING_QUOTE | | |
| 10 | CLOSING_QUOTE | | |
| 11 | MARKET_MAKER_CLOSED | | |
| 12 | DEPTH_ON_ASK | x | |
| 13 | DEPTH_ON_BID | x | |
| 14 | DEPTH_ON_BID_ASK | x | |
| 15 | TIER_3 | x | |
| 16 | CROSSED | x | |
| 17 | HALTED | | x |
| 18 | OPERATIONAL_HALT | | x |
| 19 | NEWS_OUT | | x |
| 20 | NEWS_PENDING | | x |
| 21 | NON_FIRM | | |
| 22 | DUE_TO_RELATED | | x |
| 23 | RESUME | | |
| 24 | NO_MARKET_MAKERS | | x |
| 25 | ORDER_IMBALANCE | | x |
| 26 | ORDER_INFLUX | | x |
| 27 | INDICATED | | x |
| 28 | PRE_OPEN | | |
| 29 | IN_VIEW_OF_COMMON | | x |
| 30 | RELATED_NEWS_OUT | | x |
| 32 | ADDITIONAL_INFO | | x |
| 33 | RELATED_ADD_INFO | | x |
| 34 | NO_OPEN_RESUME | | x |
| 35 | DELETED | | x |
| 36 | REGULATORY_HALT | | x |
| 37 | SEC_SUSPENSION | | x |
| 38 | NON_COMPLIANCE | | x |
| 39 | FILINGS_NOT_CURRENT | | x |
| 40 | CATS_HALTED | | x |
| 41 | CATS | | |
| 42 | EX_DIV_OR_SPLIT | x | |
| 43 | UNASSIGNED | | |
| 44 | INSIDE_OPEN | | |
| 45 | INSIDE_CLOSED | | |
| 46 | OFFER_WANTED | | |
| 47 | BID_WANTED | | |
| 48 | CASH | x | |
| 49 | INACTIVE | x | |
| 50 | NATIONAL_BBO | x | |
| 51 | NOMINAL | x | |
| 52 | CABINET | x | |
| 53 | NOMINAL_CABINET | x | |
| 54 | BLANK_PRICE | x | |
| 55 | SLOW_BID_ASK | | |
| 56 | SLOW_LIST | x | |
| 57 | SLOW_BID | | |
| 58 | SLOW_ASK | | |
| 59 | BID_OFFER_WANTED | | |
| 60 | SUBPENNY | | |
| 61 | NON_BBO | | |
| 62 | SPECIAL_OPEN | | |
| 63 | BENCHMARK | | |
| 64 | IMPLIED | | |
| 65 | EXCHANGE_BEST | | |
| 66 | MKT_WIDE_HALT_1 | | |
| 67 | MKT_WIDE_HALT_2 | | |
| 68 | MKT_WIDE_HALT_3 | | |
| 69 | ON_DEMAND_AUCTION | | |
| 70 | NON_FIRM_BID | | |
| 71 | NON_FIRM_ASK | | |
| 72 | RETAIL_BID | | |
| 73 | RETAIL_ASK | | |
| 74 | RETAIL_QTE | | |

---

## 16. Reference — trade conditions

148 entries. Columns indicate whether the condition is a cancel, late
report, auto-executed, opening report; and whether it updates volume,
high, low, last (asterisk = "updates last only if it's the only/first
qualifying trade").

| Code | Name | Cancel | Late | Auto | Open | Vol | High | Low | Last | Description |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | REGULAR | | | | | x | x | x | x | Regular trade. |
| 1 | FORM_T | | | | | x | | | | Pre/post-market. (NYSE/AMEX previously used 'T' for BurstBasket.) |
| 2 | OUT_OF_SEQ | | x | | | x | x | x | * | Out-of-sequence report. Updates last if it becomes only trade. |
| 3 | AVG_PRC | | | | | x | | | | Average Price (NYSE/AMEX). |
| 4 | AVG_PRC_NASDAQ | | | | | x | | | | Average Price (Nasdaq). Does not set high/low/last. |
| 5 | OPEN_REPORT_LATE | | x | | | x | x | x | * | Late open report. *update last if only trade. |
| 6 | OPEN_REPORT_OUT_OF_SEQ | | x | | | x | x | x | | Out-of-sequence open report. |
| 7 | OPEN_REPORT_IN_SEQ | | x | | | x | x | x | x | In-sequence opening report. First price. |
| 8 | PRIOR_REFERENCE_PRICE | | x | | | x | x | x | * | References an earlier price. *update last if only trade. |
| 9 | NEXT_DAY_SALE | | | | | x | | | | NYSE/AMEX next-day clearing; Nasdaq 1–4 day delivery. (Discontinued at NYSE 2017-09-05.) |
| 10 | BUNCHED | | | | | x | x | x | x | Aggregate of 2+ Regular trades at same price within 60s, each ≤10,000 shares. |
| 11 | CASH_SALE | | | | | x | | | | Same-day delivery + payment. (Discontinued at NYSE 2017-09-05.) |
| 12 | SELLER | | | | | x | | | | Stock can be delivered up to 60 days later. (Discontinued at NYSE 2017-09-05.) |
| 13 | SOLD_LAST | | x | | | x | x | x | * | Late report. *Sets consolidated last under specific conditions. |
| 14 | RULE_127 | | | | | x | x | x | x | NYSE Rule 127 (block trade). |
| 15 | BUNCHED_SOLD | | x | | | x | x | x | * | Bunched + late. *Update last if first trade. |
| 16 | NON_BOARD_LOT | | | | | x | | | | Odd lot (less than a board lot). Canadian markets. |
| 17 | POSIT | | | | | x | x | x | | POSIT Canada — trades priced at bid/ask midpoint. |
| 18 | AUTO_EXECUTION | | | x | | x | x | x | x | Electronic execution. Common in OPRA. |
| 19 | HALT | | | | | | | | | Trading halt. |
| 20 | DELAYED | | | | | x | | | | Delayed opening. |
| 21 | REOPEN | | | | | x | x | x | x | Reopening of a halted contract. |
| 22 | ACQUISITION | | | | | x | x | x | x | Exchange Acquisition. |
| 23 | CASH_MARKET | | | | | x | x | x | x | Cash-only market session — all trades settled in cash. |
| 24 | NEXT_DAY_MARKET | | | | | x | x | x | x | Next-day-only market session. |
| 25 | BURST_BASKET | | | | | x | x | x | x | Specialist trade as part of basket execution. |
| 26 | OPEN_DETAIL | | x | | | | | | | (107-113, 130, 160 deleted condition.) Detail trade making up an open report. |
| 27 | INTRA_DETAIL | | x | | | | | | | Detail trade making up a previous trade. |
| 28 | BASKET_ON_CLOSE | | x | | | x | | | | Paired basket order priced on closing index value. |
| 29 | RULE_155 | | | | | x | x | x | x | AMEX Rule 155 — block at one cleanup price. |
| 30 | DISTRIBUTION | | | | | x | x | x | x | Sale of large block without adverse price impact. |
| 31 | SPLIT | | | | | x | x | x | x | Two-market split execution. |
| 32 | REGULAR_SETTLE | | | | | x | x | x | x | Regular settle. |
| 33 | CUSTOM_BASKET_CROSS | | | | | x | | | | Two paired basket orders, MM facilitates. |
| 34 | ADJ_TERMS | | | | | x | x | x | x | Terms adjusted for split/dividend. |
| 35 | SPREAD | | | | | x | x | x | x | Spread between 2 options in same class. |
| 36 | STRADDLE | | | | | x | x | x | x | Straddle between 2 options in same class. |
| 37 | BUY_WRITE | | | | | x | x | x | x | Option leg of a covered call. |
| 38 | COMBO | | | | | x | x | x | x | Buy + sell across 2+ options in same class. |
| 39 | STPD | | | | | x | x | x | x | Stopped trade priced post-non-stopped trade. |
| 40 | CANC | x | | | | | | | | Cancel a previously reported trade (not first/last). |
| 41 | CANC_LAST | x | | | | | | | | Cancel most recent trade qualifying as last. |
| 42 | CANC_OPEN | x | | | | | | | | Cancel opening trade. |
| 43 | CANC_ONLY | x | | | | | | | | Cancel the only trade. |
| 44 | CANC_STPD | x | | | | | | | | Cancel a STPD trade. |
| 45 | MATCH_CROSS | | | | | x | x | x | x | Cross trade from market center crossing session. |
| 46 | FAST_MARKET | | | | | x | x | x | x | Hectic market conditions. |
| 47 | NOMINAL | | | | | x | x | x | x | Nominal price (calculated, e.g. for inactive futures/options). |
| 48 | CABINET | | | x | | | | | | Deep OTM option at one-half tick value. |
| 49 | BLANK_PRICE | | | | | | | | | Blanks the bid/ask/trade price. |
| 50 | NOT_SPECIFIED | | | | | | | | | Generic/unspecified. |
| 51 | MC_OFFICIAL_CLOSE | | | | | | | | | Market Center official close. |
| 52 | SPECIAL_TERMS | | | | | x | x | x | x | Settled in non-regular manner. |
| 53 | CONTINGENT_ORDER | | | | | x | x | x | x | Contingent on execution of another order. |
| 54 | INTERNAL_CROSS | | | | | x | x | x | x | Cross between two client accounts under one PM (TSX origin). |
| 55 | STOPPED_REGULAR | | | | | x | x | x | x | Stopped Stock — Regular Trade. |
| 56 | STOPPED_SOLD_LAST | | | | | | x | x | x | Stopped Stock — SoldLast. |
| 57 | STOPPED_OUT_OF_SEQ | | x | | | x | x | x | | Stopped Stock — Out of sequence. |
| 58 | BASIS | | | | | x | x | x | x | Basket/index trade priced via related derivatives. |
| 59 | VWAP | | | | | x | | | | Volume Weighted Average Price. |
| 60 | SPECIAL_SESSION | | | | | x | | | | Last-sale priced order in Special Trading Session. |
| 61 | NANEX_ADMIN | | | | | | | | | Nanex admin volume/price corrections. |
| 62 | OPEN_REPORT | | | | | x | x | x | | Opening trade report. |
| 63 | MARKET_ON_CLOSE | | | | | x | x | x | x | Market Center official close. |
| 64 | SETTLE_PRICE | | | | | | | | | Settlement price. |
| 65 | OUT_OF_SEQ_PRE_MKT | | x | | | x | | | | Out-of-sequence pre/post-market (FormT + OutOfSeq). |
| 66 | MC_OFFICIAL_OPEN | | | | | | | | | Market Center official open. |
| 67 | FUTURES_SPREAD | | | | | x | x | x | x | Futures spread leg. |
| 68 | OPEN_RANGE | | | | | | x | x | | Opening range high/low (~first 30s). |
| 69 | CLOSE_RANGE | | | | | | x | x | | Closing range high/low (~last 30s). |
| 70 | NOMINAL_CABINET | | | | | | | | | Nominal cabinet. |
| 71 | CHANGING_TRANS | | | | | x | x | x | x | Changing transaction. |
| 72 | CHANGING_TRANS_CAB | | | | | | | | | Changing cabinet transaction. |
| 73 | NOMINAL_UPDATE | | | | | | | | | Nominal price update. |
| 74 | PIT_SETTLEMENT | | | | | | | | | Pit-session settlement price for electronic session net-change. |
| 75 | BLOCK_TRADE | | | | | x | x | x | x | Block trade (≥10,000 shares typical). |
| 76 | EXG_FOR_PHYSICAL | | | | | x | x | x | x | Exchange For Physical (futures). |
| 77 | VOLUME_ADJUSTMENT | | | | | x | | | | Cumulative volume adjustment. |
| 78 | VOLATILITY_TRADE | | | | | x | x | x | x | Volatility trade. |
| 79 | YELLOW_FLAG | | | | | x | x | x | x | Reporting venue may have technical issues. |
| 80 | FLOOR_PRICE | | | | | x | x | x | x | Floor bid/ask vs member bid/ask (LME). |
| 81 | OFFICIAL_PRICE | | | | | x | x | x | x | LME official bid/ask. |
| 82 | UNOFFICIAL_PRICE | | | | | x | x | x | x | LME unofficial bid/ask. |
| 83 | MID_BID_ASK_PRICE | | | | | x | x | x | x | LME midpoint. |
| 84 | END_SESSION_HIGH | | | | | | x | | | End-of-session high. |
| 85 | END_SESSION_LOW | | | | | | | x | | End-of-session low. |
| 86 | BACKWARDATION | | | | | x | x | x | x | Spot > futures (opposite of contango). |
| 87 | CONTANGO | | | | | x | x | x | x | Futures > spot. |
| 88 | HOLIDAY | | | | | x | x | x | x | (in development) |
| 89 | PRE_OPENING | | | | | x | | | | Pre-opening order entry period. |
| 90 | POST_FULL | | | | | | | | | (no flags) |
| 91 | POST_RESTRICTED | | | | | | | | | (no flags) |
| 92 | CLOSING_AUCTION | | | | | | | | | (no flags) |
| 93 | BATCH | | | | | | | | | (no flags) |
| 94 | TRADING | | | | | | | | | (no flags) |
| 95 | INTERMARKET_SWEEP | | | | | x | x | x | x | Intermarket Sweep Order execution. See SEC NMS regulation 2005-06-29. |
| 96 | DERIVATIVE | | | | | x | x | x | * | Derivatively priced. |
| 97 | REOPENING | | | | | x | x | x | x | Market center re-opening prints. |
| 98 | CLOSING | | | | | x | x | x | * | Market center closing prints (e.g. NYSE closing auction). |
| 99 | CAPELECTION | | | | | x | x | x | | Cap election trade / Odd Lot trade (CTA redefinition). |
| 100 | SPOT_SETTLEMENT | | | | | x | x | x | x | Spot settlement. |
| 101 | BASIS_HIGH | | | | | x | x | x | | Basis high. |
| 102 | BASIS_LOW | | | | | x | x | x | | Basis low. |
| 103 | YIELD | | | | | | | | | Cantor Treasuries bid/ask yield updates. |
| 104 | PRICE_VARIATION | | | | | | | | | (no flags) |
| 105 | CONTINGENT_TRADE | | | | | x | | | | Contingent on an event (effective 2015-07; previously StockOption). |
| 106 | STOPPED_IM | | | | | x | x | x | | Order stopped at non-trade-through price. |
| 107 | BENCHMARK | | | | | | | | x | Benchmark trade — eligible for OHLLV on tapes A/B/UTP; volume-only on OPRA. |
| 108 | TRADE_THRU_EXEMPT | | | | | | | | x | Trade-Through-Exempt indicator set. |
| 109 | IMPLIED | | | | | x | | | | Spread leg from a futures spread. Updates volume, not OHLL. |
| 110 | OTC | | | | | | | | | (no flags) |
| 111 | MKT_SUPERVISION | | | | | | | | | (no flags) |
| 112 | RESERVED_77 | | | | | | | | | reserved |
| 113 | RESERVED_91 | | | | | | | | | reserved |
| 114 | CONTINGENT_UTP | | | | | | | | | (no flags) |
| 115 | ODD_LOT | | | | | x | | | | Trade with size 1–99. |
| 116 | RESERVED_89 | | | | | | | | | reserved |
| 117 | CORRECTED_CS_LAST | | | | | | x | x | x | Mechanism to correct official close on consolidated tape. |
| 118 | OPRA_EXT_HOURS | | | | | | | | | Pre-market OPRA extended-hours session ('X' indicator). Obsolete; see 148. |
| 119 | RESERVED_78 | | | | | | | | | reserved |
| 120 | RESERVED_81 | | | | | | | | | reserved |
| 121 | RESERVED_84 | | | | | | | | | reserved |
| 122 | RESERVED_878 | | | | | | | | | reserved |
| 123 | RESERVED_90 | | | | | | | | | reserved |
| 124 | QUALIFIED_CONTINGENT_TRADE | | | | | x | | | | QCT — multi-component contingent trade per SEC rules (effective 2015-07). |
| 125 | SINGLE_LEG_AUCTION_NON_ISO | | | | | x | x | x | x | Single-leg auction (non-ISO). |
| 126 | SINGLE_LEG_AUCTION_ISO | | | | | x | x | x | x | Single-leg auction (ISO). |
| 127 | SINGLE_LEG_CROSS_NON_ISO | | | | | x | x | x | x | Single-leg cross (non-ISO). |
| 128 | SINGLE_LEG_CROSS_ISO | | | | | x | x | x | x | Single-leg cross (ISO). |
| 129 | SINGLE_LEG_FLOOR_TRADE | | | | | x | x | x | x | Non-electronic floor trade. |
| 130 | MULTI_LEG_AUTOELEC_TRADE | | | | | x | x | x | x | Electronic multi-leg complex trade. |
| 131 | MULTI_LEG_AUCTION | | | | | x | x | x | x | Multi-leg auction in complex order book. |
| 132 | MULTI_LEG_CROSS | | | | | x | x | x | x | Multi-leg cross (e.g. customer-customer, QCC). |
| 133 | MULTI_LEG_FLOOR_TRADE | | | | | x | x | x | x | Non-electronic multi-leg floor trade. |
| 134 | ML_AUTO_ELEC_TRADE_AGSL | | | | | x | x | x | x | Multi-leg electronic vs single-leg orders/quotes. |
| 135 | STOCK_OPTIONS_AUCTION | | | | | x | x | x | x | Stock-options multi-leg auction. |
| 136 | ML_AUCTION_AGSL | | | | | x | x | x | x | Multi-leg auction vs single-leg orders/quotes. |
| 137 | ML_FLOOR_TRADE_AGSL | | | | | x | x | x | x | Non-electronic multi-leg floor trade vs single-leg orders/quotes. |
| 138 | STK_OPT_AUTO_ELEC_TRADE | | | | | x | x | x | x | Stock-options electronic multi-leg in complex order book. |
| 139 | STOCK_OPTIONS_CROSS | | | | | x | x | x | x | Stock-options multi-leg cross. |
| 140 | STOCK_OPTIONS_FLOOR_TRADE | | | | | x | x | x | x | Non-electronic stock-options multi-leg in complex order book. |
| 141 | STK_OPT_AE_TRD_AGSL | | | | | x | x | x | x | Stock-options electronic multi-leg vs single-leg. |
| 142 | STK_OPT_AUCTION_AGSL | | | | | x | x | x | x | Stock-options multi-leg auction vs single-leg. |
| 143 | STK_OPT_FLOOR_TRADE_AGSL | | | | | x | x | x | x | Non-electronic stock-options multi-leg vs single-leg. |
| 144 | ML_FLOOR_TRADE_OF_PP | | | | | x | x | x | x | Multi-leg proprietary product floor trade (≥3 legs). May trade outside NBBO. |
| 145 | BID_AGGRESSOR | | | | | x | x | x | x | Buy-side aggressor. |
| 146 | ASK_AGGRESSOR | | | | | x | x | x | x | Sell-side aggressor. |
| 147 | MULTILAT_COMP_TR_PDP | | | | | x | | | | Multilateral compression trade in proprietary product (outside RTH, derived end-of-day prices). |
| 148 | EXTENDED_HOURS_TRADE | | | | | x | | | | Trade executed outside regular market hours. |

---

## 17. REST API endpoints

This section enumerates the v3 endpoints the Terminal exposes, plus the
two alternative client surfaces Theta ships: the **`thetadata` Python
library** (gRPC, no Terminal needed) and the **WebSocket streaming
endpoint** (FPSS, on the Terminal). The Smart Wheel Engine connector
in `engine/theta_connector.py` consumes the REST endpoints below; the
Python library and WebSocket are documented here as alternative
surfaces, not as the current integration path.

### 17.1 Conventions

- **Base URL:** `http://127.0.0.1:25503/v3`
- **Auth:** none at the HTTP layer — the Terminal authenticates upstream
  via `creds.txt`. If MDDS is `UNVERIFIED`, requests will fail with
  permission errors.
- **Output formats** (`format` query param):
  `csv` (default) · `json` · `ndjson` · `html`
- **Streaming responses:** every documented endpoint can be consumed
  line-by-line via `httpx.stream("GET", ...)`. Recommended for any
  multi-day or full-chain pull (see §17.17).
- **Strike encoding:** the REST response returns `strike` in **dollars
  as float** (`180.00`). The **wire format** used by the WebSocket
  protocol (§17.16) and exchange messages encodes strike in **1/10 of a
  cent** — i.e. `$140` → `140000`.
- **Time zone:** all wall-clock times are America/New_York. Timestamps
  in the response are formatted `YYYY-MM-DDTHH:mm:ss.SSS`.
- **Multi-day limits:** historical option endpoints (`history/ohlc`,
  `history/trade`, `history/quote`, `history/trade_quote`,
  `history/greeks/*`, `history/trade_greeks/*`) are capped at
  **1 month per request** and **must specify an expiration** (no `*`).
- **Bulk requests:** wherever `expiration=*` is supported, the
  endpoint must be called day-by-day. The wildcard works on a single
  date but cannot be combined with `start_date..end_date`.

### 17.2 Endpoint index

Every documented v3 endpoint, grouped by family. The `T` column shows
the minimum tier (`F`ree / `V`alue / `S`tandard / `P`ro).

| Family | Path | T |
|---|---|---|
| **List** | `/v3/option/list/symbols` | F |
| | `/v3/option/list/dates/{trade\|quote}` | F |
| | `/v3/option/list/expirations` | F |
| | `/v3/option/list/strikes` | F |
| | `/v3/option/list/contracts/{trade\|quote}` | V |
| **Snapshot** | `/v3/option/snapshot/ohlc` | V |
| | `/v3/option/snapshot/trade` | S |
| | `/v3/option/snapshot/quote` | V |
| | `/v3/option/snapshot/open_interest` | V |
| | `/v3/option/snapshot/market_value` | S |
| | `/v3/option/snapshot/greeks/implied_volatility` | S |
| | `/v3/option/snapshot/greeks/all` | P |
| | `/v3/option/snapshot/greeks/first_order` | S |
| | `/v3/option/snapshot/greeks/second_order` | P |
| | `/v3/option/snapshot/greeks/third_order` | P |
| **History** | `/v3/option/history/eod` | F |
| | `/v3/option/history/ohlc` | V |
| | `/v3/option/history/trade` | S |
| | `/v3/option/history/quote` | V |
| | `/v3/option/history/trade_quote` | S |
| | `/v3/option/history/open_interest` | V |
| **History Greeks** (interval) | `/v3/option/history/greeks/eod` | S |
| | `/v3/option/history/greeks/all` | P |
| | `/v3/option/history/greeks/first_order` | S |
| | `/v3/option/history/greeks/second_order` | P |
| | `/v3/option/history/greeks/third_order` | P |
| | `/v3/option/history/greeks/implied_volatility` | S |
| **History Trade Greeks** (per-trade) | `/v3/option/history/trade_greeks/all` | P |
| | `/v3/option/history/trade_greeks/first_order` | P |
| | `/v3/option/history/trade_greeks/second_order` | P |
| | `/v3/option/history/trade_greeks/third_order` | P |
| | `/v3/option/history/trade_greeks/implied_volatility` | P |
| **At-time** | `/v3/option/at_time/trade` | S |
| | `/v3/option/at_time/quote` | V |
| **Calendar** | `/v3/calendar/today` | F |
| | `/v3/calendar/on_date` | V |
| | `/v3/calendar/year_holidays` | V |
| **Stock** | `/v3/stock/list/{symbols,dates}`, `/v3/stock/snapshot/{ohlc,trade,quote,market_value}`, `/v3/stock/history/{eod,ohlc,trade,quote,trade_quote}`, `/v3/stock/at_time/{trade,quote}` | varies |
| **Index** | `/v3/index/list/{symbols,dates}`, `/v3/index/snapshot/{ohlc,price,market_value}`, `/v3/index/history/{eod,ohlc,price}`, `/v3/index/at_time/price` | varies |
| **Terminal control** | `/v3/terminal/{shutdown,mdds/status,fpss/status}` | — |

(Stock and Index path shapes mirror the option family. The Python
library — §17.15 — documents the exact parameter list for each.)

### 17.3 Common parameters

Defined once here; per-endpoint sections list only the additional or
overridden parameters.

| Param | Required | Type | Default | Description |
|---|---|---|---|---|
| `symbol` | yes (most) | string | — | Underlying. Some list endpoints accept a comma-separated array; some accept `*`. |
| `expiration` | yes (most option endpoints) | string | — | `YYYYMMDD` or `YYYY-MM-DD`; `*` for all expirations (where supported). |
| `strike` | no | string | `*` | Dollars (`140.00`); `*` for all strikes. |
| `right` | no | enum | `both` | `call` · `put` · `both`. |
| `max_dte` | no | int | — | Filter to contracts with calendar-day DTE ≤ this. |
| `strike_range` | no | int | — | Strikes-around-spot filter; returns N above + N below + ATM. |
| `format` | no | enum | `csv` | `csv` · `json` · `ndjson` · `html`. |
| `start_date` | yes (history, at-time) | string | — | Inclusive. |
| `end_date` | yes (history, at-time) | string | — | Inclusive. |
| `date` | no | string | — | Single-day shortcut for some history endpoints; **overrides `start_date`/`end_date` if present**. |
| `start_time` | no | string | `09:30:00` | `HH:MM:SS.SSS`. |
| `end_time` | no | string | `16:00:00` | `HH:MM:SS.SSS`. |
| `time_of_day` | yes (at-time) | string | — | `HH:mm:ss.SSS`, ET. |
| `interval` | yes (some history) | enum | `1s` | `tick`, `10ms`, `100ms`, `500ms`, `1s`, `5s`, `10s`, `15s`, `30s`, `1m`, `5m`, `10m`, `15m`, `30m`, `1h`. **Sub-1m intervals only allowed on single-day requests.** |
| `min_time` | no | string | — | Snapshots only — filter to ts ≥ this `HH:mm:ss.SSS`. |

**Greeks-only:**

| Param | Type | Default | Description |
|---|---|---|---|
| `version` | enum | `latest` | `latest` = real TTE down to 1 hour minimum; `1` = legacy fixed `0.15` DTE for 0DTE. |
| `rate_type` | enum | `sofr` | `sofr` · `treasury_m1` · `treasury_m3` · `treasury_m6` · `treasury_y1` · `treasury_y2` · `treasury_y3` · `treasury_y5` · `treasury_y7` · `treasury_y10` · `treasury_y20` · `treasury_y30`. |
| `rate_value` | float | — | Override rate, expressed as a percent. |
| `annual_dividend` | float | 0 | Annualised dividend amount; if provided, Theta computes per-tick yield and plugs into BSM. |
| `stock_price` | float | — | Override the underlying price (snapshots only). |
| `use_market_value` | bool | `false` | Use market-value bid/ask/price instead of NBBO. |
| `underlyer_use_nbbo` | bool | `false` | Only on `history/greeks/eod`: `true` = midpoint, `false` = last trade. |

**Trade-quote-only:**

| Param | Type | Default | Description |
|---|---|---|---|
| `exclusive` | bool | `true` | If `true`, match quotes with `ts < trade_ts`; if `false`, match with `ts ≤ trade_ts`. |

**Stock-only:**

| Param | Type | Default | Description |
|---|---|---|---|
| `venue` | enum | `nqb` | `nqb` (Nasdaq Basic, real-time) or `utp_cta` (15-min delayed). |

### 17.4 Common response fields

Used across multiple endpoints; per-endpoint sections list only what is
*added* on top of the common identifiers.

**Identifiers** (on every option response):

| Field | Type | Description |
|---|---|---|
| `symbol` | string | Underlying. |
| `expiration` | date | `YYYY-MM-DD`. |
| `strike` | number | Dollars. |
| `right` | string | `call` · `put`. |
| `timestamp` | datetime | `YYYY-MM-DDTHH:mm:ss.SSS`, ET. |

**Quote fields** (NBBO):

| Field | Type | Description |
|---|---|---|
| `bid_size`, `ask_size` | int | NBBO sizes. |
| `bid_exchange`, `ask_exchange` | int | Exchange codes — see §14. |
| `bid`, `ask` | number | NBBO prices. |
| `bid_condition`, `ask_condition` | int | Quote-condition codes — see §15. |

**Trade fields**:

| Field | Type | Description |
|---|---|---|
| `sequence` | int | Exchange sequence (32-bit signed; can overflow — see §4 / Trade sequence overflow). |
| `condition` | int | Trade-condition code — see §16. |
| `ext_condition1` … `ext_condition4` | int | Additional trade conditions. **Not reported by OPRA for options — ignore.** |
| `size` | int | Contracts/shares. |
| `exchange` | int | Exchange code — see §14. |
| `price` | number | Trade price. |

**OHLCV fields**:

| Field | Type | Description |
|---|---|---|
| `open`, `high`, `low`, `close` | number | Trade prices for the bar. |
| `volume` | int | Sum of trade sizes. |
| `count` | int | Number of trades. |
| `vwap` | number | Volume-weighted average price (history only). |

**Greeks fields** (BSM; see §7 for unit conventions):

- *First order:* `delta`, `theta`, `vega`, `rho`, `epsilon`, `lambda`
- *Second order:* `gamma`, `vanna`, `charm`, `vomma`, `veta`, `vera`
- *Third order:* `speed`, `zomma`, `color`, `ultima`
- *Auxiliary:* `d1`, `d2`, `dual_delta`, `dual_gamma`
- *IV:* `implied_vol`, `iv_error`, `bid_implied_vol`, `ask_implied_vol`,
  `midpoint`
- *Underlying:* `underlying_timestamp`, `underlying_price`

> **Vega and Rho must be divided by 100** to convert to conventional
> units. (See §7.)
>
> *Upstream typing quirk:* some Greek fields (`theta`, `epsilon`,
> `vanna`, `charm`, `color`, `ultima`, `dual_delta`, `iv_error`,
> sometimes `d1`/`d2`) are documented as `string` in upstream response
> schemas — typed inconsistently across endpoints. They are **numeric
> values in practice**; the `string` typing is an upstream
> documentation artifact, not a real wire format. Cast / parse them
> as numbers in your client.

**EOD-only** (regular EOD endpoints only — `option/history/eod` and `stock/history/eod`. The `option/history/greeks/eod` variant uses `timestamp` instead and does **not** carry these two fields):

| Field | Type | Description |
|---|---|---|
| `created` | datetime | When Theta generated the EOD report (17:15 ET). |
| `last_trade` | datetime | Time of the last contributing trade. |

---

### 17.5 Option list endpoints

Updated overnight. Used for symbol/expiration/strike discovery before
issuing a snapshot or history call.

#### `GET /v3/option/list/symbols`
**Tier:** Free+. **No params** beyond `format`. **Returns:** array of `{ symbol }`.

#### `GET /v3/option/list/dates/{request_type}`
**Tier:** Free+. `{request_type}` ∈ `trade` · `quote`.
**Required:** `symbol`, `expiration`.
**Optional:** `strike`, `right`, `format`.
**Returns:** array of `{ date }`.

#### `GET /v3/option/list/expirations`
**Tier:** Free+.
**Required:** `symbol` (string or comma-separated list, supports `*`).
**Returns:** array of `{ symbol, expiration }`.

#### `GET /v3/option/list/strikes`
**Tier:** Free+.
**Required:** `symbol` (or comma list / `*`), `expiration`.
**Returns:** array of `{ symbol, strike }`.

#### `GET /v3/option/list/contracts/{request_type}`
**Tier:** Value+. `{request_type}` ∈ `trade` · `quote`. Updated **real-time**.
**Required:** `date`.
**Optional:** `symbol` (or comma list), `max_dte`, `format`.
**Returns:** array of `{ symbol, expiration, strike, right }`.

> Use this when you want every contract that traded/quoted on a given
> day, e.g. to enumerate the universe before pulling EOD data.

---

### 17.6 Option snapshot endpoints

All snapshots **return no data on weekends** and reset at midnight ET.
For weekend / overnight access, use history endpoints instead.

#### `GET /v3/option/snapshot/ohlc`
**Tier:** Value+. Real-time current-day OHLCV for option contracts.
**Required:** `symbol`, `expiration` (or `*`).
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `min_time`, `format`.
**Returns** *(per row)*: identifiers + `open`, `high`, `low`, `close`, `volume`, `count`.

#### `GET /v3/option/snapshot/trade`
**Tier:** Standard+. Last trade for the contract.
**Required:** `symbol`, `expiration`.
**Optional:** `strike`, `right`, `strike_range`, `min_time`, `format`.
**Returns:** identifiers + trade fields (`sequence`, `condition`, `size`, `exchange`, `price`, `ext_condition1..4`).

#### `GET /v3/option/snapshot/quote`
**Tier:** Value+. Last NBBO quote.
**Required:** `symbol`, `expiration`.
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `min_time`, `format`.
**Returns:** identifiers + quote fields.

#### `GET /v3/option/snapshot/open_interest`
**Tier:** Value+. Last OI message; OPRA reports OI ~06:30 ET reflecting *previous* trading day.
**Required:** `symbol`, `expiration`.
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `min_time`, `format`.
**Returns:** identifiers + `open_interest`.

#### `GET /v3/option/snapshot/market_value`
**Tier:** Standard+. Real-time market value derived from the last NBBO.
**Required:** `symbol`, `expiration`.
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `min_time`, `format`.
**Returns:** identifiers + `market_bid`, `market_ask`, `market_price`.

#### `GET /v3/option/snapshot/greeks/implied_volatility`
**Tier:** Standard+. IV from NBBO bid / mid / ask.
**Required:** `symbol`, `expiration`.
**Optional (Greeks):** `annual_dividend`, `rate_type`, `rate_value`, `stock_price`, `version`, `use_market_value`. Plus the standard option filters.
**Returns:** identifiers + `bid`, `ask`, `implied_vol`, `iv_error`, `underlying_timestamp`, `underlying_price`.

> The history variant of this endpoint (§17.8) returns `bid_implied_vol`
> / `midpoint` / `ask_implied_vol` separately. The snapshot returns a
> single `implied_vol` (mid-based).

#### `GET /v3/option/snapshot/greeks/all`
**Tier:** Pro. Full Greeks (1st + 2nd + 3rd order + auxiliaries).
**Params:** same as `implied_volatility` snapshot.
**Returns:** identifiers + `bid`, `ask` + first-order + second-order + third-order + `d1`, `d2`, `dual_delta`, `dual_gamma` + `implied_vol`, `iv_error`, `underlying_timestamp`, `underlying_price`.

#### `GET /v3/option/snapshot/greeks/first_order`
**Tier:** Standard+. Only first-order (`delta`, `theta`, `vega`, `rho`, `epsilon`, `lambda`) plus `bid`, `ask`, `implied_vol`, `iv_error`, underlying.

#### `GET /v3/option/snapshot/greeks/second_order`
**Tier:** Pro. Only second-order (`gamma`, `vanna`, `charm`, `vomma`, `veta`) plus `bid`, `ask`, `implied_vol`, `iv_error`, underlying.

> Note: `vera` is in the `all` and `eod` responses but not in the
> `second_order` snapshot.

#### `GET /v3/option/snapshot/greeks/third_order`
**Tier:** Pro. Only third-order (`speed`, `zomma`, `color`, `ultima`) plus `bid`, `ask`, `implied_vol`, `iv_error`, underlying.

---

### 17.7 Option historical endpoints

Multi-day requests are limited to **1 month** and require an explicit
expiration (no wildcard).

#### `GET /v3/option/history/eod`
**Tier:** Free+. Theta-generated EOD report (17:15 ET).
**Required:** `start_date`, `end_date`, `symbol`, `expiration` (or `*`, day-by-day).
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + `created`, `last_trade` + OHLCV (`open`, `high`, `low`, `close`, `volume`, `count`) + last NBBO quote fields.

#### `GET /v3/option/history/ohlc`
**Tier:** Value+. SIP-rule-filtered interval OHLC (see §8 for OHLC method).
Bar timestamp = bar open. A trade belongs to bar `t` iff `t ≤ trade_ts < t + interval`.
**Required:** `symbol`, `expiration`, `interval`. **And** either `date` (single day) or `start_date` + `end_date` (≤ 1 month).
**Optional:** `strike`, `right`, `start_time`, `end_time`, `strike_range`, `format`.
**Returns:** identifiers + OHLCV + `vwap`.

#### `GET /v3/option/history/trade`
**Tier:** Standard+. Every trade reported by OPRA.
**Required:** `symbol`, `expiration` (or `*`, single-day only). And `date` or `start_date`+`end_date`.
**Optional:** `strike`, `right`, `start_time`, `end_time`, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + trade fields. `ext_condition1..4` are present but **always 0/ignorable** for options.

#### `GET /v3/option/history/quote`
**Tier:** Value+. Every NBBO quote reported by OPRA, **or** the last quote at each interval boundary if `interval` is specified.
**Required:** `symbol`, `expiration` (or `*`, single-day only), `interval`. And `date` or `start_date`+`end_date`.
**Optional:** `strike`, `right`, `start_time`, `end_time`, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + quote fields.

#### `GET /v3/option/history/trade_quote`
**Tier:** Standard+. Each trade paired with the prior NBBO quote.
**Required:** `symbol`, `expiration` (or `*`, single-day only). And `date` or `start_date`+`end_date`.
**Optional:** `strike`, `right`, `start_time`, `end_time`, `exclusive` (default `true`), `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + `trade_timestamp`, `quote_timestamp`, trade fields, quote fields.

> Theta recommends `exclusive=true` ("yields better results for various
> applications") — this is the **default**. With `exclusive=false`, a
> quote with the *same* timestamp as the trade will be paired with it.
>
> *Upstream caveat:* Theta's prose on this endpoint claims the
> "default" matches with `≤`, but its parameter table says
> `Default: true` (i.e. strict `<`). The param table is the
> authoritative source; treat default behaviour as `<`. If you need
> `≤` matching for a specific use case, set `exclusive=false`
> explicitly rather than relying on omission.

#### `GET /v3/option/history/open_interest`
**Tier:** Value+. OI snapshots (one per day per contract).
**Required:** `symbol`, `expiration` (or `*`). And `date` or `start_date`+`end_date`.
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + `open_interest`.

> OPRA may not send a new OI message if there is no OI for the contract.
> Empty rows in OI history are usually genuine ("contract had zero OI"),
> not a data gap.

---

### 17.8 Option historical Greeks endpoints

Calculated using **midpoint** of the option NBBO (or `use_market_value=true`)
and the underlying tick at each interval. **Multi-day ≤ 1 month;
no expiration wildcard.** All Greeks endpoints accept the standard
Greeks parameters from §17.3 (`version`, `rate_type`, `rate_value`,
`annual_dividend`).

#### `GET /v3/option/history/greeks/eod`
**Tier:** Standard+. EOD-snapshot Greeks (closing option price + closing underlying).
**Required:** `symbol`, `expiration` (or `*`, day-by-day), `start_date`, `end_date`.
**Optional:** `strike`, `right`, `underlyer_use_nbbo` (`false` = last trade, `true` = NBBO mid), Greeks params, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + OHLCV + last NBBO + first/second/third-order Greeks + `d1`, `d2`, `dual_delta`, `dual_gamma` + `implied_vol`, `iv_error`, `underlying_timestamp`, `underlying_price`.

#### `GET /v3/option/history/greeks/all`
**Tier:** Pro. Full-set Greeks at each `interval`.
**Required:** `symbol`, `expiration` (no `*`), `interval`. And `date` or `start_date`+`end_date`.
**Optional:** standard option filters, Greeks params, `start_time`, `end_time`, `strike_range`, `format`.
**Returns:** identifiers + `bid`, `ask` + full Greeks set + `implied_vol`, `iv_error`, underlying.

#### `GET /v3/option/history/greeks/first_order` · `.../second_order` · `.../third_order`
**Tier:** Standard / Pro / Pro respectively. Same params as `greeks/all`. Returns the matching subset of Greeks (see §17.4).

> Same `vera` quirk as the snapshot family (§17.6): `vera` appears in
> `history/greeks/all` and `history/greeks/eod` but **not** in the
> standalone `history/greeks/second_order` response. Request `all` or
> `eod` if you need it.

#### `GET /v3/option/history/greeks/implied_volatility`
**Tier:** Standard+. IV with bid / mid / ask broken out.
**Required:** `symbol`, `expiration`, `interval`. And `date` or `start_date`+`end_date`.
**Returns:** identifiers + `bid`, `bid_implied_vol`, `midpoint`, `implied_vol`, `ask`, `ask_implied_vol`, `iv_error`, underlying.

---

### 17.9 Option historical Trade Greeks endpoints

Greeks computed at **every trade** (not at intervals). `expiration=*`
is allowed for single-day requests. Multi-day requests still ≤ 1 month
and require an explicit expiration. All accept the standard Greeks
params.

#### `GET /v3/option/history/trade_greeks/all`
**Tier:** Pro. Full Greeks per trade.
**Required:** `symbol`, `expiration`. And `date` or `start_date`+`end_date`.
**Returns:** identifiers + trade fields + full Greeks set + `implied_vol`, `iv_error`, underlying.

#### `GET /v3/option/history/trade_greeks/first_order` · `.../second_order` · `.../third_order`
**Tier:** Pro. Trade fields + the matching Greeks subset.

#### `GET /v3/option/history/trade_greeks/implied_volatility`
**Tier:** Pro. IV computed from each **trade** price.
**Returns:** identifiers + trade fields + `implied_vol`, `iv_error`, underlying.

---

### 17.10 Option at-time endpoints

Return the last trade or quote prior to a specific millisecond on each
day in a date range.

#### `GET /v3/option/at_time/trade`
**Tier:** Standard+.
**Required:** `symbol`, `start_date`, `end_date`, `time_of_day`, `expiration` (or `*`).
**Optional:** `strike`, `right`, `max_dte`, `strike_range`, `format`.
**Returns:** identifiers + trade fields.

#### `GET /v3/option/at_time/quote`
**Tier:** Value+. Same shape as `at_time/trade`, returns quote fields instead.

> `time_of_day` resolves to America/New_York. `0DTE` contracts may have
> no trade prior to a very early time (premarket); the endpoint then
> returns no row for that day.

---

### 17.11 Stock REST endpoints (path shape)

Theta's docs do not duplicate the parameter detail for stock endpoints
because the Python library — §17.15 — covers them. The HTTP path
shape mirrors the option family. URLs:

| Family | Path |
|---|---|
| List | `/v3/stock/list/symbols`, `/v3/stock/list/dates/{trade\|quote}` |
| Snapshot | `/v3/stock/snapshot/{ohlc,trade,quote,market_value}` |
| History | `/v3/stock/history/{eod,ohlc,trade,quote,trade_quote}` |
| At-time | `/v3/stock/at_time/{trade,quote}` |

**Stock-specific notes:**
- `venue` parameter (default `nqb`) selects Nasdaq Basic (real-time)
  or `utp_cta` (15-min delayed). Standard+ tier required for `nqb`
  real-time.
- Stock streaming uses **BBO** (Nasdaq Basic only), not NBBO.
- Multi-day stock history limited to 1 month per request.

For exact parameter and response detail, see §17.15
(`stock_history_eod`, `stock_history_ohlc`, etc.) — the Python library
methods document the same surface area.

---

### 17.12 Index REST endpoints (path shape)

Same convention — Python library (§17.15) is the parameter source of
truth.

| Family | Path |
|---|---|
| List | `/v3/index/list/symbols`, `/v3/index/list/dates` |
| Snapshot | `/v3/index/snapshot/{ohlc,price,market_value}` |
| History | `/v3/index/history/{eod,ohlc,price}` |
| At-time | `/v3/index/at_time/price` |

**Index-specific notes:**
- Indices on the **CGIF** feed publish on price-change only — there is
  no tick when the price stays the same. A "missing" tick = "no
  change since last tick" (see §4 / §5).
- Indices on the Nasdaq Indices feed (including `NDX`) are **not
  supported** by Theta as of v3.
- Index `history/price` accepts `interval`. When set, the row at each
  timestamp is the price at that exact moment, not an aggregate.

---

### 17.13 Calendar endpoints

Equity-market schedule. Holiday data covers `2012-01-01` through the
end of the year following the current calendar year.

#### `GET /v3/calendar/today`
**Tier:** Free+. Current-day schedule.
**No params** beyond `format`.
**Returns:** `{ type, open, close }` where `type` ∈ `open` · `full_close` · `early_close` · `weekend`. Times are `HH:mm:ss`.

#### `GET /v3/calendar/on_date`
**Tier:** Value+.
**Required:** `date`.
**Returns:** same shape as `today`.

#### `GET /v3/calendar/year_holidays`
**Tier:** Value+.
**Required:** `year`.
**Returns:** array of `{ date, type, open, close }` where `type` ∈ `full_close` · `early_close`.

> On early-close days (1:00 PM ET), eligible options trade until 1:15
> PM. Some NYSE exchanges continue late trading until 5:00 PM ET on
> early-close days. Plan for this when filtering bar data by trading
> hours.

---

### 17.14 Quirks + gotchas

Worth keeping in mind when integrating against the REST API. These are
in addition to §9 (general data-issue checklist):

- **Multi-day historical option calls require an expiration** (no `*`).
  For full-chain history, loop date-by-date with `expiration=*`.
- **Sub-1-minute intervals only on single-day requests.** The 1-month
  cap on multi-day requests forces minimum `interval=1m`.
- **Snapshot cache resets at midnight ET.** After the reset, snapshot
  endpoints return no data until live data starts flowing in (i.e.
  market open). On weekends and holidays they return no data all day.
  Use history endpoints when you need last-trading-day values
  overnight. Related: §9 documents that previous-day data is
  unavailable from 00:00 ET to 01:45 ET (Theta's announced midnight
  reset gap).
- **OI is reported around 06:30 ET and reflects the *previous* day**.
  If you query OI snapshot during market hours, you are reading
  yesterday's OI.
- **`expiration=*` is single-day only on most snapshot/history
  endpoints.** When in a date range, loop day-by-day in your client
  (the orchestrator at `scripts/pull_all.py` already does this).
- **`bid_implied_vol`, `ask_implied_vol`, and `midpoint` only appear
  on `history/greeks/implied_volatility`** — the corresponding
  *snapshot* endpoint returns only `implied_vol` (mid-based).
- **`vera` Greek**: appears in `history/greeks/eod` and
  `history/greeks/all`, but not in the standalone `second_order`
  endpoint. If you need `vera`, request `all` or `eod`.
- **Trade `condition` codes** are the same enum as §16. `ext_condition1..4`
  are reserved for non-options venues — for options they are emitted
  but always carry the default value; treat them as ignorable.
- **`exclusive=true` is the default** for `trade_quote` endpoints.
  Quotes with `quote_ts == trade_ts` are *excluded* from the match
  (per Theta's recommendation). Set `exclusive=false` only if you
  specifically want simultaneous-timestamp matching.
- **Strike encoding mismatch**: REST returns dollars (`140.00`); the
  WebSocket protocol and OPRA wire format use 1/10-cent
  (`140000`). Be careful when joining streamed data with REST history.

---

### 17.15 Python library — `thetadata` (alternative to the Terminal)

Theta also publishes a Python package (`pip install thetadata`) that
connects directly to Theta servers over **HTTPS + gRPC**. **It does
not require the Terminal to be running.** Returns Polars or Pandas
DataFrames per call.

> **This is a different deployment topology** from the REST + Terminal
> path that `engine/theta_connector.py` currently uses. It is documented
> here as an option, not as the current integration. Switching the
> engine to the Python library would remove the Java JAR dependency
> but is an architectural change — not part of this doc pass.

#### Install + auth

```
pip install thetadata          # or: uv add thetadata
```

Requires Python 3.12+. Authenticate one of three ways:

1. `creds.txt` in the cwd (or `THETADATA_CREDENTIALS_FILE` env var) —
   email on line 1, password on line 2.
2. Constructor args:
   ```python
   client = ThetaClient(email="...", password="...")
   ```
3. Custom file path:
   ```python
   client = ThetaClient(creds_file="/path/to/creds.txt")
   ```

#### Quick start

```python
from datetime import date
from thetadata import ThetaClient

client = ThetaClient()                                  # polars by default
print(client.stock_list_symbols())
print(client.stock_history_eod(symbol="AAPL",
                                start_date=date(2024, 1, 1),
                                end_date=date(2024, 1, 31)))
print(client.stock_snapshot_quote(symbol=["AAPL"]))
```

#### Polars vs Pandas

Default returns `polars.DataFrame`. Switch with:

```python
pandas_client = ThetaClient(dataframe_type="pandas")
```

Or share a session between two clients:

```python
polars_client = ThetaClient()
pandas_client = ThetaClient(
    existing_authorized_client=polars_client,
    dataframe_type="pandas",
)
```

Theta recommends Polars for analytical workloads (faster, more
memory-efficient, multithreaded).

#### Method ↔ REST mapping

The Python methods are 1:1 with REST endpoints. Naming convention:
`{stock|option|index}_{list|snapshot|history|at_time}_{verb}` plus
calendar.

**Stocks** — `stock_list_symbols`, `stock_list_dates`,
`stock_snapshot_{ohlc,trade,quote,market_value}`,
`stock_history_{eod,ohlc,trade,quote,trade_quote}`,
`stock_at_time_{trade,quote}`.

**Options** — `option_list_{symbols,dates,expirations,strikes,contracts}`,
`option_snapshot_{ohlc,trade,quote,open_interest,market_value}`,
`option_snapshot_greeks_{implied_volatility,all,first_order,second_order,third_order}`,
`option_history_{eod,ohlc,trade,quote,trade_quote,open_interest}`,
`option_history_greeks_{eod,all,first_order,second_order,third_order,implied_volatility}`,
`option_history_trade_greeks_{all,first_order,second_order,third_order,implied_volatility}`,
`option_at_time_{trade,quote}`.

**Indices** — `index_list_{symbols,dates}`,
`index_snapshot_{ohlc,price,market_value}`,
`index_history_{eod,ohlc,price}`, `index_at_time_price`.

**Calendar** — `calendar_open_today`, `calendar_on_date`,
`calendar_year`.

Parameters mirror the REST query strings, but use Python types
(`datetime.date` for dates, `datetime.time` for times, `List[str]` for
multi-symbol lists, `int` / `float` for numbers, etc.). The library
docs render the row schema for each method.

#### Logging

```python
import logging
from thetadata import ThetaClient

logging.basicConfig(level=logging.INFO)
client = ThetaClient()                # logs auth + request info
```

---

### 17.16 WebSocket streaming

Theta's streaming surface lives on the **Terminal's FPSS** (not the
MDDS / REST endpoint). You connect to a websocket on the Terminal,
subscribe to streams, and the Terminal multiplexes upstream events
back to you over the same connection.

**Endpoint:** `ws://127.0.0.1:25520/v1/events`
**Connections:** **only one allowed** to this endpoint. Multiplex on
your client side.
**Tier:** see §17.16.4 — varies by stream type.

#### 17.16.1 Mechanics + config

The Terminal's FPSS is enabled by default. To disable or change the
upstream region, edit the Terminal config:

```
[fpss]
enable=true                  # default; set false to disable
fpss_region="fpss_nj_hosts"  # production (default)
# fpss_region="fpss_dev_hosts"   # dev — replays a random past day in a loop
# fpss_region="fpss_stage_hosts" # staging
```

The **dev** region is useful when the live market is closed: it
infinite-loops a single past day's data, sending events as fast as
possible. Some replayed contracts may be expired or not yet listed.

#### 17.16.2 Message shape

Every received message has a `header`:

```json
{
  "header": {
    "status": "CONNECTED",
    "type": "TRADE"            // QUOTE, TRADE, OHLC, REQ_RESPONSE, STATUS
  },
  "contract": {                 // present on TRADE, QUOTE, OHLC
    "security_type": "OPTION",  // or STOCK or INDEX
    "root": "QQQ",
    "expiration": 20231110,
    "strike": 360000,           // 1/10 of a cent — see §17.1
    "right": "P"                // C or P
  },
  "trade": {                    // or "quote" or "ohlc"
    "ms_of_day": 49531278,
    "sequence": -563040482,
    "size": 5,
    "condition": 18,
    "price": 1.06,
    "exchange": 65,
    "date": 20231103
  }
}
```

**Status messages** are sent every second to keep the connection alive
(`type=STATUS`, no body). Use them as a heartbeat.

#### 17.16.3 Subscribe / unsubscribe

Each subscribe request must include an `id` field that increments per
request — used by the Terminal to send back a `REQ_RESPONSE` confirming
success and to **automatically resubscribe** the stream on reconnect.
**Failure to increment the id breaks auto-resubscribe.**

**Per-contract trade stream (option):**

```json
{
  "msg_type": "STREAM",
  "sec_type": "OPTION",
  "req_type": "TRADE",
  "add": true,
  "id": 0,
  "contract": {
    "root": "SPXW",
    "expiration": 20240315,
    "strike": 4800000,
    "right": "C"
  }
}
```

**Per-contract quote stream:** identical, with `req_type: "QUOTE"`.

**Full-trade-stream (every option trade across OPRA):**

```json
{
  "msg_type": "STREAM_BULK",
  "sec_type": "OPTION",
  "req_type": "TRADE",
  "add": true,
  "id": 0
}
```

**Stock streams** — same shape with `sec_type: "STOCK"`. For
per-symbol stock streams, the contract object is just
`{ "root": "AAPL" }`.

**Index price stream** — `sec_type: "INDEX"`, `req_type: "TRADE"`,
contract is `{ "root": "SPX" }`. The trade message represents a price
report; only `price` is meaningful (other fields can be ignored).

**Unsubscribe:** flip `add` to `false` (and increment `id`):

```json
{ "msg_type": "STREAM", "sec_type": "OPTION", "req_type": "TRADE",
  "add": false, "id": 1, "contract": { ... } }
```

**Stop all subscriptions:**

```json
{ "msg_type": "STOP" }
```

#### 17.16.4 Tier matrix

| Stream | Min tier |
|---|---|
| Option full-trade stream (`STREAM_BULK` / OPTION / TRADE) | Pro |
| Option per-contract trade stream | Standard |
| Option per-contract quote stream | Standard |
| Stock full-trade stream | Pro |
| Stock per-symbol trade stream | Standard |
| Stock per-symbol quote stream | Standard |
| Index price stream | Standard |

Streamable-contract counts (see §2 streaming tables) cap how many
per-contract subscriptions you can hold open at once.

#### 17.16.5 REQ_RESPONSE codes

After subscribing, you receive:

```json
{ "header": { "type": "REQ_RESPONSE", "status": "CONNECTED",
              "response": "SUBSCRIBED", "req_id": 0 } }
```

Possible `response` values:

| Code | Meaning |
|---|---|
| `SUBSCRIBED` | Request accepted. Doesn't guarantee the contract exists — only that you'll receive data if/when it does. |
| `ERROR` | Generic subscription error. |
| `MAX_STREAMS_REACHED` | Too many open contracts — unsubscribe some, upgrade tier, or send `STOP`. |
| `INVALID_PERMS` | Tier doesn't permit this stream type. |

#### 17.16.6 Latency benchmark

Theta publishes a Python latency-test snippet for the option full-trade
stream. Typical latencies (ms-of-day delta vs system clock) on a
local machine in EST are **single-digit-to-50 ms**. Sample
Python (note: `websockets` 11.x, **Python 3.11 only — not 3.12**):

```python
import asyncio, json, time
import websockets

async def stream_trades():
    async with websockets.connect("ws://127.0.0.1:25520/v1/events") as ws:
        await ws.send(str({
            "msg_type": "STREAM_BULK", "sec_type": "OPTION",
            "req_type": "TRADE", "add": True, "id": 0,
        }))
        count = 0
        while True:
            response = await ws.recv()
            count += 1
            obj = json.loads(response)
            if obj["header"]["type"] == "TRADE" and count % 100 == 0:
                ms_now = (int(time.time() * 1000) % 86400000) - 14400000
                print("latency:", ms_now - int(obj["trade"]["ms_of_day"]))

asyncio.get_event_loop().run_until_complete(stream_trades())
```

Sync your system clock with an NJ/NY time server for accurate latency
measurement. If you see 900-ms negative latencies, add `+ 1000` to the
`ms_now` calculation (DST edge case).

---

### 17.17 Client code patterns

Recommended ways to consume the REST API from Python and JavaScript.

#### 17.17.1 Python — `httpx` synchronous

For single-shot small payloads, the whole response in memory:

```python
import httpx, csv

BASE_URL = "http://127.0.0.1:25503/v3"
params = {"symbol": "MSFT", "expiration": "*"}
r = httpx.get(BASE_URL + "/option/snapshot/ohlc",
              params=params, timeout=60)
r.raise_for_status()
for row in csv.reader(r.text.split("\n")):
    print(row)
```

#### 17.17.2 Python — `httpx` streaming (recommended for large pulls)

For multi-day, full-chain, or trade-tape pulls, stream line-by-line so
you do not hold the entire response in memory:

```python
import httpx, csv, io

with httpx.stream("GET", BASE_URL + "/option/snapshot/ohlc",
                  params=params, timeout=60) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        for row in csv.reader(io.StringIO(line)):
            print(row)
```

#### 17.17.3 Python — Polars + NDJSON

`format=ndjson` is the cleanest server-side encoding to feed into
Polars (no escape ambiguity vs CSV):

```python
import httpx, io, polars as pl

params = {
    "date": "2024-11-07", "symbol": "AAPL",
    "expiration": "2025-01-17", "interval": "5m",
    "format": "ndjson",
}
r = httpx.get(BASE_URL + "/option/history/ohlc", params=params, timeout=60)
r.raise_for_status()
df = pl.read_ndjson(io.StringIO(r.text))
```

To switch to CSV, set `format=csv` and use `pl.read_csv(io.StringIO(r.text))`.

#### 17.17.4 Python — concurrency with `httpx.AsyncClient` + semaphore

Stay under the per-tier outstanding-request cap (§3) using a semaphore:

```python
import asyncio, httpx

CONCURRENCY_LIMIT = 4   # set per your tier — Standard=4, Pro=8
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

async def fetch(client, symbol):
    async with sem:
        params = {"symbol": symbol, "expiration": "*"}
        r = await client.get(BASE_URL + "/option/snapshot/ohlc", params=params)
        r.raise_for_status()
        return {"symbol": symbol, "data": r.text}

async def run(symbols):
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await asyncio.gather(*[fetch(client, s) for s in symbols])

# results = asyncio.run(run(["AAPL", "TSLA", "META", "SPY"]))
```

#### 17.17.5 Python — basic pipeline (Polars + asyncio)

Combine date listing → concurrent fetch → concat. Useful for pulling
N trading days of seconds-level data efficiently:

```python
import asyncio, httpx, io
import polars as pl

BASE_URL = "http://127.0.0.1:25503/v3"
sem = asyncio.Semaphore(4)

# Step 1: get available trading days (NDJSON for clean Polars ingest)
r = httpx.get(BASE_URL + "/stock/list/dates/quote",
              params={"symbol": "AAPL", "format": "ndjson"}, timeout=60)
r.raise_for_status()
dates = pl.read_ndjson(io.StringIO(r.text))["date"][-30:]  # last 30

# Step 2: concurrent per-day fetch
async def fetch(client, day):
    async with sem:
        p = {"symbol": "AAPL", "date": day, "interval": "1s",
             "format": "ndjson"}
        r = await client.get(BASE_URL + "/stock/history/quote", params=p)
        r.raise_for_status()
        return pl.read_ndjson(io.StringIO(r.text))

async def run():
    async with httpx.AsyncClient(timeout=60.0) as c:
        frames = await asyncio.gather(*[fetch(c, d) for d in dates])
        return pl.concat(frames)

# results = asyncio.run(run())
```

#### 17.17.6 JavaScript — Fetch API

Works in browser and Node.js. The Terminal must be running and reachable
at `127.0.0.1:25503`.

```javascript
const BASE_URL = "http://127.0.0.1:25503/v3";
const params = new URLSearchParams({ symbol: "MSFT", expiration: "*" });
const url = `${BASE_URL}/option/snapshot/ohlc?${params}`;

try {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`status: ${r.status}`);
  console.log(await r.text());
} catch (e) {
  console.error("Error fetching data:", e);
}
```

> Cross-origin: in a browser, the Terminal's HTTP server may need
> `host` config tweaks or a local proxy if the page is served from
> anything other than `127.0.0.1`.

---

## 18. Operational pipeline — what to pull and how

The operational counterpart to the upstream reference above: how to
run the actual pulls against the Terminal, the orchestrator,
verification, and the daily production routine. **This section
absorbed and replaces the prior `scripts/theta_data_guide.md`** so
that there is a single Theta documentation file in the repo.

### 18.1 Probe what your subscription unlocks

**Run this first.** The probe hits every endpoint the engine could
consume and classifies what is and isn't on your current tier.

```
python scripts/probe_theta_capabilities.py
```

Output:
- **Console:** grouped by category (stock / option / index / future),
  one line per endpoint.
- **File:** `data_processed/theta_capabilities.json` — a
  machine-readable audit trail consumed by the connector.

Classifications:

| Symbol | HTTP | Meaning |
|---|---|---|
| `✓ OK` | 200 + non-empty body | endpoint works, you have access |
| `- EMPTY` | 200 + empty body | endpoint works, no test data returned |
| `× BLOCKED` | 403 | not on your subscription tier |
| `? MISSING` | 404 | endpoint doesn't exist or wrong path |
| `! ERROR` | other | worth investigating |

Re-run any time you upgrade your Theta plan to see what opened up.

### 18.2 Core pulls (run in this order)

#### 18.2a yfinance building blocks (no Theta needed)

These don't touch Theta — they use yfinance (free, no API key).

```
python scripts/pull_vol_indices.py --years 5                # 10 vol indices
python scripts/pull_treasury_yields_yf.py --incremental     # risk-free rate curve
python scripts/pull_fundamentals_yf.py --workers 4          # P/E, beta, sector per ticker
python scripts/pull_earnings_yf.py --workers 4              # earnings calendar
```

#### 18.2b Theta indices (supersedes yfinance with authoritative CBOE data)

If your tier includes `/v3/index/history/*` (check
`probe_theta_capabilities.py`):

```
python scripts/pull_theta_indices_history.py --years 5 --incremental
```

This **overwrites** the yfinance rows in `vol_indices.parquet` with
Theta rows — Theta wins on duplicate dates. Merge is automatic.

#### 18.2c VIX futures curve (UX1–UX8 equivalent)

If futures tier is available:

```
python scripts/pull_theta_vix_futures.py --years 5 --months 8 --incremental
```

Outputs:
- `data_processed/vix_futures.parquet` — long format, one row per
  `(date, expiry)`.
- `data_processed/vix_futures_wide.parquet` — columns `ux1, ux2, …,
  ux8` for direct joins.

#### 18.2d IV surface history per ticker

Highest-value pull for strategy logic. Per-day snapshots of the full
chain across strikes and expiries.

```
# All 500 names, last 7 days (incremental). Use daily.
python scripts/pull_theta_iv_surface_history.py --universe sp500 \
       --days 7 --workers 4

# First-time bulk load: last 2 years.
python scripts/pull_theta_iv_surface_history.py --universe sp500 \
       --start 2024-04-01 --workers 4
```

Output: `data_processed/theta/iv_surface_history/ticker=<X>/year=<Y>/date=<YYYY-MM-DD>.parquet`.

> **Coverage caveat.** `iv_surface/` (snapshot) now covers ~502/503
> symbols across 3 dates (2026-04-23/05-24/06-01); `iv_surface_history/`
> is a stalled 4-name back-solve pilot (A, AAPL, ABBV, ABNB). See
> `docs/DATA_INVENTORY.md` for current counts and `PROJECT_STATE.md` §3
> for the missing-data contract decision.

#### 18.2e Options flow

Per-ticker daily aggregates: put/call volume, OI change,
unusual-volume flags.

```
python scripts/pull_theta_options_flow.py --universe sp500 --days 30 --workers 4
```

Output: `data_processed/theta/options_flow/<TICKER>.parquet`.

#### 18.2f Corporate actions

Splits + dividends per ticker (fills the otherwise-empty
`data/bloomberg/sp500_corporate_actions.csv`).

```
python scripts/pull_theta_corp_actions.py --universe sp500 --years 10 --workers 4
```

Outputs:
- `data_processed/corporate_actions/splits.parquet`
- `data_processed/corporate_actions/dividends.parquet`
- `data/bloomberg/sp500_dividends_theta.csv` (loader-compatible
  view).

#### 18.2g Feature-store backfill (after everything else)

Uses every dataset above. Takes ~15 min for all 500 tickers.

```
python scripts/backfill_features.py --workers 6 --force
```

### 18.3 One command does it all

The orchestrator runs every step in the right order, skipping
Theta-dependent ones automatically when the Terminal is down, and
skipping the news step when no API key is in env.

```
python scripts/pull_all.py                               # live refresh
python scripts/pull_all.py --dry-run                     # print plan only
python scripts/pull_all.py --skip theta_corp_actions     # skip specific
python scripts/pull_all.py --only vol treasury           # only these
```

Safe to schedule daily via cron / Task Scheduler.

### 18.4 Heavy / optional pulls

Not part of the default refresh. Run manually only when needed.

#### 18.4a Intraday option tape

Massive data — one ticker × one expiry × 1 day can be 100k+ rows.
Use targeted queries only.

```
# 5 days of AAPL tape, ATM strike only, 35-DTE expiry
python scripts/pull_theta_option_tape.py --tickers AAPL --days 5 --atm-only
```

Output: `data_processed/theta/option_tape/ticker=<X>/date=<Y>/{trades,quotes}.parquet`.

Use case: dealer-positioning refinement (classify prints as
buy-initiated vs sell-initiated from NBBO mid). Worth doing on the
top ~20 wheel candidates.

### 18.5 Verify everything

Single command, comprehensive across all data sources:

```
python scripts/feature_smoke_test.py
```

Sections 15 (`data_connectors`), 22 (`theta_history_pulls`), and 26
(`theta_outputs`) flip from SKIP → PASS as Theta data lands on disk.

For just the Theta-related checks:

```
python scripts/feature_smoke_test.py --section theta --verbose
```

### 18.6 Coverage matrix — what's wired and what isn't

| Feature | Source | Status |
|---|---|---|
| Stock OHLCV history | Theta (`stock/history/eod`) | ✓ script ready |
| IV surface (strike × expiry × date) | Theta (`option/history/greeks/iv`) | ✓ `pull_theta_iv_surface_history.py` |
| Options daily volume + OI | Theta (`option/history/volume,open_interest`) | ✓ `pull_theta_options_flow.py` |
| Full chain snapshot | Theta (`option/snapshot/greeks`) | ✓ existing connector |
| VIX family (index) history | Theta / Yahoo fallback | ✓ `pull_theta_indices_history.py` + `pull_vol_indices.py` |
| VIX futures UX1–UX8 | Theta (`future/history/eod`) | ✓ `pull_theta_vix_futures.py` |
| Stock splits / dividends | Theta (`stock/history/{split,dividend}`) | ✓ `pull_theta_corp_actions.py` |
| Intraday option tape (trades + quotes) | Theta (`option/history/{trade,quote}`) | ✓ `pull_theta_option_tape.py` |
| Treasury yields | yfinance | ✓ `pull_treasury_yields_yf.py` |
| Fundamentals snapshot (P/E, beta, sector) | yfinance | ✓ `pull_fundamentals_yf.py` |
| Earnings calendar | yfinance | ✓ `pull_earnings_yf.py` |
| News sentiment | Polygon / Finnhub / Benzinga | ✓ `pull_news_sentiment.py` (needs API key) |
| Short interest / borrow fee | Bloomberg only | ○ no free alternative |
| Analyst revisions stream | Bloomberg only (yfinance has current snapshot) | ○ partial via yfinance |
| Macro calendar (FOMC, CPI) | Bloomberg only (Finnhub has a free limited calendar) | ○ could add Finnhub adapter |
| Point-in-time index membership | Bloomberg (`sp500_index_membership.csv` already on disk) | ✓ wired into loader |

### 18.7 Daily production routine

```
# Every business day morning, in order:
python scripts/probe_theta_capabilities.py    # optional — only when Theta has updates
python scripts/pull_all.py                    # runs everything available
python scripts/feature_smoke_test.py --fast   # verify
```

Expect the orchestrator to take ~5–10 min end-to-end (network-bound).
The feature-backfill step inside it adds ~15 min when Theta rows
changed.

---

## 19. Smart Wheel Engine integration notes

A small set of repo-specific facts that complement the upstream
documentation:

- **Provider selection.** `WheelRunner.connector` (in
  `engine/wheel_runner.py`) reads `SWE_DATA_PROVIDER`. Default is
  `bloomberg`; `theta` swaps in `engine/theta_connector.py`.
- **Tier behaviour.** Per `LAPTOP_SETUP.md` §3, our recorded tier
  is: stock EOD ✓, option chains + snapshots ✓, option history
  (EOD/quote/trade/OI/greeks/IV) ✓, index EOD (VIX + SKEW only) ✓.
  **Blocked:** futures (no UX1–UX8), index snapshots/OHLC, stock
  realtime. **Missing on v3:** corporate actions.
- **Capability cache.** `data_processed/theta_capabilities.json` is
  produced by `scripts/probe_theta_capabilities.py` and consumed by
  the connector to short-circuit calls to known-blocked endpoints.
- **`iv_surface/` coverage.** ~502/503 symbols covered (Universe A +
  8 ETFs) across 3 snapshot dates; `iv_surface_history/` is a stalled
  4-name back-solve pilot (see `docs/DATA_INVENTORY.md`).
  `engine/volatility_surface.py` SVI tools have
  zero non-test callers as of 2026-04-25 — see `PROJECT_STATE.md`
  §3 for the open contract decision.
- **Sandbox without Terminal.** Cowork / CI runs default to
  `SWE_DATA_PROVIDER=bloomberg`. Anything that needs a live chain
  snapshot is unavailable; backfilled `data_processed/theta/**` can
  be read directly when present (rehydrated via
  `scripts/pull_all.py` on a machine with Terminal, then copied in).
- **Rate handling.** Theta's default risk-free rate is SOFR; the
  engine's own rate handling lives in
  `engine/data_integration.get_current_risk_free_rate` and
  `engine/data_connector.MarketDataConnector.get_risk_free_rate`.
  Both return a **decimal** (audit-VIII fixed a percent/decimal
  unit bug at `WheelRunner.rank_candidates_by_ev`). Don't pass
  Theta a percent value.
- **Greeks unit caveat.** Theta's Vega and Rho need to be divided
  by 100 before they meet our internal Greek unit contract
  (`docs/GREEKS_UNIT_CONTRACT.md`). The connector handles this
  conversion.
- **Python version split — footgun.** Two Python tools in the Theta
  ecosystem have **incompatible** version requirements:
  - The `thetadata` Python library (§17.15) requires **Python 3.12+**.
  - Theta's own latency-test snippet for the WebSocket stream
    (§17.16.6) uses an older `websockets` API that runs only on
    **Python 3.11**. It will not work on 3.12.
  
  If we adopt the Python library or use the WebSocket latency probe,
  pin the Python version per use. The current engine doesn't depend
  on either path — it consumes the REST API via `httpx` from
  `engine/theta_connector.py`, which works on any modern 3.x.
