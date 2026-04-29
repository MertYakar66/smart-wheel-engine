# Laptop setup — rehydrate the engine from scratch

Use this when you move to a new machine. Everything you need is either in
this repo or regeneratable on the laptop with your Theta subscription.

---

## 1. Clone + Python env

```bash
git clone https://github.com/MertYakar66/smart-wheel-engine.git
cd smart-wheel-engine
git checkout claude/map-codebase-architecture-aBvbq   # or main, after merge

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Secrets — configure `.env`

```bash
cp .env.example .env
```

Fill in only the keys you actually use. For the current pipeline, the only
optional keys are news providers:

- `POLYGON_API_KEY` / `FINNHUB_API_KEY` / `BENZINGA_API_KEY` — any one
  activates `pull_news_sentiment.py`. Skip if you're not running that step.

No other key is required. Broker creds (IBKR / TDA / Alpaca) are only
needed for live execution, not research / backfill.

---

## 3. Theta Terminal

1. Install the Terminal tray app on the laptop (same Java `.jar` you used here).
2. Log in with your subscription.
3. Confirm it's reachable:
   ```bash
   python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 25503)); print('UP')"
   ```

Your current tier (recorded in the last probe):
- **Works:** stock EOD, option chain lists + snapshots, option history
  (EOD / quote / trade / OI / greeks/IV), index EOD (VIX + SKEW only)
- **Blocked:** futures (no UX1-UX8), index snapshots/OHLC, stock realtime
- **Missing on v3:** corporate actions

---

## 4. Regenerate local data (everything not in git)

```bash
# Verify your tier hasn't changed
python scripts/probe_theta_capabilities.py

# Pull everything your subscription + env keys allow
python scripts/pull_all.py --skip theta_vix_futures theta_corp_actions

# Rebuild the feature store for all 500 S&P names (~15 min)
python scripts/backfill_features.py --workers 6 --force
```

What this rehydrates:
- `data_processed/theta/**` — IV surface, options flow, indices history
- `data_processed/vol_indices{,_wide}.parquet` — VIX family
- `data/features/**/ticker=<X>/` — full feature store (1.2 GB)
- `data/bloomberg/sp500_{earnings,fundamentals}_yf.csv` — refreshed via
  yfinance if you want newer numbers

---

## 5. Verify

```bash
python scripts/feature_smoke_test.py        # 127 checks across 26 sections
pytest -q                                    # regression suite
```

Expect ~107 PASS / 0 FAIL / ~20 SKIP on the smoke test. SKIPs are mostly
for tiers you don't have (futures, realtime snapshots).

---

## What's already in git (no action needed)

- All engine code, scripts, tests
- All 25 Bloomberg CSVs (~260 MB committed — OHLCV, IV, liquidity, index
  membership, dividends, earnings, fundamentals, short interest, etc.)
- `data_processed/trade_universe/2025-11-22_trade_universe.csv`
- AAPL feature-store sample

## What is NOT in git (you regenerate on the laptop)

- `data/features/ticker=<other tickers>/` — 1.2 GB derived shards
- `data_processed/theta/**` — Theta pulls
- `data_processed/vol_indices*.parquet` — vol index history
- `data_processed/theta_capabilities.json` — probe output
- `.env` — secrets (gitignored; re-create from `.env.example`)

---

## Daily routine on the laptop

```bash
python scripts/pull_all.py --skip theta_vix_futures theta_corp_actions
python scripts/feature_smoke_test.py --fast
```

That's it — same workflow as on the desktop, just fewer tiers available.
