# Smart Wheel Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Probabilistic expected-value (EV) decision engine for wheel strategies on
S&P 500 names** — short cash-secured puts → covered calls, with timing-gated
strangles. Every tradeable candidate is ranked through a single authoritative
ranker (`EVEngine.evaluate`); reviewers can downgrade a verdict but never
rescue a negative-EV trade.

> **AI agent / fresh contributor — start here:**
>
> 1. [`AGENTS.md`](AGENTS.md) — read order for any agent (Claude, Codex, Cursor, Copilot, Aider).
> 2. [`CLAUDE.md`](CLAUDE.md) — structural contract; the four-layer mental model and the hard EV invariant.
> 3. [`PROJECT_STATE.md`](PROJECT_STATE.md) — what's authoritative right now, what's deprecated.
> 4. [`MODULE_INDEX.md`](MODULE_INDEX.md) — per-module map.
> 5. [`TESTING.md`](TESTING.md) — test taxonomy + launch-blocker subset.
>
> Other entry points: [`DECISIONS.md`](DECISIONS.md), [`ROADMAP.md`](ROADMAP.md), [`CHANGELOG.md`](CHANGELOG.md), [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md), [`docs/LAUNCH_READINESS.md`](docs/LAUNCH_READINESS.md), [`COMMIT_GUIDE.md`](COMMIT_GUIDE.md), [`FILE_MANIFEST.md`](FILE_MANIFEST.md), [`docs/TRADINGVIEW_INTEGRATION.md`](docs/TRADINGVIEW_INTEGRATION.md).

---

## The four layers

| Layer | Lives at | What it does |
|---|---|---|
| **Data** | `data/`, `data_processed/`, `scripts/pull_*.py` | OHLCV, IV history, option chains, fundamentals, macro, news. Two providers selected by `SWE_DATA_PROVIDER` (default `bloomberg`). |
| **Quant** | `engine/` | Black-Scholes-Merton + Greeks to 3rd order, empirical forward distributions, POT-GPD tails, 4-state Gaussian HMM regime, Nelson-Siegel skew, Student-t copula CVaR, dealer GEX / walls / gamma flip. |
| **Decision** | `engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py` | `EVEngine.evaluate` (the authoritative ranker), `WheelRunner.rank_candidates_by_ev` (the one ranker every tradeable path routes through), `EnginePhaseReviewer` (rules R1–R6, downgrade-only). |
| **Interface** | `engine_api.py`, `dashboard/`, `engine/tradingview_bridge.py`, `advisors/` | HTTP API on `:8787`, Next.js dashboard, TradingView chart bridge (sanity check, not a decider), Buffett/Munger/Simons/Taleb advisor committee (advisory only). |

See [`CLAUDE.md`](CLAUDE.md) for the full four-layer model and the hard EV
invariant. See [`MODULE_INDEX.md`](MODULE_INDEX.md) for the per-module map.

---

## What this is not

Out of scope by design (see `CLAUDE.md`'s NEVER list):

- **No auto-execution.** The engine produces ranked candidates and memos. No broker wiring, no OMS, no order routing.
- **No tick-level order flow.** Theta v3 doesn't expose realtime stock quotes at this tier.
- **No non-S&P-500 universe.** Constituent list at `data_raw/sp500_constituents_current.csv`.
- **No non-wheel strategies.** Short puts + covered calls + timing-gated strangles only.

---

## Quick start

### Install

```bash
git clone https://github.com/MertYakar66/smart-wheel-engine.git
cd smart-wheel-engine
pip install -r requirements.txt
```

Python 3.11+ is required. For Theta Terminal bring-up on a new machine, see
[`docs/LAPTOP_SETUP.md`](docs/LAPTOP_SETUP.md).

### Smoke test (5 tickers, ~2 s)

The fastest way to confirm the data + connector + EV-engine path is healthy:

```python
from engine.wheel_runner import WheelRunner

runner = WheelRunner()
print("connector:", type(runner.connector).__name__)

df = runner.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10,
    min_ev_dollars=-1e9,
    include_diagnostic_fields=True,
)
print(df[["ticker", "ev_dollars", "iv", "premium"]])
```

Five rows with non-null `ev_dollars`, `iv`, and `premium` mean the
Bloomberg-CSV path is healthy. Same snippet is wired as the `/ev-smoke`
slash command.

### Run the HTTP API + dashboard

```bash
# Terminal 1: engine API on :8787
python engine_api.py

# Terminal 2: Next.js dashboard at :3000 (proxies the API)
cd dashboard && npm install && npm run dev
```

`engine_api.py` serves 32 endpoints — see the file header for the catalog.

### Daily news pipeline (optional, no API cost)

```bash
python morning_run.py
```

Browser-driven multi-LLM (Claude / ChatGPT / Gemini paid sessions); the
output feeds the operator dashboard. As of D18 (2026-05-26), no news
subsystem feeds the EV authority — `engine/news_sentiment.py` is now
an operator-transparency layer with a constant-1.0 multiplier stub.
See `DECISIONS.md` D18 and `docs/NEWS_REDESIGN_CAMPAIGN.md` for the
in-flight quantitative replacements (EDGAR earnings, fundamentals
quality score, FRED macro).

---

## Provider selection

```bash
# Cowork sandbox / fresh laptop without Terminal (default)
export SWE_DATA_PROVIDER=bloomberg

# Laptop with Theta Terminal up
export SWE_DATA_PROVIDER=theta
```

`bloomberg` reads the committed CSVs under `data/bloomberg/`. `theta` reads
live from the Theta Terminal at `127.0.0.1:25503`. Full capability matrix
in [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) §2.

The `.claude/settings.json` SessionStart hook warns when the variable is
unset and defaults to `bloomberg`.

---

## Project layout

```
smart-wheel-engine/
├── engine/          # quant + decision layer (EVEngine, WheelRunner, dossier, dealer positioning, …)
├── advisors/        # Buffett/Munger/Simons/Taleb committee (advisory only)
├── scripts/         # data pullers (pull_*.py) + diagnostics + Bloomberg-export assets
├── tests/           # test suite (taxonomy in TESTING.md)
├── dashboard/       # Next.js dashboard consuming engine_api.py (+ legacy Python CLI)
├── data/            # Bloomberg CSVs + feature store (AAPL committed as sample)
├── data_raw/        # universe list + raw fixtures
├── data_processed/  # regenerable Theta/yfinance pulls (gitignored)
├── financial_news/  # standalone news platform (not on the EV path)
├── news_pipeline/   # browser-agent pipeline driving morning_run.py
├── local_agent/     # experimental local agent + UI
├── ml/              # research ML models
├── backtests/       # research backtesting
├── src/             # feature-engineering / schema modules (legacy scaffold — see DECISIONS.md D2)
├── config/          # configuration
├── utils/           # shared utilities
├── tradingview/     # Pine indicator + analyst-workspace assets
├── docs/            # documentation set (operational + reference)
├── archive/         # superseded / point-in-time artifacts
├── notebooks/       # exploratory notebooks
├── models/          # ML model output directory
├── engine_api.py    # HTTP API entry point (:8787)
├── morning_run.py   # news-pipeline entry point
└── *.md             # AGENTS / CLAUDE / README + the Tier-2 index docs
```

The exhaustive per-file index is [`FILE_MANIFEST.md`](FILE_MANIFEST.md) —
grep it; don't read it in full.

---

## Testing

```bash
# Full suite
pytest tests/ -v

# Launch-blocker subset (decision-layer gate)
/launch-blockers          # slash command, wraps the subset
# or directly:
pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py \
       tests/test_authority_hardening.py tests/test_audit_viii_*.py \
       tests/test_launch_blockers.py -v
```

Any change touching `engine/ev_engine.py`, `engine/wheel_runner.py`, or
`engine/candidate_dossier.py` must run the full suite — the invariants are
cross-cutting. See [`TESTING.md`](TESTING.md) for the live count, the
launch-blocker subset, and the "what to run when you touch X" map.

---

## Documentation

| Document | Description |
|---|---|
| [AGENTS.md](AGENTS.md) | AI-agent onboarding contract — the canonical read order |
| [CLAUDE.md](CLAUDE.md) | Structural contract — four-layer model + hard EV invariant + NEVER list |
| [PROJECT_STATE.md](PROJECT_STATE.md) | Temporal state — what's authoritative / in progress / deprecated |
| [MODULE_INDEX.md](MODULE_INDEX.md) | Per-module purpose + decision-layer role classification |
| [FILE_MANIFEST.md](FILE_MANIFEST.md) | Exhaustive per-file index (grep, don't read) |
| [TESTING.md](TESTING.md) | Test taxonomy + launch-blocker subset |
| [DECISIONS.md](DECISIONS.md) | Architectural decision log with rationale |
| [ROADMAP.md](ROADMAP.md) | Intentional next work by track |
| [CHANGELOG.md](CHANGELOG.md) | Recently-shipped, grouped by month |
| [COMMIT_GUIDE.md](COMMIT_GUIDE.md) | Commit-message and PR format |
| [docs/DATA_POLICY.md](docs/DATA_POLICY.md) | Data tiers, provider matrix, refresh procedures, sandbox caveats |
| [docs/LAUNCH_READINESS.md](docs/LAUNCH_READINESS.md) | Launch-blocker invariants before merging |
| [docs/LAPTOP_SETUP.md](docs/LAPTOP_SETUP.md) | Bring-up on a new machine (Theta Terminal, feature store) |
| [docs/TRADINGVIEW_INTEGRATION.md](docs/TRADINGVIEW_INTEGRATION.md) | Engine bridge + analyst workspace (MCP) |
| [docs/GREEKS_UNIT_CONTRACT.md](docs/GREEKS_UNIT_CONTRACT.md) | Canonical Greeks unit conventions |
| [docs/GOVERNANCE.md](docs/GOVERNANCE.md) | Model-governance framework |
| [docs/MODEL_CARDS.md](docs/MODEL_CARDS.md) | Model documentation |
| [docs/SECURITY.md](docs/SECURITY.md) | Security policy |

---

## Contributing

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the human-side
workflow and [`AGENTS.md`](AGENTS.md) + [`COMMIT_GUIDE.md`](COMMIT_GUIDE.md)
for the AI-agent handoff and commit-message standard.

Hard rules in brief:

- Branch + PR for everything; never commit to `main`.
- Run the full test suite for any decision-layer change.
- New inputs wire in as chained-provider participants or downgrade-only
  reviewers — never as a path that rescues a negative-EV trade.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Black-Scholes-Merton implementation follows Hull (11th edition).
- American option pricing uses Barone-Adesi & Whaley (1987).
- Tail-risk methodology references the POT-GPD literature (Embrechts et al.).
- Regime detection draws on the 4-state Gaussian HMM regime-switching literature.
