# Smart Wheel Engine — Dashboard

Next.js 16 dashboard for the Smart Wheel Engine. Consumes the engine HTTP
API at `:8787` (served by [`engine_api.py`](../engine_api.py)) and exposes
two surfaces:

1. **Engine cockpit** — EV-ranked candidates, dossier (engine + chart),
   strangle timing, dealer-positioning overlay, payoff diagrams, AI memos.
2. **Financial-news component** — feed, ticker pages, watchlist, macro
   calendar, research chat against local Ollama. This is the news-side UI
   that piggybacks on the same Next.js app; it pre-dates the engine
   integration and is kept alongside it.

The repo root has an alternate legacy Python CLI dashboard at
[`quant_dashboard.py`](quant_dashboard.py); it is **not** the primary UI
and is retained as a research-tier surface only (see
[`MODULE_INDEX.md`](../MODULE_INDEX.md)).

---

## Tech stack

- **Framework**: Next.js 16 (App Router, TypeScript)
- **UI**: Tailwind CSS v4 + shadcn/ui components
- **Database**: SQLite via better-sqlite3 + Drizzle ORM
- **Charts**: Recharts
- **AI**: Local Ollama via Vercel AI SDK (research chat, memo summarisation)
- **Data**: the engine API (`:8787`) for ranking + analysis; RSS feeds,
  Finnhub free tier, SEC EDGAR, FRED for the news component

---

## Getting started

### Prerequisites

- Node.js 20+ (required by Next.js 16)
- The engine API up at `:8787` for the engine cockpit
  (`python engine_api.py` from the repo root — see the
  [root README](../README.md))
- Optional: [Ollama](https://ollama.ai) for memo / research-chat AI features
- Optional: Finnhub free-tier API key for live quotes in the news component
- Optional: FRED API key for macro indicators

### Setup

```bash
# From the repo root
cd dashboard

# Install dependencies
npm install

# Environment template (Finnhub, FRED, Ollama, etc.)
cp .env.example .env.local

# Dev server at :3000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The engine cockpit
pages assume `python engine_api.py` is running in another terminal.

### Optional: Ollama

For memo summarisation and the research chat:

```bash
curl -fsSL https://ollama.ai/install.sh | sh   # macOS / Linux
ollama pull qwen2.5:7b
# Ollama serves on http://localhost:11434 by default
```

The engine's own memo path (`engine/trade_memo.py`) uses Ollama 72B / 32B
locally; the dashboard's research chat uses the same Ollama instance via
the Vercel AI SDK.

---

## App layout

```
dashboard/src/
├── app/
│   ├── (main)/              # Financial-news app (feed, ticker, watchlist, calendar, research)
│   ├── (terminal)/          # Engine surfaces — Decision Cockpit (/cockpit) + Terminal (/terminal)
│   ├── api/                 # Next.js API routes
│   │   ├── engine/         # Proxies / wrappers around engine_api.py :8787
│   │   ├── execute/        # Trade-execution surfaces (committee verdicts)
│   │   ├── exposure/       # Portfolio Greeks / exposure
│   │   ├── briefings/      # AI briefings (Ollama-backed)
│   │   ├── stories/        # News stories (financial-news component)
│   │   ├── ingest/         # RSS ingestion trigger
│   │   ├── market/         # Live quotes (Finnhub)
│   │   ├── events/         # Macro-event calendar
│   │   ├── watchlist/      # Ticker watchlist
│   │   ├── alerts/         # Watchlist alerts
│   │   ├── chat/           # Ollama research chat (streaming)
│   │   ├── schedule/       # Scheduled tasks
│   │   ├── stream/         # Server-sent events
│   │   └── categories/     # News classification
│   ├── layout.tsx
│   └── page.tsx
├── components/              # React components (incl. shadcn/ui)
├── db/                      # Drizzle schema + connection
├── services/                # Business logic (RSS, entity extraction, market data, EDGAR, FRED)
├── types/                   # TypeScript types
└── lib/                     # Utilities
```

`node_modules/` and the built `.next/` directory are gitignored; everything
in `src/` is tracked.

---

## Scripts

```bash
npm run dev          # Development server at :3000
npm run build        # Production build
npm run start        # Production server
npm run lint         # ESLint
npm run db:generate  # Generate Drizzle migrations
npm run db:migrate   # Run Drizzle migrations
```

---

## Data sources (zero cost)

| Source | Data | Cost |
|---|---|---|
| Engine API (`:8787`) | EV ranking, analysis, dossier, payoff, dealer positioning | Local |
| RSS feeds (8 sources) | Financial headlines | Free |
| Finnhub | Stock quotes | Free tier (60 calls / min) |
| SEC EDGAR | Company filings | Free (public API) |
| FRED | Macro indicators | Free (API key required) |
| Ollama | Local AI inference | Free (local) |

---

## How the engine cockpit talks to the engine

The Next.js API routes under `src/app/api/engine/` are thin proxies that
forward to the Python HTTP API at `http://localhost:8787`. The engine
endpoint catalog (34 endpoints — `/api/candidates`, `/api/dossier`,
`/api/strangle`, `/api/dealer_positioning`, `/api/memo`, `/api/tv/*`,
etc.) is documented in the [`engine_api.py`](../engine_api.py) header.

The dashboard does **not** reimplement any of the engine's quantitative
logic; it renders what the API returns. The hard EV invariant
(`CLAUDE.md`'s "no tradeable candidate bypasses `EVEngine.evaluate`")
holds because the API itself enforces it.
