# Smart Wheel Engine — Dashboard

Next.js 16 dashboard for the Smart Wheel Engine. It consumes the engine HTTP
API at `:8787` (served by [`engine_api.py`](../engine_api.py)) and exposes
three pages, all in the `(terminal)` route group:

| Page | What it shows |
|---|---|
| `/cockpit` | Decision Cockpit — EV-ranked candidates, dossier and reviewer verdicts (R1–R11), regime banner, concentration meters, drop funnel. Read-only over `/api/engine`. |
| `/portfolio` | Live IBKR book viewer (design D26) — KPIs, holdings, allocation, equity curve, income, margin, risk, trade history. Read-only over `/api/portfolio/*`; each slice is labelled with its source. |
| `/terminal` | Options terminal — market/vol panel, options engine, live book, ticker watchlist, events calendar, Ollama research chat, command line, and a symbol workbench (engine read, dealer positioning, TradingView handoff). |

`/` redirects to `/cockpit`. The three pages share the Wheelhouse header and
its cross-page tabs.

The dashboard renders what the engine returns and reimplements none of its
quantitative logic. The hard EV invariant (`OPERATING_MODEL.md` §7 — no
tradeable candidate bypasses `EVEngine.evaluate`) holds because the engine
API enforces it; nothing here ranks, upgrades or executes.

The repo root has an alternate legacy Python CLI dashboard at
[`quant_dashboard.py`](quant_dashboard.py); it is **not** the primary UI
and is retained as a research-tier surface only (see
[`MODULE_INDEX.md`](../MODULE_INDEX.md)).

---

## Tech stack

- **Framework**: Next.js 16 (App Router, TypeScript)
- **UI**: Tailwind CSS v4 + shadcn/ui primitives
- **Local store**: SQLite via better-sqlite3 + Drizzle ORM, holding only the
  terminal's operator state — ticker watchlist, manually curated events,
  cached quote snapshots, research-chat sessions. Lives at
  `dashboard/data/finance-news.db` (gitignored; the legacy file name is kept
  so an existing local watchlist is still found).
- **Charts**: Recharts
- **AI**: local Ollama via the Vercel AI SDK (terminal research chat)
- **Data**: the engine API (`:8787`) for ranking, analysis, regime, calendar
  and the live book; Finnhub (optional key) for realtime quotes, falling back
  to the engine's EOD close

---

## Getting started

### Prerequisites

- Node.js 20+ (required by Next.js 16)
- The engine API up at `:8787` (`python engine_api.py` from the repo root —
  see the [root README](../README.md)); without it the pages render their
  explicit engine-offline states
- Optional: [Ollama](https://ollama.ai) for the terminal research chat
- Optional: Finnhub free-tier API key for realtime quotes

### Setup

```bash
# From the repo root
cd dashboard

# Install dependencies
npm install

# Environment template (engine URL, Finnhub, Ollama)
cp .env.example .env.local

# Dev server at :3000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000); it lands on `/cockpit`.

### Optional: Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh   # macOS / Linux
ollama pull qwen2.5:7b
# Ollama serves on http://localhost:11434 by default
```

The engine's own memo path (`engine/trade_memo.py`) uses Ollama 72B / 32B
locally; the dashboard's research chat uses the same Ollama instance via
the Vercel AI SDK (`OLLAMA_URL` / `OLLAMA_MODEL`).

---

## App layout

```
dashboard/src/
├── app/
│   ├── (terminal)/          # /cockpit, /portfolio, /terminal (+ layout, error, loading)
│   ├── api/
│   │   ├── engine/          # GET ?action=… proxy to engine_api.py :8787 (never cached)
│   │   ├── portfolio/[sub]/ # GET proxy to the engine's read-only /api/portfolio/<sub>
│   │   ├── chat/            # POST — streaming Ollama research chat
│   │   ├── market/          # GET ?ticker= — quote: Finnhub → <24h snapshot → engine EOD
│   │   ├── watchlist/       # GET / POST / DELETE — SQLite ticker watchlist with quotes
│   │   └── events/          # GET / POST — SQLite events merged with the engine calendar
│   ├── layout.tsx           # Root layout + metadata
│   ├── page.tsx             # Redirects to /cockpit
│   ├── not-found.tsx, global-error.tsx
│   └── globals.css
├── components/
│   ├── cockpit/             # Decision Cockpit panels
│   ├── portfolio/           # Live-book viewer panels + data hooks
│   ├── terminal/            # Terminal panels, command line, panel error boundary
│   ├── shell/               # Wheelhouse header + cross-page tabs
│   └── ui/                  # shadcn/ui primitives
├── db/                      # Drizzle schema + SQLite connection (local store)
├── hooks/                   # useEngineData / useLiveBook / useTickerAnalysis
├── services/market-data.ts  # Quote resolution (Finnhub, snapshot cache, engine EOD)
├── types/                   # Wire shapes (engine, cockpit, live book)
└── lib/                     # cockpit-trust (null-honest formatting), cn()
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
npm run db:generate  # Generate Drizzle migrations for the local store
npm run db:migrate   # Run Drizzle migrations
```

The runtime creates the local store's tables itself (`src/db/index.ts`); the
Drizzle Kit scripts are an optional migration workflow, not a startup step.

---

## Data sources

| Source | Data | Cost |
|---|---|---|
| Engine API (`:8787`) | EV ranking, dossier, regime / VIX, calendar, ticker analysis, dealer positioning, live IBKR book | Local |
| Finnhub | Realtime stock quotes (optional key; engine EOD close otherwise) | Free tier |
| Ollama | Local AI inference for the research chat | Free (local) |

---

## How the dashboard talks to the engine

`src/app/api/engine/route.ts` and `src/app/api/portfolio/[sub]/route.ts` are
thin server-side proxies that forward to the Python HTTP API at
`ENGINE_API_URL` (default `http://localhost:8787`) with `cache: "no-store"`.
The engine endpoint catalog is documented in the
[`engine_api.py`](../engine_api.py) header. When the engine is unreachable the
proxies answer 503 with a hint and the panels render an explicit
offline / unavailable state.
