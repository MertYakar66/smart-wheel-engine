# FinanceNews - AI Financial News Platform

A local-first, zero-cost financial news aggregator with AI-powered analysis via Ollama. Built with Next.js 15, SQLite, and free data sources.

## Features

- **News Feed** - Aggregated financial news from 8+ RSS sources with automatic entity extraction, story clustering, and sector classification
- **Ticker Pages** - Per-stock view with price charts (Recharts), related news, and SEC EDGAR filing links
- **Watchlist** - Track tickers with price alerts and filtered news
- **Macro Calendar** - FOMC meetings, CPI releases, jobs reports, GDP estimates
- **Research Chat** - AI-powered research assistant via local Ollama models
- **Alert System** - Notifications when watched tickers appear in new stories

## Tech Stack

- **Framework**: Next.js 15 (App Router, TypeScript)
- **UI**: Tailwind CSS v4 + custom shadcn/ui components
- **Database**: SQLite via better-sqlite3 + Drizzle ORM
- **Charts**: Recharts
- **AI**: Ollama (local) via Vercel AI SDK
- **Data Sources**: RSS feeds, Finnhub (free tier), SEC EDGAR, FRED

## Getting Started

### Prerequisites

- Node.js 18+
- (Optional) [Ollama](https://ollama.ai) for AI features
- (Optional) Finnhub API key for live quotes
- (Optional) FRED API key for macro data

### Setup

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

### Optional: Ollama Setup

For AI-powered entity extraction and research chat:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b

# Ollama runs on http://localhost:11434 by default
```

### Optional: API Keys

Set in `.env.local`:
- `FINNHUB_API_KEY` - Free tier at [finnhub.io](https://finnhub.io/register) (60 calls/min)
- `FRED_API_KEY` - Free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

## Architecture

```
src/
├── app/                    # Next.js App Router pages
│   ├── feed/              # News feed with story cards
│   ├── ticker/[symbol]/   # Per-ticker view with chart
│   ├── watchlist/         # Ticker watchlist management
│   ├── calendar/          # Macro event calendar
│   ├── research/          # AI chat research surface
│   └── api/               # API routes
│       ├── stories/       # Story CRUD + filtering
│       ├── ingest/        # RSS ingestion trigger
│       ├── market/        # Market data quotes
│       ├── watchlist/     # Watchlist management
│       ├── alerts/        # Alert management
│       ├── events/        # Calendar events
│       └── chat/          # Ollama chat streaming
├── components/            # React components
│   ├── nav.tsx           # Navigation bar
│   └── ui/               # shadcn/ui base components
├── db/                    # Database layer
│   ├── schema.ts         # Drizzle ORM schema
│   └── index.ts          # Database connection + init
├── services/              # Business logic
│   ├── data-provider.ts  # DataProvider interface
│   ├── rss-ingestion.ts  # RSS feed parser
│   ├── rss-feeds.ts      # Feed source config
│   ├── entity-extraction.ts # NLP via Ollama/regex
│   ├── story-clustering.ts  # Jaccard similarity dedup
│   ├── market-data.ts    # Finnhub integration
│   ├── edgar.ts          # SEC EDGAR API
│   └── macro-data.ts     # FRED API
├── types/                 # TypeScript type definitions
└── lib/                   # Utilities (cn, etc.)
```

## Data Sources (Zero Cost)

| Source | Data | Cost |
|--------|------|------|
| RSS Feeds (8 sources) | Financial headlines | Free |
| Finnhub | Stock quotes | Free tier (60/min) |
| SEC EDGAR | Company filings | Free (public API) |
| FRED | Macro indicators | Free (API key required) |
| Ollama | AI inference | Free (local) |

## Scripts

```bash
npm run dev          # Development server
npm run build        # Production build
npm run start        # Production server
npm run lint         # ESLint
npm run db:generate  # Generate Drizzle migrations
npm run db:migrate   # Run Drizzle migrations
```
