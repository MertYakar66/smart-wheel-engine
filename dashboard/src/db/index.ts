import Database from "better-sqlite3";
import { drizzle, type BetterSQLite3Database } from "drizzle-orm/better-sqlite3";
import * as schema from "./schema";
import path from "path";
import fs from "fs";

let _db: BetterSQLite3Database<typeof schema> | null = null;

function initDb(): BetterSQLite3Database<typeof schema> {
  if (_db) return _db;

  const DATA_DIR = path.join(process.cwd(), "data");
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }

  const DB_PATH = path.join(DATA_DIR, "finance-news.db");
  const sqlite = new Database(DB_PATH);

  // Enable WAL mode for better concurrent read performance
  sqlite.pragma("journal_mode = WAL");
  sqlite.pragma("busy_timeout = 5000");

  // Initialize tables if they don't exist
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS stories (
      story_id TEXT PRIMARY KEY,
      canonical_title TEXT NOT NULL,
      summary TEXT,
      impact_score REAL DEFAULT 0,
      sector TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      source_count INTEGER DEFAULT 1,
      impact_tags TEXT,
      exposure_mechanisms TEXT,
      impact_horizon TEXT,
      story_status TEXT DEFAULT 'developing',
      contradiction_flag INTEGER DEFAULT 0,
      first_seen_at TEXT,
      why_it_matters TEXT,
      corroboration_score REAL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS story_sources (
      source_id TEXT PRIMARY KEY,
      story_id TEXT REFERENCES stories(story_id),
      url TEXT NOT NULL,
      publisher TEXT,
      headline TEXT NOT NULL,
      published_at TEXT,
      snippet TEXT,
      retrieval_provider TEXT DEFAULT 'rss',
      rights_restricted INTEGER DEFAULT 0,
      sentiment TEXT,
      geography TEXT
    );

    CREATE TABLE IF NOT EXISTS story_entities (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      story_id TEXT REFERENCES stories(story_id),
      entity_type TEXT NOT NULL,
      entity_value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS story_timeline (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      story_id TEXT REFERENCES stories(story_id),
      event_type TEXT NOT NULL,
      description TEXT NOT NULL,
      metadata TEXT,
      occurred_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS user_exposures (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      exposure_type TEXT NOT NULL,
      exposure_value TEXT NOT NULL,
      weight REAL DEFAULT 1,
      added_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS news_categories (
      category_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      description TEXT,
      query_keywords TEXT NOT NULL,
      query_entities TEXT,
      query_tickers TEXT,
      query_sectors TEXT,
      query_countries TEXT,
      icon TEXT,
      color TEXT,
      sort_order INTEGER DEFAULT 0,
      enabled INTEGER DEFAULT 1,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS story_categories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      story_id TEXT REFERENCES stories(story_id),
      category_id TEXT REFERENCES news_categories(category_id),
      match_score REAL DEFAULT 0,
      matched_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS ingestion_runs (
      run_id TEXT PRIMARY KEY,
      run_type TEXT NOT NULL,
      started_at TEXT NOT NULL,
      completed_at TEXT,
      headlines_ingested INTEGER DEFAULT 0,
      stories_clustered INTEGER DEFAULT 0,
      impact_analyzed INTEGER DEFAULT 0,
      categories_matched INTEGER DEFAULT 0,
      status TEXT DEFAULT 'running',
      error TEXT
    );

    CREATE TABLE IF NOT EXISTS briefings (
      briefing_id TEXT PRIMARY KEY,
      briefing_type TEXT NOT NULL,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      story_ids TEXT NOT NULL,
      generated_at TEXT NOT NULL,
      period_start TEXT NOT NULL,
      period_end TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS market_snapshots (
      snapshot_id TEXT PRIMARY KEY,
      ticker TEXT NOT NULL,
      price REAL,
      change_pct REAL,
      volume INTEGER,
      captured_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS watchlists (
      ticker TEXT PRIMARY KEY,
      added_at TEXT NOT NULL,
      alert_threshold_pct REAL DEFAULT 5
    );

    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      event_type TEXT NOT NULL,
      ticker TEXT,
      event_date TEXT NOT NULL,
      description TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
      alert_id TEXT PRIMARY KEY,
      story_id TEXT REFERENCES stories(story_id),
      ticker TEXT,
      trigger_type TEXT NOT NULL,
      triggered_at TEXT NOT NULL,
      dismissed INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS chat_sessions (
      session_id TEXT PRIMARY KEY,
      title TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
      message_id TEXT PRIMARY KEY,
      session_id TEXT REFERENCES chat_sessions(session_id),
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_story_sources_story ON story_sources(story_id);
    CREATE INDEX IF NOT EXISTS idx_story_entities_story ON story_entities(story_id);
    CREATE INDEX IF NOT EXISTS idx_story_entities_value ON story_entities(entity_value);
    CREATE INDEX IF NOT EXISTS idx_story_timeline_story ON story_timeline(story_id);
    CREATE INDEX IF NOT EXISTS idx_user_exposures_type ON user_exposures(exposure_type);
    CREATE INDEX IF NOT EXISTS idx_story_categories_story ON story_categories(story_id);
    CREATE INDEX IF NOT EXISTS idx_story_categories_cat ON story_categories(category_id);
    CREATE INDEX IF NOT EXISTS idx_ingestion_runs_type ON ingestion_runs(run_type);
    CREATE INDEX IF NOT EXISTS idx_briefings_type ON briefings(briefing_type);
    CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker ON market_snapshots(ticker);
    CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker);
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
  `);

  // Migrations for existing databases: add new columns safely
  const migrations = [
    "ALTER TABLE stories ADD COLUMN impact_tags TEXT",
    "ALTER TABLE stories ADD COLUMN exposure_mechanisms TEXT",
    "ALTER TABLE stories ADD COLUMN impact_horizon TEXT",
    "ALTER TABLE stories ADD COLUMN story_status TEXT DEFAULT 'developing'",
    "ALTER TABLE stories ADD COLUMN contradiction_flag INTEGER DEFAULT 0",
    "ALTER TABLE stories ADD COLUMN first_seen_at TEXT",
    "ALTER TABLE stories ADD COLUMN why_it_matters TEXT",
    "ALTER TABLE stories ADD COLUMN corroboration_score REAL DEFAULT 0",
    "ALTER TABLE story_sources ADD COLUMN retrieval_provider TEXT DEFAULT 'rss'",
    "ALTER TABLE story_sources ADD COLUMN rights_restricted INTEGER DEFAULT 0",
    "ALTER TABLE story_sources ADD COLUMN sentiment TEXT",
    "ALTER TABLE story_sources ADD COLUMN geography TEXT",
  ];

  for (const migration of migrations) {
    try {
      sqlite.exec(migration);
    } catch {
      // Column already exists — safe to ignore
    }
  }

  _db = drizzle(sqlite, { schema });
  return _db;
}

// Lazy getter — DB is only initialized on first access
export const db = new Proxy({} as BetterSQLite3Database<typeof schema>, {
  get(_target, prop, receiver) {
    const realDb = initDb();
    return Reflect.get(realDb, prop, receiver);
  },
});
