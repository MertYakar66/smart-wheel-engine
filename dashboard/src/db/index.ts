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

  // The file name predates the 2026-09-16 news removal and is kept so an
  // existing local watchlist is still found. Tables that earlier versions
  // created in an existing file (the news pipeline, the research chat removed
  // on 2026-09-17) are left alone — never dropped.
  const DB_PATH = path.join(DATA_DIR, "finance-news.db");
  const sqlite = new Database(DB_PATH);

  // Enable WAL mode for better concurrent read performance
  sqlite.pragma("journal_mode = WAL");
  sqlite.pragma("busy_timeout = 5000");

  // Initialize tables if they don't exist
  sqlite.exec(`
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
      added_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      event_type TEXT NOT NULL,
      ticker TEXT,
      event_date TEXT NOT NULL,
      description TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker ON market_snapshots(ticker);
  `);

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
