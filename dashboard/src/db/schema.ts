import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";

// Local SQLite store for the terminal's operator state — the ticker
// watchlist, manually curated calendar events, cached quote snapshots and
// research-chat sessions. Nothing here feeds the EV path; the engine
// (engine_api.py) is the only decision authority.

// ─── Market Snapshots ──────────────────────────────────────────────────
export const marketSnapshots = sqliteTable("market_snapshots", {
  snapshotId: text("snapshot_id").primaryKey(),
  ticker: text("ticker").notNull(),
  price: real("price"),
  changePct: real("change_pct"),
  volume: integer("volume"),
  capturedAt: text("captured_at").notNull(),
});

// ─── Watchlists ────────────────────────────────────────────────────────
export const watchlists = sqliteTable("watchlists", {
  ticker: text("ticker").primaryKey(),
  addedAt: text("added_at").notNull(),
});

// ─── Events ────────────────────────────────────────────────────────────
export const events = sqliteTable("events", {
  eventId: text("event_id").primaryKey(),
  eventType: text("event_type").notNull(), // earnings | fomc | cpi | jobs | gdp
  ticker: text("ticker"),
  eventDate: text("event_date").notNull(),
  description: text("description"),
});

// ─── Chat Sessions ─────────────────────────────────────────────────────
export const chatSessions = sqliteTable("chat_sessions", {
  sessionId: text("session_id").primaryKey(),
  title: text("title"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

// ─── Messages ──────────────────────────────────────────────────────────
export const messages = sqliteTable("messages", {
  messageId: text("message_id").primaryKey(),
  sessionId: text("session_id").references(() => chatSessions.sessionId),
  role: text("role").notNull(), // user | assistant | system
  content: text("content").notNull(),
  createdAt: text("created_at").notNull(),
});
