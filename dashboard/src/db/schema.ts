import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";

// Local SQLite store for the terminal's operator state — the ticker
// watchlist, manually curated calendar events and cached quote snapshots.
// Nothing here feeds the EV path; the engine (engine_api.py) is the only
// decision authority.

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
