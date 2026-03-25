import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";

// ─── Stories ───────────────────────────────────────────────────────────
export const stories = sqliteTable("stories", {
  storyId: text("story_id").primaryKey(),
  canonicalTitle: text("canonical_title").notNull(),
  summary: text("summary"),
  impactScore: real("impact_score").default(0),
  sector: text("sector"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
  sourceCount: integer("source_count").default(1),
  // ─── Story Graph Intelligence fields ──────────────────────────────
  impactTags: text("impact_tags"),           // JSON: ["rates","oil","fx","regulation","earnings"]
  exposureMechanisms: text("exposure_mechanisms"), // JSON: [{"factor":"rates","direction":"negative","confidence":0.8}]
  impactHorizon: text("impact_horizon"),     // "intraday" | "days" | "weeks" | "quarters"
  storyStatus: text("story_status").default("developing"), // "developing" | "evolving" | "resolved"
  contradictionFlag: integer("contradiction_flag").default(0),
  firstSeenAt: text("first_seen_at"),
  whyItMatters: text("why_it_matters"),      // AI-generated narrative
  corroborationScore: real("corroboration_score").default(0), // 0-1 multi-source agreement
});

// ─── Story Sources ─────────────────────────────────────────────────────
export const storySources = sqliteTable("story_sources", {
  sourceId: text("source_id").primaryKey(),
  storyId: text("story_id").references(() => stories.storyId),
  url: text("url").notNull(),
  publisher: text("publisher"),
  headline: text("headline").notNull(),
  publishedAt: text("published_at"),
  snippet: text("snippet"),
  retrievalProvider: text("retrieval_provider").default("rss"), // "rss" | "valyu" | "manual"
  rightsRestricted: integer("rights_restricted").default(0),
  sentiment: text("sentiment"),              // "positive" | "negative" | "neutral" | "mixed"
  geography: text("geography"),              // country/region code
});

// ─── Story Entities ────────────────────────────────────────────────────
export const storyEntities = sqliteTable("story_entities", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  storyId: text("story_id").references(() => stories.storyId),
  entityType: text("entity_type").notNull(), // ticker | person | org | topic | factor
  entityValue: text("entity_value").notNull(),
});

// ─── Story Timeline ───────────────────────────────────────────────────
export const storyTimeline = sqliteTable("story_timeline", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  storyId: text("story_id").references(() => stories.storyId),
  eventType: text("event_type").notNull(),   // "created" | "source_added" | "merged" | "status_change" | "contradiction_detected"
  description: text("description").notNull(),
  metadata: text("metadata"),                // JSON: additional data
  occurredAt: text("occurred_at").notNull(),
});

// ─── User Exposures (for exposure-first ranking) ──────────────────────
export const userExposures = sqliteTable("user_exposures", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  exposureType: text("exposure_type").notNull(), // "ticker" | "sector" | "factor" | "country" | "theme"
  exposureValue: text("exposure_value").notNull(),
  weight: real("weight").default(1),             // importance weight for ranking
  addedAt: text("added_at").notNull(),
});

// ─── News Categories (saved queries / Bloomberg MYN-style) ────────────
export const newsCategories = sqliteTable("news_categories", {
  categoryId: text("category_id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  // Query definition as JSON
  queryKeywords: text("query_keywords").notNull(),       // JSON: ["fed","rate","fomc"]
  queryEntities: text("query_entities"),                 // JSON: ["FED","ECB","BOJ"]
  queryTickers: text("query_tickers"),                   // JSON: ["SPY","TLT"]
  querySectors: text("query_sectors"),                   // JSON: ["Macro","Financials"]
  queryCountries: text("query_countries"),                // JSON: ["US","EU","JP"]
  // Display
  icon: text("icon"),                                     // lucide icon name
  color: text("color"),                                   // hex or tailwind color
  sortOrder: integer("sort_order").default(0),
  enabled: integer("enabled").default(1),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

// ─── Story Categories (many-to-many mapping) ─────────────────────────
export const storyCategories = sqliteTable("story_categories", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  storyId: text("story_id").references(() => stories.storyId),
  categoryId: text("category_id").references(() => newsCategories.categoryId),
  matchScore: real("match_score").default(0),            // 0-1 relevance to category
  matchedAt: text("matched_at").notNull(),
});

// ─── Ingestion Runs (scheduled pipeline tracking) ─────────────────────
export const ingestionRuns = sqliteTable("ingestion_runs", {
  runId: text("run_id").primaryKey(),
  runType: text("run_type").notNull(),                   // "morning" | "evening" | "manual" | "hot"
  startedAt: text("started_at").notNull(),
  completedAt: text("completed_at"),
  headlinesIngested: integer("headlines_ingested").default(0),
  storiesClustered: integer("stories_clustered").default(0),
  impactAnalyzed: integer("impact_analyzed").default(0),
  categoriesMatched: integer("categories_matched").default(0),
  status: text("status").default("running"),             // "running" | "completed" | "failed"
  error: text("error"),
});

// ─── Briefings (morning/evening digests) ──────────────────────────────
export const briefings = sqliteTable("briefings", {
  briefingId: text("briefing_id").primaryKey(),
  briefingType: text("briefing_type").notNull(),         // "morning" | "evening" | "breaking"
  title: text("title").notNull(),
  content: text("content").notNull(),                    // JSON: sections with story references
  storyIds: text("story_ids").notNull(),                 // JSON: ordered list of story IDs
  generatedAt: text("generated_at").notNull(),
  periodStart: text("period_start").notNull(),           // coverage window start
  periodEnd: text("period_end").notNull(),               // coverage window end
});

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
  alertThresholdPct: real("alert_threshold_pct").default(5),
});

// ─── Events ────────────────────────────────────────────────────────────
export const events = sqliteTable("events", {
  eventId: text("event_id").primaryKey(),
  eventType: text("event_type").notNull(), // earnings | fomc | cpi | jobs | gdp
  ticker: text("ticker"),
  eventDate: text("event_date").notNull(),
  description: text("description"),
});

// ─── Alerts ────────────────────────────────────────────────────────────
export const alerts = sqliteTable("alerts", {
  alertId: text("alert_id").primaryKey(),
  storyId: text("story_id").references(() => stories.storyId),
  ticker: text("ticker"),
  triggerType: text("trigger_type").notNull(), // news | price_move
  triggeredAt: text("triggered_at").notNull(),
  dismissed: integer("dismissed").default(0),
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
