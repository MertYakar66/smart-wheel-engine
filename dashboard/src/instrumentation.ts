// Next.js instrumentation hook — runs once per server boot. The only job
// here is starting the scheduled news-ingestion cron; everything else
// (DB init, category seeding) is lazy or owned by the cron starter.
export async function register() {
  // node-cron and better-sqlite3 only exist in the nodejs runtime.
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const { startNewsCron } = await import("@/services/news-cron");
  startNewsCron();
}
