# Worklog — the one place a task records what it learned

This directory replaces two things that did not scale:

- the **490 KB append-only `docs/USAGE_TEST_LEDGER.md` monolith** — one giant
  file every task had to edit (a rebase magnet across parallel terminals) and
  far too big to read whole; and
- the **"dated report → hand-maintained index → archive"** treadmill for
  backtest / verification write-ups.

Instead: **one fragment file per task or scenario**, written once, by the
terminal doing the work. The index (`INDEX.md`) is **generated** from the
fragments' front-matter — never hand-maintained.

This is also a coordination win (see `docs/PARALLEL_SESSIONS.md`): because each
task writes its *own* file, there is no shared file to collide on. The old
"one `USAGE_TEST_LEDGER.md` owner per cycle" rule goes away.

---

## What goes in a fragment

The answer to *"what did we try, what worked, what didn't, how did we fix it"* —
so the next agent learns from your work instead of re-deriving it. The
negative results (what you tried that **didn't** work, and why) are the
highest-value part: they are exactly what a fresh agent would otherwise repeat.

Each fragment is `docs/worklog/<id>-<slug>.md` with front-matter + a fixed body.

### Front-matter (machine-read by `gen_worklog_index.py`)

```yaml
---
id: S31                 # REQUIRED. Canonical id: Sn for a usage/backtest
                        #   scenario (assigned at MERGE, see PARALLEL_SESSIONS
                        #   rule 9), else a short slug.
title: Sever verbal news from the EV path    # REQUIRED.
kind: feature           # REQUIRED. feature|fix|backtest|verification|usage|
                        #   refactor|docs|research
status: merged          # REQUIRED. planned|in-flight|completed|merged|
                        #   abandoned|superseded
terminal: A             # optional — which terminal worked it
pr: 249                 # optional — the PR that shipped it
decisions: [D18]        # optional — DECISIONS.md entries this established
date: 2026-05-27        # optional
headline: news_mult pinned to 1.0; verbal news has zero EV influence  # optional
                        #   — the one-line takeaway shown in INDEX.md
surface: [engine/news_sentiment.py, tests/test_news_severance.py]     # optional
---
```

### Body (fixed sections — omit one only if it is genuinely empty)

```markdown
## Goal            — what we set out to do, and why
## What we tried   — approaches in the order we tried them
## What worked
## What didn't     — the dead ends + WHY (the part that saves the next agent)
## How we fixed it — the approach that shipped
## Evidence        — exact commands, numbers, links to raw artifacts
## Unresolved / handoff — what's still open; what the next agent should look at
```

---

## How to create one

```bash
python scripts/new_worklog.py S31 --title "Sever verbal news from the EV path" --kind feature
# writes docs/worklog/s31-sever-verbal-news-from-the-ev-path.md from the template
```

Then fill the sections and (re)generate the index:

```bash
python scripts/gen_worklog_index.py        # rewrites docs/worklog/INDEX.md
python scripts/gen_worklog_index.py --check # what CI runs — fails if INDEX is stale
```

`INDEX.md` is generated — **do not edit it by hand.** CI regenerates and diffs
it (`--check`); a stale index fails the build.

---

## What the index covers

`gen_worklog_index.py` indexes:

1. every fragment in this directory, and
2. the **dated backtest / verification reports** that still live in `docs/`
   (`ENGINE_BACKTEST_*.md`, `*REALISM_VERIFICATION*.md`, …). These are **indexed
   in place, not moved** — they carry 243 inbound references across 43 files
   (including `CLAUDE.md` and decision-layer docstrings), so relocating them is
   pure link-breakage risk for no functional gain. The generated `INDEX.md`
   gives the unified view the old `VERIFICATION_INDEX_*.md` provided by hand.
   Adding front-matter to one of those reports just enriches its index row.

## Migrated history

`docs/USAGE_TEST_LEDGER.md` is **frozen** (a banner points here). Its `S1`–`S46`
entries were split verbatim into fragments here, so nothing was lost and the
file stopped being an unbounded rebase magnet. New scenarios are fragments from
now on; never reopen the monolith.
