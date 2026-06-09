# AGENTS.md

This repository follows the Claude Code orientation contract. Any AI
agent — Claude Code, OpenAI Codex, Cursor, GitHub Copilot Agent, Aider,
or otherwise — entering this repo should follow the same canonical
read order.

## Read first

`CLAUDE.md` is auto-loaded on every Claude Code session (and re-injected
after `/compact`) — you have it by the time you read this. The mandatory
companion is:

1. **`PROJECT_STATE.md`** — what is authoritative right now, what is
   work-in-progress, what is deprecated/phantom, and where the live
   documentation drift is. Read this before assuming any file or
   pyproject entry is current.

For a fast "where does X live / what tests cover Y / what is authoritative for Z"
lookup, **`docs/REPO_MAP.md`** routes each question to the one owning doc (so you
don't open three and reconcile them).

## Consult when your task requires it

| Trigger | Read |
|---|---|
| Fast where/what/authoritative router (one stop) | `docs/REPO_MAP.md` |
| Navigating `engine/`, `advisors/`, or a parallel news tree | `MODULE_INDEX.md` |
| Touching code that needs tests run | `TESTING.md` |
| **Locating a file by name or directory — `grep` it, do not read in full** | `FILE_MANIFEST.md` |
| Rehydrating data on a fresh machine (Theta Terminal, feature store) | `docs/LAPTOP_SETUP.md` |
| Another session may be active on this repo (shared working tree) | `docs/PARALLEL_SESSIONS.md` (and the coordination board pinned in GitHub issues) |
| You're told **"You are responsible for the Dashboard"** (the dashboard terminal) | `docs/DASHBOARD_TERMINAL.md` |

## Read on demand

Once oriented, pick the doc that matches your task. Each one is
purpose-built; they don't overlap.

| You want to… | Read |
|---|---|
| See *why* a structural choice was made | `DECISIONS.md` |
| Find what's intentionally next | `ROADMAP.md` |
| See what shipped recently | `CHANGELOG.md` |
| See what *usage* tests we've run + every backtest / verification record | `docs/worklog/INDEX.md` (per-task fragments; `docs/USAGE_TEST_LEDGER.md` is the frozen pre-2026-05-29 monolith) |
| Understand the data layer (what's tracked, regen rules, secrets) | `docs/DATA_POLICY.md` |
| Know the launch-blocker invariants before merging | `docs/LAUNCH_READINESS.md` |
| Write a commit message or PR | `COMMIT_GUIDE.md` |
| Wire anything to TradingView (engine bridge or analyst workspace) | `docs/TRADINGVIEW_INTEGRATION.md`, then `docs/TRADINGVIEW_MCP_INTEGRATION.md` for MCP design |
| Run a Theta data refresh | `docs/THETA_PULL_SESSION_NOTES.md`, then `docs/THETA_USAGE.md` |
| Touch Greek calculations | `docs/GREEKS_UNIT_CONTRACT.md` (canonical) |

## The hard rule

From `CLAUDE.md` §2:

> **No tradeable candidate bypasses `EVEngine.evaluate`.**

Reviewers (chart provider, advisor committee, dealer positioning,
TradingView bridge) can **downgrade** a verdict. None of them can
**rescue** a negative-EV trade. (News sentiment used to be a
downgrade-only reviewer; D18 severed it from the EV path —
`engine/news_sentiment.py` is now an operator-transparency layer.)

If you add a new input — new data source, new advisor, a TradingView
MCP feed, a quantitative news layer — wire it as a chained-provider
participant or a downgrade-only reviewer. Do not introduce a code path
that converts a non-tradeable candidate into a tradeable one without
a fresh `EVEngine.evaluate` call.

The canonical reviewer rule set is **R1–R11** (`CLAUDE.md` §2). R11 is
the elevated-vol top-bin size-down (`DECISIONS.md` D23) — wired live,
downgrade-only. This contract is enforced structurally by
`tests/test_audit_invariants.py`, `tests/test_dossier_invariant.py`
(R1–R10), `tests/test_r11_elevated_vol.py` (R11), and
`tests/test_authority_hardening.py`. Never move or merge a change
touching these without §2-owner sign-off.

## What you can change without asking

- Module-level docstrings; per-function docstrings.
- New tests under `tests/` that exercise existing public surface.
- A **worklog fragment** under `docs/worklog/` recording your task — what you
  tried, what worked, what didn't, how you fixed it. Scaffold with
  `python scripts/new_worklog.py`; refresh the index with
  `python scripts/gen_worklog_index.py`. This is *the* place to document work
  (see `docs/PARALLEL_SESSIONS.md` "Where to document").
- Documentation (`docs/`, `*.md` at repo root) **except** `CLAUDE.md`,
  which the user maintains directly.
- Files under your active feature branch (never `main`).
- Updates to `CHANGELOG.md` / `PROJECT_STATE.md` / `ROADMAP.md` /
  `DECISIONS.md` that reflect the work you just did. Keep these
  in sync with the code as you ship.

## What requires explicit ask

- Any change under `engine/ev_engine.py`,
  `engine/wheel_runner.py`, or `engine/candidate_dossier.py`.
- Any new data-provider class, advisor, or chart provider.
- Adding broker / OMS / order-routing surface — out of scope per
  `CLAUDE.md`'s NEVER list.
- Editing `CLAUDE.md` itself.
- Committing to `main` directly (see `COMMIT_GUIDE.md` §5 — branch
  + PR for everything, even one-line typos).
- Refreshing the tracked Bloomberg CSVs under `data/bloomberg/`
  (`sp500_*.csv`, `treasury_yields.csv`) — these are data commits with
  audit value (see `ROADMAP.md` C1 — the track-vs-gitignore decision
  is open).

## Commit format

The repo uses a structured commit format (`type(scope): summary`
plus a body with `Changed:` / `Why:` / `Tested:` / `Tried but
rejected:` / `Unresolved:` / `AI handoff:` sections). Read
`COMMIT_GUIDE.md` before your first commit.
