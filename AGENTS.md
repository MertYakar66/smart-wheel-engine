# AGENTS.md

This repository follows the Claude Code orientation contract. Any AI
agent — Claude Code, OpenAI Codex, Cursor, GitHub Copilot Agent, Aider,
or otherwise — entering this repo should follow the same canonical
read order.

## Read first

1. **`CLAUDE.md`** — single source of orientation. Pins the four-layer
   mental model (data → quant → decision → interface), the hard EV
   invariant, the data-provider matrix (`SWE_DATA_PROVIDER`), what is
   explicitly out of scope, and the bring-up checklist for a fresh
   session.

2. **`PROJECT_STATE.md`** — what is authoritative right now, what is
   work-in-progress, what is deprecated/phantom, and where the live
   documentation drift is. Read this before assuming any file or
   pyproject entry is current.

3. **`MODULE_INDEX.md`** — per-module purpose and decision-layer
   relationship. Use this to navigate `engine/`, `advisors/`, and the
   parallel news/agent trees without rediscovering the topology.

4. **`TESTING.md`** — test taxonomy, launch-blocker subset, what to
   run when you touch a given file.

5. **`LAPTOP_SETUP.md`** — only if rehydrating data (Theta Terminal,
   feature store) on a new machine. Not needed for code-only work.

## Read on demand

Once oriented, pick the doc that matches your task. Each one is
purpose-built; they don't overlap.

| You want to… | Read |
|---|---|
| See *why* a structural choice was made | `DECISIONS.md` |
| Find what's intentionally next | `ROADMAP.md` |
| See what shipped recently | `CHANGELOG.md` |
| Understand the data layer (what's tracked, regen rules, secrets) | `DATA_POLICY.md` |
| Know the launch-blocker invariants before merging | `LAUNCH_READINESS.md` |
| Write a commit message or PR | `COMMIT_GUIDE.md` |
| Wire anything to TradingView (engine bridge or analyst workspace) | `TRADINGVIEW_INTEGRATION.md`, then `docs/TRADINGVIEW_MCP_INTEGRATION.md` for MCP design |
| Run a Theta data refresh | `docs/THETA_PULL_SESSION_NOTES.md`, then `docs/THETA_USAGE.md` |
| Touch Greek calculations | `docs/GREEKS_UNIT_CONTRACT.md` (canonical) |

## The hard rule

From `CLAUDE.md` §2:

> **No tradeable candidate bypasses `EVEngine.evaluate`.**

Reviewers (chart provider, news sentiment, advisor committee, dealer
positioning, TradingView bridge) can **downgrade** a verdict.
None of them can **rescue** a negative-EV trade.

If you add a new input — new data source, new advisor, a TradingView
MCP feed, a sentiment model — wire it as a chained-provider
participant or a downgrade-only reviewer. Do not introduce a code path
that converts a non-tradeable candidate into a tradeable one without
a fresh `EVEngine.evaluate` call. This is enforced structurally by
`tests/test_audit_invariants.py`, `tests/test_dossier_invariant.py`,
and `tests/test_authority_hardening.py`.

## What you can change without asking

- Module-level docstrings; per-function docstrings.
- New tests under `tests/` that exercise existing public surface.
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
  `CLAUDE.md` §4.
- Editing `CLAUDE.md` itself.
- Touching `pyproject.toml` `[project.scripts]` or
  `[tool.hatch] packages` (currently in a known-stale state — see
  `PROJECT_STATE.md`).
- Committing to `main` directly (see `COMMIT_GUIDE.md` §5 — branch
  + PR for everything, even one-line typos).
- Refreshing the tracked Bloomberg CSVs in
  `data/bloomberg/sp500_*.csv`, `treasury_yields.csv` — these are
  data commits with audit value (see `ROADMAP.md` C1 — the
  track-vs-gitignore decision is open).

## Commit format

The repo uses a structured commit format (`type(scope): summary`
plus a body with `Changed:` / `Why:` / `Tested:` / `Tried but
rejected:` / `Unresolved:` / `AI handoff:` sections). Read
`COMMIT_GUIDE.md` before your first commit.
