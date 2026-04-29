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
