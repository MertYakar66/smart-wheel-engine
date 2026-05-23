# Contributing to Smart Wheel Engine

> **AI agents (Claude, Codex, Cursor, Copilot, Aider, …):** Start with
> [`AGENTS.md`](../AGENTS.md), then [`CLAUDE.md`](../CLAUDE.md) for the
> structural contract, [`PROJECT_STATE.md`](../PROJECT_STATE.md) for
> what's authoritative right now, and [`COMMIT_GUIDE.md`](../COMMIT_GUIDE.md)
> before your first commit. The sections below are the human-side
> open-source workflow; the AI-agent handoff path is more direct.

---

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+ if you'll touch the Next.js dashboard
- Git

### Setup

```bash
git clone https://github.com/MertYakar66/smart-wheel-engine.git
cd smart-wheel-engine

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\Activate.ps1    # Windows PowerShell

# Install runtime + test deps
pip install -r requirements.txt
pip install pytest pytest-cov hypothesis ruff pre-commit

# Pre-commit hooks
pre-commit install
```

> **Note on `pyproject.toml`.** The project carries a `pyproject.toml`
> for tooling config (ruff, mypy, pytest, coverage) and a packaging
> surface (hatchling). `pip install -e ".[dev]"` is what CI uses
> and is the recommended install for contributors who want the
> wheel build path. The plain `requirements.txt` install above is
> equivalent for runtime + test deps and is the bring-up path used
> on fresh machines. The known-stale state flagged in earlier
> `PROJECT_STATE.md` §5 (broken `wheel = "src.cli:app"` script,
> `packages = ["src"]` excluding the live code, two phantom
> `[project.dependencies]` entries) was closed by ROADMAP Track B5
> — see `CHANGELOG.md` 2026-05.

For Theta Terminal bring-up on a new machine (not needed if you're
using the committed Bloomberg CSVs, which is the default), see
[`docs/LAPTOP_SETUP.md`](LAPTOP_SETUP.md).

---

## Development workflow

### 1. Branch off `main`

```bash
git checkout -b agent/<short-slug>          # for AI-agent work
git checkout -b claude/<short-slug>         # also accepted
git checkout -b feature/<short-slug>        # human convention
```

Never commit to `main` directly (`CLAUDE.md`'s NEVER list).

### 2. Make changes

- Follow existing code style — Ruff is the formatter.
- Type-hint new functions; the codebase is mypy-strict in scope.
- Add tests for new functionality.
- If you're touching a doc, prefer **symbol names** over
  `file.py:line` references — line numbers drift.

### 3. Run the relevant tests

```bash
# Full suite (always run for decision-layer changes)
pytest tests/ -v

# Launch-blocker subset (decision-layer gate)
pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py \
       tests/test_authority_hardening.py tests/test_audit_viii_*.py \
       tests/test_launch_blockers.py -v

# Targeted (e.g. when touching engine/ev_engine.py)
pytest tests/test_ev_engine_upgrades.py -v
```

See [`TESTING.md`](../TESTING.md) for the full taxonomy, launch-blocker
subset, and the "what to run when you touch X" map.

For any change touching `engine/ev_engine.py`, `engine/wheel_runner.py`,
or `engine/candidate_dossier.py`: run the **full** suite, not just the
targeted file. Invariants are cross-cutting.

### 4. Pre-commit + lint

```bash
# Run pre-commit on all changed files
pre-commit run --files <path1> <path2>

# Or let it run automatically on commit
git commit -m "type(scope): ..."

# Explicit lint check (CI hard-fails on dirty lint)
ruff format --check .
ruff check .
```

### 5. Submit the PR

```bash
git push -u origin <your-branch>
gh pr create --base main --title "type(scope): ..." --body "..."
```

CI runs eight jobs (Environment Validation, Lint & Type Check,
Security Scan, Test Suite × 2, Quantitative Validation, Integration
Tests, FILE_MANIFEST Coverage). All eight must pass before merge.

---

## Code guidelines

### Python style

- PEP 8 + Ruff defaults.
- Type hints on all functions; mypy-strict expected.
- Docstrings on public functions (Args / Returns / Raises).
- Keep functions focused; if a function grows past ~80 lines, ask
  whether it's two functions.

```python
def calculate_ev_dollars(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    *,
    multiplier: float = 1.0,
) -> float:
    """Compute expected-value dollars for a short-put leg.

    Args:
        spot: Underlying spot price.
        strike: Put strike.
        iv: Annualised implied volatility (decimal, not percent).
        dte: Days to expiration.
        multiplier: Regime / dealer-positioning multiplier (clamped
            to [0.70, 1.05] by the dealer overlay).

    Returns:
        Expected-value dollars; negative if the trade is non-tradeable.
    """
    ...
```

Greeks units are pinned in [`docs/GREEKS_UNIT_CONTRACT.md`](GREEKS_UNIT_CONTRACT.md).
Read it before touching any Greek code.

### Testing

- Tests live in `tests/`, file pattern `test_*.py`.
- Use descriptive test names (`test_atm_call_delta_near_half`,
  not `test_1`).
- For decision-layer changes, add a test that pins the invariant,
  not just one that covers the new code path.

```python
class TestEnginePhaseReviewer:
    def test_negative_ev_blocked_even_with_perfect_chart(self):
        """R1 invariant — chart cannot upgrade a negative-EV verdict."""
        ...
```

The launch-blocker test family (`test_audit_*`,
`test_dossier_invariant`, `test_authority_hardening`,
`test_launch_blockers`) is what enforces the hard EV invariant
across the codebase. New invariants belong there.

### Commits

The repo uses a structured `type(scope): summary` format with a
labelled body (`Changed:` / `Why:` / `Tested:` / `Tried but
rejected:` / `Unresolved:` / `AI handoff:`). The full standard with
worked examples lives in [`COMMIT_GUIDE.md`](../COMMIT_GUIDE.md) —
read it before your first commit.

Quick template:

```
type(scope): one-line summary in imperative voice

Changed:
- bullet

Why:
- the motivation, the constraint, the past incident

Tested:
- the exact command(s) you ran

AI handoff:
- a hint for the next agent (what to look at next, what this
  exposes, what test to add later)
```

---

## What requires explicit ask

Per [`AGENTS.md`](../AGENTS.md), the following are *not* covered by
the "what you can change without asking" allow-list:

- Any change under `engine/ev_engine.py`, `engine/wheel_runner.py`,
  or `engine/candidate_dossier.py`.
- New data-provider classes, advisors, or chart providers.
- Broker / OMS / order-routing surface (out of scope per
  `CLAUDE.md`'s NEVER list).
- Editing `CLAUDE.md` itself.
- Refreshing the committed Bloomberg CSVs.

When in doubt, ask before doing.

---

## Areas open for contribution

The current backlog is tracked in [`ROADMAP.md`](../ROADMAP.md):

- **Track A** — decision-layer correctness (highest priority).
- **Track B** — documentation drift (a few items still pending).
- **Track C** — hygiene + governance follow-ups.
- **Track F** — lint debt (44 ruff errors on main).

`DECISIONS.md` records the architectural choices and what
was rejected. `PROJECT_STATE.md` records the temporal state — read
it before assuming any file or pyproject entry is current.

---

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
