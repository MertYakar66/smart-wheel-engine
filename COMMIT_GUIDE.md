# Commit & PR Governance

The repo's history is its second test suite. A commit message that
explains *why* (not just *what*) lets the next agent — human or
LLM — recover context months later from `git log` alone, without
having to re-read the whole tree.

This file pins the standard. It is short on purpose; the value is in
the structure, not the rules.

---

## 1. Commit message format

```
type(scope): one-line summary in imperative voice

Changed:
- bullet

Why:
- bullet (the motivation, the constraint, the past incident)

Tested:
- bullet (the command you actually ran)

Tried but rejected:
- bullet (the alternative that didn't work and the reason)

Unresolved:
- bullet (anything you noticed but didn't fix in this commit)

AI handoff:
- bullet (a hint for the next agent — what to look at next, what
  this exposes, what test to add later)
```

The first line is what shows in `git log --oneline`. The body is the
five labelled sections. **Omit a section entirely if it's empty** —
don't write "Tried but rejected: N/A". An empty section is noise;
a missing section is silence.

### Allowed `type` values

| Type | Use for |
|---|---|
| `feat` | A new capability that didn't exist before |
| `fix` | A bug fix |
| `docs` | Documentation only — no runtime change |
| `refactor` | Internal restructure with no behaviour change |
| `test` | Test-only changes (added, refactored, deleted) |
| `chore` | Hygiene: gitignore, CI, dependencies, formatting |
| `audit` | Audit-cycle work that pins invariants (see audit-i to audit-viii history) |
| `perf` | Performance improvement with no behaviour change |

### `scope` examples

Pick the smallest module the commit really touches. Examples from
existing history:

`pull_theta_iv_surface_history`, `theta_connector`, `wheel_runner`,
`ev_engine`, `candidate_dossier`, `tradingview_bridge`,
`session_start`, `pyproject`, `gitignore`, `pull_all`,
`PROJECT_STATE`, `MODULE_INDEX`, `THETA_USAGE`, `iv_surface`.

If you genuinely touch many scopes (a real cross-cutting change),
use a category: `engine`, `scripts`, `docs`, `tests`, `infra`.

---

## 2. Worked example

A real commit that follows the standard:

```
fix(pull_theta_iv_surface_history): per-bucket fallback to next-nearest with data

Changed:
- get_iv_surface_history now retries the next-nearest expiry bucket
  when the requested bucket returns an empty frame.
- Strict-mode partial-coverage rejection upgraded to record the
  rejected bucket count in the run summary.

Why:
- The 2026-04-23 puller hit ~120 strict-mode rejections (PR #58) on
  tickers where one of the three target buckets had no quotes that
  day. Falling through to the next bucket is correct: the surface
  is interpolated across buckets anyway, and a 30d ↔ 45d swap is
  inside the noise floor of an SVI fit.
- Without fallback, those 120 tickers showed as missing in
  iv_surface_history, masking the difference between "ticker has
  no IV history at all" (real failure) and "this single bucket
  was empty for this date" (recoverable).

Tested:
- pytest tests/test_iv_surface_history_puller.py -v (4 new cases)
- python scripts/pull_theta_iv_surface_history.py --tickers AAPL,MSFT,SPY
  on the laptop with Terminal up: 0 rejections after the fix.

Tried but rejected:
- Linearly interpolating between adjacent buckets in-puller.
  Defers the surface fit to a place that doesn't have the SVI
  machinery; better to leave interpolation to volatility_surface.py.

Unresolved:
- The 4 persistent failures (BF.B / BRK.B / NVR / DAY) remain. Root
  cause is upstream-data gap, not format — see PROJECT_STATE.md
  §3.4. Will not be fixed by this puller.

AI handoff:
- If you wire this into a feature, see DECISIONS.md D9 (iv_surface
  contract is still open). Don't silently substitute a flat-IV
  fallback.
```

---

## 3. PR description format

PRs go on top of commits with the same five-section body, plus a
header summary. Example:

```markdown
## Summary
<2-3 bullets — what, why, scope>

## Changes
- file.py — what changed
- other.py — what changed

## Why
- one paragraph or bullets

## Tests
- exact commands run
- what they verified

## Tried but rejected
- alternative + reason

## Unresolved
- known follow-ups

## AI handoff
- next-step hints; what would benefit from a deeper review;
  open questions a future agent should track
```

PR titles follow the same `type(scope): summary` format as commit
subjects. Keep them under 70 characters.

---

## 4. The five sections, expanded

### `Changed`
The factual list of what's different now versus before. Files +
behaviour. No motivation here. One bullet per concrete change; if
you can't list them in 3-5 bullets, the commit is probably too big.

### `Why`
The motivation. Past incidents are gold here — "we got burned last
quarter when X" is exactly the kind of context that doesn't survive
in code comments but is critical to keep. Constraints (vendor API
limits, regulatory requirements, latency budgets) belong here too.

### `Tested`
The exact commands you ran, not "I tested it". A future agent should
be able to copy-paste your tested commands to reproduce. If you
manually verified something (browser, dashboard, external service),
say so — manual verification is fine, but flag it as manual.

### `Tried but rejected`
The single highest-value section for keeping context. Future agents
will independently consider the same alternatives; without this
section, they re-litigate. Common reasons to use it:
- Performance trade-offs explored
- Alternative API designs you considered
- Library choices made and why
- Test approaches that didn't catch the bug

If you didn't try alternatives, omit the section.

### `Unresolved`
Things you noticed but didn't fix. Better here than in a TODO comment
that rots in code. Examples:
- A nearby code smell you saw but is out of scope
- A test that flakes intermittently and you didn't investigate
- A doc that's stale and you didn't update
- An assumption you made that should probably be a config flag

### `AI handoff`
A hint for the next agent. The next agent doesn't know what you
intended to do next. Tell them. Examples:
- "Wire this into the EV path next — see ROADMAP.md A2"
- "If you add a new puller, copy this same retry pattern"
- "The 4 persistent failures are upstream — don't waste time
  debugging the puller for them"
- "This change should probably be paired with a doc update in
  TRADINGVIEW_INTEGRATION.md §3"

---

## 5. Branch naming

| Convention | Use |
|---|---|
| `claude/<short-slug>` | Default for Claude Code work |
| `codex/<short-slug>` | OpenAI Codex / Cursor agent work |
| `feat/<short-slug>` | Human-driven feature work |
| `fix/<short-slug>` | Human-driven bug fix |

Slugs are kebab-case, 2-4 words, descriptive. Examples from existing
history: `claude/fix-iv-surface-bucket-fallback`,
`claude/handoff-docs`, `claude/theta-usage-doc`.

**Never edit `main` directly.** Even a one-line README typo fix gets
a branch + PR. The audit trail is part of governance.

---

## 6. AI agent attribution

When an AI agent commits, use a `Co-Authored-By` trailer in the
commit body so the history reflects who shipped what. Example:

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

The model name + capability hint (1M context, fast mode, etc.) helps
later debugging when a class of bugs traces to a specific model
generation.

---

## 7. Anti-patterns to avoid

- **"WIP" or "stuff" commit messages.** Every commit goes through
  PR review eventually; rename before pushing.
- **`git commit -m "fix typo"` 17 times.** Squash before opening
  the PR. Each commit on `main` should stand alone as a meaningful
  unit.
- **`--no-verify` to skip pre-commit hooks.** If a hook fails,
  the right move is to fix the underlying issue, not bypass.
- **Force-pushing to a shared branch.** Branches with PRs open are
  shared; reviewers may have local checkouts. Rebase locally and
  push as a new commit, or open a follow-up PR.
- **Committing files containing secrets** — `.env`, `creds.txt`,
  `*_credentials.json`. The `.gitignore` covers known paths;
  `git diff --staged` before commit catches anything new.

---

## 8. The bare minimum

If you remember nothing else from this file: every non-trivial
commit should answer **why** in the body. Code comments rot; commit
messages are immutable history. Use them.
