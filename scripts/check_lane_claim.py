#!/usr/bin/env python3
"""Guard: a PR that edits a decision-layer file must explicitly claim it.

The three decision-layer files are the highest-contention surface in the
repo and the one place where two terminals editing concurrently silently
corrupt each other's work (the `select_book` double-build, PR #107 vs #109,
is the canonical near-miss). ``docs/PARALLEL_SESSIONS.md`` requires that
decision-layer edits be **serialised** — one terminal at a time, claimed on
the coordination board before branching.

Historically that rule was policed by prose: every claim comment carried a
manual sentence like *"Checked the board — no open claim touches
wheel_runner.py."* A human reading 275 freeform comments and hoping they did
not miss one is not an enforcement mechanism. This script turns the rule into
a CI gate.

What it enforces (the "hard gate on decision-layer only" from the 2026-05
coordination redesign — see ``docs/PARALLEL_SESSIONS.md``):

  * If a PR's diff touches NONE of the decision-layer files, it passes
    unconditionally. Non-decision-layer lane ownership stays advisory
    (Major-Session allocation + the board), by design — this gate is
    deliberately narrow so routine refactors never fight it.
  * If a PR's diff touches a decision-layer file, the PR description MUST
    carry a ``lane-claim`` block that names that file. A decision-layer edit
    with no claim fails the build, naming the offending file.

The claim is the conscious, auditable act of saying "I hold the
decision-layer lock for this file this cycle." It does NOT, by itself,
prove no *other* open PR also holds it — that cross-PR mutual exclusion is
guaranteed upstream by the single Major Session at allocation time (one
allocator => no race). This gate closes the much more common failure: a
terminal editing a decision-layer file without coordinating at all.

The claim block lives in the PR description (not a committed file) so it
needs no merge and cannot collide across branches::

    <!-- lane-claim
    files: engine/wheel_runner.py, engine/candidate_dossier.py
    board: https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-NNN
    -->

Matching is intentionally forgiving: a decision-layer path counts as claimed
if its exact repo-relative path appears anywhere inside the block. The fixed
``files:`` / ``board:`` keys are a convention for humans; the parser only
needs the path to be present.

Sources of the claim text, in priority order:

  1. ``--claim-file PATH`` — read the claim from a file (used by tests and
     by anyone who wants to dry-run a PR body).
  2. ``PR_BODY`` environment variable — set by the CI workflow to
     ``github.event.pull_request.body``.

If neither is provided the script is NOT in a PR context (e.g. a ``push``
build or a local invocation), so it prints a notice and exits 0 — the gate
only enforces on pull requests, where a description exists.

Changed files are computed as ``git diff --name-only <base>...HEAD`` against
``--base`` (default ``origin/main``); tests inject an explicit list via
``--changed-files`` to stay hermetic.

Exit 0 = OK / not-a-PR-context / no decision-layer files touched.
Exit 1 = a decision-layer file was touched without a matching claim.
Stdlib + git only — no third-party deps, same as the other CI guards.

Run:  python scripts/check_lane_claim.py --base origin/main
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# The decision-layer trio (CLAUDE.md §2 / PROJECT_STATE.md §1). Editing any of
# these is serialised one-terminal-at-a-time per docs/PARALLEL_SESSIONS.md.
DECISION_LAYER_FILES: tuple[str, ...] = (
    "engine/ev_engine.py",
    "engine/wheel_runner.py",
    "engine/candidate_dossier.py",
)

_CLAIM_BLOCK_RE = re.compile(r"<!--\s*lane-claim\b(.*?)-->", re.IGNORECASE | re.DOTALL)


def _git(args: list[str]) -> str:
    """Run a git command and return stdout (raises on non-zero)."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _changed_files(base: str) -> list[str]:
    """Repo-relative paths changed between ``base`` and HEAD.

    Uses the three-dot form so the comparison is against the merge-base of
    ``base`` and HEAD — i.e. only what this branch changed, not commits that
    landed on ``base`` after the branch was cut.
    """
    out = _git(["diff", "--name-only", f"{base}...HEAD"])
    return [line for line in out.splitlines() if line]


def _claim_text(claim_file: str | None) -> str | None:
    """Return the claim text, or None when there is no PR context.

    Priority: ``--claim-file`` then the ``PR_BODY`` env var. An *empty* body
    in a PR context is still a context (returns ``""`` -> enforced); only the
    total absence of both sources means "not a PR" (returns ``None`` -> skip).
    """
    if claim_file is not None:
        with open(claim_file, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    return os.environ.get("PR_BODY")


def _claimed_files(claim_text: str, touched: list[str]) -> set[str]:
    """Subset of ``touched`` that appears inside a lane-claim block."""
    blocks = _CLAIM_BLOCK_RE.findall(claim_text)
    if not blocks:
        return set()
    joined = "\n".join(blocks)
    return {f for f in touched if f in joined}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail a PR that edits a decision-layer file without claiming it."
    )
    ap.add_argument("--base", default=os.environ.get("LANE_CLAIM_BASE", "origin/main"))
    ap.add_argument("--claim-file", default=None, help="Read the PR body/claim from this file.")
    ap.add_argument(
        "--changed-files",
        nargs="*",
        default=None,
        help="Explicit changed-file list (bypasses git; for tests).",
    )
    args = ap.parse_args(argv)

    changed = args.changed_files if args.changed_files is not None else _changed_files(args.base)
    touched = [f for f in changed if f in DECISION_LAYER_FILES]

    if not touched:
        print("lane-claim: OK — no decision-layer files touched.")
        return 0

    claim_text = _claim_text(args.claim_file)
    if claim_text is None:
        print(
            "lane-claim: no claim source (PR_BODY unset and no --claim-file); "
            "not a PR context — gate not enforced."
        )
        print(f"  (decision-layer files touched: {', '.join(touched)})")
        return 0

    claimed = _claimed_files(claim_text, touched)
    unclaimed = sorted(set(touched) - claimed)

    print(f"decision-layer files touched : {', '.join(sorted(touched))}")
    print(f"claimed in PR description    : {', '.join(sorted(claimed)) or '(none)'}")

    if not unclaimed:
        print("lane-claim: OK — every decision-layer file touched is claimed.")
        return 0

    print()
    print("FAIL: decision-layer file(s) edited without a lane-claim:")
    for f in unclaimed:
        print(f"  unclaimed decision-layer edit : {f}")
    print()
    print("Fix: claim the file on the coordination board (#113), then add a")
    print("lane-claim block to the PR description naming it, e.g.:")
    print()
    print("  <!-- lane-claim")
    print(f"  files: {', '.join(unclaimed)}")
    print("  board: https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-NNN")
    print("  -->")
    print()
    print("Decision-layer edits are serialised one terminal at a time —")
    print("see docs/PARALLEL_SESSIONS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
