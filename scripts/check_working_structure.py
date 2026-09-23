#!/usr/bin/env python3
"""Does the working structure still match the repository? (DECISIONS.md D32)

The rule-books are prose, and prose rots without breaking any build. This check
makes the countable and the linkable parts loud. Each failure names its fix:

1. The ``AGENTS.md`` Appendix is ``CLAUDE.md`` §1–§6, word for word (Codex loads
   the one, Claude the other; they must say the same thing).
2. ``PROJECT_STATE.md`` records the main commit (a "`main` is at `<hash>`" line),
   which session-open measures drift from; in a full clone the hash must exist.
3. ``docs/deadlines.md`` has its table, and every row has a Due (an ISO date,
   "Conditional" or "Undated") and a Status.
4. ``data/DATA_MANIFEST.json`` records a frontier date for every dated dataset
   (the data slot of the pen's mark).
5. The marks ``CLAUDE.md`` shows are the ones ``scripts/session_open.py`` prints.
6. Every repository path the rule-books cite exists. Paths under the data trees
   (git holds no data since D31), placeholders and fenced examples are skipped.

What this cannot do: judge whether prose is true. When it fails, that is the
system working: fix the document it names in the same commit; never weaken the
check to make it pass. Stdlib and git only.

    python scripts/check_working_structure.py        # exit 0 = all pass, 1 = a failure
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RULE_BOOKS = (
    "CLAUDE.md",
    "AGENTS.md",
    "OPERATING_MODEL.md",
    "docs/PROMPTING_STANDARD.md",
    "docs/deadlines.md",
)
MARK_PREFIXES = (
    "Report from the Strategist, Sir —",
    "Report from the Executor, Sir —",
    "Second opinion from Codex, Sir —",
)
PEN_SLOTS = (
    "main `",
    "docs ",
    "commits behind",
    "other branches",
    "data ",
    "days old",
    "nearest deadline:",
)
PATH_EXTENSIONS = (
    ".py",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".sh",
    ".ps1",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".csv",
)
DATA_TREES = ("data", "data_raw", "data_processed", "data_archive")


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _section_block(text: str, first: str, stop: str | None) -> list[str]:
    """Lines from the heading starting ``first`` up to (not including) ``stop``."""
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith(first)), None)
    if start is None:
        return []
    end = len(lines)
    if stop:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(stop)), end)
    block = [line.rstrip() for line in lines[start:end]]
    while block and not block[-1]:
        block.pop()
    return block


def check_appendix(root: Path) -> list[str]:
    try:
        claude, agents = _read(root, "CLAUDE.md"), _read(root, "AGENTS.md")
    except OSError as e:
        return [f"AGENTS.md / CLAUDE.md: {e}"]
    want = _section_block(claude, "## 1. ", "## 7. ")
    if not want:
        return ["CLAUDE.md: no '## 1. ' … '## 7. ' checklist block found"]
    idx = agents.find("\n## Appendix")
    if idx < 0:
        return ["AGENTS.md: no '## Appendix' heading — copy CLAUDE.md §1–§6 under it"]
    got = _section_block(agents[idx:], "## 1. ", None)
    if got == want:
        return []
    for n, (a, b) in enumerate(zip(want, got, strict=False), start=1):
        if a != b:
            return [
                "AGENTS.md Appendix differs from CLAUDE.md §1–§6 at checklist line "
                f"{n}:\n    CLAUDE.md: {a!r}\n    AGENTS.md: {b!r}\n"
                "  fix: copy CLAUDE.md §1–§6 into the AGENTS.md Appendix, word for word"
            ]
    return [
        f"AGENTS.md Appendix has {len(got)} lines, CLAUDE.md §1–§6 has {len(want)}"
        " — copy CLAUDE.md §1–§6 into the AGENTS.md Appendix, word for word"
    ]


def check_main_hash(root: Path) -> list[str]:
    try:
        state = _read(root, "PROJECT_STATE.md")
    except OSError as e:
        return [f"PROJECT_STATE.md: {e}"]
    m = re.search(r"`?main`? is at `([0-9a-f]{7,40})`", state)
    if not m:
        return [
            "PROJECT_STATE.md records no main commit — add the Branches line "
            '"`main` is at `<hash>` (<date>)" (session-open measures drift from it)'
        ]
    shallow = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
    )
    if shallow.returncode == 0 and shallow.stdout.strip() == "false":
        found = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{m.group(1)}^{{commit}}"],
            capture_output=True,
        )
        if found.returncode != 0:
            return [f"PROJECT_STATE.md records main at {m.group(1)}, which is not a commit here"]
    return []


def check_deadlines(root: Path) -> list[str]:
    try:
        text = _read(root, "docs/deadlines.md")
    except OSError:
        return ["docs/deadlines.md is missing — it feeds the pen's mark (nearest deadline)"]
    lines = text.split("\n")
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(
                r"^\|\s*Due\s*\|\s*What\s*\|\s*Owner\s*\|\s*Status\s*\|\s*Source\s*\|", line
            )
        ),
        None,
    )
    if header is None:
        return [
            "docs/deadlines.md: the table header must be | Due | What | Owner | Status | Source |"
        ]
    failures = []
    for line in lines[header + 1 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set(cells[0]) <= set("-: "):
            continue
        due = re.sub(r"[*`]", "", cells[0]).strip()
        ok_due = due.lower().startswith(("conditional", "undated"))
        m = re.match(r"\d{4}-\d{2}-\d{2}", due)
        if m:
            try:  # a real calendar date: 2026-02-30 would crash session-open
                dt.date.fromisoformat(m.group(0))
                ok_due = True
            except ValueError:
                ok_due = False
        if len(cells) != 5 or not ok_due or not cells[3].strip():
            failures.append(f"docs/deadlines.md: malformed row: {line.strip()[:100]}")
    return failures


def check_frontier(root: Path) -> list[str]:
    try:
        manifest = json.loads(_read(root, "data/DATA_MANIFEST.json"))
    except (OSError, ValueError) as e:
        return [f"data/DATA_MANIFEST.json: {e}"]
    spec = importlib.util.spec_from_file_location(
        "data_manifest", root / "scripts" / "data_manifest.py"
    )
    if spec is None or spec.loader is None:
        return ["scripts/data_manifest.py: cannot load"]
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    frontier = manifest.get("frontier", {})
    failures = []
    for name, _rel in dm.FRONTIER_FILES:
        last = frontier.get(name, {}).get("last_date", "")
        try:
            dt.date.fromisoformat(last)
        except ValueError:
            failures.append(
                f"data/DATA_MANIFEST.json: no frontier date for '{name}' — rebuild the manifest "
                "with the data present (python scripts/data_manifest.py build --root <root>)"
            )
    return failures


def check_marks(root: Path) -> list[str]:
    try:
        claude, script = _read(root, "CLAUDE.md"), _read(root, "scripts/session_open.py")
    except OSError as e:
        return [f"marks: {e}"]
    failures = [
        f"mark {p!r} missing from {where}"
        for p in MARK_PREFIXES
        for where, text in (("CLAUDE.md", claude), ("scripts/session_open.py", script))
        if p not in text
    ]
    pen_line = next((line for line in claude.split("\n") if MARK_PREFIXES[0] in line), "")
    failures += [
        f"CLAUDE.md: the pen's mark lacks the slot {s!r}" for s in PEN_SLOTS if s not in pen_line
    ]
    return failures


def _cited_paths(text: str) -> list[str]:
    """Backticked repository paths: files with a known extension, or directories ending in '/'.

    A bare name without a '/' counts only when it is a root Markdown document
    (``CLAUDE.md``); other bare file names are prose, not paths.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced examples may name hypothetical files
    found = []
    for token in re.findall(r"`([^`\s]+)`", text):
        token = token.split("::", 1)[0].rstrip(".,;:")
        if any(ch in token for ch in "<>*{}$%\\|()") or token.startswith(("http", "~", "-", "/")):
            continue
        if token.startswith(".") and "/" not in token:
            continue
        is_file = token.endswith(PATH_EXTENSIONS)
        is_dir = token.endswith("/") and len(token) > 1
        if not (is_file or is_dir):
            continue
        if "/" not in token.rstrip("/") and not token.endswith(".md"):
            continue
        found.append(token)
    return found


def _is_data_path(rel: str) -> bool:
    top = rel.split("/", 1)[0]
    if top not in DATA_TREES:
        return False
    if rel.endswith("/"):
        return True
    return not rel.endswith((".py", ".md", ".gitkeep")) and not rel.endswith("DATA_MANIFEST.json")


def check_cited_paths(root: Path, files: tuple[str, ...] = RULE_BOOKS) -> list[str]:
    failures = []
    for rel in files:
        try:
            text = _read(root, rel)
        except OSError:
            continue
        for token in sorted(set(_cited_paths(text))):
            if _is_data_path(token):
                continue
            if not (root / token).exists():
                failures.append(f"{rel}: cites `{token}`, which does not exist")
    return failures


CHECKS = (
    ("AGENTS.md Appendix = CLAUDE.md §1–§6", check_appendix),
    ("PROJECT_STATE.md records the main commit", check_main_hash),
    ("docs/deadlines.md is well formed", check_deadlines),
    ("the manifest records the data frontier", check_frontier),
    ("the marks match scripts/session_open.py", check_marks),
    ("cited paths exist", check_cited_paths),
)


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]).resolve() if argv else REPO
    failed = 0
    for label, fn in CHECKS:
        problems = fn(root)
        print(f"{'ok  ' if not problems else 'FAIL'} {label}")
        for p in problems:
            print(f"     {p}")
        failed += bool(problems)
    print(f"working structure: {len(CHECKS) - failed} of {len(CHECKS)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
