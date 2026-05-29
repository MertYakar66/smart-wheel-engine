#!/usr/bin/env python3
"""Generate docs/TESTED_SURFACE_MAP.md from coverage.json.

The TESTED_SURFACE_MAP answers "what is and isn't covered" in one place.
It is generated from `coverage.json` (machine-readable, produced by
pytest-cov's --cov-report=json) rather than hand-written so that the
numbers track the live suite — re-run after a meaningful coverage shift
and commit the regenerated doc.

Per-module row columns:
  - module path
  - statements (total executable)
  - % covered (line+branch summary from coverage.py)
  - notable untested functions (functions whose definitions are fully or
    mostly missed, via AST cross-reference with coverage's missing_lines)
  - test files that cover it (heuristic: tests/ files whose `import` /
    `from X import` mentions the module's dotted path; coverage contexts
    are not enabled so this is a static-import approximation, not a
    runtime trace)

Plus a Top-N coverage gaps section ranked by uncovered statements
(missed × statement-count), which is the direct "where is the next
test investment best spent" answer.

Run:  python scripts/generate_tested_surface_map.py
       [--coverage coverage.json] [--out docs/TESTED_SURFACE_MAP.md]
       [--top 15]

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE = REPO_ROOT / "coverage.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "TESTED_SURFACE_MAP.md"
TESTS_DIR = REPO_ROOT / "tests"


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _module_dotted(rel_path: str) -> str:
    """Map a tracked file path to its importable dotted name."""
    p = _norm(rel_path)
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith(".py"):
        p = p[: -len(".py")]
    return p.replace("/", ".")


def _scan_test_imports() -> dict[str, set[str]]:
    """tests/test_*.py -> {dotted modules it imports}.

    Static import scan, not runtime trace. Covers `import X` and
    `from X import Y` (recording `X`). Conditional / late imports
    inside functions are included (ast.walk is exhaustive).
    """
    out: dict[str, set[str]] = {}
    if not TESTS_DIR.exists():
        return out
    for tpath in TESTS_DIR.rglob("test_*.py"):
        rel = _norm(str(tpath.relative_to(REPO_ROOT)))
        try:
            tree = ast.parse(tpath.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        out[rel] = imports
    return out


def _module_to_tests(
    files: dict[str, dict],
    test_imports: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Build module-path -> [test files] mapping via import grep."""
    out: dict[str, list[str]] = defaultdict(list)
    # Cache dotted -> file for fast lookup
    dotted_to_file: dict[str, str] = {}
    for fpath in files:
        dotted_to_file[_module_dotted(fpath)] = fpath
    for tpath, imps in test_imports.items():
        matched: set[str] = set()
        for imp in imps:
            # exact match
            if imp in dotted_to_file:
                matched.add(dotted_to_file[imp])
                continue
            # prefix match (e.g. `from engine.wheel_runner import X` ->
            # `engine.wheel_runner` is the imp itself; `import engine`
            # would only match the package __init__)
            parts = imp.split(".")
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in dotted_to_file:
                    matched.add(dotted_to_file[candidate])
                    break
        for fpath in matched:
            out[fpath].append(tpath)
    # stable order
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def _untested_functions(
    file_path: Path,
    missing_lines: set[int],
) -> list[tuple[str, int, int, int]]:
    """Return (name, def_line, missed_count, total_lines) for functions
    with material untested surface. Methods are namespaced as
    `Class.method`. Sorted by missed_count desc.

    A function qualifies when it satisfies BOTH of:

      - body is at least 3 lines (filter out trivial properties), and
      - either ≥33% of body lines are missed *or* ≥10 raw missed lines.

    The 33% / 10-line floor is intentionally looser than "function is
    essentially untested" so that top-gap modules (`wheel_runner.py`,
    `wheel_tracker.py`) surface specific partial-coverage hotspots
    instead of collapsing to an unhelpful "—".
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):
        return []
    out: list[tuple[str, int, int, int]] = []

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, prefix=f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                start = child.lineno
                end = getattr(child, "end_lineno", start)
                fn_lines = set(range(start, end + 1))
                missed = fn_lines & missing_lines
                total = len(fn_lines)
                if (
                    total >= 3
                    and len(missed) >= 3
                    and (len(missed) >= 10 or len(missed) * 3 >= total)
                ):
                    out.append((name, start, len(missed), total))
                walk(child, prefix=prefix)
            else:
                walk(child, prefix=prefix)

    walk(tree)
    out.sort(key=lambda r: (-r[2], r[1]))
    return out


def _fmt_pct(p: float) -> str:
    return f"{p:.1f}%"


def _fmt_untested(rows: list[tuple[str, int, int, int]], cap: int = 3) -> str:
    if not rows:
        return "—"
    parts = [f"`{name}` (L{ln}, {mis}/{tot})" for name, ln, mis, tot in rows[:cap]]
    if len(rows) > cap:
        parts.append(f"… +{len(rows) - cap} more")
    return "; ".join(parts)


def _fmt_tests(tests: list[str], cap: int = 4) -> str:
    if not tests:
        return "—"
    parts = [f"`{t}`" for t in tests[:cap]]
    if len(tests) > cap:
        parts.append(f"… +{len(tests) - cap} more")
    return ", ".join(parts)


def generate(coverage_path: Path, out_path: Path, top_n: int) -> None:
    cov = json.loads(coverage_path.read_text(encoding="utf-8"))
    files: dict[str, dict] = cov["files"]
    totals = cov["totals"]
    meta = cov.get("meta", {})

    test_imports = _scan_test_imports()
    mod_to_tests = _module_to_tests(files, test_imports)

    rows: list[dict] = []
    for fpath, data in sorted(files.items()):
        summary = data["summary"]
        missing_lines = set(data.get("missing_lines", []))
        untested_fns = _untested_functions(REPO_ROOT / fpath, missing_lines)
        rows.append(
            {
                "path": _norm(fpath),
                "stmts": summary["num_statements"],
                "covered": summary["covered_lines"],
                "missed_stmts": summary.get("missing_lines", 0),
                "pct": summary["percent_covered"],
                "untested": untested_fns,
                "tests": mod_to_tests.get(fpath, []),
            }
        )

    # Top-N gaps: ranked by raw uncovered statements (the count that
    # most directly answers "how much surface is dark").
    gaps = sorted(
        [r for r in rows if r["missed_stmts"] > 0],
        key=lambda r: (-r["missed_stmts"], r["pct"]),
    )[:top_n]

    now = datetime.now(UTC).strftime("%Y-%m-%d")
    cov_ts = meta.get("timestamp", "—")

    out_lines: list[str] = []
    out_lines.append("# Tested-surface map")
    out_lines.append("")
    out_lines.append(
        f"_Generated {now} from `coverage.json` (suite timestamp `{cov_ts}`) "
        "by `scripts/generate_tested_surface_map.py`. Regenerate after a "
        "meaningful coverage shift._"
    )
    out_lines.append("")
    out_lines.append(
        "This file answers _what is and isn't covered by the test suite_ at a "
        "module granularity. The numbers come from coverage.py's branch-aware "
        "report; the module → test mapping is a static import grep of "
        "`tests/test_*.py` (not a runtime trace), so a test file is listed if "
        "it imports the module — not necessarily if it exercises every line."
    )
    out_lines.append("")
    out_lines.append("**CI scope** (per `pyproject.toml [tool.coverage.run]`):")
    out_lines.append("`src` · `engine` · `advisors` · `financial_news` · `data`.")
    out_lines.append(
        "Modules listed in `[tool.coverage.run] omit` (research-tier ETL, "
        "Ollama-dependent memo generator, UI, etc.) are excluded by design "
        "— see `DECISIONS.md` D10 for the rationale on the 80% floor."
    )
    out_lines.append("")
    out_lines.append("## Suite totals")
    out_lines.append("")
    out_lines.append("| Metric | Value |")
    out_lines.append("|---|---|")
    out_lines.append(f"| Total statements (CI scope) | {totals['num_statements']:,} |")
    out_lines.append(f"| Covered statements | {totals['covered_lines']:,} |")
    out_lines.append(f"| Missing statements | {totals.get('missing_lines', 0):,} |")
    out_lines.append(f"| Excluded statements | {totals.get('excluded_lines', 0):,} |")
    if "num_branches" in totals:
        out_lines.append(f"| Total branches | {totals['num_branches']:,} |")
        out_lines.append(f"| Covered branches | {totals.get('covered_branches', 0):,} |")
        out_lines.append(f"| Partial branches | {totals.get('num_partial_branches', 0):,} |")
        out_lines.append(f"| Missing branches | {totals.get('missing_branches', 0):,} |")
    out_lines.append(f"| **Suite % covered** | **{_fmt_pct(totals['percent_covered'])}** |")
    out_lines.append(f"| Files in scope | {len(files):,} |")
    out_lines.append("")

    # Top gaps section first — this is the direct answer.
    out_lines.append(f"## Top {top_n} coverage gaps")
    out_lines.append("")
    out_lines.append(
        "Ranked by **uncovered statements** (raw count). These are where "
        "additional tests would buy the most coverage; review the "
        "untested-function column before adding a test to confirm the gap "
        "is on a path that warrants exercise rather than an `omit`-candidate "
        "research module."
    )
    out_lines.append("")
    out_lines.append("| Rank | Module | Stmts | Missed | % | Notable untested |")
    out_lines.append("|---:|---|---:|---:|---:|---|")
    for i, r in enumerate(gaps, 1):
        out_lines.append(
            f"| {i} | `{r['path']}` | {r['stmts']:,} | {r['missed_stmts']:,} | "
            f"{_fmt_pct(r['pct'])} | {_fmt_untested(r['untested'], cap=3)} |"
        )
    out_lines.append("")

    # Full per-module table — grouped by top-level package for readability.
    out_lines.append("## Per-module coverage")
    out_lines.append("")
    out_lines.append(
        "One row per CI-scope file. The **Tests** column lists `tests/` "
        "files that statically import the module; this is a coverage proxy "
        "(static import) not a runtime exercise trace."
    )
    out_lines.append("")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        top = r["path"].split("/", 1)[0]
        grouped[top].append(r)
    for top in sorted(grouped):
        out_lines.append(f"### `{top}/`")
        out_lines.append("")
        out_lines.append("| Module | Stmts | % | Notable untested | Tests |")
        out_lines.append("|---|---:|---:|---|---|")
        for r in sorted(grouped[top], key=lambda x: x["path"]):
            out_lines.append(
                f"| `{r['path']}` | {r['stmts']:,} | {_fmt_pct(r['pct'])} | "
                f"{_fmt_untested(r['untested'], cap=2)} | "
                f"{_fmt_tests(r['tests'], cap=3)} |"
            )
        out_lines.append("")

    # Methodology notes.
    out_lines.append("## Methodology notes")
    out_lines.append("")
    out_lines.append(
        "- **Numbers** come from `coverage.json` (`pytest-cov "
        "--cov-report=json`), which itself reflects "
        "`pyproject.toml [tool.coverage.run]` — `source`, `omit`, "
        "`branch = true`, and `[tool.coverage.report] exclude_lines`."
    )
    out_lines.append(
        '- **"Notable untested"** functions are functions with ≥3 line body '
        "where either ≥33% of body lines or ≥10 raw lines intersect "
        "`missing_lines`. Methods are namespaced as `Class.method`. "
        "Truncated to the top 2–3 per row by missed count. A `—` means no "
        "single function clears the threshold; the row's gap is fragmented "
        "across many small misses (still surfaced via the row's `Stmts` / "
        "% columns)."
    )
    out_lines.append(
        '- **"Tests"** column is built from `tests/test_*.py` AST '
        "imports (`import X` / `from X import Y`). A test file appears if "
        "it imports the module — it does not assert exercise. Enabling "
        '`coverage.dynamic_context = "test_function"` would give a true '
        "runtime mapping; the trade-off is run time (~1.5–2× slower) and "
        "a much larger `.coverage` SQLite file. Out of scope for this "
        "first artifact."
    )
    out_lines.append(
        "- **`__init__.py`** files appear when they hold re-exports or "
        "logic; pure namespace inits show 100% with a small statement "
        "count."
    )
    out_lines.append(
        "- **Branches** are counted but not enumerated per row — the "
        "branch totals at the top of the file give the suite-level "
        "branch picture; per-line branch detail lives in "
        "`coverage.json[files][...].missing_branches` for anyone wanting "
        "a deeper dive."
    )
    out_lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(
        f"wrote {out_path.relative_to(REPO_ROOT)}: "
        f"{len(rows)} modules, top-{top_n} gaps, "
        f"suite {_fmt_pct(totals['percent_covered'])}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--coverage",
        default=str(DEFAULT_COVERAGE),
        help=f"Path to coverage.json (default: {DEFAULT_COVERAGE.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output markdown path (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many gaps to surface in the Top-N section (default: 15)",
    )
    args = ap.parse_args()
    generate(Path(args.coverage), Path(args.out), args.top)


if __name__ == "__main__":
    main()
