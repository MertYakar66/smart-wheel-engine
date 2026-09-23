#!/usr/bin/env python3
"""The data manifest — the contract between the desktop data root, the backup and git.

``data/DATA_MANIFEST.json`` lists every dataset the engine owns: repository-
relative path, byte size and sha256. It was first generated from the git
objects themselves (2026-09-18: the tracked ``data/`` tree on ``main``, the 13
deep-history slices on ``deep-history/bloomberg-raw`` and the 15 day-bot tick
files on ``claude/daybot-bloomberg-pull``), so it is the exact checklist a
data root must satisfy before anything is removed from git (DECISIONS.md D31).

Usage (from the repository root, or anywhere with ``--manifest``)::

    python scripts/data_manifest.py check  --root D:\\smart-wheel-data   # 0 missing / 0 mismatched → exit 0
    python scripts/data_manifest.py census --root D:\\smart-wheel-data   # what is present, by dataset group
    python scripts/data_manifest.py build  --root D:\\smart-wheel-data --out data/DATA_MANIFEST.json
    python scripts/data_manifest.py check  --root .  --group deep          # only one dataset group
    python scripts/data_manifest.py materialize --root D:\\smart-wheel-data  # copy git-held files into the root

``materialize`` fills a root from the git objects the manifest names
(``git_sources``: branch → commit). It only ever *creates* files that are
missing from the root — an existing file is never overwritten, a mismatching one
is reported — and it verifies every byte it writes against the manifest sha256.
Run it BEFORE any data branch is deleted (``git fetch origin <branch>`` first).

``data_archive/`` holds bytes kept for the record, not read by the engine: every
distinct data file found at the tip of a branch other than ``main`` that no
other manifest row already carries, under ``data_archive/<branch>/<path>``. Such
a row names the path the bytes had in git as ``git_path`` (default: ``path``),
which is where ``materialize`` reads them from. With these rows present, a root
that passes ``check`` holds every data file any branch tip ever carried, so a
branch can be deleted without losing a dataset (DECISIONS.md D31).

``--root`` defaults to ``SWE_DATA_ROOT`` when set, else the repository root.
``check`` exits 1 on any missing or mismatched file, 0 otherwise; ``--extra``
also lists data files present under the root that the manifest does not know.
Pure standard library; no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "data" / "DATA_MANIFEST.json"
SCHEMA = 2

# Dataset groups: a file belongs to the first group whose prefix matches.
GROUPS: tuple[tuple[str, str], ...] = (
    ("deep", "data/bloomberg/deep/"),
    ("broad_pull", "data/bloomberg/broad_pull/"),
    ("bloomberg", "data/bloomberg/"),
    ("features", "data/features/"),
    ("ticks", "data_raw/bloomberg/ticks/"),
    ("raw", "data_raw/"),
    ("processed", "data_processed/"),
    ("archive", "data_archive/"),
    ("data", "data/"),
)
# What ``build`` walks under a root (relative). Code files under data/ are skipped.
WALK_DIRS: tuple[str, ...] = ("data", "data_raw", "data_processed", "data_archive")
SKIP_SUFFIXES: tuple[str, ...] = (".py", ".md", ".pyc", ".gitkeep", ".log", ".lock")
SKIP_NAMES: tuple[str, ...] = (
    "DATA_MANIFEST.json",
    "__pycache__",
    "_locks",
    "_inventory_scan.json",
)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def group_of(rel: str) -> str:
    for name, prefix in GROUPS:
        if rel.startswith(prefix):
            return name
    return "other"


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    if m.get("schema") != SCHEMA:
        raise SystemExit(f"manifest schema {m.get('schema')!r} != {SCHEMA} ({path})")
    return m


def resolve_root(arg: str | None) -> Path:
    raw = arg or os.environ.get("SWE_DATA_ROOT", "").strip()
    return (Path(raw).expanduser() if raw else REPO).resolve()


def iter_root_files(root: Path):
    for top in WALK_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_NAMES]
            for fn in sorted(filenames):
                if fn in SKIP_NAMES or fn.endswith(SKIP_SUFFIXES):
                    continue
                p = Path(dirpath) / fn
                yield p.relative_to(root).as_posix(), p


def cmd_build(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    out = Path(args.out) if args.out else DEFAULT_MANIFEST
    previous = _load_previous(out)
    prev_files = {f["path"]: f for f in previous.get("files", [])}
    files = []
    for rel, p in iter_root_files(root):
        row = {
            "path": rel,
            "size": p.stat().st_size,
            "sha256": sha256_of(p),
            "group": group_of(rel),
        }
        before = prev_files.get(rel)
        if before and before.get("sha256") == row["sha256"] and before.get("git_source"):
            row["git_source"] = before["git_source"]  # ledger: still the bytes git once held
            if before.get("git_path"):
                row["git_path"] = before["git_path"]
        files.append(row)
    files.sort(key=lambda r: r["path"])
    manifest = {
        "schema": SCHEMA,
        "generated": date.today().isoformat(),
        "root_used": str(root),
        "note": (
            "Every dataset the engine owns: path relative to the data root, byte size, sha256. "
            "The data root is the operator's desktop (SWE_DATA_ROOT); git holds no data (D31). "
            "Verify a root with: python scripts/data_manifest.py check --root <root>."
        ),
    }
    for key in ("git_sources", "drive"):  # ledger metadata survives a rebuild
        if key in previous:
            manifest[key] = previous[key]
    manifest["counts"] = _counts(files)
    manifest["files"] = files
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out}: {len(files)} files, {sum(f['size'] for f in files) / 1e6:.1f} MB")
    return 0


def _counts(files: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for f in files:
        g = out.setdefault(f["group"], {"files": 0, "bytes": 0})
        g["files"] += 1
        g["bytes"] += f["size"]
    return dict(sorted(out.items()))


def _select(manifest: dict, group: str | None) -> list[dict]:
    files = manifest["files"]
    return [f for f in files if f.get("group") == group] if group else files


def cmd_check(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    manifest = load_manifest(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    files = _select(manifest, args.group)
    missing, mismatched, ok = [], [], 0
    for f in files:
        p = root / f["path"]
        if not p.is_file():
            missing.append(f["path"])
            continue
        size = p.stat().st_size
        if size != f["size"]:
            mismatched.append(f"{f['path']} (size {size} != {f['size']})")
            continue
        if not args.size_only:
            digest = sha256_of(p)
            if digest != f["sha256"]:
                mismatched.append(f"{f['path']} (sha256 differs)")
                continue
        ok += 1
    print(f"root: {root}")
    print(
        f"checked {len(files)} manifest files{' in group ' + args.group if args.group else ''}: "
        f"{ok} ok, {len(missing)} missing, {len(mismatched)} mismatched"
        f"{' (size only)' if args.size_only else ''}"
    )
    for m in missing:
        print(f"  MISSING    {m}")
    for m in mismatched:
        print(f"  MISMATCH   {m}")
    if args.extra:
        known = {f["path"] for f in manifest["files"]}
        extra = [rel for rel, _ in iter_root_files(root) if rel not in known]
        print(f"extra data files under the root not in the manifest: {len(extra)}")
        for e in extra[: args.limit]:
            print(f"  EXTRA      {e}")
    return 1 if (missing or mismatched) else 0


def _load_previous(path: Path) -> dict:
    """The manifest already at ``path`` (for ledger carry-over), or ``{}``."""
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return {}
    return m if isinstance(m, dict) and m.get("schema") == SCHEMA else {}


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=check, text=False
    )


def _git_has_object(repo: Path, spec: str) -> bool:
    return _git(repo, "cat-file", "-e", spec, check=False).returncode == 0


def _git_write_blob(repo: Path, spec: str, dest: Path) -> str:
    """Stream ``git cat-file blob <spec>`` into ``dest`` (via a temp file); return sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".materializing")
    h = hashlib.sha256()
    with open(tmp, "wb") as fh:
        proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "blob", spec], stdout=subprocess.PIPE
        )
        assert proc.stdout is not None
        for block in iter(lambda: proc.stdout.read(1 << 20), b""):
            h.update(block)
            fh.write(block)
        if proc.wait() != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"git cat-file blob {spec} failed")
    os.replace(tmp, dest)
    return h.hexdigest()


def cmd_materialize(args: argparse.Namespace) -> int:
    """Create every manifest file missing from the root from the git objects named
    in ``git_sources`` and verify it. Existing files are never overwritten."""
    root = resolve_root(args.root)
    repo = Path(args.repo).resolve() if args.repo else REPO
    manifest = load_manifest(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    sources: dict[str, str] = manifest.get("git_sources") or {}
    files = _select(manifest, args.group)
    present, written, mismatched, unavailable, failed = 0, 0, [], [], []
    print(f"root: {root}\nrepo: {repo}")
    if not sources:
        print("manifest has no git_sources: nothing to materialize from git")
    needed = sorted({f.get("git_source") for f in files if f.get("git_source")} - set(sources))
    for src in needed:
        unavailable.append(f"{src}: not in the manifest's git_sources")
    for f in files:
        dest = root / f["path"]
        if dest.is_file():
            if dest.stat().st_size == f["size"] and (
                args.size_only or sha256_of(dest) == f["sha256"]
            ):
                present += 1
            else:
                mismatched.append(f["path"])
            continue
        src = f.get("git_source")
        commit = sources.get(src or "")
        if not commit:
            unavailable.append(f"{f['path']}: no git source")
            continue
        spec = f"{commit}:{f.get('git_path') or f['path']}"
        if not _git_has_object(repo, spec):
            branch = src.split(":", 1)[1] if src and ":" in src else src
            unavailable.append(
                f"{f['path']}: object {commit[:9]} absent — git fetch origin {branch}"
            )
            continue
        if args.dry_run:
            print(f"  WOULD WRITE {f['path']} ({f['size']} bytes from {src})")
            written += 1
            continue
        try:
            digest = _git_write_blob(repo, spec, dest)
        except (OSError, RuntimeError) as exc:
            failed.append(f"{f['path']}: {exc}")
            continue
        if digest != f["sha256"] or dest.stat().st_size != f["size"]:
            dest.unlink(missing_ok=True)
            failed.append(f"{f['path']}: bytes from git do not match the manifest (removed)")
            continue
        written += 1
        print(f"  WROTE      {f['path']} ({f['size']} bytes from {src}, sha256 ok)")
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"materialize {len(files)} manifest files: {present} already present, {verb} {written}, "
        f"{len(mismatched)} mismatched (kept, not overwritten), {len(unavailable)} unavailable, "
        f"{len(failed)} failed"
    )
    for m in mismatched:
        print(f"  MISMATCH   {m}")
    for u in unavailable:
        print(f"  UNAVAILABLE {u}")
    for x in failed:
        print(f"  FAILED     {x}")
    return 1 if (mismatched or unavailable or failed) else 0


def cmd_census(args: argparse.Namespace) -> int:
    root = resolve_root(args.root)
    manifest = load_manifest(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    by_group: dict[str, list[dict]] = {}
    for f in manifest["files"]:
        by_group.setdefault(f["group"], []).append(f)
    print(f"root: {root}")
    print(f"{'group':<12}{'manifest':>10}{'present':>10}{'size-ok':>10}{'MB':>10}")
    total_missing = 0
    for g, fs in sorted(by_group.items()):
        present = [f for f in fs if (root / f["path"]).is_file()]
        size_ok = [f for f in present if (root / f["path"]).stat().st_size == f["size"]]
        total_missing += len(fs) - len(present)
        print(
            f"{g:<12}{len(fs):>10}{len(present):>10}{len(size_ok):>10}{sum(f['size'] for f in fs) / 1e6:>10.1f}"
        )
    print(f"missing overall: {total_missing}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    commands = (
        ("build", cmd_build),
        ("check", cmd_check),
        ("census", cmd_census),
        ("materialize", cmd_materialize),
    )
    for name, fn in commands:
        sp = sub.add_parser(name)
        sp.add_argument("--root", help="data root (default: $SWE_DATA_ROOT or the repo root)")
        sp.add_argument("--manifest", help=f"manifest path (default: {DEFAULT_MANIFEST})")
        if name == "build":
            sp.add_argument(
                "--out", help="where to write the manifest (default: data/DATA_MANIFEST.json)"
            )
        if name in ("check", "materialize"):
            sp.add_argument(
                "--group",
                help="only this dataset group (deep, broad_pull, bloomberg, features, ticks, raw, processed, archive, data)",
            )
            sp.add_argument(
                "--size-only",
                action="store_true",
                help="treat an existing file with the right size as present (no hashing)",
            )
        if name == "materialize":
            sp.add_argument(
                "--repo", help="git repository holding the objects (default: this repository)"
            )
            sp.add_argument(
                "--dry-run", action="store_true", help="report what would be written; write nothing"
            )
        if name == "check":
            sp.add_argument(
                "--extra",
                action="store_true",
                help="also list data files under the root that the manifest does not know",
            )
            sp.add_argument("--limit", type=int, default=50, help="max extra files to print")
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
