#!/usr/bin/env python3
"""Drive consolidation: find what only Drive holds and copy it home — never overwrite, never delete (D33).

D33 makes the desktop data root the main copy and one Drive folder, ``swe-data/``,
the complete second copy. Before that folder exists, project data sits in older
Drive areas that overlap (``docs/DATA_INVENTORY.md`` §C.3). This tool proves,
object by object, what those areas hold that the root lacks, and copies it home::

    python scripts/drive_consolidate.py census    --remote gdrive: --out census.json
    python scripts/drive_consolidate.py inventory --root C:\\swe-data --out root.json
    python scripts/drive_consolidate.py plan      --census census.json --inventory root.json --out plan.json
    python scripts/drive_consolidate.py bytecheck --plan plan.json --inventory root.json --root C:\\swe-data --remote gdrive:
    python scripts/drive_consolidate.py copy      --plan plan.json --root C:\\swe-data --remote gdrive:
    python scripts/drive_consolidate.py verify    --plan plan.json --root C:\\swe-data
    python scripts/drive_consolidate.py sums      --root C:\\swe-data
    python scripts/drive_consolidate.py filters   > exclude.txt   # rclone --exclude-from, with --ignore-case

``census`` lists every object in each area through raw Drive queries by parent id
(``rclone backend query``), so a shortcut is recorded as a shortcut and never
followed, and a Google-format file, a duplicate name or a second parent stays
visible. Each area's root folder is verified by its parent and name first.

``plan`` gives every Drive object one class, written to the ledger
(``plan.json`` and ``ledger.csv``):

    folder            a folder; nothing to do
    redundant         same size, MD5 and SHA-256 as a root file (``twin``)
    duplicate         same size, MD5 and SHA-256 as another Drive object that is copied
    copy              only Drive holds these bytes: copied to ``dest`` under
                      data_archive/drive-legacy/<area>/…, never over an existing file
    needs-byte-check  Drive lists no SHA-256 (or no MD5) and the bytes may exist
                      elsewhere; ``bytecheck`` downloads and settles it
    unresolved        left where it is and listed: a shortcut, a Google-format
                      file, a credential-shaped name, several parents

Only bytes count (D33). A git object in the old ``.git`` upload is not
byte-identical to the bundle that holds the same logical object, so it comes home
like any other file.

A missing hash never counts as a match. An empty file is never matched by content:
it comes home at its own path. ``dest`` keeps the Drive path; a component Windows
cannot hold becomes ``_drive-<id>``, and a destination that is taken (any case) goes
to ``<area>/_conflicts/<id>/<name>``.

``copy`` copies by Drive id (``rclone backend copyid``). It never overwrites: a
destination that already holds the right bytes counts as done, so a rerun resumes,
and one holding other bytes stops the run before anything is copied. Every batch
is re-hashed. ``verify`` re-hashes every copy.

Nothing here deletes or moves a Drive object or a root file. The only files the
tool removes or replaces are its own: the temporary downloads of ``bytecheck``
(in a folder outside the root) and its own output files, written atomically.
Deletion is a separate step with the Operator's yes (D33, card 3).

Exit codes: 0 ok · 1 verify found a difference · 2 configuration error ·
3 a destination holds other bytes · 4 a copy failed (rerun resumes) · 5 not enough
free space. Standard library only; the Drive side is rclone ≥ 1.65 (Drive SHA-256).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = 1
LEGACY = "data_archive/drive-legacy"
FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
GOOGLE = "application/vnd.google-apps."
SUMS = "SHA256SUMS"
LOGS = "_logs"
MIN_RCLONE = (1, 65)  # Drive SHA-1/SHA-256 support (rclone changelog, v1.65.0)
QUERY_BATCH = 50  # parent ids per Drive query, as rclone groups them itself
RETRY_SLEEP = 2.0  # seconds, times the attempt number, between query retries
MAX_DEST = 400  # characters under the root; beyond this the object stays on Drive

RCLONE: list[str] = ["rclone"]  # the command; tests substitute a fake

# The Drive areas of docs/DATA_INVENTORY.md §C.3 (D33). ``name`` is the folder under
# data_archive/drive-legacy/; ``parent`` and ``folder`` verify the id before a census;
# ``read-only`` areas belong to another project: copied from, never cleaned.
DEFAULT_AREAS: tuple[dict, ...] = (
    {
        "name": "swe-local-only",
        "id": "1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn",
        "parent": "root",
        "folder": "swe-local-only",
        "mode": "consolidate",
    },
    {
        "name": "SmartWheelData",
        "id": "1wCFPBf0o9PJMy2f2vy34S316XFc1Sq3e",
        "parent": "1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4",
        "folder": "SmartWheelData",
        "mode": "consolidate",
    },
    {
        "name": "smart-wheel-engine-git",
        "id": "1dA_fq1MorvsqUWeVxqR0aAaJ9XMwEjUY",
        "parent": "root",
        "folder": "smart-wheel-engine",
        "mode": "consolidate",
    },
    {
        "name": "day-bot-local-archive/vendor_swe_data",
        "id": "1EewDv70haKPVzjmhTDvo27lLsMjlqyzY",
        "parent": "1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0",
        "folder": "vendor_swe_data",
        "mode": "read-only",
    },
    {
        "name": "day-bot-local-archive/vendor_swe_data_raw",
        "id": "1_u85pi25w-H5HynRH3-WvGv1tMd8MW71",
        "parent": "1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0",
        "folder": "vendor_swe_data_raw",
        "mode": "read-only",
    },
    {
        "name": "day-bot-local-archive/vendor_swe_data_processed",
        "id": "1WXeonbDMTT_VsGDizQD32Rw0V14xLxHE",
        "parent": "1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0",
        "folder": "vendor_swe_data_processed",
        "mode": "read-only",
    },
    {
        "name": "day-bot-local-archive/data_raw",
        "id": "1uHSbrEaZoyW_Tgn1BimSz02KOJ18e616",
        "parent": "1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0",
        "folder": "data_raw",
        "mode": "read-only",
    },
)

# A path with any component matching one of these (case-insensitive) is never read,
# copied, mirrored or listed in SHA256SUMS: it stays where it is (D33 exclusions).
CREDENTIAL_PATTERNS: tuple[str, ...] = (
    "*credential*",
    "*creds*",
    "*secret*",
    "*token*",
    "*password*",
    "*passwd*",
    "*apikey*",
    "*api_key*",
    "*service_account*",
    "*private_key*",
    "*oauth*",
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
)

LEDGER_FIELDS = (
    "area",
    "mode",
    "id",
    "parent",
    "path",
    "kind",
    "mime",
    "size",
    "md5",
    "sha256",
    "created",
    "modified",
    "class",
    "reason",
    "twin",
    "dest",
    "path_unique",
    "deletable",
)

_WIN_BAD = set('<>:"/\\|?*')
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)
}


class ToolError(Exception):
    """A configuration or safety failure; carries the exit code."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- small helpers


def credential_shaped(rel: str) -> bool:
    parts = [p.lower() for p in rel.replace("\\", "/").split("/") if p]
    return any(fnmatch.fnmatchcase(part, pat) for part in parts for pat in CREDENTIAL_PATTERNS)


def win_safe(name: str) -> bool:
    """Can Windows hold a file or folder with exactly this name?"""
    if not name or name in (".", "..") or name[-1] in " .":
        return False
    if any(c in _WIN_BAD or ord(c) < 32 for c in name):
        return False
    return name.split(".")[0].upper() not in _WIN_RESERVED


def native(path: Path | str) -> str:
    """An absolute path usable for file access, extended-length on Windows."""
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s
    return s


def hash_file(path: Path | str) -> tuple[int, str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with open(native(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            md5.update(block)
            sha.update(block)
            size += len(block)
    return size, md5.hexdigest(), sha.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, data) -> None:
    """Write our own output file atomically (temp file, then replace)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_root(arg: str | None) -> Path:
    raw = (arg or os.environ.get("SWE_DATA_ROOT", "")).strip()
    if not raw:
        raise ToolError("no data root: pass --root or set SWE_DATA_ROOT")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ToolError(f"data root {root} is not a directory")
    return root


def same_path(a: str, b: str) -> bool:
    na, nb = os.path.normcase(os.path.abspath(a)), os.path.normcase(os.path.abspath(b))
    return na == nb


# ---------------------------------------------------------------- rclone


def run_cmd(cmd: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def rclone_version() -> tuple[int, int, int]:
    cp = run_cmd([*RCLONE, "version"])
    m = re.search(r"rclone v(\d+)\.(\d+)\.(\d+)", cp.stdout or "")
    if cp.returncode != 0 or not m:
        raise ToolError(f"cannot run rclone ({' '.join(RCLONE)}): {cp.stderr.strip()[:300]}")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def require_rclone() -> str:
    v = rclone_version()
    if v[:2] < MIN_RCLONE:
        raise ToolError(
            f"rclone v{'.'.join(map(str, v))} is too old: Drive SHA-256 needs v{MIN_RCLONE[0]}.{MIN_RCLONE[1]}"
        )
    return ".".join(map(str, v))


def q_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def drive_query(remote: str, query: str, tries: int = 3) -> list[dict]:
    """Raw Drive objects matching a query; retried, and INCOMPLETE is an error."""
    last = ""
    for attempt in range(tries):
        cp = run_cmd([*RCLONE, "backend", "query", remote, query])
        if cp.returncode == 0 and "INCOMPLETE" not in (cp.stderr or ""):
            out = json.loads(cp.stdout or "null")
            return out or []
        last = (cp.stderr or "").strip()[-400:]
        time.sleep(RETRY_SLEEP * (attempt + 1))
    raise ToolError(f"Drive query failed after {tries} tries: {query[:120]}…\n{last}")


def slim(obj: dict) -> dict:
    sc = obj.get("shortcutDetails") or {}
    size = obj.get("size")
    return {
        "id": obj["id"],
        "name": obj.get("name", ""),
        "mime": obj.get("mimeType", ""),
        "size": int(size) if size not in (None, "") else None,
        "md5": (obj.get("md5Checksum") or "").lower() or None,
        "sha256": (obj.get("sha256Checksum") or "").lower() or None,
        "parents": list(obj.get("parents") or []),
        "created": obj.get("createdTime"),
        "modified": obj.get("modifiedTime"),
        "target": sc.get("targetId"),
        "target_mime": sc.get("targetMimeType"),
    }


# ---------------------------------------------------------------- census


def census(remote: str, areas: list[dict]) -> dict:
    out = []
    for area in areas:
        where = (
            "'root' in parents"
            if area["parent"] == "root"
            else f"{q_quote(area['parent'])} in parents"
        )
        hits = drive_query(
            remote, f"{where} and name = {q_quote(area['folder'])} and trashed = false"
        )
        ids = [h["id"] for h in hits if h.get("mimeType") == FOLDER]
        if area["id"] not in ids:
            raise ToolError(
                f"area {area['name']}: no folder {area['folder']!r} with id {area['id']} under "
                f"{area['parent']} (found {ids})"
            )
        seen, objects, queue = {area["id"]}, [], [area["id"]]
        while queue:
            batch, queue = queue[:QUERY_BATCH], queue[QUERY_BATCH:]
            parents = " or ".join(f"{q_quote(i)} in parents" for i in batch)
            for obj in drive_query(remote, f"({parents}) and trashed = false"):
                if obj["id"] in seen:
                    continue
                seen.add(obj["id"])
                objects.append(slim(obj))
                if obj.get("mimeType") == FOLDER:
                    queue.append(obj["id"])
        objects.sort(key=lambda o: o["id"])
        out.append({**area, "objects": objects})
        print(f"census: {area['name']}: {len(objects)} objects", file=sys.stderr)
    swe = drive_query(remote, "'root' in parents and name = 'swe-data' and trashed = false")
    return {
        "schema": SCHEMA,
        "kind": "census",
        "generated": utc_now(),
        "remote": remote,
        "areas": out,
        "swe_data": [slim(o) for o in swe],
    }


# ---------------------------------------------------------------- inventory


def _walk(root: Path):
    """(relative posix path, native path) of every file under the root but ``_logs/``."""
    base = native(root)
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = os.path.relpath(dirpath, base).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        if not rel_dir:
            dirnames[:] = [d for d in dirnames if d != LOGS]
        dirnames.sort()
        yield rel_dir, dirnames, sorted(filenames), dirpath


def inventory(root: Path) -> dict:
    files, dirs, links = [], [], []
    count = 0
    for rel_dir, dirnames, filenames, dirpath in _walk(root):
        dirs.extend(f"{rel_dir}/{d}" if rel_dir else d for d in dirnames)
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                links.append(rel)
                continue
            if credential_shaped(rel):
                files.append({"path": rel, "size": os.stat(full).st_size, "excluded": True})
                continue
            size, md5, sha = hash_file(full)
            files.append({"path": rel, "size": size, "md5": md5, "sha256": sha, "excluded": False})
            count += 1
            if count % 2000 == 0:
                print(f"inventory: {count} files hashed", file=sys.stderr)
    return {
        "schema": SCHEMA,
        "kind": "inventory",
        "generated": utc_now(),
        "root": str(root),
        "files": files,
        "dirs": dirs,
        "links": links,
    }


# ---------------------------------------------------------------- plan


def _area_tree(area: dict) -> tuple[dict, dict, dict]:
    """Per object id: its path in the area, its chain of ids, and whether its path is unambiguous."""
    objs = {o["id"]: o for o in area["objects"]}
    top = area["id"]
    parent_of = {}
    for o in objs.values():
        inside = [p for p in o["parents"] if p == top or p in objs]
        if not inside:
            raise ToolError(f"area {area['name']}: object {o['id']} has no parent inside the area")
        parent_of[o["id"]] = inside[0]
    siblings = Counter((parent_of[i], objs[i]["name"]) for i in objs)
    paths, chains, unique = {}, {}, {}

    def resolve(oid: str, depth: int = 0):
        if oid in paths:
            return
        if depth > 200:
            raise ToolError(f"area {area['name']}: folder chain deeper than 200 at {oid}")
        par = parent_of[oid]
        name = objs[oid]["name"]
        own = siblings[(par, name)] == 1
        if par == top:
            paths[oid], chains[oid], unique[oid] = name, [oid], own
        else:
            resolve(par, depth + 1)
            paths[oid] = f"{paths[par]}/{name}"
            chains[oid] = [*chains[par], oid]
            unique[oid] = own and unique[par]

    for oid in objs:
        resolve(oid)
    return paths, chains, unique


def build_plan(census_doc: dict, inv: dict) -> dict:
    root_files = inv["files"]
    usable = [f for f in root_files if not f.get("excluded")]
    by_full: dict[tuple, list[str]] = defaultdict(list)
    md5_count: Counter = Counter()
    for f in usable:
        if f["size"]:
            by_full[(f["size"], f["md5"], f["sha256"])].append(f["path"])
            md5_count[(f["size"], f["md5"])] += 1
    for paths in by_full.values():
        paths.sort()
    empty_root = {f["path"].casefold() for f in root_files if f["size"] == 0}
    taken_files = {f["path"].casefold() for f in root_files}
    taken_dirs = {d.casefold() for d in inv.get("dirs", [])}
    for f in root_files:
        parts = f["path"].casefold().split("/")
        taken_dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))

    trees = {a["id"]: _area_tree(a) for a in census_doc["areas"]}
    for a in census_doc["areas"]:
        for o in a["objects"]:
            if o["mime"] not in (FOLDER, SHORTCUT) and not o["mime"].startswith(GOOGLE):
                if o["size"] and o["md5"]:
                    md5_count[(o["size"], o["md5"])] += 1

    rows, first_copy = [], {}
    for a in census_doc["areas"]:
        paths, chains, unique = trees[a["id"]]
        names = {o["id"]: o["name"] for o in a["objects"]}
        for o in sorted(a["objects"], key=lambda o: (paths[o["id"]], o["id"])):
            path = paths[o["id"]]
            kind = (
                "folder"
                if o["mime"] == FOLDER
                else "shortcut"
                if o["mime"] == SHORTCUT
                else "google"
                if o["mime"].startswith(GOOGLE)
                else "file"
            )
            row = {
                "area": a["name"],
                "mode": a["mode"],
                "id": o["id"],
                "parent": o["parents"][0] if o["parents"] else "",
                "path": path,
                "kind": kind,
                "mime": o["mime"],
                "size": o["size"],
                "md5": o["md5"],
                "sha256": o["sha256"],
                "created": o["created"],
                "modified": o["modified"],
                "class": "",
                "reason": "",
                "twin": "",
                "dest": "",
                "path_unique": unique[o["id"]],
            }
            comps = [
                n if win_safe(n) else f"_drive-{cid}"
                for n, cid in ((names[c], c) for c in chains[o["id"]])
            ]
            natural = f"{LEGACY}/{a['name']}/{'/'.join(comps)}"
            cls, reason, twin = _classify(
                o, path, by_full, md5_count, first_copy, natural, empty_root
            )
            row.update({"class": cls, "reason": reason, "twin": twin})
            if cls in ("copy", "needs-byte-check"):
                if len(natural) > MAX_DEST:
                    row.update(
                        {"class": "unresolved", "reason": "destination path too long", "twin": ""}
                    )
                else:
                    row["dest"] = _assign_dest(
                        natural, a["name"], o["id"], comps[-1], taken_files, taken_dirs
                    )
                    if cls == "copy" and o["size"] and o["md5"] and o["sha256"]:
                        first_copy.setdefault((o["size"], o["md5"], o["sha256"]), row["dest"])
            rows.append(row)
    _mark_deletable(rows)
    return {
        "schema": SCHEMA,
        "kind": "plan",
        "generated": utc_now(),
        "root": inv["root"],
        "census_generated": census_doc["generated"],
        "inventory_generated": inv["generated"],
        "areas": [
            {"name": a["name"], "id": a["id"], "mode": a["mode"]} for a in census_doc["areas"]
        ],
        "swe_data": census_doc.get("swe_data", []),
        "root_bytes": sum(f["size"] for f in usable),
        "root_files": len(usable),
        "excluded": [f["path"] for f in root_files if f.get("excluded")],
        "rows": rows,
    }


def _classify(o, path, by_full, md5_count, first_copy, natural, empty_root):
    if o["mime"] == FOLDER:
        return "folder", "", ""
    if o["mime"] == SHORTCUT:
        return (
            "unresolved",
            f"shortcut to {o['target'] or '?'} ({o['target_mime'] or '?'}); never followed",
            "",
        )
    if o["mime"].startswith(GOOGLE):
        return "unresolved", "Google-format file: no bytes to compare", ""
    if credential_shaped(path):
        return "unresolved", "credential-shaped name: never read or copied", ""
    if len(o["parents"]) > 1:
        return "unresolved", "several parents", ""
    if o["size"] is None:
        return "needs-byte-check", "Drive lists no size", ""
    if o["size"] == 0:
        if natural.casefold() in empty_root:
            return "redundant", "empty file already at its destination", natural
        return "copy", "empty file: kept at its own path", ""
    if not o["md5"]:
        return "needs-byte-check", "Drive lists no MD5", ""
    if o["sha256"]:
        key = (o["size"], o["md5"], o["sha256"])
        if key in by_full:
            return "redundant", "", by_full[key][0]
        if key in first_copy:
            return "duplicate", "same bytes as another Drive object that is copied", first_copy[key]
        return "copy", "", ""
    if md5_count[(o["size"], o["md5"])] > 1:
        return "needs-byte-check", "no SHA-256 on Drive, and its MD5 is not unique", ""
    return "copy", "no SHA-256 on Drive; its MD5 matches nothing else", ""


def _assign_dest(natural, area_name, oid, leaf, taken_files, taken_dirs) -> str:
    def free(p: str) -> bool:
        cf = p.casefold()
        if cf in taken_files or cf in taken_dirs:
            return False
        parts = cf.split("/")
        return not any("/".join(parts[:i]) in taken_files for i in range(1, len(parts)))

    for cand in (natural, f"{LEGACY}/{area_name}/_conflicts/{oid}/{leaf}"):
        if free(cand):
            cf = cand.casefold()
            taken_files.add(cf)
            parts = cf.split("/")
            taken_dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))
            return cand
    raise ToolError(f"no free destination for Drive object {oid} ({natural})")


def _mark_deletable(rows: list[dict]) -> None:
    """A forecast for card 3: which objects may go once their bytes are proven at home."""
    for r in rows:
        r["deletable"] = (
            r["mode"] == "consolidate"
            and bool(r["path_unique"])
            and r["class"] in ("redundant", "duplicate", "copy")
        )


def write_ledger(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in LEDGER_FIELDS})
    os.replace(tmp, path)


def summarize(plan: dict, about: dict | None = None, limit: int = 40) -> list[str]:
    lines = []
    for a in plan["areas"]:
        rows = [r for r in plan["rows"] if r["area"] == a["name"]]
        cls = Counter(r["class"] for r in rows)
        size = defaultdict(int)
        for r in rows:
            size[r["class"]] += r["size"] or 0
        parts = ", ".join(f"{c} {n} ({size[c]:,} B)" for c, n in sorted(cls.items()))
        lines.append(f"{a['name']} [{a['mode']}]: {len(rows)} objects: {parts}")
    copy = [r for r in plan["rows"] if r["class"] == "copy"]
    copy_bytes = sum(r["size"] or 0 for r in copy)
    lines.append(f"to copy home: {len(copy)} files, {copy_bytes:,} B")
    need = plan["root_bytes"] + copy_bytes
    lines.append(f"swe-data will hold about {plan['root_files'] + len(copy)} files, {need:,} B")
    if about:
        free = about.get("free")
        lines.append(f"Drive free: {free:,} B" if isinstance(free, int) else "Drive free: unknown")
        if isinstance(free, int):
            lines.append(f"Drive margin after swe-data: {free - need:,} B")
    if plan.get("swe_data"):
        lines.append(
            f"WARNING: a top-level swe-data already exists on Drive ({len(plan['swe_data'])})"
        )
    if plan.get("excluded"):
        lines.append(f"root files excluded as credential-shaped: {len(plan['excluded'])}")
        lines.extend(f"  excluded  {p}" for p in plan["excluded"][:limit])
    todo = [r for r in plan["rows"] if r["class"] in ("unresolved", "needs-byte-check")]
    lines.append(f"left to settle or listed: {len(todo)}")
    lines.extend(f"  {r['class']:<17} {r['area']}/{r['path']}  {r['reason']}" for r in todo[:limit])
    if len(todo) > limit:
        lines.append(f"  … {len(todo) - limit} more in the ledger")
    return lines


# ---------------------------------------------------------------- bytecheck


def _outside(tmp: Path, root: Path) -> bool:
    t, r = os.path.normcase(str(tmp.resolve())), os.path.normcase(str(root.resolve()))
    return not (t == r or t.startswith(r.rstrip("\\/") + os.sep))


def bytecheck(plan: dict, inv: dict, root: Path, remote: str, tmp: Path) -> dict:
    if inv.get("generated") != plan.get("inventory_generated"):
        raise ToolError("this inventory is not the one the plan was made from")
    if not _outside(tmp, root):
        raise ToolError(f"the temporary folder {tmp} must be outside the root {root}")
    tmp.mkdir(parents=True, exist_ok=True)
    rows = plan["rows"]
    by_full: dict[tuple, str] = {}
    for f in sorted(inv["files"], key=lambda f: f["path"]):
        if not f.get("excluded") and f["size"]:
            by_full.setdefault((f["size"], f["md5"], f["sha256"]), f["path"])
    copy_keys = {}
    for r in rows:
        if r["class"] == "copy" and r["size"] and r["md5"] and r["sha256"]:
            copy_keys.setdefault((r["size"], r["md5"], r["sha256"]), r["dest"])
    made: list[Path] = []

    def fetch(r: dict) -> Path:
        dest = tmp / r["id"]
        if dest.exists():
            raise ToolError(f"temporary file {dest} already exists; use an empty --tmp")
        cp = run_cmd([*RCLONE, "backend", "copyid", remote, r["id"], str(dest)])
        if cp.returncode != 0 or not dest.exists():
            raise ToolError(
                f"download of {r['area']}/{r['path']} failed: {cp.stderr.strip()[-300:]}", 4
            )
        made.append(dest)
        return dest

    try:
        for r in sorted(
            (r for r in rows if r["class"] == "needs-byte-check"),
            key=lambda r: (r["area"], r["path"]),
        ):
            local = fetch(r)
            size, md5, sha = hash_file(local)
            if (r["size"] is not None and size != r["size"]) or (r["md5"] and md5 != r["md5"]):
                r.update(
                    {
                        "class": "unresolved",
                        "reason": "download differs from Drive's size or MD5",
                        "dest": "",
                    }
                )
            else:
                r.update({"size": size, "md5": md5, "sha256": r["sha256"] or sha})
                key = (size, md5, sha)
                if size == 0:
                    r.update({"class": "copy", "reason": "empty file: kept at its own path"})
                elif key in by_full:
                    r.update(
                        {
                            "class": "redundant",
                            "reason": "SHA-256 computed from a download",
                            "twin": by_full[key],
                            "dest": "",
                        }
                    )
                elif key in copy_keys:
                    r.update(
                        {
                            "class": "duplicate",
                            "reason": "SHA-256 computed from a download",
                            "twin": copy_keys[key],
                            "dest": "",
                        }
                    )
                else:
                    r.update({"class": "copy", "reason": "SHA-256 computed from a download"})
                    copy_keys[key] = r["dest"]
            os.remove(native(local))  # our own temporary download
            made.remove(local)
    finally:
        for p in made:
            if os.path.exists(native(p)):
                os.remove(native(p))  # our own temporary download
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()  # the empty temporary folder
    _mark_deletable(rows)
    plan["bytechecked"] = utc_now()
    return plan


# ---------------------------------------------------------------- copy and verify


def _matches(path: Path, r: dict) -> tuple[bool, str]:
    size, md5, sha = hash_file(path)
    if size != (r["size"] or 0):
        return False, f"size {size} != {r['size']}"
    if r["md5"] and md5 != r["md5"]:
        return False, "MD5 differs"
    if r["sha256"] and sha != r["sha256"]:
        return False, "SHA-256 differs"
    return True, sha


def copy_all(
    plan: dict,
    root: Path,
    remote: str,
    batch: int,
    jobs: int,
    reserve: int,
    dry_run: bool,
    log: Path,
) -> int:
    if not same_path(plan["root"], str(root)):
        raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
    rows = [r for r in plan["rows"] if r["class"] == "copy"]
    for r in rows:
        parts = r["dest"].split("/")
        if (
            not r["dest"].startswith(LEGACY + "/")
            or ".." in parts
            or "" in parts
            or ":" in r["dest"]
        ):
            raise ToolError(f"refusing destination {r['dest']!r}: copies go only under {LEGACY}/")
    todo, done, conflicts = [], [], []
    for r in rows:
        dest = root / r["dest"]
        if os.path.lexists(native(dest)):
            ok, why = _matches(dest, r) if os.path.isfile(native(dest)) else (False, "not a file")
            (done if ok else conflicts).append((r, why))
        else:
            todo.append(r)
    print(
        f"copy: {len(rows)} to copy home; {len(done)} already there with the right bytes; {len(todo)} to fetch"
    )
    if conflicts:
        for r, why in conflicts:
            print(f"  CONFLICT  {r['dest']}: {why} (Drive {r['area']}/{r['path']})")
        raise ToolError(f"{len(conflicts)} destination(s) hold other bytes; nothing was copied", 3)
    need = sum(r["size"] or 0 for r in todo)
    free = shutil.disk_usage(native(root)).free
    print(f"copy: {need:,} B to download; {free:,} B free; reserve {reserve:,} B")
    if free - need < reserve:
        raise ToolError("not enough free space for the download and the reserve", 5)
    if dry_run or not todo:
        return 0
    require_rclone()
    chunks = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    results: list[dict] = []

    def run(chunk: list[dict]) -> list[dict]:
        out, send = [], []
        for r in chunk:
            if os.path.lexists(native(root / r["dest"])):
                out.append(
                    {"id": r["id"], "dest": r["dest"], "status": "conflict: appeared before copy"}
                )
            else:
                send.append(r)
        if send:
            args = [*RCLONE, "backend", "copyid", remote]
            for r in send:
                args += [r["id"], str(root / r["dest"])]
            cp = run_cmd(args)
            for r in send:
                dest = root / r["dest"]
                if not os.path.isfile(native(dest)):
                    out.append(
                        {
                            "id": r["id"],
                            "dest": r["dest"],
                            "status": f"missing after copy (rclone exit {cp.returncode})",
                        }
                    )
                    continue
                ok, why = _matches(dest, r)
                out.append(
                    {
                        "id": r["id"],
                        "dest": r["dest"],
                        "status": "ok" if ok else f"MISMATCH: {why}",
                        "sha256": why if ok else "",
                    }
                )
        return out

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for i, part in enumerate(ex.map(run, chunks), 1):
            results.extend(part)
            if i % 10 == 0 or i == len(chunks):
                print(f"copy: {i}/{len(chunks)} batches", file=sys.stderr)
    with open(log, "a", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps({**rec, "at": utc_now()}) + "\n")
    bad = [x for x in results if x["status"] != "ok"]
    print(f"copy: {len(results) - len(bad)} copied and verified; {len(bad)} failed")
    for x in bad[:50]:
        print(f"  FAILED  {x['dest']}: {x['status']}")
    if any(x["status"].startswith(("MISMATCH", "conflict")) for x in bad):
        raise ToolError(
            "a copy does not match Drive, or a destination appeared; nothing was overwritten", 3
        )
    return 4 if bad else 0


def verify(plan: dict, root: Path) -> int:
    rows = [r for r in plan["rows"] if r["class"] == "copy"]
    ok = missing = bad = 0
    for r in rows:
        dest = root / r["dest"]
        if not os.path.isfile(native(dest)):
            missing += 1
            print(f"  MISSING   {r['dest']}")
            continue
        good, why = _matches(dest, r)
        if good:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH  {r['dest']}: {why}")
    partial = []
    legacy = root / LEGACY
    if legacy.is_dir():
        for rel_dir, _d, filenames, _p in _walk(legacy):
            partial.extend(
                f"{rel_dir}/{f}" if rel_dir else f for f in filenames if f.endswith(".partial")
            )
    print(f"verify: {len(rows)} copies: {ok} ok, {missing} missing, {bad} mismatched")
    if partial:
        print(
            f"verify: {len(partial)} leftover .partial file(s) from an interrupted copy (not data):"
        )
        for p in partial[:20]:
            print(f"  partial   {LEGACY}/{p}")
    return 0 if missing == bad == 0 else 1


# ---------------------------------------------------------------- sums and filters


def write_sums(root: Path) -> tuple[int, int, str]:
    lines, total = [], 0
    for rel_dir, _d, filenames, dirpath in _walk(root):
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            if rel in (SUMS, SUMS + ".tmp") or os.path.islink(full) or credential_shaped(rel):
                continue
            size, _md5, sha = hash_file(full)
            total += size
            lines.append(f"{sha}  {rel}")
    lines.sort(key=lambda s: s[66:])
    body = ("\n".join(lines) + "\n") if lines else ""
    out = root / SUMS
    tmp = root / (SUMS + ".tmp")
    with open(native(tmp), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    os.replace(native(tmp), native(out))  # our own list, regenerated
    return len(lines), total, hashlib.sha256(body.encode("utf-8")).hexdigest()


def filters_text() -> str:
    lines = [
        "# rclone --exclude-from list for the data root (D33); use with --ignore-case",
        f"/{LOGS}/**",
    ]
    for pat in CREDENTIAL_PATTERNS:
        lines += [pat, f"{pat}/**"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- command line


def _areas(arg: str | None) -> list[dict]:
    return list(load_json(Path(arg))) if arg else [dict(a) for a in DEFAULT_AREAS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("census", help="list every object in the Drive areas")
    sp.add_argument("--remote", required=True, help="rclone remote, e.g. gdrive:")
    sp.add_argument("--areas", help="areas JSON (default: the §C.3 areas built in)")
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("inventory", help="size, MD5 and SHA-256 of every root file")
    sp.add_argument("--root")
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("plan", help="classify every Drive object; write the ledger")
    sp.add_argument("--census", required=True)
    sp.add_argument("--inventory", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--ledger", help="CSV ledger (default: next to --out)")
    sp.add_argument("--about", help="the output of `rclone about <remote> --json`")
    sp.add_argument("--limit", type=int, default=40)
    sp = sub.add_parser("bytecheck", help="settle objects Drive lists without a SHA-256")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--inventory", required=True, help="the root inventory the plan was made from")
    sp.add_argument("--root")
    sp.add_argument("--remote", required=True)
    sp.add_argument("--tmp", help="an empty temporary folder outside the root")
    sp = sub.add_parser("copy", help="copy Drive-only objects home by id; never overwrite")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--root")
    sp.add_argument("--remote", required=True)
    sp.add_argument("--batch", type=int, default=20)
    sp.add_argument("--jobs", type=int, default=4)
    sp.add_argument("--reserve-gb", type=float, default=10.0)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--log", help="JSON-lines log (default: next to --plan)")
    sp = sub.add_parser("verify", help="re-hash every copy against the plan")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--root")
    sp = sub.add_parser("sums", help="write SHA256SUMS for the root")
    sp.add_argument("--root")
    sub.add_parser("filters", help="print the rclone exclusion list")
    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except ToolError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "filters":
        sys.stdout.write(filters_text())
        return 0
    if args.cmd == "census":
        version = require_rclone()
        doc = census(args.remote, _areas(args.areas))
        doc["rclone"] = version
        write_json(Path(args.out), doc)
        n = sum(len(a["objects"]) for a in doc["areas"])
        print(
            f"census: {len(doc['areas'])} areas, {n} objects; swe-data exists: {bool(doc['swe_data'])}"
        )
        return 0
    if args.cmd == "inventory":
        root = resolve_root(args.root)
        doc = inventory(root)
        write_json(Path(args.out), doc)
        kept = [f for f in doc["files"] if not f["excluded"]]
        print(
            f"inventory: {len(doc['files'])} files ({len(kept)} hashed, {len(doc['files']) - len(kept)} "
            f"credential-shaped, not read), {sum(f['size'] for f in kept):,} B; {len(doc['links'])} links skipped"
        )
        return 0
    if args.cmd == "plan":
        inv = load_json(Path(args.inventory))
        plan = build_plan(load_json(Path(args.census)), inv)
        _save_plan(plan, Path(args.out), Path(args.ledger) if args.ledger else None)
        about = load_json(Path(args.about)) if args.about else None
        print("\n".join(summarize(plan, about, args.limit)))
        return 0
    plan_path = Path(args.plan)
    plan = load_json(plan_path)
    if plan.get("kind") != "plan" or plan.get("schema") != SCHEMA:
        raise ToolError(f"{plan_path} is not a plan of schema {SCHEMA}")
    root = resolve_root(args.root)
    if args.cmd == "bytecheck":
        if not same_path(plan["root"], str(root)):
            raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
        require_rclone()
        tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="swe-bytecheck-"))
        if tmp.exists() and any(tmp.iterdir()):
            raise ToolError(f"the temporary folder {tmp} is not empty")
        inv = load_json(Path(args.inventory))
        bytecheck(plan, inv, root, args.remote, tmp)
        _save_plan(plan, plan_path, None)
        print("\n".join(summarize(plan)))
        return 0
    if args.cmd == "copy":
        log = Path(args.log) if args.log else plan_path.with_name(plan_path.stem + "_copy.jsonl")
        return copy_all(
            plan,
            root,
            args.remote,
            max(1, args.batch),
            args.jobs,
            int(args.reserve_gb * 1e9),
            args.dry_run,
            log,
        )
    if args.cmd == "verify":
        return verify(plan, root)
    if args.cmd == "sums":
        n, total, digest = write_sums(root)
        print(f"sums: {SUMS} lists {n} files, {total:,} B; its own sha256 {digest}")
        return 0
    raise ToolError(f"unknown command {args.cmd}")


def _save_plan(plan: dict, out: Path, ledger: Path | None) -> None:
    write_json(out, plan)
    write_ledger(ledger or out.with_name(out.stem + "_ledger.csv"), plan["rows"])


if __name__ == "__main__":
    sys.exit(main())
