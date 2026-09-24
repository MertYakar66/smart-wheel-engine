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
    python scripts/drive_consolidate.py sweep     --source D:\\old --inventory root.json --dest data_archive/old --root C:\\swe-data
    python scripts/drive_consolidate.py sums      --root C:\\swe-data
    python scripts/drive_consolidate.py filters   > exclude.txt   # rclone --exclude-from, with --ignore-case

``census`` lists every object in each area through raw Drive queries by parent id
(``rclone backend query``), so a shortcut is recorded as a shortcut and never
followed, and a Google-format file, a duplicate name or a second parent stays
visible. Each area's root folder is verified by its parent and name first. Drive
sometimes answers an OR of several parents with nothing (Google issue 149522397;
rclone works around it the same way), so an empty answer is asked again one parent
at a time. Last, each area's file count and bytes must equal ``rclone size``, which
walks folder by folder: any difference stops the run.

``plan`` gives every Drive object one class, written to the ledger
(``plan.json`` and ``plan_ledger.csv``):

    folder            a folder; nothing to do
    redundant         same size, MD5 and SHA-256 as a root file (``twin``); an
                      empty file only when an empty file already sits at its own
                      destination
    duplicate         same size, MD5 and SHA-256 as another Drive object that is copied
    copy              only Drive holds these bytes: copied to ``dest`` under
                      data_archive/drive-legacy/<area>/…, never over an existing file
    needs-byte-check  Drive lists no size, MD5 or SHA-256 for it; ``bytecheck``
                      downloads it and settles it (``copy`` refuses to run before)
    unresolved        left where it is and listed: a shortcut, a Google-format
                      file, a credential-shaped name, several parents, a
                      destination that cannot be placed

Only bytes count (D33): a missing hash never counts as a match, and a git object in
the old ``.git`` upload is not a copy of the bundle that holds the same logical
object. ``dest`` keeps the Drive path; a name Windows cannot hold, or that rclone
would rewrite, becomes ``_drive-<id>``; a destination already taken (any case) goes
to ``<area>/_conflicts/<id>/<name>``. A Drive id is ``deletable`` (a forecast for
card 3) only if every row that names it is.

``copy`` downloads by Drive id (``rclone backend copyid``) into a staging folder
next to the root (``<root>.d33-staging``, same volume, never inside the root),
re-hashes each file there, and only then publishes it with a hard link, which
fails if the destination exists: nothing is ever written over. A destination that
already holds the right bytes counts as done, so a rerun resumes; one holding other
bytes stops the run before anything is copied. A failed batch is retried one object
at a time, so one object that cannot be downloaded never blocks the others.
``sweep`` does the same for a local folder, by bytes, with an exclusive create.

Nothing here deletes or moves a Drive object or a root file. The only files the
tool removes or replaces are its own: temporary downloads outside the root, the
staging names of files once published, and its own output files, which never go
inside the root except under ``_logs/``. Deletion is a separate step with the
Operator's yes (D33, card 3).

Exit codes: 0 ok · 1 verify found a difference · 2 configuration or safety error ·
3 a destination holds other bytes, or a download does not match Drive · 4 a copy
failed (a rerun resumes) · 5 not enough free space. Standard library only; the
Drive side is rclone ≥ 1.65 (Drive SHA-256).
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
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = 2
LEGACY = "data_archive/drive-legacy"
FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
GOOGLE = "application/vnd.google-apps."
SUMS = "SHA256SUMS"
LOGS = "_logs"
LIVE_TREES = ("data", "data_raw", "data_processed")
EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
MIN_RCLONE = (1, 65)  # Drive SHA-1/SHA-256 support (rclone changelog, v1.65.0)
QUERY_BATCH = 50  # parent ids per Drive query, as rclone groups them itself
RETRY_SLEEP = 2.0  # seconds, times the attempt number, between query retries
MAX_DEST = 400  # characters under the root; beyond this the object stays on Drive
MAX_CMDLINE = 20000  # characters per rclone call; Windows allows 32,767
STAGING_SUFFIX = ".d33-staging"

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
    "*.ppk",
    "*.kdbx",
    "*.jks",
    "*.keystore",
    "rclone.conf",  # holds the Drive token
    ".netrc",
    "_netrc",
    ".pgpass",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
)
# Whole path tails treated the same way: a git config can carry a token in a remote URL.
CREDENTIAL_PATHS: tuple[str, ...] = (".git/config",)

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

AREA_NAME = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
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
    if any(fnmatch.fnmatchcase(part, pat) for part in parts for pat in CREDENTIAL_PATTERNS):
        return True
    joined = "/".join(parts)
    return any(joined == tail or joined.endswith("/" + tail) for tail in CREDENTIAL_PATHS)


def _rclone_rewrites(c: str) -> bool:
    """Characters rclone's local backend rewrites in a file name (its escape character,
    control pictures, full-width ASCII): such a file would not land where planned."""
    return c == "\u201b" or "\u2400" <= c <= "\u2421" or "\uff01" <= c <= "\uff5e"


def win_safe(name: str) -> bool:
    """Can Windows hold a file or folder with exactly this name, and rclone write it as is?"""
    if not name or name in (".", "..") or name[-1] in " .":
        return False
    if len(name.encode("utf-16-le")) // 2 > 255:
        return False
    if any(c in _WIN_BAD or ord(c) < 32 or ord(c) == 127 or _rclone_rewrites(c) for c in name):
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


def _inside(path: Path | str, folder: Path | str) -> bool:
    p = os.path.normcase(os.path.abspath(str(path)))
    f = os.path.normcase(os.path.abspath(str(folder)))
    return p == f or p.startswith(f.rstrip("\\/") + os.sep)


def guard_out(path: Path, root: Path | str | None) -> Path:
    """Our own output files never land inside the data root, except under its _logs/."""
    if root is None:
        env = os.environ.get("SWE_DATA_ROOT", "").strip()
        root = env or None
    if root is not None and _inside(path, root) and not _inside(path, Path(root) / LOGS):
        raise ToolError(
            f"refusing to write {path} inside the data root; use {LOGS}/ or a folder outside it"
        )
    return path


def write_outputs(pairs: list[tuple[Path, str]]) -> None:
    """Write our own output files: every temp file first, then replace them all."""
    temps = []
    for path, text in pairs:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        temps.append((tmp, path))
    for tmp, path in temps:
        os.replace(tmp, path)


def json_text(data) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def _check_areas(areas: list[dict]) -> None:
    seen = set()
    for a in areas:
        name = a.get("name", "")
        if not AREA_NAME.match(name) or ".." in name.split("/"):
            raise ToolError(
                f"area name {name!r} must be letters, digits, '.', '_', '-' and '/' only"
            )
        if name.casefold() in seen:
            raise ToolError(f"two areas share the name {name!r}")
        seen.add(name.casefold())
        if a.get("mode") not in ("consolidate", "read-only"):
            raise ToolError(f"area {name}: mode must be consolidate or read-only")


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
            f"rclone v{'.'.join(map(str, v))} is too old: Drive SHA-256 needs "
            f"v{MIN_RCLONE[0]}.{MIN_RCLONE[1]}"
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
            return json.loads(cp.stdout or "null") or []
        last = (cp.stderr or "").strip()[-400:]
        time.sleep(RETRY_SLEEP * (attempt + 1))
    raise ToolError(f"Drive query failed after {tries} tries: {query[:120]}…\n{last}")


def _children(remote: str, ids: list[str]) -> list[dict]:
    parents = " or ".join(f"{q_quote(i)} in parents" for i in ids)
    found = drive_query(remote, f"({parents}) and trashed = false")
    if found or len(ids) == 1:
        return found
    # Drive sometimes answers (A in parents) or (B in parents) with nothing
    # (issuetracker.google.com/issues/149522397); ask each parent alone, as rclone does.
    out: list[dict] = []
    for i in ids:
        out.extend(drive_query(remote, f"{q_quote(i)} in parents and trashed = false"))
    return out


def rclone_size(remote: str, folder_id: str) -> dict:
    """File count and bytes under a folder, from rclone's own folder-by-folder walk."""
    cp = run_cmd(
        [
            *RCLONE,
            "size",
            "--json",
            "--drive-root-folder-id",
            folder_id,
            "--drive-skip-gdocs",
            "--drive-skip-shortcuts",
            remote,
        ]
    )
    if cp.returncode != 0:
        raise ToolError(f"rclone size failed for {folder_id}: {cp.stderr.strip()[-300:]}")
    d = json.loads(cp.stdout)
    return {"count": int(d["count"]), "bytes": int(d["bytes"])}


def slim(obj: dict) -> dict:
    sc = obj.get("shortcutDetails") or {}
    size = obj.get("size")
    md5 = (obj.get("md5Checksum") or "").lower() or None
    size = int(size) if size not in (None, "") else None
    if size is None and md5 == EMPTY_MD5:
        size = 0  # rclone's JSON omits a zero size (``size,omitempty``)
    return {
        "id": obj["id"],
        "name": obj.get("name", ""),
        "mime": obj.get("mimeType", ""),
        "size": size,
        "md5": md5,
        "sha256": (obj.get("sha256Checksum") or "").lower() or None,
        "parents": list(obj.get("parents") or []),
        "created": obj.get("createdTime"),
        "modified": obj.get("modifiedTime"),
        "target": sc.get("targetId"),
        "target_mime": sc.get("targetMimeType"),
    }


def is_file(o: dict) -> bool:
    return o["mime"] not in (FOLDER, SHORTCUT) and not o["mime"].startswith(GOOGLE)


def census_totals(objects: list[dict], top: str) -> dict:
    """File count and bytes as rclone's walk sees them: once per parent folder."""
    folders = {top} | {o["id"] for o in objects if o["mime"] == FOLDER}
    count = size = 0
    for o in objects:
        if is_file(o):
            k = sum(1 for p in o["parents"] if p in folders)
            count += k
            size += k * (o["size"] or 0)
    return {"count": count, "bytes": size}


# ---------------------------------------------------------------- census


def census(remote: str, areas: list[dict]) -> dict:
    _check_areas(areas)
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
            for obj in _children(remote, batch):
                if obj["id"] in seen:
                    continue
                seen.add(obj["id"])
                objects.append(slim(obj))
                if obj.get("mimeType") == FOLDER:
                    queue.append(obj["id"])
        objects.sort(key=lambda o: o["id"])
        mine = census_totals(objects, area["id"])
        walk = rclone_size(remote, area["id"])
        if mine != walk:
            raise ToolError(
                f"area {area['name']}: the census found {mine['count']} files, {mine['bytes']:,} B, "
                f"but rclone size finds {walk['count']} files, {walk['bytes']:,} B. A listing is "
                "incomplete (or the area has duplicate folder names, which rclone's walk merges); "
                "nothing is planned from it"
            )
        out.append({**area, "objects": objects, "check": walk})
        print(
            f"census: {area['name']}: {len(objects)} objects; {walk['count']} files, "
            f"{walk['bytes']:,} B, equal to rclone size",
            file=sys.stderr,
        )
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
    """(relative dir, dirnames, filenames, native dir) under the root, without ``_logs/``."""
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


class Taken:
    """Paths under the root that a new file may not take (case-insensitive, as Windows)."""

    def __init__(self, files: list[str], dirs: list[str]):
        self.files: set[str] = set()
        self.dirs = {d.casefold() for d in dirs}
        for p in files:
            self.take(p)

    def free(self, p: str) -> bool:
        cf = p.casefold()
        if cf in self.files or cf in self.dirs:
            return False
        parts = cf.split("/")
        return not any("/".join(parts[:i]) in self.files for i in range(1, len(parts)))

    def take(self, p: str) -> None:
        cf = p.casefold()
        self.files.add(cf)
        parts = cf.split("/")
        self.dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))

    def assign(self, candidates: list[str]) -> str | None:
        for cand in candidates:
            if len(cand) <= MAX_DEST and self.free(cand):
                self.take(cand)
                return cand
        return None


def _conflict_path(area: str, oid: str, leaf: str) -> str:
    return f"{LEGACY}/{area}/_conflicts/{oid}/{leaf}"


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


def _empty_ok(o: dict) -> bool:
    """An empty object whose listed hashes are the empty file's (a missing SHA-256 is not)."""
    return o["md5"] == EMPTY_MD5 and o["sha256"] == EMPTY_SHA256


def build_plan(census_doc: dict, inv: dict) -> dict:
    _check_areas(census_doc["areas"])
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
    empty_root = {
        f["path"].casefold() for f in usable if f["size"] == 0 and f["sha256"] == EMPTY_SHA256
    }
    taken = Taken([f["path"] for f in root_files], inv.get("dirs", []))
    trees = {a["id"]: _area_tree(a) for a in census_doc["areas"]}
    for a in census_doc["areas"]:
        for o in a["objects"]:
            if is_file(o) and o["size"] and o["md5"]:
                md5_count[(o["size"], o["md5"])] += 1

    rows, first_copy, first_row = [], {}, {}
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
            comps = [
                n if win_safe(n) else f"_drive-{c}"
                for n, c in ((names[c], c) for c in chains[o["id"]])
            ]
            natural = f"{LEGACY}/{a['name']}/{'/'.join(comps)}"
            conflict = _conflict_path(a["name"], o["id"], comps[-1])
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
                "twin": "",
                "dest": "",
                "candidates": [natural, conflict],
                "path_unique": unique[o["id"]],
            }
            cls, reason, twin = _classify(
                o, path, by_full, md5_count, first_copy, row["candidates"], empty_root
            )
            row.update({"class": cls, "reason": reason, "twin": twin})
            prev = first_row.get(o["id"])
            if prev is not None and cls in ("copy", "duplicate", "redundant", "needs-byte-check"):
                # One Drive object reached through two areas (a folder with two
                # parents): it comes home once, through its first row.
                row.update(
                    {
                        "class": "duplicate",
                        "reason": f"the same Drive object as {prev['area']}/{prev['path']}",
                        "twin": prev["dest"] or prev["twin"],
                    }
                )
            elif cls == "copy":
                _place(row, taken)
                if row["class"] == "copy" and o["size"] and o["md5"] and o["sha256"]:
                    first_copy.setdefault((o["size"], o["md5"], o["sha256"]), row["dest"])
            first_row.setdefault(o["id"], row)
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


def _place(row: dict, taken: Taken) -> None:
    dest = taken.assign(row["candidates"])
    if dest is None:
        row.update(
            {"class": "unresolved", "reason": "no free destination under the root", "dest": ""}
        )
    else:
        row["dest"] = dest


def _classify(o, path, by_full, md5_count, first_copy, candidates, empty_root):
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
    if not o["md5"]:
        return "needs-byte-check", "Drive lists no MD5", ""
    if o["size"] == 0:
        if not _empty_ok(o):
            return "needs-byte-check", "empty, but Drive does not list the empty file's hashes", ""
        for cand in candidates:
            if cand.casefold() in empty_root:
                return "redundant", "empty file already at its destination", cand
        return "copy", "empty file: kept at its own path", ""
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


def _mark_deletable(rows: list[dict]) -> None:
    """A forecast for card 3: an id may go only if every row naming it may."""
    ok: dict[str, bool] = {}
    for r in rows:
        each = (
            r["mode"] == "consolidate"
            and bool(r["path_unique"])
            and r["class"] in ("redundant", "duplicate", "copy")
        )
        ok[r["id"]] = ok.get(r["id"], True) and each
    for r in rows:
        r["deletable"] = ok[r["id"]]


def ledger_text(rows: list[dict]) -> str:
    from io import StringIO

    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=LEDGER_FIELDS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in LEDGER_FIELDS})
    return buf.getvalue()


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


def bytecheck(plan: dict, inv: dict, root: Path, remote: str, tmp: Path) -> dict:
    if inv.get("generated") != plan.get("inventory_generated"):
        raise ToolError("this inventory is not the one the plan was made from")
    if _inside(tmp, root) or _inside(root, tmp):
        raise ToolError(f"the temporary folder {tmp} must be outside the root {root}")
    rows = plan["rows"]
    by_full: dict[tuple, str] = {}
    for f in sorted(inv["files"], key=lambda f: f["path"]):
        if not f.get("excluded") and f["size"]:
            by_full.setdefault((f["size"], f["md5"], f["sha256"]), f["path"])
    empty_root = {
        f["path"].casefold()
        for f in inv["files"]
        if not f.get("excluded") and f["size"] == 0 and f["sha256"] == EMPTY_SHA256
    }
    copy_keys = {}
    for r in rows:
        if r["class"] == "copy" and r["size"] and r["md5"] and r["sha256"]:
            copy_keys.setdefault((r["size"], r["md5"], r["sha256"]), r["dest"])
    taken = Taken(
        [f["path"] for f in inv["files"]] + [r["dest"] for r in rows if r["dest"]],
        inv.get("dirs", []),
    )
    todo = sorted(
        (r for r in rows if r["class"] == "needs-byte-check"), key=lambda r: (r["area"], r["path"])
    )
    tmp.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    try:
        for r in todo:
            local = tmp / r["id"]
            if os.path.lexists(native(local)):
                raise ToolError(f"temporary file {local} already exists; use an empty --tmp")
            cp = run_cmd([*RCLONE, "backend", "copyid", remote, r["id"], str(local)])
            if cp.returncode != 0 or not os.path.isfile(native(local)):
                raise ToolError(
                    f"download of {r['area']}/{r['path']} failed: {cp.stderr.strip()[-300:]}", 4
                )
            made.append(native(local))
            size, md5, sha = hash_file(local)
            os.remove(native(local))  # our own temporary download
            made.remove(native(local))
            listed = [
                ("size", r["size"], size),
                ("MD5", r["md5"], md5),
                ("SHA-256", r["sha256"], sha),
            ]
            differ = [name for name, want, got in listed if want is not None and want != got]
            if differ:
                r.update(
                    {
                        "class": "unresolved",
                        "reason": f"download differs from Drive's listed {', '.join(differ)}",
                        "dest": "",
                    }
                )
                continue
            r.update({"size": size, "md5": md5, "sha256": sha})
            key = (size, md5, sha)
            empty_twin = next((c for c in r["candidates"] if c.casefold() in empty_root), None)
            if size == 0 and empty_twin:
                r.update(
                    {
                        "class": "redundant",
                        "reason": "empty file already at its destination",
                        "twin": empty_twin,
                    }
                )
            elif size and key in by_full:
                r.update(
                    {
                        "class": "redundant",
                        "reason": "SHA-256 from a download",
                        "twin": by_full[key],
                    }
                )
            elif size and key in copy_keys:
                r.update(
                    {
                        "class": "duplicate",
                        "reason": "SHA-256 from a download",
                        "twin": copy_keys[key],
                    }
                )
            else:
                r.update({"class": "copy", "reason": "settled by a download"})
                _place(r, taken)
                if r["class"] == "copy" and size:
                    copy_keys[key] = r["dest"]
    finally:
        for p in made:
            if os.path.exists(p):
                os.remove(p)  # our own temporary download
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()  # the empty temporary folder
    _mark_deletable(rows)
    plan["bytechecked"] = utc_now()
    return plan


# ---------------------------------------------------------------- copy and verify


def _matches(path: Path | str, r: dict) -> tuple[bool, str]:
    size, md5, sha = hash_file(path)
    if size != (r["size"] or 0):
        return False, f"size {size} != {r['size']}"
    if r["md5"] and md5 != r["md5"]:
        return False, "MD5 differs"
    if r["sha256"] and sha != r["sha256"]:
        return False, "SHA-256 differs"
    return True, sha


def _copy_new(src: str, dst: Path) -> None:
    """Copy src to a file that must not exist yet (exclusive create: never overwrites).

    A copy that fails part-way removes the file it created, so no partial file stays.
    """
    os.makedirs(native(dst.parent), exist_ok=True)
    fd = os.open(native(dst), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, 1 << 20)
        shutil.copystat(src, native(dst))
    except BaseException:
        os.remove(native(dst))  # the partial file this call created (O_EXCL), never another's
        raise


def _publish(src: Path, dst: Path) -> None:
    """Make a verified staging file appear at dst, which must not exist.

    A hard link fails if dst exists, so nothing is ever written over; where hard links
    are not available, an exclusive create gives the same guarantee. The staging name
    is ours and is removed once the bytes are at dst.
    """
    os.makedirs(native(dst.parent), exist_ok=True)
    try:
        os.link(native(src), native(dst))
    except FileExistsError:
        raise
    except OSError:
        _copy_new(native(src), dst)
    os.remove(native(src))  # our own staging name; the bytes stay at dst


def _staging(root: Path) -> Path:
    return root.parent / (root.name + STAGING_SUFFIX)


def _run_folder(root: Path, kind: str) -> Path:
    """A new, empty folder of our own beside the root (same volume), unique to this run."""
    base = _staging(root)
    if _inside(base, root):
        raise ToolError(f"the staging folder {base} must be outside the root")
    os.makedirs(native(base), exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.mkdtemp(prefix=f"{kind}-{stamp}-", dir=str(base)))


def _tidy(staging: Path) -> None:
    for p in (staging, staging.parent):
        if os.path.isdir(native(p)) and not os.listdir(native(p)):
            os.rmdir(native(p))  # our own staging folder, once empty


def _batches(rows: list[dict], batch: int, staging: Path) -> list[list[dict]]:
    out, cur, length = [], [], 0
    for r in rows:
        extra = len(r["id"]) + len(str(staging / r["id"])) + 6
        if cur and (len(cur) >= batch or length + extra > MAX_CMDLINE):
            out.append(cur)
            cur, length = [], 0
        cur.append(r)
        length += extra
    if cur:
        out.append(cur)
    return out


def _check_dest(dest: str) -> None:
    parts = dest.split("/")
    if (
        not dest.startswith(LEGACY + "/")
        or "\\" in dest
        or ":" in dest
        or ".." in parts
        or "" in parts
        or "." in parts
    ):
        raise ToolError(f"refusing destination {dest!r}: copies go only under {LEGACY}/")


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
    pending = [r for r in plan["rows"] if r["class"] == "needs-byte-check"]
    if pending:
        raise ToolError(f"{len(pending)} object(s) still need a byte check; run bytecheck first")
    rows = [r for r in plan["rows"] if r["class"] == "copy"]
    for r in rows:
        _check_dest(r["dest"])
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
    guard_out(log, root)
    staging = _run_folder(root, "copy")
    lock = threading.Lock()
    results: list[dict] = []

    def fetch(pairs: list[tuple[dict, Path]]) -> str:
        args = [*RCLONE, "backend", "copyid", remote]
        for r, s in pairs:
            args += [r["id"], str(s)]
        cp = run_cmd(args)
        return "" if cp.returncode == 0 else (cp.stderr or "").strip()[-300:]

    def run(chunk: list[dict]) -> None:
        pairs = [(r, staging / r["id"]) for r in chunk]
        err = fetch(pairs)
        errors: dict[str, str] = {}
        missing = [(r, s) for r, s in pairs if not os.path.isfile(native(s))]
        if missing and len(pairs) > 1:
            # copyid stops at its first failing pair: ask for each missing one alone.
            for r, s in missing:
                errors[r["id"]] = fetch([(r, s)])
        elif missing:
            errors[missing[0][0]["id"]] = err
        out = []
        for r, s in pairs:
            rec = {"id": r["id"], "dest": r["dest"]}
            if not os.path.isfile(native(s)):
                rec["status"] = f"download failed: {errors.get(r['id']) or err or 'no file'}"
            else:
                ok, why = _matches(s, r)
                if not ok:
                    rec["status"] = f"MISMATCH with Drive's listing: {why}; kept at {s}"
                else:
                    try:
                        _publish(s, root / r["dest"])
                        rec.update({"status": "ok", "sha256": why})
                    except FileExistsError:
                        rec["status"] = f"conflict: the destination appeared; kept at {s}"
            out.append(rec)
        with lock:
            results.extend(out)
            with open(log, "a", encoding="utf-8") as fh:
                for rec in out:
                    fh.write(json.dumps({**rec, "at": utc_now()}) + "\n")

    chunks = _batches(todo, batch, staging)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for i, _ in enumerate(ex.map(run, chunks), 1):
            if i % 10 == 0 or i == len(chunks):
                print(f"copy: {i}/{len(chunks)} batches", file=sys.stderr)
    _tidy(staging)
    bad = [x for x in results if x["status"] != "ok"]
    print(f"copy: {len(results) - len(bad)} copied and verified; {len(bad)} not")
    for x in bad[:50]:
        print(f"  FAILED  {x['dest']}: {x['status']}")
    if any(x["status"].startswith(("MISMATCH", "conflict")) for x in bad):
        raise ToolError(
            "a download does not match Drive, or a destination appeared; nothing was overwritten", 3
        )
    return 4 if bad else 0


def verify(plan: dict, root: Path) -> int:
    pending = [r for r in plan["rows"] if r["class"] == "needs-byte-check"]
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
    leftovers = []
    staging = _staging(root)
    if os.path.isdir(native(staging)):
        for dirpath, _d, filenames in os.walk(native(staging)):
            leftovers.extend(os.path.join(dirpath, f) for f in filenames)
    partial = []
    legacy = root / LEGACY
    if legacy.is_dir():
        for rel_dir, _d, filenames, _p in _walk(legacy):
            partial.extend(
                f"{rel_dir}/{f}" if rel_dir else f for f in filenames if f.endswith(".partial")
            )
    print(
        f"verify: {len(rows)} copies: {ok} ok, {missing} missing, {bad} mismatched; "
        f"{len(pending)} object(s) still need a byte check"
    )
    for p in leftovers[:20]:
        print(f"  staging   {p} (a download kept for inspection; outside the root)")
    for p in partial[:20]:
        print(f"  partial   {LEGACY}/{p} (not data)")
    return 0 if missing == bad == len(pending) == 0 else 1


# ---------------------------------------------------------------- sweep (a local folder)

# What counts as a data file in a local folder, as scripts/data_manifest.py builds.
SWEEP_SKIP_SUFFIXES = (".py", ".md", ".pyc", ".gitkeep", ".log", ".lock")
SWEEP_SKIP_NAMES = ("__pycache__", "_locks", ".git", "DATA_MANIFEST.json", "_inventory_scan.json")


def sweep(source: Path, root: Path, inv: dict, dest: str, dry_run: bool) -> int:
    """Copy the data files of a local folder whose bytes the root lacks to root/dest."""
    parts = dest.split("/")
    if not AREA_NAME.match(dest) or ".." in parts or parts[0] in (*LIVE_TREES, LOGS):
        raise ToolError(f"refusing destination {dest!r}: sweep writes only outside the live trees")
    if not source.is_dir():
        raise ToolError(f"source {source} is not a directory")
    if _inside(source, root) or _inside(root, source):
        raise ToolError("the source and the root must not contain each other")
    if not same_path(inv["root"], str(root)):
        raise ToolError(f"the inventory was made for root {inv['root']}, not {root}")
    have = {(f["size"], f["md5"], f["sha256"]) for f in inv["files"] if not f.get("excluded")}
    base = native(source)
    todo, done, conflicts, home, skipped = [], [], [], 0, 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SWEEP_SKIP_NAMES)
        rel_dir = os.path.relpath(dirpath, base).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        for fn in sorted(filenames):
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            if (
                fn in SWEEP_SKIP_NAMES
                or fn.endswith(SWEEP_SKIP_SUFFIXES)
                or os.path.islink(full)
                or credential_shaped(rel)
                or not all(win_safe(p) for p in rel.split("/"))
            ):
                skipped += 1
                continue
            size, md5, sha = hash_file(full)
            if size and (size, md5, sha) in have:
                home += 1
                continue
            target = root / dest / rel
            if os.path.lexists(native(target)):
                got = hash_file(target) if os.path.isfile(native(target)) else None
                (done if got == (size, md5, sha) else conflicts).append(rel)
                continue
            todo.append((rel, full, size, sha))
    print(
        f"sweep: {home + len(todo) + len(done) + len(conflicts)} data files in {source}; "
        f"{home} already in the root by bytes; {len(done)} already swept; {len(todo)} to copy "
        f"({sum(t[2] for t in todo):,} B); {skipped} skipped (code, logs, credential names, links)"
    )
    for rel in conflicts:
        print(f"  CONFLICT  {dest}/{rel} holds other bytes")
    if conflicts:
        raise ToolError(f"{len(conflicts)} destination(s) hold other bytes; nothing was copied", 3)
    for rel, _full, size, _sha in todo:
        print(f"  {'would copy' if dry_run else 'copy'}  {dest}/{rel}  ({size:,} B)")
    if dry_run:
        return 0
    if todo:
        # Copy beside the root, re-hash there, then publish with a hard link: a bad copy
        # never reaches the root, and a file that appears meanwhile is never replaced.
        staging = _run_folder(root, "sweep")
        bad = []
        for n, (rel, full, _size, sha) in enumerate(todo):
            tmp = staging / f"{n:06d}"
            _copy_new(full, tmp)
            if hash_file(tmp)[2] != sha:
                bad.append(f"{dest}/{rel}: the copy does not match its source; kept at {tmp}")
                continue
            try:
                _publish(tmp, root / dest / rel)
            except FileExistsError:
                bad.append(f"{dest}/{rel}: the destination appeared; kept at {tmp}")
        _tidy(staging)
        for b in bad:
            print(f"  FAILED  {b}")
        if bad:
            raise ToolError(f"{len(bad)} file(s) not copied; nothing was overwritten", 3)
    print(f"sweep: copied and verified {len(todo)} file(s)")
    return 0


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
    lines += list(CREDENTIAL_PATHS)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- command line


def _areas(arg: str | None) -> list[dict]:
    return list(load_json(Path(arg))) if arg else [dict(a) for a in DEFAULT_AREAS]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")  # Drive names on a Windows pipe
        except (AttributeError, ValueError):
            pass
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
    sp.add_argument("--ledger", help="CSV ledger (default: <out>_ledger.csv)")
    sp.add_argument("--about", help="the output of `rclone about <remote> --json`")
    sp.add_argument("--limit", type=int, default=40)
    sp = sub.add_parser(
        "bytecheck", help="settle objects Drive lists without a size, MD5 or SHA-256"
    )
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
    sp.add_argument("--log", help="JSON-lines log (default: <plan>_copy.jsonl)")
    sp = sub.add_parser("verify", help="re-hash every copy against the plan")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--root")
    sp = sub.add_parser("sweep", help="copy a local folder's data files the root lacks, by bytes")
    sp.add_argument("--source", required=True)
    sp.add_argument("--root")
    sp.add_argument("--inventory", required=True)
    sp.add_argument("--dest", required=True, help="a folder under the root, outside the live trees")
    sp.add_argument("--dry-run", action="store_true")
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
        out = guard_out(Path(args.out), None)
        version = require_rclone()
        doc = census(args.remote, _areas(args.areas))
        doc["rclone"] = version
        write_outputs([(out, json_text(doc))])
        n = sum(len(a["objects"]) for a in doc["areas"])
        print(
            f"census: {len(doc['areas'])} areas, {n} objects; swe-data exists: {bool(doc['swe_data'])}"
        )
        return 0
    if args.cmd == "inventory":
        root = resolve_root(args.root)
        out = guard_out(Path(args.out), root)
        doc = inventory(root)
        write_outputs([(out, json_text(doc))])
        kept = [f for f in doc["files"] if not f["excluded"]]
        print(
            f"inventory: {len(doc['files'])} files ({len(kept)} hashed, {len(doc['files']) - len(kept)} "
            f"credential-shaped, not read), {sum(f['size'] for f in kept):,} B; {len(doc['links'])} links skipped"
        )
        return 0
    if args.cmd == "plan":
        inv = load_json(Path(args.inventory))
        out = guard_out(Path(args.out), inv["root"])
        ledger = guard_out(
            Path(args.ledger) if args.ledger else out.with_name(out.stem + "_ledger.csv"),
            inv["root"],
        )
        plan = build_plan(load_json(Path(args.census)), inv)
        write_outputs([(ledger, ledger_text(plan["rows"])), (out, json_text(plan))])
        about = load_json(Path(args.about)) if args.about else None
        print("\n".join(summarize(plan, about, args.limit)))
        return 0
    root = resolve_root(args.root)
    if args.cmd == "sums":
        n, total, digest = write_sums(root)
        print(f"sums: {SUMS} lists {n} files, {total:,} B; its own sha256 {digest}")
        return 0
    if args.cmd == "sweep":
        return sweep(
            Path(args.source), root, load_json(Path(args.inventory)), args.dest, args.dry_run
        )
    plan_path = Path(args.plan)
    plan = load_json(plan_path)
    if plan.get("kind") != "plan" or plan.get("schema") != SCHEMA:
        raise ToolError(f"{plan_path} is not a plan of schema {SCHEMA}")
    if args.cmd == "bytecheck":
        if not same_path(plan["root"], str(root)):
            raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
        guard_out(plan_path, root)
        require_rclone()
        tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="swe-bytecheck-"))
        if tmp.exists() and any(tmp.iterdir()):
            raise ToolError(f"the temporary folder {tmp} is not empty")
        bytecheck(plan, load_json(Path(args.inventory)), root, args.remote, tmp)
        ledger = plan_path.with_name(plan_path.stem + "_ledger.csv")
        write_outputs([(ledger, ledger_text(plan["rows"])), (plan_path, json_text(plan))])
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
    raise ToolError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
