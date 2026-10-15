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
    python scripts/drive_consolidate.py trashlist --plan plan.json --census census.json --swe-census swe.json --root C:\\swe-data --out C:\\swe-data\\_logs\\card3a\\list
    python scripts/drive_consolidate.py recount   --proof trashlist.json --summary trashlist_summary.json --expect-sha256 <sha256> --census fresh.json --root C:\\swe-data --remote gdrive:

``census`` lists every object in each area through raw Drive queries by parent id
(``rclone backend query``), so a shortcut is recorded as a shortcut and never
followed, and a Google-format file, a duplicate name or a second parent stays
visible. Each area's root folder is verified by its parent and name first. Drive
sometimes answers an OR of several parents with nothing (Google issue 149522397;
rclone works around it the same way), so an empty answer is asked again one parent
at a time. Last, each area's file count and bytes must equal ``rclone size`` run
with ``--disable ListR``, which lists one folder at a time: any difference stops the
run. A file Drive lists without an MD5 is listed but not counted, as rclone can
neither count nor download it. Every ``backend query`` carries
``--drive-auth-owner-only``, which there only adds each object's owners, so the census
records whether the account owns each object (``owned``); no other command gets it,
since ``lsjson`` and ``size`` would drop what the account does not own. The ``census``
command then records where it stands (D36), with queries that have no trashed clause:
the chain My Drive › The_Works › Projects › OptionsEngine_Project, each link read by
name and the last by its pinned id; ``swe-data/``, read by its recorded id in its
recorded parent, never by its name (missing, trashed or not a folder stops it); and the
sharing of each link, of ``swe-data/`` and of each area folder, read with
``backend query -M``, which returns the permissions Drive lists. None of that runs in
the re-census ``copy`` and ``bytecheck`` make.

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
                      file, a credential-shaped name (any file named config
                      among them), a path through a folder the census never listed,
                      a destination that cannot be placed

A Drive id is one object wherever it is seen, and areas can overlap: a path that
keeps it on Drive in one area keeps it there in every area, and the census keeps
no hash of it in any.

Only bytes count (D33): a missing hash never counts as a match, and a git object in
the old ``.git`` upload is not a copy of the bundle that holds the same logical
object. ``dest`` keeps the Drive path; a name Windows cannot hold, or that rclone
would rewrite, becomes ``_drive-<id>``; a destination already taken (any case) goes
to ``<area>/_conflicts/<id>/<name>``. A Drive id is ``deletable`` (a forecast for
card 3) only when its bytes already sit in the root (``redundant``), in a consolidate
area, at an unambiguous path, for every row that names it; a copy becomes deletable
only in the plan made after it lands.

``copy`` downloads by Drive id (``rclone backend copyid``) into a staging folder
next to the root (``<root>.d33-staging``, same volume, never inside the root),
re-hashes each file there, and only then publishes it with a hard link, which
fails if the destination exists: nothing is ever written over. A destination that
already holds the right bytes counts as done, so a rerun resumes; one holding other
bytes stops the run before anything is copied. A failed batch is retried one object
at a time, so one object that cannot be downloaded never blocks the others. The plan
is a snapshot: ``copy`` and ``bytecheck`` census the plan's areas again before the
first download, and an object gone, moved or renamed since the plan, or no longer
safe to read, stops the run before anything is fetched. They census once more after
the last download: ``copy`` publishes only what is unchanged (a changed object's
download is removed, never published), and ``bytecheck`` keeps no verdict at all.
``sweep`` does the same for a local folder, by bytes, through the same staging
folder; a root file the inventory lists counts only if it still holds its bytes.

A symbolic link or a Windows junction is never a file of the root's own: the
inventory lists it without following it, the plan places nothing at or under it,
and ``copy``, ``sweep`` and ``verify`` refuse a destination with one on its path.
``verify`` re-hashes every copy, and every root file that a Drive object was
matched to: one changed or removed since the inventory fails the check, because
its Drive twin is the only copy again.

``trashlist`` (card 3a, D36's middle route) accounts for every object the census lists
in the three consolidate areas, and for each area folder: each gets one class.
``listed`` is a file proven twice: every plan row naming it is ``deletable``; Drive
lists its size, MD5 and SHA-256, and it has one parent, is owned by the account, is
untrashed, and is neither a Google-format file nor a shortcut; its root or archive
twin, hashed again now, matches; ``swe-data/``'s census holds a file at exactly that
path, with one parent and another id, listing the same size, MD5 and SHA-256; and its
name has no credential shape. ``folder`` is a folder with one parent, owned, untrashed,
whose whole subtree holds only ``listed`` files and such folders. Everything else is a
``blocker``, with one reason; so is an object a read-only area also shows. What counts
comes from the census, never from the plan's rows, and a hash learned from a download
never counts. Its inputs are a plan over exactly card 2b's six areas, with no byte
check pending, the census it was made from, and a census of ``swe-data/`` alone; its
outputs go into a new folder under the root's ``_logs/``: ``trashlist.json`` (the
proof), ``trashlist.csv``, ``kept.csv`` (the blockers), ``sharing.json`` and
``trashlist_summary.json``. **The proof's sha256 is the sha256 of ``trashlist.json``'s
exact bytes**: canonical JSON (keys sorted, no spaces, UTF-8, rows in area, path and id
order, one newline at the end), with no time of its making in it. The Operator's yes
names that sha256. An area is "clear to trash by hand" only with no ``blocker`` row.

``recount``, on the day of the trash, reads only. Given the approved sha256 and a census
made after the proof, it says "unchanged: yes" for the areas checked only when the
proof and its summary are the approved ones, each area has no ``blocker`` row, the
census lists the same ids with the same kind, MIME type, parents, owner, name, size,
hashes and modified time, every listed file's desktop twin hashed again still matches,
and every ``swe-data/`` twin, read again in its folder, has the same id, bytes and
place and is untrashed. It writes its verdict to a new file under the root's
``_logs/``.

Nothing here deletes or moves a Drive object or a root file. The only files the
tool removes or replaces are its own: temporary downloads outside the root, the
staging names of files once published, the unfinished file a failed copy has just
created (exclusive creation proves it is its own), and its own output files, which
never go inside the root except under ``_logs/``. Deletion is a separate step with
the Operator's yes (D33, card 3).

Exit codes: 0 ok · 1 verify or recount found a difference · 2 configuration or safety error,
or an unexpected one · 3 a destination holds other bytes, a download does not
match Drive, or Drive changed since the plan · 4 a copy or file operation failed (a
rerun resumes) · 5 not enough free space. Standard library only; the
Drive side is rclone ≥ 1.65 (Drive SHA-256).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
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
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = 5  # 5: owners; swe-data/ by its id; the chain and sharing; trashlist and recount
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
MAX_PATHS = 64  # paths to one object through folders with two parents; beyond, left on Drive
OUTSIDE = "\x00outside"  # a path segment no Drive name holds: a parent the census never listed
STAGING_SUFFIX = ".d33-staging"

RCLONE: list[str] = ["rclone"]  # the command; tests substitute a fake

# Passed to ``backend query`` and to no other command. There it only adds each object's
# owners to what Drive returns (rclone v1.74.4 getFileFields); lsjson and size would drop
# every object the account does not own, and the census must stay complete.
OWNER_ONLY = "--drive-auth-owner-only"
# rclone's global --metadata. With ``backend query`` it adds Drive's permissions to what
# is returned (getFileFields with metadataFields): the sharing read, and only that read.
SHARING_FLAG = "-M"
# What the census keeps of each permission Drive lists: who is reached and how, never a
# name or an email address.
SHARING_KEYS = (
    "id",
    "type",
    "role",
    "domain",
    "allowFileDiscovery",
    "deleted",
    "expirationTime",
    "pendingOwner",
    "view",
)

# Where the project's folders sit since 2026-10-12 (D36): My Drive › The_Works › Projects ›
# OptionsEngine_Project, found by name from My Drive's root, the last by its pinned id.
CHAIN = ("The_Works", "Projects", "OptionsEngine_Project")
PROJECT_FOLDER_ID = "1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4"  # OptionsEngine_Project
# swe-data/, D33's second copy, read by its id in its recorded parent, never by its name.
SWE_DATA: dict = {
    "name": "swe-data",
    "id": "13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b",
    "parent": PROJECT_FOLDER_ID,
    "folder": "swe-data",
    "mode": "read-only",
}

# The Drive areas of docs/DATA_INVENTORY.md §C.3 (D33). ``name`` is the folder under
# data_archive/drive-legacy/; ``parent`` and ``folder`` verify the id before a census;
# ``read-only`` areas belong to another project: copied from, never cleaned.
DEFAULT_AREAS: tuple[dict, ...] = (
    {
        "name": "swe-local-only",
        "id": "1JwPWszfyggUDT1vYaRjZ8nlHEDR3vEOn",
        "parent": PROJECT_FOLDER_ID,
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
        "parent": PROJECT_FOLDER_ID,
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
# Card 2b's six areas: the areas above without the day-bot's own data_raw (D34 point 3).
# trashlist takes a plan over exactly these.
CARD2B_AREAS: tuple[dict, ...] = tuple(
    a for a in DEFAULT_AREAS if a["name"] != "day-bot-local-archive/data_raw"
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
    "*.env",
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
# Whole file names, in any folder: a git config can carry a token in a remote URL, and a
# git folder can come under any name (a bare mirror, an upload without its ".git" name).
# A folder named config is not affected.
CREDENTIAL_PATHS: tuple[str, ...] = ("config", "config.worktree")

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

# Plain folder names, each with at most one "=" inside (a hive partition folder: ticker=AAPL).
_AREA_PART = r"[A-Za-z0-9._-]+(?:=[A-Za-z0-9._-]+)?"
AREA_NAME = re.compile(rf"^{_AREA_PART}(?:/{_AREA_PART})*\Z")  # \Z: no trailing newline
_WIN_BAD = set('<>:"/\\|?*')
_WIN_RESERVED = (
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
    | {f"{p}{s}" for p in ("COM", "LPT") for s in "\u00b9\u00b2\u00b3"}
)


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


def _fold(name: str) -> str:
    """The key under which two names clash on Windows. NTFS upcases, so "ı" and "i"
    clash there, which casefold alone misses; a key that clashes more is only safer."""
    return name.upper().casefold()


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
    na = os.path.normcase(_plain(os.path.abspath(_plain(a))))
    nb = os.path.normcase(_plain(os.path.abspath(_plain(b))))
    return na == nb


# The two extended-length forms native() writes: a drive letter, and a server share.
_EXTENDED_DRIVE = re.compile(r"\\\\\?\\[A-Za-z]:\\")
_EXTENDED_UNC = re.compile(r"\\\\\?\\UNC\\", re.IGNORECASE)


def _plain(path: str) -> str:
    """A Windows extended-length path (\\\\?\\C:\\... or \\\\?\\UNC\\...) spelled the ordinary way.

    Any other such form (a volume GUID, GLOBALROOT, a \\\\.\\ device path) is refused:
    without its prefix it would read as a relative path, and be compared as one.
    """
    if _EXTENDED_UNC.match(path):
        return "\\\\" + path[8:]
    if _EXTENDED_DRIVE.match(path):
        return path[4:]
    odd = ("\\\\?\\", "\\\\.\\") + (("//?/", "//./") if os.name == "nt" else ())
    if path.startswith(odd):
        raise ToolError(f"unsupported path form {path!r}: spell it with a drive letter or a share")
    return path


def _inside(path: Path | str, folder: Path | str) -> bool:
    """Is path the folder or under it, once links are resolved on both sides?"""
    p = os.path.normcase(_plain(os.path.realpath(_plain(str(path)))))
    f = os.path.normcase(_plain(os.path.realpath(_plain(str(folder)))))
    return p == f or p.startswith(f.rstrip("\\/") + os.sep)


# Links followed on the way to one path before it is refused (Linux stops at 40 too).
MAX_HOPS = 40
# What readlink gives for a volume mounted in a folder: that volume's top, and nothing below.
_VOLUME_ROOT = re.compile(r"\\\\\?\\Volume\{[0-9a-f-]+\}\\?", re.IGNORECASE)


def _link_in_root_on(path: Path, root: Path | str, hops: int = 0) -> str | None:
    """A link or junction in the root that the way to path passes through, however spelled.

    An output meant for the root's _logs/ must land there, not wherever a link under the
    root leads (outside it, where it could replace a file of someone else's). Links are
    followed hop by hop, as the OS follows them, so such a link is found also when the
    path reaches it by another name, or through where another link leads. A link above
    the root, or the root itself reached through one, is not refused.
    """
    for cur in (*reversed(path.parents), path):
        if cur.parent == cur or not _is_link(native(cur)):
            continue
        if _inside(cur.parent, root):
            return str(cur)
        if hops >= MAX_HOPS:
            raise ToolError(f"refusing {path}: more than {MAX_HOPS} links on the way")
        try:
            raw = os.readlink(native(cur))
        except OSError as e:
            raise ToolError(f"refusing {path}: cannot follow {cur} ({e.strerror or e})") from None
        if _VOLUME_ROOT.fullmatch(raw):
            continue  # a volume mounted in a folder outside the root: walk on from its top
        target = Path(_plain(raw))  # a place inside a volume, or GLOBALROOT: refused there
        if not target.is_absolute():
            target = cur.parent / target
        # The OS goes on from where the link leads: every earlier part was no link.
        return _link_in_root_on(target / path.relative_to(cur), root, hops + 1)
    return None


def guard_out(path: Path, root: Path | str | None) -> Path:
    """Our own output files never land inside the data root, except under its _logs/.

    Returns the path to write: absolute, as native() spells it. Every check is made on
    that path, so the caller must write there.
    """
    spelled = _plain(str(path))
    out = Path(os.path.abspath(spelled))
    if os.path.realpath(spelled) != os.path.realpath(out):
        # A ".." after a link: the OS goes up from where the link leads, native() from the
        # link's own folder. The checks and the write must mean one place.
        raise ToolError(f"refusing to write {path}: a '..' in it follows a link")
    if root is None:
        env = os.environ.get("SWE_DATA_ROOT", "").strip()
        root = env or None
    if root is None:
        return out
    link = _link_in_root_on(out, root)
    if link:
        raise ToolError(f"refusing to write {path}: {link} is a link or junction in the data root")
    if _inside(out, root) and _is_link(native(Path(root) / LOGS)):
        # However the output is spelled, a linked _logs could lead into the live trees.
        raise ToolError(f"refusing to write {path}: {Path(root) / LOGS} is a link or junction")
    if _inside(out, root) and not _inside(out, Path(root) / LOGS):
        # Links resolved: a link inside _logs that leads into the live trees is refused too.
        raise ToolError(
            f"refusing to write {path} inside the data root; use {LOGS}/ or a folder outside it"
        )
    return out


def _file_key(path: Path | str) -> str:
    """One file's name for comparing: links resolved, by the platform's case rule."""
    return os.path.normcase(_plain(os.path.realpath(_plain(str(path)))))


def _not_an_input(outputs: list[Path], inputs: list) -> None:
    """Refuse, before any work, an output that is also one of the command's own inputs."""
    ins = {_file_key(i) for i in inputs if i}
    for out in outputs:
        if _file_key(out) in ins:
            raise ToolError(f"refusing to write {out}: it is also an input of this command")


def write_outputs(pairs: list[tuple[Path, str]]) -> None:
    """Write our own output files: every temp file first, then replace them all.

    Each temp file is a new name of our own (an exclusive create, so never an existing
    file or link), and a destination that is a link is refused: an output never lands
    in whatever a link points to.
    """
    names = [_file_key(p) for p, _t in pairs]
    if len(set(names)) < len(names):  # the second would silently replace the first
        raise ToolError("two outputs name the same file: " + ", ".join(str(p) for p, _t in pairs))
    temps: list[tuple[str, Path]] = []
    try:
        for path, text in pairs:
            if _is_link(native(path)):
                raise ToolError(f"refusing to write {path}: it is a link or junction")
            fd, tmp = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=native(path.parent)
            )
            temps.append((tmp, path))
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        while temps:
            tmp, path = temps[0]
            os.replace(tmp, native(path))
            temps.pop(0)  # replaced: that name is the output now, not ours to clean up
    except BaseException:
        for tmp, _path in temps:
            if os.path.lexists(tmp):
                os.remove(tmp)  # our own temporary output (mkstemp created it)
        raise


def _load_doc(path: Path, kind: str) -> dict:
    """One of the tool's own documents, of this schema: an older one may mean other rules."""
    doc = load_json(path)
    if not isinstance(doc, dict) or doc.get("kind") != kind or doc.get("schema") != SCHEMA:
        raise ToolError(f"{path} is not a {kind} of schema {SCHEMA}; make it again")
    return doc


def json_text(data) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def _check_areas(areas: list[dict]) -> None:
    seen = set()
    for a in areas:
        name = a.get("name", "")
        if not AREA_NAME.match(name) or ".." in name.split("/"):
            raise ToolError(
                f"area name {name!r} must be letters, digits, '.', '_', '-' and '/' only, "
                "with at most one '=' inside each folder name (ticker=AAPL)"
            )
        if not all(win_safe(p) for p in name.split("/")):
            raise ToolError(f"area name {name!r} holds a part Windows cannot keep")
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


def drive_query(remote: str, query: str, tries: int = 3, sharing: bool = False) -> list[dict]:
    """Raw Drive objects matching a query; retried, and INCOMPLETE is an error.

    Every query carries OWNER_ONLY, so each object comes with its owners. ``sharing``
    adds SHARING_FLAG, so each comes with the permissions Drive lists too.
    """
    flags = [OWNER_ONLY, SHARING_FLAG] if sharing else [OWNER_ONLY]
    last = ""
    for attempt in range(tries):
        cp = run_cmd([*RCLONE, "backend", "query", *flags, remote, query])
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
    """File count and bytes under a folder, from rclone's own walk.

    ``--disable ListR`` makes rclone list one folder at a time. Its default listing
    asks about many parents in one OR query, the query Google issue 149522397
    affects, so it would not be an independent check of the census.
    """
    cp = run_cmd(
        [
            *RCLONE,
            "size",
            "--disable",
            "ListR",
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
    sha = (obj.get("sha256Checksum") or "").lower() or None
    size = int(size) if size not in (None, "") else None
    if size is None and md5 == EMPTY_MD5:
        size = 0  # rclone's JSON omits a zero size (``size,omitempty``)
    return {
        "id": obj["id"],
        "name": obj.get("name", ""),
        "mime": obj.get("mimeType", ""),
        "size": size,
        "md5": md5,
        "sha256": sha,
        "parents": list(obj.get("parents") or []),
        "created": obj.get("createdTime"),
        "modified": obj.get("modifiedTime"),
        "target": sc.get("targetId"),
        "target_mime": sc.get("targetMimeType"),
        "owned": owned_by_me(obj),
        "trashed": bool(obj.get("trashed")),
    }


def owned_by_me(obj: dict) -> bool | None:
    """True when one of the object's owners is the signed-in account (Drive's ``me``).

    None when Drive returned no owners at all: the query lacked OWNER_ONLY, so ownership is
    not known, and unknown never counts as owned.
    """
    owners = obj.get("owners")
    if owners is None:
        return None
    return any(bool(u.get("me")) for u in owners if isinstance(u, dict))


def sharing_of(obj: dict) -> list[dict] | None:
    """The permissions Drive lists for an object, each cut to SHARING_KEYS; None if none listed."""
    perms = obj.get("permissions")
    if perms is None:
        return None
    kept = [{k: p[k] for k in SHARING_KEYS if k in p} for p in perms if isinstance(p, dict)]
    return sorted(kept, key=lambda p: json.dumps(p, sort_keys=True))


def sharing_flags(perms: list[dict] | None) -> list[str]:
    """Every permission other than the owner's, in words; and a missing list is flagged too."""
    if perms is None:
        return ["Drive listed no permissions: its sharing is not known"]
    return [
        f"{p.get('type', '?')} {p.get('role', '?')}"
        + (f" ({p['domain']})" if p.get("domain") else "")
        + (f" [permission {p['id']}]" if p.get("id") else "")
        for p in perms
        if p.get("role") != "owner"
    ]


def is_file(o: dict) -> bool:
    return o["mime"] not in (FOLDER, SHORTCUT) and not o["mime"].startswith(GOOGLE)


def census_totals(objects: list[dict], top: str) -> dict:
    """File count and bytes as rclone's walk sees them: once per parent folder, and only
    files with an MD5. rclone neither lists nor downloads a non-Google file without one
    (``newObjectWithExportInfo`` in its Drive backend skips it)."""
    folders = {top} | {o["id"] for o in objects if o["mime"] == FOLDER}
    count = size = 0
    for o in objects:
        if is_file(o) and o["md5"]:
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
        top = next(h for h in hits if h["id"] == area["id"])
        root_parents = list(top.get("parents") or [])
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
            names = Counter(
                (p, o["name"]) for o in objects if o["mime"] == FOLDER for p in o["parents"]
            )
            twins = sorted({n for (_p, n), k in names.items() if k > 1})
            forks = sorted(
                o["name"] for o in objects if o["mime"] == FOLDER and len(o["parents"]) > 1
            )
            raise ToolError(
                f"area {area['name']}: the census found {mine['count']} files, {mine['bytes']:,} B, "
                f"but rclone size finds {walk['count']} files, {walk['bytes']:,} B. A listing is "
                "incomplete, or the area holds folders rclone's walk by path counts differently: "
                "two folders that share a name under one parent"
                + (f" (here: {', '.join(twins[:5])})" if twins else "")
                + ", or a folder with two parents"
                + (f" (here: {', '.join(forks[:5])})" if forks else "")
                + "; nothing is planned from it"
            )
        nomd5 = sum(1 for o in objects if is_file(o) and not o["md5"])
        root = slim(top)
        out.append(
            {
                **area,
                "root_parents": root_parents,
                "root_owned": root["owned"],
                "root_modified": root["modified"],
                "objects": objects,
                "check": walk,
            }
        )
        print(
            f"census: {area['name']}: {len(objects)} objects; {walk['count']} files, "
            f"{walk['bytes']:,} B, equal to rclone size"
            + (f"; {nomd5} file(s) without an MD5, which rclone cannot download" if nomd5 else ""),
            file=sys.stderr,
        )
    # One Drive object is one listing: an id listed differently in two areas changed
    # while the census ran (updated, or moved from one area to the next).
    first: dict[str, tuple[str, dict]] = {}
    for a in out:
        for o in a["objects"]:
            seen_in, was = first.setdefault(o["id"], (a["name"], o))
            if was != o:
                raise ToolError(
                    f"object {o['id']} ({o['name']}) is listed differently in area {seen_in} "
                    f"and area {a['name']}: it changed while the census ran; run it again"
                )
    # One verdict per Drive id across every area: an object any area sees on a path the
    # tool must not read keeps no hash anywhere (areas can overlap, nested in each other).
    bad = _unsafe_ids(out)
    for a in out:
        for o in a["objects"]:
            if o["id"] in bad:
                o["md5"] = "withheld" if o["md5"] else None
                o["sha256"] = None
    return {
        "schema": SCHEMA,
        "kind": "census",
        "generated": utc_now(),
        "remote": remote,
        "areas": out,
    }


def _folder_record(obj: dict) -> dict:
    """A folder the census records by id: where it is, whose it is, and its sharing."""
    s = slim(obj)
    return {
        "id": s["id"],
        "name": s["name"],
        "mime": s["mime"],
        "parents": s["parents"],
        "owned": s["owned"],
        "trashed": s["trashed"],
        "modified": s["modified"],
        "sharing": sharing_of(obj),
    }


def _where(parent: str) -> str:
    return "'root' in parents" if parent == "root" else f"{q_quote(parent)} in parents"


def read_chain(remote: str) -> list[dict]:
    """The folders from My Drive's root down to OptionsEngine_Project, each read by name.

    Three queries with no trashed clause, so a trashed or second folder of the name is
    seen: each must return exactly one object, an untrashed folder, and the last must be
    OptionsEngine_Project's pinned id. Any other result stops the census.
    """
    links, parent = [], "root"
    for name in CHAIN:
        hits = drive_query(remote, f"{_where(parent)} and name = {q_quote(name)}", sharing=True)
        if len(hits) != 1:
            raise ToolError(
                f"the chain: {len(hits)} objects named {name!r} in {parent}, not exactly one: "
                f"{sorted(h.get('id', '?') for h in hits)}; nothing is recorded"
            )
        hit = hits[0]
        if hit.get("mimeType") != FOLDER:
            raise ToolError(f"the chain: {name!r} ({hit['id']}) in {parent} is not a folder")
        if hit.get("trashed"):
            raise ToolError(f"the chain: {name!r} ({hit['id']}) in {parent} is in Drive's trash")
        links.append(_folder_record(hit))
        parent = hit["id"]
    if links[-1]["id"] != PROJECT_FOLDER_ID:
        raise ToolError(
            f"the chain: OptionsEngine_Project is {links[-1]['id']}, not the pinned "
            f"{PROJECT_FOLDER_ID}; nothing is recorded"
        )
    if len(links[0]["parents"]) != 1:
        raise ToolError(
            f"the chain: The_Works has parents {links[0]['parents']}, not My Drive's root alone"
        )
    return links


def read_by_id(remote: str, spec: dict) -> dict:
    """A folder read by its recorded id, in its recorded parent (design v6 §3).

    One query of the parent with no trashed clause: exactly one child carries the id,
    and it must be an untrashed folder. Its name plays no part.
    """
    hits = drive_query(remote, _where(spec["parent"]), sharing=True)
    mine = [h for h in hits if h.get("id") == spec["id"]]
    label = f"{spec['name']}/ (id {spec['id']})"
    if not mine:
        raise ToolError(
            f"{label} is missing from its recorded parent {spec['parent']}: moved, deleted "
            "or not visible; nothing is recorded"
        )
    if len(mine) > 1:
        raise ToolError(f"{label} is listed {len(mine)} times in {spec['parent']}")
    hit = mine[0]
    if hit.get("trashed"):
        raise ToolError(f"{label} is in Drive's trash; nothing is recorded")
    if hit.get("mimeType") != FOLDER:
        raise ToolError(f"{label} is not a folder ({hit.get('mimeType')}); nothing is recorded")
    return _folder_record(hit)


def locate(remote: str, doc: dict) -> None:
    """Record where the census stands: the chain, swe-data/ by its id, and each area's sharing.

    Run by the ``census`` command only, never inside census(), so the re-census that copy
    and bytecheck make (``_drifted``) never reads swe-data/ or the chain.
    """
    chain = read_chain(remote)
    doc["chain"] = chain
    doc["my_drive_root"] = chain[0]["parents"][0]  # the API never returns the alias 'root'
    doc["swe_data"] = read_by_id(remote, SWE_DATA)
    for a in doc["areas"]:
        hits = drive_query(
            remote, f"{_where(a['parent'])} and name = {q_quote(a['folder'])}", sharing=True
        )
        mine = [h for h in hits if h.get("id") == a["id"]]
        if len(mine) != 1:
            raise ToolError(f"area {a['name']}: its folder {a['id']} could not be read again")
        a["sharing"] = sharing_of(mine[0])


# ---------------------------------------------------------------- inventory


# A Windows junction's reparse tag (stat.IO_REPARSE_TAG_MOUNT_POINT, defined on Windows only).
JUNCTION_TAG = 0xA0000003


def _is_link(path: str) -> bool:
    """A symbolic link, or on Windows a junction: never part of the root's own tree.

    os.path.isjunction is Python 3.12's. On 3.11 a junction is known by its reparse tag,
    which os.lstat reports on Windows since 3.8.
    """
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None:
        return bool(isjunction(path))
    try:
        return getattr(os.lstat(path), "st_reparse_tag", 0) == JUNCTION_TAG
    except OSError:
        return False


def _link_on_path(root: Path, rel: str) -> str | None:
    """The first part of root/rel that is a link or junction, if any (the root itself may be)."""
    cur = root
    for part in rel.split("/"):
        cur = cur / part
        if _is_link(native(cur)):
            return str(cur)
    return None


def _walk_error(e: OSError) -> None:
    raise ToolError(f"cannot list {e.filename}: {e.strerror or e}; nothing may be left out")


def _walk(root: Path):
    """(relative dir, dirnames, filenames, native dir, linked dirnames) under the root.

    Skips ``_logs/`` (in any case, as the rclone filter does), never descends into a
    linked folder (os.walk follows a Windows junction, whose bytes live outside the
    root), and stops on a folder it cannot list rather than leave it out.
    """
    base = native(root)
    for dirpath, dirnames, filenames in os.walk(base, onerror=_walk_error):
        rel_dir = os.path.relpath(dirpath, base).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        if not rel_dir:
            dirnames[:] = [d for d in dirnames if _fold(d) != _fold(LOGS)]
        linked = sorted(d for d in dirnames if _is_link(os.path.join(dirpath, d)))
        dirnames[:] = sorted(d for d in dirnames if d not in linked)
        yield rel_dir, dirnames, sorted(filenames), dirpath, linked


def inventory(root: Path) -> dict:
    files, dirs, links = [], [], []
    count = 0
    for rel_dir, dirnames, filenames, dirpath, linked in _walk(root):
        links.extend(f"{rel_dir}/{d}" if rel_dir else d for d in linked)
        dirs.extend(f"{rel_dir}/{d}" if rel_dir else d for d in dirnames)
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            if _is_link(full):
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
        self.dirs = {_fold(d) for d in dirs}
        for p in files:
            self.take(p)

    def free(self, p: str) -> bool:
        cf = _fold(p)
        if cf in self.files or cf in self.dirs:
            return False
        parts = cf.split("/")
        return not any("/".join(parts[:i]) in self.files for i in range(1, len(parts)))

    def take(self, p: str) -> None:
        cf = _fold(p)
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
    # The area root itself in a second folder (another project's, say): every item in
    # it is in that folder too, so none is ever cleaned by path.
    # A census without the root's parents cannot show it has only one: ambiguous.
    top_own = len(area.get("root_parents") or []) == 1
    inside_of = {}
    for o in objs.values():
        inside = [p for p in o["parents"] if p == top or p in objs]
        if not inside:
            raise ToolError(f"area {area['name']}: object {o['id']} has no parent inside the area")
        inside_of[o["id"]] = inside
    # A name counts under every folder it is in, so a second parent's clash is seen too.
    siblings = Counter((p, objs[i]["name"]) for i in objs for p in inside_of[i])
    paths, chains, unique = {}, {}, {}

    def resolve(oid: str, chain: frozenset) -> bool:
        """Its path through the first parent that does not lead back into the chain."""
        if oid in paths:
            return True
        if len(chain) > 200:
            raise ToolError(f"area {area['name']}: folder chain deeper than 200 at {oid}")
        name = objs[oid]["name"]
        # One parent and no sibling of the same name. An item with a second parent,
        # even outside every area, is also in another folder: never cleaned by path.
        own = len(objs[oid]["parents"]) == 1 and all(
            siblings[(p, name)] == 1 for p in inside_of[oid]
        )
        for par in inside_of[oid]:
            if par == top:
                paths[oid], chains[oid], unique[oid] = name, [oid], own and top_own
                return True
            if par != oid and par not in chain and resolve(par, chain | {oid}):
                paths[oid] = f"{paths[par]}/{name}"
                chains[oid] = [*chains[par], oid]
                unique[oid] = own and unique[par]
                return True
        return False  # every parent leads back into the chain: a loop

    for oid in objs:
        if not resolve(oid, frozenset()):
            raise ToolError(f"area {area['name']}: {oid} is reached only through a loop of folders")
    return paths, chains, unique


def _every_path(area: dict) -> dict[str, list[str]]:
    """Per object id: its path through every in-area parent chain (up to MAX_PATHS + 1).

    A folder with two parents gives what is under it two paths; a rule that reads a
    path (a credential-shaped name) must hold on every one of them. A parent outside
    the census gives a path starting with OUTSIDE: its names are not known.
    """
    objs = {o["id"]: o for o in area["objects"]}
    top = area["id"]
    top_single = len(area.get("root_parents") or []) == 1
    memo: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def walk(oid: str) -> list[str]:
        if oid in memo:
            return memo[oid]
        if oid in visiting:  # a loop of folders: that path cannot be known
            return [OUTSIDE]
        visiting.add(oid)
        name, out = objs[oid]["name"], []
        for p in objs[oid]["parents"]:
            if p == top:
                out.append(name)
                if not top_single:  # the area root is in another folder too
                    out.append(f"{OUTSIDE}/{name}")
            elif p in objs:
                out.extend(f"{q}/{name}" for q in walk(p))
            else:  # a parent this census never listed: that path is not known
                out.append(f"{OUTSIDE}/{name}")
            if len(out) > MAX_PATHS:
                break
        visiting.discard(oid)
        memo[oid] = out[: MAX_PATHS + 1]
        return memo[oid]

    return {oid: walk(oid) for oid in objs}


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
    empty_root = {f["path"] for f in usable if f["size"] == 0 and f["sha256"] == EMPTY_SHA256}
    # A link is occupied too: nothing is placed at or under it.
    taken = Taken([f["path"] for f in root_files] + inv.get("links", []), inv.get("dirs", []))
    trees = {a["id"]: _area_tree(a) for a in census_doc["areas"]}
    bad = _unsafe_ids(census_doc["areas"])
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
            if o["id"] in bad:  # whatever the census kept, the plan keeps no hash of it
                row["md5"] = "withheld" if row["md5"] else None
                row["sha256"] = None
            if kind == "file" and o["id"] in bad:
                why, seen = bad[o["id"]]
                cls, twin = "unresolved", ""
                reason = why if seen == a["name"] else f"{why} (as seen in area {seen})"
            else:
                cls, reason, twin = _classify(
                    o, by_full, md5_count, first_copy, row["candidates"], empty_root
                )
            row.update({"class": cls, "reason": reason, "twin": twin})
            prev = first_row.get(o["id"])
            if prev is not None and cls in ("copy", "duplicate", "redundant", "needs-byte-check"):
                # One Drive object reached through two areas (a folder with two
                # parents): it comes home once, through its first row (_mirror).
                row["same_object"] = True
            elif cls == "copy":
                _place(row, taken)
                if row["class"] == "copy" and o["size"] and o["md5"] and o["sha256"]:
                    first_copy.setdefault((o["size"], o["md5"], o["sha256"]), row["dest"])
            first_row.setdefault(o["id"], row)
            rows.append(row)
    _mirror(rows)
    _mark_deletable(rows)
    return {
        "schema": SCHEMA,
        "kind": "plan",
        "generated": utc_now(),
        "root": inv["root"],
        "census_generated": census_doc["generated"],
        "inventory_generated": inv["generated"],
        "areas": [
            {k: a.get(k) for k in ("name", "id", "parent", "folder", "mode")}
            for a in census_doc["areas"]
        ],
        "swe_data": census_doc.get("swe_data"),
        "root_bytes": sum(f["size"] for f in usable),
        "root_files": len(usable),
        "excluded": [f["path"] for f in root_files if f.get("excluded")],
        "rows": rows,
    }


def _place(row: dict, taken: Taken) -> None:
    # A Drive id inside a path (_drive-<id>, _conflicts/<id>/) can look credential-shaped;
    # such a copy would be left out of the inventory, SHA256SUMS and swe-data.
    dest = taken.assign([c for c in row["candidates"] if not credential_shaped(c)])
    if dest is None:
        row.update(
            {"class": "unresolved", "reason": "no free destination under the root", "dest": ""}
        )
    else:
        row["dest"] = dest


def _classify(o, by_full, md5_count, first_copy, candidates, empty_root):
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
    if o["md5"] == "withheld":  # the census kept no hash: never read or copied
        return "unresolved", "hashes withheld by the census: never read or copied", ""
    if o["size"] is None:
        return "needs-byte-check", "Drive lists no size", ""
    if not o["md5"]:
        return "needs-byte-check", "Drive lists no MD5", ""
    if o["size"] == 0:
        if not _empty_ok(o):
            return "needs-byte-check", "empty, but Drive does not list the empty file's hashes", ""
        for cand in candidates:
            if cand in empty_root:  # exactly this name: a case variant is copied beside it
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


def _unsafe_ids(areas: list[dict]) -> dict[str, tuple[str, str]]:
    """Drive ids the tool must never read or copy: id -> (reason, the area that showed it).

    Judged on every path in every area, since areas can overlap (one nested in
    another) and an object is one Drive id wherever it is seen: a credential-shaped
    name on any path (a file named config among them), a path through a parent no
    census listed, or more than MAX_PATHS paths.
    """
    bad: dict[str, tuple[str, str]] = {}
    for area in areas:
        every = _every_path(area)
        for o in area["objects"]:
            paths = every[o["id"]]
            if len(paths) > MAX_PATHS:
                reason = f"reached through more than {MAX_PATHS} paths"
            elif any(credential_shaped(p) for p in paths):  # every path, not just the first
                reason = "credential-shaped name: never read or copied"
            elif any(p.startswith(OUTSIDE) for p in paths):  # names that cannot be checked
                reason = "a path through a folder outside the area, or a loop: not known"
            else:
                continue
            bad.setdefault(o["id"], (reason, area["name"]))
    return bad


def _mirror(rows: list[dict]) -> None:
    """A Drive object reached through two areas comes home once, through its first row.

    Each later row mirrors that row: a duplicate of it when its bytes come home, and
    otherwise its class, so no row claims a home the object will not have.
    """
    first: dict[str, dict] = {}
    for r in rows:
        f = first.setdefault(r["id"], r)
        if f is r or not r.get("same_object"):
            continue
        where = f"the same Drive object as {f['area']}/{f['path']}"
        if f["class"] in ("copy", "redundant", "duplicate"):
            r.update({"class": "duplicate", "reason": where, "twin": f["dest"] or f["twin"]})
        elif f["class"] == "needs-byte-check":
            r.update({"class": f["class"], "reason": f"{where}, which still needs a byte check"})
            r["twin"] = ""
        else:
            r.update({"class": f["class"], "reason": f"{where}: {f['reason']}", "twin": ""})
        r["dest"] = ""


def _mark_deletable(rows: list[dict]) -> None:
    """A forecast for card 3: an id may go only when its bytes already sit in the root.

    That is a redundant row (or a later row of the same object mirroring one), in a
    consolidate area, at an unambiguous path, for every row naming the id. A copy is
    never deletable here: only the plan made after it lands can call it redundant.
    """
    ok: dict[str, bool] = {}
    for r in rows:
        each = (
            r["mode"] == "consolidate"
            and bool(r["path_unique"])
            and (
                r["class"] == "redundant"
                or (r["class"] == "duplicate" and bool(r.get("same_object")))
            )
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
    swe = plan.get("swe_data")
    if isinstance(swe, dict):
        lines.append(f"swe-data/: id {swe['id']}, in {', '.join(swe['parents']) or '?'}")
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


def _drifted(plan: dict, remote: str, rows: list[dict]) -> list[tuple[dict, str]]:
    """The rows whose Drive object changed since the plan, each with what changed.

    The plan is a snapshot. A fresh census of its areas must show each object at the
    same path, with the same size and hashes wherever both list them, and still safe to
    read (``_unsafe_ids``): one renamed to a credential-shaped name since, moved under
    one, moved at all, or edited in place, has changed.
    """
    specs = []
    for a in plan["areas"]:
        if not a.get("parent") or not a.get("folder"):
            raise ToolError(f"the plan does not record where area {a['name']} is; plan again")
        specs.append({k: a[k] for k in ("name", "id", "parent", "folder", "mode")})
    fresh = census(remote, specs)
    bad = _unsafe_ids(fresh["areas"])
    now: dict[tuple[str, str], tuple[str, dict]] = {}
    for a in fresh["areas"]:
        paths = _area_tree(a)[0]
        now.update({(a["name"], o["id"]): (paths[o["id"]], o) for o in a["objects"]})
    changed = []
    for r in rows:
        where, o = now.get((r["area"], r["id"]), (None, None))
        if r["id"] in bad:  # a second folder can make it unsafe while its path stays
            changed.append((r, bad[r["id"]][0]))
        elif where != r["path"]:
            changed.append((r, f"now at {r['area']}/{where}" if where else "gone from the area"))
        elif any(o[k] is not None and r[k] is not None and o[k] != r[k] for k in HASHED):
            changed.append((r, "edited in place: its size or hashes differ from the plan's"))
    for r, why in changed[:40]:
        print(f"  CHANGED  {r['area']}/{r['path']}: {why}")
    return changed


HASHED = ("size", "md5", "sha256")


def _refuse_drift(plan: dict, remote: str, rows: list[dict], when: str) -> None:
    """Stop if any of these rows' Drive objects changed since the plan."""
    changed = _drifted(plan, remote, rows)
    if changed:
        raise ToolError(
            f"{len(changed)} object(s) changed on Drive {when}; run census and plan again",
            3,
        )


def bytecheck(plan: dict, inv: dict, root: Path, remote: str, tmp: Path | None) -> dict:
    """Settle what Drive lists without a size or hash, by download (tmp: None makes one)."""
    if inv.get("generated") != plan.get("inventory_generated"):
        raise ToolError("this inventory is not the one the plan was made from")
    if tmp is not None and (_inside(tmp, root) or _inside(root, tmp)):
        raise ToolError(f"the temporary folder {tmp} must be outside the root {root}")
    rows = plan["rows"]
    by_full: dict[tuple, str] = {}
    for f in sorted(inv["files"], key=lambda f: f["path"]):
        if not f.get("excluded") and f["size"]:
            by_full.setdefault((f["size"], f["md5"], f["sha256"]), f["path"])
    empty_root = {
        f["path"]
        for f in inv["files"]
        if not f.get("excluded") and f["size"] == 0 and f["sha256"] == EMPTY_SHA256
    }
    copy_keys = {}
    for r in rows:
        if r["class"] == "copy" and r["size"] and r["md5"] and r["sha256"]:
            copy_keys.setdefault((r["size"], r["md5"], r["sha256"]), r["dest"])
    taken = Taken(
        [f["path"] for f in inv["files"]]
        + inv.get("links", [])
        + [r["dest"] for r in rows if r["dest"]],
        inv.get("dirs", []),
    )
    todo = sorted(
        (r for r in rows if r["class"] == "needs-byte-check" and not r.get("same_object")),
        key=lambda r: (r["area"], r["path"]),
    )
    if todo:
        _refuse_drift(plan, remote, todo, "since the plan. Nothing was fetched")
    made_tmp = tmp is None or not os.path.lexists(native(tmp))
    if tmp is None:  # made only now, after every check that can stop the run
        tmp = Path(os.path.abspath(tempfile.mkdtemp(prefix="swe-bytecheck-")))
        if _inside(tmp, root) or _inside(root, tmp):
            tmp.rmdir()  # the empty folder just made
            raise ToolError(f"the temporary folder {tmp} must be outside the root {root}")
    tmp.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    try:
        for r in todo:
            local = tmp / r["id"]
            if os.path.lexists(native(local)):
                raise ToolError(f"temporary file {local} already exists; use an empty --tmp")
            cp = run_cmd([*RCLONE, "backend", "copyid", remote, r["id"], str(local)])
            if os.path.lexists(native(local)):
                made.append(native(local))
            if cp.returncode != 0 or not os.path.isfile(native(local)):
                # One object Drive will not hand over must not stop the others: it stays
                # on Drive, listed, and is never deletable.
                if os.path.lexists(native(local)):
                    os.remove(native(local))  # our own partial download
                    made.remove(native(local))
                err = (cp.stderr or "").strip()[-300:] or "no file"
                r.update({"class": "unresolved", "reason": f"download failed: {err}", "dest": ""})
                continue
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
            empty_twin = next((c for c in r["candidates"] if c in empty_root), None)
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
        if made_tmp and tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()  # the empty temporary folder
    if todo:  # an object changed while it was checked keeps no verdict and no hash
        _refuse_drift(plan, remote, todo, "during the byte check. The plan was not changed")
    _mirror(rows)
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
        st = os.stat(src)  # the times only: a read-only flag would pin the staging name
        os.utime(native(dst), ns=(st.st_atime_ns, st.st_mtime_ns))
    except BaseException:
        os.remove(native(dst))  # the partial file this call created (O_EXCL), never another's
        raise


_NO_LINKS_ERRNO = {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK}
_NO_LINKS_WINERROR = {1, 17, 50}  # invalid function, not the same device, not supported


def _links_unsupported(e: OSError) -> bool:
    if getattr(e, "winerror", None) is not None:
        return e.winerror in _NO_LINKS_WINERROR  # type: ignore[attr-defined]
    return e.errno in _NO_LINKS_ERRNO


def _publish(src: Path, dst: Path) -> str:
    """Make a verified staging file appear at dst, which must not exist.

    A hard link fails if dst exists, so nothing is ever written over. Only where the
    volume cannot hold one does an exclusive create take its place; any other failure
    (a file another program holds, say) is raised. The staging name is ours and is
    removed once the bytes are at dst; if that fails, the note says where it stays.
    """
    os.makedirs(native(dst.parent), exist_ok=True)
    try:
        os.link(native(src), native(dst))
    except FileExistsError:
        raise
    except OSError as e:
        if not _links_unsupported(e):
            raise
        _copy_new(native(src), dst)
    try:
        os.remove(native(src))  # our own staging name; the bytes stay at dst
    except OSError as e:
        return f"published; its staging name stays at {src} ({e.strerror or e})"
    return ""


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


def _discard(run: Path) -> list[str]:
    """Remove every file in this run's own staging folder; return any that would not go."""
    left = []
    for dirpath, _dirs, filenames in os.walk(native(run)):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                os.remove(full)  # downloaded by this run into its own new folder
            except OSError:
                left.append(full)
    return left


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
        link = _link_on_path(root, r["dest"])
        if link:
            conflicts.append(
                (r, f"a link or junction on the path ({link}), not a file of the root")
            )
        elif os.path.lexists(native(dest)):
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
    _refuse_drift(plan, remote, todo, "since the plan. Nothing was fetched")
    log = _start_log(log, root)
    print(f"copy: log {log}")
    staging = _run_folder(root, "copy")
    lock = threading.Lock()
    results: list[dict] = []
    fetched: list[tuple[dict, Path, bool, str]] = []  # (row, download, bytes match, why)

    def settle(rec: dict) -> None:
        with lock:  # logged as each object settles, so a crash loses no record
            results.append(rec)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({**rec, "at": utc_now()}) + "\n")

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
        for r, s in pairs:
            if not os.path.isfile(native(s)):
                why = errors.get(r["id"]) or err or "no file"
                settle({"id": r["id"], "dest": r["dest"], "status": f"download failed: {why}"})
                continue
            try:
                ok, why = _matches(s, r)
            except OSError as e:  # say an antivirus holds the fresh download: it alone fails
                rec = {"id": r["id"], "dest": r["dest"]}
                rec["status"] = f"download unreadable: {e.strerror or e}"
                try:
                    os.remove(native(s))  # this run's own download, never published
                except FileNotFoundError:
                    pass  # gone already (quarantined, say)
                except OSError as e2:
                    rec["status"] += f"; kept at {s} ({e2.strerror or e2})"
                settle(rec)
                continue
            with lock:
                fetched.append((r, s, ok, why))

    checked = False
    try:
        chunks = _batches(todo, batch, staging)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            for i, _ in enumerate(ex.map(run, chunks), 1):
                if i % 10 == 0 or i == len(chunks):
                    print(f"copy: {i}/{len(chunks)} batches", file=sys.stderr)
        # The downloads took time: Drive is censused again, and only an object still at
        # its path and still safe to read is published. What changed is not kept,
        # whatever its bytes: it may be a credential now.
        changed: dict[str, str] = {}
        if fetched:
            changed = {r["id"]: why for r, why in _drifted(plan, remote, [f[0] for f in fetched])}
        stuck: dict[str, str] = {}
        for r, s, _ok, _sha in fetched:  # before any publish, so no stop can leave one behind
            if r["id"] in changed:
                try:
                    os.remove(native(s))  # our own download of it, never published
                except OSError as e:
                    stuck[r["id"]] = f"; kept at {s} ({e.strerror or e})"
        checked = True
    finally:
        if not checked:  # stopped before Drive was checked again: nothing fetched is kept
            for f in _discard(staging):
                print(f"  NOT REMOVED  {f}", file=sys.stderr)
            _tidy(staging)
    for r, s, ok, sha in sorted(fetched, key=lambda f: f[0]["dest"]):
        rec = {"id": r["id"], "dest": r["dest"]}
        if r["id"] in changed:
            rec["status"] = (
                f"changed on Drive during the copy ({changed[r['id']]}); not published"
                + stuck.get(r["id"], "")
            )
        elif not ok:
            rec["status"] = f"MISMATCH with Drive's listing: {sha}; kept at {s}"
        elif _link_on_path(root, r["dest"]):
            rec["status"] = f"conflict: a link appeared on the path; kept at {s}"
        else:
            try:
                note = _publish(s, root / r["dest"])
                rec.update({"status": "ok", "sha256": sha})
                if note:
                    rec["note"] = note
            except FileExistsError:
                rec["status"] = f"conflict: the destination appeared; kept at {s}"
            except OSError as e:
                rec["status"] = f"publish failed: {e.strerror or e}; kept at {s}"
        settle(rec)
    _tidy(staging)
    bad = [x for x in results if x["status"] != "ok"]
    print(f"copy: {len(results) - len(bad)} copied and verified; {len(bad)} not")
    for x in bad[:50]:
        print(f"  FAILED  {x['dest']}: {x['status']}")
    for x in [x for x in results if x.get("note")][:50]:
        print(f"  NOTE    {x['dest']}: {x['note']}")
    if any(x["status"].startswith(("MISMATCH", "conflict", "changed on Drive")) for x in bad):
        raise ToolError(
            "a download does not match Drive, a destination appeared, or Drive changed during "
            "the copy; nothing was overwritten",
            3,
        )
    return 4 if bad else 0


def _log_name(base: Path) -> Path:
    """The copy log for this run: base's name with the run's UTC time and a random part.

    So a rerun of the same command, --log included, starts a new log of its own.
    """
    if base.name in ("", ".", ".."):
        raise ToolError(f"--log {base}: name a file")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return base.with_name(f"{base.stem}-{stamp}-{os.urandom(3).hex()}.jsonl")


def _start_log(log: Path, root: Path) -> Path:
    """Create the copy log: always a new file of this run's own (an exclusive create).

    So nothing that was there before is ever read, followed or appended to: a file of any
    kind (a ledger, someone else's, an empty one), a link or a folder at that path refuses
    the run before any download. Each run writes its own log; a rerun starts a new one.
    """
    log = guard_out(log, root)
    if os.path.lexists(native(log)):  # Windows would create where a dangling link points
        raise ToolError(f"{log} exists already; the copy log is always a new file")
    try:
        fd = os.open(native(log), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    except FileExistsError:
        raise ToolError(f"{log} exists already; the copy log is always a new file") from None
    except OSError as e:  # a folder there (Windows may say "access denied"), or no such folder
        raise ToolError(f"cannot start the copy log {log}: {e.strerror or e}") from None
    os.close(fd)
    return log


def _check_home(root: Path, rel: str, r: dict) -> str:
    """'' when root/rel is a file of the root's own holding the row's bytes, else why not."""
    if credential_shaped(rel):
        return "a credential-shaped name, never read"
    link = _link_on_path(root, rel)
    if link:
        return f"a link or junction on the path ({link})"
    if not os.path.isfile(native(root / rel)):
        return "missing"
    good, why = _matches(root / rel, r)
    return "" if good else why


def verify(plan: dict, root: Path) -> int:
    """Re-hash every file the plan relies on: each copy, and each root file a Drive object matches.

    A root file changed or removed since the inventory makes its Drive twin the only
    copy again, so it fails here rather than reaching card 3 as deletable.
    """
    if not same_path(plan["root"], str(root)):
        raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
    pending = [r for r in plan["rows"] if r["class"] == "needs-byte-check"]
    rows = [r for r in plan["rows"] if r["class"] == "copy"]
    ok = missing = bad = 0
    for r in rows:
        why = _check_home(root, r["dest"], r)
        if not why:
            ok += 1
        elif why == "missing":
            missing += 1
            print(f"  MISSING   {r['dest']}")
        else:
            bad += 1
            print(f"  MISMATCH  {r['dest']}: {why}")
    copied = {r["dest"] for r in rows}
    twins: dict[str, dict] = {}
    for r in plan["rows"]:
        if r["class"] in ("redundant", "duplicate") and r["twin"] and r["twin"] not in copied:
            twins.setdefault(r["twin"], r)
    twin_bad = 0
    for rel, r in sorted(twins.items()):
        why = _check_home(root, rel, r)
        if why:
            twin_bad += 1
            print(f"  TWIN      {rel}: {why} (Drive {r['area']}/{r['path']} relies on it)")
    leftovers = []
    staging = _staging(root)
    if os.path.isdir(native(staging)):
        for dirpath, _d, filenames in os.walk(native(staging), onerror=_walk_error):
            leftovers.extend(os.path.join(dirpath, f) for f in filenames)
    partial = []
    legacy = root / LEGACY
    if legacy.is_dir():
        for rel_dir, _d, filenames, _p, _l in _walk(legacy):
            partial.extend(
                f"{rel_dir}/{f}" if rel_dir else f for f in filenames if f.endswith(".partial")
            )
    print(
        f"verify: {len(rows)} copies: {ok} ok, {missing} missing, {bad} mismatched; "
        f"{len(twins)} root files Drive objects match: {len(twins) - twin_bad} ok, "
        f"{twin_bad} missing or changed; {len(pending)} object(s) still need a byte check"
    )
    for p in leftovers[:20]:
        print(f"  staging   {p} (a download kept for inspection; outside the root)")
    for p in partial[:20]:
        print(f"  partial   {LEGACY}/{p} (not data)")
    return 0 if missing == bad == twin_bad == len(pending) == 0 else 1


# ---------------------------------------------------------------- trashlist and recount

# One row per object of the three consolidate areas, and per area folder (card 3, D36).
TRASHLIST_FIELDS = (
    "area",
    "path",
    "id",
    "class",
    "reason",
    "kind",
    "mime",
    "name",
    "parents",
    "owned",
    "size",
    "md5",
    "sha256",
    "modified",
    "twin",
    "twin_size",
    "twin_modified",
    "swe_twin_id",
    "swe_twin_parent",
)
# What recount compares, object by object, between the proof and a fresh census.
COMPARED = ("kind", "mime", "parents", "owned", "name", "size", "md5", "sha256", "modified")
PROOF = "trashlist.json"
PROOF_SUMMARY = "trashlist_summary.json"
KIND_COUNTS = (
    "files with an MD5",
    "files without an MD5",
    "Google-format files",
    "shortcuts",
    "folders",
)


def kind_of(mime: str) -> str:
    if mime == FOLDER:
        return "folder"
    if mime == SHORTCUT:
        return "shortcut"
    if mime.startswith(GOOGLE):
        return "google"
    return "file"


def _kind_count(row: dict) -> str:
    if row["kind"] == "file":
        return "files with an MD5" if row["md5"] else "files without an MD5"
    return {"google": "Google-format files", "shortcut": "shortcuts", "folder": "folders"}[
        row["kind"]
    ]


def _area_objects(area: dict) -> list[tuple[str, dict]]:
    """(path, object) for every object the census lists in an area, its own folder first.

    The folder is built from the census's area record: its id and folder name, the parents
    its root was found with (``root_parents``), whether the account owns it, and its
    modified time. trashlist and recount both build it here, so their id sets can match.
    """
    paths = _area_tree(area)[0]
    top = {
        "id": area["id"],
        "name": area["folder"],
        "mime": FOLDER,
        "size": None,
        "md5": None,
        "sha256": None,
        "parents": list(area.get("root_parents") or []),
        "owned": area.get("root_owned"),
        "trashed": False,  # the census found it with "trashed = false"
        "modified": area.get("root_modified"),
    }
    return [("", top)] + [(paths[o["id"]], o) for o in area["objects"]]


def _proof_row(area_name: str, path: str, o: dict) -> dict:
    row = dict.fromkeys(TRASHLIST_FIELDS)
    row.update(
        {
            "area": area_name,
            "path": path,
            "id": o["id"],
            "reason": "",
            "kind": kind_of(o["mime"]),
            "mime": o["mime"],
            "name": o["name"],
            "parents": list(o["parents"]),
            "owned": o.get("owned"),
            "size": o["size"],
            "md5": o["md5"],
            "sha256": o["sha256"],
            "modified": o.get("modified"),
        }
    )
    return row


def _disk_time(ns: int) -> str:
    sec, frac = divmod(ns, 10**9)
    return dt.datetime.fromtimestamp(sec, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:09d}Z"


def _differ(want: tuple, got: tuple) -> list[str]:
    """Which of size, MD5 and SHA-256 differ: a missing value never matches."""
    return [
        n
        for n, a, b in zip(("size", "MD5", "SHA-256"), want, got, strict=True)
        if a is None or b is None or a != b
    ]


class _Twins:
    """Desktop twins hashed now, each path once: why not, or (size, MD5, SHA-256, modified)."""

    def __init__(self, root: Path):
        self.root = root
        self.seen: dict[str, tuple[str, tuple | None]] = {}

    def get(self, rel: str) -> tuple[str, tuple | None]:
        if rel not in self.seen:
            self.seen[rel] = self._read(rel)
        return self.seen[rel]

    def _read(self, rel: str) -> tuple[str, tuple | None]:
        parts = rel.split("/")
        if not rel or "\\" in rel or ":" in rel or any(p in ("", ".", "..") for p in parts):
            return "not a plain path under the root", None
        if _fold(parts[0]) == _fold(LOGS):
            return f"under the root's {LOGS}/", None
        if credential_shaped(rel):
            return "a credential-shaped name, never read", None
        link = _link_on_path(self.root, rel)
        if link:
            return f"a link or junction on its path ({link})", None
        full = self.root / rel
        if not os.path.isfile(native(full)):
            return "missing", None
        st = os.stat(native(full))
        size, md5, sha = hash_file(full)
        return "", (size, md5, sha, _disk_time(st.st_mtime_ns))


def _my_drive_roots(doc: dict) -> set[str]:
    return {"root"} | ({doc["my_drive_root"]} if doc.get("my_drive_root") else set())


def _check_trashlist_inputs(plan: dict, census_doc: dict, swe_doc: dict, root: Path) -> None:
    """Refuse, before any work, inputs the proof must not be made from."""
    if not same_path(plan["root"], str(root)):
        raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
    pending = [r for r in plan["rows"] if r["class"] == "needs-byte-check"]
    if pending:
        raise ToolError(f"{len(pending)} object(s) of the plan still need a byte check")
    six = sorted((a["id"], a["mode"]) for a in CARD2B_AREAS)
    if sorted((a.get("id"), a.get("mode")) for a in plan["areas"]) != six:
        raise ToolError("the plan's areas are not exactly card 2b's six, by id and mode")
    if census_doc.get("generated") != plan.get("census_generated"):
        raise ToolError("this census is not the one the plan was made from")
    if sorted((a.get("id"), a.get("mode")) for a in census_doc["areas"]) != six:
        raise ToolError("the census's areas are not exactly card 2b's six, by id and mode")
    if [{k: a.get(k) for k in SWE_DATA} for a in swe_doc["areas"]] != [SWE_DATA]:
        raise ToolError(
            f"--swe-census must be a census of swe-data/ alone, made with the one area "
            f"{json.dumps(SWE_DATA)}"
        )
    for doc, label in ((census_doc, "the census"), (swe_doc, "the swe-data/ census")):
        swe = doc.get("swe_data")
        if not doc.get("chain") or not doc.get("my_drive_root") or not isinstance(swe, dict):
            raise ToolError(f"{label} records no chain or swe-data/; make it with `census`")
        if swe.get("id") != SWE_DATA["id"]:
            raise ToolError(f"{label} records swe-data/ as {swe.get('id')}, not {SWE_DATA['id']}")
        if any("sharing" not in x for x in [*doc["chain"], swe, *doc["areas"]]):
            raise ToolError(f"{label} records no sharing for a folder; make it with `census`")
        tops = _my_drive_roots(doc)
        if tops & set(swe.get("parents") or []):
            raise ToolError(f"{label}: swe-data/ has My Drive's root as its parent")
        for a in doc["areas"]:
            if a["mode"] == "consolidate" or a["id"] == SWE_DATA["id"]:
                if tops & set(a.get("root_parents") or []):
                    raise ToolError(
                        f"{label}: area {a['name']}'s folder has My Drive's root as its parent"
                    )


def _common_block(o: dict, read_only: dict[str, str], places: dict[str, list[str]]) -> str:
    """Why an object of a consolidate area, file or folder, cannot go: '' when nothing blocks."""
    if o["id"] in read_only:
        return f"also seen in the read-only area {read_only[o['id']]}"
    if len(places[o["id"]]) > 1:
        return f"seen {len(places[o['id']])} times in the consolidate areas"
    if len(o["parents"]) != 1:
        return f"{len(o['parents'])} parents, not one"
    if o.get("owned") is not True:
        return "not owned by the account" if o.get("owned") is False else "owner not recorded"
    if o.get("trashed"):
        return "in Drive's trash"
    return ""


def _judge_file(
    area: str,
    path: str,
    o: dict,
    read_only: dict[str, str],
    places: dict[str, list[str]],
    plan_rows: dict[str, list[dict]],
    swe: dict[str, list[dict]],
    twins: _Twins,
) -> tuple[str, str, dict]:
    """(class, reason, the twin fields) for one object that is not a folder."""
    kind = kind_of(o["mime"])
    if kind == "shortcut":
        return "blocker", "a shortcut: never followed, never listed", {}
    if kind == "google":
        return "blocker", "a Google-format file: no bytes to compare", {}
    why = _common_block(o, read_only, places)
    if why:
        return "blocker", why, {}
    if o["md5"] == "withheld":
        return "blocker", "hashes withheld by the census: a path the tool must not read", {}
    for key, label in (("size", "size"), ("md5", "MD5"), ("sha256", "SHA-256")):
        if o[key] is None:
            return "blocker", f"Drive lists no {label}", {}
    if credential_shaped(path) or credential_shaped(o["name"]):
        return "blocker", "a credential-shaped name", {}
    rows = plan_rows.get(o["id"], [])
    if not any(r["area"] == area for r in rows):
        return "blocker", "not in the plan", {}
    bad = next((r for r in rows if not r.get("deletable")), None)
    if bad is not None:
        why = f"{bad['class']}: {bad['reason']}" if bad.get("reason") else bad["class"]
        return "blocker", f"not deletable in the plan ({bad['area']}, {why})", {}
    if any(r["area"] == area and r["path"] != path for r in rows):
        return "blocker", "the plan lists it at another path", {}
    if any("download" in (r.get("reason") or "") for r in rows):
        return "blocker", "its hashes in the plan came from a download, which never counts", {}
    want = (o["size"], o["md5"], o["sha256"])
    if any((r["size"], r["md5"], r["sha256"]) != want for r in rows):
        return "blocker", "the plan's size or hashes differ from the census's listed ones", {}
    named = {r.get("twin") or "" for r in rows}
    if len(named) != 1 or "" in named:
        return "blocker", "the plan names no single root twin", {}
    twin = named.pop()
    why, got = twins.get(twin)
    if why or got is None:
        return "blocker", f"root twin {twin}: {why}", {}
    differ = _differ(want, got[:3])
    if differ:
        return "blocker", f"root twin {twin}: its {', '.join(differ)} differ from Drive's", {}
    there = swe.get(twin, [])
    if len(there) != 1:
        return "blocker", f"swe-data/ holds {len(there)} objects at {twin}, not one file", {}
    s = there[0]
    if not is_file(s):
        return "blocker", f"swe-data/'s object at {twin} is not a file", {}
    if len(s["parents"]) != 1:
        return "blocker", f"swe-data/'s copy at {twin} has {len(s['parents'])} parents", {}
    if s["id"] == o["id"]:
        return "blocker", f"swe-data/'s copy at {twin} is this same Drive object", {}
    differ = _differ(want, (s["size"], s["md5"], s["sha256"]))
    if differ:
        return "blocker", f"swe-data/'s copy at {twin}: its {', '.join(differ)} differ", {}
    return (
        "listed",
        "",
        {
            "twin": twin,
            "twin_size": got[0],
            "twin_modified": got[3],
            "swe_twin_id": s["id"],
            "swe_twin_parent": s["parents"][0],
        },
    )


def _swe_index(swe_doc: dict) -> dict[str, list[dict]]:
    """swe-data/'s objects by their path under it."""
    area = swe_doc["areas"][0]
    paths = _area_tree(area)[0]
    out: dict[str, list[dict]] = defaultdict(list)
    for o in area["objects"]:
        out[paths[o["id"]]].append(o)
    return out


def trashlist(plan: dict, census_doc: dict, swe_doc: dict, root: Path) -> tuple[list, list]:
    """Account for every object of the three consolidate areas: (proof rows, area summaries).

    Every object the census lists there, and each area folder, gets exactly one class:
    ``listed`` (a file proven twice: by its desktop twin, hashed now, and by its swe-data/
    twin), ``folder`` (one parent, owned, untrashed, holding only listed files and such
    folders), or ``blocker``, with one reason. What counts comes from the census, never from
    the plan's rows; the plan only says which ids are deletable and names each one's twin.
    An object a read-only area shows is never classed there, and blocks here.
    """
    _check_trashlist_inputs(plan, census_doc, swe_doc, root)
    read_only: dict[str, str] = {}
    for a in census_doc["areas"]:
        if a["mode"] == "read-only":
            read_only.setdefault(a["id"], a["name"])
            for o in a["objects"]:
                read_only.setdefault(o["id"], a["name"])
    cons = [a for a in census_doc["areas"] if a["mode"] == "consolidate"]
    places: dict[str, list[str]] = defaultdict(list)
    for a in cons:
        for _path, o in _area_objects(a):
            places[o["id"]].append(a["name"])
    plan_rows: dict[str, list[dict]] = defaultdict(list)
    for r in plan["rows"]:
        plan_rows[r["id"]].append(r)
    swe = _swe_index(swe_doc)
    twins = _Twins(root)
    rows: list[dict] = []
    summaries: list[dict] = []
    for a in cons:
        area_rows = _classify_area(a, read_only, places, plan_rows, swe, twins)
        rows.extend(area_rows)
        summaries.append(_area_summary(a, area_rows))
    rows.sort(key=lambda r: (r["area"], r["path"], r["id"]))
    return rows, summaries


def _classify_area(
    a: dict,
    read_only: dict[str, str],
    places: dict[str, list[str]],
    plan_rows: dict[str, list[dict]],
    swe: dict[str, list[dict]],
    twins: _Twins,
) -> list[dict]:
    """One consolidate area's proof rows: its files judged first, then its folders bottom-up."""
    objs = _area_objects(a)
    own = {o["id"]: (path, o) for path, o in objs}
    out = {o["id"]: _proof_row(a["name"], path, o) for path, o in objs}
    children: dict[str, list[str]] = defaultdict(list)
    for _path, o in objs:
        for p in o["parents"]:
            if p in own and p != o["id"]:
                children[p].append(o["id"])
    for path, o in objs:
        if kind_of(o["mime"]) != "folder":
            cls, why, extra = _judge_file(
                a["name"], path, o, read_only, places, plan_rows, swe, twins
            )
            out[o["id"]].update({"class": cls, "reason": why, **extra})
    memo: dict[str, tuple[str, str]] = {}

    def folder_class(oid: str, stack: frozenset) -> tuple[str, str]:
        if oid in memo:
            return memo[oid]
        _p, o = own[oid]
        why = _common_block(o, read_only, places)
        if not why and oid in stack:
            why = "a loop of folders"
        if not why:
            for cid in sorted(children[oid], key=lambda c: (own[c][0], c)):
                if kind_of(own[cid][1]["mime"]) == "folder":
                    cls = folder_class(cid, stack | {oid})[0]
                else:
                    cls = out[cid]["class"]
                if cls == "blocker":
                    why = f"holds a blocker: {own[cid][0]}"
                    break
        memo[oid] = ("blocker", why) if why else ("folder", "")
        return memo[oid]

    for _path, o in objs:
        if kind_of(o["mime"]) == "folder":
            cls, why = folder_class(o["id"], frozenset())
            out[o["id"]].update({"class": cls, "reason": why})
    return sorted(out.values(), key=lambda r: (r["path"], r["id"]))


def _area_summary(area: dict, rows: list[dict]) -> dict:
    cls = Counter(r["class"] for r in rows)
    kinds = Counter(_kind_count(r) for r in rows)
    blockers = cls.get("blocker", 0)
    return {
        "name": area["name"],
        "folder_id": area["id"],
        "folder": area["folder"],
        "parents": list(area.get("root_parents") or []),
        "rows": len(rows),
        "classes": {k: cls.get(k, 0) for k in ("listed", "folder", "blocker")},
        "kinds": {k: kinds.get(k, 0) for k in KIND_COUNTS},
        "listed_bytes": sum(r["size"] or 0 for r in rows if r["class"] == "listed"),
        "sharing": area.get("sharing"),
        "sharing_flags": sharing_flags(area.get("sharing")),
        "clear_to_trash_by_hand": "no" if blockers else "yes",
        "blockers": blockers,
    }


def proof_text(rows: list[dict]) -> str:
    """The proof as canonical JSON: keys sorted, no spaces, UTF-8, one newline at the end.

    Its sha256 is the sha256 of these exact bytes (``.encode("utf-8")``), and nothing like a
    timestamp of its making is in it.
    """
    doc = {"kind": "trashlist", "schema": SCHEMA, "rows": rows}
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return ";".join(map(str, v))
    return str(v)


def rows_csv(rows: list[dict]) -> str:
    from io import StringIO

    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=TRASHLIST_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r[k]) for k in TRASHLIST_FIELDS})
    return buf.getvalue()


def sharing_record(doc: dict) -> dict:
    """The census's sharing records: the chain, swe-data/ and each area folder, with flags."""

    def one(x: dict, sharing) -> dict:
        return {
            "id": x["id"],
            "name": x.get("name") or x.get("folder"),
            "parents": x.get("parents") or x.get("root_parents") or [],
            "owned": x.get("owned", x.get("root_owned")),
            "sharing": sharing,
            "flags": sharing_flags(sharing),
        }

    return {
        "census_generated": doc.get("generated"),
        "chain": [one(x, x.get("sharing")) for x in doc.get("chain") or []],
        "swe_data": one(doc["swe_data"], doc["swe_data"].get("sharing")),
        "areas": [{**one(a, a.get("sharing")), "area": a["name"]} for a in doc["areas"]],
    }


def _new_logs_folder(out: Path, root: Path) -> Path:
    """A folder under the root's _logs/ that does not exist yet: a list is never replaced."""
    out = guard_out(out, root)
    logs = root / LOGS
    if not _inside(out, logs) or _file_key(out) == _file_key(logs):
        raise ToolError(f"--out {out}: trashlist writes only into a new folder under {logs}")
    if os.path.lexists(native(out)):
        raise ToolError(f"--out {out} exists already; a list is never replaced")
    return out


def _sha_file(path: Path) -> str:
    with open(native(path), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_trashlist(
    out: Path, rows: list[dict], areas: list[dict], census_doc: dict, inputs: dict
) -> dict:
    """Write the five outputs into ``out``, a new folder made now; return the summary."""
    text = proof_text(rows)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    summary = {
        "kind": "trashlist_summary",
        "schema": SCHEMA,
        "ran": utc_now(),
        "inputs": inputs,
        "areas": areas,
        "chain": [{"name": x["name"], "id": x["id"]} for x in census_doc["chain"]],
        "swe_data": {
            "id": census_doc["swe_data"]["id"],
            "parents": census_doc["swe_data"]["parents"],
        },
        "proof": PROOF,
        "sha256": sha,
    }
    os.makedirs(native(out.parent), exist_ok=True)
    try:
        os.mkdir(native(out))
    except FileExistsError:
        raise ToolError(f"--out {out} exists already; a list is never replaced") from None
    kept = [r for r in rows if r["class"] == "blocker"]
    write_outputs(
        [
            (out / PROOF, text),
            (out / "trashlist.csv", rows_csv(rows)),
            (out / "kept.csv", rows_csv(kept)),
            (out / "sharing.json", json_text(sharing_record(census_doc))),
            (out / PROOF_SUMMARY, json_text(summary)),
        ]
    )
    if _sha_file(out / PROOF) != sha:
        raise ToolError(f"{out / PROOF} does not hold the bytes just written")
    return summary


def trashlist_lines(summary: dict, out: Path) -> list[str]:
    lines = []
    for a in summary["areas"]:
        c, k = a["classes"], a["kinds"]
        lines.append(
            f"trashlist: {a['name']} (folder {a['folder_id']}, in {', '.join(a['parents'])}): "
            f"listed {c['listed']} ({a['listed_bytes']:,} B), folder {c['folder']}, "
            f"blocker {c['blocker']}"
        )
        lines.append("  kinds: " + ", ".join(f"{n} {k[n]}" for n in KIND_COUNTS))
        flags = a["sharing_flags"]
        lines.append("  sharing: " + ("; ".join(flags) if flags else "the owner alone"))
        verdict = a["clear_to_trash_by_hand"]
        extra = f" ({a['blockers']} blocker(s); see kept.csv)" if verdict == "no" else ""
        lines.append(f"  clear to trash by hand: {verdict}{extra}")
    lines.append(f"trashlist: proof {out / PROOF}")
    lines.append(f"trashlist: sha256 {summary['sha256']}")
    return lines


def _later(a, b) -> bool:
    try:
        return dt.datetime.fromisoformat(a) > dt.datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return False


def _children_any(remote: str, ids: list[str]) -> list[dict]:
    """Everything in these folders, trashed or not (no trashed clause)."""
    parents = " or ".join(f"{q_quote(i)} in parents" for i in ids)
    found = drive_query(remote, f"({parents})")
    if found or len(ids) == 1:
        return found
    out: list[dict] = []  # Google issue 149522397, as in _children
    for i in ids:
        out.extend(drive_query(remote, f"{q_quote(i)} in parents"))
    return out


def _recheck_swe(remote: str, listed: list[dict]) -> list[str]:
    """Each listed row's swe-data/ twin, read again in its folder: same id, bytes and place."""
    parents = sorted({r["swe_twin_parent"] for r in listed})
    found: dict[str, list[dict]] = defaultdict(list)
    for i in range(0, len(parents), QUERY_BATCH):
        for obj in _children_any(remote, parents[i : i + QUERY_BATCH]):
            found[obj["id"]].append(obj)
    diffs = []
    for r in listed:
        label = f"{r['area']}/{r['path']}: its swe-data/ twin {r['twin']} ({r['swe_twin_id']})"
        seen = {json.dumps(o, sort_keys=True) for o in found.get(r["swe_twin_id"], [])}
        if not seen:
            diffs.append(f"{label} is gone from its folder {r['swe_twin_parent']}")
            continue
        if len(seen) > 1:
            diffs.append(f"{label} was listed twice, differently")
            continue
        s = slim(found[r["swe_twin_id"]][0])
        if s["trashed"]:
            diffs.append(f"{label} is in Drive's trash")
            continue
        bad = [] if s["parents"] == [r["swe_twin_parent"]] else ["parents"]
        if s["name"] != r["twin"].rsplit("/", 1)[-1]:
            bad.append("name")
        bad += _differ((r["size"], r["md5"], r["sha256"]), (s["size"], s["md5"], s["sha256"]))
        if bad:
            diffs.append(f"{label}: its {', '.join(bad)} changed")
    return diffs


def _norm(key: str, value):
    return sorted(value) if key == "parents" and isinstance(value, list) else value


def _place_of(r: dict) -> str:
    return f"{r['area']}/{r['path']}" if r["path"] else f"{r['area']} (the area folder)"


def recount(
    proof_path: Path,
    summary_path: Path,
    expect: str,
    census_doc: dict,
    names: list[str],
    root: Path,
    remote: str,
) -> dict:
    """On the day of the trash: has anything in the checked areas changed since the proof?

    It reads only. "unchanged: yes" needs every check to hold; the record names each
    difference otherwise. Inputs that do not hold are differences too, and then nothing
    more is compared.
    """
    consolidate = {a["name"]: a for a in DEFAULT_AREAS if a["mode"] == "consolidate"}
    unknown = sorted(set(names) - set(consolidate))
    if unknown:
        raise ToolError(f"--area {unknown}: not one of the consolidate areas {sorted(consolidate)}")
    names = [n for n in consolidate if n in names]
    data = Path(proof_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    try:
        proof = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ToolError(f"{proof_path} is not a trashlist proof") from None
    if not isinstance(proof, dict) or proof.get("kind") != "trashlist":
        raise ToolError(f"{proof_path} is not a trashlist proof")
    if proof.get("schema") != SCHEMA:
        raise ToolError(f"{proof_path} is not a proof of schema {SCHEMA}; make it again")
    summary = load_json(summary_path)
    if not isinstance(summary, dict) or summary.get("kind") != "trashlist_summary":
        raise ToolError(f"{summary_path} is not a trashlist summary")
    record = {
        "kind": "recount",
        "schema": SCHEMA,
        "ran": utc_now(),
        "proof_sha256": sha,
        "expected_sha256": expect,
        "proof_made": summary.get("ran"),
        "census_generated": census_doc.get("generated"),
        "areas": names,
    }
    diffs: list[str] = []
    if sha != expect.strip().lower():
        diffs.append(f"the proof's sha256 is {sha}, not the approved {expect}")
    if summary.get("sha256") != sha:
        diffs.append(f"the summary is not this proof's: it names sha256 {summary.get('sha256')}")
    want = sorted((consolidate[n]["id"], consolidate[n]["mode"]) for n in names)
    if sorted((a.get("id"), a.get("mode")) for a in census_doc["areas"]) != want:
        diffs.append(f"the census's areas are not exactly the areas checked ({', '.join(names)})")
    if not _later(census_doc.get("generated"), summary.get("ran")):
        diffs.append(
            f"the census ({census_doc.get('generated')}) is not later than the proof "
            f"({summary.get('ran')})"
        )
    # An area is known in the proof by its folder's row (path ""), whatever it was named.
    named = {r["id"]: r["area"] for r in proof["rows"] if r["path"] == ""}
    label = {n: named.get(consolidate[n]["id"], n) for n in names}
    rows = [r for r in proof["rows"] if r["area"] in label.values()]
    for n in names:
        mine = [r for r in rows if r["area"] == label[n]]
        nb = sum(1 for r in mine if r["class"] == "blocker")
        if consolidate[n]["id"] not in named:
            diffs.append(f"area {n}: the proof has no rows for it")
        if nb:
            diffs.append(f"area {n}: the proof has {nb} blocker row(s): not clear to trash by hand")
    if diffs:
        return {**record, "unchanged": "no", "differences": diffs}
    by_id = {a["id"]: a for a in census_doc["areas"]}
    for n in names:
        fresh = {
            o["id"]: _proof_row(label[n], path, o)
            for path, o in _area_objects(by_id[consolidate[n]["id"]])
        }
        old = {r["id"]: r for r in rows if r["area"] == label[n]}
        for oid in sorted(set(fresh) - set(old), key=lambda i: (fresh[i]["path"], i)):
            r = fresh[oid]
            diffs.append(f"new since the proof: {_place_of(r)} ({oid}, {r['kind']}, {r['mime']})")
        for oid in sorted(set(old) - set(fresh), key=lambda i: (old[i]["path"], i)):
            diffs.append(f"gone since the proof: {_place_of(old[oid])} ({oid})")
        for oid in sorted(set(old) & set(fresh), key=lambda i: (old[i]["path"], i)):
            changed = [k for k in COMPARED if _norm(k, old[oid][k]) != _norm(k, fresh[oid][k])]
            if changed:
                diffs.append(f"changed: {_place_of(old[oid])} ({oid}): its {', '.join(changed)}")
    listed = [r for r in rows if r["class"] == "listed"]
    twins = _Twins(root)
    for r in listed:  # D36 point 1: "and again before the trash"
        why, got = twins.get(r["twin"])
        if why or got is None:
            diffs.append(f"{r['area']}/{r['path']}: its desktop twin {r['twin']}: {why}")
            continue
        differ = _differ((r["size"], r["md5"], r["sha256"]), got[:3])
        if differ:
            diffs.append(
                f"{r['area']}/{r['path']}: its desktop twin {r['twin']}: its "
                f"{', '.join(differ)} changed"
            )
    if listed:
        diffs.extend(_recheck_swe(remote, listed))
    return {**record, "unchanged": "no" if diffs else "yes", "differences": diffs}


def _write_new(path: Path, text: str) -> None:
    """Write a file that must not exist yet (an exclusive create: never replaces one)."""
    os.makedirs(native(path.parent), exist_ok=True)
    try:
        fd = os.open(
            native(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        )
    except FileExistsError:
        raise ToolError(f"{path} exists already; a recount never replaces a file") from None
    with os.fdopen(fd, "wb") as fh:
        fh.write(text.encode("utf-8"))


# ---------------------------------------------------------------- sweep (a local folder)

# What counts as a data file in a local folder, as scripts/data_manifest.py builds.
SWEEP_SKIP_SUFFIXES = (".py", ".md", ".pyc", ".gitkeep", ".log", ".lock")
SWEEP_SKIP_NAMES = ("__pycache__", "_locks", ".git", "DATA_MANIFEST.json", "_inventory_scan.json")
# Compared by the Windows case rule: on the desktop .GIT is .git and MODEL.PY is code.
_SKIP_NAMES = {_fold(n) for n in SWEEP_SKIP_NAMES}
_SKIP_SUFFIXES = tuple(_fold(x) for x in SWEEP_SKIP_SUFFIXES)


def _count_files(top: str) -> tuple[int, bool]:
    """Files under a folder sweep passes over, for its report: never a stop, never a link."""
    errors: list[OSError] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(top, onerror=errors.append):
        dirnames[:] = [d for d in dirnames if not _is_link(os.path.join(dirpath, d))]
        n += len(filenames)
    return n, bool(errors)


def sweep(source: Path, root: Path, inv: dict, dest: str, dry_run: bool) -> int:
    """Copy the data files of a local folder whose bytes the root lacks to root/dest.

    Nothing is left out silently. A data file whose name Windows or rclone cannot keep
    stops the run before anything is copied, and so does a folder that cannot be
    listed; each credential-shaped file and each link it passes over is named.
    """
    parts = dest.split("/")
    if (
        not AREA_NAME.match(dest)
        or not all(win_safe(p) for p in parts)  # refuses "", "." and ".." too
        or _fold(parts[0]) in {_fold(t) for t in (*LIVE_TREES, LOGS)}
        or credential_shaped(dest)
    ):
        raise ToolError(
            f"refusing destination {dest!r}: sweep writes only to a plain folder outside the "
            "live trees"
        )
    if not source.is_dir():
        raise ToolError(f"source {source} is not a directory")
    for spelled in (os.path.abspath(source), os.path.realpath(source)):
        for part in Path(spelled).parts:
            if any(fnmatch.fnmatchcase(part.lower(), pat) for pat in CREDENTIAL_PATTERNS):
                raise ToolError(
                    f"refusing source {source}: {part!r} is a credential-shaped name, never read"
                )
    if _inside(source, root) or _inside(root, source):
        raise ToolError("the source and the root must not contain each other")
    if not same_path(inv["root"], str(root)):
        raise ToolError(f"the inventory was made for root {inv['root']}, not {root}")
    have: dict[tuple, list[str]] = defaultdict(list)
    for f in inv["files"]:
        if not f.get("excluded") and f["size"]:
            have[(f["size"], f["md5"], f["sha256"])].append(f["path"])
    confirmed: dict[tuple, bool] = {}

    def at_home(key: tuple) -> bool:
        # The inventory may be stale: a root file it lists counts only if it still
        # holds these bytes now.
        if key not in confirmed:
            confirmed[key] = any(
                not _link_on_path(root, p)
                and os.path.isfile(native(root / p))
                and hash_file(root / p) == key
                for p in have.get(key, [])
            )
        return confirmed[key]

    base = native(source)
    todo, done, conflicts, home, code = [], [], [], 0, 0
    passed: list[tuple[str, str]] = []
    unsafe: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base, onerror=_walk_error):
        rel_dir = os.path.relpath(dirpath, base).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        linked = [d for d in dirnames if _is_link(os.path.join(dirpath, d))]
        passed.extend(
            (f"{rel_dir}/{d}" if rel_dir else d, "a linked folder, not followed") for d in linked
        )
        for d in sorted(d for d in dirnames if _fold(d) in _SKIP_NAMES and d not in linked):
            n, unreadable = _count_files(os.path.join(dirpath, d))
            more = "; some of it unreadable" if unreadable else ""
            passed.append(
                (f"{rel_dir}/{d}" if rel_dir else d, f"a {d} folder, not data ({n} files{more})")
            )
        dirnames[:] = sorted(d for d in dirnames if _fold(d) not in _SKIP_NAMES and d not in linked)
        for fn in sorted(filenames):
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            if _fold(fn) in _SKIP_NAMES or _fold(fn).endswith(_SKIP_SUFFIXES):
                code += 1
                continue
            if _is_link(full):
                passed.append((rel, "a link, not followed"))
                continue
            if credential_shaped(rel):
                passed.append((rel, "a credential-shaped name, never read"))
                continue
            if not all(win_safe(p) for p in rel.split("/")):
                unsafe.append(rel)
                continue
            size, md5, sha = hash_file(full)
            if size and at_home((size, md5, sha)):
                home += 1
                continue
            target = root / dest / rel
            if _link_on_path(root, f"{dest}/{rel}"):
                conflicts.append(rel)
            elif os.path.lexists(native(target)):
                got = hash_file(target) if os.path.isfile(native(target)) else None
                (done if got == (size, md5, sha) else conflicts).append(rel)
            else:
                todo.append((rel, full, size, sha))
    print(
        f"sweep: {home + len(todo) + len(done) + len(conflicts) + len(unsafe)} data files in "
        f"{source}; {home} already in the root by bytes; {len(done)} already swept; "
        f"{len(todo)} to copy ({sum(t[2] for t in todo):,} B); {code} code or log files skipped"
    )
    for rel, why in passed:
        print(f"  PASSED    {rel}: {why}")
    for rel in unsafe:
        print(f"  UNSAFE    {rel}: a name Windows or rclone cannot keep")
    for rel in conflicts:
        print(f"  CONFLICT  {dest}/{rel} holds other bytes, or a link is on its path")
    if conflicts:
        raise ToolError(f"{len(conflicts)} destination(s) hold other bytes; nothing was copied", 3)
    if unsafe:
        raise ToolError(
            f"{len(unsafe)} data file(s) have names Windows or rclone cannot keep; rename them "
            "in the source and run again; nothing was copied"
        )
    for rel, _full, size, _sha in todo:
        print(f"  {'would copy' if dry_run else 'copy'}  {dest}/{rel}  ({size:,} B)")
    if dry_run:
        return 0
    if todo:
        # Copy beside the root, re-hash there, then publish with a hard link: a bad copy
        # never reaches the root, and a file that appears meanwhile is never replaced.
        staging = _run_folder(root, "sweep")
        bad: list[str] = []
        failed: list[str] = []
        notes: list[str] = []
        for n, (rel, full, _size, sha) in enumerate(todo):
            tmp = staging / f"{n:06d}"
            _copy_new(full, tmp)
            if hash_file(tmp)[2] != sha:
                bad.append(f"{dest}/{rel}: the copy does not match its source; kept at {tmp}")
                continue
            if _link_on_path(root, f"{dest}/{rel}"):
                bad.append(f"{dest}/{rel}: a link appeared on the path; kept at {tmp}")
                continue
            try:
                note = _publish(tmp, root / dest / rel)
                if note:
                    notes.append(f"{dest}/{rel}: {note}")
            except FileExistsError:
                bad.append(f"{dest}/{rel}: the destination appeared; kept at {tmp}")
            except OSError as e:
                failed.append(f"{dest}/{rel}: publish failed ({e.strerror or e}); kept at {tmp}")
        _tidy(staging)
        for b in bad + failed:
            print(f"  FAILED  {b}")
        for x in notes:
            print(f"  NOTE    {x}")
        if bad:
            raise ToolError(f"{len(bad)} file(s) not copied; nothing was overwritten", 3)
        if failed:
            raise ToolError(f"{len(failed)} file(s) not copied; a rerun resumes", 4)
    print(f"sweep: copied and verified {len(todo)} file(s)")
    return 0


# ---------------------------------------------------------------- sums and filters


def write_sums(root: Path) -> tuple[int, int, str]:
    lines, total = [], 0
    for rel_dir, _d, filenames, dirpath, _l in _walk(root):
        for fn in filenames:
            rel = f"{rel_dir}/{fn}" if rel_dir else fn
            full = os.path.join(dirpath, fn)
            key = _fold(rel)  # on Windows sha256sums is the same file as SHA256SUMS
            ours = key == _fold(SUMS) or (
                key.startswith(_fold(f".{SUMS}.")) and key.endswith(_fold(".tmp"))
            )
            if ours or _is_link(full) or credential_shaped(rel):
                continue
            size, _md5, sha = hash_file(full)
            total += size
            lines.append(f"{sha}  {rel}")
    lines.sort(key=lambda s: s[66:])
    body = ("\n".join(lines) + "\n") if lines else ""
    write_outputs([(root / SUMS, body)])  # our own list, regenerated; never through a link
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
    sp.add_argument("--root", help="the data root, so --out never lands inside it")
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
    sp.add_argument(
        "--log",
        help="JSON-lines log: a file name, or a folder to put it in; each run writes "
        "<name>-<UTC time>-<random>.jsonl there (default name: <plan>_copy)",
    )
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
    sp = sub.add_parser(
        "trashlist", help="account for every object of the consolidate areas; write the proof"
    )
    sp.add_argument("--plan", required=True, help="a whole-area plan over card 2b's six areas")
    sp.add_argument("--census", required=True, help="the census that plan was made from")
    sp.add_argument("--swe-census", required=True, help="a census of swe-data/ alone")
    sp.add_argument("--root")
    sp.add_argument("--out", required=True, help="a new folder under <root>/_logs/")
    sp = sub.add_parser("recount", help="check nothing changed since the proof; reads only")
    sp.add_argument("--proof", required=True, help="the trashlist.json the Operator approved")
    sp.add_argument("--summary", required=True, help="its trashlist_summary.json")
    sp.add_argument("--expect-sha256", required=True, help="the sha256 the Operator approved")
    sp.add_argument("--census", required=True, help="a census of the checked areas, made now")
    sp.add_argument(
        "--area",
        action="append",
        help="a consolidate area to check; may be repeated (default: all three)",
    )
    sp.add_argument("--root")
    sp.add_argument("--remote", required=True)
    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except ToolError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code
    except OSError as e:  # a file operation failed; nothing is overwritten, a rerun resumes
        print(f"ERROR: {e}", file=sys.stderr)
        return 4
    except Exception:  # never exit 1, which means "verify found a difference"
        traceback.print_exc()
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "filters":
        sys.stdout.write(filters_text())
        return 0
    if args.cmd == "census":
        out = guard_out(Path(args.out), args.root)
        _not_an_input([out], [args.areas])
        version = require_rclone()
        doc = census(args.remote, _areas(args.areas))
        doc["rclone"] = version
        locate(args.remote, doc)
        write_outputs([(out, json_text(doc))])
        n = sum(len(a["objects"]) for a in doc["areas"])
        swe = doc["swe_data"]
        print(
            f"census: {len(doc['areas'])} areas, {n} objects; swe-data/ is {swe['id']}, "
            f"in {', '.join(swe['parents'])}"
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
        inv = _load_doc(Path(args.inventory), "inventory")
        out = guard_out(Path(args.out), inv["root"])
        ledger = guard_out(
            Path(args.ledger) if args.ledger else out.with_name(out.stem + "_ledger.csv"),
            inv["root"],
        )
        _not_an_input([out, ledger], [args.census, args.inventory, args.about])
        plan = build_plan(_load_doc(Path(args.census), "census"), inv)
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
            Path(args.source),
            root,
            _load_doc(Path(args.inventory), "inventory"),
            args.dest,
            args.dry_run,
        )
    if args.cmd == "trashlist":
        out = _new_logs_folder(Path(args.out), root)
        plan = _load_doc(Path(args.plan), "plan")
        census_doc = _load_doc(Path(args.census), "census")
        swe_doc = _load_doc(Path(args.swe_census), "census")
        rows, areas = trashlist(plan, census_doc, swe_doc, root)
        inputs = {
            "root": str(root),
            "plan": {"file": Path(args.plan).name, "generated": plan.get("generated")},
            "census": {"file": Path(args.census).name, "generated": census_doc.get("generated")},
            "swe_census": {
                "file": Path(args.swe_census).name,
                "generated": swe_doc.get("generated"),
            },
        }
        summary = write_trashlist(out, rows, areas, census_doc, inputs)
        print("\n".join(trashlist_lines(summary, out)))
        return 0
    if args.cmd == "recount":
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        log = guard_out(root / LOGS / f"recount-{stamp}-{os.urandom(3).hex()}.json", root)
        census_doc = _load_doc(Path(args.census), "census")
        names = args.area or [a["name"] for a in DEFAULT_AREAS if a["mode"] == "consolidate"]
        require_rclone()
        record = recount(
            Path(args.proof),
            Path(args.summary),
            args.expect_sha256,
            census_doc,
            names,
            root,
            args.remote,
        )
        _write_new(log, json_text(record))
        print(f"recount: {', '.join(record['areas'])}: unchanged: {record['unchanged']}")
        for d in record["differences"][:200]:
            print(f"  DIFFERENT  {d}")
        if len(record["differences"]) > 200:
            print(f"  … {len(record['differences']) - 200} more in {log}")
        print(f"recount: verdict written to {log}")
        return 0 if record["unchanged"] == "yes" else 1
    plan_path = Path(args.plan)
    plan = load_json(plan_path)
    if plan.get("kind") != "plan" or plan.get("schema") != SCHEMA:
        raise ToolError(f"{plan_path} is not a plan of schema {SCHEMA}")
    if args.cmd == "bytecheck":
        if not same_path(plan["root"], str(root)):
            raise ToolError(f"the plan was made for root {plan['root']}, not {root}")
        out = guard_out(plan_path, root)  # the plan is rewritten in place
        require_rclone()
        # Absolute, so rclone never reads a colon in it as a remote name.
        tmp = Path(os.path.abspath(args.tmp)) if args.tmp else None
        if tmp is not None and tmp.exists() and any(tmp.iterdir()):
            raise ToolError(f"the temporary folder {tmp} is not empty")
        ledger = out.with_name(out.stem + "_ledger.csv")
        _not_an_input([ledger], [args.inventory])
        bytecheck(plan, _load_doc(Path(args.inventory), "inventory"), root, args.remote, tmp)
        write_outputs([(ledger, ledger_text(plan["rows"])), (out, json_text(plan))])
        print("\n".join(summarize(plan)))
        return 0
    if args.cmd == "copy":
        own = f"{plan_path.stem}_copy.jsonl"
        base = Path(args.log) if args.log else plan_path.with_name(own)
        if args.log and (args.log.endswith(("/", os.sep)) or os.path.isdir(args.log)):
            if not os.path.isdir(args.log):
                raise ToolError(f"--log {args.log}: no such folder")
            base = Path(args.log) / own  # a folder: the log goes in it
        log = _log_name(base)  # a new file each run, so never the plan, its ledger or a rerun's
        guard_out(log, root)  # before the census: a wrong place stops the run at once
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
