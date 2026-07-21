#!/usr/bin/env python3
"""Fetch engine data from Google Drive using data/data_manifest.json.

The Bloomberg data monoliths are NOT stored in git (they live in Google Drive).
This script hydrates a fresh checkout so the engine can run: it reads the
manifest, finds each file in its Drive folder, downloads it to the correct
on-disk path, and verifies the sha256.

Usage
-----
  python scripts/fetch_data.py --check          # verify local files vs manifest (no Drive, no network)
  python scripts/fetch_data.py --served-only    # download ONLY the 10 _FILES the live engine needs
  python scripts/fetch_data.py                  # download all git-tracked data (served + broad_pull)
  python scripts/fetch_data.py --include-deep    # also the deep-history archive (1994-2018 + delisted)

Auth (download only; --check needs none)
----------------------------------------
Provide Google Drive read access via ONE of:
  * A service account:  export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json
    (share the Drive folder with the service-account email, Viewer is enough)
  * An OAuth token file: put it at ~/.config/swe/drive_token.json
Requires:  pip install google-api-python-client google-auth
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "data_manifest.json"
TOKEN = Path.home() / ".config" / "swe" / "drive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST} (regenerate with scripts/gen_data_manifest.py)")
    return json.loads(MANIFEST.read_text())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select(files: list[dict], served_only: bool, include_deep: bool) -> list[dict]:
    out = []
    for f in files:
        is_deep = f["tier"].startswith("deep")
        if is_deep and not include_deep:
            continue
        if served_only and not f["served"]:
            continue
        out.append(f)
    return out


def check(files: list[dict]) -> int:
    ok = missing = stale = 0
    for f in files:
        p = REPO / f["path"]
        tag = "served" if f["served"] else f["tier"]
        if not p.exists():
            print(f"  MISSING  {f['path']}  ({tag})")
            missing += 1
        elif f["sha256"] and sha256(p) != f["sha256"]:
            print(f"  STALE    {f['path']}  (sha256 mismatch)")
            stale += 1
        else:
            ok += 1
    print(f"\n{ok} OK · {missing} missing · {stale} stale  (of {len(files)} checked)")
    return 1 if (missing or stale) else 0


def drive_service():
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit("pip install google-api-python-client google-auth  (needed for download; --check does not)")
    sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa and Path(sa).exists():
        creds = service_account.Credentials.from_service_account_file(sa, scopes=SCOPES)
    elif TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    else:
        sys.exit(
            "No Drive credentials. Set GOOGLE_APPLICATION_CREDENTIALS to a service-account "
            f"JSON (share the folder with its email), or place an OAuth token at {TOKEN}."
        )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_id(svc, folder_id: str, name: str) -> str | None:
    safe = name.replace("'", "\\'")
    q = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = svc.files().list(q=q, fields="files(id,name,size)", pageSize=5).execute()
    hits = resp.get("files", [])
    return hits[0]["id"] if hits else None


def download(svc, file_id: str, dest: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        dl = MediaIoBaseDownload(fh, svc.files().get_media(fileId=file_id), chunksize=8 << 20)
        done = False
        while not done:
            _status, done = dl.next_chunk()
    tmp.replace(dest)


def fetch(files: list[dict]) -> int:
    svc = drive_service()
    got = skipped = failed = 0
    for f in files:
        p = REPO / f["path"]
        if p.exists() and f["sha256"] and sha256(p) == f["sha256"]:
            skipped += 1
            continue
        fid = resolve_id(svc, f["drive_folder"], f["basename"])
        if not fid:
            print(f"  NOT ON DRIVE  {f['path']}  (folder {f['drive_folder']})")
            failed += 1
            continue
        print(f"  ↓ {f['path']}")
        download(svc, fid, p)
        if f["sha256"] and sha256(p) != f["sha256"]:
            print(f"    ⚠ sha256 mismatch after download: {f['path']}")
            failed += 1
        else:
            got += 1
    print(f"\n{got} downloaded · {skipped} already-current · {failed} failed  (of {len(files)})")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify local files against the manifest; no Drive")
    ap.add_argument("--served-only", action="store_true", help="only the 10 _FILES the live engine reads")
    ap.add_argument("--include-deep", action="store_true", help="also the opt-in deep-history archive")
    args = ap.parse_args()

    m = load_manifest()
    files = select(m["files"], args.served_only, args.include_deep)
    print(f"{'CHECK' if args.check else 'FETCH'}: {len(files)} file(s) "
          f"[served_only={args.served_only} include_deep={args.include_deep}]\n")
    return check(files) if args.check else fetch(files)


if __name__ == "__main__":
    raise SystemExit(main())
