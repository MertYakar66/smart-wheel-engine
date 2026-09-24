"""scripts/drive_consolidate.py — the D33 Drive consolidation, against a fake Drive.

Pins:
- the census lists every object by parent id, verifies each area root, and never
  follows a shortcut;
- the plan classifies each case:
  - redundant (size, MD5 and SHA-256 equal a root file; nothing else counts,
    so a git object comes home like any file);
  - copy, with a collision-safe destination: same path with other bytes, a case
    clash, a duplicate name, a Windows-unsafe or reserved name;
  - duplicate (a second Drive copy of the same bytes);
  - needs-byte-check (no SHA-256 on Drive: a missing hash never matches);
  - unresolved: a shortcut, a Google-format file, a credential name, several
    parents;
- a read-only area is never deletable;
- the inventory never reads a credential-shaped file;
- bytecheck settles by download, in a folder outside the root;
- copy never overwrites: other bytes at a destination stop the run before
  anything is copied, and an interrupted run resumes without copying a file
  twice;
- verify re-hashes;
- SHA256SUMS excludes the logs, credential names and itself;
- the rclone filter agrees with the tool's own exclusion rule (when rclone is
  installed).

The fake rclone is a Python script run with this interpreter. Fixture files are
written as bytes, so the tests behave the same on Windows. No git is needed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "drive_consolidate", _REPO / "scripts" / "drive_consolidate.py"
)
dc = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(dc)

FAKE_RCLONE = r"""
import json, os, re, sys
state = json.load(open(os.environ["FAKE_DRIVE"], encoding="utf-8"))
args = sys.argv[1:]
if args[:1] == ["version"]:
    print("rclone v1.68.2")
    sys.exit(0)
if args[:2] == ["backend", "query"]:
    q = args[3]
    parents = set(re.findall(r"'([^']+)' in parents", q))
    name = re.search(r"name = '([^']*)'", q)
    out = [o for o in state["objects"] if parents & set(o.get("parents", []))
           and (name is None or o["name"] == name.group(1))]
    if os.environ.get("FAKE_INCOMPLETE"):
        sys.stderr.write("ERROR : search result INCOMPLETE\n")
    print(json.dumps(out or None))
    sys.exit(0)
if args[:2] == ["backend", "copyid"]:
    pairs = args[3:]
    budget = os.environ.get("FAKE_BUDGET")
    for i in range(0, len(pairs), 2):
        oid, dest = pairs[i], pairs[i + 1]
        if budget:
            left = int(open(budget).read())
            if left <= 0:
                sys.stderr.write("ERROR : simulated interruption\n")
                sys.exit(1)
            open(budget, "w").write(str(left - 1))
        data = bytes.fromhex(state["content"][oid])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as fh:
            fh.write(oid + "\n")
        squat = os.environ.get("FAKE_SQUAT")
        if squat and not os.path.exists(squat):
            os.makedirs(os.path.dirname(squat), exist_ok=True)
            with open(squat, "wb") as fh:
                fh.write(b"appeared mid-run\n")
    sys.exit(0)
sys.exit(2)
"""


class FakeDrive:
    def __init__(self) -> None:
        self.objects: list[dict] = []
        self.content: dict[str, str] = {}
        self.n = 0

    def _id(self) -> str:
        self.n += 1
        return f"id{self.n:04d}"

    def folder(self, name: str, parent: str, oid: str | None = None) -> str:
        oid = oid or self._id()
        self.objects.append({"id": oid, "name": name, "mimeType": dc.FOLDER, "parents": [parent]})
        return oid

    def file(
        self,
        name,
        parent,
        data: bytes,
        *,
        sha=True,
        md5=None,
        parents=None,
        mime="text/csv",
        size=None,
    ):
        oid = self._id()
        obj = {
            "id": oid,
            "name": name,
            "mimeType": mime,
            "parents": parents or [parent],
            "size": str(len(data) if size is None else size),
            "md5Checksum": md5 or hashlib.md5(data).hexdigest(),
            "createdTime": "2026-07-01T00:00:00Z",
            "modifiedTime": "2026-07-02T00:00:00Z",
        }
        if sha:
            obj["sha256Checksum"] = hashlib.sha256(data).hexdigest()
        self.objects.append(obj)
        self.content[oid] = data.hex()
        return oid

    def special(self, name, parent, mime, **extra) -> str:
        oid = self._id()
        self.objects.append(
            {"id": oid, "name": name, "mimeType": mime, "parents": [parent], **extra}
        )
        return oid

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"objects": self.objects, "content": self.content}), encoding="utf-8"
        )


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A root, a fake Drive with three areas, and the fake rclone."""
    monkeypatch.setattr(dc, "RETRY_SLEEP", 0.0)
    root = tmp_path / "root"
    files = {
        "data/x.csv": b"x-bytes\n",
        "data/nosha_twin.csv": b"nosha twin\n",
        "data_archive/drive-legacy/A/data/conflict.csv": b"LOCAL DIFFERENT\n",
        "data_archive/drive-legacy/A/markers/done.flag": b"",
        "data_processed/ibkr/flex_credentials.json": b'{"t":"s"}',
        "_logs/run.log": b"log\n",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    d = FakeDrive()
    d.folder("A", "root", "areaA")
    d.folder("parentB", "root", "pB")
    d.folder("ro", "pB", "areaB")
    d.folder("G", "root", "areaG")
    data = d.folder("data", "areaA")
    other = d.folder("other", "areaA")
    ibkr = d.folder("ibkr", "areaA")
    markers = d.folder("markers", "areaA")
    ids = {
        "x": d.file("x.csv", data, b"x-bytes\n"),
        "new": d.file("new.csv", data, b"only on drive\n"),
        "conflict": d.file("conflict.csv", data, b"DRIVE VERSION\n"),
        "nosha": d.file("nosha.csv", data, b"nosha twin\n", sha=False),
        "nosha_unique": d.file("nosha_unique.csv", data, b"no sha, unique\n", sha=False),
        "lying": d.file(
            "lying.csv",
            data,
            b"NOSHA TWIN\n",
            sha=False,
            md5=hashlib.md5(b"nosha twin\n").hexdigest(),
        ),
        "dup1": d.file("dup1.csv", data, b"shared\n"),
        "dup2": d.file("dup2.csv", other, b"shared\n"),
        "Case": d.file("Case.csv", data, b"upper\n"),
        "case": d.file("case.csv", data, b"lower\n"),
        "bad": d.file("bad:name.csv", data, b"bad name\n"),
        "con": d.file("CON.csv", data, b"reserved\n"),
        "twin_a": d.file("twin.csv", data, b"twin a\n"),
        "twin_b": d.file("twin.csv", data, b"twin b\n"),
        "cred": d.file("flex_credentials.json", ibkr, b'{"t":"drive"}'),
        "done": d.file("done.flag", markers, b""),
        "newflag": d.file("new.flag", markers, b""),
        "multi": d.file("multi.csv", data, b"two parents\n", parents=[data, "elsewhere"]),
        "vendor": d.file("vendor.csv", "areaB", b"vendor only\n"),
        "x_ro": d.file("x_copy.csv", "areaB", b"x-bytes\n"),
    }
    outside = d.folder("personal", "root")
    d.file("private.txt", outside, b"never listed\n")
    ids["shortcut"] = d.special(
        "link",
        data,
        dc.SHORTCUT,
        shortcutDetails={"targetId": outside, "targetMimeType": dc.FOLDER},
    )
    ids["gdoc"] = d.special("notes", data, "application/vnd.google-apps.document")
    gitdir = d.folder(".git", "areaG")
    objects = d.folder("objects", gitdir)
    ids["gitcfg"] = d.file("config", gitdir, b"[core]\n")
    ids["loose"] = d.file("c" * 38, d.folder("ab", objects), b"zlib-ish")
    ids["pack"] = d.file("pack-" + "d" * 40 + ".pack", d.folder("pack", objects), b"PACK...")

    state = tmp_path / "drive.json"
    d.save(state)
    script = tmp_path / "fake_rclone.py"
    script.write_bytes(FAKE_RCLONE.encode())
    log = tmp_path / "copyid.log"
    log.write_bytes(b"")
    monkeypatch.setenv("FAKE_DRIVE", str(state))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(dc, "RCLONE", [sys.executable, str(script)])
    areas = [
        {"name": "A", "id": "areaA", "parent": "root", "folder": "A", "mode": "consolidate"},
        {"name": "B/ro", "id": "areaB", "parent": "pB", "folder": "ro", "mode": "read-only"},
        {"name": "G", "id": "areaG", "parent": "root", "folder": "G", "mode": "consolidate"},
    ]
    areas_path = tmp_path / "areas.json"
    areas_path.write_text(json.dumps(areas), encoding="utf-8")
    return {
        "tmp": tmp_path,
        "root": root,
        "ids": ids,
        "areas": areas_path,
        "log": log,
        "drive": d,
        "state": state,
    }


def _plan(w) -> dict:
    t = w["tmp"]
    assert (
        dc.main(
            ["census", "--remote", "fake:", "--areas", str(w["areas"]), "--out", str(t / "c.json")]
        )
        == 0
    )
    assert dc.main(["inventory", "--root", str(w["root"]), "--out", str(t / "i.json")]) == 0
    args = [
        "plan",
        "--census",
        str(t / "c.json"),
        "--inventory",
        str(t / "i.json"),
        "--out",
        str(t / "p.json"),
    ]
    assert dc.main(args) == 0
    return json.loads((t / "p.json").read_text(encoding="utf-8"))


def _by_id(plan: dict) -> dict:
    return {r["id"]: r for r in plan["rows"]}


def _bytecheck(w) -> dict:
    t = w["tmp"]
    args = ["bytecheck", "--plan", str(t / "p.json"), "--inventory", str(t / "i.json")]
    args += ["--root", str(w["root"]), "--remote", "fake:", "--tmp", str(t / "bc")]
    assert dc.main(args) == 0
    return json.loads((t / "p.json").read_text(encoding="utf-8"))


def _root_hashes(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_census_lists_every_object_and_never_follows_a_shortcut(world):
    t = world["tmp"]
    assert (
        dc.main(
            [
                "census",
                "--remote",
                "fake:",
                "--areas",
                str(world["areas"]),
                "--out",
                str(t / "c.json"),
            ]
        )
        == 0
    )
    doc = json.loads((t / "c.json").read_text(encoding="utf-8"))
    names = {o["name"] for a in doc["areas"] for o in a["objects"]}
    assert "private.txt" not in names and "personal" not in names  # behind the shortcut
    assert "link" in names and "notes" in names and "flex_credentials.json" in names
    assert doc["swe_data"] == []
    a = next(a for a in doc["areas"] if a["name"] == "A")
    link = next(o for o in a["objects"] if o["name"] == "link")
    assert link["mime"] == dc.SHORTCUT and link["target_mime"] == dc.FOLDER


def test_census_refuses_an_area_whose_id_does_not_match(world, tmp_path):
    bad = json.loads(world["areas"].read_text(encoding="utf-8"))
    bad[0]["id"] = "not-the-folder"
    p = tmp_path / "bad_areas.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    assert (
        dc.main(
            ["census", "--remote", "fake:", "--areas", str(p), "--out", str(tmp_path / "c.json")]
        )
        == 2
    )


def test_census_treats_an_incomplete_search_as_an_error(world, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_INCOMPLETE", "1")
    args = [
        "census",
        "--remote",
        "fake:",
        "--areas",
        str(world["areas"]),
        "--out",
        str(tmp_path / "c.json"),
    ]
    assert dc.main(args) == 2


def test_plan_classifies_every_case(world):
    plan = _plan(world)
    rows, ids = _by_id(plan), world["ids"]
    cls = {k: rows[v]["class"] for k, v in ids.items()}
    assert cls["x"] == "redundant" and rows[ids["x"]]["twin"] == "data/x.csv"
    assert (
        cls["new"] == "copy"
        and rows[ids["new"]]["dest"] == "data_archive/drive-legacy/A/data/new.csv"
    )
    # Same path as a root file with other bytes: never over it.
    assert cls["conflict"] == "copy"
    assert (
        rows[ids["conflict"]]["dest"]
        == f"data_archive/drive-legacy/A/_conflicts/{ids['conflict']}/conflict.csv"
    )
    # No SHA-256 on Drive: a missing hash never counts as a match.
    assert cls["nosha"] == "needs-byte-check" and cls["lying"] == "needs-byte-check"
    assert cls["nosha_unique"] == "copy"
    assert cls["dup1"] == "copy" and cls["dup2"] == "duplicate"
    assert rows[ids["dup2"]]["twin"] == rows[ids["dup1"]]["dest"]
    # Case clash: one keeps its path, the other goes to _conflicts.
    assert {rows[ids["Case"]]["dest"], rows[ids["case"]]["dest"]} == {
        "data_archive/drive-legacy/A/data/Case.csv",
        f"data_archive/drive-legacy/A/_conflicts/{ids['case']}/case.csv",
    }
    assert rows[ids["bad"]]["dest"] == f"data_archive/drive-legacy/A/data/_drive-{ids['bad']}"
    assert rows[ids["con"]]["dest"] == f"data_archive/drive-legacy/A/data/_drive-{ids['con']}"
    # A duplicate name in one folder: both come home, neither is deletable.
    assert cls["twin_a"] == cls["twin_b"] == "copy"
    assert rows[ids["twin_a"]]["dest"] != rows[ids["twin_b"]]["dest"]
    assert not rows[ids["twin_a"]]["deletable"] and not rows[ids["twin_b"]]["deletable"]
    assert cls["shortcut"] == cls["gdoc"] == cls["cred"] == cls["multi"] == "unresolved"
    assert "credential" in rows[ids["cred"]]["reason"] and rows[ids["cred"]]["dest"] == ""
    assert cls["done"] == "redundant" and cls["newflag"] == "copy"
    # The read-only area: copied from, never deletable.
    assert cls["vendor"] == "copy" and not rows[ids["vendor"]]["deletable"]
    assert cls["x_ro"] == "redundant" and not rows[ids["x_ro"]]["deletable"]
    assert rows[ids["vendor"]]["dest"] == "data_archive/drive-legacy/B/ro/vendor.csv"
    # Only bytes count: the old .git upload's files come home like any other file.
    assert cls["gitcfg"] == cls["loose"] == cls["pack"] == "copy"
    assert rows[ids["loose"]]["dest"] == f"data_archive/drive-legacy/G/.git/objects/ab/{'c' * 38}"
    # The root's credential file is listed as excluded, never hashed.
    assert plan["excluded"] == ["data_processed/ibkr/flex_credentials.json"]
    assert all(
        not r["dest"] or r["dest"].startswith("data_archive/drive-legacy/") for r in plan["rows"]
    )


def test_plan_is_deterministic(world):
    a = _plan(world)["rows"]
    b = _plan(world)["rows"]
    assert a == b


def test_inventory_never_reads_a_credential_file(world, monkeypatch):
    seen = []
    real = dc.hash_file

    def spy(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(dc, "hash_file", spy)
    doc = dc.inventory(world["root"])
    assert not any("credential" in s for s in seen)
    cred = next(f for f in doc["files"] if f["path"].endswith("flex_credentials.json"))
    assert cred["excluded"] and "sha256" not in cred
    assert not any(f["path"].startswith("_logs/") for f in doc["files"])


def test_bytecheck_settles_by_download_and_never_trusts_a_missing_hash(world):
    _plan(world)
    rows, ids = _by_id(_bytecheck(world)), world["ids"]
    assert (
        rows[ids["nosha"]]["class"] == "redundant"
        and rows[ids["nosha"]]["twin"] == "data/nosha_twin.csv"
    )
    # Drive's MD5 said "same as the twin", the bytes say otherwise.
    assert rows[ids["lying"]]["class"] == "unresolved" and rows[ids["lying"]]["dest"] == ""
    assert not (world["tmp"] / "bc").exists()  # the temporary folder is gone
    downloaded = world["log"].read_text(encoding="utf-8").split()
    assert ids["cred"] not in downloaded


def test_bytecheck_refuses_a_temporary_folder_inside_the_root(world):
    t = world["tmp"]
    _plan(world)
    args = ["bytecheck", "--plan", str(t / "p.json"), "--inventory", str(t / "i.json")]
    args += ["--root", str(world["root"]), "--remote", "fake:", "--tmp", str(world["root"] / "tmp")]
    assert dc.main(args) == 2


def test_copy_brings_everything_home_and_overwrites_nothing(world):
    t, root = world["tmp"], world["root"]
    _plan(world)
    plan = _bytecheck(world)
    before = _root_hashes(root)
    args = [
        "copy",
        "--plan",
        str(t / "p.json"),
        "--root",
        str(root),
        "--remote",
        "fake:",
        "--reserve-gb",
        "0",
    ]
    assert dc.main(args) == 0
    after = _root_hashes(root)
    assert all(after[k] == v for k, v in before.items())  # nothing existing changed
    copies = [r for r in plan["rows"] if r["class"] == "copy"]
    assert set(after) - set(before) == {r["dest"] for r in copies}
    assert (
        root / "data_archive/drive-legacy/A/data/conflict.csv"
    ).read_bytes() == b"LOCAL DIFFERENT\n"
    conflict = (
        root / f"data_archive/drive-legacy/A/_conflicts/{world['ids']['conflict']}/conflict.csv"
    )
    assert conflict.read_bytes() == b"DRIVE VERSION\n"
    assert world["ids"]["cred"] not in world["log"].read_text(encoding="utf-8").split()
    assert dc.main(["verify", "--plan", str(t / "p.json"), "--root", str(root)]) == 0
    # A rerun finds everything done and fetches nothing.
    n = len(world["log"].read_text(encoding="utf-8").split())
    assert dc.main(args) == 0
    assert len(world["log"].read_text(encoding="utf-8").split()) == n


def test_copy_stops_before_copying_when_a_destination_holds_other_bytes(world):
    t, root = world["tmp"], world["root"]
    plan = _plan(world)
    row = next(r for r in plan["rows"] if r["id"] == world["ids"]["new"])
    squatter = root / row["dest"]
    squatter.parent.mkdir(parents=True, exist_ok=True)
    squatter.write_bytes(b"someone else's bytes\n")
    args = [
        "copy",
        "--plan",
        str(t / "p.json"),
        "--root",
        str(root),
        "--remote",
        "fake:",
        "--reserve-gb",
        "0",
    ]
    assert dc.main(args) == 3
    assert squatter.read_bytes() == b"someone else's bytes\n"
    assert world["log"].read_text(encoding="utf-8") == ""


def test_an_interrupted_copy_resumes_without_copying_twice(world, monkeypatch):
    t, root = world["tmp"], world["root"]
    _plan(world)
    budget = t / "budget"
    budget.write_text("2", encoding="utf-8")
    monkeypatch.setenv("FAKE_BUDGET", str(budget))
    args = ["copy", "--plan", str(t / "p.json"), "--root", str(root), "--remote", "fake:"]
    args += ["--reserve-gb", "0", "--batch", "1", "--jobs", "1"]
    assert dc.main(args) == 4
    assert len(world["log"].read_text(encoding="utf-8").split()) == 2
    monkeypatch.delenv("FAKE_BUDGET")
    assert dc.main(args) == 0
    fetched = world["log"].read_text(encoding="utf-8").split()
    assert len(fetched) == len(set(fetched))  # no object fetched twice
    assert dc.main(["verify", "--plan", str(t / "p.json"), "--root", str(root)]) == 0


def test_copy_rechecks_each_destination_just_before_copying(world, monkeypatch):
    t, root = world["tmp"], world["root"]
    plan = _plan(world)
    copies = sorted((r for r in plan["rows"] if r["class"] == "copy"), key=lambda r: r["dest"])
    # The fake writes a file at the last destination right after the first copy,
    # as if something else wrote there mid-run.
    squat = root / copies[-1]["dest"]
    monkeypatch.setenv("FAKE_SQUAT", str(squat))
    args = ["copy", "--plan", str(t / "p.json"), "--root", str(root), "--remote", "fake:"]
    args += ["--reserve-gb", "0", "--batch", "1", "--jobs", "1"]
    assert dc.main(args) == 3
    assert squat.read_bytes() == b"appeared mid-run\n"  # never overwritten
    assert copies[-1]["id"] not in world["log"].read_text(encoding="utf-8").split()


def test_copy_refuses_a_destination_outside_the_legacy_folder(world):
    t = world["tmp"]
    plan = _plan(world)
    for r in plan["rows"]:
        if r["class"] == "copy":
            r["dest"] = "data/bloomberg/sp500_ohlcv.csv"
            break
    (t / "p.json").write_text(json.dumps(plan), encoding="utf-8")
    args = ["copy", "--plan", str(t / "p.json"), "--root", str(world["root"]), "--remote", "fake:"]
    assert dc.main(args) == 2


def test_copy_refuses_a_plan_made_for_another_root(world, tmp_path):
    _plan(world)
    other = tmp_path / "other-root"
    other.mkdir()
    args = [
        "copy",
        "--plan",
        str(world["tmp"] / "p.json"),
        "--root",
        str(other),
        "--remote",
        "fake:",
    ]
    assert dc.main(args) == 2


def test_sums_lists_the_root_but_not_logs_credentials_or_itself(world):
    root = world["root"]
    n, total, digest = dc.write_sums(root)
    body = (root / dc.SUMS).read_bytes()
    assert hashlib.sha256(body).hexdigest() == digest
    lines = body.decode("utf-8").splitlines()
    paths = [line[66:] for line in lines]
    assert paths == sorted(paths) and n == len(lines)
    assert "data/x.csv" in paths
    assert not any(p.startswith("_logs/") or "credential" in p or p == dc.SUMS for p in paths)
    assert all(line[64:66] == "  " for line in lines)
    dc.write_sums(root)  # regenerating replaces the list and does not list it
    assert dc.SUMS not in (root / dc.SUMS).read_text(encoding="utf-8")
    rclone = shutil.which("rclone")
    if rclone:
        cp = subprocess.run(
            [
                rclone,
                "checksum",
                "sha256",
                str(root / dc.SUMS),
                str(root),
                "--exclude",
                "/SHA256SUMS",
            ]
            + ["--exclude", "/_logs/**", "--exclude", "*credential*"],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr


def test_the_rclone_filter_agrees_with_the_tool(tmp_path):
    rclone = shutil.which("rclone")
    if not rclone:
        pytest.skip("rclone is not installed here")
    samples = [
        "data/x.csv",
        "data/credit_risk.csv",
        "data/keys.csv",
        "data/pass.csv",
        "data_processed/ibkr/flex_credentials.json",
        "data_processed/ibkr/FLEX_CREDENTIALS.JSON",
        "cfg/.env",
        "cfg/.env.local",
        "cfg/server.pem",
        "secrets/inner/file.csv",
        "a/api_token.txt",
        "_logs/run.log",
        "deep/_logs/kept.log",
        "SHA256SUMS",
    ]
    root = tmp_path / "r"
    for s in samples:
        p = root / s
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    flt = tmp_path / "exclude.txt"
    flt.write_text(dc.filters_text(), encoding="utf-8")
    cp = subprocess.run(
        [
            rclone,
            "lsf",
            "-R",
            "--files-only",
            "--exclude-from",
            str(flt),
            "--ignore-case",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    kept_by_rclone = set(cp.stdout.split())
    kept_by_tool = {
        s for s in samples if not s.startswith("_logs/") and not dc.credential_shaped(s)
    }
    assert kept_by_rclone == kept_by_tool


def test_windows_safe_names():
    assert dc.win_safe("ticker=AAPL") and dc.win_safe("a.csv.gz")
    for bad in (
        "a:b",
        "CON",
        "con.txt",
        "LPT1.log",
        "trailing.",
        "trailing ",
        "",
        "..",
        "a|b",
        "tab\t",
    ):
        assert not dc.win_safe(bad), bad


def test_credential_shaped_matches_any_component():
    assert dc.credential_shaped("x/Secrets/y.csv")
    assert dc.credential_shaped("ibkr/flex_credentials.json")
    assert not dc.credential_shaped("data/bloomberg/sp500_credit_risk.csv")
    assert not dc.credential_shaped("data/keys.csv")


def test_the_default_areas_match_the_inventory_record():
    text = (_REPO / "docs" / "DATA_INVENTORY.md").read_text(encoding="utf-8")
    for a in dc.DEFAULT_AREAS:
        assert a["id"] in text, a["name"]
    assert {a["mode"] for a in dc.DEFAULT_AREAS} == {"consolidate", "read-only"}
    assert os.path.basename(dc.LEGACY) == "drive-legacy"
