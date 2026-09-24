"""scripts/drive_consolidate.py — the D33 Drive consolidation, against a fake Drive.

Pins:
- the census lists every object by parent id, verifies each area root, never
  follows a shortcut, recovers when Drive answers a query over several parents
  with nothing, and stops when its totals differ from ``rclone size``;
- the plan classifies each case:
  - redundant only when size, MD5 and SHA-256 all equal a root file (an empty file
    only when an empty file sits at its destination);
  - copy, with a collision-safe destination: same path with other bytes, a case
    clash, a duplicate name, a Windows-unsafe, reserved or rclone-rewritten name;
  - duplicate (a second Drive copy of the same bytes);
  - needs-byte-check (a missing hash never matches);
  - unresolved: a shortcut, a Google-format file, a credential name (``.git/config``
    and ``rclone.conf`` included), several parents;
- a second round after copying plans nothing new;
- a read-only area is never deletable, and a Drive id is deletable only if every row
  naming it is;
- the inventory never reads a credential-shaped file;
- bytecheck settles by download, in a folder outside the root;
- copy:
  - refuses while byte checks are pending;
  - never overwrites: other bytes at a destination stop the run before anything is
    copied, and a file that appears mid-run is never replaced;
  - publishes only re-hashed downloads;
  - survives one undownloadable object;
  - resumes after an interruption without copying a file twice;
  - runs batches in parallel;
  - handles long paths;
- a copy that fails part-way leaves no partial file;
- verify re-hashes and fails while byte checks are pending;
- sweep copies a local folder's missing bytes, and only those, through the staging
  folder: a bad copy never reaches the root, and a file that appears mid-run is
  never replaced;
- outputs never land inside the root;
- SHA256SUMS excludes the logs, credential names and itself;
- the rclone filter agrees with the tool's own exclusion rule (when rclone is
  installed).

The fake rclone is a Python script run with this interpreter. It omits a zero
``size`` and stops a ``copyid`` at its first failing pair, as rclone does. Fixture
files are written as bytes, so the tests behave the same on Windows. No git is
needed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
objs = state["objects"]
args = sys.argv[1:]

def wire(o):
    o = dict(o)
    if o.get("size") == "0":
        del o["size"]  # rclone's JSON omits a zero size (size,omitempty)
    return o

if args[:1] == ["version"]:
    print("rclone v1.68.2")
    sys.exit(0)
if args[:2] == ["backend", "query"]:
    q = args[3]
    parents = set(re.findall(r"'([^']+)' in parents", q))
    name = re.search(r"name = '([^']*)'", q)
    if os.environ.get("FAKE_MULTI_EMPTY") and len(parents) > 1:
        print("null")  # Google issue 149522397
        sys.exit(0)
    out = [wire(o) for o in objs if parents & set(o.get("parents", []))
           and (name is None or o["name"] == name.group(1))]
    if os.environ.get("FAKE_INCOMPLETE"):
        sys.stderr.write("ERROR : search result INCOMPLETE\n")
    print(json.dumps(out or None))
    sys.exit(0)
if args[:1] == ["size"]:
    top = args[args.index("--drive-root-folder-id") + 1]
    folders, frontier = {top}, [top]
    while frontier:
        f = frontier.pop()
        for o in objs:
            if (f in o.get("parents", []) and o["mimeType"] == "application/vnd.google-apps.folder"
                    and o["id"] not in folders):
                folders.add(o["id"])
                frontier.append(o["id"])
    count = size = 0
    for o in objs:
        if o["mimeType"].startswith("application/vnd.google-apps."):
            continue
        k = sum(1 for p in o.get("parents", []) if p in folders)
        count += k
        size += k * int(o.get("size", "0"))
    count += int(os.environ.get("FAKE_SIZE_OFFSET", "0"))
    print(json.dumps({"count": count, "bytes": size, "sizeless": 0}))
    sys.exit(0)
if args[:2] == ["backend", "copyid"]:
    pairs = args[3:]
    budget = os.environ.get("FAKE_BUDGET")
    fail = set(filter(None, os.environ.get("FAKE_FAIL_IDS", "").split(",")))
    for i in range(0, len(pairs), 2):
        oid, dest = pairs[i], pairs[i + 1]
        if oid in fail:
            sys.stderr.write(f'ERROR : failed copying "{oid}" to "{dest}": cannotDownloadAbusiveFile\n')
            sys.exit(1)
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

    def folder(self, name: str, parent: str, oid: str | None = None, parents=None) -> str:
        oid = oid or self._id()
        self.objects.append(
            {"id": oid, "name": name, "mimeType": dc.FOLDER, "parents": parents or [parent]}
        )
        return oid

    def file(self, name, parent, data: bytes, *, sha=True, md5=None, sha256=None, parents=None):
        oid = self._id()
        obj = {
            "id": oid,
            "name": name,
            "mimeType": "text/csv",
            "parents": parents or [parent],
            "size": str(len(data)),
            "md5Checksum": md5 or hashlib.md5(data).hexdigest(),
            "createdTime": "2026-07-01T00:00:00Z",
            "modifiedTime": "2026-07-02T00:00:00Z",
        }
        if sha:
            obj["sha256Checksum"] = sha256 or hashlib.sha256(data).hexdigest()
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


def _env(tmp_path, monkeypatch, drive: FakeDrive, areas: list[dict], files: dict[str, bytes]):
    """Write the root, the fake Drive and the areas; point the tool at the fake rclone."""
    monkeypatch.setattr(dc, "RETRY_SLEEP", 0.0)
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)
    root = tmp_path / "root"
    root.mkdir()
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    state = tmp_path / "drive.json"
    drive.save(state)
    script = tmp_path / "fake_rclone.py"
    script.write_bytes(FAKE_RCLONE.encode())
    log = tmp_path / "copyid.log"
    log.write_bytes(b"")
    monkeypatch.setenv("FAKE_DRIVE", str(state))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(dc, "RCLONE", [sys.executable, str(script)])
    areas_path = tmp_path / "areas.json"
    areas_path.write_text(json.dumps(areas), encoding="utf-8")
    return {"tmp": tmp_path, "root": root, "areas": areas_path, "log": log, "drive": drive}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A root and a fake Drive with three areas, holding every case the plan must tell apart."""
    files = {
        "data/x.csv": b"x-bytes\n",
        "data/nosha_twin.csv": b"nosha twin\n",
        "data_archive/drive-legacy/A/data/conflict.csv": b"LOCAL DIFFERENT\n",
        "data_archive/drive-legacy/A/markers/done.flag": b"",
        "data_processed/ibkr/flex_credentials.json": b'{"t":"s"}',
        "_logs/run.log": b"log\n",
    }
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
        "fullwidth": d.file("a：b.csv", data, b"rclone would rewrite this name\n"),
        "twin_a": d.file("twin.csv", data, b"twin a\n"),
        "twin_b": d.file("twin.csv", data, b"twin b\n"),
        "cred": d.file("flex_credentials.json", ibkr, b'{"t":"drive"}'),
        "rclone_conf": d.file("rclone.conf", ibkr, b"[gdrive]\n"),
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
    ids["gitcfg"] = d.file("config", gitdir, b"[remote]\n")
    ids["loose"] = d.file("c" * 38, d.folder("ab", objects), b"zlib-ish")
    ids["pack"] = d.file("pack-" + "d" * 40 + ".pack", d.folder("pack", objects), b"PACK...")
    areas = [
        {"name": "A", "id": "areaA", "parent": "root", "folder": "A", "mode": "consolidate"},
        {"name": "B/ro", "id": "areaB", "parent": "pB", "folder": "ro", "mode": "read-only"},
        {"name": "G", "id": "areaG", "parent": "root", "folder": "G", "mode": "consolidate"},
    ]
    w = _env(tmp_path, monkeypatch, d, areas, files)
    w["ids"] = ids
    return w


def _census(w, out="c.json") -> int:
    return dc.main(
        ["census", "--remote", "fake:", "--areas", str(w["areas"]), "--out", str(w["tmp"] / out)]
    )


def _plan(w) -> dict:
    t = w["tmp"]
    assert _census(w) == 0
    assert dc.main(["inventory", "--root", str(w["root"]), "--out", str(t / "i.json")]) == 0
    args = ["plan", "--census", str(t / "c.json"), "--inventory", str(t / "i.json")]
    assert dc.main([*args, "--out", str(t / "p.json")]) == 0
    return json.loads((t / "p.json").read_text(encoding="utf-8"))


def _bytecheck(w) -> dict:
    t = w["tmp"]
    args = ["bytecheck", "--plan", str(t / "p.json"), "--inventory", str(t / "i.json")]
    args += ["--root", str(w["root"]), "--remote", "fake:", "--tmp", str(t / "bc")]
    assert dc.main(args) == 0
    return json.loads((t / "p.json").read_text(encoding="utf-8"))


def _copy(w, *extra) -> int:
    args = [
        "copy",
        "--plan",
        str(w["tmp"] / "p.json"),
        "--root",
        str(w["root"]),
        "--remote",
        "fake:",
    ]
    return dc.main([*args, "--reserve-gb", "0", *extra])


def _verify(w) -> int:
    return dc.main(["verify", "--plan", str(w["tmp"] / "p.json"), "--root", str(w["root"])])


def _by_id(plan: dict) -> dict:
    return {r["id"]: r for r in plan["rows"]}


def _fetched(w) -> list[str]:
    return w["log"].read_text(encoding="utf-8").split()


def _root_hashes(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _one_area(tmp_path, monkeypatch, drive: FakeDrive, files=None, name="A", mode="consolidate"):
    areas = [{"name": name, "id": "areaA", "parent": "root", "folder": name, "mode": mode}]
    return _env(tmp_path, monkeypatch, drive, areas, files or {})


# ---------------------------------------------------------------- census


def test_census_lists_every_object_and_never_follows_a_shortcut(world):
    assert _census(world) == 0
    doc = json.loads((world["tmp"] / "c.json").read_text(encoding="utf-8"))
    names = {o["name"] for a in doc["areas"] for o in a["objects"]}
    assert "private.txt" not in names and "personal" not in names  # behind the shortcut
    assert {"link", "notes", "flex_credentials.json"} <= names
    assert doc["swe_data"] == []
    a = next(a for a in doc["areas"] if a["name"] == "A")
    link = next(o for o in a["objects"] if o["name"] == "link")
    assert link["mime"] == dc.SHORTCUT and link["target_mime"] == dc.FOLDER
    done = next(o for o in a["objects"] if o["name"] == "done.flag")
    assert done["size"] == 0  # rclone omits a zero size; the empty MD5 restores it


def test_census_recovers_when_drive_answers_several_parents_with_nothing(world, monkeypatch):
    assert _census(world, "c1.json") == 0
    monkeypatch.setenv("FAKE_MULTI_EMPTY", "1")
    assert _census(world, "c2.json") == 0
    ids = []
    for f in ("c1.json", "c2.json"):
        doc = json.loads((world["tmp"] / f).read_text(encoding="utf-8"))
        ids.append(sorted(o["id"] for a in doc["areas"] for o in a["objects"]))
    assert ids[0] == ids[1] and len(ids[0]) > 20


def test_census_stops_when_its_totals_differ_from_rclone_size(world, monkeypatch):
    monkeypatch.setenv("FAKE_SIZE_OFFSET", "1")
    assert _census(world) == 2
    assert not (world["tmp"] / "c.json").exists()


def test_census_refuses_an_area_whose_id_does_not_match(world, tmp_path):
    bad = json.loads(world["areas"].read_text(encoding="utf-8"))
    bad[0]["id"] = "not-the-folder"
    p = tmp_path / "bad_areas.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    args = ["census", "--remote", "fake:", "--areas", str(p), "--out", str(tmp_path / "c.json")]
    assert dc.main(args) == 2


@pytest.mark.parametrize("name", ["a\\b", "x/../y", "", "a b"])
def test_census_refuses_an_unsafe_area_name(world, tmp_path, name):
    bad = json.loads(world["areas"].read_text(encoding="utf-8"))
    bad[0]["name"] = name
    p = tmp_path / "bad_areas.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    args = ["census", "--remote", "fake:", "--areas", str(p), "--out", str(tmp_path / "c.json")]
    assert dc.main(args) == 2


def test_census_treats_an_incomplete_search_as_an_error(world, monkeypatch):
    monkeypatch.setenv("FAKE_INCOMPLETE", "1")
    assert _census(world) == 2


# ---------------------------------------------------------------- plan


def test_plan_classifies_every_case(world):
    plan = _plan(world)
    rows, ids = _by_id(plan), world["ids"]
    cls = {k: rows[v]["class"] for k, v in ids.items()}
    legacy = "data_archive/drive-legacy"
    assert cls["x"] == "redundant" and rows[ids["x"]]["twin"] == "data/x.csv"
    assert cls["new"] == "copy" and rows[ids["new"]]["dest"] == f"{legacy}/A/data/new.csv"
    # Same path as a root file with other bytes: never over it.
    assert cls["conflict"] == "copy"
    conflict = f"{legacy}/A/_conflicts/{ids['conflict']}/conflict.csv"
    assert rows[ids["conflict"]]["dest"] == conflict
    # No SHA-256 on Drive: a missing hash never counts as a match; no destination yet.
    assert cls["nosha"] == cls["lying"] == "needs-byte-check"
    assert rows[ids["nosha"]]["dest"] == ""
    assert cls["nosha_unique"] == "copy"
    assert cls["dup1"] == "copy" and cls["dup2"] == "duplicate"
    assert rows[ids["dup2"]]["twin"] == rows[ids["dup1"]]["dest"]
    assert {rows[ids["Case"]]["dest"], rows[ids["case"]]["dest"]} == {
        f"{legacy}/A/data/Case.csv",
        f"{legacy}/A/_conflicts/{ids['case']}/case.csv",
    }
    for key in ("bad", "con", "fullwidth"):  # unsafe, reserved, rewritten by rclone
        assert rows[ids[key]]["dest"] == f"{legacy}/A/data/_drive-{ids[key]}"
    assert cls["twin_a"] == cls["twin_b"] == "copy"
    assert rows[ids["twin_a"]]["dest"] != rows[ids["twin_b"]]["dest"]
    assert not rows[ids["twin_a"]]["deletable"] and not rows[ids["twin_b"]]["deletable"]
    for key in ("shortcut", "gdoc", "cred", "rclone_conf", "gitcfg", "multi"):
        assert cls[key] == "unresolved" and rows[ids[key]]["dest"] == "", key
    assert "credential" in rows[ids["gitcfg"]]["reason"]
    assert cls["done"] == "redundant" and cls["newflag"] == "copy"
    assert cls["vendor"] == "copy" and not rows[ids["vendor"]]["deletable"]
    assert cls["x_ro"] == "redundant" and not rows[ids["x_ro"]]["deletable"]
    assert rows[ids["vendor"]]["dest"] == f"{legacy}/B/ro/vendor.csv"
    assert cls["loose"] == cls["pack"] == "copy"  # only bytes count: git objects come home
    assert rows[ids["x"]]["deletable"] and rows[ids["new"]]["deletable"]
    assert plan["excluded"] == ["data_processed/ibkr/flex_credentials.json"]
    assert all(not r["dest"] or r["dest"].startswith(legacy + "/") for r in plan["rows"])


def test_plan_is_deterministic(world):
    a = _plan(world)["rows"]
    b = _plan(world)["rows"]
    assert a == b


def test_sha256_decides_even_when_size_and_md5_agree(tmp_path):
    """A Drive object listing a root file's size and MD5 but another SHA-256 is not that file."""
    body = b"same size and md5\n"
    md5 = hashlib.md5(body).hexdigest()
    obj = {
        "id": "o1",
        "name": "f.csv",
        "mime": "text/csv",
        "size": len(body),
        "md5": md5,
        "sha256": "0" * 64,
        "parents": ["areaA"],
        "created": None,
        "modified": None,
        "target": None,
        "target_mime": None,
    }
    census_doc = {
        "generated": "g",
        "areas": [{"name": "A", "id": "areaA", "mode": "consolidate", "objects": [obj]}],
    }
    root_file = {
        "path": "data/f.csv",
        "size": len(body),
        "md5": md5,
        "sha256": hashlib.sha256(body).hexdigest(),
        "excluded": False,
    }
    inv = {"generated": "i", "root": str(tmp_path), "dirs": [], "files": [root_file]}
    row = dc.build_plan(census_doc, inv)["rows"][0]
    assert row["class"] == "copy"


def test_a_drive_id_is_deletable_only_if_every_row_naming_it_is(tmp_path, monkeypatch):
    d = FakeDrive()
    d.folder("A", "root", "areaA")
    d.folder("B", "root", "areaB")
    shared = d.folder("shared", "areaA", parents=["areaA", "areaB"])
    child = d.file("c.csv", shared, b"in two areas\n")
    areas = [
        {"name": "A", "id": "areaA", "parent": "root", "folder": "A", "mode": "consolidate"},
        {"name": "B", "id": "areaB", "parent": "root", "folder": "B", "mode": "read-only"},
    ]
    w = _env(tmp_path, monkeypatch, d, areas, {"data/other.csv": b"other\n"})
    rows = [r for r in _plan(w)["rows"] if r["id"] == child]
    assert [r["class"] for r in rows] == ["copy", "duplicate"]  # it comes home once
    assert "the same Drive object" in rows[1]["reason"] and rows[1]["twin"] == rows[0]["dest"]
    assert not any(r["deletable"] for r in rows)


def test_a_destination_that_cannot_be_placed_is_listed_not_fatal(tmp_path, monkeypatch):
    d = FakeDrive()
    d.folder("A", "root", "areaA")
    oid = d.file("f.csv", "areaA", b"drive bytes\n")
    files = {
        "data_archive/drive-legacy/A/f.csv": b"other bytes 1\n",
        f"data_archive/drive-legacy/A/_conflicts/{oid}/f.csv": b"other bytes 2\n",
    }
    w = _one_area(tmp_path, monkeypatch, d, files)
    row = _by_id(_plan(w))[oid]
    assert row["class"] == "unresolved" and "no free destination" in row["reason"]
    assert row["dest"] == ""


def test_the_two_never_overwrite_primitives_refuse_an_existing_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"new bytes\n")
    dst = tmp_path / "out" / "dst.bin"
    dst.parent.mkdir()
    dst.write_bytes(b"existing bytes\n")
    with pytest.raises(FileExistsError):
        dc._copy_new(str(src), dst)
    assert dst.read_bytes() == b"existing bytes\n"
    with pytest.raises(FileExistsError):
        dc._publish(src, dst)
    assert dst.read_bytes() == b"existing bytes\n" and src.read_bytes() == b"new bytes\n"
    fresh = tmp_path / "out" / "fresh.bin"
    dc._publish(src, fresh)
    assert fresh.read_bytes() == b"new bytes\n" and not src.exists()


def test_a_copy_that_fails_part_way_leaves_no_partial_file(tmp_path, monkeypatch):
    src = tmp_path / "src.bin"
    src.write_bytes(b"0123456789" * 1000)

    def disk_full(inp, out, _length=0):
        out.write(inp.read(100))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dc.shutil, "copyfileobj", disk_full)
    dst = tmp_path / "out" / "dst.bin"
    with pytest.raises(OSError):
        dc._copy_new(str(src), dst)
    assert not dst.exists()

    # The fallback when a hard link is not possible: the staging copy stays, nothing lands.
    def no_links(*_a):
        raise OSError(1, "hard links not supported")

    monkeypatch.setattr(dc.os, "link", no_links)
    with pytest.raises(OSError):
        dc._publish(src, dst)
    assert not dst.exists() and src.read_bytes() == b"0123456789" * 1000


def test_outputs_never_land_inside_the_root_except_logs(world):
    t, root = world["tmp"], world["root"]
    assert _census(world) == 0
    assert dc.main(["inventory", "--root", str(root), "--out", str(root / "data" / "i.json")]) == 2
    assert dc.main(["inventory", "--root", str(root), "--out", str(root / "_logs" / "i.json")]) == 0
    args = ["plan", "--census", str(t / "c.json"), "--inventory", str(root / "_logs" / "i.json")]
    assert dc.main([*args, "--out", str(root / "p.json")]) == 2
    assert dc.main([*args, "--out", str(root / "_logs" / "p.json")]) == 0


# ---------------------------------------------------------------- inventory and bytecheck


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
    assert rows[ids["nosha"]]["class"] == "redundant"
    assert rows[ids["nosha"]]["twin"] == "data/nosha_twin.csv"
    # Drive's MD5 said "same as the twin"; the bytes say otherwise.
    assert rows[ids["lying"]]["class"] == "unresolved" and rows[ids["lying"]]["dest"] == ""
    assert not (world["tmp"] / "bc").exists()
    for key in ("cred", "rclone_conf", "gitcfg"):
        assert ids[key] not in _fetched(world)


def test_bytecheck_compares_sha256_not_just_size_and_md5(tmp_path, monkeypatch):
    """The root lists this object's size and MD5 with another SHA-256: not a twin."""
    body = b"bytes without a listed sha\n"
    d = FakeDrive()
    d.folder("A", "root", "areaA")
    oid = d.file("f.csv", "areaA", body, sha=False)
    w = _one_area(tmp_path, monkeypatch, d, {"data/f.csv": body})
    t = w["tmp"]
    assert _census(w) == 0
    inv = dc.inventory(w["root"])
    entry = next(f for f in inv["files"] if f["path"] == "data/f.csv")
    entry["sha256"] = "f" * 64  # same size and MD5, another SHA-256
    (t / "i.json").write_text(json.dumps(inv), encoding="utf-8")
    args = ["plan", "--census", str(t / "c.json"), "--inventory", str(t / "i.json")]
    assert dc.main([*args, "--out", str(t / "p.json")]) == 0
    assert _by_id(_bytecheck(w))[oid]["class"] == "copy"


def test_bytecheck_refuses_a_temporary_folder_inside_the_root(world):
    t = world["tmp"]
    _plan(world)
    args = ["bytecheck", "--plan", str(t / "p.json"), "--inventory", str(t / "i.json")]
    args += ["--root", str(world["root"]), "--remote", "fake:", "--tmp", str(world["root"] / "tmp")]
    assert dc.main(args) == 2


# ---------------------------------------------------------------- copy and verify


def test_copy_and_verify_refuse_while_byte_checks_are_pending(world):
    _plan(world)
    assert _copy(world) == 2
    assert _verify(world) == 1
    assert _fetched(world) == []


def test_copy_brings_everything_home_and_overwrites_nothing(world):
    root = world["root"]
    _plan(world)
    plan = _bytecheck(world)
    before = _root_hashes(root)
    assert _copy(world) == 0
    after = _root_hashes(root)
    assert all(after[k] == v for k, v in before.items())  # nothing existing changed
    copies = [r for r in plan["rows"] if r["class"] == "copy"]
    assert set(after) - set(before) == {r["dest"] for r in copies}
    local = root / "data_archive/drive-legacy/A/data/conflict.csv"
    assert local.read_bytes() == b"LOCAL DIFFERENT\n"
    drive = root / f"data_archive/drive-legacy/A/_conflicts/{world['ids']['conflict']}/conflict.csv"
    assert drive.read_bytes() == b"DRIVE VERSION\n"
    assert not (root.parent / (root.name + dc.STAGING_SUFFIX)).exists()  # staging cleaned up
    assert _verify(world) == 0
    n = len(_fetched(world))
    assert _copy(world) == 0  # a rerun finds everything done
    assert len(_fetched(world)) == n


def test_a_second_round_after_copying_plans_nothing_new(world):
    _plan(world)
    _bytecheck(world)
    assert _copy(world) == 0
    first = _by_id(json.loads((world["tmp"] / "p.json").read_text(encoding="utf-8")))
    _plan(world)
    second = _by_id(_bytecheck(world))
    assert [r for r in second.values() if r["class"] in ("copy", "needs-byte-check")] == []
    for oid, r in first.items():
        if r["class"] in ("copy", "duplicate"):
            assert second[oid]["class"] == "redundant", (r["path"], second[oid])
    assert not [r for r in second.values() if "no free destination" in r["reason"]]


def test_copy_stops_before_copying_when_a_destination_holds_other_bytes(world):
    root = world["root"]
    _plan(world)
    plan = _bytecheck(world)
    row = next(r for r in plan["rows"] if r["id"] == world["ids"]["new"])
    squatter = root / row["dest"]
    squatter.parent.mkdir(parents=True, exist_ok=True)
    squatter.write_bytes(b"someone else's bytes\n")
    before = _fetched(world)  # bytecheck's own downloads
    assert _copy(world) == 3
    assert squatter.read_bytes() == b"someone else's bytes\n"
    assert _fetched(world) == before  # nothing was fetched


def test_copy_never_replaces_a_file_that_appears_mid_run(world, monkeypatch):
    root = world["root"]
    _plan(world)
    plan = _bytecheck(world)
    last = [r for r in plan["rows"] if r["class"] == "copy"][-1]  # --batch 1: fetched last
    squat = root / last["dest"]
    monkeypatch.setenv("FAKE_SQUAT", str(squat))
    assert _copy(world, "--batch", "1", "--jobs", "1") == 3
    assert squat.read_bytes() == b"appeared mid-run\n"


def test_copy_publishes_only_what_matches_drive_sha256(tmp_path, monkeypatch):
    d = FakeDrive()
    d.folder("A", "root", "areaA")
    d.file("f.csv", "areaA", b"listed with the wrong sha256\n", sha256="0" * 64)
    w = _one_area(tmp_path, monkeypatch, d)
    plan = _plan(w)
    assert _copy(w) == 3
    assert not (w["root"] / plan["rows"][0]["dest"]).exists()  # never published


def test_one_undownloadable_object_does_not_block_its_batch(world, monkeypatch):
    root = world["root"]
    _plan(world)
    plan = _bytecheck(world)
    blocked = world["ids"]["new"]
    monkeypatch.setenv("FAKE_FAIL_IDS", blocked)
    assert _copy(world) == 4
    for r in plan["rows"]:
        if r["class"] == "copy" and r["id"] != blocked:
            assert (root / r["dest"]).is_file(), r["dest"]
    log = (world["tmp"] / "p_copy.jsonl").read_text(encoding="utf-8")
    assert "cannotDownloadAbusiveFile" in log


def test_an_interrupted_copy_resumes_without_copying_twice(world, monkeypatch):
    t = world["tmp"]
    _plan(world)
    _bytecheck(world)
    budget = t / "budget"
    budget.write_text("2", encoding="utf-8")
    monkeypatch.setenv("FAKE_BUDGET", str(budget))
    base = len(_fetched(world))  # bytecheck's own downloads
    assert _copy(world, "--batch", "1", "--jobs", "1") == 4
    assert len(_fetched(world)) == base + 2
    monkeypatch.delenv("FAKE_BUDGET")
    assert _copy(world, "--batch", "1", "--jobs", "1") == 0
    fetched = _fetched(world)[base:]
    assert len(fetched) == len(set(fetched))  # no object fetched twice
    assert _verify(world) == 0


def test_parallel_batches_copy_each_object_once(world):
    _plan(world)
    plan = _bytecheck(world)
    base = len(_fetched(world))  # bytecheck's own downloads
    assert _copy(world, "--batch", "1", "--jobs", "4") == 0
    copies = sorted(r["id"] for r in plan["rows"] if r["class"] == "copy")
    assert sorted(_fetched(world)[base:]) == copies
    assert _verify(world) == 0


def test_long_paths_come_home(tmp_path, monkeypatch):
    d = FakeDrive()
    d.folder("A", "root", "areaA")
    parent = "areaA"
    for i in range(8):
        parent = d.folder(f"{i}" + "n" * 39, parent)
    d.file("deep.csv", parent, b"deep\n")
    w = _one_area(tmp_path, monkeypatch, d)
    plan = _plan(w)
    row = next(r for r in plan["rows"] if r["class"] == "copy")
    assert len(str(w["root"] / row["dest"])) > 260
    assert _copy(w) == 0 and _verify(w) == 0


def test_copy_refuses_a_destination_outside_the_legacy_folder(world):
    t = world["tmp"]
    plan = _plan(world)
    for r in plan["rows"]:
        if r["class"] == "needs-byte-check":
            r["class"] = "unresolved"
    next(r for r in plan["rows"] if r["class"] == "copy")["dest"] = "data/bloomberg/sp500_ohlcv.csv"
    (t / "p.json").write_text(json.dumps(plan), encoding="utf-8")
    assert _copy(world) == 2


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


# ---------------------------------------------------------------- sweep


@pytest.fixture
def sweep_world(world, tmp_path):
    src = tmp_path / "old"
    for rel, data in {
        "a/same.csv": b"x-bytes\n",  # already in the root, by bytes
        "a/new.csv": b"only here\n",
        "a/tool.py": b"print('code')\n",
        "a/secret_token.txt": b"hunter2\n",
        "a/empty.flag": b"",
        ".git/HEAD": b"ref\n",
    }.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    inv = tmp_path / "si.json"
    assert dc.main(["inventory", "--root", str(world["root"]), "--out", str(inv)]) == 0
    return {**world, "src": src, "inv": inv}


def _sweep(w, dest="data_archive/old", source=None) -> int:
    args = ["sweep", "--source", str(source or w["src"]), "--root", str(w["root"])]
    return dc.main([*args, "--inventory", str(w["inv"]), "--dest", dest])


def test_sweep_copies_only_the_bytes_the_root_lacks(sweep_world, monkeypatch):
    root = sweep_world["root"]
    seen = []
    real = dc.hash_file

    def spy(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(dc, "hash_file", spy)
    assert _sweep(sweep_world) == 0
    out = root / "data_archive/old"
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert got == ["a/empty.flag", "a/new.csv"]
    assert (out / "a/new.csv").read_bytes() == b"only here\n"
    assert not any("secret_token" in s for s in seen)
    assert _sweep(sweep_world) == 0  # a rerun copies nothing new
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == got
    assert not (root.parent / (root.name + dc.STAGING_SUFFIX)).exists()  # staging cleaned up


def test_sweep_stops_on_a_destination_with_other_bytes(sweep_world):
    root = sweep_world["root"]
    squat = root / "data_archive/old/a/new.csv"
    squat.parent.mkdir(parents=True)
    squat.write_bytes(b"other bytes\n")
    assert _sweep(sweep_world) == 3
    assert squat.read_bytes() == b"other bytes\n"
    assert not (root / "data_archive/old/a/empty.flag").exists()


def _staged(root):
    staging = root.parent / (root.name + dc.STAGING_SUFFIX)
    return sorted(p for p in staging.rglob("*") if p.is_file()) if staging.exists() else []


def test_sweep_never_puts_a_bad_copy_in_the_root(sweep_world, monkeypatch):
    root = sweep_world["root"]
    real = dc._copy_new

    def corrupting(src, dst):
        real(src, dst)
        with open(dst, "ab") as fh:
            fh.write(b"!")

    monkeypatch.setattr(dc, "_copy_new", corrupting)
    assert _sweep(sweep_world) == 3
    assert not (root / "data_archive/old").exists()
    assert len(_staged(root)) == 2  # kept beside the root for inspection


def test_sweep_never_replaces_a_file_that_appears_mid_run(sweep_world, monkeypatch):
    root = sweep_world["root"]
    target = root / "data_archive/old/a/new.csv"
    real = dc._copy_new

    def squatting(src, dst):
        real(src, dst)
        if str(src).endswith("new.csv"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"someone else's bytes\n")

    monkeypatch.setattr(dc, "_copy_new", squatting)
    assert _sweep(sweep_world) == 3
    assert target.read_bytes() == b"someone else's bytes\n"
    assert (root / "data_archive/old/a/empty.flag").read_bytes() == b""
    assert [p.read_bytes() for p in _staged(root)] == [b"only here\n"]


@pytest.mark.parametrize("dest", ["data/x", "data_raw", "_logs/x", "a\\b", "../out", ""])
def test_sweep_refuses_the_live_trees_and_odd_destinations(sweep_world, dest):
    assert _sweep(sweep_world, dest=dest) == 2


def test_sweep_refuses_a_source_inside_the_root(sweep_world):
    assert _sweep(sweep_world, source=sweep_world["root"] / "data") == 2


# ---------------------------------------------------------------- sums, filters, names


def test_sums_lists_the_root_but_not_logs_credentials_or_itself(world):
    root = world["root"]
    n, _total, digest = dc.write_sums(root)
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
        "cfg/rclone.conf",
        "cfg/config",
        "repo/.git/config",
        "repo/.git/HEAD",
        "keys/id_rsa",
        "keys/id_ed25519.pub",
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
    assert dc.win_safe("ticker=AAPL") and dc.win_safe("a.csv.gz") and dc.win_safe("é.csv")
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
        "del\x7f",
        "q‛z.csv",  # rclone's escape character
        "c␁d.csv",  # a control picture
        "a：b.csv",  # a full-width colon
        "x" * 256,  # over NTFS's 255 per name
    ):
        assert not dc.win_safe(bad), bad


def test_credential_shaped_matches_any_component():
    assert dc.credential_shaped("x/Secrets/y.csv")
    assert dc.credential_shaped("ibkr/flex_credentials.json")
    assert dc.credential_shaped("cfg/rclone.conf") and dc.credential_shaped("repo/.git/config")
    assert not dc.credential_shaped("data/bloomberg/sp500_credit_risk.csv")
    assert not dc.credential_shaped("data/keys.csv") and not dc.credential_shaped("cfg/config")


def test_the_default_areas_match_the_inventory_record_and_their_modes():
    text = (_REPO / "docs" / "DATA_INVENTORY.md").read_text(encoding="utf-8")
    for a in dc.DEFAULT_AREAS:
        assert a["id"] in text, a["name"]
    modes = {a["name"]: a["mode"] for a in dc.DEFAULT_AREAS}
    assert modes == {
        "swe-local-only": "consolidate",
        "SmartWheelData": "consolidate",
        "smart-wheel-engine-git": "consolidate",
        "day-bot-local-archive/vendor_swe_data": "read-only",
        "day-bot-local-archive/vendor_swe_data_raw": "read-only",
        "day-bot-local-archive/vendor_swe_data_processed": "read-only",
        "day-bot-local-archive/data_raw": "read-only",
    }
    dc._check_areas([dict(a) for a in dc.DEFAULT_AREAS])
