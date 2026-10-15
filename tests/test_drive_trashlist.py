"""scripts/drive_consolidate.py: card 3's census fields, swe-data/ by id, trashlist and recount.

Card 3's code run, for D37, the middle route (design version 6 §2 (a), (b), (d) and (e), §3's
reading by id and §7, with version 4's §2 (a) and (b)). Pins:
- census: ``owned`` comes from ``backend query`` with ``--drive-auth-owner-only``, which no
  other command gets; ``-M`` goes only to the sharing reads; ``swe-data/`` is read by its
  id (missing, trashed or not a folder stops it; a folder of its name elsewhere, the top of
  My Drive included, changes nothing); the chain from My Drive's root to
  OptionsEngine_Project is recorded, and any other answer stops the census; the sharing of
  each link, of ``swe-data/`` and of each area folder is recorded with no name or email;
  the re-census that copy and bytecheck make reads none of it;
- trashlist: its refusals; only ``deletable`` files of a six-area plan, never one a
  read-only area shows; each kind of object never listed; the ``swe-data/`` twin and root
  twin cases, archive twins included; an empty file needs both copies too; outputs only
  in a new folder under ``_logs/``;
- the whole-folder accounting: every object of the three areas, and each area folder,
  gets exactly one class; each blocker kind makes its area's verdict "no"; a folder
  holding a blocker, with two parents or not owned, is a blocker; a clean area is "yes";
- the proof's sha256 is the sha256 of ``trashlist.json``'s bytes alone, the same twice
  over the same inputs, and changes with any row of any class;
- recount: "unchanged: yes" on an unchanged fake Drive, and "no" for each change it must
  see, its inputs included;
- the tool runs rclone only with an allowlist of verbs.

It uses the fake rclone and ``FakeDrive`` of ``tests/test_drive_consolidate.py``, which
always hold the chain and ``swe-data/`` at their pinned ids. Most cases change the
documents of one proven world in memory, so each rule is tested on its own; the rest run
the commands end to end.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_drive_consolidate as tdc
from tests.test_drive_consolidate import (  # noqa: F401  (fixtures)
    every_fake_rclone_call_is_allowed,  # autouse: the session's check of every rclone verb
    no_rclone_outside_the_fake,  # autouse: dc.RCLONE is a sentinel until a test sets the fake
    world,  # for the verb test
)

dc = tdc.dc
FakeDrive = tdc.FakeDrive

AREA = {a["name"]: a for a in dc.DEFAULT_AREAS}
CONS = ("swe-local-only", "SmartWheelData", "smart-wheel-engine-git")
SLO, SWD, GIT = (AREA[n]["id"] for n in CONS)
VENDOR = AREA["day-bot-local-archive/vendor_swe_data"]["id"]
LOCAL_ARCHIVE = "1BBSXZIZBF8xwqIvkybN9WK8GOSVDiwo0"
SWE = dc.SWE_DATA["id"]
T0 = dt.datetime(2026, 10, 15, 8, 0, 0, tzinfo=dt.UTC)
LEGACY = dc.LEGACY


class Clock:
    """utc_now, a second later at every call: each document is later than the one before."""

    def __init__(self, start: dt.datetime):
        self.t = start

    def __call__(self) -> str:
        self.t += dt.timedelta(seconds=1)
        return self.t.isoformat()


class World:
    """A fake Drive with card 2b's six areas where §C.3 puts them, a root, and swe-data/."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch):
        self.tmp, self.mp = tmp_path, mp
        self.d = FakeDrive()
        self.files: dict[str, bytes] = {}
        self.ids: dict[str, str] = {}
        self.folders: dict[tuple[str, str], str] = {}
        for n in CONS:
            a = AREA[n]
            self.d.folder(a["folder"], a["parent"], a["id"])
        self.d.folder("Day_Trading_Bot", "chain-projects", "dtb")
        self.d.folder("_local_archive", "dtb", LOCAL_ARCHIVE)
        for a in dc.DEFAULT_AREAS:
            if a["mode"] == "read-only":
                self.d.folder(a["folder"], a["parent"], a["id"])

    def dir(self, top: str, path: str) -> str:
        cur, parts = top, path.split("/")
        for i, part in enumerate(parts):
            key = (top, "/".join(parts[: i + 1]))
            if key not in self.folders:
                self.folders[key] = self.d.folder(part, cur)
            cur = self.folders[key]
        return cur

    def drive_file(self, key: str, top: str, path: str, data: bytes, **kw) -> str:
        folder, _, name = path.rpartition("/")
        parent = self.dir(top, folder) if folder else top
        self.ids[key] = self.d.file(name, parent, data, **kw)
        return self.ids[key]

    def home(self, rel: str, data: bytes, swe: bool = True) -> None:
        """A root file, and (as card 2c made it) its copy at the same path in swe-data/."""
        self.files[rel] = data
        if swe:
            self.drive_file(f"swe:{rel}", SWE, rel, data)

    def env(self) -> dict:
        areas = [dict(a) for a in dc.CARD2B_AREAS]
        w = tdc._env(self.tmp, self.mp, self.d, areas, self.files)
        w["world"] = self
        return w


def _clean(w: World) -> None:
    """One proven file per area (a root twin, two archive twins), an empty folder, a vendor file."""
    w.drive_file("a", SLO, "data/a.csv", b"alpha\n")
    w.home("data/a.csv", b"alpha\n")
    w.dir(SLO, "empty")
    w.drive_file("t1", SWD, "data_processed/theta/t1.parquet", b"theta one\n")
    w.home(f"{LEGACY}/SmartWheelData/data_processed/theta/t1.parquet", b"theta one\n")
    w.drive_file("git", GIT, "objects/ab/cdef", b"git object\n")
    w.home(f"{LEGACY}/smart-wheel-engine-git/objects/ab/cdef", b"git object\n")
    w.drive_file("vendor", VENDOR, "v.csv", b"vendor bytes\n")
    w.home("data/v.csv", b"vendor bytes\n")


def _census_of(w: dict, areas, name: str, preset: str | None = None, only=()) -> int:
    """The census command over these areas, or over one of the tool's presets (card 3's)."""
    args = ["census", "--remote", "fake:", "--out", str(w["tmp"] / name)]
    for area in only:
        args += ["--only", area]
    if preset:
        return dc.main([*args, "--preset", preset])
    spec = w["tmp"] / f"{name}.areas.json"
    spec.write_text(json.dumps([dict(a) for a in areas]), encoding="utf-8")
    return dc.main([*args, "--areas", str(spec)])


def _prepare(w: dict, bytecheck: bool = False) -> None:
    """Card 3a's documents: the six-area census, swe-data/'s, the inventory and the plan."""
    t = w["tmp"]
    assert _census_of(w, None, "c6.json", "card2b") == 0
    assert _census_of(w, None, "swe.json", "swe-data") == 0
    assert dc.main(["inventory", "--root", str(w["root"]), "--out", str(t / "i.json")]) == 0
    args = ["plan", "--census", str(t / "c6.json"), "--inventory", str(t / "i.json")]
    assert dc.main([*args, "--out", str(t / "p.json")]) == 0
    if bytecheck:
        args = ["bytecheck", "--plan", str(t / "p.json"), "--inventory", str(t / "i.json")]
        args += ["--root", str(w["root"]), "--remote", "fake:", "--tmp", str(t / "bc")]
        assert dc.main(args) == 0


def _trashlist(w: dict, out: str = "list", **names) -> int:
    t = w["tmp"]
    args = ["trashlist", "--plan", str(t / names.get("plan", "p.json"))]
    args += ["--census", str(t / names.get("census", "c6.json"))]
    args += ["--swe-census", str(t / names.get("swe", "swe.json"))]
    args += ["--root", str(names.get("root", w["root"]))]
    return dc.main([*args, "--out", str(names.get("out_path", w["root"] / "_logs" / out))])


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _docs(w: dict) -> tuple[dict, dict, dict]:
    return tuple(_load(w["tmp"] / n) for n in ("p.json", "c6.json", "swe.json"))  # type: ignore[return-value]


@pytest.fixture(scope="module")
def proven(tmp_path_factory):
    """One clean world, proven once: census, plan and trashlist, as card 3a would run them."""
    tmp = tmp_path_factory.mktemp("proven")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dc, "utc_now", Clock(T0))
        w = World(tmp, mp)
        _clean(w)
        env = w.env()
        _prepare(env)
        assert _census_of(env, None, "early.json", "consolidate") == 0  # older than the proof
        assert _trashlist(env) == 0
        env["docs"] = _docs(env)
        env["rclone"] = list(dc.RCLONE)
        env["summary"] = _load(env["root"] / "_logs" / "list" / dc.PROOF_SUMMARY)
    return env


@pytest.fixture
def case(proven, tmp_path):
    """The proven world's documents, copied for this test, with a copy of its root."""
    root = tmp_path / "root"
    shutil.copytree(proven["root"], root, ignore=shutil.ignore_patterns("_logs"))
    plan, census, swe = copy.deepcopy(proven["docs"])
    plan["root"] = str(root)
    world_ = proven["world"]
    return SimpleNamespace(
        plan=plan, census=census, swe=swe, root=root, ids=world_.ids, folders=world_.folders
    )


def _classes(c) -> dict[str, dict]:
    rows, _ = dc.trashlist(c.plan, c.census, c.swe, c.root)
    out: dict[str, dict] = {}
    for r in rows:
        assert r["id"] not in out, f"{r['id']} has two rows"
        out[r["id"]] = r
    return out


def _verdicts(c) -> dict[str, str]:
    _rows, areas = dc.trashlist(c.plan, c.census, c.swe, c.root)
    return {a["name"]: a["clear_to_trash_by_hand"] for a in areas}


def _listings(doc: dict, oid: str) -> list[dict]:
    """Every listing of one object in a census (an id can show in more than one area)."""
    found = [o for a in doc["areas"] for o in a["objects"] if o["id"] == oid]
    assert found, oid
    return found


def _area_of(doc: dict, aid: str) -> dict:
    return next(a for a in doc["areas"] if a["id"] == aid)


def _plan_rows(plan: dict, oid: str) -> list[dict]:
    rows = [r for r in plan["rows"] if r["id"] == oid]
    assert rows, oid
    return rows


# ---------------------------------------------------------------- census: owners, chain, swe-data


@pytest.fixture
def small(tmp_path, monkeypatch):
    """One consolidate area in OptionsEngine_Project, with a file the account does not own."""
    w = World(tmp_path, monkeypatch)
    w.drive_file("mine", SLO, "data/mine.csv", b"mine\n")
    w.drive_file("theirs", SLO, "data/theirs.csv", b"theirs\n", owned=False)
    env = tdc._env(tmp_path, monkeypatch, w.d, [AREA["swe-local-only"]], {})
    env["world"] = w
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    env["calls"] = calls
    return env


def _calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _query_of(call: list[str]) -> str:
    return [a for a in call[2:] if not a.startswith("-")][1]


def test_owned_comes_from_backend_query_alone(small):
    assert tdc._census(small) == 0
    doc = _load(small["tmp"] / "c.json")
    objs = {o["name"]: o for o in doc["areas"][0]["objects"]}
    assert objs["mine.csv"]["owned"] is True and objs["theirs.csv"]["owned"] is False
    assert objs["data"]["owned"] is True and doc["areas"][0]["root_owned"] is True
    calls = _calls(small["calls"])
    assert calls
    for call in calls:
        if dc.OWNER_ONLY in call:
            assert call[:2] == ["backend", "query"], call
        if call[:2] == ["backend", "query"]:
            assert dc.OWNER_ONLY in call, call
    listings = [c for c in calls if c[:1] == ["lsjson"]]
    assert listings and all(dc.OWNER_ONLY not in c and "-M" not in c for c in listings)
    # The fake's lsjson drops what the account does not own when given the flag, as rclone's
    # listing does: the census's cross-check would then fail. Here it passed.
    assert doc["areas"][0]["check"]["count"] == 2


def test_the_sharing_flag_goes_only_to_the_sharing_reads(small):
    assert tdc._census(small) == 0
    for call in _calls(small["calls"]):
        if "-M" not in call:
            continue
        assert call[:2] == ["backend", "query"], call
        q = _query_of(call)
        assert "trashed" not in q and " or " not in q, q  # one folder's own read
    walks = [c for c in _calls(small["calls"]) if c[:2] == ["backend", "query"]]
    walks = [c for c in walks if "trashed = false" in _query_of(c)]
    assert walks and all("-M" not in c for c in walks)


def test_the_census_records_the_chain_swe_data_and_sharing(small):
    assert tdc._census(small) == 0
    doc = _load(small["tmp"] / "c.json")
    assert [(x["name"], x["id"]) for x in doc["chain"]] == [
        ("The_Works", "chain-the-works"),
        ("Projects", "chain-projects"),
        ("OptionsEngine_Project", dc.PROJECT_FOLDER_ID),
    ]
    assert [x["parents"] for x in doc["chain"]] == [
        [tdc.MY_DRIVE_ROOT],
        ["chain-the-works"],
        ["chain-projects"],
    ]
    assert doc["my_drive_root"] == tdc.MY_DRIVE_ROOT
    swe = doc["swe_data"]
    assert (swe["id"], swe["name"], swe["parents"]) == (SWE, "swe-data", [dc.PROJECT_FOLDER_ID])
    assert swe["owned"] is True and swe["trashed"] is False
    a = doc["areas"][0]
    assert a["root_parents"] == [dc.PROJECT_FOLDER_ID]
    for rec in [*doc["chain"], swe, a]:
        assert rec["sharing"] == [{"id": "perm-owner", "role": "owner", "type": "user"}]
    text = (small["tmp"] / "c.json").read_text(encoding="utf-8")
    assert "example.invalid" not in text and "The Owner" not in text  # no email, no name


@pytest.mark.parametrize("change", ["missing", "trashed", "not a folder", "moved"])
def test_the_census_stops_unless_swe_data_is_where_its_id_says(small, change, capsys):
    d = small["world"].d
    swe = d.obj(SWE)
    if change == "missing":
        d.objects.remove(swe)
    elif change == "trashed":
        swe["trashed"] = True
    elif change == "not a folder":
        swe["mimeType"] = "text/csv"
    else:
        swe["parents"] = ["chain-projects"]
    d.save(small["tmp"] / "drive.json")
    assert tdc._census(small) == 2
    err = capsys.readouterr().err
    words = {"missing": "missing", "moved": "missing", "trashed": "in Drive's trash"}
    assert SWE in err and words.get(change, change) in err
    assert not (small["tmp"] / "c.json").exists()


def test_swe_data_is_found_by_its_id_whatever_else_bears_its_name(small):
    d = small["world"].d
    d.folder("swe-data", tdc.MY_DRIVE_ROOT, "decoy-at-the-top")  # a folder of its name at the top
    d.folder("swe-data", "chain-projects", "decoy-in-projects")
    d.obj(SWE)["name"] = "swe-data (renamed)"  # its own name plays no part
    d.save(small["tmp"] / "drive.json")
    assert tdc._census(small) == 0
    swe = _load(small["tmp"] / "c.json")["swe_data"]
    assert (swe["id"], swe["parents"]) == (SWE, [dc.PROJECT_FOLDER_ID])
    for call in _calls(small["calls"]):
        if call[:2] == ["backend", "query"]:
            assert "swe-data" not in _query_of(call), call  # never looked up by name


@pytest.mark.parametrize(
    "change",
    [
        "The_Works missing",
        "a second The_Works, trashed",
        "Projects trashed",
        "Projects not a folder",
        "OptionsEngine_Project another id",
        "The_Works with two parents",
        "a second The_Works, another case",
    ],
)
def test_the_census_stops_on_any_other_chain(small, change, capsys):
    # An area outside the chain, so that a trashed link (which Drive also shows on every
    # folder below it) is found by the chain's own read, not first by the area's lookup.
    d = small["world"].d
    d.folder("Elsewhere", tdc.MY_DRIVE_ROOT, "area-elsewhere")
    elsewhere = {"name": "Elsewhere", "id": "area-elsewhere", "parent": "root"}
    elsewhere.update({"folder": "Elsewhere", "mode": "read-only"})
    small["areas"].write_text(json.dumps([elsewhere]), encoding="utf-8")
    if change == "a second The_Works, another case":
        d.folder("the_works", tdc.MY_DRIVE_ROOT, "lower-the-works")  # Drive's name = ignores case
    elif change == "The_Works missing":
        d.obj("chain-the-works")["name"] = "The Works"
    elif change == "a second The_Works, trashed":
        d.folder("The_Works", tdc.MY_DRIVE_ROOT, "old-the-works")
        d.obj("old-the-works")["trashed"] = True
    elif change == "Projects trashed":
        d.obj("chain-projects")["trashed"] = True
    elif change == "Projects not a folder":
        d.obj("chain-projects")["mimeType"] = "text/plain"
    elif change == "OptionsEngine_Project another id":
        d.obj(dc.PROJECT_FOLDER_ID)["name"] = "OptionsEngine_Project (old)"
        d.folder("OptionsEngine_Project", "chain-projects", "another-oep")
    else:
        d.obj("chain-the-works")["parents"] = [tdc.MY_DRIVE_ROOT, "elsewhere"]
    d.save(small["tmp"] / "drive.json")
    assert tdc._census(small) == 2
    assert "the chain" in capsys.readouterr().err
    assert not (small["tmp"] / "c.json").exists()


def test_the_census_stops_when_an_area_folder_cannot_be_read_for_its_sharing(
    small, monkeypatch, capsys
):
    real = dc.drive_query

    def lose_the_folder(remote, query, tries=3, sharing=False):
        found = real(remote, query, tries, sharing)
        if sharing and "name = 'swe-local-only'" in query:
            return [h for h in found if h["id"] != SLO]  # gone between the walk and this read
        return found

    monkeypatch.setattr(dc, "drive_query", lose_the_folder)
    assert tdc._census(small) == 2
    assert f"its folder {SLO} could not be read again" in capsys.readouterr().err
    assert not (small["tmp"] / "c.json").exists()


def test_the_re_census_of_copy_and_bytecheck_reads_no_chain_or_swe_data(small):
    small["calls"].write_text("", encoding="utf-8")
    doc = dc.census("fake:", [AREA["swe-local-only"]])
    assert "swe_data" not in doc and "chain" not in doc
    for call in _calls(small["calls"]):
        assert "-M" not in call
        if call[:2] == ["backend", "query"]:
            q = _query_of(call)
            assert dc.PROJECT_FOLDER_ID not in q or "swe-local-only" in q, q
            assert "The_Works" not in q and "Projects" not in q, q


def test_the_plan_says_where_swe_data_is_and_warns_of_nothing(small, capsys):
    t = small["tmp"]
    assert tdc._census(small) == 0
    assert dc.main(["inventory", "--root", str(small["root"]), "--out", str(t / "i.json")]) == 0
    args = ["plan", "--census", str(t / "c.json"), "--inventory", str(t / "i.json")]
    assert dc.main([*args, "--out", str(t / "p.json")]) == 0
    out = capsys.readouterr().out
    assert f"swe-data/: id {SWE}, in {dc.PROJECT_FOLDER_ID}" in out
    assert "WARNING" not in out
    assert _load(t / "p.json")["swe_data"]["id"] == SWE


def test_the_built_in_areas_sit_in_options_engine_project():
    parents = {a["name"]: a["parent"] for a in dc.DEFAULT_AREAS if a["mode"] == "consolidate"}
    assert parents == dict.fromkeys(CONS, dc.PROJECT_FOLDER_ID)
    assert [a["name"] for a in dc.CARD2B_AREAS] == [
        a["name"] for a in dc.DEFAULT_AREAS if a["name"] != "day-bot-local-archive/data_raw"
    ]
    assert len(dc.CARD2B_AREAS) == 6
    assert dc.SWE_DATA == {
        "name": "swe-data",
        "id": "13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b",
        "parent": "1niufzSC5-C5fMJ0XZg8LSR53Tm1fp2-4",
        "folder": "swe-data",
        "mode": "read-only",
    }


# ---------------------------------------------------------------- trashlist: a clean world


def test_a_clean_world_is_listed_and_every_area_is_clear(proven):
    out = proven["root"] / "_logs" / "list"
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(
        [dc.PROOF, "trashlist.csv", "kept.csv", "sharing.json", dc.PROOF_SUMMARY]
    )
    summary = proven["summary"]
    assert [a["clear_to_trash_by_hand"] for a in summary["areas"]] == ["yes"] * 3
    rows = _load(out / dc.PROOF)["rows"]
    ids = proven["world"].ids
    listed = {r["id"]: r for r in rows if r["class"] == "listed"}
    assert set(listed) == {ids["a"], ids["t1"], ids["git"]}
    assert ids["vendor"] not in {r["id"] for r in rows}  # a read-only area's file: never classed
    a = listed[ids["a"]]
    assert a["twin"] == "data/a.csv" and a["twin_size"] == len(b"alpha\n")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{9}Z", a["twin_modified"])
    assert a["swe_twin_id"] == ids["swe:data/a.csv"]
    assert a["swe_twin_parent"] == proven["world"].folders[(SWE, "data")]
    t1 = listed[ids["t1"]]  # an archive twin counts
    assert t1["twin"] == f"{LEGACY}/SmartWheelData/data_processed/theta/t1.parquet"
    tops = {r["id"]: r for r in rows if r["path"] == ""}
    assert set(tops) == {SLO, SWD, GIT}
    assert all(
        r["class"] == "folder" and r["parents"] == [dc.PROJECT_FOLDER_ID] for r in tops.values()
    )
    kept = (out / "kept.csv").read_text(encoding="utf-8").splitlines()
    assert kept == [",".join(dc.TRASHLIST_FIELDS)]
    with open(out / "trashlist.csv", encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == len(rows) + 1


def test_every_object_of_the_three_areas_and_each_area_folder_gets_one_class(case):
    rows, _ = dc.trashlist(case.plan, case.census, case.swe, case.root)
    want = []
    for a in case.census["areas"]:
        if a["mode"] == "consolidate":
            want += [(a["name"], a["id"])] + [(a["name"], o["id"]) for o in a["objects"]]
    assert sorted((r["area"], r["id"]) for r in rows) == sorted(want)
    assert len({(r["area"], r["id"]) for r in rows}) == len(rows)
    assert {r["class"] for r in rows} <= {"listed", "folder", "blocker"}
    assert all(bool(r["reason"]) == (r["class"] == "blocker") for r in rows)


def test_an_empty_folder_is_a_folder_and_an_area_of_listed_files_is_clear(case):
    cls = _classes(case)
    assert cls[case.folders[(SLO, "empty")]]["class"] == "folder"
    assert cls[case.folders[(SLO, "data")]]["class"] == "folder"
    assert cls[SLO]["class"] == "folder"
    assert _verdicts(case) == dict.fromkeys(CONS, "yes")


# ---------------------------------------------------------------- the five listing conditions


def test_condition_1_every_plan_row_naming_it_is_deletable(case):
    _plan_rows(case.plan, case.ids["a"])[0]["deletable"] = False
    r = _classes(case)[case.ids["a"]]
    assert r["class"] == "blocker" and r["reason"].startswith("not deletable in the plan")
    assert _verdicts(case)["swe-local-only"] == "no"


def test_condition_1_an_object_the_plan_does_not_name_is_never_listed(case):
    case.plan["rows"] = [r for r in case.plan["rows"] if r["id"] != case.ids["a"]]
    r = _classes(case)[case.ids["a"]]
    assert (r["class"], r["reason"]) == ("blocker", "not in the plan")


CONDITION_2 = {
    "no size": ({"size": None}, "Drive lists no size"),
    "no MD5": ({"md5": None}, "Drive lists no MD5"),
    "no SHA-256": ({"sha256": None}, "Drive lists no SHA-256"),
    "two parents": ({"parents": "+elsewhere"}, "2 parents, not one"),
    "not owned": ({"owned": False}, "not owned by the account"),
    "owner not recorded": ({"owned": None}, "owner not recorded"),
    "trashed": ({"trashed": True}, "in Drive's trash"),
    "Google-format": (
        {"mime": "application/vnd.google-apps.spreadsheet"},
        "a Google-format file: no bytes to compare",
    ),
    "shortcut": ({"mime": dc.SHORTCUT}, "a shortcut: never followed, never listed"),
    "withheld": ({"md5": "withheld", "sha256": None}, "hashes withheld by the census"),
}


@pytest.mark.parametrize("change", sorted(CONDITION_2))
def test_condition_2_what_drive_lists_for_it(case, change):
    edit, reason = CONDITION_2[change]
    for o in _listings(case.census, case.ids["a"]):
        for k, v in edit.items():
            o[k] = [*o[k], "elsewhere"] if v == "+elsewhere" else v
    r = _classes(case)[case.ids["a"]]
    assert r["class"] == "blocker" and r["reason"].startswith(reason), r["reason"]
    assert _verdicts(case) == {
        "swe-local-only": "no",
        "SmartWheelData": "yes",
        "smart-wheel-engine-git": "yes",
    }


@pytest.mark.parametrize("twin", ["root", "archive"])
@pytest.mark.parametrize("change", ["changed", "removed"])
def test_condition_3_the_desktop_twin_hashed_again_now(case, twin, change):
    key = "a" if twin == "root" else "t1"
    rel = _plan_rows(case.plan, case.ids[key])[0]["twin"]
    p = case.root / rel
    if change == "changed":
        p.write_bytes(p.read_bytes().upper())  # same size, other bytes
    else:
        p.unlink()
    r = _classes(case)[case.ids[key]]
    assert r["class"] == "blocker" and r["reason"].startswith(f"root twin {rel}"), r["reason"]
    assert ("missing" in r["reason"]) == (change == "removed")


def test_condition_3_the_plan_must_name_one_twin(case):
    _plan_rows(case.plan, case.ids["a"])[0]["twin"] = ""
    r = _classes(case)[case.ids["a"]]
    assert (r["class"], r["reason"]) == ("blocker", "the plan names no single root twin")


SWE_TWIN = {
    "same size and MD5, another SHA-256": ("sha256", "f" * 64, "SHA-256 differ"),
    "no SHA-256": ("sha256", None, "SHA-256 differ"),
    "another size": ("size", 999, "size"),
    "missing": ("remove", None, "holds 0 objects"),
    "at another path": ("name", "a-moved.csv", "holds 0 objects"),
    "the listed id itself": ("id", "LISTED", "this same Drive object"),
    "two parents": ("parents", "+elsewhere", "2 parents"),
    "not a file": ("mime", "application/vnd.google-apps.document", "is not a file"),
    "two objects at its path": ("twice", None, "holds 2 objects"),
}


@pytest.mark.parametrize("change", sorted(SWE_TWIN))
def test_condition_4_the_swe_data_twin_at_exactly_the_twin_path(case, change):
    key, value, reason = SWE_TWIN[change]
    objs = case.swe["areas"][0]["objects"]
    sid = case.ids["swe:data/a.csv"]
    o = next(o for o in objs if o["id"] == sid)
    if key == "remove":
        objs.remove(o)
    elif key == "twice":
        objs.append({**o, "id": "a-second-copy"})
    elif key == "id":
        o["id"] = case.ids["a"]
    elif value == "+elsewhere":
        o[key] = [*o[key], "elsewhere"]
    else:
        o[key] = value
    r = _classes(case)[case.ids["a"]]
    assert r["class"] == "blocker" and r["reason"].startswith("swe-data/"), r["reason"]
    assert reason in r["reason"], r["reason"]


def test_condition_5_a_credential_shaped_name_is_never_listed(case):
    for o in _listings(case.census, case.ids["a"]):
        o["name"] = "api_token.csv"
    r = _classes(case)[case.ids["a"]]
    assert (r["class"], r["reason"]) == ("blocker", "a credential-shaped name")


def test_what_counts_comes_from_the_census_never_the_plan(case):
    rows = _plan_rows(case.plan, case.ids["a"])
    rows[0]["sha256"] = "0" * 64  # the plan's hash, not the census's
    r = _classes(case)[case.ids["a"]]
    assert r["reason"] == "the plan's size or hashes differ from the census's listed ones"


def test_a_hash_learned_from_a_download_never_counts(case):
    _plan_rows(case.plan, case.ids["a"])[0]["reason"] = "SHA-256 from a download"
    r = _classes(case)[case.ids["a"]]
    assert r["class"] == "blocker" and "download" in r["reason"]


def test_the_plan_must_list_it_at_the_census_path(case):
    _plan_rows(case.plan, case.ids["a"])[0]["path"] = "data/elsewhere.csv"
    r = _classes(case)[case.ids["a"]]
    assert (r["class"], r["reason"]) == ("blocker", "the plan lists it at another path")


def test_an_object_a_read_only_area_shows_is_a_blocker_here(case):
    o = copy.deepcopy(_listings(case.census, case.ids["a"])[0])
    o["parents"] = [VENDOR]
    _area_of(case.census, VENDOR)["objects"].append(o)
    r = _classes(case)[case.ids["a"]]
    assert r["reason"] == "also seen in the read-only area day-bot-local-archive/vendor_swe_data"
    assert _verdicts(case)["swe-local-only"] == "no"


def test_an_object_seen_twice_in_the_consolidate_areas_is_a_blocker(case):
    o = copy.deepcopy(_listings(case.census, case.ids["a"])[0])
    o["parents"] = [GIT]
    _area_of(case.census, GIT)["objects"].append(o)
    rows, _ = dc.trashlist(case.plan, case.census, case.swe, case.root)
    mine = [r for r in rows if r["id"] == case.ids["a"]]
    assert len(mine) == 2 and all(r["class"] == "blocker" for r in mine)
    assert all("seen 2 times" in r["reason"] for r in mine)


# ---------------------------------------------------------------- the folder rule


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"parents": "+elsewhere"}, "2 parents, not one"),
        ({"owned": False}, "not owned by the account"),
        ({"trashed": True}, "in Drive's trash"),
    ],
)
def test_a_folder_must_have_one_parent_be_owned_and_untrashed(case, change, reason):
    fid = case.folders[(SLO, "empty")]
    for o in _listings(case.census, fid):
        for k, v in change.items():
            o[k] = [*o[k], "elsewhere"] if v == "+elsewhere" else v
    cls = _classes(case)
    assert (cls[fid]["class"], cls[fid]["reason"]) == ("blocker", reason)
    assert cls[SLO]["reason"] == "holds a blocker: empty"
    assert _verdicts(case)["swe-local-only"] == "no"


def test_a_folder_holding_a_blocker_is_a_blocker_up_to_its_area(case):
    for o in _listings(case.census, case.ids["t1"]):
        o["owned"] = False
    cls = _classes(case)
    theta = case.folders[(SWD, "data_processed/theta")]
    dp = case.folders[(SWD, "data_processed")]
    assert cls[theta]["reason"] == "holds a blocker: data_processed/theta/t1.parquet"
    assert cls[dp]["reason"] == "holds a blocker: data_processed/theta"
    assert cls[SWD]["reason"] == "holds a blocker: data_processed"
    assert _verdicts(case) == {
        "swe-local-only": "yes",
        "SmartWheelData": "no",
        "smart-wheel-engine-git": "yes",
    }


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"root_parents": [dc.PROJECT_FOLDER_ID, "elsewhere"]}, "2 parents, not one"),
        ({"root_owned": False}, "not owned by the account"),
    ],
)
def test_the_area_folder_itself_is_held_to_the_folder_rule(case, change, reason):
    _area_of(case.census, GIT).update(change)
    cls = _classes(case)
    assert (cls[GIT]["class"], cls[GIT]["reason"]) == ("blocker", reason)
    assert _verdicts(case)["smart-wheel-engine-git"] == "no"


# ---------------------------------------------------------------- a rich world, end to end


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    """Every kind of object that must never go, through census, plan, bytecheck and trashlist.

    swe-local-only stays clean; SmartWheelData holds every blocker; the git area holds a file
    a read-only area also shows; a file only a read-only area holds is never classed.
    """
    tmp = tmp_path_factory.mktemp("rich")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dc, "utc_now", Clock(T0))
        w = World(tmp, mp)
        _clean(w)
        rich_dir = w.dir(SWD, "rich")
        other = w.dir(SWD, "data_processed")
        w.ids["gdoc"] = w.d.special("notes", rich_dir, "application/vnd.google-apps.document")
        w.ids["link"] = w.d.special(
            "link",
            rich_dir,
            dc.SHORTCUT,
            shortcutDetails={"targetId": SWE, "targetMimeType": dc.FOLDER},
        )
        w.drive_file("nomd5", SWD, "rich/nomd5.bin", b"no md5 listed\n", md5="", sha=False)
        w.drive_file("nosha", SWD, "rich/nosha.csv", b"no sha-256 listed\n", sha=False)
        w.drive_file(
            "twoparents", SWD, "rich/two.csv", b"in two folders\n", parents=[rich_dir, other]
        )
        w.home("data/two.csv", b"in two folders\n")
        w.drive_file("notmine", SWD, "rich/notmine.csv", b"someone else's\n", owned=False)
        w.home("data/notmine.csv", b"someone else's\n")
        w.drive_file("cred", SWD, "rich/api_token.txt", b"a token\n")
        w.home("data/api_token.txt", b"a token\n")
        w.drive_file("noswe", SWD, "rich/noswe.csv", b"only on the desktop\n")
        w.home("data/noswe.csv", b"only on the desktop\n", swe=False)
        w.drive_file("emptyfile", SLO, "markers/done.flag", b"")
        w.home(f"{LEGACY}/swe-local-only/markers/done.flag", b"")
        w.drive_file("empty_noswe", SWD, "rich/empty.flag", b"")
        w.home(f"{LEGACY}/SmartWheelData/rich/empty.flag", b"", swe=False)
        w.ids["theirs_dir"] = w.d.folder("theirs", rich_dir, owned=False)
        w.ids["forked"] = w.d.folder("forked", rich_dir, parents=[rich_dir, other])
        w.drive_file(
            "shared",
            GIT,
            "objects/sh/ared",
            b"shared with a vendor area\n",
            parents=[w.dir(GIT, "objects/sh"), VENDOR],
        )
        w.home("data/shared.csv", b"shared with a vendor area\n")
        env = w.env()
        _prepare(env, bytecheck=True)
        assert _trashlist(env) == 0
        assert _trashlist(env, "again") == 0
        env["rows"] = _load(env["root"] / "_logs" / "list" / dc.PROOF)["rows"]
        env["summary"] = _load(env["root"] / "_logs" / "list" / dc.PROOF_SUMMARY)
    return env


RICH_REASONS = {
    "gdoc": "a Google-format file: no bytes to compare",
    "link": "a shortcut: never followed, never listed",
    "nomd5": "Drive lists no MD5",
    "nosha": "Drive lists no SHA-256",
    "twoparents": "2 parents, not one",
    "notmine": "not owned by the account",
    "cred": "hashes withheld by the census: a path the tool must not read",
    "noswe": "swe-data/ holds 0 objects at data/noswe.csv, not one file",
    "empty_noswe": f"swe-data/ holds 0 objects at {LEGACY}/SmartWheelData/rich/empty.flag, not one file",
    "theirs_dir": "not owned by the account",
    "forked": "2 parents, not one",
    "shared": "also seen in the read-only area day-bot-local-archive/vendor_swe_data",
}


@pytest.mark.parametrize("key", sorted(RICH_REASONS))
def test_each_kind_of_object_is_never_listed(rich, key):
    by_id = {}
    for r in rich["rows"]:
        by_id.setdefault(r["id"], []).append(r)
    rows = by_id[rich["world"].ids[key]]
    assert [r["class"] for r in rows] == ["blocker"] * len(rows)
    assert rows[0]["reason"] == RICH_REASONS[key]


def test_each_blocker_makes_its_area_no_and_a_clean_area_stays_yes(rich):
    verdicts = {
        a["name"]: (a["clear_to_trash_by_hand"], a["blockers"]) for a in rich["summary"]["areas"]
    }
    assert verdicts["swe-local-only"] == ("yes", 0)
    assert verdicts["SmartWheelData"][0] == "no" and verdicts["smart-wheel-engine-git"][0] == "no"
    ids = rich["world"].ids
    rows = {(r["area"], r["id"]): r for r in rich["rows"]}
    assert rows[("swe-local-only", ids["emptyfile"])]["class"] == "listed"  # both copies held
    rich_dir = rich["world"].folders[(SWD, "rich")]
    assert rows[("SmartWheelData", rich_dir)]["reason"].startswith("holds a blocker: rich/")
    assert ids["vendor"] not in {r["id"] for r in rich["rows"]}
    kinds = {a["name"]: a["kinds"] for a in rich["summary"]["areas"]}["SmartWheelData"]
    assert kinds["Google-format files"] == 1 and kinds["shortcuts"] == 1
    assert kinds["files without an MD5"] == 1
    kept = (rich["root"] / "_logs" / "list" / "kept.csv").read_text(encoding="utf-8")
    assert len(kept.splitlines()) == 1 + sum(1 for r in rich["rows"] if r["class"] == "blocker")


def test_an_empty_file_needs_its_desktop_copy_too(rich, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(rich["root"], root, ignore=shutil.ignore_patterns("_logs"))
    plan, census, swe = (copy.deepcopy(d) for d in _docs(rich))
    plan["root"] = str(root)
    twin = f"{LEGACY}/swe-local-only/markers/done.flag"
    (root / twin).unlink()
    rows, _ = dc.trashlist(plan, census, swe, root)
    r = next(r for r in rows if r["id"] == rich["world"].ids["emptyfile"])
    assert (r["class"], r["reason"]) == ("blocker", f"root twin {twin}: missing")


def test_the_proof_is_the_same_twice_over_the_same_inputs(rich):
    one = (rich["root"] / "_logs" / "list" / dc.PROOF).read_bytes()
    two = (rich["root"] / "_logs" / "again" / dc.PROOF).read_bytes()
    assert one == two
    assert rich["summary"]["sha256"] == hashlib.sha256(one).hexdigest()


# ---------------------------------------------------------------- the proof and its sha256


def test_the_printed_summary_names_its_inputs_and_when_it_ran(proven):
    out = proven["root"] / "_logs" / "list"
    lines = dc.trashlist_lines(proven["summary"], out)
    ins = proven["summary"]["inputs"]
    assert lines[0] == f"trashlist: ran {proven['summary']['ran']}, root {ins['root']}"
    assert lines[1:4] == [
        f"  --plan p.json (made {ins['plan']['generated']})",
        f"  --census c6.json (made {ins['census']['generated']})",
        f"  --swe-census swe.json (made {ins['swe_census']['generated']})",
    ]
    assert lines[-1] == f"trashlist: sha256 {proven['summary']['sha256']}"
    assert sum(1 for x in lines if x.startswith("  clear to trash by hand: yes")) == 3


def test_the_sha256_is_of_the_proof_bytes_alone_in_canonical_form(proven):
    out = proven["root"] / "_logs" / "list"
    data = (out / dc.PROOF).read_bytes()
    assert proven["summary"]["sha256"] == hashlib.sha256(data).hexdigest()
    doc = json.loads(data.decode("utf-8"))
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    assert (
        data == canonical.encode("utf-8") and data.endswith(b"}\n") and not data.endswith(b"\n\n")
    )
    keys = [(r["area"], r["path"], r["id"]) for r in doc["rows"]]
    assert keys == sorted(keys)
    assert set(doc) == {"kind", "rows", "schema"}  # no time of its making
    assert proven["summary"]["ran"] not in data.decode("utf-8")


def test_the_sha256_changes_with_any_row_of_any_class(case):
    for o in _listings(case.census, case.ids["t1"]):
        o["owned"] = False  # a blocker too
    rows, _ = dc.trashlist(case.plan, case.census, case.swe, case.root)
    base = hashlib.sha256(dc.proof_text(rows).encode("utf-8")).hexdigest()
    assert {r["class"] for r in rows} == {"listed", "folder", "blocker"}
    for i, r in enumerate(rows):
        for field in ("id", "class", "reason", "modified", "parents"):
            changed = copy.deepcopy(rows)
            v = changed[i][field]
            changed[i][field] = [*v, "x"] if isinstance(v, list) else f"{v}x"
            sha = hashlib.sha256(dc.proof_text(changed).encode("utf-8")).hexdigest()
            assert sha != base, (r["class"], field)
    assert hashlib.sha256(dc.proof_text(rows).encode("utf-8")).hexdigest() == base


# ---------------------------------------------------------------- trashlist refuses


def _write(tmp: Path, name: str, doc: dict) -> str:
    p = tmp / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return name


REFUSALS = {
    "an older plan": ("plan", lambda p, c, s: p.update(schema=dc.SCHEMA - 1), "schema"),
    "an older census": ("census", lambda p, c, s: c.update(schema=dc.SCHEMA - 1), "schema"),
    "an older swe-data census": ("swe", lambda p, c, s: s.update(schema=dc.SCHEMA - 1), "schema"),
    "a byte check pending": (
        "plan",
        lambda p, c, s: p["rows"][-1].update({"class": "needs-byte-check"}),
        "byte check",
    ),
    "five areas": ("plan", lambda p, c, s: p["areas"].pop(), "card 2b's six"),
    "data_raw too": (
        "plan",
        lambda p, c, s: p["areas"].append(dict(AREA["day-bot-local-archive/data_raw"])),
        "card 2b's six",
    ),
    "a mode changed": (
        "plan",
        lambda p, c, s: p["areas"][3].update(mode="consolidate"),
        "card 2b's six",
    ),
    "another census": (
        "census",
        lambda p, c, s: c.update(generated="2026-01-01T00:00:00+00:00"),
        "not the one",
    ),
    "census areas not the plan's": (
        "census",
        lambda p, c, s: c["areas"][4].update(mode="consolidate"),
        "census's areas",
    ),
    "swe-data census of another mode": (
        "swe",
        lambda p, c, s: s["areas"][0].update(mode="consolidate"),
        "--swe-census",
    ),
    "swe-data census with another area": (
        "swe",
        lambda p, c, s: s["areas"].append(copy.deepcopy(c["areas"][0])),
        "--swe-census",
    ),
    "swe-data census of another parent": (
        "swe",
        lambda p, c, s: s["areas"][0].update(parent="root"),
        "--swe-census",
    ),
    "an area folder at the top of My Drive": (
        "census",
        lambda p, c, s: c["areas"][0].update(root_parents=[c["my_drive_root"]]),
        "My Drive's root",
    ),
    "an area folder under the root alias": (
        "census",
        lambda p, c, s: c["areas"][2].update(root_parents=["root"]),
        "My Drive's root",
    ),
    "swe-data at the top of My Drive": (
        "census",
        lambda p, c, s: c["swe_data"].update(parents=[c["my_drive_root"]]),
        "My Drive's root",
    ),
    "swe-data's own census at the top": (
        "swe",
        lambda p, c, s: s["areas"][0].update(root_parents=[s["my_drive_root"]]),
        "My Drive's root",
    ),
    "a census with no chain": ("census", lambda p, c, s: c.pop("chain"), "records no chain"),
    "a census recording another swe-data": (
        "census",
        lambda p, c, s: c["swe_data"].update(id="another-folder"),
        "records swe-data/ as another-folder",
    ),
    "a census with no sharing for an area folder": (
        "census",
        lambda p, c, s: c["areas"][1].pop("sharing"),
        "records no sharing",
    ),
    "a swe-data census with no sharing for a link": (
        "swe",
        lambda p, c, s: s["chain"][2].pop("sharing"),
        "records no sharing",
    ),
}


@pytest.mark.parametrize("change", sorted(REFUSALS))
def test_trashlist_refuses(proven, change, tmp_path, capsys):
    which, edit, words = REFUSALS[change]
    plan, census, swe = copy.deepcopy(proven["docs"])
    edit(plan, census, swe)
    w = {"tmp": tmp_path, "root": proven["root"]}
    names = {
        "plan": _write(tmp_path, "p.json", plan),
        "census": _write(tmp_path, "c6.json", census),
    }
    names["swe"] = _write(tmp_path, "swe.json", swe)
    out = proven["root"] / "_logs" / ("refused-" + re.sub(r"\W+", "-", change))
    assert _trashlist(w, out_path=out, **names) == 2
    assert words in capsys.readouterr().err
    assert not out.exists()


def test_trashlist_refuses_another_root(proven, tmp_path, capsys):
    w = {"tmp": proven["tmp"], "root": proven["root"]}
    other = tmp_path / "other-root"
    other.mkdir()
    assert _trashlist(w, root=other, out_path=other / "_logs" / "x") == 2
    assert "made for root" in capsys.readouterr().err
    assert not (other / "_logs").exists()


@pytest.mark.parametrize(
    "where", ["in the live trees", "outside the root", "_logs itself", "exists"]
)
def test_trashlist_writes_only_into_a_new_folder_under_logs(proven, where, tmp_path, capsys):
    root = proven["root"]
    out = {
        "in the live trees": root / "data" / "list",
        "outside the root": tmp_path / "list",
        "_logs itself": root / "_logs",
        "exists": root / "_logs" / "list",
    }[where]
    before = sorted(p.name for p in (root / "_logs" / "list").iterdir())
    w = {"tmp": proven["tmp"], "root": root}
    assert _trashlist(w, out_path=out) == 2
    err = capsys.readouterr().err
    assert "refusing to write" in err or "_logs" in err or "exists already" in err
    assert not (root / "data" / "list").exists() and not (tmp_path / "list").exists()
    assert sorted(p.name for p in (root / "_logs" / "list").iterdir()) == before


# ---------------------------------------------------------------- recount


@pytest.fixture
def live(proven, tmp_path, monkeypatch):
    """The proven world's fake Drive, root and list, copied and live for this test alone."""
    root = tmp_path / "root"
    shutil.copytree(proven["root"], root)
    d = copy.deepcopy(proven["world"].d)
    state = tmp_path / "drive.json"
    d.save(state)
    (tmp_path / "copyid.log").write_bytes(b"")
    monkeypatch.setattr(dc, "RETRY_SLEEP", 0.0)
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)
    monkeypatch.setenv("FAKE_DRIVE", str(state))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "copyid.log"))
    monkeypatch.setattr(dc, "RCLONE", proven["rclone"])
    monkeypatch.setattr(dc, "utc_now", Clock(T0 + dt.timedelta(days=1)))
    shutil.copy(proven["tmp"] / "early.json", tmp_path / "early.json")
    # The copy stands for the proven root: its summary names this root (the sha256 is the
    # proof's, which this does not touch).
    summary_path = root / "_logs" / "list" / dc.PROOF_SUMMARY
    summary = _load(summary_path)
    summary["inputs"]["root"] = str(root)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    world_ = proven["world"]
    return SimpleNamespace(
        tmp=tmp_path,
        root=root,
        d=d,
        state=state,
        list=root / "_logs" / "list",
        ids=world_.ids,
        folders=world_.folders,
        sha=proven["summary"]["sha256"],
        save=lambda: d.save(state),
    )


def _census_now(lv, names=CONS, out="now.json") -> None:
    only = () if tuple(names) == CONS else tuple(names)
    assert _census_of({"tmp": lv.tmp}, None, out, "consolidate", only) == 0


def _recount(lv, *areas, sha=None, census="now.json", proof=None, summary=None):
    args = ["recount", "--proof", str(proof or lv.list / dc.PROOF)]
    args += ["--summary", str(summary or lv.list / dc.PROOF_SUMMARY)]
    args += ["--expect-sha256", sha or lv.sha, "--census", str(lv.tmp / census)]
    args += ["--root", str(lv.root), "--remote", "fake:"]
    for a in areas:
        args += ["--area", a]
    before = set((lv.root / "_logs").glob("recount-*.json"))
    rc = dc.main(args)
    made = set((lv.root / "_logs").glob("recount-*.json")) - before
    assert len(made) == 1  # a new file each time, never one replaced
    return rc, _load(made.pop())


def test_recount_says_unchanged_on_an_unchanged_drive(live, capsys):
    _census_now(live)
    rc, rec = _recount(live)
    assert (rc, rec["unchanged"], rec["differences"]) == (0, "yes", [])
    assert rec["areas"] == list(CONS) and rec["proof_sha256"] == live.sha
    assert "unchanged: yes" in capsys.readouterr().out
    rc, rec = _recount(live)  # twice: two records, neither replaced
    assert rc == 0


ADDED = {
    "a Google-format file": lambda lv, f: lv.d.special(
        "doc", f, "application/vnd.google-apps.document"
    ),
    "a shortcut": lambda lv, f: lv.d.special(
        "link", f, dc.SHORTCUT, shortcutDetails={"targetId": SWE}
    ),
    "a file without an MD5": lambda lv, f: lv.d.file("odd.bin", f, b"no md5\n", md5="", sha=False),
    "an empty folder": lambda lv, f: lv.d.folder("new-empty", f),
}


@pytest.mark.parametrize("what", sorted(ADDED))
def test_recount_sees_anything_added(live, what):
    ADDED[what](live, live.folders[(SLO, "data")])
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    assert (rc, rec["unchanged"]) == (1, "no")
    assert any(
        d.startswith("new since the proof: swe-local-only/data/") for d in rec["differences"]
    )


@pytest.mark.parametrize("what", ["a listed file", "an empty folder"])
def test_recount_sees_anything_removed(live, what):
    oid = live.ids["a"] if what == "a listed file" else live.folders[(SLO, "empty")]
    live.d.objects.remove(live.d.obj(oid))
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    assert rc == 1 and any(
        d.startswith("gone since the proof") and oid in d for d in rec["differences"]
    )


def test_recount_sees_a_file_replaced_by_another_of_the_same_size(live):
    old = live.d.obj(live.ids["a"])
    live.d.objects.remove(old)
    new = live.d.file("a.csv", old["parents"][0], b"ALPHA\n")
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    assert rc == 1
    assert any("gone since the proof" in d and live.ids["a"] in d for d in rec["differences"])
    assert any("new since the proof" in d and new in d for d in rec["differences"])


@pytest.mark.parametrize(
    "field", ["modified", "parents", "owned", "name", "size", "md5", "sha256", "mime"]
)
def test_recount_sees_a_change_in_place(live, field):
    o = live.d.obj(live.ids["a"])
    if field == "modified":
        o["modifiedTime"] = "2026-10-16T00:00:00Z"
    elif field == "parents":
        o["parents"] = [live.folders[(SLO, "empty")]]
    elif field == "owned":
        o["owners"] = tdc.SOMEONE_ELSE
    elif field == "name":
        o["name"] = "b.csv"
    elif field == "size":
        o["size"] = "7"
    elif field == "md5":
        o["md5Checksum"] = "d" * 32
    elif field == "mime":
        o["mimeType"] = "text/plain"
    else:
        o["sha256Checksum"] = "e" * 64
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    assert rc == 1
    assert any(
        d.startswith("changed: swe-local-only/data/a.csv") and field in d
        for d in rec["differences"]
    ), rec


def test_recount_rehashes_a_desktop_twin_whose_size_and_time_are_kept(live):
    p = live.root / "data" / "a.csv"
    st = p.stat()
    p.write_bytes(b"ALPHA\n")  # the same size
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    _census_now(live)
    rc, rec = _recount(live)
    assert rc == 1
    assert rec["differences"] == [
        "swe-local-only/data/a.csv: its desktop twin data/a.csv: its MD5, SHA-256 changed"
    ]


@pytest.mark.parametrize(
    "change", ["changed", "removed", "trashed", "moved", "a second parent", "renamed"]
)
def test_recount_reads_each_swe_data_twin_again(live, change):
    o = live.d.obj(live.ids["swe:data/a.csv"])
    if change == "changed":
        o["md5Checksum"] = hashlib.md5(b"ALPHA\n").hexdigest()
        o["sha256Checksum"] = hashlib.sha256(b"ALPHA\n").hexdigest()
    elif change == "removed":
        live.d.objects.remove(o)
    elif change == "trashed":
        o["trashed"] = True
    elif change == "a second parent":
        o["parents"] = [*o["parents"], live.folders[(SWE, LEGACY)]]
    elif change == "renamed":
        o["name"] = "a-renamed.csv"
    else:
        o["parents"] = [live.folders[(SWE, LEGACY)]]
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    assert rc == 1
    words = {
        "changed": "MD5, SHA-256 changed",
        "removed": "is gone",
        "trashed": "in Drive's trash",
        "moved": "is gone",
        "a second parent": "its parents changed",
        "renamed": "its name changed",
    }
    assert any(
        "its swe-data/ twin data/a.csv" in d and words[change] in d for d in rec["differences"]
    ), rec


def test_recount_refuses_another_sha256_or_a_summary_not_the_proofs(live):
    _census_now(live)
    rc, rec = _recount(live, sha="0" * 64)
    assert rc == 1 and rec["differences"][0].startswith("the proof's sha256 is")
    summary = _load(live.list / dc.PROOF_SUMMARY)
    summary["sha256"] = "1" * 64
    other = live.tmp / "other_summary.json"
    other.write_text(json.dumps(summary), encoding="utf-8")
    rc, rec = _recount(live, summary=other)
    assert rc == 1 and rec["differences"] == [
        f"the summary is not this proof's: it names sha256 {'1' * 64}"
    ]


def test_recount_needs_a_census_made_after_the_proof(live):
    rc, rec = _recount(live, census="early.json")  # made before the proof, of the same areas
    assert rc == 1 and len(rec["differences"]) == 1
    assert rec["differences"][0].startswith("the census started at ")
    assert "not after the proof was made" in rec["differences"][0]


@pytest.mark.parametrize("given, checked", [(CONS, ["swe-local-only"]), (CONS[:1], [])])
def test_recount_needs_a_census_of_exactly_the_areas_checked(live, given, checked):
    _census_now(live, given)
    rc, rec = _recount(live, *checked)
    assert rc == 1 and rec["differences"][0].startswith("the census's areas are not exactly")


def test_recount_refuses_an_area_with_a_blocker_and_checks_one_area_alone(live):
    gdoc = live.d.special(
        "notes", live.folders[(SWD, "data_processed")], "application/vnd.google-apps.document"
    )
    live.save()
    w = {"tmp": live.tmp, "root": live.root}
    _prepare(w)
    assert _trashlist(w, "list-with-a-blocker") == 0
    out = live.root / "_logs" / "list-with-a-blocker"
    sha = _load(out / dc.PROOF_SUMMARY)["sha256"]
    rows = _load(out / dc.PROOF)["rows"]
    assert next(r for r in rows if r["id"] == gdoc)["class"] == "blocker"
    _census_now(live)
    paths = {"proof": out / dc.PROOF, "summary": out / dc.PROOF_SUMMARY, "sha": sha}
    rc, rec = _recount(live, **paths)
    # The Google file, the folder that holds it and the area folder: three blocker rows.
    assert rc == 1
    assert rec["differences"] == [
        "area SmartWheelData: the proof has 3 row(s) neither listed nor a folder: not clear "
        "to trash by hand"
    ]
    _census_now(live, ["swe-local-only"], "slo.json")
    rc, rec = _recount(live, "swe-local-only", census="slo.json", **paths)
    assert (rc, rec["unchanged"], rec["areas"]) == (0, "yes", ["swe-local-only"])


def test_recount_hashes_the_twins_in_the_proofs_own_root(live, tmp_path):
    other = tmp_path / "an-older-copy"
    shutil.copytree(live.root, other)
    (live.root / "data" / "a.csv").write_bytes(b"ALPHA\n")  # the real root's twin changed
    _census_now(live)
    rc, _rec = _recount(live)
    assert rc == 1
    args = ["recount", "--proof", str(live.list / dc.PROOF)]
    args += ["--summary", str(live.list / dc.PROOF_SUMMARY), "--expect-sha256", live.sha]
    args += ["--census", str(live.tmp / "now.json"), "--root", str(other), "--remote", "fake:"]
    assert dc.main(args) == 1
    rec = _load(next((other / "_logs").glob("recount-*.json")))
    assert rec["differences"] == [f"the proof was made for root {live.root}, not {other}"]


def test_recount_compares_the_area_folders_name_as_drive_lists_it(live):
    _census_now(live)
    now = _load(live.tmp / "now.json")
    _area_of(now, SLO)["root_name"] = "Swe-Local-Only"  # Drive's name = '…' ignores case
    (live.tmp / "now.json").write_text(json.dumps(now), encoding="utf-8")
    rc, rec = _recount(live)
    assert rc == 1
    assert rec["differences"] == [f"changed: swe-local-only (the area folder) ({SLO}): its name"]


def test_recount_refuses_an_area_that_is_not_a_consolidate_area(live, capsys):
    _census_now(live)
    rc = dc.main(
        [
            "recount",
            "--proof",
            str(live.list / dc.PROOF),
            "--summary",
            str(live.list / dc.PROOF_SUMMARY),
            "--expect-sha256",
            live.sha,
            "--census",
            str(live.tmp / "now.json"),
            "--root",
            str(live.root),
            "--remote",
            "fake:",
            "--area",
            "day-bot-local-archive/vendor_swe_data",
        ]
    )
    assert rc == 2 and "not one of the consolidate areas" in capsys.readouterr().err


# ---------------------------------------------------------------- the verbs rclone is given


def _check_call(call: list[str]) -> None:
    """One rclone call: an allowed verb, with only the flags that verb may carry."""
    if call == ["version"]:
        return
    if call[:2] == ["backend", "query"]:
        flags = {a for a in call[2:] if a.startswith("-")}
        assert dc.OWNER_ONLY in flags and flags <= {dc.OWNER_ONLY, dc.SHARING_FLAG}, call
        pos = [a for a in call[2:] if not a.startswith("-")]
        assert len(pos) == 2, call
        if dc.SHARING_FLAG in flags:  # the sharing read, and only that read
            assert re.fullmatch(r"'[^']+' in parents( and name = '.*')?", pos[1]), call
        return
    if call[:1] == ["lsjson"]:  # the census's cross-check, and only that
        assert call == [
            "lsjson",
            "-R",
            "--disable",
            "ListR",
            "--drive-root-folder-id",
            call[5],
            "--drive-skip-shortcuts",
            "--drive-show-all-gdocs",
            "--hash-type",
            "MD5",
            call[-1],
        ], call
        return
    if call[:2] == ["backend", "copyid"]:
        pairs = call[3:]
        assert pairs and len(pairs) % 2 == 0 and not any(a.startswith("-") for a in call[2:]), call
        for dest in pairs[1::2]:
            assert os.path.isabs(dest) and ":" not in dest.replace(
                os.path.splitdrive(dest)[0], ""
            ), call
        return
    raise AssertionError(f"a verb outside the allowlist: {call}")


def _spy(monkeypatch) -> list[list[str]]:
    """Every rclone call the tool makes from now on, without the command itself."""
    seen: list[list[str]] = []
    real = dc.run_cmd

    def spy(cmd, stdin=None):
        assert cmd[: len(dc.RCLONE)] == dc.RCLONE
        seen.append(list(cmd[len(dc.RCLONE) :]))
        return real(cmd, stdin)

    monkeypatch.setattr(dc, "run_cmd", spy)
    return seen


def _verbs(seen: list[list[str]]) -> set[str]:
    return {" ".join(c[:2]) if c[0] == "backend" else c[0] for c in seen}


def test_the_d33_commands_run_rclone_only_with_allowed_verbs(world, monkeypatch):  # noqa: F811
    seen = _spy(monkeypatch)
    tdc._plan(world)  # census, inventory, plan
    tdc._bytecheck(world)
    assert tdc._copy(world) == 0 and tdc._verify(world) == 0
    assert _verbs(seen) == {"version", "backend query", "lsjson", "backend copyid"}
    for call in seen:
        _check_call(call)


def test_card_3s_commands_run_rclone_only_with_allowed_verbs(live, monkeypatch):
    seen = _spy(monkeypatch)
    _census_now(live)  # the census command: the walk, the chain, swe-data/ and sharing
    assert _recount(live)[0] == 0
    assert _verbs(seen) == {"version", "backend query", "lsjson"}  # trashlist calls none
    for call in seen:
        _check_call(call)
    sharing = [c for c in seen if dc.SHARING_FLAG in c]
    assert len(sharing) == 3 + 1 + 3  # the chain's links, swe-data/, the three area folders


def test_no_rclone_verb_in_the_source_is_outside_the_allowlist():
    src = (tdc._REPO / "scripts" / "drive_consolidate.py").read_text(encoding="utf-8")
    calls = re.findall(r"\*RCLONE,((?:\s*\"[^\"]*\",?)+)", src)
    assert len(calls) == len(re.findall(r"\*RCLONE,", src)) >= 5
    verbs = set()
    for args in calls:
        words = re.findall(r"\"([^\"]*)\"", args)
        verbs.add(" ".join(words[:2]) if words[0] == "backend" else words[0])
    assert verbs == {"version", "backend query", "lsjson", "backend copyid"}, verbs
    for word in (
        "deletefile",
        "delete",
        "purge",
        "rmdir",
        "moveid",
        "move",
        "moveto",
        "sync",
        "copyto",
        "mkdir",
        "untrash",
        "cleanup",
        "dedupe",
        "rescue",
        "touch",
        "settier",
    ):
        assert not re.search(rf"\"{word}\"", src), word


# ---------------------------------------------------------------- fix round 1 (the pen's A to F)


# G. Each census is cross-checked against rclone's own listing, one folder per query.


@pytest.mark.parametrize(
    "what", ["a Google-format file", "a folder", "a file without an MD5", "a file with an MD5"]
)
def test_the_cross_check_stops_a_census_whose_batches_missed_an_object(
    small, monkeypatch, capsys, what
):
    w = small["world"]
    w.dir(SLO, "empty")  # with data/, the walk's second round is an OR of two parents
    data = w.dir(SLO, "data")
    if what == "a folder":
        lost, kind = w.d.folder("sub", data), "folders"
    elif what == "a file without an MD5":
        lost = w.d.file("odd.bin", data, b"no md5", md5="", sha=False)
        kind = "files without an MD5"
    elif what == "a file with an MD5":  # its count would differ too: the ids are named first
        lost, kind = w.d.file("extra.csv", data, b"extra bytes"), "files with an MD5"
    else:
        lost = w.d.special("notes", data, "application/vnd.google-apps.document")
        kind = "Google-format files"
    w.d.save(small["tmp"] / "drive.json")
    monkeypatch.setenv("FAKE_DROP_IN_OR", lost)  # Drive answers that OR, but without it
    assert tdc._census(small) == 2
    err = capsys.readouterr().err
    assert f"{kind} only rclone lists: {lost}" in err, err
    assert not (small["tmp"] / "c.json").exists()


@pytest.mark.parametrize(
    "what", ["a clean area", "a Google Form", "a file without an MD5", "a shortcut"]
)
def test_the_cross_check_passes_a_whole_census(small, what):
    w = small["world"]
    data = w.dir(SLO, "data")
    if what == "a Google Form":  # rclone lists it only with --drive-show-all-gdocs
        w.d.special("survey", data, "application/vnd.google-apps.form")
    elif what == "a file without an MD5":
        w.d.file("odd.bin", data, b"no md5", md5="", sha=False)
    elif what == "a shortcut":  # left out of both on purpose: trashing one deletes no bytes
        w.d.special("link", data, dc.SHORTCUT, shortcutDetails={"targetId": SWE})
    w.d.save(small["tmp"] / "drive.json")
    assert tdc._census(small) == 0
    doc = _load(small["tmp"] / "c.json")
    assert doc["cross_checked"] == dc.CROSS_CHECK
    cross = doc["areas"][0]["cross_check"]
    assert cross["files with an MD5"] == 2 and cross["folders"] == 1
    assert cross["Google-format files"] == (what == "a Google Form")
    assert cross["files without an MD5"] == (what == "a file without an MD5")


def test_the_census_records_its_start_before_its_first_query(small, monkeypatch):
    events: list[tuple[str, object]] = []
    clock = Clock(T0)

    def now():
        value = clock()
        events.append(("clock", value))
        return value

    real = dc.run_cmd

    def run(cmd, stdin=None):
        events.append(("call", list(cmd[len(dc.RCLONE) :][:2])))
        return real(cmd, stdin)

    monkeypatch.setattr(dc, "utc_now", now)
    monkeypatch.setattr(dc, "run_cmd", run)
    assert tdc._census(small) == 0
    doc = _load(small["tmp"] / "c.json")
    first = next(i for i, e in enumerate(events) if e == ("call", ["backend", "query"]))
    before = [v for kind, v in events[:first] if kind == "clock"]
    assert before and doc["started"] == before[-1]
    assert dc._later(doc["generated"], doc["started"])


def test_the_census_walk_keeps_its_batches(small):
    w = small["world"]
    w.dir(SLO, "empty")
    w.d.save(small["tmp"] / "drive.json")
    small["calls"].write_text("", encoding="utf-8")
    dc.census("fake:", [AREA["swe-local-only"]])  # what _drifted runs before copy, too
    queries = [_query_of(c) for c in _calls(small["calls"]) if c[:2] == ["backend", "query"]]
    assert any(" or " in q for q in queries)
    assert [c[0] for c in _calls(small["calls"]) if c[0] != "backend"] == ["lsjson"]


def test_trashlist_refuses_a_census_without_its_cross_check(proven, tmp_path, capsys):
    for drop in ("the census's record", "an area's record"):
        plan, census, swe = copy.deepcopy(proven["docs"])
        if drop == "the census's record":
            census.pop("cross_checked")
        else:
            census["areas"][2].pop("cross_check")
        w = {"tmp": tmp_path, "root": proven["root"]}
        names = {
            "plan": _write(tmp_path, "p.json", plan),
            "census": _write(tmp_path, "c6.json", census),
        }
        names["swe"] = _write(tmp_path, "swe.json", swe)
        out = proven["root"] / "_logs" / "refused-unchecked"
        assert _trashlist(w, out_path=out, **names) == 2, drop
        assert "no cross-check against rclone's listing" in capsys.readouterr().err
        assert not out.exists()


def test_recount_needs_a_cross_checked_census(live):
    _census_now(live)
    census = _load(live.tmp / "now.json")
    census.pop("cross_checked")
    (live.tmp / "now.json").write_text(json.dumps(census), encoding="utf-8")
    rc, rec = _recount(live)
    assert rc == 1
    assert rec["differences"] == ["the census records no cross-check against rclone's listing"]


# B. The verdict fails closed.


def test_a_verdict_is_yes_only_when_every_row_is_listed_or_a_folder(case):
    rows, _ = dc.trashlist(case.plan, case.census, case.swe, case.root)
    area = _area_of(case.census, SLO)
    mine = [r for r in rows if r["area"] == "swe-local-only"]
    assert dc._area_summary(area, mine)["clear_to_trash_by_hand"] == "yes"
    odd = copy.deepcopy(mine)
    next(r for r in odd if r["class"] == "listed")["class"] = "undecided"  # neither, nor blocker
    s = dc._area_summary(area, odd)
    assert (s["clear_to_trash_by_hand"], s["blockers"]) == ("no", 1)
    no_top = [r for r in mine if r["path"] != ""]  # every row clean, but the folder's own missing
    assert dc._area_summary(area, no_top)["clear_to_trash_by_hand"] == "no"
    assert dc._area_summary(area, [])["clear_to_trash_by_hand"] == "no"


def test_recount_says_no_for_a_row_neither_listed_nor_a_folder(live):
    proof = _load(live.list / dc.PROOF)
    next(r for r in proof["rows"] if r["id"] == live.ids["a"])["class"] = "undecided"
    text = dc.proof_text(proof["rows"])
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (live.tmp / "odd.json").write_bytes(text.encode("utf-8"))
    summary = _load(live.list / dc.PROOF_SUMMARY)
    summary["sha256"] = sha
    (live.tmp / "odd_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _census_now(live)
    rc, rec = _recount(
        live, sha=sha, proof=live.tmp / "odd.json", summary=live.tmp / "odd_summary.json"
    )
    assert rc == 1
    assert rec["differences"] == [
        "area swe-local-only: the proof has 1 row(s) neither listed nor a folder: not clear "
        "to trash by hand"
    ]


# C. Freshness: a census from the 24 hours before the recount, and later than the proof.


@pytest.mark.parametrize(
    "case_, start_after_proof, end_after_start, now_after_start, unchanged, why",
    [
        ("started at the proof time itself", 0, 60, 3600, "no", "not after the proof"),
        ("started before the proof was made", -60, 120, 3600, "no", "not after the proof"),
        ("started just after the proof", 1, 60, 3600, "yes", None),
        ("started 24 hours before", 1, 60, 24 * 3600, "yes", None),
        ("started just over 24 hours before", 1, 3600, 24 * 3600 + 1, "no", "not within"),
        ("dated after the recount", 1, 60, -10, "no", "not within"),
        ("finished before its start", 60, -70, 3600, "no", "finished"),
    ],
)
def test_recount_counts_freshness_from_the_census_start(
    live,
    monkeypatch,
    capsys,
    case_,
    start_after_proof,
    end_after_start,
    now_after_start,
    unchanged,
    why,
):
    _census_now(live)
    made = dt.datetime.fromisoformat(_load(live.list / dc.PROOF_SUMMARY)["ran"])
    started = made + dt.timedelta(seconds=start_after_proof)
    finished = started + dt.timedelta(seconds=end_after_start)
    now = started + dt.timedelta(seconds=now_after_start)
    census = _load(live.tmp / "now.json")
    census["started"], census["generated"] = started.isoformat(), finished.isoformat()
    (live.tmp / "now.json").write_text(json.dumps(census), encoding="utf-8")
    monkeypatch.setattr(dc, "utc_now", lambda: now.isoformat())
    rc, rec = _recount(live)
    assert (rec["unchanged"], rc) == (unchanged, 0 if unchanged == "yes" else 1), rec
    assert (rec["census_started"], rec["census_generated"]) == (
        started.isoformat(),
        finished.isoformat(),
    )
    assert rec["census_age_seconds"] == now_after_start
    out = capsys.readouterr().out
    assert f"unchanged: {unchanged} (census started {started.isoformat()}, " in out
    assert f"; finished {finished.isoformat()})" in out
    if why == "not after the proof":
        assert rec["differences"] == [
            f"the census started at {started.isoformat()}, not after the proof was made "
            f"({made.isoformat()})"
        ]
    if case_ == "started just over 24 hours before":  # though it finished within them
        assert rec["differences"] == [
            f"the census started at {started.isoformat()}, not within the 24 hours before this "
            f"recount ({now.isoformat()})"
        ]
    if case_ == "finished before its start":
        assert rec["differences"] == [
            f"the census finished at {finished.isoformat()}, not between its start and this recount"
        ]
    if case_ == "dated after the recount":
        assert rec["differences"] == [
            f"the census started at {started.isoformat()}, not within the 24 hours before this "
            f"recount ({now.isoformat()})",
            f"the census finished at {finished.isoformat()}, not between its start and this "
            "recount",
        ]


# D. The allowlist, on every call the fake rclone receives.


def test_trashlist_gives_rclone_nothing(live):
    w = {"tmp": live.tmp, "root": live.root}
    _prepare(w)
    before = len(tdc.fake_calls())
    assert _trashlist(w, "no-calls") == 0
    assert tdc.fake_calls()[before:] == []


def test_recount_gives_rclone_only_allowed_verbs(live):
    _census_now(live)
    before = len(tdc.fake_calls())
    assert _recount(live)[0] == 0
    calls = tdc.fake_calls()[before:]
    assert {tdc.verb_of(c) for c in calls} == {"version", "backend query"}
    for call in calls:
        _check_call(call)


def test_no_process_or_rclone_call_in_the_source_is_outside_the_allowlist():
    import ast

    src = (tdc._REPO / "scripts" / "drive_consolidate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    up = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def function_of(node):
        while node in up:
            node = up[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    def led_by_rclone(lst) -> bool:
        first = lst.elts[0] if isinstance(lst, ast.List) and lst.elts else None
        return isinstance(first, ast.Starred) and getattr(first.value, "id", "") == "RCLONE"

    def verb(lst) -> str:
        words = []
        for e in lst.elts[1:]:
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                break
            words.append(e.value)
        return " ".join(words[:2]) if words[:1] == ["backend"] else "".join(words[:1])

    problems = []
    for node in ast.walk(tree):
        where = getattr(node, "lineno", "?")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            f = function_of(node)
            if node.value.id == "subprocess" and (f is None or f.name != "run_cmd"):
                problems.append(f"subprocess outside run_cmd at line {where}")
            launch = node.attr in {"system", "popen", "startfile"}
            if node.value.id == "os" and (launch or node.attr.startswith(("exec", "spawn"))):
                problems.append(f"os.{node.attr} at line {where}")
        if isinstance(node, ast.Name) and node.id == "RCLONE" and isinstance(node.ctx, ast.Load):
            par = up[node]
            if isinstance(par, ast.Starred) and led_by_rclone(up[par]):
                if verb(up[par]) not in tdc.ALLOWED_VERBS:
                    problems.append(f"rclone verb {verb(up[par])!r} at line {where}")
            else:  # joined into a string, added to, indexed, passed on: refused
                problems.append(f"RCLONE used another way at line {where}")
        if isinstance(node, ast.Constant) and node.value == "rclone":
            owner = up.get(up.get(node))
            key = isinstance(up.get(node), ast.Subscript)  # a document's key: doc["rclone"]
            if not key and not (
                isinstance(owner, ast.AnnAssign) and getattr(owner.target, "id", "") == "RCLONE"
            ):
                problems.append(f"the string 'rclone' at line {where}")
        if isinstance(node, ast.keyword) and node.arg == "shell":
            problems.append(f"shell= at line {getattr(up.get(node), 'lineno', '?')}")
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_cmd":
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.List):
                ok = led_by_rclone(arg)
            elif isinstance(arg, ast.Name):
                sets = [
                    n.value
                    for n in ast.walk(function_of(node) or tree)
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == arg.id for t in n.targets)
                ]
                ok = bool(sets) and all(led_by_rclone(v) for v in sets)
            else:
                ok = False
            if not ok:
                problems.append(f"run_cmd not given an [*RCLONE, verb, ...] list at line {where}")
    assert not problems, problems
    assert sum(1 for n in ast.walk(tree) if getattr(n, "id", "") == "RCLONE") >= 6


# E1. The folder rule's own registrations.


def _folder_obj(oid: str, name: str, parent: str) -> dict:
    return {
        "id": oid,
        "name": name,
        "mime": dc.FOLDER,
        "size": None,
        "md5": None,
        "sha256": None,
        "parents": [parent],
        "created": None,
        "modified": "2026-07-03T00:00:00Z",
        "target": None,
        "target_mime": None,
        "owned": True,
        "trashed": False,
    }


def test_an_empty_folder_a_read_only_area_also_shows_is_a_blocker(case):
    fid = case.folders[(SLO, "empty")]
    _area_of(case.census, VENDOR)["objects"].append(_folder_obj(fid, "empty", VENDOR))
    cls = _classes(case)
    assert (
        cls[fid]["reason"]
        == "also seen in the read-only area day-bot-local-archive/vendor_swe_data"
    )
    assert _verdicts(case)["swe-local-only"] == "no"


def test_an_empty_folder_in_two_consolidate_areas_is_a_blocker_in_both(case):
    fid = case.folders[(SLO, "empty")]
    _area_of(case.census, GIT)["objects"].append(_folder_obj(fid, "empty", GIT))
    rows, areas = dc.trashlist(case.plan, case.census, case.swe, case.root)
    mine = [r for r in rows if r["id"] == fid]
    assert [(r["area"], r["class"]) for r in mine] == [
        ("smart-wheel-engine-git", "blocker"),
        ("swe-local-only", "blocker"),
    ]
    assert all(r["reason"] == "seen 2 times in the consolidate areas" for r in mine)
    assert [a["clear_to_trash_by_hand"] for a in areas] == ["no", "yes", "no"]


def test_a_read_only_areas_own_folder_inside_a_consolidate_area_is_a_blocker(case):
    _area_of(case.census, VENDOR)["objects"] = []
    _area_of(case.census, SLO)["objects"].append(_folder_obj(VENDOR, "vendor_swe_data", SLO))
    cls = _classes(case)
    assert cls[VENDOR]["area"] == "swe-local-only" and cls[VENDOR]["class"] == "blocker"
    assert (
        cls[VENDOR]["reason"]
        == "also seen in the read-only area day-bot-local-archive/vendor_swe_data"
    )
    assert _verdicts(case)["swe-local-only"] == "no"


def test_a_consolidate_area_folder_a_read_only_area_shows_is_a_blocker(case):
    _area_of(case.census, VENDOR)["objects"].append(_folder_obj(SLO, "swe-local-only", VENDOR))
    cls = _classes(case)
    assert cls[SLO]["class"] == "blocker"
    assert (
        cls[SLO]["reason"]
        == "also seen in the read-only area day-bot-local-archive/vendor_swe_data"
    )
    assert _verdicts(case)["swe-local-only"] == "no"


# E2, E3 and F. Owners, sharing and Drive's own names, end to end.

OTHERS = [
    {
        "id": "p-user",
        "type": "user",
        "role": "writer",
        "emailAddress": "someone@example.invalid",
        "displayName": "Some One",
    },
    {
        "id": "p-group",
        "type": "group",
        "role": "reader",
        "emailAddress": "team@example.invalid",
        "displayName": "A Team",
    },
    {"id": "p-domain", "type": "domain", "role": "reader", "domain": "example.org"},
    {"id": "anyoneWithLink", "type": "anyone", "role": "reader", "allowFileDiscovery": False},
]


@pytest.fixture(scope="module")
def odd(tmp_path_factory):
    """A world whose git area folder another account owns, and whose swe-local-only folder
    is shared beyond its owner: with a user, a group, a domain and anyone with the link."""
    tmp = tmp_path_factory.mktemp("odd")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dc, "utc_now", Clock(T0))
        w = World(tmp, mp)
        _clean(w)
        w.d.obj(GIT)["owners"] = tdc.SOMEONE_ELSE
        w.d.obj(SLO)["permissions"] = [*tdc.OWNER_ONLY_PERMS, *OTHERS]
        env = w.env()
        _prepare(env)
        assert _trashlist(env) == 0
        out = env["root"] / "_logs" / "list"
        env["summary"] = _load(out / dc.PROOF_SUMMARY)
        env["rows"] = _load(out / dc.PROOF)["rows"]
        env["sharing_text"] = (out / "sharing.json").read_text(encoding="utf-8")
        env["census"] = _load(env["tmp"] / "c6.json")
    return env


def test_an_area_folder_another_account_owns_is_a_blocker_from_the_census_on(odd):
    assert _area_of(odd["census"], GIT)["root_owned"] is False
    top = next(r for r in odd["rows"] if r["id"] == GIT)
    assert (top["class"], top["reason"], top["owned"]) == (
        "blocker",
        "not owned by the account",
        False,
    )
    verdicts = {a["name"]: a["clear_to_trash_by_hand"] for a in odd["summary"]["areas"]}
    assert verdicts == {
        "swe-local-only": "yes",  # shared, which is recorded beside the verdict, not in it
        "SmartWheelData": "yes",
        "smart-wheel-engine-git": "no",
    }


def test_sharing_beyond_the_owner_is_flagged_with_no_email_or_name(odd, capsys):
    slo = next(a for a in odd["summary"]["areas"] if a["name"] == "swe-local-only")
    want = {
        "user writer [permission p-user]",
        "group reader [permission p-group]",
        "domain reader (example.org) [permission p-domain]",
        "anyone reader [permission anyoneWithLink]",
    }
    assert set(slo["sharing_flags"]) == want and len(slo["sharing_flags"]) == 4
    sharing = json.loads(odd["sharing_text"])
    rec = next(a for a in sharing["areas"] if a["id"] == SLO)
    assert set(rec["flags"]) == want
    assert {p["id"] for p in rec["sharing"]} == {"perm-owner", *(p["id"] for p in OTHERS)}
    assert "@" not in odd["sharing_text"] and "Some One" not in odd["sharing_text"]
    assert "A Team" not in odd["sharing_text"] and "The Owner" not in odd["sharing_text"]
    others = [a for a in odd["summary"]["areas"] if a["name"] != "swe-local-only"]
    assert all(a["sharing_flags"] == [] for a in others)
    lines = dc.trashlist_lines(odd["summary"], odd["root"] / "_logs" / "list")
    assert any(x.startswith("  sharing: ") and "anyone reader" in x for x in lines)
    assert "  sharing: the owner alone" in lines


def test_sharing_json_names_each_area_folder_as_drive_does(odd):
    sharing = json.loads(odd["sharing_text"])
    names = {a["area"]: a["name"] for a in sharing["areas"]}
    assert names["smart-wheel-engine-git"] == "smart-wheel-engine"  # Drive's, not the tool's
    assert names["swe-local-only"] == "swe-local-only"
    assert len(names) == 6


@pytest.mark.parametrize("what", ["a file", "the area folder"])
def test_an_object_drive_returns_without_owners_is_not_owned(small, what):
    w = small["world"]
    oid = w.ids["mine"] if what == "a file" else SLO
    del w.d.obj(oid)["owners"]
    w.d.save(small["tmp"] / "drive.json")
    assert tdc._census(small) == 0
    area = _load(small["tmp"] / "c.json")["areas"][0]
    if what == "a file":
        assert next(o for o in area["objects"] if o["id"] == oid)["owned"] is None
    else:
        assert area["root_owned"] is None


def test_an_object_without_owners_never_reaches_listed(case):
    for o in _listings(case.census, case.ids["a"]):
        o["owned"] = None
    _area_of(case.census, GIT)["root_owned"] = None
    cls = _classes(case)
    assert (cls[case.ids["a"]]["class"], cls[case.ids["a"]]["reason"]) == (
        "blocker",
        "owner not recorded",
    )
    assert cls[GIT]["reason"] == "owner not recorded"


# E4. The desktop twin's guards.


def _twin_to(c, rel: str) -> None:
    for r in _plan_rows(c.plan, c.ids["a"]):
        r["twin"] = rel


@pytest.mark.parametrize(
    "guard, rel, reason",
    [
        ("a plain path", "data/./a.csv", "not a plain path under the root"),
        ("not under _logs", "_logs/a.csv", "under the root's _logs/"),
        ("no credential shape", "data/token_a.csv", "a credential-shaped name, never read"),
    ],
)
def test_a_desktop_twin_must_be_a_plain_readable_path(case, guard, rel, reason):
    body = (case.root / "data" / "a.csv").read_bytes()
    (case.root / "_logs").mkdir(exist_ok=True)
    (case.root / "_logs" / "a.csv").write_bytes(body)
    (case.root / "data" / "token_a.csv").write_bytes(body)
    _twin_to(case, rel)
    r = _classes(case)[case.ids["a"]]
    assert (r["class"], r["reason"]) == ("blocker", f"root twin {rel}: {reason}")


def test_a_desktop_twin_through_a_link_or_junction_is_never_read(case):
    real = case.root / "real-data"
    (case.root / "data").rename(real)
    tdc._symlink_or_skip(case.root / "data", real, is_dir=True)
    r = _classes(case)[case.ids["a"]]
    assert r["class"] == "blocker"
    assert r["reason"].startswith("root twin data/a.csv: a link or junction on its path")


# E5. The two "listed twice" checks.


def test_the_census_stops_if_swe_data_is_listed_twice(small, monkeypatch, capsys):
    real = dc.drive_query

    def twice(remote, query, tries=3, sharing=False):
        found = real(remote, query, tries, sharing)
        if query == f"'{dc.PROJECT_FOLDER_ID}' in parents":
            found += [h for h in found if h["id"] == SWE]
        return found

    monkeypatch.setattr(dc, "drive_query", twice)
    assert tdc._census(small) == 2
    assert f"swe-data/ (id {SWE}) is listed 2 times" in capsys.readouterr().err


def test_recount_says_no_for_a_swe_data_twin_listed_twice_differently(live, monkeypatch):
    _census_now(live)
    parent = live.folders[(SWE, "data")]
    real = dc.drive_query

    def twice(remote, query, tries=3, sharing=False):
        found = real(remote, query, tries, sharing)
        if not sharing and f"'{parent}' in parents" in query and "trashed" not in query:
            twin = next(h for h in found if h["id"] == live.ids["swe:data/a.csv"])
            found.append({**twin, "md5Checksum": "0" * 32})  # after the true one
        return found

    monkeypatch.setattr(dc, "drive_query", twice)
    rc, rec = _recount(live)
    assert rc == 1
    sid = live.ids["swe:data/a.csv"]
    assert rec["differences"] == [
        f"swe-local-only/data/a.csv: its swe-data/ twin data/a.csv ({sid}) was listed twice, "
        "differently"
    ]


# E6. A folder above a swe-data/ twin goes to the trash.


def test_a_swe_data_twin_whose_folder_is_trashed_fails_both_checks(live):
    live.d.obj(live.folders[(SWE, "data")])["trashed"] = True  # Drive trashes what it holds too
    live.save()
    _census_now(live)
    rc, rec = _recount(live)
    sid = live.ids["swe:data/a.csv"]
    assert rc == 1
    assert rec["differences"] == [
        f"swe-local-only/data/a.csv: its swe-data/ twin data/a.csv ({sid}) is in Drive's trash"
    ]
    w = {"tmp": live.tmp, "root": live.root}
    _prepare(w)
    assert _trashlist(w, "after-the-trash") == 0
    rows = _load(live.root / "_logs" / "after-the-trash" / dc.PROOF)["rows"]
    r = next(r for r in rows if r["id"] == live.ids["a"])
    assert (r["class"], r["reason"]) == (
        "blocker",
        "swe-data/ holds 0 objects at data/a.csv, not one file",
    )


# I. Presets: card 3a cannot make the wrong census by hand.


@pytest.mark.parametrize("preset", sorted(dc.PRESETS))
def test_each_preset_gives_exactly_its_areas(small, preset):
    assert _census_of(small, None, f"{preset}.json", preset) == 0
    doc = _load(small["tmp"] / f"{preset}.json")
    got = [(a["name"], a["id"], a["parent"], a["folder"], a["mode"]) for a in doc["areas"]]
    want = {
        "card2b": [a for a in dc.DEFAULT_AREAS if a["name"] != "day-bot-local-archive/data_raw"],
        "consolidate": [a for a in dc.DEFAULT_AREAS if a["mode"] == "consolidate"],
        "swe-data": [dc.SWE_DATA],
    }[preset]
    assert got == [(a["name"], a["id"], a["parent"], a["folder"], a["mode"]) for a in want]
    assert doc["preset"] == preset


def test_a_preset_is_refused_with_areas(small, capsys):
    args = ["census", "--remote", "fake:", "--areas", str(small["areas"]), "--preset", "card2b"]
    assert dc.main([*args, "--out", str(small["tmp"] / "both.json")]) == 2
    assert "give --preset or --areas, not both" in capsys.readouterr().err
    assert not (small["tmp"] / "both.json").exists()


def test_trashlist_and_recount_take_the_presets(proven, live):
    plan, census, swe = proven["docs"]
    assert (census["preset"], swe["preset"]) == ("card2b", "swe-data")
    assert proven["summary"]["areas"]  # trashlist took them: the proven world's own list
    _census_now(live)  # the consolidate preset
    assert _load(live.tmp / "now.json")["preset"] == "consolidate"
    rc, rec = _recount(live)
    assert (rc, rec["unchanged"]) == (0, "yes")


# J. What was left unpinned.


def test_recount_needs_the_proofs_row_for_each_area_folder(live):
    proof = _load(live.list / dc.PROOF)
    proof["rows"] = [r for r in proof["rows"] if r["id"] != SLO]
    text = dc.proof_text(proof["rows"])
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (live.tmp / "no_top.json").write_bytes(text.encode("utf-8"))
    summary = _load(live.list / dc.PROOF_SUMMARY)
    summary["sha256"] = sha
    (live.tmp / "no_top_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _census_now(live)
    paths = {"proof": live.tmp / "no_top.json", "summary": live.tmp / "no_top_summary.json"}
    rc, rec = _recount(live, sha=sha, **paths)
    assert rc == 1
    assert rec["differences"] == ["area swe-local-only: the proof has no row for its folder"]


def test_kept_csv_holds_every_blocker_row_and_only_those(rich):
    import csv

    with open(rich["root"] / "_logs" / "list" / "kept.csv", encoding="utf-8", newline="") as fh:
        kept = list(csv.DictReader(fh))
    blockers = [r for r in rich["rows"] if r["class"] not in dc.CLEARED]
    assert blockers and all(r["class"] == "blocker" for r in blockers)
    assert sorted((r["area"], r["id"]) for r in kept) == sorted(
        (r["area"], r["id"]) for r in blockers
    )
    assert {r["class"] for r in kept} == {"blocker"} and all(r["reason"] for r in kept)


def test_a_test_that_reaches_rclone_outside_the_fake_fails(
    no_rclone_outside_the_fake,  # noqa: F811  (the imported fixture)
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)
    sentinel = no_rclone_outside_the_fake
    assert dc.RCLONE is sentinel  # every test starts with it; only the fake replaces it
    assert dc.main(["census", "--remote", "fake:", "--out", str(tmp_path / "c.json")]) == 2
    assert sentinel.used  # so this fixture's teardown would fail any other test here
    sentinel.used = False  # this test's own, deliberate use


@pytest.mark.parametrize(
    "old, new, rule",
    [
        (  # inside run_cmd itself, where subprocess may be used: only shell= is wrong
            "cmd, input=stdin, capture_output=True",
            "cmd, input=stdin, shell=True, capture_output=True",
            "shell=",
        ),
        (  # RCLONE joined into a string, in a function that runs nothing
            "def trashlist(plan: dict",
            "def _planted():\n    return ' '.join(RCLONE)\n\n\ndef trashlist(plan: dict",
            "RCLONE used another way",
        ),
    ],
)
def test_the_source_scan_refuses_a_shell_or_a_joined_rclone(old, new, rule, tmp_path, monkeypatch):
    src = (tdc._REPO / "scripts" / "drive_consolidate.py").read_text(encoding="utf-8")
    assert src.count(old) == 1
    copy_ = tmp_path / "scripts" / "drive_consolidate.py"
    copy_.parent.mkdir()
    copy_.write_text(src.replace(old, new), encoding="utf-8")
    monkeypatch.setattr(tdc, "_REPO", tmp_path)
    with pytest.raises(AssertionError) as caught:
        test_no_process_or_rclone_call_in_the_source_is_outside_the_allowlist()
    first = str(caught.value).splitlines()[0]  # the list of problems, as the scan reports it
    assert first.startswith("['" + rule) and first.count(" at line ") == 1, first


# L. Same-name sibling folders: rclone lists one twice, and the other's contents never.


def _twin_world(tmp_path, monkeypatch, empty_twin_last=True):
    """A clean world plus a folder 'dup' in swe-local-only holding a file the desktop never
    had, and a same-name sibling (the twin a retried mkdir leaves; card 2c met one)."""
    monkeypatch.setattr(dc, "utc_now", Clock(T0))
    w = World(tmp_path, monkeypatch)
    _clean(w)
    if empty_twin_last:
        full = w.d.folder("dup", SLO)
        lost = w.d.file("lost.csv", full, b"bytes the desktop never had\n")
        w.d.folder("dup", SLO)  # empty, listed last: rclone's cache resolves 'dup' to it
    else:
        w.d.folder("dup", SLO)
        full = w.d.folder("dup", SLO)
        lost = w.d.file("lost.csv", full, b"bytes the desktop never had\n")
    return w, lost


@pytest.mark.parametrize("empty_twin_last", [True, False])
@pytest.mark.parametrize("dropped", [True, False])
def test_same_name_sibling_folders_stop_the_census(
    tmp_path, monkeypatch, capsys, empty_twin_last, dropped
):
    # The final check's case: the batched census misses lost.csv, and rclone's walk never
    # lists the twin that holds it. Before this round both agreed and trashlist said yes.
    w, lost = _twin_world(tmp_path, monkeypatch, empty_twin_last)
    env = w.env()
    if dropped:
        monkeypatch.setenv("FAKE_DROP_IN_OR", lost)
    assert _census_of(env, None, "c6.json", "card2b") == 2
    err = capsys.readouterr().err
    assert "one path for two folders in rclone's listing: dup (" in err, err
    assert not (tmp_path / "c6.json").exists()  # so no proof, and no "yes", can follow


def test_different_case_names_are_not_one_path(tmp_path, monkeypatch):
    # rclone's cache keys on the exact path, case and all: Dup and dup are two paths.
    monkeypatch.setattr(dc, "utc_now", Clock(T0))
    w = World(tmp_path, monkeypatch)
    _clean(w)
    w.drive_file("upper", SLO, "Dup/u.csv", b"upper case\n")
    w.home("upper/u.csv", b"upper case\n")  # desktop paths that Windows keeps apart
    w.drive_file("lower", SLO, "dup/l.csv", b"lower case\n")
    w.home("lower/l.csv", b"lower case\n")
    env = w.env()
    _prepare(env)
    slo = _area_of(_load(tmp_path / "c6.json"), SLO)
    assert slo["same_name_folders"] == []
    assert _trashlist(env) == 0
    rows = _load(env["root"] / "_logs" / "list" / dc.PROOF)["rows"]
    mine = {r["id"]: r for r in rows}
    assert mine[w.ids["upper"]]["class"] == mine[w.ids["lower"]]["class"] == "listed"


def test_same_name_sibling_folders_and_all_they_hold_are_blockers(case):
    slo = _area_of(case.census, SLO)
    data = case.folders[(SLO, "data")]
    slo["objects"].append(_folder_obj("twin-data", "data", SLO))
    inside = {
        **_listings(case.census, case.ids["a"])[0],
        "id": "twin-file",
        "parents": ["twin-data"],
    }
    slo["objects"].append(inside)
    cls = _classes(case)
    for fid in (data, "twin-data"):
        assert (cls[fid]["class"], cls[fid]["reason"]) == ("blocker", dc.SAME_NAME), fid
    for oid in (case.ids["a"], "twin-file"):  # a proven file too: rclone cannot list it
        assert (cls[oid]["class"], cls[oid]["reason"]) == ("blocker", dc.SHADOWED), oid
    assert cls[SLO]["reason"].startswith("holds a blocker: data")
    assert _verdicts(case) == {
        "swe-local-only": "no",
        "SmartWheelData": "yes",
        "smart-wheel-engine-git": "yes",
    }


def test_the_census_records_same_name_sibling_folders():
    objs = [
        {"id": "f1", "name": "dup", "mime": dc.FOLDER, "parents": ["top"]},
        {"id": "f2", "name": "dup", "mime": dc.FOLDER, "parents": ["top"]},
        {"id": "f3", "name": "Dup", "mime": dc.FOLDER, "parents": ["top"]},
        {"id": "f4", "name": "dup", "mime": dc.FOLDER, "parents": ["f1"]},
        {"id": "x1", "name": "dup", "mime": "text/csv", "parents": ["top"]},
    ]
    assert dc.same_name_folders(objs) == [{"parent": "top", "name": "dup", "ids": ["f1", "f2"]}]
    entries = [
        {"Path": "dup", "IsDir": True, "ID": "f1"},
        {"Path": "dup", "IsDir": True, "ID": "f2"},
        {"Path": "dup/x", "IsDir": False, "ID": "x1"},
        {"Path": "Dup", "IsDir": True, "ID": "f3"},
    ]
    assert dc.one_path_two_folders(entries) == ["dup (f1, f2)"]


def test_recount_says_no_for_an_area_holding_same_name_sibling_folders(live):
    _census_now(live)
    census = _load(live.tmp / "now.json")
    _area_of(census, SLO)["objects"].append(_folder_obj("twin-data", "data", SLO))
    (live.tmp / "now.json").write_text(json.dumps(census), encoding="utf-8")
    rc, rec = _recount(live)
    data = live.folders[(SLO, "data")]
    assert rc == 1
    assert rec["differences"] == [
        f"area swe-local-only: same-name sibling folders 'data' in {SLO} "
        f"({', '.join(sorted([data, 'twin-data']))}): rclone cannot list both"
    ]


# M. An INCOMPLETE listing is a failed one.


def test_an_incomplete_listing_stops_the_census(small, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_LSJSON_INCOMPLETE", "1")
    assert tdc._census(small) == 2
    assert "rclone lsjson failed" in capsys.readouterr().err
    assert not (small["tmp"] / "c.json").exists()


# N. The swe-data/ census needs its cross-check too.


def test_trashlist_refuses_a_swe_data_census_without_its_cross_check(proven, tmp_path, capsys):
    for drop in ("its record", "its area's record"):
        plan, census, swe = copy.deepcopy(proven["docs"])
        if drop == "its record":
            swe.pop("cross_checked")
        else:
            swe["areas"][0].pop("cross_check")
        w = {"tmp": tmp_path, "root": proven["root"]}
        names = {
            "plan": _write(tmp_path, "p.json", plan),
            "census": _write(tmp_path, "c6.json", census),
        }
        names["swe"] = _write(tmp_path, "swe.json", swe)
        out = proven["root"] / "_logs" / "refused-swe-unchecked"
        assert _trashlist(w, out_path=out, **names) == 2, drop
        assert "the swe-data/ census records no cross-check" in capsys.readouterr().err
        assert not out.exists()


# Q. census --only: one area of a set, with no hand-written file.


def test_only_keeps_the_named_areas_of_a_set(small, capsys):
    assert _census_of(small, None, "one.json", "consolidate", ("swe-local-only",)) == 0
    doc = _load(small["tmp"] / "one.json")
    assert [a["name"] for a in doc["areas"]] == ["swe-local-only"]
    assert doc["only"] == ["swe-local-only"] and doc["preset"] == "consolidate"
    two = ("SmartWheelData", "swe-local-only")
    assert _census_of(small, None, "two.json", "consolidate", two) == 0
    assert [a["name"] for a in _load(small["tmp"] / "two.json")["areas"]] == [
        "swe-local-only",
        "SmartWheelData",
    ]
    file_set = [AREA["swe-local-only"], AREA["smart-wheel-engine-git"]]
    assert _census_of(small, file_set, "file.json", None, ("smart-wheel-engine-git",)) == 0
    assert [a["name"] for a in _load(small["tmp"] / "file.json")["areas"]] == [
        "smart-wheel-engine-git"
    ]
    assert _census_of(small, None, "bad.json", "consolidate", ("vendor_swe_data",)) == 2
    assert "--only ['vendor_swe_data']: not an area of this census's set" in (
        capsys.readouterr().err
    )
    assert not (small["tmp"] / "bad.json").exists()


def test_a_one_area_recount_needs_no_hand_written_file(live):
    _census_now(live, ["swe-local-only"], "slo.json")  # --preset consolidate --only ...
    rc, rec = _recount(live, "swe-local-only", census="slo.json")
    assert (rc, rec["unchanged"], rec["areas"]) == (0, "yes", ["swe-local-only"])
    assert not list(live.tmp.glob("slo.json.areas.json"))


# D, last: every call the fake rclone has received in this session is an allowed one.


def test_every_call_the_fake_rclone_received_in_this_session_is_allowed(small):
    assert tdc._census(small) == 0  # so that it has calls to check, run alone too
    calls = tdc.fake_calls()
    assert {"version", "backend query", "lsjson"} <= {tdc.verb_of(c) for c in calls}
    bad = [c for c in calls if tdc.verb_of(c) not in tdc.ALLOWED_VERBS]
    assert not bad, bad[:5]
