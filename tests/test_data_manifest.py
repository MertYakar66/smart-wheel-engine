"""scripts/data_manifest.py — build / check / census on a temporary data root (D31).

Pins: build hashes every data file under the three conventional dirs and skips
code; check exits 0 on a byte-identical root and 1 on a missing or altered
file, naming it; --group narrows the check; census counts presence by group;
materialize creates only the files missing from a root, byte-verified from the
git objects the manifest names, never overwrites, and names the branch to
fetch when an object is absent; an archive row (data_archive/, D31) is read
from its git_path and written to its own path; build carries the ledger
metadata (git_sources, drive, per-file git_source and git_path) over from the
manifest it replaces; and the committed data/DATA_MANIFEST.json parses with the
expected schema, covers the datasets git holds today (bloomberg, broad_pull,
deep, ticks), and has a row for every data file tracked on this checkout.

Fixture files are written as bytes so the git round trip is byte-stable on
Windows, where ``write_text`` emits CRLF and ``core.autocrlf`` may rewrite it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "data_manifest", _REPO / "scripts" / "data_manifest.py"
)
dm = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(dm)


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "data" / "bloomberg" / "deep").mkdir(parents=True)
    (root / "data_raw" / "bloomberg" / "ticks").mkdir(parents=True)
    (root / "data_processed").mkdir()
    (root / "data" / "bloomberg" / "sp500_ohlcv.csv").write_bytes(
        b"date,ticker,close\n2026-01-02,AAPL,1\n"
    )
    (root / "data" / "bloomberg" / "deep" / "slice.csv.gz").write_bytes(b"\x1f\x8b" + b"x" * 50)
    (root / "data_raw" / "bloomberg" / "ticks" / "SPY_ticks.csv.gz").write_bytes(
        b"\x1f\x8b" + b"y" * 20
    )
    (root / "data" / "schemas.py").write_text("# code, not data\n")
    (root / "data" / "README.md").write_text("docs, not data\n")
    return root


def test_build_then_check_roundtrip(tmp_path, capsys):
    root = _make_root(tmp_path)
    out = tmp_path / "m.json"
    assert dm.main(["build", "--root", str(root), "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["schema"] == dm.SCHEMA
    paths = {f["path"] for f in m["files"]}
    assert paths == {
        "data/bloomberg/sp500_ohlcv.csv",
        "data/bloomberg/deep/slice.csv.gz",
        "data_raw/bloomberg/ticks/SPY_ticks.csv.gz",
    }
    groups = {f["path"]: f["group"] for f in m["files"]}
    assert groups["data/bloomberg/deep/slice.csv.gz"] == "deep"
    assert groups["data_raw/bloomberg/ticks/SPY_ticks.csv.gz"] == "ticks"
    assert groups["data/bloomberg/sp500_ohlcv.csv"] == "bloomberg"
    assert dm.main(["check", "--root", str(root), "--manifest", str(out)]) == 0
    assert "3 ok, 0 missing, 0 mismatched" in capsys.readouterr().out


def test_check_flags_missing_and_altered(tmp_path, capsys):
    root = _make_root(tmp_path)
    out = tmp_path / "m.json"
    dm.main(["build", "--root", str(root), "--out", str(out)])
    (root / "data_raw" / "bloomberg" / "ticks" / "SPY_ticks.csv.gz").unlink()
    (root / "data" / "bloomberg" / "sp500_ohlcv.csv").write_text(
        "date,ticker,close\n2026-01-02,AAPL,2\n"
    )
    rc = dm.main(["check", "--root", str(root), "--manifest", str(out)])
    text = capsys.readouterr().out
    assert rc == 1
    assert "MISSING    data_raw/bloomberg/ticks/SPY_ticks.csv.gz" in text
    assert "MISMATCH   data/bloomberg/sp500_ohlcv.csv (sha256 differs)" in text
    # same-size content change is caught by the hash, not the size
    assert "size" not in text.split("MISMATCH")[1].split("\n")[0]


def test_check_group_and_size_only(tmp_path, capsys):
    root = _make_root(tmp_path)
    out = tmp_path / "m.json"
    dm.main(["build", "--root", str(root), "--out", str(out)])
    (root / "data_raw" / "bloomberg" / "ticks" / "SPY_ticks.csv.gz").unlink()
    assert (
        dm.main(
            ["check", "--root", str(root), "--manifest", str(out), "--group", "deep", "--size-only"]
        )
        == 0
    )
    assert dm.main(["check", "--root", str(root), "--manifest", str(out), "--group", "ticks"]) == 1
    capsys.readouterr()


def test_census_counts_presence_by_group(tmp_path, capsys):
    root = _make_root(tmp_path)
    out = tmp_path / "m.json"
    dm.main(["build", "--root", str(root), "--out", str(out)])
    (root / "data" / "bloomberg" / "deep" / "slice.csv.gz").unlink()
    assert dm.main(["census", "--root", str(root), "--manifest", str(out)]) == 0
    text = capsys.readouterr().out
    assert "missing overall: 1" in text
    assert "deep" in text


def test_extra_lists_unknown_files(tmp_path, capsys):
    root = _make_root(tmp_path)
    out = tmp_path / "m.json"
    dm.main(["build", "--root", str(root), "--out", str(out)])
    (root / "data" / "bloomberg" / "new_panel.csv").write_text("x\n")
    dm.main(["check", "--root", str(root), "--manifest", str(out), "--extra"])
    assert "EXTRA      data/bloomberg/new_panel.csv" in capsys.readouterr().out


@pytest.mark.skipif(
    not (_REPO / "data" / "DATA_MANIFEST.json").exists(), reason="manifest not committed"
)
def test_committed_manifest_covers_the_git_held_datasets():
    m = dm.load_manifest(_REPO / "data" / "DATA_MANIFEST.json")
    groups = {f["group"] for f in m["files"]}
    assert {"bloomberg", "broad_pull", "deep", "ticks"} <= groups
    assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in m["files"])
    assert m["counts"]["deep"]["files"] == 13
    assert m["counts"]["ticks"]["files"] == 15
    # data_archive rows: bytes from a branch tip, read from git_path, never a duplicate
    sources = m.get("git_sources", {})
    archive = [f for f in m["files"] if f["group"] == "archive"]
    assert archive, "the branch archive rows are part of the manifest (D31)"
    for f in archive:
        assert f["path"].startswith("data_archive/") and f.get("git_path")
        assert f["path"].endswith("/" + f["git_path"])
        assert f.get("git_source") in sources
    shas = [f["sha256"] for f in m["files"]]
    assert len(shas) == len(set(shas)), "each distinct file is carried by exactly one row"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_every_tracked_data_file_has_a_manifest_row():
    """``check`` walks manifest rows only, so a tracked file without a row is
    invisible to it — and untracking it would drop the only copy (D31 gap A,
    2026-09-23: 16 feature sidecars). Whatever git still tracks under the data
    trees must be in the manifest."""
    m = dm.load_manifest(_REPO / "data" / "DATA_MANIFEST.json")
    known = {f["path"] for f in m["files"]}
    tracked = _git(_REPO, "ls-files", "--", *dm.WALK_DIRS).splitlines()
    data_files = [
        p
        for p in tracked
        if not p.endswith(dm.SKIP_SUFFIXES) and not set(p.split("/")) & set(dm.SKIP_NAMES)
    ]
    assert [p for p in data_files if p not in known] == []


# --------------------------------------------------------------------------
# materialize — from git objects into a root, verified, never overwriting
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def git_repo_with_data(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A throwaway git repo holding a data root at one commit, plus a manifest
    of that root whose git_sources points at the commit."""
    if shutil.which("git") is None:  # pragma: no cover - CI always has git
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    root = _make_root(tmp_path)
    shutil.copytree(root, repo)
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")  # store the fixture bytes as written
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "data")
    commit = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "m.json"
    dm.main(["build", "--root", str(root), "--out", str(out)])
    m = json.loads(out.read_text())
    for f in m["files"]:
        f["git_source"] = "git:data-branch"
    m["git_sources"] = {"git:data-branch": commit}
    out.write_text(json.dumps(m))
    return repo, out, m


def test_materialize_fills_an_empty_root_and_verifies(tmp_path, git_repo_with_data, capsys):
    repo, manifest, m = git_repo_with_data
    target = tmp_path / "desktop"
    rc = dm.main(
        ["materialize", "--root", str(target), "--manifest", str(manifest), "--repo", str(repo)]
    )
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "3 already present" not in text and "wrote 3" in text
    assert dm.main(["check", "--root", str(target), "--manifest", str(manifest)]) == 0
    assert "3 ok, 0 missing, 0 mismatched" in capsys.readouterr().out
    # second run: everything present, nothing written, still exit 0
    assert (
        dm.main(
            ["materialize", "--root", str(target), "--manifest", str(manifest), "--repo", str(repo)]
        )
        == 0
    )
    assert "3 already present, wrote 0" in capsys.readouterr().out
    assert not list(target.rglob("*.materializing"))


def test_materialize_never_overwrites_and_names_the_branch_to_fetch(
    tmp_path, git_repo_with_data, capsys
):
    repo, manifest, m = git_repo_with_data
    target = tmp_path / "desktop"
    (target / "data" / "bloomberg").mkdir(parents=True)
    altered = target / "data" / "bloomberg" / "sp500_ohlcv.csv"
    altered.write_text("operator's newer bytes\n")
    # point one file at a commit the repo does not have
    m2 = json.loads(manifest.read_text())
    m2["git_sources"]["git:data-branch"] = "0" * 40
    manifest.write_text(json.dumps(m2))
    rc = dm.main(
        ["materialize", "--root", str(target), "--manifest", str(manifest), "--repo", str(repo)]
    )
    text = capsys.readouterr().out
    assert rc == 1
    assert altered.read_text() == "operator's newer bytes\n"  # kept, not overwritten
    assert "MISMATCH   data/bloomberg/sp500_ohlcv.csv" in text
    assert "git fetch origin data-branch" in text
    assert "1 mismatched (kept, not overwritten), 2 unavailable" in text


def test_materialize_dry_run_writes_nothing(tmp_path, git_repo_with_data, capsys):
    repo, manifest, m = git_repo_with_data
    target = tmp_path / "desktop"
    rc = dm.main(
        [
            "materialize",
            "--root",
            str(target),
            "--manifest",
            str(manifest),
            "--repo",
            str(repo),
            "--dry-run",
        ]
    )
    text = capsys.readouterr().out
    assert rc == 0 and "would write 3" in text
    assert not target.exists() or not list(target.rglob("*"))


def test_materialize_writes_archive_rows_from_their_git_path(tmp_path, git_repo_with_data, capsys):
    """An archive row keeps a branch's bytes under data_archive/<branch>/<path>:
    materialize reads them from git_path, check and census see them as 'archive',
    and a rebuild carries git_path over."""
    repo, manifest, m = git_repo_with_data
    m2 = json.loads(manifest.read_text())
    src = next(f for f in m2["files"] if f["path"] == "data/bloomberg/sp500_ohlcv.csv")
    m2["files"].append(
        {
            "path": "data_archive/data-branch/data/bloomberg/sp500_ohlcv.csv",
            "size": src["size"],
            "sha256": src["sha256"],
            "group": "archive",
            "git_source": "git:data-branch",
            "git_path": "data/bloomberg/sp500_ohlcv.csv",
        }
    )
    manifest.write_text(json.dumps(m2))
    target = tmp_path / "desktop"
    rc = dm.main(
        ["materialize", "--root", str(target), "--manifest", str(manifest), "--repo", str(repo)]
    )
    assert rc == 0, capsys.readouterr().out
    archived = target / "data_archive" / "data-branch" / "data" / "bloomberg" / "sp500_ohlcv.csv"
    assert archived.read_bytes() == (target / "data" / "bloomberg" / "sp500_ohlcv.csv").read_bytes()
    capsys.readouterr()
    assert dm.main(["check", "--root", str(target), "--manifest", str(manifest)]) == 0
    assert "4 ok, 0 missing, 0 mismatched" in capsys.readouterr().out
    assert dm.group_of("data_archive/data-branch/data/bloomberg/sp500_ohlcv.csv") == "archive"
    assert dm.main(["build", "--root", str(target), "--out", str(manifest)]) == 0
    rebuilt = {f["path"]: f for f in json.loads(manifest.read_text())["files"]}
    row = rebuilt["data_archive/data-branch/data/bloomberg/sp500_ohlcv.csv"]
    assert row["group"] == "archive" and row["git_path"] == "data/bloomberg/sp500_ohlcv.csv"
    assert row["git_source"] == "git:data-branch"


def test_build_carries_ledger_metadata_over(tmp_path, git_repo_with_data, capsys):
    repo, manifest, m = git_repo_with_data
    m2 = json.loads(manifest.read_text())
    m2["drive"] = {"root": "folder-id"}
    manifest.write_text(json.dumps(m2))
    root = tmp_path / "root"
    # change one file: its git_source must drop, the others keep theirs
    (root / "data" / "bloomberg" / "sp500_ohlcv.csv").write_text(
        "date,ticker,close\n2026-01-03,AAPL,3\n"
    )
    assert dm.main(["build", "--root", str(root), "--out", str(manifest)]) == 0
    m3 = json.loads(manifest.read_text())
    assert m3["git_sources"] == m2["git_sources"]
    assert m3["drive"] == {"root": "folder-id"}
    by_path = {f["path"]: f for f in m3["files"]}
    assert "git_source" not in by_path["data/bloomberg/sp500_ohlcv.csv"]
    assert by_path["data/bloomberg/deep/slice.csv.gz"]["git_source"] == "git:data-branch"
