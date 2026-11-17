"""scripts/copy_dated_sources.py — the weekly dated copies (DECISIONS.md D29, amended 2026-11-17).

Every page, calendar and table below is hand-made in this file; nothing is
copied from any site. No test opens a network connection or a data root: the
fetch function is a fake that stands for one HTTP request and follows nothing,
the date is injected, ``SWE_DATA_ROOT`` is removed before each test and set only
to a temporary folder, and a socket that tries to connect fails the test.

Pins: the script refuses without a data root (exit 3) and on a date that is not
today's in UTC (exit 5), before anything is fetched or written; ``--lists`` is
required and ``schedules`` writes nothing for ``members``; a body is kept byte
for byte (both line endings, a non-UTF-8 byte) and the record's sha256 is the
body's; an existing file is never written over and its address is not fetched;
one request per address, one more per followed redirect; a redirect is followed
only over https to the same host or its ``www`` form, at most three times, and
any other is a failure; the User-Agent names the repository and is no browser's;
everything written lies in the two copy folders or the run's log folder, and no
other file under the root is opened; a refused or unanswered address gets a row
and no file (exit 10); a file that fails its light check is kept and flagged
(exit 10); ``members.csv`` is the first table with a "Symbol" heading, in UTF-8
with LF endings, and is listed in the record; no response header reaches the
screen or the log; ``--dry-run`` fetches and writes nothing.
"""

from __future__ import annotations

import ast
import hashlib
import http.client
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import urllib.request
import urllib.response
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "copy_dated_sources.py"
_SPEC = importlib.util.spec_from_file_location("copy_dated_sources", _SCRIPT)
cds = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = cds
_SPEC.loader.exec_module(cds)
_REAL_FETCH_ONCE = cds.fetch_once

DAY = "2026-11-17"
NOW = datetime(2026, 11, 17, 14, 3, 22, tzinfo=UTC)
STAMP = "140322Z"

FED = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_PAGE = "https://www.bls.gov/schedule/news_release/"
BLS_ICS = "https://www.bls.gov/schedule/news_release/bls.ics"
BEA_PAGE = "https://www.bea.gov/news/schedule"
BEA_ICS = "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics"
BEA_JSON = "https://apps.bea.gov/API/signup/release_dates.json"
CENSUS = "https://www.census.gov/economic-indicators/calendar-listview.html"
CENSUS_2026 = "https://www.census.gov/economic-indicators/calendar-listview-2026.html"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# (address, file name, marker): the nine addresses the Operator approved, as D29 names them.
NAMED = [
    (FED, "fed_fomccalendars.htm", "FOMC"),
    (BLS_PAGE, "bls_news_release.htm", "Schedule"),
    (BLS_ICS, "bls.ics", None),
    (BEA_PAGE, "bea_schedule.htm", "Release"),
    (BEA_ICS, "bea_schedule.ics", None),
    (BEA_JSON, "bea_release_dates.json", None),
    (CENSUS, "census_calendar_listview.html", "Economic Indicator"),
    (CENSUS_2026, "census_calendar_listview_2026.html", "Economic Indicator"),
    (WIKI, "members_page.html", "500 companies"),
]
SCHEDULE_ADDRESSES = [address for address, _, _ in NAMED[:8]]
SCHEDULE_FILES = [name for _, name, _ in NAMED[:8]]

HEADERS = (
    ("Content-Type", "text/html; charset=utf-8"),
    ("X-Served-By", "hand-made-cache-7f3a"),
    ("Set-Cookie", "visit=4be1c0; Path=/; Secure"),
)

ICS = (
    b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n"
    b"SUMMARY:Hand-made test event\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n"
)
JSON_BODY = b'{"hand_made": true, "releases": []}\n'

MEMBERS_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hand-made members page</title>
<style>table { border: 1px solid }</style></head>
<body>
<h1>A hand-made list of 500 companies</h1>
<table class="note"><tr><th>Note</th></tr><tr><td>Not this table.</td></tr></table>
<table id="members">
<tbody>
<tr><th>Symbol</th><th>Company</th><th>Sector</th><th>Head&nbsp;office</th></tr>
<tr><td><a href="/aaa">AAA</a></td><td>Alpha &amp; Sons</td><td>Industrials</td><td>Zürich,
    Switzerland</td></tr>
<tr><td>BBB</td><td><style>.x { color: red }</style>Beta<sup>[1]</sup> Works</td>
<td>Health&#32;Care</td><td>S&atilde;o Paulo</td></tr>
</tbody>
</table>
<table><tr><th>Symbol</th></tr><tr><td>ZZZ</td></tr></table>
</body></html>
""".encode()
MEMBERS_CSV = (
    "Symbol,Company,Sector,Head office\n"
    'AAA,Alpha & Sons,Industrials,"Zürich, Switzerland"\n'
    "BBB,Beta[1] Works,Health Care,São Paulo\n"
).encode()
PAGE_WITHOUT_TABLE = (
    b"<html><body><h1>A hand-made list of 500 companies</h1>"
    b"<table><tr><th>Ticker</th></tr><tr><td>AAA</td></tr></table></body></html>\n"
)

DECOY = b"ticker\nAAA\n"  # stands for a file the engine's scripts read; never to be opened


def _page(marker: str) -> bytes:
    return (
        "<!DOCTYPE html>\n<html><head><title>Hand-made test page</title></head>\n"
        f"<body><h1>{marker}, written for the tests</h1></body></html>\n"
    ).encode()


def _default_body(address: str) -> bytes:
    for named, filename, marker in NAMED:
        if named == address:
            if filename.endswith(".ics"):
                return ICS
            if filename.endswith(".json"):
                return JSON_BODY
            return MEMBERS_PAGE if address == WIKI else _page(marker)
    raise AssertionError(f"the fake was asked for an address no test gave it: {address}")


def _ok(body: bytes, headers=HEADERS):
    return cds.Response(200, tuple(headers), body)


def _redirect(to: str | None, status: int = 301):
    headers = (("X-Served-By", "hand-made-cache-7f3a"),)
    if to is not None:
        headers = (("Location", to), *headers)
    return cds.Response(status, headers, b"")


class FakeWeb:
    """Stands for the network. One call is one HTTP request; it follows nothing itself."""

    def __init__(self, answers: dict | None = None) -> None:
        self.answers = dict(answers or {})
        self.calls: list[str] = []

    def __call__(self, address: str):
        self.calls.append(address)
        answer = self.answers.get(address)
        if answer is None:
            return _ok(_default_body(address))
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def _no_network_and_no_ambient_root(monkeypatch):
    monkeypatch.delenv("SWE_DATA_ROOT", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(cds, "fetch_once", refuse)


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """A temporary data root with one decoy file, and an empty working directory."""
    data_root = tmp_path / "root"
    (data_root / "data_raw").mkdir(parents=True)
    (data_root / "data_raw" / "sp500_constituents_current.csv").write_bytes(DECOY)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("SWE_DATA_ROOT", str(data_root))
    return data_root


def _schedules(root: Path) -> Path:
    return root / "data_raw" / "macro_schedule" / f"date={DAY}"


def _members(root: Path) -> Path:
    return root / "data_raw" / "sector_dated" / f"date={DAY}"


def _logs(root: Path) -> Path:
    return root / "_logs" / "dated-copies" / DAY


def _run(lists=("schedules",), web: FakeWeb | None = None, now=NOW, date=DAY, extra=()):
    web = web if web is not None else FakeWeb()
    code = cds.main(["--lists", *lists, "--date", date, *extra], fetch=web, now=lambda: now)
    return code, web


def _files(base: Path) -> dict[str, bytes]:
    return {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _record(folder: Path, stamp: str = STAMP) -> dict:
    return json.loads((folder / f"_copy-{stamp}.json").read_bytes())


def _row(record: dict, filename: str) -> dict:
    (row,) = [row for row in record["files"] if row["file"] == filename]
    return row


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── what is copied ────────────────────────────────────────────────────────────


def test_the_lists_are_the_nine_named_addresses():
    found = [
        (source.address, source.filename, source.marker)
        for copy_list in cds.LISTS.values()
        for source in copy_list.sources
    ]
    assert found == NAMED
    assert list(cds.LISTS) == ["schedules", "members"]
    assert cds.LISTS["schedules"].folder == ("data_raw", "macro_schedule")
    assert cds.LISTS["members"].folder == ("data_raw", "sector_dated")
    assert cds.LISTS["members"].table_file == "members.csv"
    assert all("&" not in (marker or "") for _, _, marker in NAMED)


def test_the_script_is_standard_library_only_and_imports_nothing_from_the_engine():
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported, "no imports found: the check read the wrong file"
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
    assert "engine" not in imported
    assert "requests" not in imported and "pandas" not in imported


# ── the refusals: no root, wrong date, no list ────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("date", [DAY, "2000-01-01"])
def test_without_a_data_root_it_exits_3_and_does_nothing(
    value, date, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    if value is not None:
        monkeypatch.setenv("SWE_DATA_ROOT", value)
    code, web = _run(date=date)
    assert code == 3
    assert web.calls == []
    assert _files(tmp_path) == {} and list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1 and "SWE_DATA_ROOT" in captured.err


def test_a_root_that_is_not_a_folder_is_no_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWE_DATA_ROOT", str(tmp_path / "not-there"))
    code, web = _run()
    assert code == 3 and web.calls == []
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err.count("\n") == 1


@pytest.mark.parametrize(
    ("date", "now"),
    [
        ("2026-11-16", NOW),
        ("2026-11-18", NOW),
        ("2026-11-18", datetime(2026, 11, 17, 23, 59, 59, tzinfo=UTC)),
        # 01:30 on the 18th in a +03:00 zone is still the 17th in UTC.
        ("2026-11-18", datetime(2026, 11, 18, 1, 30, tzinfo=timezone(timedelta(hours=3)))),
    ],
)
def test_a_date_that_is_not_todays_utc_date_exits_5_and_does_nothing(date, now, root, capsys):
    before = _files(root.parent)
    code, web = _run(date=date, now=now)
    assert code == 5
    assert web.calls == []
    assert _files(root.parent) == before
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.count("\n") == 1 and date in captured.err


def test_the_copy_date_is_the_utc_date_whatever_the_local_zone(root):
    local = datetime(2026, 11, 18, 1, 30, tzinfo=timezone(timedelta(hours=3)))
    code, _ = _run(date=DAY, now=local)
    assert code == 0
    assert (_schedules(root) / "_copy-223000Z.json").is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["--date", DAY],
        ["--lists", "--date", DAY],
        ["--lists", "everything", "--date", DAY],
        ["--lists", "schedules"],
        ["--lists", "schedules", "--date", "17-11-2026"],
        ["--lists", "schedules", "--date", "2026-11-7"],
        ["--lists", "schedules", "--date", "2026-02-30"],
        ["--list", "schedules", "--date", DAY],
    ],
)
def test_a_usage_error_keeps_argparses_exit_code_and_does_nothing(argv, root, capsys):
    before = _files(root.parent)
    web = FakeWeb()
    with pytest.raises(SystemExit) as refusal:
        cds.main(argv, fetch=web, now=lambda: NOW)
    assert refusal.value.code == 2
    assert web.calls == []
    assert _files(root.parent) == before


def test_schedules_alone_writes_nothing_for_members(root):
    code, web = _run(("schedules",))
    assert code == 0
    assert web.calls == SCHEDULE_ADDRESSES
    assert not (root / "data_raw" / "sector_dated").exists()
    assert sorted(path.name for path in _schedules(root).iterdir()) == sorted(
        [*SCHEDULE_FILES, f"_copy-{STAMP}.json"]
    )


def test_members_alone_writes_nothing_for_schedules(root):
    code, web = _run(("members",))
    assert code == 0
    assert web.calls == [WIKI]
    assert not (root / "data_raw" / "macro_schedule").exists()


def test_naming_a_list_twice_copies_it_once(root):
    code, web = _run(("members", "schedules", "members"))
    assert code == 0
    assert web.calls == [*SCHEDULE_ADDRESSES, WIKI]


# ── bytes as served, and never over a file ────────────────────────────────────


def test_a_body_is_kept_byte_for_byte_and_the_record_holds_its_sha256(root):
    odd = b"one\r\ntwo\nthree\r\n\xff\xfe latin-1 \xe9 and a nul \x00 end"
    page = b"FOMC, hand-made\r\n" + odd
    calendar = b"BEGIN:VCALENDAR\r\n" + odd
    code, _ = _run(web=FakeWeb({FED: _ok(page), BLS_ICS: _ok(calendar)}))
    assert code == 0
    with pytest.raises(UnicodeDecodeError):
        page.decode("utf-8")
    record = _record(_schedules(root))
    for filename, body in (("fed_fomccalendars.htm", page), ("bls.ics", calendar)):
        assert (_schedules(root) / filename).read_bytes() == body
        row = _row(record, filename)
        assert row["sha256"] == _sha(body) and row["bytes"] == len(body)
        assert row["result"] == "saved" and row["looks_right"] == "yes"


def test_an_existing_file_is_never_written_over_and_its_address_is_not_fetched(root, capsys):
    kept = b"kept from an earlier run\r\n\xff"
    _schedules(root).mkdir(parents=True)
    (_schedules(root) / "bls.ics").write_bytes(kept)
    code, web = _run()
    assert code == 0
    assert (_schedules(root) / "bls.ics").read_bytes() == kept
    assert BLS_ICS not in web.calls
    assert web.calls == [address for address in SCHEDULE_ADDRESSES if address != BLS_ICS]
    for address, filename, _ in NAMED[:8]:
        if address != BLS_ICS:
            assert (_schedules(root) / filename).read_bytes() == _default_body(address)
    row = _row(_record(_schedules(root)), "bls.ics")
    assert row["result"] == "already saved"
    assert row["sha256"] == _sha(kept) and row["bytes"] == len(kept)
    assert "status" not in row and "requested_utc" not in row
    assert f"bls.ics  {len(kept)}  {_sha(kept)}  already saved" in capsys.readouterr().out


def test_a_second_run_that_day_fetches_nothing_and_writes_only_its_own_record_and_log(root):
    assert _run(("schedules", "members"))[0] == 0
    first = _files(root)
    later = NOW + timedelta(minutes=5)
    code, web = _run(("schedules", "members"), now=later)
    assert code == 0
    assert web.calls == []
    second = _files(root)
    assert {name: second[name] for name in first} == first
    assert sorted(set(second) - set(first)) == [
        f"_logs/dated-copies/{DAY}/run-140822Z.log",
        f"data_raw/macro_schedule/date={DAY}/_copy-140822Z.json",
        f"data_raw/sector_dated/date={DAY}/_copy-140822Z.json",
    ]
    rows = (
        _record(_schedules(root), "140822Z")["files"] + _record(_members(root), "140822Z")["files"]
    )
    assert [row["result"] for row in rows] == ["already saved"] * 10


def test_two_runs_in_one_second_the_second_refuses_and_writes_nothing(root, capsys):
    assert _run()[0] == 0
    before = _files(root.parent)
    capsys.readouterr()
    code, web = _run()
    assert code == 1
    assert web.calls == []
    assert _files(root.parent) == before
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.count("\n") == 1


# ── requests and redirects ────────────────────────────────────────────────────


def test_one_request_per_address_when_nothing_redirects(root):
    code, web = _run(("schedules", "members"))
    assert code == 0
    assert web.calls == [address for address, _, _ in NAMED]
    for filename in SCHEDULE_FILES:
        row = _row(_record(_schedules(root)), filename)
        assert row["redirects"] == [] and row["status"] == 200
        assert row["final_address"] == row["address"]
        assert row["requested_utc"] == "2026-11-17T14:03:22Z"


def test_each_followed_redirect_is_one_more_request_and_is_recorded(root):
    bare = "https://bea.gov/news/schedule"
    web = FakeWeb(
        {
            BEA_PAGE: _redirect(bare, 301),
            bare: _redirect("/news/schedule/", 302),
            bare + "/": _ok(_page("Release")),
        }
    )
    code, web = _run(web=web)
    assert code == 0
    assert web.calls.count(BEA_PAGE) == 1
    assert [
        call for call in web.calls if "bea.gov/news/schedule" in call and ".ics" not in call
    ] == [
        BEA_PAGE,
        bare,
        bare + "/",
    ]
    assert len(web.calls) == 8 + 2
    row = _row(_record(_schedules(root)), "bea_schedule.htm")
    assert row["redirects"] == [
        {"from": BEA_PAGE, "status": 301, "to": bare},
        {"from": bare, "status": 302, "to": bare + "/"},
    ]
    assert row["final_address"] == bare + "/" and row["status"] == 200
    assert row["result"] == "saved"
    assert (_schedules(root) / "bea_schedule.htm").read_bytes() == _page("Release")


@pytest.mark.parametrize(
    ("address", "filename", "www_form"),
    [
        (
            BEA_JSON,
            "bea_release_dates.json",
            "https://www.apps.bea.gov/API/signup/release_dates.json",
        ),
        (
            FED,
            "fed_fomccalendars.htm",
            "https://federalreserve.gov/monetarypolicy/fomccalendars.htm",
        ),
    ],
)
def test_a_redirect_between_a_host_and_its_www_form_is_followed_and_recorded(
    address, filename, www_form, root
):
    web = FakeWeb({address: _redirect(www_form, 308), www_form: _ok(_default_body(address))})
    code, web = _run(web=web)
    assert code == 0
    assert web.calls.count(address) == 1 and web.calls.count(www_form) == 1
    row = _row(_record(_schedules(root)), filename)
    assert row["redirects"] == [{"from": address, "status": 308, "to": www_form}]
    assert row["final_address"] == www_form and row["result"] == "saved"
    assert (_schedules(root) / filename).read_bytes() == _default_body(address)


@pytest.mark.parametrize(
    "target",
    [
        "https://calendar.example.org/fomccalendars.htm",  # another host
        "https://apps.federalreserve.gov/fomccalendars.htm",  # a sub-host that is not www
        "https://www.federalreserve.gov.example.org/x",  # a look-alike host
        "http://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",  # off https
        "https://www.federalreserve.gov:8443/fomccalendars.htm",  # another port
        "https://someone@www.federalreserve.gov/fomccalendars.htm",  # with a user name
        "//calendar.example.org/fomccalendars.htm",  # another host, scheme left out
        None,  # no target at all
    ],
)
def test_any_other_redirect_is_not_followed_and_counts_as_a_failure(target, root, capsys):
    code, web = _run(web=FakeWeb({FED: _redirect(target, 302)}))
    assert code == 10
    assert web.calls == SCHEDULE_ADDRESSES  # one request for it, and the target never asked
    assert not (_schedules(root) / "fed_fomccalendars.htm").exists()
    row = _row(_record(_schedules(root)), "fed_fomccalendars.htm")
    assert row["result"] == "failed" and row["status"] == 302
    assert row["redirect_not_followed"] == target and row["redirects"] == []
    assert "bytes" not in row and "sha256" not in row and "final_address" not in row
    for filename in SCHEDULE_FILES[1:]:
        assert (_schedules(root) / filename).is_file()
    out = capsys.readouterr().out
    assert "fed_fomccalendars.htm  -  -  failed (redirect" in out
    assert target is None or target not in out


def test_three_redirects_are_followed_and_a_fourth_is_not(root):
    hops = [FED, *(f"https://www.federalreserve.gov/hop{n}" for n in (1, 2, 3, 4))]
    three = {hops[n]: _redirect(hops[n + 1]) for n in range(3)}
    code, web = _run(web=FakeWeb({**three, hops[3]: _ok(_page("FOMC"))}))
    assert code == 0
    assert [call for call in web.calls if "federalreserve" in call] == hops[:4]
    row = _row(_record(_schedules(root)), "fed_fomccalendars.htm")
    assert len(row["redirects"]) == 3 and row["final_address"] == hops[3]


def test_a_fourth_redirect_is_a_failure(root):
    hops = [FED, *(f"https://www.federalreserve.gov/hop{n}" for n in (1, 2, 3, 4))]
    four = {hops[n]: _redirect(hops[n + 1]) for n in range(4)}
    code, web = _run(web=FakeWeb({**four, hops[4]: _ok(_page("FOMC"))}))
    assert code == 10
    assert [call for call in web.calls if "federalreserve" in call] == hops[:4]
    assert hops[4] not in web.calls
    row = _row(_record(_schedules(root)), "fed_fomccalendars.htm")
    assert row["result"] == "failed" and len(row["redirects"]) == 3
    assert not (_schedules(root) / "fed_fomccalendars.htm").exists()


# ── how the job names itself, and the one real request function ───────────────


def test_the_user_agent_names_the_repository_and_is_no_browsers():
    agent = cds.USER_AGENT
    assert agent == (
        "smart-wheel-engine-dated-copies/1.0 (+https://github.com/MertYakar66/smart-wheel-engine)"
    )
    assert "Mozilla" not in agent
    assert "@" not in agent and "＠" not in agent  # no email address
    assert "https://github.com/MertYakar66/smart-wheel-engine" in agent
    request = cds.build_request(FED)
    assert request.get_header("User-agent") == agent
    assert request.header_items() == [("User-agent", agent)]
    assert request.get_method() == "GET" and request.data is None


class _Canned(urllib.request.HTTPSHandler):
    """Answers in place of the network, inside urllib's own machinery: no socket is opened."""

    def __init__(self, status: int, headers, body: bytes) -> None:
        super().__init__()
        self.status, self.canned_headers, self.body = status, headers, body
        self.seen: list[urllib.request.Request] = []

    def https_open(self, req):
        self.seen.append(req)
        message = http.client.HTTPMessage()
        for name, value in self.canned_headers:
            message[name] = value
        reply = urllib.response.addinfourl(
            io.BytesIO(self.body), message, req.full_url, self.status
        )
        reply.msg = "hand-made"
        return reply


def _fetch_through_urllib(address: str, canned: _Canned):
    opener = cds.build_opener(canned, urllib.request.ProxyHandler({}))
    return _REAL_FETCH_ONCE(address, opener)


def test_the_real_fetch_sends_one_get_with_our_user_agent_and_a_30_second_timeout():
    body = b"FOMC, hand-made\r\n\xff"
    canned = _Canned(200, HEADERS, body)
    response = _fetch_through_urllib(FED, canned)
    assert response == cds.Response(200, HEADERS, body)
    (request,) = canned.seen
    assert request.full_url == FED and request.get_method() == "GET"
    assert request.get_header("User-agent") == cds.USER_AGENT
    assert request.timeout == 30 == cds.TIMEOUT_SECONDS
    sent = dict(request.header_items())
    assert not any("Mozilla" in value for value in sent.values())
    assert "Cookie" not in sent and "Authorization" not in sent and "From" not in sent


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_the_real_fetch_does_not_follow_a_redirect_by_itself(status):
    canned = _Canned(status, (("Location", "https://calendar.example.org/elsewhere"),), b"moved")
    response = _fetch_through_urllib(FED, canned)
    assert len(canned.seen) == 1  # urllib asked once and did not go on to the target
    assert response.status == status
    assert dict(response.headers)["Location"] == "https://calendar.example.org/elsewhere"


def test_the_real_fetch_returns_an_error_status_as_a_response():
    canned = _Canned(403, (("Content-Type", "text/plain"),), b"hand-made refusal")
    response = _fetch_through_urllib(BLS_PAGE, canned)
    assert len(canned.seen) == 1
    assert response == cds.Response(403, (("Content-Type", "text/plain"),), b"hand-made refusal")


# ── where it writes, and what it opens ────────────────────────────────────────


def test_every_path_written_lies_in_the_copy_folders_or_the_runs_log_folder(root):
    before = _files(root.parent)
    assert list(before) == ["root/data_raw/sp500_constituents_current.csv"]
    code, _ = _run(("schedules", "members"))
    assert code == 0
    after = _files(root.parent)
    allowed = (
        f"root/data_raw/macro_schedule/date={DAY}/",
        f"root/data_raw/sector_dated/date={DAY}/",
        f"root/_logs/dated-copies/{DAY}/",
    )
    new = sorted(set(after) - set(before))
    assert len(new) == 8 + 1 + 2 + 1 + 1  # schedules, their record, page + table, record, log
    assert all(name.startswith(allowed) for name in new), new
    assert after["root/data_raw/sp500_constituents_current.csv"] == DECOY
    assert list((root.parent / "cwd").iterdir()) == []
    folders = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir())
    assert folders == [
        "_logs",
        "_logs/dated-copies",
        f"_logs/dated-copies/{DAY}",
        "data_raw",
        "data_raw/macro_schedule",
        f"data_raw/macro_schedule/date={DAY}",
        "data_raw/sector_dated",
        f"data_raw/sector_dated/date={DAY}",
    ]


def test_a_write_outside_the_copy_and_log_folders_is_refused(root):
    allowed = frozenset({_schedules(root)})
    with pytest.raises(RuntimeError):
        cds._create(root / "data_raw" / "sp500_constituents_current.csv", b"x", allowed)
    with pytest.raises(RuntimeError):
        cds._create(root / "data_raw" / "new.csv", b"x", allowed)
    assert _files(root) == {"data_raw/sp500_constituents_current.csv": DECOY}


def test_a_file_is_created_and_never_opened_again_for_writing(root):
    folder = _schedules(root)
    folder.mkdir(parents=True)
    allowed = frozenset({folder})
    cds._create(folder / "bls.ics", b"first\r\n\xff", allowed)
    with pytest.raises(FileExistsError):
        cds._create(folder / "bls.ics", b"second", allowed)
    assert (folder / "bls.ics").read_bytes() == b"first\r\n\xff"


_AUDIT_DRIVER = r"""
import contextlib, importlib.util, io, json, os, socket, sys
from datetime import UTC, datetime, timedelta

script = sys.argv[1]


def no_network(*args, **kwargs):
    raise AssertionError("the audit driver tried to open a network connection")


socket.socket.connect = no_network
socket.create_connection = no_network

spec = importlib.util.spec_from_file_location("copy_dated_sources", script)
cds = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cds
spec.loader.exec_module(cds)

PAGE = (
    "<html><body><h1>{marker}, hand-made</h1><table><tr><th>Symbol</th><th>Company</th></tr>"
    "<tr><td>AAA</td><td>Alpha Works</td></tr></table></body></html>"
)


def fetch(address):
    for copy_list in cds.LISTS.values():
        for source in copy_list.sources:
            if source.address == address:
                if source.filename.endswith(".ics"):
                    body = b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n"
                elif source.filename.endswith(".json"):
                    body = b"{}"
                else:
                    body = PAGE.format(marker=source.marker).encode()
                return cds.Response(200, (("Content-Type", "text/html"),), body)
    raise AssertionError(address)


WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
events = []
watching = False


def hook(event, args):
    if not watching or not (event == "open" or event.startswith(("os.", "shutil."))):
        return
    paths = [os.fsdecode(arg) for arg in args if isinstance(arg, (str, bytes, os.PathLike))]
    writes = True
    if event == "open":
        paths = paths[:1]
        writes = bool(isinstance(args[2], int) and args[2] & WRITE)
    events.append({"event": event, "paths": paths, "writes": writes})


sys.addaudithook(hook)
start = datetime(2026, 11, 17, 14, 3, 22, tzinfo=UTC)
codes = []
with contextlib.redirect_stdout(io.StringIO()):
    watching = True
    for minutes in (0, 5):  # the second run finds every file already saved, and reads them
        moment = start + timedelta(minutes=minutes)
        codes.append(
            cds.main(
                ["--lists", "schedules", "members", "--date", "2026-11-17"],
                fetch=fetch,
                now=lambda: moment,
            )
        )
    watching = False
print(json.dumps({"codes": codes, "events": events}))
"""


def test_no_file_under_the_root_is_opened_outside_the_copy_and_log_folders(root):
    """Python's own audit events, in a child process: every open, mkdir, rename or
    removal under the data root during two runs, the second of which reads the copies."""
    env = {key: value for key, value in os.environ.items() if key != "SWE_DATA_ROOT"}
    env.update(SWE_DATA_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-c", _AUDIT_DRIVER, str(_SCRIPT)],
        env=env,
        cwd=root.parent / "cwd",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["codes"] == [0, 0]

    def norm(path: str | Path) -> str:
        return os.path.normcase(os.path.realpath(path))

    base = norm(root)
    folders = [norm(_schedules(root)), norm(_members(root)), norm(_logs(root))]
    opened_to_write, opened_to_read = set(), set()
    for event in report["events"]:
        for path in map(norm, event["paths"]):
            if path != base and not path.startswith(base + os.sep):
                continue
            inside = any(path.startswith(folder + os.sep) for folder in folders)
            on_the_way = any(
                folder == path or folder.startswith(path + os.sep) for folder in folders
            )
            if event["event"] == "os.mkdir":
                assert inside or on_the_way, event
                continue
            assert event["event"] == "open", event  # nothing is renamed, removed or linked
            assert inside, event
            (opened_to_write if event["writes"] else opened_to_read).add(path)
    copies = [_schedules(root) / name for name in SCHEDULE_FILES]
    copies += [_members(root) / "members_page.html", _members(root) / "members.csv"]
    own = [
        folder / f"_copy-{stamp}.json"
        for folder in (_schedules(root), _members(root))
        for stamp in ("140322Z", "140822Z")
    ]
    own += [_logs(root) / "run-140322Z.log", _logs(root) / "run-140822Z.log"]
    assert opened_to_write == {norm(path) for path in [*copies, *own]}
    assert opened_to_read == {norm(path) for path in copies}
    assert (root / "data_raw" / "sp500_constituents_current.csv").read_bytes() == DECOY
    assert list((root.parent / "cwd").iterdir()) == []


# ── refused, unanswered, looking wrong ────────────────────────────────────────


@pytest.mark.parametrize("status", [403, 404, 429, 500, 503, 204, 304])
def test_a_refused_address_gets_a_row_and_no_file_and_the_rest_are_saved(status, root, capsys):
    refusal = cds.Response(status, HEADERS, b"<html>hand-made refusal</html>")
    code, web = _run(web=FakeWeb({BLS_PAGE: refusal}))
    assert code == 10
    assert web.calls == SCHEDULE_ADDRESSES  # asked once, never again
    assert not (_schedules(root) / "bls_news_release.htm").exists()
    record = _record(_schedules(root))
    row = _row(record, "bls_news_release.htm")
    assert row["result"] == "refused" and row["status"] == status
    assert row["address"] == BLS_PAGE and row["headers"] == [list(pair) for pair in HEADERS]
    assert "bytes" not in row and "sha256" not in row and "looks_right" not in row
    others = [row for row in record["files"] if row["file"] != "bls_news_release.htm"]
    assert [row["result"] for row in others] == ["saved"] * 7
    for row in others:
        assert _sha((_schedules(root) / row["file"]).read_bytes()) == row["sha256"]
    assert f"bls_news_release.htm  -  -  refused (HTTP {status})" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("hand-made: timed out"),
        OSError("hand-made: connection refused"),
        ValueError("x"),
    ],
)
def test_an_address_that_does_not_answer_gets_a_row_and_no_file(error, root, capsys):
    code, web = _run(web=FakeWeb({BEA_ICS: error}))
    assert code == 10
    assert web.calls == SCHEDULE_ADDRESSES  # no retry
    assert not (_schedules(root) / "bea_schedule.ics").exists()
    row = _row(_record(_schedules(root)), "bea_schedule.ics")
    assert row["result"] == "failed" and type(error).__name__ in row["error"]
    assert "status" not in row and "headers" not in row and "sha256" not in row
    assert len([name for name in SCHEDULE_FILES if (_schedules(root) / name).is_file()]) == 7
    assert "bea_schedule.ics  -  -  failed (" in capsys.readouterr().out


def test_when_every_address_fails_the_record_and_the_log_are_still_written(root):
    web = FakeWeb(dict.fromkeys(SCHEDULE_ADDRESSES, OSError("hand-made: no network")))
    code, web = _run(web=web)
    assert code == 10 and web.calls == SCHEDULE_ADDRESSES
    assert sorted(path.name for path in _schedules(root).iterdir()) == [f"_copy-{STAMP}.json"]
    assert [row["result"] for row in _record(_schedules(root))["files"]] == ["failed"] * 8
    assert (_logs(root) / f"run-{STAMP}.log").read_bytes().count(b"\n") == 8


@pytest.mark.parametrize(
    ("address", "filename", "body"),
    [
        (
            FED,
            "fed_fomccalendars.htm",
            b"<html><body>hand-made, the marker is missing</body></html>",
        ),
        (CENSUS, "census_calendar_listview.html", b"<html>Economic  Indicator, two spaces</html>"),
        (BLS_ICS, "bls.ics", b"<html>hand-made: a page where a calendar should be</html>"),
        (BEA_JSON, "bea_release_dates.json", b'{"hand_made": tru'),
        (BEA_JSON, "bea_release_dates.json", b""),
    ],
)
def test_a_file_that_fails_its_light_check_is_saved_and_flagged(
    address, filename, body, root, capsys
):
    code, _ = _run(web=FakeWeb({address: _ok(body)}))
    assert code == 10
    assert (_schedules(root) / filename).read_bytes() == body  # the check never stops a save
    row = _row(_record(_schedules(root)), filename)
    assert row["result"] == "looks wrong" and row["looks_right"] == "no"
    assert row["sha256"] == _sha(body) and row["bytes"] == len(body)
    assert f"{filename}  {len(body)}  {_sha(body)}  looks wrong (" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("address", "filename", "body"),
    [
        (FED, "fed_fomccalendars.htm", b"<html>the fomc calendar, hand-made, lower case</html>"),
        (CENSUS, "census_calendar_listview.html", b"<html>ECONOMIC INDICATOR CALENDAR</html>"),
        (BLS_ICS, "bls.ics", b"\xef\xbb\xbfBEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n"),
        (BEA_JSON, "bea_release_dates.json", b"[]"),
    ],
)
def test_the_light_check_passes_whatever_the_letter_case(address, filename, body, root):
    code, _ = _run(web=FakeWeb({address: _ok(body)}))
    assert code == 0
    row = _row(_record(_schedules(root)), filename)
    assert row["result"] == "saved" and row["looks_right"] == "yes"


# ── the members table ─────────────────────────────────────────────────────────


def test_members_csv_is_built_from_the_page_and_listed_in_the_record(root, capsys):
    code, web = _run(("members",))
    assert code == 0 and web.calls == [WIKI]
    assert (_members(root) / "members_page.html").read_bytes() == MEMBERS_PAGE
    table = (_members(root) / "members.csv").read_bytes()
    assert table == MEMBERS_CSV
    assert b"\r" not in table and "Zürich".encode() in table
    record = _record(_members(root))
    assert [row["file"] for row in record["files"]] == ["members_page.html", "members.csv"]
    page_row, table_row = record["files"]
    assert page_row["result"] == "saved" and page_row["looks_right"] == "yes"
    assert page_row["sha256"] == _sha(MEMBERS_PAGE)
    assert table_row == {
        "file": "members.csv",
        "derived_from": "members_page.html",
        "result": "saved",
        "bytes": len(MEMBERS_CSV),
        "sha256": _sha(MEMBERS_CSV),
        "rows": 2,
    }
    assert capsys.readouterr().out.splitlines() == [
        f"members_page.html  {len(MEMBERS_PAGE)}  {_sha(MEMBERS_PAGE)}  saved",
        f"members.csv  {len(MEMBERS_CSV)}  {_sha(MEMBERS_CSV)}  saved",
    ]


def test_a_page_with_no_such_table_is_kept_and_no_members_csv_is_written(root, capsys):
    code, _ = _run(("members",), web=FakeWeb({WIKI: _ok(PAGE_WITHOUT_TABLE)}))
    assert code == 10
    assert (_members(root) / "members_page.html").read_bytes() == PAGE_WITHOUT_TABLE
    assert not (_members(root) / "members.csv").exists()
    page_row, table_row = _record(_members(root))["files"]
    assert page_row["result"] == "looks wrong" and page_row["looks_right"] == "no"
    assert page_row["sha256"] == _sha(PAGE_WITHOUT_TABLE)
    assert table_row["file"] == "members.csv" and table_row["result"] == "not written"
    assert "sha256" not in table_row
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("members_page.html  ") and "looks wrong (" in line


def test_when_the_page_is_refused_no_table_is_written(root):
    code, _ = _run(("members",), web=FakeWeb({WIKI: cds.Response(403, HEADERS, b"no")}))
    assert code == 10
    assert sorted(path.name for path in _members(root).iterdir()) == [f"_copy-{STAMP}.json"]
    page_row, table_row = _record(_members(root))["files"]
    assert page_row["result"] == "refused" and table_row["result"] == "not written"


@pytest.mark.parametrize(
    ("content_type", "encoding"),
    [
        ("text/html; charset=ISO-8859-1", "iso-8859-1"),
        ('text/html; charset="utf-8"', "utf-8"),
        ("text/html", "utf-8"),
        ("text/html; charset=no-such-charset", "utf-8"),
        (None, "utf-8"),
    ],
)
def test_the_page_is_decoded_with_its_content_type_charset_utf8_if_none(
    content_type, encoding, root
):
    served = MEMBERS_PAGE.decode("utf-8").encode(encoding)
    headers = () if content_type is None else (("Content-Type", content_type),)
    code, _ = _run(("members",), web=FakeWeb({WIKI: _ok(served, headers)}))
    assert code == 0
    assert (_members(root) / "members_page.html").read_bytes() == served
    assert (_members(root) / "members.csv").read_bytes() == MEMBERS_CSV  # always UTF-8


def test_a_missing_table_is_built_from_the_page_an_earlier_run_saved(root):
    _members(root).mkdir(parents=True)
    (_members(root) / "members_page.html").write_bytes(MEMBERS_PAGE)
    code, web = _run(("members",))
    assert code == 0 and web.calls == []
    assert (_members(root) / "members_page.html").read_bytes() == MEMBERS_PAGE
    assert (_members(root) / "members.csv").read_bytes() == MEMBERS_CSV
    page_row, table_row = _record(_members(root))["files"]
    assert page_row["result"] == "already saved" and table_row["result"] == "saved"


def test_an_earlier_page_with_no_such_table_still_counts_as_looking_wrong(root, capsys):
    _members(root).mkdir(parents=True)
    (_members(root) / "members_page.html").write_bytes(PAGE_WITHOUT_TABLE)
    code, web = _run(("members",))
    assert code == 10 and web.calls == []
    assert not (_members(root) / "members.csv").exists()
    assert (_members(root) / "members_page.html").read_bytes() == PAGE_WITHOUT_TABLE
    assert "looks wrong (saved by an earlier run; " in capsys.readouterr().out


def test_an_existing_members_csv_is_never_written_over(root):
    kept = b"kept,from,an,earlier,run\r\n"
    _members(root).mkdir(parents=True)
    (_members(root) / "members.csv").write_bytes(kept)
    code, web = _run(("members",))
    assert code == 0 and web.calls == [WIKI]
    assert (_members(root) / "members.csv").read_bytes() == kept
    table_row = _record(_members(root))["files"][1]
    assert table_row["result"] == "already saved" and table_row["sha256"] == _sha(kept)


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        # the first table with the heading, not an earlier or a later one
        (
            "<table><tr><th>Name</th></tr><tr><td>no</td></tr></table>"
            "<table><tr><th>Symbol</th><th>Name</th></tr><tr><td>AAA</td><td>One</td></tr></table>"
            "<table><tr><th>Symbol</th></tr><tr><td>ZZZ</td></tr></table>",
            [["Symbol", "Name"], ["AAA", "One"]],
        ),
        # end tags left out, as HTML allows
        (
            "<table><tr><th>Symbol<th>Name<tr><td>AAA<td>One<tr><td>BBB<td>Two</table>",
            [["Symbol", "Name"], ["AAA", "One"], ["BBB", "Two"]],
        ),
        # tags removed, entities decoded, whitespace made one space, ends trimmed
        (
            "<table><tr><td> Symbol </td><td><b>Na</b>me</td></tr>"
            "<tr><td>\n  <a href='#'>A&amp;A</a>\t</td><td>One&nbsp;&nbsp;\n two &#233;</td></tr></table>",
            [["Symbol", "Name"], ["A&A", "One two é"]],
        ),
        # script and style are not cell text; a table inside a cell is its own table
        (
            "<table><tr><th>Symbol</th><th>Name</th></tr><tr><td><script>var x = 1;</script>AAA</td>"
            "<td>One<table><tr><td>inner</td></tr></table></td></tr></table>",
            [["Symbol", "Name"], ["AAA", "One"]],
        ),
        # a table never closed is still read to the end of the page
        ("<table><tr><th>Symbol</th></tr><tr><td>AAA</td></tr>", [["Symbol"], ["AAA"]]),
    ],
)
def test_the_table_rules(page, expected):
    assert cds.members_table(page) == expected


@pytest.mark.parametrize(
    "page",
    [
        "",
        "<p>no table at all</p>",
        "<table><tr><th>symbol</th></tr></table>",  # the heading reads "Symbol", exactly
        "<table><tr><th>Symbols</th></tr></table>",
        "<table><tr><th>Name</th></tr><tr><td>Symbol</td></tr></table>",  # not the heading row
    ],
)
def test_no_table_with_the_heading_is_no_table(page):
    assert cds.members_table(page) is None


def test_the_table_file_is_utf8_with_lf_endings_and_quotes_what_needs_it():
    data = cds.table_csv([["Symbol", "Name"], ["AAA", 'One, "Two"'], ["BBB", "Zürich"]])
    assert data == 'Symbol,Name\nAAA,"One, ""Two"""\nBBB,Zürich\n'.encode()


# ── the output, the log and the record ────────────────────────────────────────


def test_no_output_line_and_no_log_line_holds_a_response_header(root, capsys):
    elsewhere = "https://calendar.example.org/hand-made-target"
    bare = "https://bea.gov/news/schedule"
    web = FakeWeb(
        {
            FED: _redirect(elsewhere, 302),  # not followed
            BLS_PAGE: cds.Response(403, HEADERS, b"hand-made refusal"),
            BEA_PAGE: _redirect(bare, 301),  # followed
            bare: _ok(_page("Release")),
            BEA_ICS: TimeoutError("hand-made: timed out"),
            CENSUS: _ok(b"<html>hand-made, the marker is missing</html>"),
        }
    )
    code, _ = _run(("schedules", "members"), web=web)
    assert code == 10
    out = capsys.readouterr().out
    log = (_logs(root) / f"run-{STAMP}.log").read_bytes()
    assert b"\r" not in log
    assert log.decode("utf-8") == out  # the same lines, in the same order
    lines = out.splitlines()
    assert [line.split("  ")[0] for line in lines] == [
        *SCHEDULE_FILES,
        "members_page.html",
        "members.csv",
    ]
    assert all(len(line.split("  ")) == 4 for line in lines)
    served = [
        *HEADERS,
        ("Location", elsewhere),
        ("Location", bare),
        ("X-Served-By", "hand-made-cache-7f3a"),
    ]
    for name, value in served:
        assert name not in out and name.lower() not in out.lower()
        assert value not in out
    for part in ("charset", "visit=4be1c0", "hand-made-cache", "Secure", "example.org"):
        assert part not in out
    # The record is where the headers go.
    record = _record(_schedules(root))
    assert _row(record, "bls_news_release.htm")["headers"] == [list(pair) for pair in HEADERS]
    assert _row(record, "fed_fomccalendars.htm")["redirect_not_followed"] == elsewhere
    assert _row(record, "bea_schedule.htm")["redirects"][0]["to"] == bare


def test_the_record_names_the_run_the_list_and_how_the_job_named_itself(root):
    assert _run(("schedules", "members"))[0] == 0
    for folder, name, count in ((_schedules(root), "schedules", 8), (_members(root), "members", 2)):
        raw = (folder / f"_copy-{STAMP}.json").read_bytes()
        assert b"\r" not in raw and raw.endswith(b"}\n")
        record = json.loads(raw)
        assert record["list"] == name and record["copy_date"] == DAY
        assert record["run_utc"] == "2026-11-17T14:03:22Z"
        assert record["user_agent"] == cds.USER_AGENT
        assert len(record["files"]) == count
    row = _row(_record(_schedules(root)), "bea_release_dates.json")
    assert row == {
        "address": BEA_JSON,
        "file": "bea_release_dates.json",
        "requested_utc": "2026-11-17T14:03:22Z",
        "redirects": [],
        "final_address": BEA_JSON,
        "status": 200,
        "headers": [list(pair) for pair in HEADERS],
        "bytes": len(JSON_BODY),
        "sha256": _sha(JSON_BODY),
        "looks_right": "yes",
        "check": "parses as JSON",
        "result": "saved",
    }


def test_an_error_text_is_one_line_of_plain_ascii(root, capsys):
    error = OSError("hand-made: bağlantı\r\n  kurulamadı")
    code, _ = _run(web=FakeWeb({FED: error}))
    assert code == 10
    (line,) = [line for line in capsys.readouterr().out.splitlines() if line.startswith("fed_")]
    assert line.isascii() and "ba\\u011flant\\u0131 kurulamad\\u0131" in line


# ── the dry run, and the command line ─────────────────────────────────────────


def test_a_dry_run_fetches_nothing_and_writes_nothing(root, capsys):
    before = _files(root.parent)
    code, web = _run(("schedules", "members"), extra=("--dry-run",))
    assert code == 0
    assert web.calls == []
    assert _files(root.parent) == before
    assert sorted(path.name for path in root.iterdir()) == ["data_raw"]
    assert sorted(path.name for path in (root / "data_raw").iterdir()) == [
        "sp500_constituents_current.csv"
    ]
    lines = capsys.readouterr().out.splitlines()
    resolved = root.resolve()
    expected = [
        f"{address} -> {resolved / 'data_raw' / 'macro_schedule' / f'date={DAY}' / filename}"
        for address, filename, _ in NAMED[:8]
    ]
    expected.append(
        f"{WIKI} -> {resolved / 'data_raw' / 'sector_dated' / f'date={DAY}' / 'members_page.html'}"
    )
    expected.append(
        "(a table built from members_page.html) -> "
        f"{resolved / 'data_raw' / 'sector_dated' / f'date={DAY}' / 'members.csv'}"
    )
    assert lines == expected


def test_a_dry_run_still_needs_the_root_and_todays_date(root, monkeypatch):
    assert _run(date="2026-11-16", extra=("--dry-run",))[0] == 5
    monkeypatch.delenv("SWE_DATA_ROOT")
    assert _run(extra=("--dry-run",))[0] == 3


def _command_line(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != "SWE_DATA_ROOT"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_command_line_without_a_data_root_exits_3_with_one_line(tmp_path):
    done = _command_line(tmp_path, "--lists", "schedules", "--date", "2000-01-01")
    assert done.returncode == 3
    assert done.stdout == ""
    assert done.stderr.count("\n") == 1 and "SWE_DATA_ROOT" in done.stderr
    assert list(tmp_path.iterdir()) == []


def test_the_command_line_without_lists_is_a_usage_error(tmp_path):
    done = _command_line(tmp_path, "--date", "2000-01-01")
    assert done.returncode == 2 and "--lists" in done.stderr
    assert list(tmp_path.iterdir()) == []
