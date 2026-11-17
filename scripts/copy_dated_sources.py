#!/usr/bin/env python3
"""Weekly dated copies of the official release schedules and the S&P 500 members list.

``DECISIONS.md`` D29, "Amended on 2026-11-17 (ruling 8)", is the rule, and
``docs/DATED_COPIES.md`` is how the job is run. This script takes the copy and
does nothing else: it copies nothing to Drive, it changes no file the engine or
its scripts read, and it is never added to ``scripts/pull_all.py``.

Usage (on the desktop, where ``SWE_DATA_ROOT`` names the data root)::

    python scripts/copy_dated_sources.py --lists schedules --date 2026-11-17
    python scripts/copy_dated_sources.py --lists schedules members --date 2026-11-17
    python scripts/copy_dated_sources.py --lists schedules --date 2026-11-17 --dry-run

``--lists`` and ``--date`` are both required; there is no default list. The
date must be today's date in UTC. ``--dry-run`` prints each address and the
path its copy would get, and fetches and writes nothing.

**What it writes**, and nothing else:

- ``<root>/data_raw/macro_schedule/date=<copy date>/``: the eight schedule files;
- ``<root>/data_raw/sector_dated/date=<copy date>/``: ``members_page.html`` as
  served, and ``members.csv``, a table built from that page;
- beside the copies, one ``_copy-<HHMMSS>Z.json`` record per run and list;
- ``<root>/_logs/dated-copies/<copy date>/run-<HHMMSS>Z.log``.

A file is created in binary mode with the exclusive flag, so the bytes are kept
as served and an existing file is never opened for writing. If an address's
file is already there, the address is not fetched and is reported as already
saved, with the size and sha256 of the file found. The script reads no file
under the data root outside those folders.

**Fetching.** Only the standard library (``urllib``). One request per address,
a 30-second timeout and no retry. ``urllib`` is stopped from following a
redirect by itself: each one is checked here, and it is followed only over
https and only to the same host and port, or between a host and its ``www``
form, at most three times. Each followed redirect is one more request. Any
other redirect is not followed and counts as a failure. Only a 200 answer is
saved. The job names itself by the project and its public repository, with no
email address and never a browser's User-Agent (D29).

**The table.** ``members.csv`` is the first table in the page whose heading row
(its first row) has a cell reading "Symbol". Each cell is its text with tags
removed, entities decoded, runs of whitespace made one space and the ends
trimmed; the text of ``<script>`` and ``<style>`` is not cell text. The headings
are the page's own. The page is decoded with the charset in its Content-Type
header, UTF-8 if there is none. The file is UTF-8 with LF line endings. With no
such table the page is still kept, no ``members.csv`` is written, and the
address counts as looking wrong. If the page was saved by an earlier run that
day and ``members.csv`` is missing, the table is built from the saved page,
with the Content-Type that run's record holds for it.

**The light check** never stops a save: an ``.ics`` file should begin with
``BEGIN:VCALENDAR`` (one byte-order mark and blank space before it are
allowed), a ``.json`` file should parse, and an HTML page should contain its
marker, whatever the letter case. The record says whether each file looks
right.

**The record** (``_copy-<HHMMSS>Z.json``) lists for each address: the address,
each redirect and the final address, the file name, the UTC time of the
request, the HTTP status, the response headers, the byte count, the sha256 and
the result of the light check. ``members.csv`` is listed too, as derived from
``members_page.html``. An address that answers with another status, or does not
answer, gets its row with the status or the error and no file. Response headers
go only into the record: never to the screen or the log. The record is written
even when the run is stopped part-way (``"finished": false``), so every file
saved so far has its row.

**Output.** One line per file: its name, bytes, sha256 and result (saved,
already saved, looks wrong, refused with its status, or failed). The same lines
go to the run's log. A failure after a followed redirect names only the kind of
error: its text may quote the address the server sent, and stays in the record.

**Exit codes.** 0: every address in the named lists is saved, now or by an
earlier run that day, and every file saved in this run looks right. 10: one or
more addresses were refused or failed, or look wrong, and the rest were saved.
3: ``SWE_DATA_ROOT`` is unset or blank, or names no existing folder. 5:
``--date`` is not today's date in UTC. 2: a usage error (argparse's own). 1:
this run's own log or record file is already there (two runs in one second);
nothing is fetched or written.

The script imports nothing from ``engine/``. Pure standard library.
"""

import argparse
import codecs
import csv
import hashlib
import io
import json
import os
import re
import string
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

USER_AGENT = (
    "smart-wheel-engine-dated-copies/1.0 (+https://github.com/MertYakar66/smart-wheel-engine)"
)
TIMEOUT_SECONDS = 30
MAX_REDIRECTS = 3
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

EXIT_OK = 0
EXIT_CLASH = 1
EXIT_NO_ROOT = 3
EXIT_WRONG_DATE = 5
EXIT_INCOMPLETE = 10

NO_TABLE = 'no table whose heading row has a cell reading "Symbol"'


@dataclass(frozen=True)
class Source:
    """One address, the name its copy gets, and the marker an HTML page should hold."""

    address: str
    filename: str
    marker: str | None = None


@dataclass(frozen=True)
class CopyList:
    """One list: its folder under the data root, its addresses, and its derived table."""

    name: str
    folder: tuple[str, ...]
    sources: tuple[Source, ...]
    table_from: str | None = None
    table_file: str | None = None


LISTS: dict[str, CopyList] = {
    "schedules": CopyList(
        name="schedules",
        folder=("data_raw", "macro_schedule"),
        sources=(
            Source(
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                "fed_fomccalendars.htm",
                "FOMC",
            ),
            Source(
                "https://www.bls.gov/schedule/news_release/",
                "bls_news_release.htm",
                "Schedule",
            ),
            Source("https://www.bls.gov/schedule/news_release/bls.ics", "bls.ics"),
            Source("https://www.bea.gov/news/schedule", "bea_schedule.htm", "Release"),
            Source(
                "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics",
                "bea_schedule.ics",
            ),
            Source(
                "https://apps.bea.gov/API/signup/release_dates.json",
                "bea_release_dates.json",
            ),
            Source(
                "https://www.census.gov/economic-indicators/calendar-listview.html",
                "census_calendar_listview.html",
                "Economic Indicator",
            ),
            Source(
                "https://www.census.gov/economic-indicators/calendar-listview-2026.html",
                "census_calendar_listview_2026.html",
                "Economic Indicator",
            ),
        ),
    ),
    "members": CopyList(
        name="members",
        folder=("data_raw", "sector_dated"),
        sources=(
            Source(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "members_page.html",
                "500 companies",
            ),
        ),
        table_from="members_page.html",
        table_file="members.csv",
    ),
}


@dataclass(frozen=True)
class Response:
    """What one HTTP request brought back. Nothing here has followed a redirect."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


# ── fetching ──────────────────────────────────────────────────────────────────


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop urllib following a redirect by itself; ``copy_address`` checks each one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_opener(*handlers: Any) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect, *handlers)


def build_request(address: str) -> urllib.request.Request:
    return urllib.request.Request(address, headers={"User-Agent": USER_AGENT}, method="GET")


def fetch_once(address: str, opener: urllib.request.OpenerDirector | None = None) -> Response:
    """One HTTP request. A redirect or an error status comes back as a ``Response``;
    no answer at all (a timeout, a refused connection, a broken reply) raises."""
    opener = opener or build_opener()
    try:
        with opener.open(build_request(address), timeout=TIMEOUT_SECONDS) as reply:
            return Response(reply.status, tuple(reply.headers.items()), reply.read())
    except urllib.error.HTTPError as err:
        with err:
            headers = tuple(err.headers.items()) if err.headers is not None else ()
            return Response(err.code, headers, err.read())


def _port(parts: urllib.parse.SplitResult) -> int:
    return 443 if parts.port is None else parts.port


def redirect_target(
    current: str, location: str | None, followed: int
) -> tuple[str | None, str | None]:
    """Where a redirect is followed to, or why it is not. Returns (address, reason);
    one of the two is None. The reason never repeats what the server sent."""
    if not location:
        return None, "redirect without a target address, not followed"
    try:
        # Quoted as urllib's own redirect handler quotes it, so a space or a
        # letter outside ASCII in the target can still be sent.
        target = urllib.parse.quote(
            urllib.parse.urljoin(current, location),
            encoding="iso-8859-1",
            safe=string.punctuation,
        )
        here, there = urllib.parse.urlsplit(current), urllib.parse.urlsplit(target)
        same_port = _port(here) == _port(there)
    except ValueError:
        return None, "redirect to an address that cannot be read, not followed"
    host, new_host = (here.hostname or "").lower(), (there.hostname or "").lower()
    same_site = bool(new_host) and (
        new_host == host or new_host == "www." + host or host == "www." + new_host
    )
    if (
        there.scheme != "https"
        or not same_site
        or not same_port
        or there.username is not None
        or there.password is not None
    ):
        return None, "redirect to another host or off https, not followed"
    if followed >= MAX_REDIRECTS:
        return None, "more than three redirects, not followed"
    return target, None


# ── the light check ───────────────────────────────────────────────────────────


def looks_right(source: Source, body: bytes) -> tuple[bool, str]:
    """The light check for one saved file: (looks right, what was checked)."""
    name = source.filename.lower()
    if name.endswith(".ics"):
        start = body[3:] if body.startswith(b"\xef\xbb\xbf") else body
        ok = start.lstrip(b" \t\r\n").startswith(b"BEGIN:VCALENDAR")
        return ok, "begins with BEGIN:VCALENDAR" if ok else "does not begin with BEGIN:VCALENDAR"
    if name.endswith(".json"):
        try:
            json.loads(body)
        except (ValueError, RecursionError):
            return False, "does not parse as JSON"
        return True, "parses as JSON"
    marker = source.marker or ""
    ok = bool(marker) and marker.lower().encode("ascii") in body.lower()
    return ok, f'contains "{marker}"' if ok else f'marker "{marker}" not found'


# ── the table ─────────────────────────────────────────────────────────────────


class _OpenTable:
    def __init__(self, rows: list[list[str]]) -> None:
        self.rows = rows
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def end_cell(self) -> None:
        if self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
        self.cell = None

    def end_row(self) -> None:
        self.end_cell()
        if self.row:
            self.rows.append(self.row)
        self.row = None


class _Tables(HTMLParser):
    """Every table in a page as rows of cell texts, in the order the tables open."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._open: list[_OpenTable] = []
        self._not_text = 0  # inside <script> or <style>

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._not_text += 1
        elif tag == "table":
            rows: list[list[str]] = []
            self.tables.append(rows)
            self._open.append(_OpenTable(rows))
        elif self._open and tag == "tr":
            self._open[-1].end_row()
            self._open[-1].row = []
        elif self._open and tag in ("td", "th"):
            table = self._open[-1]
            table.end_cell()
            if table.row is None:
                table.row = []
            table.cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._not_text = max(0, self._not_text - 1)
        elif not self._open:
            return
        elif tag in ("td", "th"):
            self._open[-1].end_cell()
        elif tag == "tr":
            self._open[-1].end_row()
        elif tag == "table":
            self._open.pop().end_row()

    def handle_data(self, data: str) -> None:
        if self._open and not self._not_text and self._open[-1].cell is not None:
            self._open[-1].cell.append(data)

    def finish(self) -> None:
        self.close()
        while self._open:
            self._open.pop().end_row()


def members_table(page_text: str) -> list[list[str]] | None:
    """The first table whose heading row has a cell reading "Symbol", or None."""
    parser = _Tables()
    try:
        parser.feed(page_text)
        parser.finish()
    except Exception:  # a page the parser cannot read holds no table we can use
        return None
    for rows in parser.tables:
        if rows and "Symbol" in rows[0]:
            return rows
    return None


def table_csv(rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8")


def charset_of(headers: tuple[tuple[str, str], ...]) -> str:
    """The charset in the Content-Type header; UTF-8 if there is none we know."""
    try:
        message = Message()
        message["Content-Type"] = _header(headers, "Content-Type") or "application/octet-stream"
        return codecs.lookup(message.get_content_charset() or "utf-8").name
    except Exception:  # a header we cannot read names no charset we know
        return "utf-8"


def decode_page(body: bytes, charset: str) -> str:
    """The page as text. A charset that is no text encoding falls back to UTF-8."""
    try:
        text = body.decode(charset, errors="replace")
        text.encode("utf-8")  # a codec that hands back lone surrogates is no text encoding
        return text
    except (LookupError, UnicodeError):
        return body.decode("utf-8", errors="replace")


def recorded_headers(folder: Path, filename: str, digest: str) -> tuple[tuple[str, str], ...]:
    """The response headers an earlier run that day recorded for these bytes; () if none."""
    for path in sorted(folder.glob("_copy-*.json"), reverse=True):
        try:
            for row in json.loads(path.read_bytes())["files"]:
                if row.get("file") == filename and row.get("sha256") == digest:
                    if row.get("headers"):
                        return tuple((str(name), str(value)) for name, value in row["headers"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return ()


# ── small helpers ─────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header(headers: tuple[tuple[str, str], ...], name: str) -> str | None:
    for key, value in headers:
        if key.lower() == name.lower():
            return value
    return None


def _one_line(text: str) -> str:
    """One line of plain ASCII, so it prints on any console."""
    return " ".join(text.split()).encode("ascii", "backslashreplace").decode("ascii")


def _say(text: str) -> None:
    """Print one line, escaped if the console cannot show it as it is."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _create(path: Path, data: bytes, allowed: frozenset[Path]) -> None:
    """Write a new file. Mode ``xb``: the bytes as given, and never over a file."""
    if path.parent not in allowed:
        raise RuntimeError(f"refusing to write outside the copy and log folders: {path}")
    with open(path, "xb") as handle:
        handle.write(data)


def _read_existing(path: Path, row: dict[str, Any]) -> bytes | None:
    """The bytes of a file that is already there; None, with the row filled in, if it
    cannot be read (a folder or a broken link under that name)."""
    try:
        return path.read_bytes()
    except OSError as exc:
        row.update(
            result="failed",
            error=f"{path.name} is there but cannot be read ({type(exc).__name__})",
        )
        return None


def _line(row: dict[str, Any]) -> str:
    result = row["result"]
    if result == "refused":
        result = f"refused (HTTP {row['status']})"
    elif result == "failed":
        shown = row["error"]
        if row.get("error_type") and row.get("redirects"):
            # The address asked came from the server, and the error may quote it.
            shown = f"{row['error_type']} after a redirect; the record holds the text"
        result = f"failed ({shown})"
    elif result == "looks wrong":
        result = f"looks wrong ({row['check']})"
    size = row.get("bytes")
    return f"{row['file']}  {'-' if size is None else size}  {row.get('sha256') or '-'}  {result}"


# ── one address, one table ────────────────────────────────────────────────────


def copy_address(
    source: Source,
    folder: Path,
    fetch: Callable[[str], Response],
    now: Callable[[], datetime],
    allowed: frozenset[Path],
) -> tuple[dict[str, Any], Response | None]:
    """Fetch one address and save its body as a new file. Returns the record's row,
    and the response when its body was saved in this call."""
    target = folder / source.filename
    row: dict[str, Any] = {"address": source.address, "file": source.filename}
    if os.path.lexists(target):
        found = _read_existing(target, row)
        if found is not None:
            row.update(result="already saved", bytes=len(found), sha256=_sha256(found))
        return row, None

    redirects: list[dict[str, Any]] = []
    row.update(requested_utc=_iso(now()), redirects=redirects)
    address = source.address
    while True:
        try:
            response = fetch(address)
        except Exception as exc:  # no answer: a timeout, a refused connection, a broken reply
            row.update(
                result="failed",
                error=_one_line(f"{type(exc).__name__}: {exc}"),
                error_type=type(exc).__name__,
            )
            return row, None
        if response.status not in REDIRECT_STATUSES:
            break
        location = _header(response.headers, "Location")
        target_address, reason = redirect_target(address, location, len(redirects))
        if target_address is None:
            row.update(
                result="failed",
                error=reason,
                status=response.status,
                headers=[list(pair) for pair in response.headers],
                redirect_not_followed=location,
            )
            return row, None
        redirects.append({"from": address, "status": response.status, "to": target_address})
        address = target_address

    row.update(
        final_address=address,
        status=response.status,
        headers=[list(pair) for pair in response.headers],
    )
    if response.status != 200:
        row["result"] = "refused"
        return row, None
    try:
        _create(target, response.body, allowed)
    except FileExistsError:
        row.update(
            result="failed",
            error=f"{source.filename} appeared while it was being fetched; not written over",
        )
        return row, None
    except OSError as exc:
        row.update(
            result="failed",
            error=f"{source.filename} could not be written in full ({type(exc).__name__}); "
            "what is on disk under that name is not a copy",
        )
        return row, None
    ok, what = looks_right(source, response.body)
    row.update(
        bytes=len(response.body),
        sha256=_sha256(response.body),
        looks_right="yes" if ok else "no",
        check=what,
        result="saved" if ok else "looks wrong",
    )
    return row, response


def build_table(
    copy_list: CopyList, folder: Path, response: Response | None, allowed: frozenset[Path]
) -> dict[str, Any]:
    """Write the list's table from its page, unless it is already there. ``response``
    is the page as just fetched, or None when an earlier run that day saved it."""
    assert copy_list.table_file and copy_list.table_from
    row: dict[str, Any] = {"file": copy_list.table_file, "derived_from": copy_list.table_from}
    target = folder / copy_list.table_file
    if os.path.lexists(target):
        found = _read_existing(target, row)
        if found is None:
            row.update(result="not written", reason=row.pop("error"))
        else:
            row.update(result="already saved", bytes=len(found), sha256=_sha256(found))
        return row
    page = folder / copy_list.table_from
    if response is not None:
        body, headers = response.body, response.headers
    elif os.path.lexists(page):
        try:
            body = page.read_bytes()
        except OSError:
            row.update(result="not written", reason=f"{copy_list.table_from} cannot be read")
            return row
        # Saved by an earlier run that day: its record holds the page's headers.
        headers = recorded_headers(folder, copy_list.table_from, _sha256(body))
    else:
        row.update(result="not written", reason=f"{copy_list.table_from} was not saved")
        return row
    charset = charset_of(headers)
    rows = members_table(decode_page(body, charset))
    if rows is None:
        row.update(result="not written", reason=NO_TABLE)
        return row
    data = table_csv(rows)
    try:
        _create(target, data, allowed)
    except OSError as exc:
        row.update(
            result="not written",
            reason=f"{copy_list.table_file} could not be written in full ({type(exc).__name__})",
        )
        return row
    row.update(
        result="saved",
        bytes=len(data),
        sha256=_sha256(data),
        rows=len(rows) - 1,
        page_read_as=charset,
    )
    return row


# ── the run ───────────────────────────────────────────────────────────────────


def _date(value: str) -> date:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise argparse.ArgumentTypeError(f"{value!r} is not a date written YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copy_dated_sources.py",
        description="Take the weekly dated copy of the named lists (DECISIONS.md D29, "
        "amended 2026-11-17; docs/DATED_COPIES.md). Needs SWE_DATA_ROOT.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--lists",
        nargs="+",
        action="extend",
        required=True,
        choices=tuple(LISTS),
        metavar="LIST",
        help="the lists to copy: schedules, members, or both; there is no default",
    )
    parser.add_argument(
        "--date",
        required=True,
        type=_date,
        metavar="YYYY-MM-DD",
        help="the copy date; it must be today's date in UTC",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the addresses and the paths their copies would get; fetch and write nothing",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[str], Response] | None = None,
    now: Callable[[], datetime] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    fetch = fetch or fetch_once
    now = now or utc_now
    wanted = [copy_list for name, copy_list in LISTS.items() if name in args.lists]

    raw_root = os.environ.get("SWE_DATA_ROOT", "").strip()
    if not raw_root:
        print(
            "SWE_DATA_ROOT is unset or blank, so there is no data root to copy into: "
            "nothing fetched, nothing written.",
            file=sys.stderr,
        )
        return EXIT_NO_ROOT
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        print(
            f"SWE_DATA_ROOT names {root}, which is not an existing folder: "
            "nothing fetched, nothing written.",
            file=sys.stderr,
        )
        return EXIT_NO_ROOT

    started = now().astimezone(UTC)
    if args.date != started.date():
        print(
            f"--date {args.date.isoformat()} is not today's date in UTC "
            f"({started.date().isoformat()}): nothing fetched, nothing written.",
            file=sys.stderr,
        )
        return EXIT_WRONG_DATE

    day = args.date.isoformat()
    folders = {cl.name: root.joinpath(*cl.folder, f"date={day}") for cl in wanted}
    if args.dry_run:
        for copy_list in wanted:
            folder = folders[copy_list.name]
            for source in copy_list.sources:
                _say(f"{source.address} -> {folder / source.filename}")
            if copy_list.table_file:
                _say(
                    f"(a table built from {copy_list.table_from}) -> "
                    f"{folder / copy_list.table_file}"
                )
        return EXIT_OK

    stamp = started.strftime("%H%M%S") + "Z"
    log_dir = root / "_logs" / "dated-copies" / day
    log_path = log_dir / f"run-{stamp}.log"
    records = {name: folder / f"_copy-{stamp}.json" for name, folder in folders.items()}
    taken = [path for path in (log_path, *records.values()) if os.path.lexists(path)]
    if taken:
        print(
            f"a run at {stamp} today already wrote {taken[0].name}, and this run never writes "
            "over a file: nothing fetched, nothing written. Run it again.",
            file=sys.stderr,
        )
        return EXIT_CLASH

    allowed = frozenset({log_dir, *folders.values()})
    log_dir.mkdir(parents=True, exist_ok=True)
    incomplete = False
    with open(log_path, "x", encoding="utf-8", newline="\n") as log:

        def emit(row: dict[str, Any]) -> None:
            line = _line(row)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        for copy_list in wanted:
            folder = folders[copy_list.name]
            folder.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, Any]] = []
            finished = False
            try:
                for source in copy_list.sources:
                    row, response = copy_address(source, folder, fetch, now, allowed)
                    rows.append(row)
                    if source.filename != copy_list.table_from:
                        emit(row)
                        continue
                    table_row = build_table(copy_list, folder, response, allowed)
                    rows.append(table_row)
                    if table_row["result"] == "not written" and "sha256" in row:
                        # The page is on disk and no table came of it: it looks wrong.
                        before = row.get("check") or "saved by an earlier run"
                        row.update(
                            looks_right="no",
                            check=f"{before}; {table_row['reason']}; "
                            f"{copy_list.table_file} not written",
                            result="looks wrong",
                        )
                    emit(row)
                    if "sha256" in table_row:
                        emit(table_row)
                finished = True
            finally:
                # Written even when the run is stopped part-way, so that every file
                # saved so far has its record.
                record = {
                    "script": "scripts/copy_dated_sources.py",
                    "rule": 'DECISIONS.md D29, "Amended on 2026-11-17 (ruling 8)"',
                    "list": copy_list.name,
                    "copy_date": day,
                    "run_utc": _iso(started),
                    "user_agent": USER_AGENT,
                    "finished": finished,
                    "files": rows,
                }
                text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
                _create(records[copy_list.name], text.encode("utf-8", "replace"), allowed)
            if any(row["result"] in ("refused", "failed", "looks wrong") for row in rows):
                incomplete = True
    return EXIT_INCOMPLETE if incomplete else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
