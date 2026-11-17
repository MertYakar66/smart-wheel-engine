---
id: dated-copies-script
title: The weekly dated copies script, its tests and its page
kind: feature
status: in-flight
terminal: remote-sandbox
pr:
decisions: [D29]
date: 2026-11-17
headline: scripts/copy_dated_sources.py takes the weekly dated copy that D29's amendment of 2026-11-17 allows; its tests use a fake fetch, no network and no data; no copy has been taken yet
surface: [scripts/copy_dated_sources.py, tests/test_copy_dated_sources.py, docs/DATED_COPIES.md, FILE_MANIFEST.md, TESTING.md, CHANGELOG.md]
---

## Goal

The Operator wants a dated copy, taken once a week, of the official economic
release schedules and of the S&P 500 members list, without touching any file
the engine or its scripts read (`DECISIONS.md` D29, "Amended on 2026-11-17
(ruling 8)"). This run is the third step of the plan the Operator approved on
2026-11-17: it builds the script that takes the copy, its tests and the pen's
page. It takes no copy itself. It fetched none of the sources and opened no
data; it ran in a cloud session, which holds no data root.

## What we tried

One approach, the one the Execution Prompt specified.

- **Standard library only.** `pyproject.toml` lists neither `requests` nor an
  HTML parser, and the run may add no dependency. The script uses `urllib`,
  `html.parser` and `csv`.
- **One function is one HTTP request.** `fetch_once` sends one GET. `urllib`
  follows redirects by itself, so its redirect handler is replaced by one that
  refuses, and each redirect comes back to the script as an answer. The script
  then decides: https only, the same host or its `www` form, at most three.
- **Create-only files.** Every file is opened with mode `xb` (the log with
  `x`). The bytes are written as served, and an existing file cannot be opened
  for writing even by mistake.
- **A fake in place of the network.** The tests pass `main` a fake fetch
  function and a fixed clock. `SWE_DATA_ROOT` is removed before each test and
  set only to a temporary folder, and a socket that tries to connect fails the
  test.

## What worked

- The real request function is tested inside `urllib`'s own machinery without a
  socket: a canned HTTPS handler answers in place of the network. That proves
  one GET goes out with our User-Agent and a 30-second timeout, that `urllib`
  does not follow a 301, 302, 303, 307 or 308 by itself, and that an error
  status comes back as an answer and not as an exception.
- "It opens no other file under the data root" is proved with Python's audit
  events. A child process runs the script twice on a temporary root and records
  every `open`, `mkdir`, rename and removal. The second run finds every copy
  already saved, so it also covers the reads.
- The page `docs/DATED_COPIES.md` was taken byte for byte from the prompt as
  delivered. Its sha256 matches the one the pen gave.

## What didn't

- **The first tests missed one mutation.** The script was broken on purpose in
  27 ways to see whether the tests notice. 26 were caught. Opening files with
  `wb` in place of `xb` was not: the "already saved" check comes first, so the
  mode never mattered in a whole run. A direct test of the create-only write
  now pins it.
- **Loading the script by path breaks dataclasses with postponed annotations.**
  `from __future__ import annotations` makes `dataclasses` look the module up
  in `sys.modules`, and a module loaded with `spec_from_file_location` is not
  there unless the loader puts it there. The script does not use that import,
  and the test registers the module before running it.

## How we fixed it

`scripts/copy_dated_sources.py`:

- `--lists` (one or both of `schedules`, `members`) and `--date` are required.
  The root is `SWE_DATA_ROOT`, read directly. Unset or blank: one line, exit 3.
  A date that is not today's in UTC: one line, exit 5. Both come before
  anything is fetched or written.
- `schedules` saves eight files under
  `data_raw/macro_schedule/date=<copy date>/`. `members` saves
  `members_page.html` and `members.csv` under
  `data_raw/sector_dated/date=<copy date>/`.
- Each run writes `_copy-<HHMMSS>Z.json` beside the copies and
  `_logs/dated-copies/<copy date>/run-<HHMMSS>Z.log`. Response headers go only
  into the record.
- Exit 0 when everything is saved and looks right, 10 when an address was
  refused, failed or looks wrong.

Points the prompt left open, and what the script does. Each is for the pen to
accept or re-cut:

1. **A root that is set but is no existing folder** exits 3, like an unset one.
   The script does not create a data root from a mistyped path.
2. **Only a 200 answer is saved.** Any other final status is "refused".
3. **Two runs in the same second** would need the same record and log names.
   The second refuses with exit 1 before it fetches or writes anything.
4. **An already saved file** is reported with the size and sha256 of the file
   found, so a later run's lines can be compared with the first run's.
5. **A missing `members.csv` beside a page saved earlier that day** is built
   from the saved page, read as UTF-8. If that page holds no such table, the
   run exits 10 again.
6. **The heading row is the table's first row.** The text of `<script>` and
   `<style>` is not cell text. A `<br>` adds no space: tags are removed and
   nothing is put in their place.
7. **The `.ics` check allows a byte-order mark or blank space** before
   `BEGIN:VCALENDAR`.
8. **A redirect that is not followed** is reported without its target: the
   target is a response header, and those go only into the record.
9. **`--dry-run` still needs the root and today's date**, since it prints the
   paths a real run would use.

## Evidence

All on 2026-11-17, in the cloud session, with `SWE_DATA_ROOT` unset and no
`data/bloomberg/sp500_ohlcv.csv`.

- `sha256sum docs/DATED_COPIES.md`:
  `dac627c6760ebbfaabc38e66b0e5a4b4e719e85490f89b83df35d68469b039ae`, 14,079
  bytes, 181 lines.
- `python -m pytest tests/test_copy_dated_sources.py -q`: `104 passed` on
  Python 3.11, 3.12 and 3.13.
- `ruff check` and `ruff format --check` on `engine/ data/ tests/ scripts/
  utils/ dashboard/ backtests/regression/`: `All checks passed!` and `372 files
  already formatted` (ruff 0.17.5).
- `python scripts/copy_dated_sources.py --lists schedules --date 2000-01-01`
  with `SWE_DATA_ROOT` unset: one line, exit 3.
- `python scripts/check_manifest_coverage.py`, `python
  scripts/gen_worklog_index.py --check`, `python
  scripts/check_working_structure.py` and `python
  scripts/check_doc_currency.py`: all pass.
- The mutation check: 27 deliberate breakages of the script, one at a time, on
  a scratch copy outside the repository. The tests fail on all 27.

## Unresolved / handoff

- **Nothing here has met a real site.** The addresses for BEA's ICS and JSON
  files and the five marker words are the pen's, unverified until the first
  copy. A wrong marker only flags a file. A site may also refuse a client that
  is not a browser; the script then reports the status and saves nothing for
  that address.
- **A body served compressed** would be saved as served and would fail its
  light check. `urllib` asks for no compression, so this needs a server that
  ignores the request.
- **The first copy** is the desktop run in `docs/DATED_COPIES.md`. It runs this
  test file on Windows first, with `SWE_DATA_ROOT` unset for that one command.
- The pen sets this fragment's `status` and `pr` at the merge.
