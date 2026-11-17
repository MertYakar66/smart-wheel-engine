# Weekly dated copies — what they are, and the prompt that runs them

**Status:** in force under `DECISIONS.md` D29, amended 2026-11-17 (ruling 8). That amendment is the rule;
this page is how the job is run. The pen wrote the prompt below; an Executor never edits it. A change to
it is the pen's, checked by a context that did not write it.

## What the job does

Once a week it saves a dated copy, as served, of the official economic release schedules and, once the
Operator has released it, of the S&P 500 members list. It changes no file the engine or its scripts
read. Nothing reads the copies except the run that makes them and the check of that run.

| List | Addresses | Where the copy goes (under the data root) |
| --- | --- | --- |
| Release schedules (`schedules`) | The Federal Reserve's `federalreserve.gov/monetarypolicy/fomccalendars.htm`; BLS's `bls.gov/schedule/news_release/` and its `bls.ics`; BEA's `bea.gov/news/schedule` with its ICS and JSON files; Census's `census.gov/economic-indicators/calendar-listview.html` and `calendar-listview-2026.html` | `data_raw/macro_schedule/date=<copy date>/` |
| Members list (`members`) | The Wikipedia page `en.wikipedia.org/wiki/List_of_S%26P_500_companies`, kept as served and also as a table built from that page | `data_raw/sector_dated/date=<copy date>/` |

The script is `scripts/copy_dated_sources.py`. It copies only the lists it is told to; it has no default
list. It refuses to run when `SWE_DATA_ROOT` is unset, never writes over an existing file, and writes
only inside the two folders above and the run's own folder under `_logs/`.

The copied pages, the table and the `_copy-*.json` files beside them stay under the data root and in
`swe-data/`. They are never pasted or committed anywhere. The public record of a copy is its date and,
for each file, its name, size and sha256.

## The three slots

The prompt below is used word for word each week. Only three slots change:

| Slot | What goes in it | Where the value comes from |
| --- | --- | --- |
| `<DATE>` | The day of the run, `YYYY-MM-DD`, in UTC | The day the Operator says "authorize the weekly copy" |
| `<LISTS>`, with `<RELEASE>` | `<LISTS>` is `schedules`, or `schedules members`. `<RELEASE>` is `not released` until the Operator releases the members list; from then on it is their releasing words, quoted, with their date | D29's amendment and any dated line under it; the Operator's own words |
| `<REHASH>` | `does not run` | D29's amendment. This prompt has no steps for the whole-root re-hash. If a dated line under the amendment says it may run, the pen re-cuts this prompt before that week's run |

The order each week: the Operator signs in to Drive and says "authorize the weekly copy"; the pen fills
the slots; a context that did not write the filled-in prompt checks it; the pen sends it to an Executor
on the desktop.

## The prompt

This run needs the desktop: the data root and rclone exist only there. The commands are written for Git
Bash.

```
<run-mode>       change
<confirmed-request> "authorize the weekly copy" — confirmed <DATE>
<request-as-sharpened>
  Goal:            keep this week's dated copy of the lists the Operator has released, as D29's
                   amendment of 2026-11-17 allows.
  Success:         the copy of each list in <LISTS> is under the data root and in swe-data/, the same
                   both ways by checksum; nothing else under the data root changed; the record pull
                   request is open with the Run Summary on it.
  Tried before:    the earlier weekly copies; their records are in docs/worklog/.
  Must not break:  no frozen or served file is touched; data_raw/sp500_constituents_current.csv keeps
                   its sha256; the manifest's 144 files keep their sizes; nothing on Drive is deleted,
                   moved or synced.
  Scope of run:    one weekly copy, dated <DATE>.
  Operator answers: not warranted, because D29's amendment and its dated lines settle the job.
                   Lists released: <LISTS>. The members list: <RELEASE>. The routine's whole-root
                   re-hash (items 1 and 3): <REHASH>.
  Premises:        scripts/copy_dated_sources.py is on main (checked: the file exists); the Operator
                   signed in to Drive within the last 7 days (unverified: step a settles it); swe-data/
                   is reached by its id, as docs/DATA_INVENTORY.md section C gives it (checked there).
  Approved as:     "The draft of the exception in this chat, and its check. After your two yeses: the
                   records pull request and its merge with CI green. The script's change run in a
                   cloud Executor session, its check, and its merge with CI green. Neither the run
                   nor its check opens any data. One desktop run. Before it asks for your go-ahead it
                   runs the script's own test file, which opens no data. After your go-ahead it tests
                   the Drive sign-in with a read-only call. It fetches the public pages of the lists
                   you have released. It writes the new copy folders (`data_raw/macro_schedule/` and,
                   once released, `data_raw/sector_dated/`) and `_logs/` in the root, and copies them
                   to `swe-data/` through rclone, by copy only. In the rest of the data root it opens
                   one file, `data_raw/sp500_constituents_current.csv`, to take its sha256, and reads
                   the sizes of the manifest's 144 files. That run's record pull request, which
                   publishes the copy date, file names, sizes and sha256s; its check by a fresh
                   context, which reads the pull request, the new copy folders and the run's logs,
                   and nothing else in the data root; and its merge with CI green. Only on your yes
                   to the re-hash: that pass, the rebuilt `SHA256SUMS`, and its replacement in
                   `swe-data/`, as the routine's items 1 and 3 say. The same run each later week,
                   only on your "authorize the weekly copy". The close after each merge. And nothing
                   on the line above." — approved 2026-11-17 (PROJECT_STATE.md section 0 B,
                   Authorized).
<objective>      Take the dated copy of <LISTS> for <DATE>, copy it to swe-data/, check it both ways,
                 and record it in one small pull request. Done means every verification step below
                 shows its expected output, or the run has stopped and said why.
<context>        DECISIONS.md D29, "Amended on 2026-11-17 (ruling 8)", and any dated line under it;
                 docs/DATED_COPIES.md (this prompt's home); docs/DATA_INVENTORY.md section C, the
                 routine that keeps swe-data/ current, and the line after its item 6;
                 docs/deadlines.md, the Drive sign-in row. SWE_DATA_ROOT names the data root on the
                 desktop. rclone's remote is gdrive:. This run needs the desktop, where the data root
                 and rclone are; the commands are written for Git Bash.
<scope>          1. The sign-in test. 2. The two before-checks. 3. The script. 4. The routine's items
                 2, 4, 5 and 6 over the new folders only. 5. The two after-checks. 6. The record pull
                 request.
<out-of-scope>   The routine's items 1 and 3 (the whole-root re-hash and the replacement of
                 SHA256SUMS). Any other file under the data root. scripts/data_manifest.py build.
                 scripts/pull_all.py. Any change to the script or to this prompt. Any list not in
                 <LISTS>. Any source the amendment does not name. Merging the pull request. Posting
                 on any issue.
<owns>           Under the data root: data_raw/macro_schedule/date=<DATE>/,
                 data_raw/sector_dated/date=<DATE>/ (only if <LISTS> holds members), and
                 _logs/dated-copies/<DATE>/. In the repository: the run's worklog fragment,
                 docs/worklog/INDEX.md and one CHANGELOG.md bullet.
<reads>          Under the data root, outside <owns>: data_raw/sp500_constituents_current.csv, to take
                 its sha256; the sizes of the manifest's 144 files. Nothing else there is opened.
<invariants>     Data integrity: frozen and served files are never rewritten; no silent substitution.
                 Process: work on a branch; nobody pushes to main; nothing is deleted. Honesty: every
                 figure in the Run Summary comes from pasted output. Nothing private goes on GitHub.
<gates>          None: nothing here is on the CLAUDE.md section 6 list. The run stops by itself, and
                 says BLOCKED with what is needed, if step a, b or c fails, or if step d exits with
                 3 or 5.
<verification>   Work in a worktree of origin/main on the branch named under <push>. Run the steps in
                 order and paste the output of each, except where a step says what to paste.
                 a. rclone lsf gdrive: --drive-root-folder-id 13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b
                    --max-depth 1 --dirs-only
                    Expected: exit 0 and at least one name. Paste the exit code and the number of
                    names. If it fails, stop before anything is fetched or written: BLOCKED, the
                    Operator signs in again.
                 b. python scripts/data_manifest.py check --root "$SWE_DATA_ROOT" --size-only
                    Expected: "144 ok, 0 missing, 0 mismatched (size only)".
                 c. sha256sum "$SWE_DATA_ROOT/data_raw/sp500_constituents_current.csv"
                    Expected: a sha256; keep it.
                 d. python scripts/copy_dated_sources.py --lists <LISTS> --date <DATE>
                    Expected: one line per file with its name, bytes, sha256 and result. Exit 0:
                    every address was saved and looks right. Exit 10: one or more addresses were
                    refused, failed or look wrong; report each by name and result, and go on with the
                    rest.
                    Exit 3 (no data root) or 5 (the date is not today's UTC date): stop. Any other
                    exit code: stop.
                 e. Make the filter list outside the data root and the repository, and print its
                    path:
                    F="$(mktemp)"; python scripts/drive_consolidate.py filters | grep -vx
                    '/_logs/\*\*' > "$F"; echo "$F"
                    A shell variable does not last from one command to the next, so write that
                    printed path out in full wherever "$F" stands below.
                    Then, for each new folder D (the copy folders first, _logs/dated-copies/<DATE>
                    last, and nothing is written to that log folder after this step):
                    rclone copy "$SWE_DATA_ROOT/D" "gdrive:D" --drive-root-folder-id
                    13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b --immutable --checksum --ignore-checksum
                    --ignore-size --no-update-dir-modtime --drive-use-trash=true --exclude-from "$F"
                    --ignore-case
                    Expected: exit 0 for each.
                 f. For each new folder D:
                    rclone check "$SWE_DATA_ROOT/D" "gdrive:D" --drive-root-folder-id
                    13h0HIy_kLjsN431YJFRewJ8FAZGjxe3b --checksum --disable ListR --exclude-from "$F"
                    --ignore-case
                    Expected: 0 differences, and the matching count equals the number of files in D.
                 g. Repeat b and c. Expected: the same line as b, and the same sha256 as c.
                 h. python scripts/check_working_structure.py; python scripts/gen_worklog_index.py
                    --check; python scripts/check_manifest_coverage.py; python
                    scripts/check_doc_currency.py. Expected: all pass.
<push>           claude/dated-copy-<DATE>, from origin/main. Push before the Run Summary.
<summary>        Write the Run Summary per OPERATING_MODEL.md §4.4 as a comment on the record pull
                 request. The worklog fragment carries the copy date; for each saved file its name,
                 size and sha256; for each address that was refused or failed, or whose file looks
                 wrong, its file name and result; and, if <LISTS> holds members, the line "The members
                 list: <RELEASE>".
<constraints>    Never rclone sync, move, delete, purge or dedupe. Never write over a file, at home or
                 on Drive. Never open or print rclone's configuration or any credential. Open no file
                 under the data root beyond <owns> and <reads>. Never paste or commit a copied page,
                 the table, a _copy-*.json file or any response header: they stay under the data
                 root. Run step d once; no retry. Never merge. Put nothing private on GitHub. If the
                 copy to Drive fails part-way, stop, leave the copy at home as it is, and say plainly
                 that the root is ahead of swe-data/.
```

## The first copy only

The first copy uses the prompt above with these differences, and no others:

1. Line two quotes the Operator's approval of the plan, with its date, in place of "authorize the weekly
   copy": "I approve plan 1 (the weekly dated copies, as an exception before D40) and plan 2 (D40's
   draft, read and draft only), as rewritten." — confirmed 2026-11-17.
2. "Tried before" reads: nothing; this is the first copy.
3. Before step a, the run runs the script's own test file, and nothing else, with `SWE_DATA_ROOT` unset
   for that one command: `env -u SWE_DATA_ROOT python -m pytest tests/test_copy_dated_sources.py -q`.
   Expected: all pass. This shows on Windows that the script keeps bytes unchanged.
4. `<gates>` gains one gate, added to what it already says: G1, one yes/no question to the Operator,
   after that test and before step a, so before anything is fetched or written: "Your Drive sign-in is
   made. Go?"
