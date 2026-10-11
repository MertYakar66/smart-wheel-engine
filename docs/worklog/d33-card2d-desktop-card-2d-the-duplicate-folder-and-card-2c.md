---
id: d33-card2d
title: Desktop card 2d: the duplicate folder, and card 2c's check and restore tests (D33)
kind: verification
status: merged
terminal: desktop
pr: 556
decisions: [D33]
date: 2026-10-09
headline: The duplicate Drive folder is cleared and swe-data/ is proven: 147,811 files the same both ways by MD5, 147,811 files and 18,644,236,641 B by count with no filter, and both restore tests pass — so D31 step 6's evidence now exists.
surface: []
---

the duplicate cleared; swe-data on Drive, checked both ways; restore tests passed

## Goal

Finish D33's second copy. Card 2c (#554) uploaded the 147,811 files of its baseline into one
Drive folder, `swe-data/` (the desktop root but its own `_logs/` and the one credential-shaped
file, both of which the tool leaves out), and then stopped at its own check: Drive held two folders of one name under
`data_archive/drive-legacy/SmartWheelData/data_processed/theta/option_history/ticker=FDX/`, both
named `expiration=20201113`, so rclone ignored one, could not reach the single `data.parquet`
behind it, and counted it missing. Clearing a duplicate needs a deletion on Drive, which card 2c
forbids outright.

This card clears that duplicate with rclone's own repair for duplicate folders, and then runs card
2c's check, count and restore tests to the end. Those last two are what D33 point 7 and D31 step 6
have been waiting for: the four data branches may be deleted only once the full-history bundle
restores from Drive's copy into an empty repository. The deletion itself is not in this card.

## What we tried

In order, each step writing its own file in `%TEMP%`:

- **Before anything on Drive changed.** `rclone about` read the sign-in as good (`about exit 0`).
  The end check compared the root with card 2c's baseline by name, size and bytes and read
  `then 147812 files; now 147812; gone 0, size changed 0, bytes changed 0 of 147811 hashed
  (credential-shaped names by size only), added 0`, with the steps `plan DONE; upload DONE; check
  not done; restore not done` and the manifest `checked 144 manifest files: 144 ok, 0 missing,
  0 mismatched`. So nothing had written to the root since card 2c, and the Dashboard's `update`
  had not run.
- **The duplicate, read only.** Four reads of Drive, one by path under `swe-data/`'s id and three
  by folder id, found `ticker=FDX` holding 515 objects with exactly one name twice; the folder made
  first empty, trashed objects included; the folder
  made 95 seconds later holding `data.parquet`, 147,522 B — exactly as the pen had recorded it on
  2026-10-07. `ticker=FDX`'s object ids were pinned by a sha256 so step 6 could prove Drive had
  not moved under it.
- **The dry run.** rclone's repair with `--dry-run` showed one change and nothing else:
  `NOTICE: expiration=20201113: Skipped merge duplicate directories as --dry-run is set (size 0)`,
  exit 0.
- **G1, the gate.** The Operator was shown that line and asked whether to repair and run the
  checks. They answered `yes`, recorded in `swe-card2d-g1.txt`, which reads `yes` on two lines.
- **The repair**, then **the run**: the driver's `run` mode, which skipped the plan and the upload
  as DONE and worked the check and the restore tests. It took two runs and five check attempts, `a2` to
  `a6` (`a1` is card 2c's); see "What didn't".

## What worked

**The repair did exactly what the gate promised.** It re-read the folders first, ran the dry run
again seconds before acting, and merged:

```
expiration=20201113: Merging contents of duplicate directories
expiration=20201113: removing empty directory
```

Then it re-read `ticker=FDX` by id and checked the result: the folder holding `data.parquet`
(147,522 B) remains, the empty one is in Drive's trash, and `ticker=FDX` holds **514 objects, each
name once** — one fewer than step 4's 515, and the remaining ids are step 4's set less the trashed
folder. Nothing else on Drive was written, moved or deleted.

**The run then closed all three clauses card 2c could not.** From the driver's own lines:

| What | The driver's line |
| ---- | ----------------- |
| The check, both ways, by checksum | `== check, both ways, by checksum (md5): 147811 the same; 0 only in the root; 0 only in swe-data/; 0 differ; 0 could not be checked` |
| The count, with no filter | `== swe-data/, with no filter: 147811 files, 18,644,236,641 B: the baseline's count and bytes` |
| The manifest's 144 files, from Drive | `== the manifest's 144 files, restored from swe-data/ into an empty folder: 144 ok` |
| The bundle, from Drive | `== the bundle, restored from swe-data/: 1,698,024,795 B, sha256 ee15dd9cbd0132fa4195947893f9a3531da50a4df69749bcbe8ab026caf8fa84: card 1's` |
| The bundle into an empty repository | `== the bundle restores into an empty repository: fsck clean, and the four tips present` |

The git lines under the last of those read `fsck exit 0`, `config lines that would tie it to
another repository: 0`, and `ok` for each of the four data-branch tips:
`68a48b245cea285d98f86ab7e6ba1b5cdca7002d` (`deep-history/bloomberg-raw`),
`2abf850d76b3de9a1e04301d333485e64061f3cd` (`claude/daybot-bloomberg-pull`),
`597cc6af6e2b579d667ca24dc2b324cd766a2a58` (`backup/drive-tier-c-2026-07-22`) and
`24835719ffa6d83a2c5e1dce4a7605b356695ede` (`data/drive-migration`).

The run ended `== END run` with `run exit 0` and all four steps DONE.

## What didn't

Three failures, none of them the data's, and all worth the next agent's time:

1. **A name-lookup failure cost one attempt of 986.4 minutes, `a2`.** Thirty-eight minutes into
   the first check attempt the desktop began failing, intermittently, to resolve Google's
   hostnames, and failures were still being seen the next morning. That attempt ran **986.4 min**
   and ended `exit 1`, having lost **38,780 folder
   listings**; rclone reports every file under a folder it cannot list as one-sided, so its
   combined report held 41,773 "only in the root" lines that were not real differences. The driver
   caught this correctly: it tests for unlisted folders *before* it looks at differences, and
   marks that stop retryable (`UNLISTED` in the driver, citing rclone's `fs/march/march.go`).
   The Executor read the System event log for this outage (its queries were not recorded) across
   2026-10-07 18:30Z to 2026-10-08 08:00Z, and reported no DHCP, DNS-client, Tcpip or
   Ethernet-driver entry. That window misses the outage's onset: rclone stamps its log in local
   time (UTC+3), so the first failure, stamped `2026/10/07 18:42:17`, was at 15:42:17 UTC. The
   addendum below summarises the queries for 15:30Z to 07:45Z; the queries and their output are
   on PR #565 (comment 6109160781).
2. **Two attempts were spent on a program that could not finish starting.** The next two attempts
   each ended `rclone about exited 3221225794` — `0xC0000142`, a process that started and then
   failed while loading its libraries — five minutes apart, about ten minutes in total. These were
   the first programs the driver started after the Executor's session had ended, and the pen's
   inference, not proven, is that the driver's console went away with that session. That used the
   run's third attempt and wrote `check/RUN-STOP-1.txt`, so one further three-attempt failure of
   the check would have been final.
3. **A single eighty-second outage then failed the count, in `a5`.** In the second run, the first
   attempt's check passed cleanly (`exit 0 (220.1 min)`) but its count ran 219.6 min and ended
   `Failed to size with 900 errors`, all of them name-lookup failures packed into 15:10:06Z to
   15:11:27Z. Because the driver restarts a failed step from its beginning, that threw away a
   good four-hour check as well. For this burst the Executor read the System event log (its
   queries were not recorded): no DHCP, DNS-client, Tcpip or Ethernet-driver entry from 15:08:00Z
   to 15:14:00Z, nor from 14:50Z to 15:30Z. The only two entries from 15:08 to 15:14 were
   informational, provider `mtkwecx` (a MediaTek wireless component), id 7501 "Country Info", at
   15:09:10Z and 15:13:39Z, one on each side of the burst. What failed was name resolution: every
   error reads `no such host`. Why is not established; the addendum below summarises the queries,
   and PR #565 (comment 6109160781) holds them with their output.

The lesson: **in rclone v1.74.4 a failed name lookup is never retried,
while a failed connection is retried about 100 times.** The pen traced the error chain to Go's
`*net.DNSError`, which is neither a timeout nor temporary, so rclone's Windows retry list never
matches it.

Two changes were made, and the record should not credit them more than the evidence allows. The
Operator set the desktop's DNS by hand to 1.1.1.1 and 8.8.8.8 at about 2026-10-08T08:30Z, and
`www.googleapis.com` and `oauth2.googleapis.com` were pinned in the desktop's hosts file at about
2026-10-08T18:00Z (the file's own mtime is 2026-10-08T18:02:52Z). **The pin landed inside a
running attempt, not before one.** The driver had already started `a6` on its five-minute pause
after `a5`'s count failed, so `a6`'s check began at 2026-10-08T17:26:59Z — about 36 minutes before
the pin — and ran that part on the hand-set DNS alone. Only `a6`'s count, 21:07:08Z to
00:47:16Z, ran wholly behind the pin. Both of `a6`'s steps ended with **0 name-lookup failures and
0 errors**.

So what the evidence supports is narrower than "the pin fixed it": the hand-set DNS alone was
**not** sufficient, because `a5`'s count failed under it with 900 lookup failures; and after the
pin no lookup failed again in about seven hours of listing. `a5`'s check and `a6`'s first 36
minutes also ran clean without the pin, which shows the fault was intermittent. The pin is
consistent with being the fix and is the only change that covers every clean minute after it, but
this run cannot prove it was the cause.

One other thing cost time and is worth knowing: the card's longer blocks would not survive the
shell tool's command-line transport — step 6's block, a long Python heredoc, came back as
`unexpected EOF while looking for matching '`, having run nothing. The Executor wrote the block to a file and ran
`bash` on it, after `bash -n`; steps 7, 8 and 10, the Resuming state block and step 7's rule took the
same route. Which files those were, and how each compares with the card's block, is in the
addendum below.

## How we fixed it

The repair is rclone's own `dedupe`, run on `ticker=FDX/` alone — given to rclone by that folder's
Drive id as the root, so it can see nothing outside it — with `--fast-list` so rclone lists each
folder by id rather than by a path that reaches only one of two folders of one name:

```
rclone dedupe --dedupe-mode skip --drive-root-folder-id <ticker=FDX's Drive id> gdrive: --max-depth 2 --fast-list --max-delete 0 --drive-use-trash=true -v
```

`--max-delete 0` refuses any file deletion outright, and `--drive-use-trash=true` sends whatever
is removed to Drive's trash, where it stays recoverable for 30 days. The dry run was the same
command with `--dry-run` added, and step 6 ran it again seconds before the repair to show that
nothing had changed while the gate waited.

The check and the count, as printed in step 7's file (`$ROOT` and `$WORK` are the shell's own
names for the data root and the run folder):

```
rclone check $ROOT gdrive: --drive-root-folder-id <swe-data/'s Drive id> --checksum --exclude-from $WORK\plan\a1\filters.txt --ignore-case --disable ListR --combined $WORK\check\a6\combined.txt --low-level-retries 100 -v --stats 10m --stats-one-line --stats-log-level NOTICE
rclone size gdrive: --drive-root-folder-id <swe-data/'s Drive id> --disable ListR --low-level-retries 100 --json
```

## Evidence

**The timings**, each from its own log's last line under the run folder:

| Log | Outcome |
| --- | ------- |
| `check/a2/06-check.txt` | `exit 1 (986.4 min)` — the name-lookup failures |
| `check/a5/06-check.txt` | `exit 0 (220.1 min)` |
| `check/a5/07-size.txt` | `exit 1 (219.6 min)` — 900 name-lookup failures |
| `check/a6/06-check.txt` | `exit 0 (220.2 min)` |
| `check/a6/07-size.txt` | `exit 0 (220.1 min)` |
| `restore/a1/08-files.txt` | `exit 0 (3.9 min)` — the manifest's 144 files |
| `restore/a1/10-bundle.txt` | `exit 0 (4.9 min)` — the bundle |

**The attempts.** The check took six attempt folders in all: `a1` is card 2c's; `a2`, `a3` and
`a4` are this card's first run, which ended
`STOPPED: check: 3 attempts this run; the last: rclone about exited 3221225794 … (a rerun may clear
this)` and left `check/RUN-STOP-1.txt`; `a5` and `a6` are the second run, and `a6` is the one that
is DONE. The restore step took one attempt, `a1`, and passed on it. The three `RETRY:` lines are in
step 7's output on PR #556: `a2`'s and `a3`'s in part 1 (comment 6072529006), and `a5`'s in part 13
(comment 6072532780). `a4` has none: it ended the run with the `STOPPED:` line above.

**The end state**, after the run: `then 147812 files; now 147812; gone 0, size changed 0, bytes
changed 0 of 147811 hashed (credential-shaped names by size only), added 0`; the steps
`plan DONE; upload DONE; check DONE; restore DONE`; `== endstate: no finding`; and the verdict
`== the end state: no finding`. **No finding of any kind**, and the manifest check read `checked
144 manifest files: 144 ok, 0 missing, 0 mismatched` by full hash. The end check also wrote the
fingerprints of the four attempts that had been left without them (`check/a2` to `check/a5`);
those are not findings.

**Where the record lives.** Outside git, in the data root's `_logs\d33-card2c\` — card 2c's run
folder, because the driver keeps its state there and this card ran the same driver. This card's
driver run log is `driver-run-20261008T100110Z-f986.txt`, and the first run's is
`driver-run-20261007T150429Z-2405.txt`. The attempts' fingerprint files, by sha256:

| Pin | sha256 |
| --- | ------ |
| `check/a1/FINGERPRINTS.txt` | `24189f023a0b4280ce26f293e2971e2addf19ab954d870c1470242f98f29a277` |
| `check/a2/FINGERPRINTS.txt` | `9d33c8c908f182f3b3e1ee90f2dfc9e85b79b7316ce9b910160fc861965d514b` |
| `check/a3/FINGERPRINTS.txt` | `c1ae346c5f3959fd88f369beb7f567e27589b55711756f5f25633c2493870d1f` |
| `check/a4/FINGERPRINTS.txt` | `b1730a0134306bcd68db9cc857371ac232d29588626ece916ca914aee251d25a` |
| `check/a5/FINGERPRINTS.txt` | `cc5f640a104c54555c0b9bc4690fc7637bbbc0627dd966ee69b6ef28608355f1` |
| `check/a6/FINGERPRINTS.txt` | `ba38b1a7f9513366e19c65c15e641f3ce55c8866cfe01e83e1c256f005246402` |
| `restore/a1/FINGERPRINTS.txt` | `de6e679d6281046ddaeef37e4f5a0fcec0a3b9a4fad15f058ab35a29798faa4f` |

The restore tests' copies are left in place outside the root, under this card's scratch folder in
`%TEMP%`: the end check reads `$SCR holds 175 files, 5,140,134,098 B: the restore tests' copies,
outside the root (left in place): a1`. Nothing was deleted anywhere but the one empty
`expiration=20201113` folder under `ticker=FDX`, which G1 authorised and which went to Drive's
trash, recoverable for 30 days.

**`SHA256SUMS`.** Card 2c's list is still the root's, unchanged, its own sha256
`e22d8579bed1ce7a81a90ccb1096d0397a29de8471f8b56f466f05f13f882989` — the fingerprint D33 asks to
be recorded in git. This card added nothing to the root and changed nothing in it.

**The notes posted on the campaign issue, #544:**

| Note | Link |
| ---- | ---- |
| REGISTER | [#544 comment 6040706812](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6040706812) |
| WAITING | [#544 comment 6055474793](https://github.com/MertYakar66/smart-wheel-engine/issues/544#issuecomment-6055474793) |

## Unresolved / handoff

1. **The four data branches are now deletable on the evidence, but not by this card.** D31 step 6
   and D33 point 7 asked for the bundle to restore from Drive's copy into an empty repository with
   `git fsck --full` clean and the four exact commits present. That is now proven. The deletion
   needs the Operator's yes in a later Execution Prompt, in one atomic push with a lease on each.
2. **The hosts-file pin must come off.** `www.googleapis.com` and `oauth2.googleapis.com` are
   pinned to fixed addresses in the desktop's hosts file, with `hosts.bak-card2d` beside it. Left
   in place, stale addresses would break Google API clients on that PC weeks from now. The pen's
   removal command restores the backup; the close records it. Note that `oauth2.googleapis.com` is
   pinned to a single address where `www.googleapis.com` has eight.
3. **The desktop's DNS is now set by hand** to 1.1.1.1 and 8.8.8.8, where it was `10.0.0.50` by
   DHCP. Whether that stays is the Operator's call. The underlying finding is durable and matters
   for every future Drive card: a long listing run needs name resolution that does not blink,
   because rclone will not retry a failed lookup.
4. **Why `0xC0000142` happened is inferred, not proven.** If a driver is ever again started from a
   session that then ends, expect it; the pen's amendment 1 now has the run started in its own
   window instead, with a tripwire that stops the driver inside its five-minute pause when a
   `RETRY:` line names a ten-digit exit, so such a failure cannot spend three attempts in ten
   minutes.
5. **`docs/DATA_INVENTORY.md` §C.3** is the pen's to write at the close, from this card's and card
   2c's records: the folder, its id, `SHA256SUMS`' own sha256, the rclone commands, and the manual
   refresh routine D33 asks for.
6. **The restore tests' copies** (175 files, 5,140,134,098 B) sit in `%TEMP%`. This card was
   forbidden to delete them. Whoever cleans them up should note they are outside the data root and
   the engine never reads them.
7. **Still open from before this card:** the Operator deletes the credential file forever from
   Drive's trash and rotates the IBKR Flex token, and decides whether the desktop's `stash@{0}`
   (the stray paste) is dropped.

## Addendum (2026-10-11): the evidence

Card 2d's addendum (#544 comment 6088910896) put on GitHub what this card's Run Summary
lacked. Its step files are pasted whole on its pull request, #565, from branch
`claude/d33-card2d-addendum`. The pen corrected this section's wording at the close after #565,
from four fresh-context checks of the step files (marked *pen* below).

- **The time-limit line.** The addendum session's own line, from its Bash tool's description: one
  timeout, in milliseconds, default 120000 and maximum 600000, that is ten minutes, and for a
  command run in the background no time limit at all — only that the command keeps running across
  turns and re-invokes the session when it exits. Card 2d's line was asked for in chat alone (step
  0, and amendment 1 item 1), and no comment on #544 or #556 states it; the addendum searched no
  file for it and opened no transcript, so card 2d's line is not recoverable from the record.
- **The tripwire.** Exactly one tripwire line survives, `== the run has ended; the tripwire stands
  down`, in the background output file created 2026-10-08T10:01:51Z and last written
  2026-10-09T00:56:48Z (69 B). Ten background output files fall in the span and that file is the
  only one of them holding a tripwire line, so the files show one arming. *Pen:* that is
  consistent with the one this card reports (10:03Z, driver pid 19560), but no step file ties the
  file to that pid, and only its one line is shown, not all 69 B. Each arming was one background command, hence one file, and an arming that printed
  nothing left a file these lines cannot tell from any other command's, so the count is a floor and
  not a proof that there was only one.
- **`swe-card2d-run7b.sh` and its log.** The script is 2,501 B, sha256
  `88cf7410159dab61ebfa09ee5e8ddef6d5e9e6a1600b746541a88a6457ea73e9` — this card's figure to the
  character; its first line is the card's definitions line, and its step 2 and step 7 lines are the
  card's blocks byte for byte, the file letter aside. The log is 652,840 B, sha256 `85423840…`, and
  after its first 343 bytes it is `swe-card2d-step7b.txt` (652,497 B, sha256 `144b2645…`) byte for
  byte. Its only other content is the seven lines before that: three tool paths, the interpreter it
  resolved, `about exit 0`, `== the Drive sign-in works` and `== END step 2`; nothing follows the
  step 7 file.
- **The blocks run from files.** Of the six blocks the pen asked about, four are byte for byte,
  with the definitions line before them: block 9, step 6, as `card2d-step6.sh` (sha256
  `60a780df…`); block 10, step 7, as `card2d-step7.sh` (`22d3b855…`), and again as
  `swe-card2d-run7b.sh`'s step 7 line, the file letter aside; block 12, step 8, as
  `card2d-step8.sh` (`2ad508da…`); and block 18, the Resuming block, as `card2d-resume.sh`
  (`ef10d809…`). Two differ, and only by what was added to run them: block 11, step 7's rule, as
  `card2d-step7-rule.sh` (`5c3c9a86…`, similarity 0.895), by a shortened definitions line and a closing
  stanza that prints the restart time two hours after the latest step 7 file; and block 14, step
  10, as `card2d-step10.sh` (`ba46b556…`, similarity 0.757), by a shortened definitions line and
  four of the card's trailing comments dropped. *Pen:* the shortened line is the card's without
  its six Drive-id and path assignments (`SWE`, `FDX`, `DA`, `DB`, `DF`, `FDXP`); the kit scored
  both files closer to their blocks without it. An eighth file, `card2d-reference.txt` (`9f7c088d…`, similarity
  0.857 against block 10), is not a script file, so the comparison reports only that it differs and
  its text is not shown.
- **The event logs.** The System log is enabled, 45,676 records, oldest event 2026-07-20T02:11:28Z,
  and `Microsoft-Windows-WLAN-AutoConfig/Operational` is enabled, 1,390 records, oldest
  2026-08-22T17:57:39Z, so both reach back well past both windows. For 2026-10-07T15:30:00Z to
  2026-10-08T07:45:00Z — the window that does cover the first outage's 15:42:17Z onset (stamped
  `2026/10/07 18:42:17` in local time in #556's step 7 output) — the System
  log holds 399 events, 367 of them informational `mtkwecx` id 7501, and five above information: a
  DistributedCOM 10016 warning, a Time-Service id 134 warning at 19:39:12Z that NtpClient could not
  set a manual peer **because of a DNS resolution error**, a Time-Service id 36 warning at
  06:48:06Z that the clock had gone 56,506 seconds without synchronising, and two WindowsUpdateClient
  id 20 installation errors; for `a5`'s burst, 15:08:00Z to 15:14:00Z, it holds two events in all,
  both informational `mtkwecx` id 7501, at 15:09:10Z and 15:13:39Z. WLAN-AutoConfig holds **0
  events** in either window: no association, roam or disconnect was logged while the lookups were
  failing. *Pen:* that rules out only a logged change of the Wi-Fi connection's state, not a Wi-Fi
  fault that changes none; the Wi-Fi driver's own 369 System-log events (`mtkwecx` id 7501) were
  counted, not read, because the kit prints a fixed label for them. The cause stays open. A second
  name-resolution failure, from a client other than rclone, falls in the same window. Each query
  and its output are in step 3's file on #565 (comment 6109160781).
- **The corrections.** The evaluation's (a) to (g) went in as nine replacements, `9 corrections
  applied … (the evaluation's corrections; 0 already present)` with `fix exit 0`, and A1 to A9 as
  eight more after the Operator's yes to G1, `8 corrections applied … (the evaluation's corrections
  and A1 to A9; 9 already present)`, also `fix exit 0`. Each replacement was checked as pending or
  already done before anything was written.
- **Since this card** (the pen's record, `PROJECT_STATE.md` §0 B): the four data branches were
  deleted (D31 step 6, #559) and the hosts pin was removed (2026-10-09), so handoff items 1 and 2
  are done.
