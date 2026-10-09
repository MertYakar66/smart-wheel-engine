---
id: d33-card2d
title: Desktop card 2d: the duplicate folder, and card 2c's check and restore tests (D33)
kind: verification
status: in-flight
terminal: desktop
pr:
decisions: [D33]
date: 2026-10-09
headline: The duplicate Drive folder is cleared and swe-data/ is proven: 147,811 files the same both ways by MD5, 147,811 files and 18,644,236,641 B by count with no filter, and both restore tests pass — so D31 step 6's evidence now exists.
surface: []
---

the duplicate cleared; swe-data on Drive, checked both ways; restore tests passed

## Goal

Finish D33's second copy. Card 2c (#554) uploaded the whole desktop root into one Drive folder,
`swe-data/`, and then stopped at its own check: Drive held two folders of one name under
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
- **The duplicate, read only.** Four reads of Drive by id found `ticker=FDX` holding 515 objects
  with exactly one name twice; the folder made first empty, trashed objects included; the folder
  made 95 seconds later holding `data.parquet`, 147,522 B — exactly as the pen had recorded it on
  2026-10-07. `ticker=FDX`'s object ids were pinned by a sha256 so step 6 could prove Drive had
  not moved under it.
- **The dry run.** rclone's repair with `--dry-run` showed one change and nothing else:
  `NOTICE: expiration=20201113: Skipped merge duplicate directories as --dry-run is set (size 0)`,
  exit 0.
- **G1, the gate.** The Operator was shown that line and asked whether to repair and run the
  checks. They answered `yes`, recorded word for word in `swe-card2d-g1.txt`.
- **The repair**, then **the run**: the driver's `run` mode, which skipped the plan and the upload
  as DONE and worked the check and the restore tests. It took two runs and six check attempts; see
  "What didn't".

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

1. **A name-lookup collapse cost one whole run.** Thirty-eight minutes into the first check
   attempt the desktop stopped resolving Google's hostnames, and kept failing for about fifteen
   hours. That attempt ran **986.4 min** and ended `exit 1`, having lost **38,780 folder
   listings**; rclone reports every file under a folder it cannot list as one-sided, so its
   combined report held 41,773 "only in the root" lines that were not real differences. The driver
   caught this correctly: it tests for unlisted folders *before* it looks at differences, and
   marks that stop retryable (`UNLISTED` in the driver, citing rclone's `fs/march/march.go`).
   The System event log holds **no DHCP, DNS-client, Tcpip or Ethernet-driver entry** anywhere in
   that window; the only entries in it are two informational ones from a MediaTek wireless
   component, and the wireless adapter was down while Ethernet carried the traffic. So nothing
   recorded a link drop, a lease change or a resolver-service fault: the failure was in name
   resolution alone.
2. **Two attempts were spent on a program that could not finish starting.** The next two attempts
   each ended `rclone about exited 3221225794` — `0xC0000142`, a process that started and then
   failed while loading its libraries — five minutes apart, about ten minutes in total. These were
   the first programs the driver started after the Executor's session had ended, and the pen's
   inference, not proven, is that the driver's console went away with that session. That used the
   run's third attempt and wrote `check/RUN-STOP-1.txt`, so one further three-attempt failure of
   the check would have been final.
3. **A single eighty-second outage then failed a second count.** In the second run, the first
   attempt's check passed cleanly (`exit 0 (220.1 min)`) but its count ran 219.6 min and ended
   `Failed to size with 900 errors`, all of them name-lookup failures packed into 15:10:06Z to
   15:11:27Z. Because the driver restarts a failed step from its beginning, that threw away a
   good four-hour check as well.

The lesson, and the fix that worked: **in rclone v1.74.4 a failed name lookup is never retried,
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
`unexpected EOF while looking for matching '`, having run nothing. Writing the identical block to a file and running
`bash` on it, after `bash -n`, worked every time.

## How we fixed it

The repair is rclone's own `dedupe`, run on `ticker=FDX/` alone — given to rclone by that folder's
Drive id as the root, so it can see nothing outside it — with `--fast-list` so rclone lists each
folder by id rather than by a path that reaches only one of two folders of one name:

```
rclone dedupe --dedupe-mode skip --drive-root-folder-id <the folder's Drive id> gdrive: --max-depth 2 --fast-list --max-delete 0 --drive-use-trash=true -v
```

`--max-delete 0` refuses any file deletion outright, and `--drive-use-trash=true` sends whatever
is removed to Drive's trash, where it stays recoverable for 30 days. The dry run was the same
command with `--dry-run` added, and step 6 ran it again seconds before the repair so that nothing
changed while the gate waited.

The check and the count, as printed in step 7's file (`$ROOT` and `$WORK` are the shell's own
names for the data root and the run folder):

```
rclone check $ROOT gdrive: --drive-root-folder-id <the folder's Drive id> --checksum --exclude-from $WORK\plan\a1\filters.txt --ignore-case --disable ListR --combined $WORK\check\a6\combined.txt --low-level-retries 100 -v --stats 10m --stats-one-line --stats-log-level NOTICE
rclone size gdrive: --drive-root-folder-id <the folder's Drive id> --disable ListR --low-level-retries 100 --json
```

## Evidence

**The timings**, each from its own log's last line under the run folder:

| Log | Outcome |
| --- | ------- |
| `check/a2/06-check.txt` | `exit 1 (986.4 min)` — the name-lookup collapse |
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
is DONE. The restore step took one attempt, `a1`, and passed on it. The three `RETRY:` lines are
quoted in "What didn't".

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
outside the root (left in place): a1`. Nothing was deleted anywhere.

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
