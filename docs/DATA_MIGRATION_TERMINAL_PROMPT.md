# TERMINAL PROMPT — migrate Bloomberg data off GitHub → Google Drive

_Paste the block below into a Claude Code terminal running on the operator's
machine (the one with the full repo, the local data, Google Drive credentials,
and the ability to run the engine). It is self-contained for a memoryless
session._

---

You are a Claude Code terminal on the operator's machine for the
**smart-wheel-engine** repo. You have the full repo, the local data under
`data/bloomberg/` and `data_processed/theta/`, Google Drive credentials, and can
run the engine. You have **no memory** of prior sessions — everything you need is
below and in the repo.

## Mission
Finish moving the ~535 MB of Bloomberg data **out of git** so it lives **only on
Google Drive**, without losing anything and without breaking the engine. The
tracking infrastructure is already built and committed on branch
`claude/smart-wheel-engine-overview-0txgye` (at commit `637b9e1` or later).

**PHASE 0 — SYNC FIRST (before anything else).** A stale local checkout will NOT
have the tooling and will make you wrongly conclude it "was never built." Run:
```bash
git fetch origin claude/smart-wheel-engine-overview-0txgye
git checkout claude/smart-wheel-engine-overview-0txgye
git pull --ff-only origin claude/smart-wheel-engine-overview-0txgye
git log --oneline -1          # MUST be 637b9e1 or LATER (e.g. dbc872b) — NOT 991a981 or earlier
ls data/data_manifest.json scripts/fetch_data.py scripts/gen_data_manifest.py   # all must exist
```
These four files provably exist on `origin` (commit `637b9e1`), so if `ls` reports
any missing AFTER this sync, the sync did not run — fix that, do not conclude the
tooling is absent. **All Drive folder ids come from `data/data_manifest.json`
(`drive_root_folder_id` + `drive_folders`) — read them from the file, NEVER
hand-transcribe an id from prose (the root is
`1xpRvaQglsmcUuTKgVKHR39_3H-vbdIFh`, 33 chars).** Then **read these four files:**
- `docs/DATA_INVENTORY.md` — the single what/where doc (§A status, §B fetch, §C locations).
- `data/data_manifest.json` — machine map of all data files (path · size · sha256 · Drive folder · role).
- `scripts/fetch_data.py` — hydrate a checkout from Drive (`--check`, `--served-only`, `--include-deep`).
- `scripts/gen_data_manifest.py` — regenerator.

## HARD RULES — do not violate
1. **Verify before you delete.** Never `git rm` any data file until you have INDEPENDENTLY confirmed Drive holds a sha256-identical copy. Losing data is the one unacceptable outcome.
2. **Branch + PR only. Never push to `main`. Never force-push. Never rewrite history.** (The history purge is a separate, later, operator-run job — NOT this task.)
3. **Do not touch engine logic**, especially the decision-layer trio (`engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`). This is a data/infra task only.
4. **Keep the working-tree data.** Use `git rm --cached` (index-only) so files stay on disk and the engine keeps running locally.
5. If any verification in Phases 1–4 fails, **STOP and report** — do not proceed to the untrack.

## Phase 1 — Drive credentials + fetch smoke test
1. Give the fetch script read access to the Drive folder holding the data
   (root folder id `1xpRvaQglsmcUuTKgVKHR39_3H-vbdIFh`; subfolder ids are in the
   manifest's `drive_folders`). Either:
   `export GOOGLE_APPLICATION_CREDENTIALS=/path/service-account.json` (share the
   folder with the SA email, Viewer), or an OAuth token at
   `~/.config/swe/drive_token.json`. Then `pip install google-api-python-client google-auth`.
2. `python scripts/fetch_data.py --check` → expect all local files OK (you have them).
3. Prove fetch really pulls from Drive: move one served file aside
   (`mv data/bloomberg/treasury_yields.csv /tmp/`), run
   `python scripts/fetch_data.py --served-only`, confirm it re-downloads and
   `--check` passes, then restore if the download differs. Record the result.

## Phase 2 — INDEPENDENT Drive-completeness gate (the safety check)
Re-verify here; do not rely on any earlier check. Confirm that **every**
git-tracked file in the manifest has a Drive copy whose sha256 matches the local
file. Practical approach:
```bash
python scripts/fetch_data.py --check     # local vs manifest sha256 → must be ALL OK
```
Then, for each manifest file, resolve it on Drive (its `drive_folder` + basename),
download to a temp path, and compare sha256 to the manifest. Produce a table:
`file → on Drive? → sha256 match?`. **Any MISSING or MISMATCH → STOP and report;
do not delete anything.**

## Phase 3 — Back up Tier C (local-only, highest risk)
The Theta option corpus (`data_processed/theta/`, ~390M rows) and the feature
store (`data/features/`) are gitignored **and not on the Drive mirror** — they
exist only on this machine and cannot be re-pulled at a Bloomberg terminal.
1. Set up `rclone` for the same Drive account (`rclone config`) — best tool for
   the large upload.
2. Create a Drive folder (e.g. `swe-local-only/`) and:
   `rclone copy data_processed/theta <remote>:swe-local-only/theta --progress`
   (also `data_processed/vol_indices*.parquet`, and `data/features/` if you want it backed up).
3. Record the new Drive folder id(s) and add a Tier-C location line to
   `docs/DATA_INVENTORY.md` §C so agents know where the corpus lives.

## Phase 4 — Refresh the manifest against THIS machine's data
Your local data is the canonical copy. Make sha256 reflect it; if anything is
newer than Drive, re-upload it first.
```bash
python scripts/gen_data_manifest.py      # regenerate data/data_manifest.json from local bytes
python scripts/fetch_data.py --check      # must be ALL OK
```
If regeneration changes any sha256, that file's Drive copy is stale → re-upload it
to its Drive folder, then re-run `--check`.

## Phase 5 — The untrack PR (only after Phases 1–4 all pass)
1. `git fetch origin && git checkout -b data/drive-migration origin/main`
2. Bring the migration scripts + doc onto this branch:
   ```bash
   git checkout claude/smart-wheel-engine-overview-0txgye -- \
     scripts/fetch_data.py scripts/gen_data_manifest.py docs/DATA_INVENTORY.md
   python scripts/gen_data_manifest.py           # generate the manifest fresh on this branch
   ```
3. Append to `.gitignore`:
   ```
   # Bloomberg data — migrated to Google Drive (fetch via scripts/fetch_data.py)
   data/bloomberg/*.csv
   data/bloomberg/*.xlsx
   data/bloomberg/broad_pull/**
   ```
   Keep tracked: `data/bloomberg/EXTRACTION_GUIDE.md`, `data/data_manifest.json`.
4. Untrack (index-only — keeps the files on disk):
   ```bash
   git rm -r --cached data/bloomberg/*.csv data/bloomberg/*.xlsx data/bloomberg/broad_pull
   ```
5. Add a CI data-fetch step so the suite still gets data. In
   `.github/workflows/ci.yml`, before the pytest step of the test job, add a step
   that writes the repo secret `GDRIVE_SA_JSON` to a temp file, points
   `GOOGLE_APPLICATION_CREDENTIALS` at it, and runs
   `python scripts/fetch_data.py --served-only` (use the full set if integration
   tests need `broad_pull`). **The operator must add the `GDRIVE_SA_JSON` secret in
   GitHub settings — the PR stays red until they do; say so in the PR body.** If
   integration/quant tests need data beyond the served set, prefer switching those
   to committed fixtures over shipping all 535 MB into CI.
6. Confirm the engine still runs locally (data is still on disk):
   `python scripts/fetch_data.py --check`, then the CLAUDE.md 5-ticker smoke
   (`WheelRunner().rank_candidates_by_ev([...], as_of=<recent>)`).
7. Commit; `git push -u origin data/drive-migration`; open a **draft** PR
   "Migrate Bloomberg data off GitHub to Google Drive". Body: what moved, the
   Phase-2 completeness table, the required `GDRIVE_SA_JSON` secret, and that the
   git-history purge is a separate later job. **Do NOT merge** — leave for operator review.

## Phase 6 — Report back
Post a structured summary: Phase 1–4 pass/fail with evidence; the Tier-C backup
location(s); the PR link; and exactly what the operator must still do (add the CI
secret; decide fetch-in-CI vs. fixtures; schedule the later history purge). If any
of Phases 1–4 failed, report it and STOP — do not open the untrack PR.
