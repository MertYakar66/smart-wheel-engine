---
id: options-sentiment-gzip
title: "options_sentiment panel gzipped before the GitHub 100 MiB ceiling (item 5c)"
kind: fix
status: shipped
terminal: X
pr: 469
decisions: []
date: 2026-07-02
headline: "The 102.3 MB options_sentiment.csv sat at 97.5% of GitHub's per-blob push limit with ~4.5 months of growth headroom (next refresh past ~Oct 2026 = hard push reject); committed gzipped at 32.0 MB (3.20:1), loader spec flipped, ~30 years of headroom"
surface:
  - data/broad_pull_loaders.py
  - tests/test_broad_pull_loaders.py
---

## Goal

Campaign item 5c — the genuinely time-dated one. Facts (recon,
first-hand): blob 102,284,594 B = 97.55 % of GitHub's 104,857,600 B
per-blob push limit; growth ~572 KB/month of data (in-file monthly
byte census); headroom ≈ 4.5 months of data → a refresh extending the
panel past ~2026-11 pushes a NEW blob GitHub hard-rejects (GH001).
No producer script exists (manual Terminal pull) and no runbook
mentioned the file — a refresh operator had nothing warning them.

## What we tried / worked

gzip in place — measured 3.20:1 (102.3 MB → 32.0 MB, 30.5 % of limit,
~30 years of headroom at observed growth). Chosen over split-by-year
(the `DatasetSpec` holds one relpath; splitting needs loader surgery)
and truncate-and-archive (loses the dormant panel's history for no
reason). In-tree precedent: `iv_surface` and `vol_term_rv` are
committed gzipped for exactly this reason. `gzip -9 -n -k` on the
EXISTING bytes (deterministic header, no pandas round-trip).

## Safety case

- The panel is **dormant** (loader module's own header: "Nothing
  consumes it"); NOT in `_FILES`, NOT in `_BROAD_PULL_PINNED`, NOT in
  `connector_data_sha256` — zero EV-path/fingerprint/re-baseline
  surface.
- The loader's `_read` is already gz-transparent (pandas
  `compression='infer'`); the only code change is the spec relpath.
- The real-data content pins (rows / dmin / dmax / tickers /
  winsorization — `tests/test_broad_pull_loaders.py`) pass unchanged
  against the .gz: byte-equivalence proven by the pins themselves.
- `.gitattributes` already covers `*.gz binary`; no LFS anywhere.
- The already-pushed 102 MB blob in history is unaffected (the limit
  is per-NEW-blob at push).

## How we fixed it

`git rm` the .csv / add `.csv.gz`; spec relpath →
`per_name/options_sentiment.csv.gz` (+ warning comment);
`test_real_gz_panels_load` parametrize gains `options_sentiment`;
docs: DATA_INVENTORY row + storage note, WIRING_CAMPAIGN §0 list +
storage gotcha + §3I header, BLOOMBERG_TERMINAL_NEXT_SESSION gains the
**staging rule** (re-pulls must stage `.csv.gz`, never raw `.csv`).

## Evidence

- `tests/test_broad_pull_loaders.py` 54/54 green against the .gz
  (includes the manifest-pin content tests).
- FILE_MANIFEST coverage clean (the .gz is glob-covered).
- New blob 32,012,405 B (30.5 % of limit).

## Unresolved / handoff

- The NEXT refresh session must follow the staging rule (documented in
  the next-session doc §1); a raw re-stage would reintroduce the
  ceiling at ~102.9 MB and get rejected outright once past ~2026-10.
- `test_real_winsorization_applied` reads the raw path from the spec —
  carried automatically; noted for future spec edits.
