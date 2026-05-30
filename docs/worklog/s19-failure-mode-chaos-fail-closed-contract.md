---
id: S19
title: Failure-mode chaos (fail-closed contract)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Characterise how the engine behaves under hostile or
malformed inputs — bogus tickers, empty / garbage `as_of`, truncated /
empty CSVs, missing fundamentals, unreachable FRED adapter, missing data
directory, corrupted `ev_row` into the dossier — and document for every
chaos vector whether the engine fails **closed** (typed exception,
all-dropped result with structured `.attrs["drops"]`, or safe empty
frame) or **open** (≥1 tradeable row a naive caller would treat as
actionable). **A single fail-open in a tradeable-verdict path is the §2
headline; everything else is operational hygiene.**

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`, baseline
`as_of=2026-03-20`. Six vector groups: C1 (8 bogus-ticker sub-cases),
C2 (8 bogus-`as_of` sub-cases, fresh `WheelRunner` per sub-case to avoid
cache bleed), C3 (6 empty / truncated CSV sub-cases under a temp-dir
subprocess so the shared `data/bloomberg/*.csv` snapshots stay
untouched), C4 (FRED / EDGAR adapter failures via in-process
monkeypatch), C5 (missing `data/` directory entirely), C6 (§2 acid test
— row-for-row diff between `[AAPL, NOTATICKER, MSFT]` and clean
`[AAPL, MSFT]` ranks), C7 (4 dossier-side corrupted `ev_row` sub-cases:
NaN, +inf, missing `ticker`, path-traversal `ticker`). Plus an extension
probe injecting a synthetic *valid* `ChartContext` to exercise the
R5-bypass path with garbage EV — the dossier reviewer test the C7
in-script vectors couldn't reach because they all hit R2 first
(`chart_context_missing`). Throwaway instrumentation script + temp dirs;
not committed. Same pattern as S5 / S15 / S16 / S17 / S18.

**Path.** Each vector wraps a `rank_candidates_by_ev` or `build_dossiers`
call in `try / except BaseException` and a `warnings.catch_warnings(
record=True)` context, then classifies the outcome
`fail_closed_exception` / `fail_closed_empty` / `fail_open_tradeable` /
`degraded`. The §2 acid test then compares the AAPL/MSFT rows from C1c
(mixed-input) row-for-row to a clean baseline AAPL/MSFT rank — drift in
any of `strike / premium / iv / ev_dollars / ev_raw / prob_profit /
hmm_regime / hmm_multiplier` is the §2 violation indicator. The
synthetic-chart probe constructs a `ChartContext(screenshot_path=
Path('/dev/null'), visible_price=spot, error='')` — `is_ok()` returns
True per `engine/chart_context.py:86` — and steps the reviewer through
a boundary EV vector (`-25, -inf, 0, 5, 10, 25, +inf, NaN`) with the
chart match keeping R3 silent.

**Status.** Done. **Verdict: PARTIAL — engine fails closed on 22 of 27
chaos vectors, but one real §2 fail-open surfaced (C7b extension probe)
and three operational fail-opens (silent `as_of` substitution,
string-iteration on the ticker input, FRED-down silent default to
`credit_multiplier=1.0`).** §2 acid test (C6) confirms: the valid rows
returned alongside dropped unknown tickers ARE faithfully
EV-engine-computed on real Bloomberg data — row-for-row identical to a
clean rank.

> **§2 headline finding — `ev_dollars=+inf` on the dossier reviewer's
> input yields `proceed`.** Reviewer rule R1
> (`engine/candidate_dossier.py:167`) is `if ev < 0: return blocked` —
> the comparison `float('inf') < 0` is `False`, so R1 doesn't fire.
> R5 (`engine/candidate_dossier.py:206-209`) is `if ev >=
> self.min_proceed_ev: return proceed` — `float('inf') >= 10.0` is
> `True`, so R5 emits `proceed / ev_above_threshold`. With a valid
> `ChartContext` attached that passes R3 (the synthetic probe used an
> `is_ok=True` chart with `visible_price=spot`), the dossier returns a
> tradeable `proceed` verdict on garbage `ev_dollars=inf` input. **NaN
> degrades safely** (R5's `nan >= 10` is False → review),
> **`-inf` degrades safely** (R1's `-inf < 0` is True → blocked), but
> **`+inf` is the gap**. On real Bloomberg data the ranker never
> produces `inf` (the engine's `round(res.ev_dollars, 2)` would need
> `mean_pnl=inf` upstream), so this is a defence-in-depth gap — a
> downstream consumer that hand-builds an `ev_row` (e.g., the
> `engine_api.py` webhook receiving external payloads, or any future
> caller of `build_dossiers` not coming from the ranker) can be tricked
> into a `proceed`. **Logged. Not fixed in this Sn — Terminal A lane.**

**Findings:**

- **Per-vector outcome table** (27 vectors total; verdicts re-classified
  after probe-level analysis where my initial classifier was too
  shallow). Wall-clock under each shows the chaos vector failed *fast*
  — no vector hangs or runs > 2.5 s.

  | Vector | Description | Wall | Verdict | Detail |
  |---|---|---|---|---|
  | C1a | `tickers=[]` | 0.39 s | **fail_closed_empty** | 0 rows, 0 drops |
  | C1b | `tickers=["NOTATICKER"]` | 1.23 s | **fail_closed_empty** | 0 rows, drop `{gate:data, reason:"no OHLCV data..."}` |
  | C1c | mixed `[AAPL, NOTATICKER, MSFT]` | 0.49 s | **fail_closed_per_ticker** | 2 rows (AAPL+MSFT, both EV-computed); 1 drop for NOTATICKER. C6 confirms row-for-row match vs clean |
  | C1d | `["aapl"]` (lower-case) | 0.25 s | **fail_closed_empty** | dropped — no normalize on the public surface |
  | C1e | `["AAPL "]` (trailing whitespace) | 0.29 s | **fail_closed_empty** | dropped — no trim |
  | C1f | `[None]` | 0.30 s | **fail_closed_empty** | dropped with `ticker:null` in drops |
  | C1g | `["AAPL"]*50` (dupes) | 0.72 s | **deduplicates implicitly** | 10 rows = `top_n` cap on AAPL strike-set; not a § 2 issue |
  | C1h | `tickers="AAPL"` (string) | 0.49 s | **DEGRADED — per-char iteration** | string iterates to `["A","A","P","L"]`; **'L' is a real SP500 ticker (Loews Corp)** so it's ranked and 3 rows return |
  | C2a | `as_of="2099-01-01"` | 1.64 s | **DEGRADED — silent date substitution** | 2 rows ranked off latest available data; no warning |
  | C2b | `as_of="1990-01-01"` (pre-coverage) | 1.40 s | **fail_closed_empty** | 0 rows, drops "no OHLCV history at or before as_of" |
  | C2c | `as_of="2026-03-21"` (Saturday) | 1.65 s | **DEGRADED — silent date substitution** | falls back to last available (Friday); no warning |
  | C2d | `as_of="2026-03-22"` (Sunday) | 1.58 s | **DEGRADED — silent date substitution** | same |
  | C2e | `as_of="2026-12-25"` (Christmas) | 1.62 s | **DEGRADED — silent date substitution** | same |
  | C2f | `as_of="not-a-date"` | 1.38 s | **fail_closed_exception** | `ValueError: Invalid isoformat string: 'not-a-date'` |
  | C2g | `as_of=None` | 1.60 s | **DEGRADED — silent fallback to date.today()** | line 902 (`date.fromisoformat(as_of) if as_of else date.today()`); today() degrades same as C2a |
  | C2h | `as_of=date(2026,3,20)` (date obj) | 1.37 s | **fail_closed_exception** | `TypeError: fromisoformat: argument must be str` — API doesn't accept `date` objects |
  | C3a | 0-byte `sp500_ohlcv.csv` (subprocess) | 1.34 s | **fail_closed_empty** | drop `{gate:data}`; `pd.read_csv` raises `EmptyDataError`, connector catches at `engine/data_connector.py:108-111` and returns empty `_cache[key]` |
  | C3b | header-only `sp500_ohlcv.csv` | 1.35 s | **fail_closed_empty** | same as C3a |
  | C3c | `sp500_ohlcv.csv` missing `close` col | 1.38 s | **fail_closed_empty** | dropped on the `'close'` check |
  | C3d | 0-byte `sp500_vol_iv_full.csv` | 2.50 s | **not load-bearing for rank** | this file feeds `connector.get_iv_history` (mark-to-market path) and the HMM-staleness fallback, **not** the rank-time IV. Rank still returns 1 row of clean AAPL data because rank IV comes from `sp500_fundamentals.csv:implied_vol_atm` (see C3* note below). **Logged as a campaign-mapping correction.** |
  | C3e | header-only `sp500_vol_iv_full.csv` | 2.43 s | **not load-bearing for rank** | same |
  | C3f | `sp500_vol_iv_full.csv` missing IV cols | 2.51 s | **not load-bearing for rank** | same |
  | C3*  | 0-byte `sp500_fundamentals.csv` (probe) | – | **fail_closed_empty** | follow-up probe with the *actual* rank-IV source corrupted: drops `{gate:data, reason:"IV missing or non-positive"}` for every ticker; no rows returned |
  | C4a | baseline ranker without FRED network | – | **baseline_pass** | clean smoke succeeded at boot; ranker doesn't hard-require live FRED |
  | C4b | `FREDAdapter.credit_regime → ConnectionError` | 1.36 s | **DEGRADED — silent default** | 2 rows ranked; `credit_multiplier=1.0` and `credit_regime="benign"` reported in row — pro cannot tell from the row that the FRED overlay was bypassed; `wheel_runner.py:709-720` wraps the call in a permissive `try` that on any failure leaves the multiplier at 1.0 and the label at "unknown" / "benign" |
  | C4c | EDGAR adapter | – | **not_applicable** | `grep EDGARAdapter\|edgar_adapter engine/` returns only the adapter file itself + `engine/external_data/__init__.py`; **EDGAR is not referenced from the ranker / EV / dossier path**. S15-style orphaned surface — exists, not wired. |
  | C5 | missing `data/` directory | 1.34 s | **fail_closed_empty** | drop `{gate:data}`; `_data_dir / "sp500_*.csv"` does not exist → `_cache[key] = pd.DataFrame()` empty fallback at `engine/data_connector.py:101-104` |
  | C6 | §2 acid test: C1c rows vs clean AAPL/MSFT | – | **match** | clean=2 rows, mixed=2 rows; row-for-row identical across `strike / premium / iv / ev_dollars / ev_raw / prob_profit / hmm_regime / hmm_multiplier`. No drift. The dropped unknown ticker did **not** poison the valid rows. |
  | C7a | `ev_row.ev_dollars=NaN` | 0.001 s | **fail_closed_empty** | R2 fires (chart_context_missing → review); synthetic-chart probe: NaN → review (R5's `nan >= 10` is False) — safe degrade |
  | C7b | `ev_row.ev_dollars=+inf` | 0.001 s | **§2 FAIL-OPEN** | R2 fires in-script (chart missing); **synthetic-chart probe with valid chart → `proceed / ev_above_threshold`**. R1 (`if ev < 0`) doesn't catch `+inf`; R5 (`if ev >= 10`) does. |
  | C7c | `ev_row` missing `ticker` key | 0.001 s | **fail_closed_empty** | `build_dossiers` line 287-289 (`if not ticker: continue`) filters out the row |
  | C7d | `ev_row.ticker="../../etc/passwd"` | 0.001 s | **fail_closed (negative_ev)** in this test, but **input not sanitised** — reviewer uppercases to `"../../ETC/PASSWD"` and uses verbatim. A chart provider that does filesystem ops with the ticker would be exploitable; the live `FilesystemChartProvider` does `base_dir / f"{ticker}.png"` which Python's `pathlib` handles correctly today (the leading `..` is treated as a path component, not parent-dir traversal, when joined this way) — but it's a brittle no-validation seam |
  | post_smoke | 5-ticker EV smoke after all chaos | – | **pass** | Confirms the chaos didn't corrupt the live data path |

  **Tallies.** Of 27 vectors: **2 fail_closed_exception, 14
  fail_closed_empty, 1 fail_closed_per-ticker (C1c — the §2 acid-test
  positive), 1 deduplicates_implicitly, 5 DEGRADED operational
  (C1h/C2a/C2c/C2d/C2e/C2g + C4b — silent substitution / default), 1
  §2 fail_open (C7b inf bypass via the synthetic-chart probe), 3 not
  load-bearing (C3d-f), 1 not_applicable (C4c), 1 baseline_pass (C4a),
  1 acid-test match (C6), 1 post-smoke pass.**

- **C7b — `ev_dollars=+inf` bypasses both R1 and R5 → tradeable
  `proceed` verdict.** Synthetic-chart probe (`is_ok=True`,
  `visible_price=spot` → R3 silent) walked an EV grid
  `(-25, -inf, 0, 5, 10, 25, +inf, NaN)`. Verdicts:
  `-25 → blocked, -inf → blocked, 0 → review, 5 → review,
  10 → proceed, 25 → proceed, +inf → proceed, NaN → review`.
  **`+inf` is the only fail-open** — R1's
  `if ev < 0` is `False` for `+inf`; R5's `if ev >= 10` is `True`. The
  reviewer is supposed to be downgrade-only (§2: "reviewers can
  downgrade; never upgrade"); `+inf` slips through both guards. On
  real Bloomberg data the ranker doesn't produce `inf` so this isn't a
  live exploit, but it's exactly the defence-in-depth gap §2 exists to
  protect against — any downstream consumer that hand-builds an
  `ev_row` (the `engine_api.py` webhook receiving external payloads is
  the obvious candidate) can produce a `proceed` on garbage. The fix
  is one-line at `candidate_dossier.py:167` — replace `if ev < 0` with
  `if not (ev > 0) or not math.isfinite(ev)` (or similar — capture
  NaN/inf explicitly). **Logged. Not fixed in this Sn — Terminal A
  decision-layer lane.**

- **C2a/c/d/e/g — `as_of` silently substituted with latest-available
  data.** A pro running `rank_candidates_by_ev(..., as_of="2099-01-01")`
  or `as_of="2026-03-22"` (Sunday) or `as_of=None` gets back a real
  EV-ranked frame computed off the latest available trading day (here
  `2026-03-20`) — with **no warning, no flag, no drop entry**. The
  `<=as_of` OHLCV lookup is silently inclusive and never indicates the
  effective date is different from the requested. A pro testing a
  "what does the engine say for next Christmas?" scenario gets today's
  numbers labelled as Christmas. **Operational fail-open** (input
  contract relaxed silently); not strict §2 — the rows ARE
  EV-computed on real data — but a real pro-usage gap. **Logged.**

- **C1h — `tickers="AAPL"` iterates per character; 'L' is ranked.** A
  pro who accidentally passes a string instead of a list gets ranked
  candidates for ticker `L` (Loews Corp — a real SP500 name that
  happens to be a single character). The API has no `isinstance(
  tickers, list)` guard at the public surface; the iteration sees
  `["A", "A", "P", "L"]`, drops 'A' / 'P' (`A` is also a real ticker,
  Agilent, but at this `as_of` happens to be dropped by gates), and
  ranks 'L'. Returned row first ticker = "L", `ev_dollars=55.86`. The
  row IS EV-computed correctly for L — so strict §2 holds — but the
  caller's *intent* was AAPL, not L. **Operational fail-open;
  input validation gap.** Fix would be a single-line type guard in
  `rank_candidates_by_ev`. **Logged.**

- **C4b — FRED `credit_regime` raising silently defaults to
  `credit_multiplier=1.0` / `credit_regime="benign"`.** When the FRED
  adapter is unreachable (sandbox or production network failure), the
  ranker continues, the credit overlay is silently bypassed, and the
  emitted row shows `credit_multiplier=1.0` and `credit_regime="benign"`
  — **identical to a row where credit truly is benign**. A pro reading
  the diagnostic surface cannot tell whether the credit overlay ran
  with real data or was bypassed. The label *should* be `"unknown"` (a
  pre-existing convention in the codebase for unmeasured overlays —
  HMM uses it; see `wheel_runner.py:1058`) but the credit-overlay
  failure path defaults the label to "benign" / multiplier to 1.0,
  conflating "credit fine" with "credit not measured". **Operational
  silent-default; not strict §2.** Fix would change the failure-default
  label to `"unknown"` to match HMM convention. **Logged.**

- **C7d — path-traversal ticker propagates unsanitized through the
  dossier.** The reviewer uppercases `"../../etc/passwd"` to
  `"../../ETC/PASSWD"` and passes it forward verbatim. Today's
  `FilesystemChartProvider` does `base_dir / f"{ticker}.png"` and
  Python's `pathlib` joining with a leading `..` doesn't escape
  `base_dir` cleanly in either direction (the lookup just fails with
  `screenshot_not_found`), so it's not a live exploit — but the
  reviewer should validate input shape rather than rely on downstream
  consumers' incidental robustness. **Defence-in-depth gap.**
  **Logged.**

- **C6 §2 acid test passes.** Mixed input `[AAPL, NOTATICKER, MSFT]`
  produced rows that are row-for-row identical to clean
  `[AAPL, MSFT]` across `strike / premium / iv / ev_dollars / ev_raw /
  prob_profit / hmm_regime / hmm_multiplier`. The dropped unknown
  ticker did not poison the valid rows; the drops list correctly
  records the rejection with a structured-enough `{ticker, gate,
  reason}` entry. **The first-class §2 invariant — valid rows alongside
  dropped invalid ones are still faithful EV computations — holds under
  chaos.** **Logged as a positive.**

- **C3d/e/f misclassification corrected.** My initial classifier
  marked these as `fail_open_tradeable` because the rank returned 1
  row of clean AAPL data despite the corrupted file. Probe-level
  follow-up (`engine/data_connector.py:74:_FILES` +
  `engine/wheel_runner.py:813-814` reads) showed the rank-time IV
  comes from `sp500_fundamentals.csv:implied_vol_atm` (or
  `volatility_30d`), NOT from `sp500_vol_iv_full.csv`. The latter
  feeds `connector.get_iv_history` which is on the mark-to-market and
  HMM-staleness paths, not the rank IV path. **Re-classified as "not
  load-bearing for rank";** C3* probe with the *actual* IV source
  (`sp500_fundamentals.csv`) corrupted correctly yields `fail_closed_
  empty` with structured drops `{gate:data, reason:"IV missing or
  non-positive"}` for every ticker. The ranker's IV validation gate
  (`wheel_runner.py:821-829` + 833-843, percent normalisation +
  degenerate-IV drop) is the load-bearing fail-closed behaviour.
  **Logged.**

- **`.attrs["drops"]` schema is unchanged under chaos** (the S16
  finding still applies): `{ticker, gate, reason}` where `reason` is
  free text (`"no OHLCV data (empty or missing 'close')"`, `"no OHLCV
  history at or before as_of"`, `"IV missing or non-positive"`,
  `"IV degenerate after percent normalisation"`, etc.). **Logged.**

- **No vector hangs or runs longer than 2.5 s.** Even the malformed
  CSV subprocess vectors (C3 / C5) return inside 2.5 s. The engine
  fails fast under hostile inputs — no infinite loops, no retry
  storms, no deadlocks. **Logged as a positive.**

- **Validation is deep in the call stack, not at the public surface.**
  Every C1 / C2 fail-open and operational-degraded finding (C1h string
  iteration, C2a/c/d/e/g silent date substitution, C2h `date`-not-str
  TypeError) is the symptom of `rank_candidates_by_ev` doing no
  surface-level argument validation. The validation that does happen
  fires deep — in the connector (`_load` raising on empty CSV,
  caught), in the IV percent-normalisation gate
  (`wheel_runner.py:821-829`), in the dossier reviewer (R1–R6). A
  one-shot input contract check at the public surface
  (`tickers: list[str]`, `as_of: str | None`, `as_of` ISO-format
  regex, `as_of` within a tracked-coverage window with a structured
  drop on out-of-range) would close every operational fail-open at
  once. **Logged.**

- **Post-chaos 5-ticker smoke remains green.** None of the chaos
  vectors corrupted in-process state for subsequent runs (per-vector
  fresh `WheelRunner()` for C2, dedicated subprocess for C3 / C5
  meant zero cross-vector contamination on the worktree's shared
  `data/bloomberg/*.csv` snapshots). **Logged as a positive.**

- **No `WheelRunner` cache pollution surfaced.** The S18 finding
  (`_hmm_regime_cache` converges and stays) extends to chaos — across
  C1's 8 sub-cases on a single runner, the cache populated normally
  on the valid C1c / C1g / C1h paths and remained stable. No cache
  state leaked from a failed call into a subsequent one. **Logged
  as a positive.**

- **EDGAR adapter is structurally orphaned.** `grep EDGAR* engine/`
  returns only `engine/external_data/edgar_adapter.py` and
  `engine/external_data/__init__.py`. No call site on the ranker / EV /
  dossier path. This is a new S15-style "exists but not wired" surface
  — `EDGARAdapter` ships `cik_for_ticker / recent_insider_trades /
  insider_activity_signal`, none referenced from the live decision
  path. Not exercised by this Sn (`C4c not_applicable`); flagged as
  campaign companion to S15's `RiskManager` / `SectorExposureManager`
  orphaning. **Logged.**

**AI handoff.**

- **C7b is the headline §2 fix-up surface.** One-line patch at
  `candidate_dossier.py:167` to make R1 reject non-finite EV:
  `import math; ...; if not math.isfinite(ev) or ev < 0: return
  "blocked", "non_finite_or_negative_ev", notes`. Add a regression
  test that exercises `+inf / -inf / NaN` against the reviewer with a
  synthetic valid chart, asserting all three → `blocked`. **Terminal
  A decision-layer lane;** not fixed in this Sn.

- **One-shot public-surface input validation** would close C1h /
  C2a / C2c / C2d / C2e / C2g / C2h in a single change.
  `rank_candidates_by_ev` would `isinstance(tickers, list)` /
  `isinstance(as_of, (str, type(None)))` /
  `coverage_min <= parsed_as_of <= coverage_max` and emit a structured
  drop or raise a typed exception on violation. Backwards-compatible:
  callers passing a valid string-list and a valid `as_of` are
  unaffected. **Terminal A decision-layer lane.**

- **`credit_regime` failure label should be `"unknown"` not
  `"benign"`.** Match the HMM convention at
  `wheel_runner.py:1057-1058`. One-line change at
  `wheel_runner.py:709-720`. **Terminal A lane.**

- **Sibling structural orphan: EDGAR.** Combines with S15's
  `RiskManager` / `SectorExposureManager` and S16's `dashboard/
  quant_dashboard.py`-only consumer pattern. Worth a future Sn or a
  Terminal A audit pass: enumerate every `engine/` and
  `engine/external_data/` symbol that has zero live callers from the
  decision path and either wire them or retire them. **Candidate
  future scope.**

- **Ruled out per the prompt (don't litigate):** any decision-layer
  fix, network load on `engine_api.py:8787` (S20 lane), concurrency,
  Theta failures (sandbox-blocked), fuzz testing / hypothesis,
  performance / load (S18 covered).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: **closed — S19's C7b §2 FAIL-OPEN is RESOLVED**. Direct probe of `EnginePhaseReviewer.review(dossier)` on the four EV vectors with a valid `ChartContext` (visible_price=spot, is_ok=True):
    - `ev_dollars=+25` → `proceed / ev_above_threshold` (control)
    - `ev_dollars=+inf` → **`blocked / ev_non_finite`** (originally `proceed`)
    - `ev_dollars=NaN` → **`blocked / ev_non_finite`** (originally degraded to `review`)
    - `ev_dollars=-inf` → **`blocked / ev_non_finite`** (originally blocked via `<0`; now goes through the explicit non-finite check)
  - Qualitative verdict: **partial — §2 C7b CLOSED**; remaining operational fail-opens persist:
    - C2a (`as_of="2099-01-01"`): **NEW behavior** — now refuses with `ValueError: as_of=2099-01-01 is beyond OHLCV data cutoff 2026-03-20` (PR #215, S32 F3). Original was a silent date substitution; **now correctly fails closed**.
    - C2c (`as_of="2026-03-21"`, Saturday): still silently substitutes to Friday close (no warning).
    - C1h (`tickers="AAPL"`, string): still iterates per-character; 'L' (Loews) still gets ranked. Input validation gap unchanged.
    - C4b (FRED `credit_regime`): re-verified in S11 — credit overlay now PIT-aware, so the silent-default-to-benign concern partly closed (the multiplier actually moves now).
  - Numerical drift > 5%: not applicable — chaos vectors are boolean fail-closed checks.
  - Notes: PR #204 (R1a `ev_non_finite` guard) is **the closer of S19 C7b**. The CLAUDE.md §2 R1a entry explicitly cites PR #204. C2a future-`as_of` is now a typed `ValueError` per PR #215. Other operational fail-opens (C1h string iteration, C2c-e silent date substitution, C2h `date`-not-`str` TypeError) remain open.

---
