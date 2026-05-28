# Architecture Decisions

Why the system is shaped the way it is. Each entry is short, pins the
problem the decision solves, and names the alternatives that were
considered and rejected. The aim is to give a fresh agent enough
context to *not re-litigate* a settled design call.

If you propose a change that touches one of these decisions, update
the entry — don't leave the rationale stale.

---

## D1. EV is the only ranker

**Decision:** Every tradeable candidate must pass through
`EVEngine.evaluate`. Reviewers (chart provider, news sentiment,
advisors, dealer positioning, TradingView bridge) can **downgrade**
verdicts but never **upgrade** them. The dealer multiplier is clamped
to `[0.70, 1.05]` and only scales `ev_dollars`, never `ev_raw`.

**Why:** Ad-hoc "but this looks great on the chart" overrides are how
strategies silently regress. A single, contractual ranker is the only
way to keep the system auditable as we add advisors / data sources /
sentiment / regime overlays.

**Rejected alternatives:**
- *Composite-score ranker (TV signal + EV + advisor average)* —
  loses traceability; one input can mask another flipping sign.
- *Per-strategy ranker (one for puts, one for calls, one for
  strangles)* — same forecasts have to be reconciled anyway; better
  to do it once in `EVEngine`.

**Pinned by:** `tests/test_audit_invariants.py`,
`tests/test_dossier_invariant.py`, `tests/test_authority_hardening.py`.

---

## D2. `engine/` is the live quant layer; `src/` is deprecated

**Decision:** All new quant + decision code goes under `engine/`.
The `src/` tree (features, data, risk, models, execution, backtest)
is a phantom from an earlier scaffold. **Do not extend it.**

**Why:** A few engine files and tests still import
`src.features.technical` and `src.features.volatility` etc., so the
tree cannot be deleted today without an import refactor we have not
budgeted. Keeping it frozen avoids the refactor without inviting new
divergence.

**Rejected alternatives:**
- *Delete `src/` now* — breaks 6 tests + `engine/strangle_timing.py`,
  `engine/tv_signals.py`, `data/bloomberg_import.py`,
  `data/feature_pipeline.py`. Need a migration window.
- *Promote `src/features/` to `engine/features/`* — same migration
  cost; deferred until the feature-store consumer story is settled.

**Pinned by:** `MODULE_INDEX.md` "Other top-level dirs" → `src/`,
`PROJECT_STATE.md` §4. `pyproject.toml` still names `src` in
`[tool.hatch.build.targets.wheel] packages` — known stale.

---

## D3. Two news subsystems coexist intentionally

**Decision:** `engine/news_sentiment.py` is the only news module on
the EV path (downgrade-only reviewer). `news_pipeline/` (browser-agent
multi-LLM pipeline driven by `morning_run.py`) and `financial_news/`
(RSS/SEC/EIA/Fed connector platform) are operational systems that
**do not** feed the decision authority.

**Why:** Each system has a different purpose and a different cost
profile. Browser agents (`news_pipeline/`) reuse paid Claude/ChatGPT
/ Gemini subscriptions for zero per-call API cost; RSS connectors
(`financial_news/`) collect structured news for the dashboard. Wiring
either of them directly into `EVEngine` would double-count news
sentiment and re-introduce the override problem D1 forbids.

**Rejected alternatives:**
- *Merge `financial_news/` and `news_pipeline/` into one tree* —
  they have different deployment models (one is daemon-style with a
  scheduler, one is a single morning batch) and would force a
  premature abstraction.
- *Wire `news_pipeline/` directly into the EV path* — duplicates
  `news_sentiment.py`, breaks downgrade-only.

**Pinned by:** `MODULE_INDEX.md` "Other top-level dirs",
`PROJECT_STATE.md` §4 ("News-stack duplication").

---

## D4. The dashboard is Next.js; the legacy CLI dashboard is retained

**Decision:** Live UI is the Next.js app under `dashboard/src/`,
served by `engine_api.py` on `:8787`. The Python CLI dashboard
(`dashboard/quant_dashboard.py`) is kept for historical reference and
its tests in `tests/test_dashboard.py` continue to pass, but is not
the primary surface.

**Why:** The Next.js app is the cockpit users actually open; the CLI
predates it. Tests still cover the CLI surface, so removing it is a
net negative until we actively migrate or delete those tests.

**Rejected alternatives:**
- *Delete the legacy CLI* — would force test removal and lose
  the offline / no-browser fallback path.

**Pinned by:** `MODULE_INDEX.md` "`dashboard/`" section,
`PROJECT_STATE.md` §4.

---

## D5. TradingView is a cockpit, not a decider — and we have two TV surfaces

**Decision:** TradingView appears in the repo in two distinct roles:

1. **Engine-side bridge** (`engine/tradingview_bridge.py`,
   `engine/tv_signals.py`, `tradingview/smart_wheel_signals.pine`,
   `tradingview/alert_payload_schema.json`) — Pine indicator on the
   chart fires webhook → engine enriches via `WheelRunner.analyze_ticker`
   + EV → returns proceed/review/skip. Engine is the decider.
2. **Analyst workspace** (`tradingview/CLAUDE.md`,
   `tradingview/OVERVIEW.md`, `tradingview/launch-tradingview-cdp.sh`,
   `tradingview/research/`, `tradingview/models/`,
   `tradingview/pine/`, `tradingview/tradingview-mcp-jackson/`) —
   Mert's research-only analyst function: Claude drives a TradingView
   Desktop instance via CDP / MCP for chart reading, Pine work, and
   filed deliverables. Decision support, not autonomous execution.

The two share a folder for ergonomic reasons but have **different
contracts**. The bridge is part of the EV path (downgrade-only, see
D1). The workspace never touches `EVEngine`.

**Why:** Putting them in separate repos would split context for the
user, who needs to switch between them daily. Putting them in the
same `tradingview/` folder with explicit per-purpose docs preserves
context without confusing the contract.

**Pinned by:** `docs/TRADINGVIEW_INTEGRATION.md`,
`docs/TRADINGVIEW_MCP_INTEGRATION.md`, `tradingview/README.md`,
`tradingview/CLAUDE.md`, `tradingview/OVERVIEW.md`.

---

## D6. Theta Terminal is local-only and never enters git

**Decision:** The `Theta/` directory (~239MB of jars + creds.txt +
config.toml) is gitignored. The Terminal is installed software, not
project code; credentials live in `Theta/creds.txt` and never travel
in any commit, branch, or PR.

**Why:** Two reasons stack: (a) the jars are huge and would bloat the
repo, (b) creds.txt is a Theta API token whose leak materially
matters. The risk-of-leak / value-of-tracking ratio is unfavourable.

**Rejected alternatives:**
- *Track everything except creds.txt* — still bloats the repo with
  binaries that update independently of code.
- *Vendor the jar via Git LFS* — adds operational complexity;
  Terminal updates would force LFS bumps; the user is the only
  consumer.

**Pinned by:** `.gitignore` ("Theta Terminal — installed software"
section), `docs/LAPTOP_SETUP.md`, `docs/THETA_USAGE.md`.

---

## D7. Data-provider matrix uses `SWE_DATA_PROVIDER`

**Decision:** Provider selection is environment-driven via
`SWE_DATA_PROVIDER` (default: `bloomberg`), read by
`WheelRunner.connector` (the lazy-load property in
`engine/wheel_runner.py`) and by the `provider` resolution at the
top of `scripts/diagnose_candidates.py`. Two providers: `bloomberg`
(CSVs in git) and `theta` (live Theta v3). The SessionStart hook
warns when the variable is unset.

**Why:** Different runtime contexts need different providers — Cowork
sandboxes have no Theta access (use bloomberg); a live laptop with
the Terminal up uses theta. Hardcoding either would force code
changes per environment.

**Pinned by:** `docs/DATA_POLICY.md` §2 (capability matrix),
`WheelRunner.connector` in `engine/wheel_runner.py`, SessionStart hook.

---

## D8. Drive mounts are eventually-consistent mirrors, not source of truth

**Decision:** Treat the Google Drive mount as a sync mirror of the
laptop's currently-checked-out branch. Always resolve via
`git fetch origin && git checkout <branch>` rather than trusting
`ls` on the Drive path. To read a newer revision without checking
it out, use `git show origin/<branch>:<path>`.

**Why:** Drive's sync is asynchronous and silent. A branch the
laptop hasn't checked out won't appear on Drive even if origin has
it. Worse, Drive denies `unlink` on existing tracked files, so
`git pull` may fetch refs but fail to update the worktree.

**Pinned by:** `docs/DATA_POLICY.md` §6 (Drive-mount caveats),
`docs/LAPTOP_SETUP.md`.

---

## D9. SVI surface tooling is dormant until missing-data contract is decided

**Decision:** `engine/volatility_surface.py`
(`VolatilitySurfaceBuilder`, `create_empirical_surface`,
`SVICalibrator`) is exported but has zero non-test callers as of
2026-04-25. Theta `iv_surface/` snapshot covers only 28/503 tickers.
`get_iv_surface()` returns an empty DataFrame on missing data — never
a flat-IV stub.

**Why:** Wiring SVI surfaces in before deciding the missing-data
contract would create a silent fallback path. The choice is between
(a) failing loudly on the ~475 uncovered tickers or (b) using a
clearly-named fallback (`flat_iv_fallback`, never silent). Either is
acceptable; a silent flat-IV stub is not.

**Pinned by:** `PROJECT_STATE.md` §3 (iv_surface integration
decision), `MODULE_INDEX.md` (`volatility_surface.py` marked
**dormant**). `CLAUDE.md`'s quant-layer entry points back at this
decision rather than re-stating it.

---

## D10. Invariants first, then 80% line coverage as a forcing function

**Decision:** The test suite enumerates structural invariants
(EV authority, dossier rules R1–R6, percent↔decimal normalisation,
P&L accumulator orthogonality) **before** chasing line coverage.
Once invariants are pinned, target **80% line coverage** on the
CI scope (`src + engine + advisors + financial_news` + `data/quality.py`, per
`.github/workflows/ci.yml`) as a forcing function for edge-case
discovery.

The launch-blocker subset in `TESTING.md` (consolidated in
`docs/LAUNCH_READINESS.md`) is the floor for any decision-layer change;
the full suite (`pytest tests/ -v`) is the floor for shipping
anything that touches `engine/ev_engine.py`,
`engine/wheel_runner.py`, or `engine/candidate_dossier.py`. The
80% coverage gate is a check on the **next layer down**: it forces
us to think about error paths, edge cases, and adversarial inputs
on every module we ship, not just the EV authority itself.

**Why 80% (not 90%):** the 2026-05-05/06 coverage push (CHANGELOG,
PRs `#63`–`#69`) landed at 82% on the CI scope. The remaining ~10pp
to 90% lives in `news_pipeline/{browser_agents,scrapers,
orchestrator}.py` — research-tier code (`MODULE_INDEX.md` "Other
top-level dirs"), not on the EV decision path. Engine consumes
those modules' output via files on disk; producers' browser
plumbing would require ~hundreds of lines of Playwright + aiohttp
mock fixture infrastructure to test. Pursuing it for the
percentage would be coverage theater (Goodhart's law). 80% pins
the EV-adjacent floor we earned, with 2pp buffer for normal PR-to-
PR noise.

**Why coverage at all:** Coverage tells you which lines ran;
invariants tell you whether the system still does what it
promises. Both matter. The audit history (`PROJECT_STATE.md` §2)
is mostly invariant-pinning work because that was the highest-
leverage gap when those audits ran. With the invariants now
pinned, covered-but-untested edge paths are the next-most-likely
source of latent bugs — the 2026-05 Phase 1 PR (`#65`) found a
real `NaT` crash in `event_gate.from_bloomberg_calendar` purely by
exercising the path. That single bug paid for the whole exercise.

**Rejected alternatives:**
- *90% as the gate, with research-tier modules excluded via
  `[tool.coverage.run] omit`.* Hides the modules from the report
  rather than admitting they're not value-bearing. 80% with full
  visibility is more honest.
- *100% coverage as the gate.* Pushes test authors toward trivial
  exercising tests for getters, dataclass fields, and `__init__`s
  that add noise without catching bugs.
- *Skipping the gate entirely.* Leaves edge-case discovery to
  bug reports from production usage, which is too late for the
  trading-decision domain.
- *82% (the current measured baseline).* Brittle — breaks on any
  small refactor that touches an untested branch. Defeats the
  forcing function: every change becomes about not regressing the
  number rather than about real edge-case discovery.

**Pinned by:** `TESTING.md`, `docs/LAUNCH_READINESS.md`, every
`test_audit_*.py`, `tests/test_dossier_invariant.py`,
`tests/test_authority_hardening.py`, `tests/test_launch_blockers.py`.
The 80% gate itself is enforced by `pyproject.toml`
`[tool.coverage.report] fail_under = 80` and the matching
`.github/workflows/ci.yml --cov-fail-under=80`. See `ROADMAP.md`
Track E for the per-PR breakdown that landed the 82% baseline.

---

## D11. Fail loudly on per-endpoint Theta failures (no silent CSV substitution)

**Decision:** A network-level failure (`ConnectionError` /
`RetryError` / `Timeout`) on any per-endpoint ThetaData v3 call
triggers a 5-second probe of
`/v3/option/list/expirations?symbol=SPY` via
`ThetaConnector.is_terminal_alive`. The outcome of that probe
determines what happens next:

  * **Probe healthy** → `ThetaConnector._fetch` records a
    `FailureRecord` and raises `PerEndpointFailure`. Callers — the
    per-ticker loops in pullers — MUST catch the exception, mark
    the ticker as failed for this run, and continue. They MUST NOT
    fall back to Bloomberg CSV; that's the contamination case.
  * **Probe also fails** → the connector enters per-instance
    "Terminal-down mode" (sets `self._terminal_down`), logs the
    event once, and returns an empty DataFrame. The existing
    empty-df → `super().get_X(...)` Bloomberg fallback at the
    caller layer takes over for the remainder of that connector
    instance's lifetime. Subsequent network failures within the
    same instance short-circuit on the flag without re-probing.

Carve-outs:

  * Globally-down Terminal → CSV fallback is acceptable.
    Backfilling with the historical store while the Terminal is
    offline is by design, not the contamination case.
  * HTTP-status responses (400 / 403 / 404 / 500 / 502 / 503) and
    empty-body "no data here" responses are NOT failures. They
    remain deliberate signals from the Terminal (tier blocks,
    symbol not in Theta) and continue to return empty DataFrames
    as before.

State scope:

  * `self._failures` and `self._terminal_down` are per-connector-
    instance. A puller using multiple `ThetaConnector` instances
    must aggregate failures manually (no puller does today; the
    contract is pinned in the docstring).
  * `self._terminal_down` does not auto-reset within an instance.
    Callers that need to retry the probe should construct a fresh
    connector.

**Implementation note — per-ticker connector pullers.** Most pullers
use a single main-scoped `ThetaConnector` and drain
`conn.get_failures()` at end-of-main into the JSON sidecar
(`data_processed/theta/_manifest_failures_<step>_<utc_ts>.json`,
written from a `try/finally` so half-run pulls still emit it). The
outliers are `scripts/pull_theta_options_flow.py`,
`scripts/pull_theta_corp_actions.py`, and
`scripts/pull_theta_option_tape.py`, which instantiate a fresh
connector inside each worker (the first to bypass the per-instance
`_MAX_CONCURRENT=4` semaphore for higher run-wide concurrency; the
others' rationale is not pinned in the codebase). The contamination
fix still operates correctly there — `PerEndpointFailure` propagates
through the worker's `except Exception` catch as a FAIL stdout row,
no parquet is written — but the connector's `_failures` list dies
with the worker, so these pullers do not write a manifest sidecar.
If structured failure observability becomes valuable for any of
them, hoist to a single shared connector first (and accept the
throughput change).

**Why:** The 2026-05-06 `pull_theta_options_flow` run logged 7
"ThetaTerminal not reachable" warnings within a 3-minute window
while the Terminal was healthy throughout — HTTP 200 on the probe
URL in ~210ms before, during, and after the warnings. The actual
failures were 30s read-timeouts on per-symbol
`/v3/option/history/eod` calls. The connector misdiagnosed these
as global Terminal-unreachable and silently substituted Bloomberg
CSV. Downstream feature computation could not distinguish
Theta-sourced rows from CSV-fallback rows; EV inputs ran on
mixed-provenance data. Issue #71 made the choice explicit:
per-endpoint failures with a healthy Terminal should kill the
per-ticker path for that run, not silently substitute. Observable
failures beat silent contamination.

The typed exception (rather than a sentinel-DataFrame discipline)
is deliberate. Empty-DataFrame was already the convention for "no
data here" — tier blocks, symbols Theta has no history for, etc. A
new contamination-bug fix needs a different signal so callers
cannot accidentally treat the two cases identically.
`PerEndpointFailure` is unmistakable; an empty DataFrame is not.

The connector-side failure accumulator (`self._failures` +
`get_failures()`) exists so each puller can add ONE end-of-run
sidecar write rather than each per-ticker `try/except` carrying
manifest-write boilerplate. Centralising the accumulator gives a
single source of truth for what failed in a run; pullers consume
it via `connector.get_failures()` (returns + clears) and write a
JSON sidecar at
`data_processed/theta/_manifest_failures_<step>_<utc_ts>.json`.

**Rejected alternatives:**

- *Sentinel-DataFrame discipline (return `None` on per-endpoint
  failure instead of raise).* Fragile, because empty-DataFrame is
  already the no-data convention. Two semantically-distinct return
  values on the same call site are too easy to confuse.
- *Per-puller probe (each puller checks `is_terminal_alive`
  itself).* Duplicates the probe across every puller and
  re-introduces the misdiagnosis bug if any puller forgets it.
- *Also abort on tier-blocked 403s.* Wrong layer — 403s are
  deliberate "your subscription doesn't cover this", not
  contamination. Keeping them as empty-df preserves the existing
  tier-aware fallback chain. `get_vix_family`'s CBOE / Yahoo
  fallback (different data sources, not Bloomberg CSV) is
  intentionally left alone for the same reason.
- *Auto-reset `self._terminal_down` after N seconds of probe-
  success.* Rejected for v1: pullers typically run end-to-end with
  one connector per run; once the Terminal is down for a run,
  retrying mid-run rarely helps. Revisit if a long-lived service
  starts using the connector.

**Pinned by:** `engine/theta_connector.py`
(`PerEndpointFailure`, `FailureRecord`, `_handle_network_failure`,
`get_failures`); `tests/test_theta_connector_v3.py::TestPerEndpointFailure`
(probe-healthy raise, probe-fails empty + flag, Terminal-down
re-probe skip, fresh-instance contract, `get_fundamentals`
propagation, JSON-serialisable record, and a parametrized trio
covering `ConnectionError` / `ReadTimeout` / `RetryError`).

---

## D12. TradingView MCP transport is the `tv` CLI (Option A)

**Decision:** The engine reaches the tradingview-mcp server by shelling
out to its `tv` command-line interface (JSON on stdout), not by
speaking the MCP-over-stdio JSON-RPC protocol and not by driving Chrome
DevTools Protocol directly. `engine/mcp_client.py`'s `MCPCLIClient` is
the concrete `MCPChartClient`: one `capture()` runs `tv symbol` /
`tv timeframe` / `tv state` / `tv quote` / `tv screenshot` as five
subprocesses, exactly once each, no retries. A mandatory step's first
failure aborts with a canonical `MCP_ERROR_MODES` value; the lone
exception is the best-effort `tv quote` step — its failure is caught
and logged, `visible_price` is left `None`, and the capture still
succeeds. It is **not** wired as a provider default —
`MCPChartProvider`'s default client stays `_UnconfiguredMCPClient`;
promoting `MCPCLIClient` to a default behind `SWE_USE_MCP_CHART`
(integration Stage 3) remains gated on a separate human decision.

**Why:** `ChartContextProvider.fetch()` is synchronous and is called in
a plain loop by `build_dossiers`. The MCP-over-stdio transport's Python
SDK is asyncio-based; bridging async into the synchronous dossier path
per call is fragile and adds a dependency. The `tv` CLI emits JSON
explicitly "for piping with jq" — it is purpose-built for programmatic
consumption, needs no new Python dependency, and is a documented,
stable surface. The engine makes four fixed calls; it needs no MCP
tool discovery, schema negotiation, or sampling.

**Rejected alternatives:**
- *Python MCP SDK (`mcp` package) over stdio.* Async; per-call
  async→sync bridging is error-prone; adds a runtime dependency; the
  tool-schema machinery is wasted on four fixed calls.
- *Direct Chrome DevTools Protocol from Python.* Re-implements what the
  MCP server already encapsulates (TradingView's undocumented internal
  APIs). Defeats the point of integrating the server.

**Live-verification (2026-05-19):** `MCPCLIClient` was verified against
a live `tv` CLI (the `LewisWJackson/tradingview-mcp-jackson` fork)
driving TradingView Desktop on Windows via CDP. Confirmed and fixed in
`engine/mcp_client.py`: Windows `cmd /c` invocation of the npm-linked
`tv` shim, the `success` status field, and the `file_path` screenshot
key. `tv state` carries no price; a fifth best-effort `tv quote
<SYMBOL>` call now supplies the live spot — verified to return a flat
`{"success": true, ..., "last": <spot>}` payload, so `visible_price`
is populated from its `last` field. Only the per-mode error strings in
`_classify` remain unverified — no live error path was exercised.

**Pinned by:** `engine/mcp_client.py` (`MCPCLIClient`),
`tests/test_mcp_client.py` (42 tests, all mocking `subprocess`),
`engine/tradingview_bridge.py` (`MCPChartClient` protocol,
`MCP_ERROR_MODES`), `docs/TRADINGVIEW_MCP_INTEGRATION.md` §9.

---

## D13. TradingView MCP is co-located and opt-in (Stage 3)

**Decision:** Integration Stage 3 wires `MCPChartProvider` into the
canonical chart-provider factory `build_default_provider`
(`engine/tradingview_bridge.py`) — the factory `WheelRunner` already
uses for dossier builds. Two boundaries are pinned, resolving the
`docs/TRADINGVIEW_MCP_INTEGRATION.md` §8 open questions q2 and q3:

- **Co-located (q2).** The engine and TradingView Desktop run on the
  same machine; the `tv` CLI reaches Desktop via CDP on
  `localhost:9222`. The operator brings Desktop + the tradingview-mcp
  server up before a run. On any auth/session failure `MCPCLIClient`
  raises a canonical `MCP_ERROR_MODES` value, `MCPChartProvider`
  downgrades only, and `ChainedChartProvider` falls through to
  `FilesystemChartProvider` — no quiet substitution.
- **Opt-in (q3).** `MCPChartProvider` is absent from the chain unless
  the `SWE_USE_MCP_CHART` environment variable is truthy
  (`1`/`true`/`yes`/`on`). When opted in it takes the §4 canonical
  first position (live MCP → cached filesystem → headless Playwright).
  An explicit `enable_mcp=` argument overrides the env var.

**Why:** Co-location matches the existing ThetaTerminal local-only
convention (D6) — one less network surface, no remote-transport work
beyond the `MCPCLIClient` subprocess design (D12). Opt-in is the
conservative default: a live five-call MCP capture costs materially
more than a cached filesystem read, so across a top-10 dossier batch
it pays only when the operator explicitly wants fresh chart state. No
live `capture_screenshot` latency measurement was available when this
shipped (no TradingView Desktop in the build sandbox); default-first
ordering should be revisited only against a measured p95 <2s.

**Rejected alternatives:**
- *MCP as the default-first provider.* Unmeasured latency; would slow
  every dossier build for all callers, including those with no MCP
  server up (each pays a full timeout before falling through).
- *Separate-host MCP server.* Needs a remote transport beyond the
  current subprocess `MCPCLIClient`; larger scope, more failure modes,
  no demand.

**Pinned by:** `engine/tradingview_bridge.py` (`build_default_provider`),
`tests/test_tv_dossier.py::TestBuildDefaultProvider`,
`docs/TRADINGVIEW_MCP_INTEGRATION.md` §9.

---

## D14. Tiered documentation layout — root holds only the entry + index docs

**Decision:** Repository documentation is organised into discovery tiers.
**Tier 1** (canonical entry, repo root): `AGENTS.md`, `CLAUDE.md`,
`README.md`. **Tier 2** (state + index, repo root): `PROJECT_STATE.md`,
`MODULE_INDEX.md`, `TESTING.md`, `DECISIONS.md`, `COMMIT_GUIDE.md`,
`FILE_MANIFEST.md` — plus `CHANGELOG.md` and `ROADMAP.md`, kept at root
as the temporal-state triad with `PROJECT_STATE.md`. `LICENSE` stays at
root by convention. Every other Markdown doc lives in `docs/`. Stale or
superseded artifacts move to `archive/YYYY-MM/`. **Tier 3**
(`.claude/commands/`) holds thin slash-command wrappers around
already-documented workflows.

This change is **structure-only**: files move, inbound references are
updated in the same commit as each move, and no doc's substantive
content is rewritten. Two clean-ups are **deferred to named follow-on
PRs** so this PR's diff stays reviewable as a pure move:

- **CLAUDE.md lean-rewrite.** `CLAUDE.md` is edited in this PR for
  moved-path references only (the `LAPTOP_SETUP.md` mentions and the §5
  docs list). A dedicated follow-on PR slims its content; this PR
  deliberately adds no `CLAUDE.md` prose.
- **Doc-truthfulness reconciliation.** Known-stale facts — code line
  numbers, the `engine_api` endpoint count, the test count, SVI
  dormancy wording, the `_yf`-merge claim, the `README.md` broker/CLI
  body, the `PROJECT_STATE.md` §5 drift list — are left exactly as
  found. This PR neither fixes nor propagates them; a dedicated
  follow-on PR reconciles them. `FILE_MANIFEST.md` and the index-doc
  refresh describe file *purpose* only and introduce no such number.

**Why:** A fresh agent landing at the repo root previously faced a wall
of Markdown files with no signal as to which to read. The tiered layout
puts the entry + index docs at root and everything else one predictable
level down in `docs/`, with `FILE_MANIFEST.md` as the exhaustive
per-file index. Splitting the cleanup across three sequenced PRs — this
move, the CLAUDE.md slim, the truth reconciliation — keeps each diff
independently reviewable: a move PR that also rewrote prose could not be
verified as behaviour-neutral, and the decision log should show that the
known-stale facts were left in place *on purpose*, not missed.

**Rejected alternatives:**
- *Move docs and fix their stale content in one PR.* Mixes a structural
  diff with prose edits; a reviewer then cannot confirm the move is
  content-neutral. Deferred to the reconciliation PR.
- *Invert `AGENTS.md` / `CLAUDE.md` (AGENTS canonical, CLAUDE
  delegating).* Considered for the Tier-1 split; rejected for this PR —
  heavy surgery on the two most sensitive docs that would relocate the
  §2 EV invariant. `CLAUDE.md` keeps its role here; its slimming is the
  separate follow-on PR.
- *Sub-folder `docs/` by topic.* Multiplies cross-reference churn for no
  navigability gain once `FILE_MANIFEST.md` + `AGENTS.md` index `docs/`.
- *Move `audit.py` into `scripts/`.* Would pull it into CI's ruff scope
  and mix a lint-scope change into a structural PR while CI lint is
  already red. Left at the repo root.
- *Delete the empty `src/` subpackages or `models/`.* `src/` is still
  imported by live modules (see D2); `models/` is `ml/wheel_model.py`'s
  default output directory. Only the genuinely dead, zero-reference
  `validation/` placeholder was removed.

**Migration path:** No compatibility shims — no moved file is an
imported Python module, and `audit.py` (the one root script with a
module name) was deliberately not moved. All inbound references were
updated in the same commit as each move. External bookmarks to
repo-root doc URLs should repoint to `docs/<name>.md`, or to
`archive/2026-05/<name>` for the three archived docs. **Shim expiry:**
n/a — no shims were created.

**Pinned by:** `FILE_MANIFEST.md` (the exhaustive index this layout
assumes), `archive/README.md`, `AGENTS.md` (the `docs/` "read on demand"
index), `PROJECT_STATE.md` §3 (the dated reorg entry).

---

## D15. Parallel-session coordination is N-generic; every terminal lives in its own worktree

**Decision:** `docs/PARALLEL_SESSIONS.md` describes the pattern for an
arbitrary number of executor terminals, not a fixed two. Roles are
**Terminal X** + **Session X** (one verifier per executor); the live
letter count comes from board #113's "Live claims" section, not from
the doc. Every terminal — *including Terminal A* — works from a
dedicated worktree (`../swe-terminal-<x>`); the primary clone
(`smart-wheel-engine/`) is reserved for Sessions, orchestration, and
safety, with no executor running in it. Per-terminal env (port,
coverage file, pytest cache, data provider) is sourced from
parametrised setup scripts: `scripts/setup-terminal.sh` and the
PowerShell companion `scripts/setup-terminal.ps1`. The `Sn` /
next-free-number rule is generalised explicitly to `D<N>` too —
both are global, both are claimed on the board before use.

**Why:** Two earlier rules were starting to bend. The doc was hardcoded
to two terminals at a time when the project was actually planning to
spin up a third for short bursts of housekeeping work, which would
have meant a doc rewrite each time the count flexed. And the
"Terminal A keeps the primary clone" carve-out put A in a privileged
position that violated the shared-working-tree hazard it was supposed
to be protecting against — the very recurring-hazards bullet in the
doc warned about exactly the failure mode A was exposed to, just for
A alone. Making the doc N-generic and moving A out of the primary
removes both the special case and the structural inconsistency in one
pass. The setup scripts pin the env conventions (port, coverage file,
pytest cache, provider) so every terminal lands in the same shape.

The Sn ↔ D-number generalisation closes a near-miss: D-numbers are
exactly the same kind of monotonic global counter as `Sn` and the
same parallel-collision risk applies. Rule 7 now names both
explicitly.

**Rejected alternatives:**
- *Keep A in the primary, just document it as the exception.* Was
  the status quo; left A exposed to the same shared-working-tree
  hazard the doc warned everyone else about. The whole point of
  rule 1 is that no terminal should run there.
- *Per-terminal `SWE_DATA_PROCESSED_DIR` and `SWE_MODELS_DIR` by
  default.* The pulled CSVs are ~140 MB combined and read-only at the
  EV-ranker layer; duplicating them per terminal pays cost for no
  observed contention. The script keeps the variables defined (so
  forward code that respects them lands in a consistent shape) but
  points them at the shared dirs; switching to per-terminal is a
  one-line edit when contention appears.
- *Wire `SWE_API_PORT` into `engine_api.py` in the same PR.* Out of
  scope for the docs/coordination PR D15 shipped in. The promotion
  was the natural follow-on and **landed as the C7 fix from audit
  issue #154** — `engine_api.py._resolve_port()` and `audit.py`'s
  `BASE` both now honour `SWE_API_PORT` with 8787 as the default
  fallback. Pinned by `tests/test_engine_api_port.py`.
- *Hardcode Terminal C in the doc.* The whole point of the rewrite
  is N-generic; the moment a third terminal is needed, it claims on
  the board, sources `setup-terminal.sh c`, and starts. The doc
  needs no edit.
- *Single setup script in one shell (just `.sh` or just `.ps1`).*
  Repo runs on Windows-local + Ubuntu-CI; the bash version covers
  Git Bash / WSL / CI shells, the PowerShell version covers native
  Windows shells without an extra shim layer.

**Pinned by:** `docs/PARALLEL_SESSIONS.md` (the rewritten doc itself
is the spec), `scripts/setup-terminal.sh`, `scripts/setup-terminal.ps1`,
`FILE_MANIFEST.md` (records both setup scripts under `scripts/`).

---

## D16. EV-authority token is verdict-bound, not just provenance-bound

**Decision:** The EV-authority token gate enforces R1 ("negative EV →
blocked") at **two** stages, not one. Issuance refuses non-tradeable
rows outright by raising :class:`EVAuthorityRefused` when
`ev_row['ev_dollars'] <= 0`. Consume re-checks a fresh
`current_ev_dollars` argument supplied at fire time: a token that was
positive at rank time but went stale by fire time is rejected with the
token **retained** for retry (the calc-happened fact is immutable; a
transient stale-EV does not invalidate it). Both legs of the wheel —
`open_short_put` and `open_covered_call` — flow through the same
`_consume_ev_authority_token` predicate; before D16 only the short-put
leg was gated despite the constructor docstring claiming both were.

The persistence schema (PR #128's `set[str]` of token hashes) is
unchanged. Tokens issued under the old code remain consumable from a
reloaded tracker as long as the caller supplies a fresh
`current_ev_dollars`.

**Why:** S8 (`docs/USAGE_TEST_LEDGER.md`) found a real DIS candidate
with `ev_dollars = -30.65` that the token gate accepted because the
gate only verified that *a calc had happened*, not that the calc said
*tradeable*. A token can outlive its underlying rank position: market
moves between rank time and fire time, and "EVEngine.evaluate ran" is
not the same property as "this is positive-EV now". The two-stage
predicate aligns the token gate with `EnginePhaseReviewer` R1 so the
token can never accept what R1 explicitly blocks. The call-leg gap
was a structural correctness bug, not just doc drift — the docstring
had promised the leg was gated since the audit-VI hardening pass
landed.

**Rejected alternatives:**
- *Option B — R5-strict / `min_proceed_ev` threshold at issuance.*
  R5 is a configurable reviewer preference (the proceed/review
  threshold); R1 is the hard CLAUDE.md §2 rule. The token must
  enforce R1, not import the reviewer's tunable. Importing R5 would
  also create a new cross-module invariant between `wheel_tracker.py`
  and `candidate_dossier.py` that would silently drift as either
  side's threshold changes.
- *Option C — verdict-encoded token; schema migration.* Encode the
  rank-time verdict into the token payload itself and migrate
  persistence to carry it. Rejected for this PR because PR #128 (the
  persistence schema) shipped ~3 days before this work and migrating
  the just-stabilised schema is disproportionate to closing the
  named S8 finding. Option A+Y (this decision) closes S8 without the
  migration cost; Option C is the right move when a second use case
  justifies it.
- *Externalised threshold knob (e.g. `min_ev_dollars` constructor
  param on `WheelTracker`).* The threshold IS zero — that's R1.
  Adding a configurable knob re-introduces the Option-B drift risk
  and gives operators a footgun (set the knob below zero and the
  gate is back to provenance-only).
- *Discard the token on stale-EV consume rejection.* The token
  represents the immutable fact "an EV calc happened on this
  canonical ev_row". A stale-EV rejection only says the trade isn't
  positive-EV right now, not that the evidence of the calc is gone.
  A subsequent fresh re-rank that returns positive should be able to
  re-fire the same token. Retaining also makes the audit log more
  useful: N×reject with `reason=stale_ev` for one token is a signal
  the system was bouncing off the gate.
- *Make `current_ev_dollars` required positional.* Would break every
  non-strict test caller in the repo for zero gate value — the gate
  only fires when `require_ev_authority=True`. Keeping the param
  optional preserves the research/test surface unchanged.
- *Gate the call leg silently (no D-entry, treat as bug fix).*
  Considered. Rejected because the leg-asymmetry has been load-bearing
  for the audit-VI hardening narrative since the constructor docstring
  was written; closing it explicitly with a D-entry beats a commit
  message that decays.

**Pinned by:** `engine/wheel_tracker.py` (`EVAuthorityRefused`,
`issue_ev_authority_token`, `_consume_ev_authority_token`,
`open_short_put`, `open_covered_call`),
`tests/test_authority_hardening.py` (D16 test block including
`test_s8_dis_negative_ev_refused_at_issue`),
`tests/test_audit_viii_e2e.py`, `tests/test_wheel_tracker_persistence.py`
(`test_persisted_token_consume_round_trip_d16`),
`docs/USAGE_TEST_LEDGER.md` S8 (the originating finding).

---

## D17. Portfolio-level risk gates are wired on both surfaces — hard-block on entry, soft-warn on review

**Decision:** Portfolio-wide risk gates (sector concentration, portfolio
delta, Kelly per-trade sizing, parametric VaR, hypothetical stress
drawdown, dealer-regime) live in one pure-function library
`engine/portfolio_risk_gates.py` and are consumed on **two** decision-
layer surfaces:

1. **Tracker hard-blocks** (`engine/wheel_tracker.py._evaluate_d17_hard_blocks`)
   — at `open_short_put` / `open_covered_call` time, in strict mode
   (`require_ev_authority=True`), three gates fire as **refusals**
   (audit-log `action="reject"` with `nav` + `nav_source` fingerprint):
   sector cap, portfolio delta cap, Kelly per-trade NAV cap. NAV is
   computed **live** via `_compute_live_nav` (mark-to-market through
   the attached connector) and threaded into every gate so all three
   see one consistent value per call. A pre-gate `nav_exhausted` refuse
   fires when live NAV drops below the operator-set
   `min_nav_for_trading` floor. The Kelly gate is intentionally
   short-circuited on the covered-call leg (stock already owned; no
   new margin).
2. **Dossier soft-warns** (`engine/candidate_dossier.py`
   `EnginePhaseReviewer` R7 + R8 + R9 + R10) — at ranking time, when
   a `PortfolioContext` is attached, the reviewer adds four
   *downgrade-only* rules on top of the existing R1–R6:
   - **R7** = `check_var`. Portfolio VaR_95 (30-day horizon) above
     `max_var_pct × NAV` → `proceed` downgrades to `review`,
     `verdict_reason="portfolio_var_breach"`.
   - **R8** = `check_stress_scenario` OR `check_dealer_regime` (one
     rule, two triggers — mirrors R6). The C4 vol-spike scenario's
     drawdown above 8% NAV fires `verdict_reason="stress_breach"`;
     the candidate's underlying being in `short_gamma_amplifying`
     regime fires `verdict_reason="short_gamma_regime"`. Distinct
     reasons per trigger so the audit trail records which one.
   - **R9** = `check_sector_cap`. Added in B2 closure (2026-05-27).
     If opening the candidate would push its GICS sector over
     `max_sector_pct × NAV` (default 25% — same gate the tracker
     applies as a HARD refusal at `open_short_put` time when
     `require_ev_authority=True`), `proceed` downgrades to `review`,
     `verdict_reason="sector_cap_breach"`. Soft-warn preview of the
     tracker's hard refusal so a trader scanning the dossier UI
     sees the warning BEFORE attempting execution. Skips silently
     when `nav == 0` or the context is missing (Q3 semantics).
   - **R10** = `check_single_name_cap`. Added 2026-05-27 as the F4
     damage-bounding addition (the F4 widening fix had to be
     rolled back per `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10 because
     HMM `crisis` labels over-fired on the calm-bull plurality;
     R10 is the orthogonal-by-design follow-up that bounds the
     damage from idiosyncratic single-name drawdowns the engine
     cannot predict). Tighter per-underlying floor that sits
     BENEATH the sector cap: a ticker concentrated as the only
     name in its sector could still pass R9 at 25% NAV, but R10
     catches it at 10% NAV first. Aggregates SHORT-option notional
     by symbol via `check_single_name_cap`. Default 10% NAV.
     `verdict_reason="single_name_breach"`. Tracker mirrors the
     same gate as a HARD refusal at `open_short_put` (added to
     `_evaluate_d17_hard_blocks` as Gate 1b, between sector and
     delta).

R1 (`negative EV → blocked`) still wins over every D17 surface — the
hard invariant from CLAUDE.md §2 / D1 / D16 is not amended.

**Locked defaults** (overridable per-call; do **not** edit the constants):

| Gate | Default | Module constant |
|---|---|---|
| Sector cap | 25% of NAV | `_DEFAULT_MAX_SECTOR_PCT = 0.25` |
| Single-name cap | 10% of NAV | `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10` |
| Portfolio delta cap | ±$300 per $100k NAV | `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` |
| Kelly per-trade fraction | 50% NAV ("half-Kelly") | `_DEFAULT_KELLY_FRACTION = 0.5` |
| VaR ceiling | 5% NAV at 95% / 30d | `_DEFAULT_MAX_VAR_PCT = 0.05` |
| Stress drawdown ceiling | 8% NAV | `_DEFAULT_MAX_STRESS_DRAWDOWN_PCT = 0.08` |
| C4 vol-spike scenario | −10% spot + 30% IV | `_C4_VOL_SPIKE_SCENARIO` |

Missing-data behaviour (soft-warns only): when the dossier's
`PortfolioContext` is absent, or `check_var` has neither correlation
matrix nor returns data, or `check_dealer_regime` has no regime map
for the ticker, the gate returns `passed=True, reason="missing_data"`
and the soft-warn does **not** fire. Q3 of the #154 C4 design
checkpoint: soft-warns should not fire on absent evidence. Matches
D11's "no silent substitution" principle — the gate refuses to
penalise a candidate for data it can't see.

The five tracker audit-log entry shapes (D17 hard-block rejects)
join the five D16 shapes in `tests/test_ev_authority_log_schema.py`'s
`_VALID_SHAPES`: `nav_exhausted`, `sector_cap_breach`,
`single_name_breach`, `portfolio_delta_breach`, `kelly_size_exceeded`
— each carries `nav` + `nav_source` plus its gate-specific details
bag.

**Why:** S15 (`docs/USAGE_TEST_LEDGER.md`) found that
`engine/risk_manager.py` and `engine/stress_testing.py` ship complete
machinery — portfolio Greeks, parametric / historical / Monte-Carlo
VaR, `SectorExposureManager`, full stress ladders — but **none of it
was imported** by `wheel_runner.py` / `wheel_tracker.py` /
`ev_engine.py`. Position-opening was happening today with no
NAV-level sector cap, no portfolio-delta cap, no Kelly check, no
tail-risk awareness. The risk infrastructure existed; the decision
layer didn't consume it. The two-surface wiring (hard-block at
opening + soft-warn on review) keeps the contracts orthogonal: the
tracker refuses positions that would breach absolute book-level
limits; the dossier flags positions where tail-risk evidence
weakens an otherwise-`proceed` verdict. R1 remains the single hard
authority — D17 never rescues a negative-EV trade.

Locked defaults are operator-tunable per call but are written into
the module as constants because they encode the C4 design's risk
tolerance and should not drift PR-to-PR during remediation work;
tuning them is a follow-on decision, not part of D17.

**Rejected alternatives:**

- **Binary Kelly (`f* = (p*b - q) / b`) — the classical formula.**
  This was Phase 1's first cut. For any realistic short put
  (`avg_win = premium × 100 ≈ $50–$300`, `avg_loss ≈ (strike −
  premium) × 100 ≈ $5k–$15k`), the loss-to-win ratio is wide enough
  that the formula returns 0 for any plausible `win_rate`. After the
  `max(0, …)` clamp the recommended exposure is $0 and the gate
  refuses every short put regardless of edge. Phase-1 unit tests
  passed because they used synthetic `(win_rate=0.7, avg_win=100,
  avg_loss=50)` inputs that happen to satisfy the formula — a
  textbook case of test inputs not matching production inputs. The
  binary form was rewritten to a **per-trade NAV cap** during Phase 2
  integration (PR #163), keeping `win_rate` / `avg_win` / `avg_loss`
  in the signature for forward-compatibility but not consuming them.
  This is the practitioner's reading of "half-Kelly" as a sizing
  ceiling rather than an edge-derived `f*`.

- **Continuous Kelly (`f* = μ/σ²`) with per-trade EV/variance
  estimates.** The "right" Kelly for size-on-edge, but requires a
  reliable per-trade σ that the current EV pipeline doesn't surface
  (the ranker outputs `ev_dollars` and `prob_profit`, not a
  distribution variance). Deferred until a per-trade variance
  estimate is available; the cap form is the conservative
  placeholder.

- **`_C4_VOL_SPIKE_SCENARIO` added to
  `stress_testing.HYPOTHETICAL_SCENARIOS`.** The natural home would
  be the existing scenario library. Rejected for Phase 1 because the
  prompt's `<rule_outs>` forbade modifying `stress_testing.py`;
  the scenario lives module-local in `portfolio_risk_gates.py` for
  now. Relocating it is a trivial follow-on when a future task wants
  it in the standard library.

- **Split R8 into R8 (stress) + R9 (dealer regime).** Considered
  for explicit accounting symmetry with R7. Rejected because R8 has
  the same shape as R6 — one rule, two trigger conditions, one
  downgrade verdict — and splitting would break the parallel with R6.
  Distinct `verdict_reason` per trigger (`stress_breach` vs
  `short_gamma_regime`) carries the per-trigger fact into the audit
  log without inflating the rule count. **Note (2026-05-27):** R9
  was later added in the B2 closure for `check_sector_cap`, NOT as
  a split of R8 — a separate gate with its own concern. R8 keeps
  its dual-trigger shape.

- **Opt-in `require_portfolio_risk_gates` flag separate from
  `require_ev_authority`.** Considered for backwards-compat: gate
  the D17 hard-blocks behind a fresh flag so existing
  `require_ev_authority=True` callers keep their old refuse-set.
  Rejected because the D16 hardening pass already established
  `require_ev_authority` as the "production-strict" mode marker,
  and adding a second flag splits the strict surface into two
  knobs with no clear use case for one-without-the-other. D17 hard-
  blocks fire under the same `require_ev_authority=True` switch.

- **Static NAV (use `initial_capital` always, not live mark-to-
  market).** Considered for determinism: every gate evaluation uses
  the same NAV regardless of P&L. Rejected because gates should
  self-adjust in drawdowns (a book down 20% should be more
  conservative, not less) — that's the *point* of NAV-relative
  caps. The static path is preserved as the `connector=None` and
  `connector raised` fallbacks, with explicit `nav_source`
  fingerprints (`"static_fallback"` / `"static_fallback_connector_error"`)
  in the audit log so any auditor can spot a gate run that landed
  on the static value.

- **Ratchet NAV (only update upward).** A floor that never lets
  recent drawdown raise the cap. Rejected for path-dependence: the
  same book at the same time would get different gate decisions
  depending on its peak-NAV history. Live mark-to-market is the
  symmetric, path-independent choice.

- **Wire D17 into `EVEngine.evaluate` directly (compute portfolio
  context as a multiplier).** Would put portfolio-level risk inside
  the EV scoring path. Rejected because it would violate the
  downgrade-only contract: a portfolio multiplier could in principle
  scale `ev_dollars` upward (against the §2 invariant). Keeping
  D17 as a tracker refuse / dossier downgrade preserves the §2
  contract — D17 can only refuse or downgrade, never rescue.

**Pinned by:** `engine/portfolio_risk_gates.py` (the library —
`check_sector_cap`, `check_portfolio_delta`, `check_kelly_size`,
`check_var`, `check_stress_scenario`, `check_dealer_regime`,
`take_snapshot`, locked-defaults constants, `_C4_VOL_SPIKE_SCENARIO`);
`engine/wheel_tracker.py` (`_evaluate_d17_hard_blocks`,
`_compute_live_nav`, `min_nav_for_trading` constructor param);
`engine/candidate_dossier.py` (`EnginePhaseReviewer` rules R7 + R8,
`_build_candidate_dict`); `tests/test_portfolio_risk_gates.py`
(per-gate unit tests against the locked defaults);
`tests/test_authority_hardening.py::TestD17HardBlocks` (tracker
hard-block end-to-end + `nav_source` fingerprint);
`tests/test_dossier_invariant.py` (R7 + R8 reviewer tests);
`tests/test_ev_authority_log_schema.py` (the four D17 hard-block
reject shapes in `_VALID_SHAPES`); `docs/USAGE_TEST_LEDGER.md` S15
(the originating finding) and S21 (end-to-end confirm-fixed at
$1M pro-account NAV).

---

## How to add a decision

1. Number it (`D11`, `D12`, …) sequentially. Don't reuse numbers.
2. Lead with **Decision** (one paragraph), then **Why**, then
   **Rejected alternatives**, then **Pinned by**.
3. Pin to code or tests, not to other docs alone — docs drift,
   tests block merges.
4. If a later decision overrides an earlier one, update the earlier
   entry with `**SUPERSEDED by D<N>**` rather than deleting it. The
   history is part of the value.
