# Usage Test Ledger

This file tracks end-to-end *usage* tests — sessions where the engine
is exercised as a real trader would use it, not by the unit /
integration test suite. Each scenario has a purpose, a setup, the
bugs and gaps it surfaced, and the PRs (if any) that closed them.

Companion to:

- `PROJECT_STATE.md` — current authoritative state.
- `ROADMAP.md` — intended next work.
- `CHANGELOG.md` — shipped per-PR detail.
- `TESTING.md` — the *unit / integration* test taxonomy. This file
  is the *usage* axis; that one is the *code* axis.

## How to update this

When a usage-test scenario completes:

1. Append an entry under the appropriate section with: name,
   purpose, setup, status, key findings, and follow-ups.
2. For each finding, link the PR that fixed it (e.g. `#102`) or
   tag it `**logged**` if not yet fixed.
3. Move scenarios between sections as their status changes
   (Candidate → Queued → In flight → Completed).
4. Keep findings inline under their scenario. Cross-cutting
   findings that recur across scenarios can be repeated; do not
   maintain a parallel flat index — it will drift.

The aim is operational, not historical: a fresh agent should be
able to read this and know which surfaces of the product have been
exercised and which have not.

---

## 1. Completed

### S1 — Single-snapshot trader session

**Purpose.** Exercise the morning-scan → dossier → sizing path as a
retail wheel trader would, top-down across the SP500.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, offline charts
(`FilesystemChartProvider`), 40-name diversified watchlist,
`as_of=2026-03-20`, $150k account, 35-DTE / 25-delta puts.

**Status.** Done. One critical bug fixed
(`#102` — dividend-yield normalization). Other findings logged.

**Findings:**

- **Dividend-yield normalization bug** (`wheel_runner.py` ~ line 655).
  Sub-1% yields skipped the `> 1.0` guard and reached BSM as a
  whole-number decimal (`0.87` used as 87% q). Corrupted the
  delta → strike solve and the synthetic premium across ~92 of 410
  priced names — MSFT, COST, AAPL surfaced as positive-EV when truly
  negative. Fixed in **`#102`** (merged `afee837`).
- **`Δ` (U+0394) Unicode crash** in `candidate_dossier.py`'s R3
  review note — crashes Windows cp1252 console on print / log.
  **Logged.**
- **Silent drops** — `rank_candidates_by_ev` returns only
  survivors; no diagnostic when a name is gated out (earnings,
  history, chain quality). **Logged.**
- **`as_of` footgun** — defaults to today; pairs stale Bloomberg
  prices with current-date event timing. **Logged.**
- **R4 reviewer rule effectively dead** in the standard ranker
  path — needs a `phase` field the ranker never emits. **Logged.**
- **Committee / ranker contract mismatch** — the advisor committee
  re-ranks at 45-DTE / 0.30-delta while the ranker and dossier use
  35-DTE / 0.25-delta. Two different contracts on two endpoints.
  **Logged.**
- **No `ev_raw` exposed** in the ranker output despite being a
  core EV-engine field. **Logged.**
- **No return-on-capital column / no account-size input** — the
  ranker optimizes absolute EV/day, structurally biased to
  expensive names. **Logged.** Addressed in part by S4 (queued).
- **Regime (HMM) multiplier unlabeled** — silently cuts EV
  50–80 % on some names with no surfaced regime. **Logged.**

### S2 — Multi-day rolling wheel campaign (4 weeks)

**Purpose.** Exercise the time dimension — managing a real book
across days: open, age, profit-take, hold, roll, accept assignment.

**Setup.** Bloomberg, offline charts, $150k account, 35-DTE /
25-delta entries, profit-take ≥ 50 %, max 25 % per name, 5-snapshot
window (`as_of` advanced 5 trading days per step) ending at the data
cutoff `2026-03-20`. Positions tracked in
`engine.wheel_tracker.WheelTracker`.

**Status.** Done. Dividend fix validated end-to-end in real flow
(4/4 closes profitable; PG loss landed honestly in the advertised
9 % tail). All management-layer findings logged.

**Findings:**

- **No persistence on `WheelTracker`.** No `to_dict` / `from_dict` /
  `to_json` / `save` / `load`. Closing a Python session loses the
  book. **Logged.**
- **No management-workflow methods.** No `suggest_rolls`,
  `suggest_actions`, `book_snapshot`, `available_buying_power`. The
  tracker is a mechanics layer (`open_short_put` / `close_short_put`
  / `roll_put` / `handle_*_assignment` / `mark_to_market`) without a
  management layer on top. The single biggest gap; addressed by S3
  (in flight).
- **`mark_to_market` silent IV-staleness.** Falls back to the
  position's entry IV when `current_ivs` is not passed — no warning,
  no `as_of` IV plumbing from the connector. **Logged.**
- **`tracker.cash` overstates buying power for CSPs.** Open credits
  the premium but does not reserve the strike collateral. Correct as
  brokerage cash; wrong as deployable capital. Workaround:
  `available = cash − Σ(strike × 100)`. **Logged.**
- **`get_performance_summary` is closed-positions-only.** Returns
  an empty DataFrame mid-campaign. No companion current-book
  snapshot. **Logged.**
- **Earnings-window-drift not surfaced** for open positions. A
  trade opened 50 days before earnings becomes "within window"
  silently as time advances. **Logged.**
- **Same-day close-and-reopen feels strange.** A name closed at
  profit-take can re-rank #1 in the same step and be reopened with
  no cooldown. Internally consistent (different contract entirely)
  but a UX surface a real trader would want. **Logged.**
- **Drop-reason silence carries into the rolling case** — same as
  S1, but more visible when a name disappears between snapshots.
  **Logged.**

---

## 2. In flight

### S3 — Build `WheelTracker.suggest_rolls(...)`

**Purpose.** Close S2's headline management-layer gap. Given a
challenged short put, return ranked candidate rolls computed through
`EVEngine.evaluate` (§2 invariant intact — uses the EV authority,
does not bypass it).

**Setup.** Branch `claude/wheel-tracker-suggest-rolls`. Scope is
`engine/wheel_tracker.py` + tests; no edits to `ev_engine.py`,
`wheel_runner.py`, or `candidate_dossier.py`. Short-put rolls only
this round (covered-call rolls deferred to a follow-up). Verifies
ruff + full `pytest tests/` + one concrete live demo (the PG
position from S2). PR opens; the human reviews before merge.

**Status.** Prompt written; executor in flight. No PR number yet.

---

## 3. Queued

### S4 — Account-size-constrained book selection

**Purpose.** Force a return-on-capital lens by setting a realistic
small account ($50k retail) as a hard constraint. Directly surfaces
the no-account-size / no-ROC gap from S1.

**Setup.** Bloomberg, offline charts, $50k account, sizing must
respect collateral. Smallest scope of the queued angles.

### S5 — Live MCP chart in the loop

**Purpose.** Exercise the just-shipped TradingView MCP integration
(Stages 1–3 + live-verify fixes + `tv quote` price wiring) in a real
dossier flow.

**Setup.** TradingView Desktop running with CDP on
`localhost:9222`; tradingview-mcp `tv` CLI on PATH;
`SWE_USE_MCP_CHART=1`. Unlocks the `proceed` verdict that offline
sessions cannot reach. Operator setup required.

### S6 — Theta provider with real chains

**Purpose.** Exercise actual chain-quoted premiums vs the synthetic
BSM premium Bloomberg uses. `edge_vs_fair` is structurally 0 on
Bloomberg; S1 flagged this as the biggest missing signal.

**Setup.** `SWE_DATA_PROVIDER=theta` with the Theta Terminal running
on `127.0.0.1:25503`. Operator setup required.

---

## 4. Candidate (not yet selected)

Worth running when scope and time allow:

- **Wheel-cycle-to-completion** — an 8–12-week campaign covering
  at least one full cycle (short put → assign → covered call →
  assign-back-to-cash). Exercises `handle_put_assignment`,
  `handle_call_assignment`, `open_covered_call`, `roll_call`.
- **Regime-shift stress** — anchor a campaign across a historical
  VIX spike. Observe whether the dealer / HMM multipliers and the
  event / stress gates respond as advertised.
- **Strangle timing-gated strategy** — the `engine/strangle_timing.py`
  path (CLAUDE.md §4 timing-gated strategy). Not yet exercised.
- **Advisor committee deep-dive** — only briefly hit in S1. Run
  all four advisors across a 10-trade book; do they disagree
  usefully? Do their verdicts move actual decisions?
- **Dashboard end-to-end** — the Next.js app under `dashboard/`
  not exercised at all. Would surface UX issues the API alone
  cannot.
- **TradingView webhook ingest** — `POST /api/tv/webhook` →
  ring buffer → `/api/tv/ranked` / `/api/tv/dossier`. The
  Pine-signal-driven entry path. Cold path right now.
- **Adversarial / gate stress** — deliberately try to break each
  gate (history, event, chain quality, stress residual,
  survivorship). Confirm fail-closed.
- **News sentiment downgrade path** — `engine/news_sentiment.py`
  is the only news-stack module on the EV path. Validate it
  actually downgrades when sentiment turns bad.
