---
id: S24
title: Multi-strategy book composition ($500k wheel + CC + strangle scan)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** S21 explicitly flagged multi-strategy book composition
as the next pro-trader-lens scenario. Build a $500k NAV book that
holds short puts + covered calls (post-assignment) simultaneously,
attach a `PortfolioContext` and run `EnginePhaseReviewer` on a fresh
candidate, then probe the D17 hard-block + dossier soft-warn gates
directly to characterise how they compose across strategies (option-
only vs option+stock). Try the strangle ranker
(`WheelRunner.rank_strangles_by_ev`, S14 / #118) on a third ticker
to confirm whether/how strangles can join the book.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20` (data cutoff), `WheelTracker(initial_capital=
$500_000, require_ev_authority=False)`. Non-strict mode keeps the
EV-authority token plumbing out of the way so the gate-composition
math is the only thing under test. Driver under `%TEMP%\s24\`,
not committed.

The book is built via the public surfaces:

1. `wr.rank_candidates_by_ev(['MRK'], delta_target=0.20)` →
   `tracker.open_short_put(...)` (Health Care, MRK $105.50, 35 DTE).
2. `wr.rank_candidates_by_ev(['KO'], delta_target=0.20)` →
   `tracker.open_short_put(...)` (Consumer Staples, KO $71.00).
3. `tracker.handle_put_assignment('KO', as_of, spot=$70.50)` to
   transition KO into `STOCK_OWNED`, then
   `wr.rank_covered_calls_by_ev('KO', shares_held=100, ...)` →
   `tracker.open_covered_call(...)` (CC at KO $80, DTE 49) to land
   the position in `COVERED_CALL`.
4. `wr.rank_strangles_by_ev('NVDA', ...)` for the strangle scan —
   pure ranking, no tracker call (no `open_strangle` surface
   exists; S14 finding).
5. `take_snapshot(tracker.positions, today=AS_OF)` →
   `PortfolioContext(held_option_positions, stock_holdings,
   spot_prices, nav)`.
6. Construct a `CandidateDossier` for a fresh CAT candidate
   (`rank_candidates_by_ev(['CAT'])`), attach a synthetic
   `ChartContext(visible_price=spot, screenshot_path=Path('.../
   synthetic-s24.png'), error='')` so R2 / R3 stay silent, then
   call `EnginePhaseReviewer.review(...)` with and without the
   `PortfolioContext`.
7. Direct probes of `check_var`, `check_stress_scenario`,
   `check_sector_cap`, `check_portfolio_delta` to read out the
   gate math on the multi-strategy book.

`returns_data` for R7 is built ad-hoc from connector OHLCV — 252-day
daily log returns per ticker (MRK + KO + CAT) — so R7 has actual
inputs rather than the missing-data skip from S21.

**Path.** `take_snapshot` at `engine/portfolio_risk_gates.py:177`
maps each `PositionState` to the snapshot shape:

  - `SHORT_PUT` → one option dict (short put leg).
  - `STOCK_OWNED` → one stock holding (no option leg).
  - `COVERED_CALL` → one option dict (short call leg) + one stock
    holding.

`EnginePhaseReviewer.review` runs R1 → R3 → R4 → R5 → R6 → R7 → R8
in sequence per `engine/candidate_dossier.py:197-353`. R7 and R8
only fire when `portfolio_context is not None and verdict ==
"proceed"`. The C4 vol-spike scenario (-10% spot + 30% IV) is
defined at `engine/portfolio_risk_gates.py:118`. The sector cap
(`check_sector_cap`, line 271) ignores stock holdings by design
(option-side concentration only); the delta cap
(`check_portfolio_delta`, line 344) includes the stock-leg
delta-dollars via the `stock_holdings` argument.

**Status.** Done. **Verdict: gate composition behaves correctly
across strategies (3 of 5 questions cleanly answered, 1 negative
finding on R8 stress, 1 pre-existing S14 strangle-integration
gap re-confirmed). One major methodology finding surfaced — the
dev-box's `sys.path` discovery silently picks up an older primary
clone, so a driver run from `%TEMP%` without explicit
`sys.path.insert(0, worktree)` evaluates against the wrong engine
SHA. Drivers fixed; S22 / S23 re-validated bit-identically against
the worktree.**

**Findings:**

- **Q1 — `take_snapshot` correctly translates the 3-state book.**
  Live driver output:

  ```
  option_positions (2):
    {'symbol': 'MRK', 'option_type': 'put',  'strike': 105.5, 'dte': 35, 'iv': 0.3128, 'contracts': 1, 'is_short': True}
    {'symbol': 'KO',  'option_type': 'call', 'strike': 80.0,  'dte': 49, 'iv': 0.2127, 'contracts': 1, 'is_short': True}
  stock_holdings (1):
    ('KO', 100)
  ```

  The covered-call position decomposes into TWO dict entries
  (option + stock); the SHORT_PUT into one; the STOCK_OWNED
  (transitional) into one stock entry. **Snapshot fidelity is
  correct across all three states. Logged as a positive.**

- **Q2 — R7 (VaR) correctly fires with real returns_data.** Driver
  built 252-day log returns per held ticker (MRK / KO) plus the
  candidate (CAT) from `conn.get_ohlcv(...)`. Direct probe:

  ```
  check_var (no returns_data):  passed=True reason='missing_data' skipped
  check_var (with returns_data): passed=True
    var_dollars=$4,344.88  cvar_dollars=$5,448.66  var_pct=0.87%  var_limit_pct=5%
  ```

  **R7 path is wired end-to-end and passes the actual VaR math
  when the inputs are supplied** — the limit was 5% NAV ($25k);
  the actual portfolio 30-day VaR_95 is 0.87% NAV. **Closes S21's
  Q2-shaped follow-up** ("a future Sn could exercise R7 with real
  returns data"). The 2-position book is well under the cap, but
  the math fires and would downgrade `proceed → review` on a
  book whose VaR breached 5%. **Logged as a positive.** **Open
  follow-up:** no upstream caller assembles `returns_data` for the
  `PortfolioContext` automatically — the dossier-builder / tracker
  hand-off has no `returns_data` plumbing; the operator-level
  caller would need to build it. Mirrors S21's "PortfolioContext
  with real returns" hand-off.

- **Q3 — R8 (C4 vol-spike stress) is benign on a $500k 2-position
  book.** Driver output:

  ```
  check_stress_scenario (C4 Vol Spike: -10% spot + 30% IV):
    passed=True  portfolio_pnl_dollars=-$5,266.72  drawdown_pct=1.05%  drawdown_limit_pct=8%
  ```

  Same shape S21 saw at $1M / 2 positions (drawdown 0.56% vs 8%
  cap). **R8 needs a larger or more concentrated book to fire.**
  Mechanically wired, contractually correct, but never bites at
  this scale. **Logged.** S21's "exercise R8 with a richer book"
  follow-up is still open; S24 corroborates it.

- **Q4 — Strangle ranker produces candidates the tracker cannot
  ever accept.** `wr.rank_strangles_by_ev('NVDA', target_dtes=
  (35,49), target_deltas=(0.20,0.15))` returned 4 candidates:

  ```
     put_strike  call_strike  dte  ev_dollars timing_recommendation    timing_phase
  0       156.0        197.0   35    -1360.22           conditional  post_expansion
  1       159.5        192.5   35    -1442.91           conditional  post_expansion
  2       153.5        202.0   49    -1734.34           conditional  post_expansion
  3       157.5        196.5   49    -1882.32           conditional  post_expansion
  ```

  All 4 are negative composed EV (timing gate "conditional", not
  "avoid"), and `WheelTracker` exposes **no** `open_strangle`,
  `SHORT_STRANGLE` state, or any way to add a strangle leg-pair
  to the book. Grep on the tracker file confirms zero matches for
  `strangle | SHORT_STRANGLE | open_strangle`. **The S14 / #118
  finding is unresolved at the tracker layer:** strangles are
  rank-only — they can score, they cannot be tracked, sized
  against NAV, or fed into R7 / R8 via `take_snapshot`. **A
  tradeable strategy with no portfolio integration.** Re-logging.

- **Q5a — `check_sector_cap` correctly ignores stock holdings.**
  Live probe:

  ```
  check_sector_cap (CAT in Industrials, $65,000 notional):
    passed=True post_open_sector_pct=0.13 sector_limit=0.25
  ```

  Post-open Industrials = 13% of NAV; the existing KO option
  (Consumer Staples) and MRK option (Health Care) don't show in
  Industrials. **And critically, the KO 100 shares from the
  assignment also don't contribute to the Industrials check** —
  matching the docstring at `engine/portfolio_risk_gates.py:286-
  293` ("stock holdings don't contribute to the option-side
  sector exposure"). **Design call** — the wheel mental model is
  "options drive risk concentration; the assigned stock is parked
  capital." A pro-trader who'd rather see *total-position* sector
  exposure (option + stock) would want a different cap. Mostly
  a transparency note. **Logged as expected behavior, with the
  design caveat.**

- **Q5b — `check_portfolio_delta` DOES include the stock holding's
  delta.** Live probe (candidate CAT 25-delta @ $625.50 against the
  multi-strategy book):

  ```
  check_portfolio_delta (CAT 25-delta @ $625.50 + stock_holdings):
    passed=False  reason='portfolio_delta_breach'
    current_portfolio_delta_dollars=$7,983.29  post_open_delta_dollars=$37,944.55  delta_cap_dollars=$1,500.00
  ```

  Current book delta = **$7,983**, dominated by the KO stock leg
  (100 shares × $74.69 spot ≈ $7,469); the short put on MRK and
  the short call on KO contribute roughly the remaining $514
  combined. **The stock holding visibly bites into the
  portfolio-delta cap** — exactly opposite of the sector cap's
  treatment. **Both behaviors are correct per their docstrings;
  the cross-gate asymmetry (sector option-only, delta total)
  is a documented design call.** **Logged.**

  Adding CAT (+$16,815 of delta-dollars per a 25-delta at $625.50
  × 100 × ~0.25) would push the book to $37,945 — 25× over the
  $1,500 cap. The delta cap remains the dominant binding
  constraint at $500k NAV (`300 × ($500k / $100k) = $1,500`),
  echoing S21.

- **`EnginePhaseReviewer` verdict-delta WITH vs WITHOUT
  `PortfolioContext`.** Same multi-strategy book, same CAT
  candidate, same synthetic chart (R2/R3 silent):

  ```
  WITHOUT PortfolioContext:
    verdict='proceed' reason='ev_above_threshold'
  WITH PortfolioContext (multi-strategy book attached):
    verdict='proceed' reason='ev_above_threshold'
    note: R7: VaR check skipped (no_correlation_matrix_or_returns_data)
  ```

  No verdict change — the `returns_data` was assembled in the
  *direct* R7 probe (Q2 above) but the `PortfolioContext` passed
  to `EnginePhaseReviewer` did **not** carry it through. **R7
  silently skipped despite the dossier path having a context
  attached.** This is the second face of the "no upstream
  `returns_data` plumbing" finding above — the field is
  optional, the reviewer doesn't fabricate it, and the
  downgrade-only contract means absent data = silent pass.
  **R7's downgrade-only-when-fired contract held.** **Logged.**

- **(F-METH-1) Dev-box `sys.path` discovery silently picks up
  the primary clone, not the worktree.** Highest-leverage finding
  from this run. A driver run as `python %TEMP%\s24\driver.py`
  with cwd at the project directory ends up with this `sys.path`:

  ```
  ''                                                       (cwd-empty - effective)
  ...
  'C:\\Users\\merty\\Desktop\\Local AI Agent'              (user-site)
  'C:\\Users\\merty\\Desktop\\smart-wheel-engine'          (older primary clone)
  ```

  When the script is invoked by path (not `-c`), Python sets
  `sys.path[0]` to the script's directory (`%TEMP%\s24\`), which
  is not the project. The cwd is *not* automatically on the path.
  But the user-site `pth` files add `C:\Users\merty\Desktop\
  smart-wheel-engine` — **a separate, older clone** currently at
  `cd16443` (pre-D17, diverged 776 lines on `portfolio_risk_gates`
  alone from `origin/main` at `86b917c`). The driver imports
  `engine.portfolio_risk_gates` **from that older clone**,
  not from the Terminal A worktree.

  **My S22 and S23 drivers ran against the older primary clone
  silently.** I noticed on S24 only because the older clone
  doesn't *have* `portfolio_risk_gates.py` (pre-D17), so the
  import failed and the masking became visible. **For S22 and
  S23 the import succeeded against the older code, and the
  driver produced findings I logged against the wrong SHA.**

  **Re-validation: I re-ran both S22 and S23 with an explicit
  `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
  prepended.** Both produced **bit-identical** numbers — every
  EV, every drop, every IV value — to the original primary-clone
  runs. The relevant code paths (`suggest_rolls`,
  `rank_covered_calls_by_ev`, `get_next_earnings`,
  `get_fundamentals`, `event_gate.is_blocked`, the IV-snapshot
  fallback at `wheel_runner.py:813-816`) are common to both
  SHAs. So the S22 and S23 findings hold against `origin/main`.
  **But the masking risk is real for any future Sn that touches
  D17 surfaces**, which would diverge silently.

  **Mitigations (fix sketch for future Sn templates):**

  ```python
  # Top of every %TEMP%\sNN\driver.py:
  import sys
  sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")
  ```

  Or, equivalently, run as `python -m runpy` with explicit
  module path. **The longer-term fix is operator-level** —
  `pip uninstall` the older primary clone or remove its `.pth`
  file from the user-site, but that's outside the scope of a
  usage-test PR. **Logged as the highest-leverage methodology
  finding in S24.**

- **(F-METH-2) `take_snapshot` falls back to `date.today()` when
  `today=None`.** At `engine/portfolio_risk_gates.py:217-218`:

  ```python
  if today is None:
      today = _date_cls.today()
  ```

  My S24 driver passed `today=AS_OF` explicitly, avoiding the
  footgun. **A naive caller who forgets to inject `today` would
  compute DTEs against the system date** — exactly the
  `date.today()` smell the prompt flagged from audit #166. Not
  yet a live bug (the current callers in `dossier_builder` and
  the tracker do inject `today`), but the latent surface remains.
  **Logged.**

- **§2 verified.** Every put opened, every CC ranked, every CAT
  dossier reviewed routed through `EVEngine.evaluate` upstream
  (via `rank_candidates_by_ev` / `rank_covered_calls_by_ev` /
  `rank_strangles_by_ev`). R7 / R8 are downgrade-only soft-warns;
  sector / delta caps are post-EV hard-blocks (live in strict mode
  per S21, mathematically observable here in non-strict mode by
  direct call). **No tradeable verdict surfaced without
  `EVEngine.evaluate` on the option leg.** The strangle scan
  returned candidates but they're not tradeable in the tracker —
  the §2 question doesn't even arise. **Logged as a positive.**

**Verdict.**

- **Multi-strategy book composition works mechanically:**
  `take_snapshot` correctly decomposes SHORT_PUT / STOCK_OWNED /
  COVERED_CALL into the option-position + stock-holding shape
  `PortfolioContext` expects. R7 / R8 / sector cap / delta cap
  all run on the composed book without special-casing.

- **R7 (VaR) is fully wired and fires with real
  `returns_data`** — closes S21's "next Sn could exercise R7
  with real data" hand-off. **But no upstream caller in the
  decision-layer assembles `returns_data` for the
  `PortfolioContext` automatically**, so the reviewer path
  silently skips R7 in the default integration. Same gap S21
  noted; the proper fix is a builder-side helper.

- **R8 (stress) is benign at $500k / 2 positions** — same
  pattern as S21 at $1M / 2 positions. Needs a larger or
  concentrated book to bite.

- **Sector cap option-only, delta cap option+stock.** Documented
  design call; cross-gate asymmetry. A trader who wants total-
  position sector concentration would need a different cap.

- **The strangle ranker is rank-only.** S14 / #118 closed the EV
  authority for strangles (`rank_strangles_by_ev` exists and
  routes through `EVEngine.evaluate` per leg), but
  `WheelTracker` has no surface to *open* a strangle. **A
  tradeable strategy with no tracker integration** — the natural
  next PR.

- **The dev-box `sys.path` discovery silently shadows the
  worktree's engine with an older primary-clone version.**
  Future drivers must explicitly prepend the worktree to
  `sys.path` to evaluate against the intended SHA. S22 / S23
  re-validated bit-identically against the worktree, so their
  findings hold — but the failure mode is silent and could mask
  serious findings on D17-era surfaces.

**AI handoff.**

- **Fix sketch for R7 / R8 reachability — `PortfolioContext`
  builder helper.** Today the reviewer's R7 / R8 paths require
  the caller to populate `returns_data` /
  `dealer_regime_by_ticker` / `volatilities` on the context.
  The dossier builder doesn't do this. Proposed helper at
  `engine/portfolio_risk_gates.py` (or a new
  `engine/portfolio_context_builder.py`):

  ```python
  def build_portfolio_context_from_tracker(
      tracker: "WheelTracker",
      *,
      today: date,
      connector,
      returns_lookback_days: int = 252,
  ) -> PortfolioContext:
      snap = take_snapshot(tracker.positions, today=today)
      spot_prices = {tk: float(connector.get_ohlcv(tk)
                                       [connector.get_ohlcv(tk).index
                                        <= pd.Timestamp(today)]
                                       ["close"].iloc[-1])
                     for tk in tracker.positions}
      # 252-day log returns per held ticker for R7's historical path.
      returns_data = {}
      for tk in tracker.positions:
          o = connector.get_ohlcv(tk)
          o = o[o.index <= pd.Timestamp(today)]
          if len(o) >= returns_lookback_days:
              returns_data[tk] = np.log(o["close"]).diff().dropna() \
                                   .iloc[-returns_lookback_days:].values
      return PortfolioContext(
          held_option_positions=snap.option_positions,
          stock_holdings=snap.stock_holdings,
          spot_prices=spot_prices,
          nav=tracker.cash + sum(s * spot_prices.get(t, 0)
                                 for t, s in snap.stock_holdings),
          returns_data=returns_data,
      )
  ```

  The dossier builder calls this once per ranking pass and
  attaches the result. R7 fires on real data; R8 stress
  unchanged (no extra input needed); `dealer_regime_by_ticker`
  remains optional (when present, R8's regime branch lights up).
  **Decision-layer surface — needs a decision-layer lock claim
  and regression test that the reviewer's R7 path consumes the
  builder-supplied `returns_data` end-to-end.** Out of scope
  for this Sn.

- **Fix sketch for the strangle tracker integration.** Add a
  `PositionState.SHORT_STRANGLE`, an `open_strangle` method on
  `WheelTracker` that opens two legs simultaneously (put + call,
  same ticker, same expiry, two strikes, two premiums), and
  extend `take_snapshot` to emit one option_position dict per
  leg under the same ticker (the snapshot shape already supports
  N option dicts per ticker, so no schema change). The delta /
  sector caps already aggregate by ticker, so they'd see the
  strangle as two-legged automatically. **Pre-flight on §2:**
  `rank_strangles_by_ev` already routes both legs through
  `EVEngine.evaluate` (per the docstring at
  `engine/wheel_runner.py:2106-2113`); the tracker just needs a
  channel to receive the result. **Larger surface; needs its
  own claim, a `WheelTracker.suggest_strangle_rolls` parallel,
  and several regression tests.** Out of scope for this Sn.

- **Fix sketch for the `sys.path` discovery footgun.** Two paths:
  (a) operator-level — remove the `.pth` file or `pip uninstall`
  the old primary clone (out of scope for any Sn); (b) campaign-
  level — every `%TEMP%\sNN\driver.py` template should start with
  `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`.
  Add this to the documented usage-test driver template in
  `docs/PARALLEL_SESSIONS.md` (or wherever the template lives).
  **The memory `[[dev-box-working-tree-is-shared]]` covers the
  git-state version of this hazard — this finding extends it to
  Python's import system.** Out of scope for this Sn entry as a
  doc edit, but worth a one-line update to that memory.

- **Fix sketch for `take_snapshot`'s `date.today()` default.**
  The default is a footgun for callers who forget to inject
  `today` — the function then computes DTEs against the system
  clock, not the data's `as_of`. Two options: (a) raise on
  `today=None` to force injection (breaking change for the
  current callers); (b) accept `today=None` but document loudly
  that it defaults to `date.today()` and warn callers to inject
  in any PIT-sensitive code path. **Tracked under the existing
  audit #166 `date.today()` thread.** Out of scope here.

**Methodology debt.**

- **F-METH-1 (sys.path) shadowed S22 / S23 originally.** I
  re-validated both bit-identically against the worktree once
  the issue was found; the findings hold. **For any future Sn
  that touches D17 / post-D17 surfaces, the masking would be
  destructive** because the primary clone diverges by 776 lines
  on `portfolio_risk_gates.py`. **Highest-priority for any
  template / docs update.**

- **R7 was exercised with synthetic `returns_data` built ad-hoc
  in the driver, not via a production code path.** The
  *reviewer* path through `PortfolioContext` (without
  `returns_data`) silently skipped R7. So **R7 is wired but
  not reachable in the standard integration** without the
  builder helper sketched above. **Logged.**

- **R8 stress remains untriggerable at any tested book size**
  (2 positions at $500k or $1M, drawdown 1.05% / 0.56% vs 8%
  cap). To exercise R8, future Sn needs either:
  (a) a 10-15 position book in concentrated sectors, OR
  (b) directly testing with a hand-crafted
  `dealer_regime_by_ticker={'CAT': 'short_gamma_amplifying'}`
  to fire R8's dealer-regime branch.

- **Strangle integration is the natural next PR** — fix sketch
  above. Not in scope for a usage-test Sn.

- **Ruled out per the campaign constraints:** Theta provider
  (other agent active), strict-mode D17 token plumbing (S21
  covered it; orthogonal to the gate-composition math under
  test here), HMM regime (no persisted model), decision-layer
  code change (S24 found surfaces, did not fix), dashboard
  surface (read-only on engine only).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `take_snapshot` is observability-only; no EV-bypass surface.
  - Qualitative verdict: match — built MRK short-put + KO short-put + KO put-assignment + (would-be CC) book on a $500k WheelTracker. `take_snapshot(tracker.positions, today=date(2026,3,20))` returned `option_positions=1` + `stock_holdings=1` (one option leg remaining + the KO 100-share assignment), confirming the 3-state schema mapping the original entry documented. **`WheelTracker.open_strangle` still does not exist** — the S14/S24 finding "a tradeable strategy with no tracker integration" is **still open** on `main`.
  - Numerical drift > 5%: not applicable — methodology Sn; not a numerical drift candidate.
  - Notes: F-METH-1 (`sys.path` discovery silently picks up older primary clone) — driver pinned to `sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")` per the [[sys-path-worktree-shadow]] memory; verified `import engine.portfolio_risk_gates` lands on the worktree's copy.
