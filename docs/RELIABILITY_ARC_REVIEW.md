# Reliability arc review — S18 / S19 / S20 (2026-05-25)

**Reviewer:** Terminal B, fresh session, no campaign context.
**Scope:** PRs #153, #157, #162, #164 — all merged to `main`.
**`origin/main` SHA at review start:** `d24f041e8447d3242055701df8dd48fb28d26df0`.

| PR | Subject | Merge SHA | Merged |
|---|---|---|---|
| #153 | S18 load / scale stress | `7ba1f33` | 2026-05-23 |
| #157 | S19 failure-mode chaos | `821c88b` | 2026-05-23 |
| #162 | S20 `engine_api.py` concurrency (original) | `9de8ec7` | 2026-05-24 |
| #164 | S20 v5 backfill amendment (race vectors clean) | `cf59636` | 2026-05-24 |

This review re-verifies the **load-bearing claims** of the reliability
arc against `origin/main` as it stands today. The §2 invariant —
"no tradeable candidate bypasses `EVEngine.evaluate`" (CLAUDE.md §2) —
is the through-line. The four claims under review:

- **C1** — S19's headline §2 finding (C7b): `ev_dollars=+inf` bypasses
  reviewer rules R1 and R5 if it reaches `build_dossiers`. Refuted on
  the network surface by S20 (G3).
- **C2** — S20's production-readiness ceiling: `request_queue_size = 5`
  (stdlib default) caps `engine_api.py`'s concurrent-accept budget.
- **C3** — S20 v5 backfill race-vector mechanisms (G1 ring-trim, G2 GET
  slice, G4 lock-free nonce, G5 per-thread isolation, G8 constant-time
  HMAC).
- **C4** — cross-arc coherency: the IV-PIT bug (PR #178 finding F8,
  fixed in #179) landed *after* S18-S20 and is orthogonal to their
  verdicts.

## Tally

- CONFIRMED:           **2**
- CONFIRMED-WITH-NOTE: **2**
- CONCERN:             **0**
- §2 BREACH:           **0**

---

## C1 — C7b mechanism (`ev_dollars=+inf` bypass)

**Verdict:** **CONFIRMED-WITH-NOTE.** Mechanism reproduces today
exactly as S19 describes. Line numbers in the S19 doc have drifted by
+35 / +35 since the D17 R7/R8 additions; the *body* of R1 and R5 is
unchanged.

**Code today** (`git show origin/main:engine/candidate_dossier.py`):

| Rule | S19-cited line | Line on `d24f041` | Code |
|---|---|---|---|
| R1 (negative-EV block) | `:167` | **`:202`** | `if ev < 0:` |
| R5 (proceed threshold) | `:206` | **`:241`** | `if ev >= self.min_proceed_ev:` |

```python
# engine/candidate_dossier.py:197-204 (d24f041)
def review(self, dossier: CandidateDossier) -> tuple[Verdict, str, list[str]]:
    notes: list[str] = []
    ev = dossier.ev_dollars

    # Rule 1: negative EV is blocked. Chart cannot save it.
    if ev < 0:
        notes.append(f"engine ev_dollars={ev:.2f} < 0 - chart cannot upgrade negative EV")
        return "blocked", "negative_ev", notes
```

```python
# engine/candidate_dossier.py:240-249 (d24f041)
# Rule 5: EV threshold.
if ev >= self.min_proceed_ev:
    verdict: Verdict = "proceed"
    reason = "ev_above_threshold"
    notes.append(f"ev_dollars={ev:.2f} >= {self.min_proceed_ev} - proceed")
else:
    verdict = "review"
    reason = "ev_below_proceed_threshold"
    notes.append(f"ev_dollars={ev:.2f} < min_proceed {self.min_proceed_ev} - human review")
```

**Live reproduction** (CPython 3.12):

```
R1 +inf:  False     # +inf >  0  → R1 doesn't fire
R1 -inf:  True      # -inf <  0  → R1 fires (-inf blocked)
R1 NaN:   False     # NaN  <  0  → R1 doesn't fire (NaN slips R1)
R5 +inf:  True      # +inf >= 10 → R5 admits +inf as proceed
R5 NaN:   False     # NaN  >= 10 → R5 doesn't fire (NaN → review)
```

The five boolean facts match S19's claim exactly. **`+inf` is the only
verdict-bypassing input on the boundary set `{-25, -inf, 0, 5, 10,
25, +inf, NaN}`:** `-inf` is safely blocked by R1, `NaN` safely
degrades to "review" (R5's strict `>=` is `False` against `NaN`), and
`+inf` slides through both guards with a valid chart attached.

**Network exposure:** S20's G3 claim — that no `engine_api.py`
endpoint admits a user-controllable `ev_dollars` — holds today.

- `engine/tv_signals.py:543-552` defines `TVAlert.parse`'s
  `known` field set: `{ticker, signal, price, timeframe, source,
  timestamp, phase, bb_width_pctl, rsi, secret}`. **`ev_dollars` is
  absent** — a payload's `ev_dollars` lands in `extras` and is
  ignored.
- `engine_api.py:2046-2072` (`_enrich_alert`): initializes
  `ev_dollars = 0.0` and overrides only from a server-side
  `runner.rank_candidates_by_ev(...)` row. The payload's
  `ev_dollars` is never read.
- `engine_api.py:346` / `:356`: `param("min_ev", "0")` populates
  `min_ev_dollars`, which is the **filter threshold** passed to
  `runner.rank_candidates_by_ev(...)` / `runner.build_candidate_dossiers(...)`.
  It excludes candidates below the bar; it cannot inject a
  candidate-level EV value.
- `engine_api.py:209-217`'s `_sanitize_nans` is a belt-and-suspenders
  reply-path defence: `float('inf')`/`float('-inf')`/`float('nan')`
  in any internal state are replaced with `None` before serialisation,
  so the response is structurally incapable of carrying a non-finite
  EV out the door.

**Notes (the "WITH-NOTE" qualifier):** S19's "fix" suggestion at
`candidate_dossier.py:167` (replace `if ev < 0` with
`if not math.isfinite(ev) or ev < 0`) remains unmerged on `main` —
intentionally so per the S19 ledger ("Logged. Not fixed in this
Sn — Terminal A decision-layer lane."). The defence-in-depth gap is
documented in S19 and S20 and structurally closed at the network
surface; the in-process gap remains for any future caller of
`build_dossiers` that hand-builds a non-ranker-sourced `ev_row`.
Line citations in the S19 doc (`:167`, `:206-209`) are stale by +35
lines on `d24f041`; the doc reads correctly with that adjustment, and
the file-relative claim ("R1 is `if ev < 0`", "R5 is `if ev >=
self.min_proceed_ev`") is true today.

---

## C2 — `request_queue_size = 5` capacity ceiling

**Verdict:** **CONFIRMED.** The stdlib default is 5, `engine_api.py`
does not override it before instantiating the server, and the
mitigation proposed in the S20 AI-handoff has not been merged.

**Stdlib default** (CPython 3.12):

```
TCPServer.request_queue_size            = 5
ThreadingHTTPServer.request_queue_size  = 5
HTTPServer.request_queue_size           = 5
```

**Server instantiation** (`engine_api.py:2245-2252` on `d24f041`):

```python
def main():
    port = _resolve_port()
    # ThreadingHTTPServer spawns one thread per request so a slow committee
    # or memo call can't block the 5+ parallel fetches the dashboard fires
    # when a trader switches tickers.
    server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
    server.daemon_threads = True
```

No `ThreadingHTTPServer.request_queue_size = N` assignment exists in
`engine_api.py` before or after the constructor call (grep confirms;
the only occurrence of the literal `request_queue_size` in the file
is *inside* the stdlib import path, not in user code). The listen-
queue depth is the OS-default 5; beyond that, the kernel returns
`ECONNREFUSED` to the next `connect()`.

The S20 finding "16 workers → 133/200 client `ConnectionRefusedError`"
is consistent with this — the math (16 concurrent clients, listen
queue of 5, one-thread-per-request acceptance) puts the workers=16
case far past the listen-queue ceiling. The v5 backfill at workers=4
saw zero drops across 414+ POSTs, also consistent (4 < 5).

**Notes:** the in-band comment at `:2247-2249` ("the 5+ parallel
fetches the dashboard fires") understates the kernel-side ceiling.
The comment is about the *thread-spawn* model handling slow
committees without head-of-line blocking, which it does — but the
*accept* side still funnels through a single listening socket with a
backlog of 5. A reader of just the comment would not realise the
server caps at ~5 in-flight accepts. The S20 mitigation
(`ThreadingHTTPServer.request_queue_size = 128`) is one line and is
flagged for follow-up.

---

## C3 — v5 race-vector mechanisms

**Verdict:** **CONFIRMED.** Each of the four code-level mechanisms
S20 v5 relied on for its race-vector results is present today, with
the documented properties.

The v5 harness itself is not checked in (throwaway driver under
`%TEMP%\s20_concurrency\`, per the S20 setup paragraph), so this
section is a mechanism-only re-verification — *not* a re-run of the
G1/G2/G4/G5/G8 numbers.

### G1 — ring-trim cap

```
# engine_api.py:106-107
_TV_ALERT_LOG: list[dict] = []
_TV_ALERT_LOG_MAX = 200
```

```python
# engine_api.py:1688-1690 (ring-trim, after each append)
_TV_ALERT_LOG.append(enriched)
if len(_TV_ALERT_LOG) > _TV_ALERT_LOG_MAX:
    del _TV_ALERT_LOG[0 : len(_TV_ALERT_LOG) - _TV_ALERT_LOG_MAX]
```

S20 cited `:103-105` for the "buffer is rebuilt on each server start"
docstring and `:1683-1684` for the ring-trim. The docstring lines
match exactly (`:103-105`); the trim has drifted by +6 to `:1688-
1690`. Mechanism is unchanged — `list.append` + tail-trim at the
fixed cap `_TV_ALERT_LOG_MAX = 200`. The v5 G1c result ("buffer
length after = exactly 200, no drift, no off-by-one") is
self-consistent: `del list[0 : len - max]` removes exactly the right
prefix when triggered.

### G2 — GET slice atomicity

```python
# engine_api.py:1712-1715
def _handle_tv_alerts(self, limit: int):
    """Return the most recent alerts held in the in-memory log."""
    limit = max(1, min(limit or 50, _TV_ALERT_LOG_MAX))
    items = list(reversed(_TV_ALERT_LOG[-limit:]))
```

S20 cited line 1714 for `_TV_ALERT_LOG[-limit:]`; today it lives at
**`:1714`** exactly. The slice is a single bytecode-op (`SLICE`) on
the GIL-protected list, atomic against concurrent `list.append` /
`del list[lo:hi]` on the writer side. The v5 G2 result
(`get_len_min = get_len_max = 30` across 40 parallel GETs) is the
expected behaviour given GIL + slice atomicity.

### G4 — lock-free nonce check-then-set

```python
# engine_api.py:126-143 (function definition)
def _tv_seen_register(digest: str, now: float) -> bool:
    """..."""
    cutoff = now - _TV_WEBHOOK_MAX_AGE_SEC
    while _TV_SEEN_NONCES and next(iter(_TV_SEEN_NONCES.values())) < cutoff:
        _TV_SEEN_NONCES.popitem(last=False)

    if digest in _TV_SEEN_NONCES:           # <- check
        return False

    _TV_SEEN_NONCES[digest] = now           # <- set
    while len(_TV_SEEN_NONCES) > _TV_SEEN_NONCES_MAX:
        _TV_SEEN_NONCES.popitem(last=False)
    return True
```

**No `threading.Lock`, no `with` block, no atomic compare-and-swap.**
The reads and writes are individually GIL-atomic on `OrderedDict`,
but the *combined* `if digest in d: return False; d[digest] = now`
pattern is not — under the GIL, two threads can both see `digest
not in d` between their bytecode boundaries before either assigns.

S20's v5 G4 result (`1 × 200, 15 × 409` on 16 same-nonce POSTs at
workers=4) does not contradict this. At workers=4 the bytecode-
window race is narrow, CPython's GIL switches at fixed intervals
(~5ms by default, see `sys.getswitchinterval()`), and the check-then-
set sequence is short enough that contention rarely lands in the
small window. The S20 doc's read — "the win is GIL + dict-op
atomicity, not explicit synchronisation; the pattern is fragile" — is
exactly correct and matches the code today.

This is a defence-in-depth gap, not an exploit on the current
deployment. A move off CPython, a higher request rate, or any future
edit that widens the check/set window (e.g., adding a logging call
between the two lines) would surface a `>1`-accept anomaly. The S20
fix (a module-level `_TV_SEEN_NONCES_LOCK = threading.Lock()`
wrapping the function body) remains unmerged.

### G5 — per-thread isolation

```python
# engine_api.py:2245-2252 (server)
server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
server.daemon_threads = True
```

```python
# engine_api.py:163-167 (shared runner — lazy single mutation)
def get_runner():
    global _runner
    if _runner is None:
        _runner = WheelRunner()
    return _runner
```

`ThreadingHTTPServer` spawns one thread per accepted request, so the
slow-dossier vs. fast-alerts test cannot serialise unless the handler
itself takes a shared lock. The only shared mutation in the request
path is the lazy `_runner` initialisation, which after the first
request is a single `return _runner` read — no serialising mutation.

`engine/data_connector.py:_load` *is* lock-free and was correctly
called out by S20 as a code-level finding ("first ~16 concurrent
requests can ALL fall into the `pd.read_csv` branch for the same key,
each loading 60 MB into memory in parallel"). The v5 G5 result
(slow=0.69s, fast=0.03s, fast not blocked by slow) is consistent with
the per-thread model; on a warm connector (which v5's pre-warmed
runner provided) the `_load` race is dormant because the cache is
already populated.

### G8 — constant-time HMAC

```python
# engine_api.py:146-160
def _tv_verify_hmac(body_bytes: bytes, provided_sig: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 verification."""
    if not secret or not provided_sig:
        return False
    expected = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    got = provided_sig.strip()
    if got.lower().startswith("sha256="):
        got = got.split("=", 1)[1]
    return hmac.compare_digest(expected, got)
```

`hmac.compare_digest` is `Lib/hmac.py`'s timing-safe comparison;
the function is **stateless** — no shared mutable state, no cache,
no nonce. Two threads computing HMACs of two different bodies are
fully independent. S20's G8 result (32 × 401 for wrong-HMAC, 16 ×
200 for correct-HMAC, all at workers=4, no auth bypass) follows
directly from the statelessness: there is no shared structure for a
race to surface in.

---

## C4 — cross-arc coherency (IV-PIT bug landed after S18-S20)

**Verdict:** **CONFIRMED.** The IV-PIT bug was discovered as
**Finding F8 on PR #178 (S22 backtest, merged 2026-05-24)**, fixed in
**PR #179 (`claude/fix-ranker-iv-pit-aware`, commit `d26a8d6`,
merged 2026-05-25 as SHA `1378a5d`)**, and re-validated in **S27
(PR #184, `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`)**. None
of S18 (#153, 2026-05-23), S19 (#157, 2026-05-23), or S20 (#162 /
#164, 2026-05-24) were affected, because none of them tested
predictive validity against historical realized P&L.

**Date sequence:**

| Date (UTC) | PR | What |
|---|---|---|
| 2026-05-23 | #153 | S18 load/scale merged |
| 2026-05-23 | #157 | S19 chaos merged |
| 2026-05-24 | #162 | S20 original concurrency merged |
| 2026-05-24 | #164 | S20 v5 amendment merged |
| 2026-05-24 | #178 | **S22 backtest merged — F8 IV-PIT bug discovered** |
| 2026-05-25 | #179 | IV-PIT fix merged (`d26a8d6` → `1378a5d`) |
| 2026-05-25 | #184 | **S27 re-run merged — Spearman ρ 0.484 → 0.218** |

**Why S18-S20 are insensitive to the IV choice:**

- **S18.** Every lane (L1–L5b) runs at `as_of=2026-03-20` against the
  503-ticker SP500 universe. The findings are *operational*:
  wall-clock latency (145s cold, 10.5s warm), peak RSS (805 MB),
  handle drift (+5 per call), cache convergence (492 entries),
  graceful overshoot at `top_n=10_000`. These quantities are
  insensitive to which IV the ranker plugs into BSM — the same
  computational path runs either way; only the *output values* shift.
  At `as_of=2026-03-20` (the snapshot's own date), the pre-fix
  `fundamentals["implied_vol_atm"]` and the post-fix
  `get_iv_history(end_date=as_of)` row are typically close (the
  snapshot is current as of the snapshot date), so even the output
  values shift modestly — but the operational verdict (would survive
  production at scale, three named conditions) does not move.

- **S19.** 27 chaos vectors at `as_of=2026-03-20` plus a handful of
  bogus / out-of-range `as_of` values. Verdicts are
  fail-closed / fail-open classifications. The C7b §2 headline
  (`+inf` bypasses R1/R5) is a pure boolean test on the comparison
  operators in `EnginePhaseReviewer` — independent of any IV. The C2
  silent-`as_of`-substitution findings (the operational fail-opens)
  are about whether the engine emits a warning when the requested
  date is outside the snapshot's coverage, not about the IV value at
  the substituted date. **No S19 verdict moves under the IV-PIT
  fix.**

- **S20.** All vectors at `as_of=2026-03-20`. The §2 G3 result
  (the campaign-headline positive) is about whether the network
  surface accepts a user-supplied `ev_dollars` and propagates it.
  The race vectors are about HTTP / threading / ring-buffer /
  HMAC mechanics. **No S20 verdict moves under the IV-PIT fix.**

The S27 doc (`docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md`)
documents the magnitude of the IV-PIT effect — Spearman ρ on
predictive validity moves from 0.484 (pre-fix) to 0.218 (post-fix),
EV mean from +$92 to −$14, IV mean from 0.327 to 0.258. This is a
substantial signal-quality shift in the *predictive-validity* axis
(S22's lane); it does not retroactively invalidate the operational-
correctness axis (S18-S20).

**No cross-talk found.** Searching `docs/USAGE_TEST_LEDGER.md` for
"IV-PIT" / "PIT IV" returns matches only in S22 / S23 / S27 / S25
context; S18-S20 do not reference the bug. Confirmed.

---

## 5-ticker EV smoke

Baseline §2 sanity (CLAUDE.md §6), run on `claude/review-reliability-arc`
@ branch HEAD (off `d24f041`), Bloomberg `as_of` implicit
(`MarketDataConnector`):

```
connector type: MarketDataConnector
rows: 5
  ticker  ev_dollars      iv  premium
3    XOM      134.46  0.3194    2.437
2    JPM      116.94  0.3215    4.445
4    UNH       64.26  0.4323    6.038
1   MSFT      -24.39  0.2657    4.837
0   AAPL      -39.05  0.2616    3.127
NaN counts:
ev_dollars    0
iv            0
premium       0
```

5 rows, 0 NaN, sensible IV values (0.26–0.43, all within historical
SP500 ATM range). AAPL and MSFT now sit at negative EV — consistent
with the post-IV-PIT-fix smoke (snapshot IV pre-fix was higher; PIT
IV pulls premiums down; small / negative EV emerges on tighter
quotes). The decision-layer path
(`rank_candidates_by_ev → EVEngine.evaluate`) is healthy.

---

## Cross-cutting observations

**1. The C7b mechanism remains an in-process gap that the network
surface structurally closes.** A future code change that exposes a
"hand-built `ev_row` → dossier" code path on `engine_api.py` (e.g., a
new admin endpoint that accepts a pre-computed `ev_row` for
validation, or a future MCP tool that synthesises an `ev_row` from
external state) would re-open the C7b surface immediately. The
defence in S19's AI-handoff (one-line `math.isfinite` check at R1)
is cheap insurance — its absence today is conscious (Terminal A
decision-layer lane), and the chained network defences
(`TVAlert.parse`'s known-field whitelist, `_enrich_alert`'s
server-side override, `_sanitize_nans`' reply-path scrub) carry the
load. The audit trail is clear; the in-process residual is the
single largest defence-in-depth-not-defence-in-fact item from the
reliability arc.

**2. The S20 capacity ceiling will become live during the dashboard
hard-reload pattern, not before.** A dashboard tab opening cold fires
5–10 simultaneous fetches against `engine_api.py:8787` (status +
candidates + portfolio + regime + universe + a few committee tiles).
At the OS-default backlog of 5, the 6th-and-beyond accept attempts
race against the handler's `accept → spawn-thread` loop. The v5
backfill at workers=4 ran clean across 414+ POSTs because it stayed
under the listen-queue; the v3 16-worker run lost 133/200 because it
overshot. A real production deployment will live in the 5–16-worker
band by default — exactly the band where measurable drops begin. The
two-line `request_queue_size = 128` fix from the S20 AI-handoff
remains the highest-value follow-up in this arc.

**3. The two divergent verdict paths in `engine_api.py` (S20 finding,
not in scope for re-verification here) are the other latent
defence-in-depth risk.** The dossier endpoint (`/api/tv/dossier`)
routes through `EnginePhaseReviewer.review` (R1–R8). The webhook
endpoint (`/api/tv/webhook` → `_enrich_alert`) runs an inline
verdict block at `engine_api.py:2082-2099` that has the *same shape*
as R1/R5 plus extras (`days_to_earnings < 5`, `prob_profit >= 0.65`,
`agrees`) but **does not invoke `EnginePhaseReviewer`**. If R1 grew a
non-finite check tomorrow, the dossier path would gain it for free;
the webhook path would not, because it has its own copy of the rule.
Same input could produce different verdicts between the two
endpoints. The S20 doc flagged this for Terminal A decision-layer
follow-up; it has not been unified on `main`. Worth surfacing again.

---

_Reviewed under read-only constraints (no edits to `engine/`,
`tests/`, or decision-layer files). All findings logged into this
document; no fixes applied._
