---
id: S20
title: `engine_api.py` concurrency & crash resilience
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Characterise `engine_api.py` (HTTP on `:8787`) under
concurrency and crash conditions — parallel POST/GET, ring-buffer race
on `_TV_ALERT_LOG`, nonce-replay race on `_TV_SEEN_NONCES`, slow-ranker
blocking responsiveness, process-kill mid-write, payload validation,
auth-under-load — and document whether the API holds invariants under
production-shaped exposure. **The campaign-headline question is G3:
does S19's C7b `+inf` defence-in-depth gap become live-exploitable via
the network surface?** Closes the reliability arc (S18 scale + S19
chaos + S20 concurrency).

**Setup.** Server spawned as subprocess on `SWE_API_PORT=18787` /
`18788` (PR #158's per-instance binding, so the test instance doesn't
collide with the production port). 8 vectors planned (G1–G8); 5
landed cleanly. Bloomberg provider. Throwaway driver under
`%TEMP%\s20_concurrency\` (`tempfile`-style, not committed; not in
the worktree per the S18/S19 desktop-clutter feedback).
`urllib.request` + `concurrent.futures.ThreadPoolExecutor` for the
client side.

**Path.** Server uses `http.server.ThreadingHTTPServer` with
`daemon_threads = True` (`engine_api.py:2250-2251`) — one thread per
request, no explicit handler-side locking. The shared mutable state
that S20 stresses: `_TV_ALERT_LOG: list[dict]` (line 106),
`_TV_SEEN_NONCES: OrderedDict[str, float]` (line 121), and the
implicit shared state inside the lazy connector cache
(`engine/data_connector.py:_load`, no lock).

**Status.** Done. **Verdict: PASS, with one production-readiness
caveat (capacity ceiling at the OS socket backlog).** §2 G3
**REFUTED** (the campaign-headline positive); 4 secondary findings
(server-capacity limit at the OS socket backlog, ungraceful 10 MB
rejection, type-coerced ticker accepted, two divergent verdict-
producing paths); 1 code-level race surface
(`engine/data_connector.py:_load` has no lock around the cache-
populate). The race vectors (G1 ring-buffer trim + dedup, G2 torn
read, G4 nonce-replay race, G5 slow-vs-fast isolation, G8 HMAC under
load) all landed **clean** in the v5 backfill — see the per-vector
table. Four driver iterations (v1 wrong path → v3 wrong response key
→ v4 PIPE deadlock → v5 stdout-to-file fix) before the race
signals were measurable; the methodology debt is itself a finding
for the next API-chaos Sn.

*(Amended 2026-05-23 same-day: original entry shipped with
G1/G2/G4/G5/G8 as partially observed; v5 backfill landed clean
race-vector data. Original PARTIAL verdict revised to
PASS-with-caveat. Methodology-debt bullet preserves the v1→v5
driver history.)*

> **§2 G3 — REFUTED on the network surface.** No live exploit of
> S19's C7b inf-bypass via either `/api/tv/dossier` or
> `/api/tv/webhook`. Three control payloads (`+inf / NaN / -inf` as
> `ev_dollars` in the webhook body) all returned the server-computed
> AAPL EV (`-95.47` on the test run's `as_of`), with `verdict=skip`
> on all three. The server-side override is **mechanically protected**
> at `engine_api.py:2061-2072`: `_enrich_alert` re-initializes
> `ev_dollars = 0.0` and then overrides from
> `runner.rank_candidates_by_ev(...).iloc[0]["ev_dollars"]` — the
> payload's `ev_dollars` field is never read on the EV path
> (`TVAlert.parse` at `engine/tv_signals.py:537-567` doesn't have
> `ev_dollars` in its known-field set; it lands in
> `extras` and is ignored). `/api/tv/dossier` (line 1843-1865)
> similarly computes via `runner.build_candidate_dossiers(...)` from
> Bloomberg data; the only user-controlled EV-related input is
> `min_ev_dollars` (the **filter threshold**, not a candidate-level
> EV value). Probed `min_ev=Infinity` → 0 dossiers returned (filter
> drops everything; no fail-open). **C7b remains defence-in-depth
> only — log into S19's AI-handoff as the closing word: no
> network-path exploit observed.**

**Findings:**

- **Per-vector outcome table.** Verdicts after the methodology
  iterations; raw timings are best-effort given the capacity-related
  noise.

  | Vector | Description | Verdict | Detail (line cites) |
  |---|---|---|---|
  | **G3a** | `/api/tv/dossier?min_ev=Infinity&tickers=AAPL` | **REFUTED — fail-closed** | 200 OK, 0 dossiers (filter drops everything); `min_ev_dollars` is the filter threshold at `engine_api.py:1860`, not a candidate-EV override |
  | **G3a'** | `/api/tv/dossier?min_ev=-1e9&tickers=AAPL` (baseline) | works as designed | 200 OK, 1 dossier returned |
  | **G3c (+inf)** | webhook POST with `payload.ev_dollars = float("inf")` | **REFUTED — server overrides** | 200 OK, response `enriched.ev_dollars=-95.47` (real AAPL EV), `verdict=skip`. Payload's `+inf` never reaches the reviewer |
  | **G3c (NaN)** | same with `NaN` | **REFUTED — server overrides** | same: `enriched.ev_dollars=-95.47`, `verdict=skip` |
  | **G3c (-inf)** | same with `-inf` | **REFUTED — server overrides** | same: `enriched.ev_dollars=-95.47`, `verdict=skip` |
  | **G6** | crash recovery (kill PID, restart, verify) | **CLEAN** | pre-restart buffer cleared on cold start (`_TV_ALERT_LOG` per `engine_api.py:106`); post-restart POST returns 200 |
  | **G7a** | empty body | **fail_closed_400** | `{"error": "alert payload missing ticker or signal"}` |
  | **G7b** | non-JSON body | **fail_closed_400** | `{"error": "Invalid JSON: Expecting value..."}` |
  | **G7c** | `{"ticker": 42, "signal": [None, None]}` | **degraded** | 200 accepted; `TVAlert.parse` at `engine/tv_signals.py:557` does `str(payload.get("ticker", "")).upper()` → `"42"`; downstream `_enrich_alert` hits the `ticker_not_in_universe` branch at `engine_api.py:1989-1994` so no decision is produced — but the type-coerced row is accepted into `_TV_ALERT_LOG` as a no-op enriched record. Defence-in-depth: a `isinstance(payload.get("ticker"), str)` guard would fail-closed-400 instead |
  | **G7d** | `__proto__` / `constructor` keys | **clean** | 200 accepted; Python's `json.loads` has no JS prototype-pollution surface; keys captured into `TVAlert.extras` and ignored |
  | **G7e** | 10 MB payload | **fail_closed_socket_abort** | `ConnectionAbortedError(10053)`. Server abruptly closes the socket rather than emitting a 413 Payload Too Large. **Ungraceful** — a real load balancer in front of this would see the connection drop and may retry. Worth a proper `Content-Length` check upstream of the body read |
  | **G1a** | ring-buffer 32 POSTs at workers=4 | **clean** | 32/32 success; buffer length 33 = 32 new + 1 warmup AAPL; 33 unique tickers — no lost appends, no duplicates |
  | **G1c** | +200 POSTs to force trim, workers=4 | **clean** | 200/200 success; buffer length after = **exactly 200**; `_TV_ALERT_LOG_MAX=200` ring-trim at `engine_api.py:1683-1684` holds precisely |
  | **G2** | 40 POST + 40 GET parallel at workers=4 | **clean** | all 40 POSTs 200; all 40 GETs 200; every GET response returned exactly 30 items (`limit=30` honored); `get_len_min=get_len_max=30` — **no torn reads observed**, Python's GIL + slice semantics keep `_TV_ALERT_LOG[-limit:]` (line 1714) atomic |
  | **G4** | 16 same-nonce POSTs at workers=4 | **clean** | **1 × 200, 15 × 409** (`replay_blocked` per `_tv_seen_register` at `engine_api.py:125-143`). No race surfaced at workers=4. **Lock-free check-then-set IS theoretically racy** (lines 137-140); GIL + dict-op atomicity protect at this concurrency, but the pattern is fragile — see code-level finding below |
  | **G5** | slow dossier + fast alerts GET parallel | **clean** | slow_wall=0.69 s (warm caches), fast_wall=0.03 s (started 0.5 s after slow). `ThreadingHTTPServer` per-thread isolation works as advertised; **no serialization behind the shared `WheelRunner`** at `engine_api.py:163-167` (`get_runner` is the only shared mutation, lazy-init, single read after first call) |
  | **G6** | crash recovery (kill PID + restart) | **clean** | pre-kill: 10 POSTs OK, buffer=100; post-restart: buffer=**0**, follow-up POST=200. In-memory ring buffer cleared on cold start per `engine_api.py:103-105` docstring |
  | **G8** | 32 wrong-HMAC + 16 correct-HMAC POSTs at workers=4 | **clean — no auth bypass** | wrong: **32 × 401**; correct: **16 × 200**. `_tv_verify_hmac` (`engine_api.py:146-160`, `hmac.compare_digest` constant-time) holds deterministically under contention. No TOCTOU surfaced. |

- **Server-capacity ceiling (the only production-readiness limiter).**
  At 16 concurrent workers driving POSTs in the original v3 driver,
  **133 of 200** got `ConnectionRefusedError` / `-1` from the client.
  The remaining 67 succeeded with 200 OK. `http.server.
  ThreadingHTTPServer` accepts connections on a single listening
  socket whose listen-queue depth is the OS default
  (`socketserver.TCPServer.request_queue_size = 5`, inherited; see
  Python `Lib/socketserver.py`). Beyond ~5 in-flight accepts, the
  kernel refuses new connections. **The v5 backfill at workers=4
  showed zero drops across G1/G2/G4/G6/G8 (414+ POSTs total)**, so
  the production-readiness boundary is between workers=5 and
  workers=16. A typical dashboard hard-reload (5–10 simultaneous
  fetches) lands **right at the boundary** — usable, but a single CI
  smoke pile-up or an oncall investigator hitting it concurrently
  with the dashboard would tip into measurable drops. **Logged as
  a production-readiness gap.** Mitigations: (a) raise
  `request_queue_size` (`ThreadingHTTPServer.request_queue_size =
  128` before instantiating the server) — the two-line fix below;
  (b) front the server with a real reverse proxy (`nginx` / `caddy`)
  that handles the connection backpressure; (c) reduce dashboard
  concurrent-fetch count.

- **`engine/data_connector.py:_load` has no lock around cache-populate.**
  Code-level finding from reading the file:

  ```python
  # engine/data_connector.py:95-131
  def _load(self, key: str) -> pd.DataFrame:
      if key in self._cache:          # <- read without lock
          return self._cache[key]
      ...
      df = pd.read_csv(path, ...)     # <- can run concurrently in N threads
      ...
      self._cache[key] = df           # <- last writer wins
      return df
  ```

  Under ThreadingHTTPServer with `daemon_threads=True` and a cold
  connector (first request after server boot), the first ~16
  concurrent requests can ALL fall into the `pd.read_csv` branch
  for the same key, each loading 60 MB into memory in parallel.
  Last writer wins on `_cache[key]`, the earlier loads' DataFrames
  are garbage-collected — but during the load the process holds
  ~16× the steady-state memory footprint. Plausibly contributory to
  what wedged the v3 server after the first concurrent burst (though
  the proximate cause was the `subprocess.Popen(stdout=PIPE)`
  deadlock — v5 with stdout-to-file ran identical concurrency cleanly,
  so the lock-free `_load` was not the actual blocker in v3 either).
  Mitigation is one line: a `threading.Lock` field guarding the
  populate branch. **Logged. Terminal A data-layer-adjacent lane** —
  `data_connector.py` is the data layer per CLAUDE.md §1, not the
  decision layer; a lock-add is safe but should be claimed on #113.

- **Two divergent verdict-producing paths in `engine_api.py`.** The
  dossier endpoint (`/api/tv/dossier`) uses `EnginePhaseReviewer` at
  `engine/candidate_dossier.py:109-247` (rules R1–R6, the canonical
  reviewer). The webhook endpoint (`/api/tv/webhook` →
  `_enrich_alert`) runs **its own inline verdict logic** at
  `engine_api.py:2082-2099`:

  ```python
  # engine_api.py:2082-2099 (inline verdict, NOT EnginePhaseReviewer)
  if days_to_earnings is not None and 0 <= days_to_earnings < 5:
      verdict = "skip"; verdict_reason = "earnings_within_5d"
  elif verdict_authority != "ev_ranked":
      verdict = "review"
  elif ev_dollars < 0:
      verdict = "skip"
  elif ev_dollars >= 10 and prob_profit >= 0.65 and agrees:
      verdict = "proceed"
  elif ev_dollars > 0:
      verdict = "review"
  else:
      verdict = "skip"
  ```

  Both paths run `EVEngine.evaluate` upstream (so §2 holds in
  both), but the **rule structure diverges**: the webhook's
  conditions check `days_to_earnings < 5` (an earnings-gate
  duplicate not in `EnginePhaseReviewer`), `prob_profit >= 0.65`
  (additional confidence threshold), and `agrees` (signal-match
  predicate). The dossier reviewer has the dealer-positioning R6
  downgrade that the webhook does NOT. Same input could produce
  *different verdicts* between the two endpoints. **Code-duplication
  + consistency risk.** Logged as a divergence finding; the right
  fix is to route both endpoints through `EnginePhaseReviewer` and
  let R1–R6 be the single source of truth — but that's a Terminal A
  decision-layer change, not in scope here.

  **Bonus observation tied to S19 C7b:** the webhook's inline rule
  has the same `if ev_dollars >= 10` admit at `engine_api.py:2091`.
  If `ev_dollars` were ever `+inf` from the ranker (which it isn't
  on real Bloomberg data — see S19 C7b), this path would also emit
  `proceed` on garbage. Two paths, same defence-in-depth gap, both
  protected today only by the ranker not producing inf.

- **`_sanitize_nans` (`engine_api.py:209-217`) is a positive
  structural defence.** Every JSON response goes through
  `_sanitize_nans` which replaces `float('inf')` / `float('-inf')` /
  `float('nan')` with `None`. So even if internal state contained a
  non-finite EV that survived processing, the response would carry
  `None`, not the malicious value. **Belt-and-suspenders for the
  network reply path.** Worth pinning with a regression test
  (Terminal A lane).

- **G6 crash recovery is clean.** Killed the server PID
  (`subprocess.Popen.kill()` — SIGKILL on Unix, `TerminateProcess`
  on Windows), re-spawned on the same port, verified
  `_TV_ALERT_LOG` is empty after restart and POSTs work again. The
  docstring claim at `engine_api.py:103-105` ("buffer is rebuilt on
  each server start") holds. **Logged as a positive.** Note: the
  ring buffer is the *only* persistence surface in
  `engine_api.py` — there's no DB, no replay log, no disk-backed
  state. A crash mid-write loses at most one alert (the in-flight
  POST that didn't finish); everything else is on-disk in
  `data/bloomberg/*.csv` and loads cleanly on cold start.

- **Methodology debt — solved in v5.** Four iterations (v1 → v3 →
  v4 → v5) before landing usable race-vector data: v1 had a wrong
  endpoint path (`/api/tv/alert` vs the actual `/api/tv/webhook`); v3
  had a wrong response-key (`body["items"]` vs the actual
  `body["alerts"]`); v4 deadlocked on `subprocess.Popen(stdout=PIPE,
  stderr=PIPE)` with no draining (server filled the 64 KB pipe buffer
  and blocked on its next `print()`, all client requests then timed
  out); **v5 fixed all three by sending server stdout/stderr to a
  log file**, which unblocked clean numbers for G1/G2/G4/G5/G6/G8.
  Future API-chaos Sns should: (a) `subprocess.Popen(stdout=open(
  log, "w"), stderr=subprocess.STDOUT)` to a real file (or
  `DEVNULL`) — **never** PIPE without an active reader thread;
  (b) probe the response shape with one request before scaling to N;
  (c) start with workers=4 to stay under the default listen-queue;
  scale up only after validating the response shape. **Logged for
  the next-Sn-prompt template.**

- **§2 verified across the network surface.** The G3 negative
  result is the campaign-headline answer: C7b is mechanically
  closed by `_enrich_alert`'s `ev_dollars = float(r0.get(
  "ev_dollars", 0) or 0)` override at line 2072, and by the
  dossier endpoint's `runner.build_candidate_dossiers(...)`
  server-computation. **No observed network path emits a tradeable
  `proceed` on garbage ev_dollars.** This closes the reliability
  arc (S18 + S19 + S20) on a structural positive.

- **v5 backfill — race-vector positives.** Once the v4 PIPE
  deadlock was unblocked (v5 stdout-to-file), all five race vectors
  came back **clean at workers=4**:
  - **G1 ring-buffer trim is precise.** 32 unique POSTs all
    landed (buffer=33 including warmup); +200 more triggered the
    `_TV_ALERT_LOG_MAX=200` trim at `engine_api.py:1683-1684` —
    buffer cap held *exactly* at 200, no drift, no off-by-one.
  - **G2 GET-during-POST is atomic.** 40 GETs concurrent with 40
    POSTs returned **exactly 30 items each**
    (`get_len_min=get_len_max=30`). Python's GIL + the
    `_TV_ALERT_LOG[-limit:]` slice at line 1714 are atomic — no
    torn read, no truncated JSON, no `IndexError` 500s.
  - **G4 nonce-replay is correct under concurrency.** 16 same-
    payload POSTs at workers=4 yielded **1 × 200, 15 × 409** —
    exactly the expected behaviour from `_tv_seen_register`
    (`engine_api.py:125-143`). **However**, the check-then-set at
    lines 137-140 is **lock-free**, so the win is the GIL + dict-
    op atomicity, not explicit synchronisation. At higher
    concurrency (or under a non-CPython interpreter, or if the
    check/set window grew with future code changes), this could
    surface as >1 accept. **Logged as a code-level finding** —
    the same `threading.Lock` pattern as `_load` would close it.
  - **G5 per-thread isolation works.** Slow dossier (0.69 s warm)
    + fast `/api/tv/alerts` (0.03 s) ran concurrently; the fast
    GET was **not blocked** by the slow ranker. ThreadingHTTPServer's
    per-request threading at `engine_api.py:2250-2251` does what it
    advertises; the shared `WheelRunner` instance does NOT
    serialise readers behind a global mutation lock.
  - **G8 auth deterministic under load.** With `TV_WEBHOOK_HMAC_SECRET`
    set, 32 wrong-signature POSTs at workers=4 all returned 401;
    16 correct-signature POSTs all returned 200. **No auth bypass,
    no TOCTOU window.** `_tv_verify_hmac` (`engine_api.py:146-160`)
    uses `hmac.compare_digest` constant-time and is stateless, so
    the deterministic result under contention matches the code.
  **Logged as the campaign reliability-arc positive.**

**AI handoff.**

- **Top fix-up surface:** raise `request_queue_size` to handle
  realistic dashboard burst load. Two-line change at
  `engine_api.py:2250` —

  ```python
  # engine_api.py:2250 (current)
  server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
  # proposed:
  ThreadingHTTPServer.request_queue_size = 128
  server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
  ```

  Default is 5 (inherited from `socketserver.TCPServer`); 128 is the
  uvicorn/gunicorn default and is more aligned with the dashboard's
  burst pattern. **Terminal A decision-layer-adjacent lane**
  (`engine_api.py` is the interface layer; not the EV decision
  layer, but is on the launch-blocker list).

- **Second fix-up surface:** add a `threading.Lock` to
  `engine/data_connector.py:_load` to prevent N-thread cold-load
  amplification. Single-line lock acquire/release around lines
  101-129.

  ```python
  # engine/data_connector.py:86-89 (current)
  def __init__(self, data_dir: str = "data/bloomberg") -> None:
      self._data_dir = Path(data_dir)
      self._cache: dict[str, pd.DataFrame] = {}
  # proposed:
  def __init__(self, data_dir: str = "data/bloomberg") -> None:
      self._data_dir = Path(data_dir)
      self._cache: dict[str, pd.DataFrame] = {}
      self._cache_lock = threading.Lock()  # NEW
  # then in _load, wrap the cache-populate branch:
  def _load(self, key: str) -> pd.DataFrame:
      if key in self._cache:
          return self._cache[key]
      with self._cache_lock:
          if key in self._cache:    # double-check after lock
              return self._cache[key]
          # ...existing populate body unchanged...
  ```

  **Data-layer change, not decision-layer.** Worth a small Sn or
  Terminal A claim.

- **Third fix-up surface:** unify the two verdict-producing paths
  in `engine_api.py`. Either (a) route `_enrich_alert`'s decision
  logic through `EnginePhaseReviewer` (preferred — single source of
  truth), or (b) explicitly document the divergence and pin both
  rule-sets in tests. The S20 inline-rules block at
  `engine_api.py:2082-2099` reads like a parallel implementation of
  R1/R5 + extras; a regression test that drives the same `ev_row`
  through both endpoints and asserts identical verdicts would catch
  any future drift. **Terminal A decision-layer-adjacent lane.**

- **G7e 10 MB payload ungraceful disconnect** — add a
  `Content-Length` size guard at the top of `do_POST` (read header,
  reject 413 if > N bytes) so the server doesn't accept a
  body-too-large connection only to abort mid-stream. Defensive
  hygiene; not exploitable, but worth a clean 413.

- **Fourth fix-up surface (new from v5):** lock the check-then-set
  in `_tv_seen_register`. Current code at
  `engine/external/engine_api.py:125-143` (well, in `engine_api.py`
  itself):

  ```python
  # engine_api.py:125-143 (current — lock-free)
  def _tv_seen_register(digest: str, now: float) -> bool:
      cutoff = now - _TV_WEBHOOK_MAX_AGE_SEC
      while _TV_SEEN_NONCES and next(iter(_TV_SEEN_NONCES.values())) < cutoff:
          _TV_SEEN_NONCES.popitem(last=False)
      if digest in _TV_SEEN_NONCES:
          return False
      _TV_SEEN_NONCES[digest] = now
      while len(_TV_SEEN_NONCES) > _TV_SEEN_NONCES_MAX:
          _TV_SEEN_NONCES.popitem(last=False)
      return True
  # proposed: add a module-level _TV_SEEN_NONCES_LOCK = threading.Lock()
  # and wrap the body in `with _TV_SEEN_NONCES_LOCK:`.
  ```

  At workers=4 the race didn't surface (CPython GIL + dict-op
  atomicity protect the small check-then-set window), but the
  pattern is fragile: a future code change that widens the window,
  or a move off CPython, or higher concurrency in production, could
  surface a >1-accept anomaly. The fix is one `threading.Lock` plus
  a `with` block wrapping the function body.

- **Ruled out per the prompt:** any decision-layer code change
  (S20 found surfaces, did not fix), real-network load from off-box
  (none available in sandbox), MCP / chart provider chaos
  (S5 / S19 covered), fuzz testing (hypothesis), performance
  tuning, Theta provider failure (sandbox-blocked).

- **Campaign arc closure:** S18 (scale) + S19 (chaos) + S20
  (concurrency) all done. The **§2 invariant holds across all three
  axes on the live decision path.** Defence-in-depth gaps named
  (C7b inf-bypass, the silent-as_of substitution, the FRED-down
  silent regime label, the data_connector race, the two verdict
  paths) are all reads-not-writes findings; the fix surface is
  small and well-scoped for a follow-up. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`). API server spun up on Terminal-A's allocated port `:8787`.
  - §2 invariant: holds — **G3 REFUTED for the second time on the network surface**. Webhook payloads with `ev_dollars` as `+inf`, `-inf`, `NaN` all return server-computed AAPL EV (`-14.46`) and `verdict=skip` — the in-body `ev_dollars` field is ignored, `_enrich_alert` re-runs the ranker. `_sanitize_nans` confirmed: alerts GET response contains no `Infinity` or `NaN` JSON tokens.
  - Qualitative verdict: match — all race vectors clean at `workers=4`:
    - G1: 220 POSTs → buffer holds **exactly 200** (trim precise).
    - G2: 40 POSTs + 40 GETs concurrent → every GET returned exactly 30 items (no torn reads).
    - G4: 16 same-nonce POSTs → **1×200, 15×409** (nonce-replay protection works under contention).
    - G7a: empty body → 400; G7c: bad ticker (ZZZZ) → 200 with soft-rejection (matches original).
    - Dossier endpoint `min_ev=Infinity` returns 200 / 0 dossiers (filter consumed); baseline `min_ev=-1e9` → 1 dossier.
  - Numerical drift > 5%:
    - metric `AAPL_webhook_enriched_ev_dollars[as_of=2026-03-20]`: orig `-95.47` → new `-14.46` (`-85% magnitude`); attributable to **PR #179** (post-IV-PIT-fix AAPL ev_dollars at this as_of is materially less negative — PIT IV for AAPL at 2026-03-20 is lower than the snapshot, reducing the synthetic premium and the magnitude of the negative EV). Sign preserved.
  - Notes: PR #216 (engine_api `request_queue_size = 128` — Terminal B's S20 fix-up #1) and PR #219 (Terminal B's `_tv_seen_register` lock — S20 fix-up #4) are flagged as merged on `main` per the board's recent activity but were not the focus of this §2 re-test. Both should harden the same surfaces in higher-concurrency regimes than the workers=4 used here.

---
