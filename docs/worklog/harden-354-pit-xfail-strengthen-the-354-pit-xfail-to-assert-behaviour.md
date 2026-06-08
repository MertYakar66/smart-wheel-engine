---
id: harden-354-pit-xfail
title: Strengthen the #354 PIT xfail to assert behaviour, not signature (kill the false-green)
kind: fix
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-08
headline: test_fundamentals_credit_are_point_in_time now asserts PIT row-SELECTION (synthetic 2-snapshot fixture, 2024 row ordered first + latest so a no-op as_of returns it) instead of just `"as_of" in signature` — so a future no-op as_of param can no longer false-green the xfail(strict) while the dateless-snapshot lookahead persists
surface: [tests/test_data_to_engine.py]
---

## Goal

`tests/test_data_to_engine.py::test_fundamentals_credit_are_point_in_time`
tracked the #354 dateless-fundamentals lookahead by asserting only
`"as_of" in inspect.signature(get_fundamentals)`. That is a **shape** check:
the day someone bolts a no-op `as_of` param onto `get_fundamentals` /
`get_credit_risk`, this `xfail(strict=True)` flips green — falsely signalling
"defect fixed" while the lookahead persists (the accessors still serve a single
dateless 2026 snapshot). Strengthen it to assert PIT **behaviour** so only a
genuine `as_of`-selection can flip it. Test-only; #354 itself stays parked
behind its PIT-data dependency.

## What we tried / what worked

A synthetic fixture decouples the guard from the production data:
- write `sp500_fundamentals.csv` + `sp500_credit_risk.csv` into `tmp_path` with
  **two dated snapshots** (2023 + 2024) for one ticker (`PITX`);
- order the **2024 row first** (so today's `df[df.ticker==t].iloc[0]` returns it)
  **and** give it the **latest date** (so a "take the most recent" stub returns
  it too) — both common no-op styles land on 2024;
- point a `MarketDataConnector(data_dir=tmp_path)` at it and assert
  `get_fundamentals(t, as_of="2024-06-01")` returns the **2023** row
  (`dividend_yield == 1.23`, not the 2024 `4.56`); same for
  `get_credit_risk` (`sp_rating == "CR2023"`).

Only a real `as_of` filter (`date <= as_of`, latest such) returns 2023.

## What didn't / why this design

A signature check can't distinguish a real fix from a stub — that's the whole
hole. Asserting against the *live* `sp500_fundamentals.csv` was a non-starter:
it's dateless (no `as_of` to select on) and editing it is a data-file change
that trips the #340 re-baseline guard (out of scope unattended). The synthetic
tmp fixture is the only additive, data-safe way to assert selection.

## How we fixed it

Replaced the signature assertion with the behavioural one above + a
`_write_pit_fundamentals_fixture` helper. The `xfail(strict=True)` + reason
stay; the reason now says "no as_of SELECTION … a no-op as_of param cannot
satisfy this assertion".

## Evidence (3-state discrimination, proven on the bytes)

- **(a) today** — `get_fundamentals(t, as_of=...)` → `TypeError: unexpected
  keyword argument 'as_of'` → `1 xfailed` (for the RIGHT reason now, not a
  signature miss).
- **(b) no-op stub** (ignore `as_of`, dateless `iloc[0]`) → returns `4.56`
  (2024) → assertion FAILS → **false-green defeated**.
- **(c) real PIT selection** (`date <= as_of`, latest) → returns `1.23` (2023)
  → assertion PASSES → `xfail(strict)` would xpass → flips red = "remove the
  xfail, #354 is fixed".

`pytest tests/test_data_to_engine.py -m "not slow"`: 8 passed / 12 xfailed
(11 blue-chip #355 + this one), no failures, no xpass. ruff check + format
clean. Test-only: no trio, no data-file edit, no parallel-session files.

## Unresolved / handoff

#354 stays parked behind its PIT-data dependency (a dated fundamentals panel —
Bloomberg pull). When it's tackled: data first, plumbing with it (the `as_of`
threading from `wheel_runner` is a separate **trio PR**), and THIS test flips
red to confirm — then delete the xfail. See memory
`xfail-signature-only-false-greens` and `docs/NEXT_DATA_SESSION_RUNBOOK.md`.
