---
id: d16-token-param-binding
title: EV-authority token parameter binding (brain-audit M1)
kind: fix
status: merged
terminal: UltraCode
pr:
decisions: [D16]
date: 2026-06-11
headline: D16 EV-authority token was an unbound bearer token — a token issued for AAPL/180/dte32 would gate any open_short_put/open_covered_call regardless of ticker/strike/expiration/side. Fix adds consume-side parameter binding (ticker, strike, derived-dte, side) via _ev_authority_payloads; hash and single-use semantics unchanged; two new refusal reasons token_param_mismatch + unbound_token; legacy snapshot rebind from audit log.
surface:
  - engine/wheel_tracker.py
  - tests/test_token_param_binding.py
  - tests/test_authority_hardening.py
  - tests/test_ev_authority_log_schema.py
  - TESTING.md
  - FILE_MANIFEST.md
  - DECISIONS.md
---

## Goal

Close the D16 unbound-bearer-token defect (brain-audit 2026-06-11 MEDIUM #1). The
`WheelTracker` EV-authority token was a bare SHA-256 hex digest stored in a
`set[str]`. The consume side (`_consume_ev_authority_token`) validated only (1)
token truthiness, (2) set membership, (3) `current_ev_dollars` presence, and (4)
`current_ev_dollars > 0` — and nothing else. This meant:

- A token issued for AAPL/strike=180/dte=32 would successfully gate an
  `open_short_put` for ZZZQ/strike=50 (cross-ticker leak confirmed by repro probe).
- A put-issued token would gate `open_covered_call` (cross-leg replay).
- The `ticker` argument at consume was logged but never compared.

The fix binds the token to (ticker, strike, derived-dte, side) at issuance time,
without changing the SHA-256 hash (token strings stay byte-identical across
versions) and without touching the trio files.

## What we tried

1. **Hash re-derivation at consume.** Recompute SHA-256 from the requested params
   and compare to the presented token. Rejected: the canonical hash includes
   `premium`, `ev_dollars`, `prob_profit`, and `distribution_source` which
   legitimately drift between rank and fire (the D16 stale-EV re-check exists for
   exactly this reason); full re-derivation over-binds, and re-deriving over a
   SUBSET requires changing hash composition.

2. **O(n) log scan at consume.** Scan `_ev_authority_log` for the matching issue
   entry. Rejected: makes a mutable audit-trail load-bearing for a security check
   on every consume; fragile to future log pruning.

3. **Separate `_ev_authority_payloads` dict (shipped).** Dedicated token-hash →
   canonical-payload mapping, populated at issuance alongside the token set. Log
   used only as a one-time legacy-rebuild fallback in `from_dict`.

## What worked

The shipped design. Key design decisions from the validator spec:

- Token hash is NOT extended — `side` is stored in the payload dict alongside
  the canonical fields but is NOT hashed.
- Side-resolution precedence: `side` kwarg > `ev_row.get('option_type')` >
  `ev_row.get('side')` > `''` (empty = legacy-unbound, still binds ticker/strike/dte).
- Mismatch retains the token (mirrors `stale_ev` semantics — a forged/buggy
  consume attempt must not burn the genuine candidate's single-use authority).
- Unbound-token path fails closed: token in set but no payload → refuse.
- BLOCKER resolution: `consume_ranker_row` overwrites the row's `dte` with the
  derived dte from the ACTUAL resolved `expiration_date` before calling
  `issue_ev_authority_token`, preserving the documented calendar-exact control
  escape hatch (explicit expiration diverging from row['dte'] no longer
  self-refuses).

## What didn't

- The "hash AAPL" approach (alternatives 1 above) breaks backward compatibility
  with persisted tokens and would require changing every existing token string.

## How we fixed it

All changes in `engine/wheel_tracker.py` (non-trio):

1. Added `self._ev_authority_payloads: dict[str, dict] = {}` to `__init__`.
2. `issue_ev_authority_token`: added `side: str | None = None` kwarg; resolves
   side via kwarg > `option_type` > `side` > `''`; stores
   `{**canonical, 'side': side_resolved}` in `_ev_authority_payloads` AFTER
   the hash (canonical stays unchanged).
3. `_consume_ev_authority_token`: added keyword-only `strike`, `entry_date`,
   `expiration_date`, `side`; after set-membership inserts (a) unbound-token
   check and (b) ticker/strike/dte/side mismatch check with structured log
   entry; success path also pops from `_ev_authority_payloads`.
4. `open_short_put` / `open_covered_call`: forward binding args with
   `side='put'` / `side='call'`.
5. `consume_ranker_row`: passes `side='put'` to issuance AND overwrites `dte`
   with `derived_dte = (expiration_date - entry_date).days` in a row copy
   (BLOCKER resolution).
6. `to_dict` / `from_dict`: serializes `ev_authority_payloads`; legacy rebuild
   from log issue entries when the key is absent.

Fixture repairs (pre-existing dte=35-vs-32 inconsistency the binding exposed):
- `tests/test_authority_hardening.py` L333, L964: add `dte=32` to covered-call
  `_ev_row` overrides (entry 2026-04-14 → expiry 2026-05-16 = 32 days).
- `tests/test_ev_authority_log_schema.py` L482: same repair.

Schema registration:
- `tests/test_ev_authority_log_schema.py` `_VALID_SHAPES`: registered
  `('reject', 'unbound_token')` with required `{action, reason, ticker, token}`
  and `('reject', 'token_param_mismatch')` with required
  `{action, reason, ticker, token, mismatched_fields, expected, requested}`.

New test file `tests/test_token_param_binding.py` (15 tests):
- Cross-ticker / cross-strike / cross-dte / cross-leg (side) mismatch refused.
- Exact match opens; single-use still enforced; mismatch-then-exact-match
  sequence (the corrected d6 probe-B/C).
- Unbound-token fail-closed; legacy snapshot rebuild; persistence round-trip.
- Schema-closure integration (validates both new shapes via real tracker paths).
- Non-strict mode untouched.
- BLOCKER resolution: escape-hatch test (explicit 67-day expiry on a dte=35 row
  succeeds); payload-dte-pinned-to-derived-dte test.

§2 safety: the change is refusal-only on an already-downgrade-only gate. No trio
file is touched. Non-strict mode (`require_ev_authority=False`) is unreachable
from the new code paths. Token hash composition is unchanged — no S27/S32/S34/S35
re-baseline required.

## Evidence

Targeted suite:

```
pytest tests/test_authority_hardening.py tests/test_ev_authority_log_schema.py
  tests/test_token_param_binding.py -q
→ 65 passed in 2.91s
```

Related suite:

```
pytest tests/test_audit_viii_e2e.py tests/test_wheel_tracker_persistence.py
  tests/test_consume_ranker_row_anchor.py tests/test_decision_layer_wiring.py -q
→ 58 passed in 28.49s
```

Manifest + taxonomy gates: `check_manifest_coverage.py` → OK (1004 tracked),
`test_testing_md_taxonomy.py` → 2 passed.

## Unresolved / handoff

- DECISIONS.md D16 annotation (docs-only addendum) was specified in the fix
  design but is a cosmetic doc change; can be a follow-on if desired.
- The brain-audit probe artifact
  `docs/verification_artifacts/brain_audit_2026-06-11/d6_probe2.py` has a
  section-C comment "expected False" that now becomes True post-fix (token
  retained on mismatch). Should be annotated when the fix lands.
