---
description: Run the launch-blocker test subset — the decision-layer invariant gate
---

Run the launch-blocker test subset — the decision-layer invariant gate
defined in `docs/LAUNCH_READINESS.md` §4 and `TESTING.md`. This is the
floor before merging any change under `engine/` that could touch the
decision layer.

Execute exactly:

```bash
pytest tests/test_audit_invariants.py \
       tests/test_dossier_invariant.py \
       tests/test_authority_hardening.py \
       tests/test_audit_viii_unit_invariants.py \
       tests/test_audit_viii_e2e.py \
       tests/test_audit_viii_real_data_smoke.py \
       tests/test_launch_blockers.py -v
```

Report the pass/fail result. If anything fails, surface the failing test
and its output — do not summarize it away. Per `CLAUDE.md` §2 and
`DECISIONS.md` D1, a failure here means the ranker is unsafe to merge.

For changes to `engine/ev_engine.py`, `engine/wheel_runner.py`, or
`engine/candidate_dossier.py`, run the full suite (`pytest tests/ -v`) as
well — the invariants are cross-cutting.

This command is a thin wrapper around the documented subset; it adds no
logic of its own.
