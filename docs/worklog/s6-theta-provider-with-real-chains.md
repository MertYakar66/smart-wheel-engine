---
id: S6
title: Theta provider with real chains
kind: usage
status: planned
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise actual chain-quoted premiums vs the synthetic
BSM premium Bloomberg uses. `edge_vs_fair` is structurally 0 on
Bloomberg; S1 flagged this as the biggest missing signal.

**Setup.** `SWE_DATA_PROVIDER=theta` with the Theta Terminal running
on `127.0.0.1:25503`. Operator setup required.

- **Re-verified 2026-05-26** by Terminal A — **SKIPPED (operator-gated, Theta lock held by another agent)**. Per the task spec skip list: "S6 operator-gated — needs Theta Terminal. Not in completed list originally." Additionally, the user-stated coordination context: another agent is actively pulling data from Theta servers (the Theta lock per PARALLEL_SESSIONS rule 7 is held); even attempting an S6 re-run would risk collision. The §2-relevant claim — that Theta-sourced chain premiums route through `EVEngine.evaluate` exactly as Bloomberg-synthetic premiums do — is structurally enforced by `WheelRunner` provider abstraction and unchanged on `main`.

---
