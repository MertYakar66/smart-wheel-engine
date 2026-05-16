# Agent Verification Workflow

This document describes a three-party verification workflow used when an
agent's findings about the Smart Wheel Engine codebase need to be audited
for accuracy before they are acted upon. Any agent joining mid-stream
should read this file first to understand the working system.

The goal of the workflow is simple: **separate the production of findings
from the verification of findings**, so that hallucinated, unsupported, or
internally inconsistent claims are caught before they drive a decision.

---

## 1. The three parties

### Executor — "Claude Code"
The agent with direct repository access. It reads files, runs commands,
runs tests, and produces findings. It holds ground truth: when a question
is about what a file *contains*, what a test *does*, or what a function
*returns*, the executor's reading of the repo is authoritative.

### Verifier — this partner
The agent that audits the executor's output. Its job is **not** to redo the
executor's work or to overrule it on facts. Its job is to check that every
substantive claim is:

- backed by a `file:line` citation or quoted command output,
- actually supported by that evidence (not merely adjacent to it),
- internally consistent with the executor's earlier statements, and
- consistent with the project profile (`CLAUDE.md` and `docs/`),
  treating the profile as evidence, not as infallible ground truth.

### Relay — the human
The human routes messages between the two agents and resolves deadlocks.
When the two agents disagree and neither has decisive evidence, the human
makes the call.

---

## 2. Repo-access model

In the canonical form of this workflow, the verifier has **no** direct repo
access and must request evidence through the relay. In the current setup
the verifier *does* have direct repo access, so we operate in an enhanced
mode:

> **Mode B — verifier with independent check.**
> The verifier audits the executor's pasted output *and* independently
> confirms claims against the real files when doing so is cheap. Findings
> are reported as, e.g., "Executor reports X; verified against
> `engine/ev_engine.py:234`, holds" or "...contradicts what that line
> shows."

Mode B collapses the relay round-trip for anything the verifier can check
directly, and only escalates to a relayed probe when the executor genuinely
needs to dig deeper (run a test, inspect a large surface, reproduce a
behaviour).

The alternative, **Mode A — strict verifier**, has the verifier ignore its
own repo access and audit only the pasted text. Mode A keeps the workflow
faithful to its repo-blind design; Mode B is faster and stronger. The
current engagement runs in **Mode B**.

---

## 3. Authority model

The verifier never overrules the executor on a **fact** about the repo. If
the verifier suspects a factual claim is wrong, it probes for evidence; if
the evidence contradicts the original claim, that is recorded as a
*finding*, not as an overrule.

The verifier **does** override the executor on:

- logical inconsistencies between the executor's own statements across
  turns,
- claims unsupported by any evidence the executor has shown,
- suggested actions that would violate a project NEVER rule or a recorded
  design decision.

Independent verification (Mode B) does not change this: when the verifier
reads a file directly and it contradicts the executor, that is still framed
as a finding for the relay to resolve, not a unilateral overrule.

---

## 4. Per-turn verification procedure

For each executor output, the verifier produces, in order:

1. **Verdict** — one of `PASS`, `FLAG`, `PROBE-NEEDED`, with a one-line
   reason.
2. **Claim audit** — for each substantive claim: is there a citation or
   quoted output; does the evidence actually support the claim; is it
   consistent with earlier turns; is it consistent with the project
   profile.
3. **Hallucination flags** — explicit list of tells (see §5). An empty list
   is a good outcome.
4. **Next message to the executor** — only when the verdict is
   `PROBE-NEEDED`: the exact text for the relay to paste back, asking for
   the *smallest sufficient* evidence. One targeted probe beats five vague
   ones, and a probe should take the executor well under a minute to run.
5. **Status read** — one to three sentences: what is resolved, what is
   still open, what the next forward step looks like.

---

## 5. Hallucination tells the verifier watches for

- Architectural claims with no grep or import evidence.
- Hedging language — "looks fine", "appears to", "should work" — with no
  quoted code.
- Plausible-sounding API names, signatures, or library behaviour that may
  not exist as stated.
- Numbers (line counts, coverage %, test counts) presented without the
  command output that produced them.
- Claims about files the executor has not shown it read this session.
- Contradictions with earlier turns.

---

## 6. Output format (verifier)

```
## Verdict
PASS | FLAG | PROBE-NEEDED — <one-line reason>

## Claim audit
| Claim (paraphrased) | Evidence shown? | Verdict | Concern |
|---|---|---|---|
| ... | yes / no / partial | ok / weak / contradicted / unverified | ... |

## Hallucination flags
- ... (empty list is fine and good)

## Next message to Claude Code
(omit this section if verdict is PASS)
<exact text for the relay to paste into the terminal>

## Status
<1-3 sentences>
```

---

## 7. Operating rules

- Do not invent file contents, line numbers, or behaviour that has not been
  seen or verified.
- Attribute clearly: "Executor reports X", not "X is the case" — unless the
  verifier has independently confirmed X, in which case say so explicitly.
- Do not speculate about repo state in a way the relay could mistake for
  verified fact.
- Once the executor produces direct contradicting evidence, the question is
  resolved — record it and move on; do not hold a position.
- Tie every concern, where possible, to a specific NEVER rule or recorded
  design decision from the project profile — those are the testable
  contract.
- No verification theater. If an output is clean, say `PASS` in one line
  and let the work continue.
- Keep probes short. A probe that takes the executor more than ~30 seconds
  to execute breaks the loop.

---

## 8. Project-specific anchors

When auditing claims about the Smart Wheel Engine specifically, the
verifier checks them against the load-bearing invariants in `CLAUDE.md`:

- **The hard invariant** — no tradeable candidate bypasses
  `EVEngine.evaluate`. Any suggested change that converts a non-tradeable
  candidate into a tradeable one without a fresh `EVEngine.evaluate` call
  is an automatic `FLAG`.
- **Downgrade-only inputs** — chart providers, news sentiment, the advisor
  committee, and the dealer overlay may only downgrade a verdict, never
  rescue a negative-EV trade.
- **Provider selection** — the active data provider (`SWE_DATA_PROVIDER`,
  default `bloomberg`) should be logged, not assumed; silent provider
  selection is a known bug source.
- **Decision-layer changes** — any change touching `engine/ev_engine.py`,
  `engine/wheel_runner.py`, or `engine/candidate_dossier.py` is expected to
  be accompanied by the full test suite, not a targeted file.

These anchors are evidence, not ground truth — if the executor shows the
repo has diverged from `CLAUDE.md`, the repo wins and the divergence is
itself a finding worth surfacing.
