# Heavy-Verify Campaign 2026-05-31 — I4: The §2 invariant under adversarial attack

**Investigation:** can a negative-EV candidate be made tradeable? Can any reviewer
or multiplier UPGRADE a verdict or flip an EV sign?
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i4_section2_probe.py`.
**Raw:** `raw_output/i4_section2_RAW.txt`. **Status:** observe-and-document; `engine/`
not modified. Crux claim independently re-verified by the lead.

---

## VERDICT

> **§2 HELD across all six attacks. I could not break it.** A negative-EV candidate
> cannot be converted into a tradeable one. `ev_dollars = ev_raw × regime_mult` where
> `regime_mult` is a product of independently non-negative, clamped factors (regime
> `[0,1.25]`, heavy-tail `0.5`, dealer `[0.70,1.05]`), so it is **sign-preserving** —
> a negative `ev_raw` can only stay ≤ 0. `ev_raw` and `prob_profit` are computed
> (ev_engine.py:383/393) **before** any multiplier, so reviewers/multipliers
> physically cannot reach them. The dossier reviewer is first-match, downgrade-only,
> with non-finite/negative-EV hard-stops (R1a/R1) ahead of every proceed path. The
> actual enforcement chokepoint is the tracker's two-stage token gate (issue refuses
> ev≤0; consume re-checks live `current_ev_dollars>0`, single-use), which a forged
> dossier verdict cannot bypass.**

Confidence: **high.** This is the single most important finding category and it is
the reassuring one: the structural firewall is intact on the current engine.

---

## Attacks and outcomes (all HELD; full evidence in the raw file)

| # | Attack | Result |
|---|---|---|
| 1 | **Sign-flip via multipliers.** Negative trade (ev_raw=−2815); swept `regime_multiplier ∈ {−100,−1,0,0.5,1,1.25,5,1e9,inf,nan}` + adversarial `MarketStructure`. | HELD. Every out-of-range/non-finite input clamped to [0,1.25] (logged `regime_anomaly`); `ev_dollars` sign always matched `ev_raw`; dealer mult always within [0.70,1.05]; at mult=1e9 → clamped 1.25 → ev_dollars=−3519 (more negative, never rescued). |
| 2 | **R6+R7–R10 composition.** Positive row + short-gamma MarketStructure + tiny-NAV PortfolioContext. | HELD. Downgraded to `review` (R6 first-match); negative-EV + both downgraders → `blocked` (R1 short-circuits). No upgrade. |
| 3 | **Forged ev_row.** Hand-crafted `ev_dollars=999`. | Dossier returns `proceed` (it trusts ev_row verbatim — **by design, NOT the chokepoint**). The same trade's real row (ev=−2815) → `issue_ev_authority_token` **refused**; consume with real negative → **refused**. §2 enforced at the rank→token→consume wire. |
| 4 | **Token gate (strict mode).** no token / forged token / stale-negative current_ev / issue(ev≤0). | HELD. All refused; valid token + positive current_ev → True (not vacuous); replay of consumed token → refused (single-use). |
| 5 | **R5 boundary.** ev=10.00 / 9.99 / 10.01. | HELD. 10.00→proceed, 9.99→review, 10.01→proceed (≥ MIN_PROCEED_EV=10.0 exact). |
| 6 | **Non-finite EV.** +inf / −inf / NaN. | HELD. All → `blocked` / `ev_non_finite` (R1a). |

**Lead's independent re-verification:** `git diff origin/main -- engine/` = 0 lines
(no engine code touched by the campaign or the probe); the clamp lines read as
claimed (ev_engine.py:493-496 — regime_multiplier validated/clamped to [0,1.25],
out-of-range logged not silently trusted; dealer [0.70,1.05] asymmetric); the probe
reproduces the negative-stays-negative result and the token-gate refusal.

## Residual concerns a skeptical PM should still know (none breach §2 today)

1. **The dossier trusts `ev_row['ev_dollars']` verbatim.** It is by design *not* the
   chokepoint, but any live path that acts on `verdict=="proceed"` directly (e.g. a
   UI "fire" button) without re-routing through the tracker token gate would inherit
   a forgeable trust boundary. §2 safety depends on every tradeable path going
   through `issue/consume_ev_authority_token`.
2. **Strict mode is opt-in** (`require_ev_authority=False` by default). The token gate
   only protects trackers explicitly built with it True. (See I3-A: it is off on
   every ranker/backtest path today.)
3. **R4 (phase contradiction) is dormant** — no provider populates the phase field;
   untested in live composition.
4. **Non-finite `regime_multiplier` is silently clamped to a neutral 1.0** (with a
   `regime_anomaly` tag in `metadata`, not the verdict). An upstream bug producing
   inf/nan multipliers degrades to neutral rather than failing loud — the only signal
   is a metadata tag.
5. **First-match returns** means downstream rule interactions (e.g. R10 when R9 also
   fires) are never co-evaluated — fine for downgrade-only, but untested past the
   first trigger.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i4_section2_probe.py
git diff origin/main -- engine/   # must be empty
```
