"""R11 — elevated-vol size-down reviewer (heavy-verify 2026-05-31 I11 + robustness
sweep 2026-06-27).

R11 downgrades proceed→review when the market-wide VIX *level* exceeds 25 (the I11
leave-one-crisis-out robust cut) under two triggers:

  * R11a (``elevated_vol_top_bin``) — a top-bin candidate (prob_profit > 0.90).
  * R11b (``elevated_vol_skew_edge``) — a candidate whose EV rests on a positive
    real-premium skew edge (premium_source == "market_mid" AND edge_vs_fair > 0),
    i.e. a trade the Phase-2 rail (#435) unlocks in the high-vol regime that the
    OOS sweep (S35, 2020) showed dragged realized NAV negative.

Both are downgrade-only (never upgrade / rescue), no-op when vix_level is absent,
and R11b additionally no-ops on the synthetic path (premium_source != "market_mid")
so a rail-absent CI run is byte-identical. Each warning payload carries the
candidate's OWN computed cvar_5 (not a hardcoded number). These pins lock that
contract.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.candidate_dossier import (
    R11_SKEW_EDGE_MIN,
    R11_TOP_BIN_PROB,
    R11_VIX_THRESHOLD,
    CandidateDossier,
    EnginePhaseReviewer,
    build_dossiers,
)
from engine.chart_context import ChartContext, Timeframe
from engine.wheel_runner import WheelRunner

REVIEWER = EnginePhaseReviewer()


def _dossier(
    prob_profit,
    vix_level,
    ev_dollars=50.0,
    cvar_5=-5000.0,
    strike=100.0,
    premium_source=None,
    edge_vs_fair=None,
):
    """A proceed-state dossier (clean chart, ev>threshold) + R11 inputs.

    ``premium_source`` / ``edge_vs_fair`` default to None and are only inserted
    into ``ev_row`` when provided, so R11a tests exercise the (realistic) path
    where those diagnostic fields are absent and R11b stays a no-op.
    """
    ev_row = {
        "ticker": "TEST",
        "strike": strike,
        "premium": 2.0,
        "ev_dollars": ev_dollars,
        "iv": 0.25,
        "dte": 30,
        "spot": strike,
        "prob_profit": prob_profit,
        "cvar_5": cvar_5,
    }
    if premium_source is not None:
        ev_row["premium_source"] = premium_source
    if edge_vs_fair is not None:
        ev_row["edge_vs_fair"] = edge_vs_fair
    return CandidateDossier(
        ticker="TEST",
        ev_row=ev_row,
        chart_context=ChartContext(
            ticker="TEST",
            timeframe="1D",
            captured_at=datetime(2026, 4, 25, 12, 0, 0),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=strike,
            visible_indicators={},
            source="test",
        ),
        vix_level=vix_level,
    )


def test_r11_fires_top_bin_elevated_vol():
    v, r, notes = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=30.0))
    assert v == "review"
    assert r == "elevated_vol_top_bin"
    assert any("R11" in n for n in notes)


def test_r11_noop_when_vix_below_threshold():
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=20.0))
    assert v == "proceed"  # VIX not elevated → R11 silent, R5 verdict stands


def test_r11_noop_when_not_top_bin():
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.85, vix_level=30.0))
    assert v == "proceed"  # not a high-confidence candidate → R11 silent


def test_r11_noop_when_vix_absent():
    """Missing-evidence semantics (like R6-R10): no vix_level → no-op."""
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.95, vix_level=None))
    assert v == "proceed"


def test_r11_never_rescues_negative_ev():
    """R11 is gated on verdict=='proceed'; a negative-EV candidate is blocked by
    R1 long before R11 and stays blocked even with elevated VIX + top-bin prob."""
    v, r, _ = REVIEWER.review(_dossier(prob_profit=0.99, vix_level=80.0, ev_dollars=-50.0))
    assert v == "blocked"
    assert r == "negative_ev"


def test_r11_boundaries_strictly_greater_than():
    # VIX exactly at threshold → no fire; just above → fire.
    assert REVIEWER.review(_dossier(0.95, R11_VIX_THRESHOLD))[0] == "proceed"
    assert REVIEWER.review(_dossier(0.95, R11_VIX_THRESHOLD + 0.01))[0] == "review"
    # prob_profit exactly at cutoff → no fire; just above → fire.
    assert REVIEWER.review(_dossier(R11_TOP_BIN_PROB, 30.0))[0] == "proceed"
    assert REVIEWER.review(_dossier(R11_TOP_BIN_PROB + 0.001, 30.0))[0] == "review"


def test_r11_payload_is_computed_not_hardcoded():
    """The warning carries THIS candidate's own cvar_5 (regime-matched, computed),
    so it tracks the data — two candidates show two different tail figures."""
    _, _, notes_a = REVIEWER.review(_dossier(0.95, 30.0, cvar_5=-5000.0))
    _, _, notes_b = REVIEWER.review(_dossier(0.95, 30.0, cvar_5=-12345.0))
    r11_a = next(n for n in notes_a if "R11" in n)
    r11_b = next(n for n in notes_b if "R11" in n)
    assert "5,000" in r11_a and "cvar_5" in r11_a
    assert "12,345" in r11_b  # payload reflects the candidate's own modeled tail


def test_build_dossiers_threads_vix_level_and_r11_fires():
    """The wiring path: build_dossiers(vix_level=...) attaches it and R11 fires."""

    class _FakeProvider:
        def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None):
            return ChartContext(
                ticker=ticker,
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=100.0,
                visible_indicators={},
                source="test",
            )

    ev = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "strike": 100.0,
                "premium": 2.0,
                "ev_dollars": 50.0,
                "iv": 0.25,
                "dte": 30,
                "spot": 100.0,
                "prob_profit": 0.95,
                "cvar_5": -5000.0,
            }
        ]
    )
    armed = build_dossiers(ev_frame=ev, provider=_FakeProvider(), top_n=1, vix_level=30.0)
    assert armed[0].verdict == "review" and armed[0].verdict_reason == "elevated_vol_top_bin"
    # Same frame, no vix_level → R11 dormant, proceeds.
    dormant = build_dossiers(ev_frame=ev, provider=_FakeProvider(), top_n=1)
    assert dormant[0].verdict == "proceed"


# ---------------------------------------------------------------------------
# R11b — elevated-vol real-premium skew-edge size-down (robustness sweep 2026-06-27).
#
# R11b downgrades a NON-top-bin candidate proceed→review when VIX is elevated AND
# its EV rests on a positive real-premium skew edge (premium_source == "market_mid"
# AND edge_vs_fair > R11_SKEW_EDGE_MIN). It bounds the procyclical over-aggressiveness
# the Phase-2 real-premium rail (#435) adds in crisis (OOS S35: 26→40 trades, −3.4%
# NAV). Crucially it no-ops on the synthetic path so a rail-absent run is unchanged.
# ---------------------------------------------------------------------------


def test_r11b_fires_skew_edge_elevated_vol():
    """Non-top-bin candidate whose EV rests on a real-premium skew edge, in
    elevated vol → downgrade with the distinct skew-edge reason."""
    v, r, notes = REVIEWER.review(
        _dossier(prob_profit=0.85, vix_level=30.0, premium_source="market_mid", edge_vs_fair=0.20)
    )
    assert v == "review"
    assert r == "elevated_vol_skew_edge"
    assert any("R11b" in n for n in notes)


def test_r11b_noop_on_synthetic_premium_source():
    """THE no-re-baseline guarantee: on the synthetic path R11b never fires, even
    with a (structurally-zero, but here forced-positive) edge and elevated VIX, so a
    rail-absent CI/regression run is byte-identical to pre-#435."""
    v, _, _ = REVIEWER.review(
        _dossier(
            prob_profit=0.85, vix_level=30.0, premium_source="synthetic_bsm", edge_vs_fair=0.20
        )
    )
    assert v == "proceed"


def test_r11b_noop_when_edge_not_positive():
    """A real quote whose mid is at/below fair (no skew lift) is not over-aggressive
    — R11b stays silent."""
    flat = REVIEWER.review(
        _dossier(prob_profit=0.85, vix_level=30.0, premium_source="market_mid", edge_vs_fair=0.0)
    )
    neg = REVIEWER.review(
        _dossier(prob_profit=0.85, vix_level=30.0, premium_source="market_mid", edge_vs_fair=-0.15)
    )
    assert flat[0] == "proceed"
    assert neg[0] == "proceed"


def test_r11b_noop_when_vix_below_threshold():
    """Calm regime → R11b silent. This is what PRESERVES the real-premium calibration
    win (rho ↑): the gate only bites in the crisis regime where impact went negative."""
    v, _, _ = REVIEWER.review(
        _dossier(prob_profit=0.85, vix_level=20.0, premium_source="market_mid", edge_vs_fair=0.20)
    )
    assert v == "proceed"


def test_r11b_noop_when_vix_absent():
    """Missing-evidence semantics: no vix_level → no-op (like R6-R10 / R11a)."""
    v, _, _ = REVIEWER.review(
        _dossier(prob_profit=0.85, vix_level=None, premium_source="market_mid", edge_vs_fair=0.20)
    )
    assert v == "proceed"


def test_r11b_noop_when_diagnostic_fields_absent():
    """No premium_source / edge_vs_fair on the row (include_diagnostic_fields=False
    path) → R11b cannot fire; a non-top-bin candidate proceeds even in elevated vol."""
    v, _, _ = REVIEWER.review(_dossier(prob_profit=0.85, vix_level=30.0))
    assert v == "proceed"


def test_r11b_never_rescues_negative_ev():
    """R11b is gated on verdict=='proceed'; R1 blocks negative EV first and it stays
    blocked even with a fat positive skew edge in extreme vol."""
    v, r, _ = REVIEWER.review(
        _dossier(
            prob_profit=0.85,
            vix_level=80.0,
            ev_dollars=-50.0,
            premium_source="market_mid",
            edge_vs_fair=1.00,
        )
    )
    assert v == "blocked"
    assert r == "negative_ev"


def test_r11a_precedes_r11b_when_both_apply():
    """When a candidate is BOTH a top bin AND carries a positive real-premium skew
    edge, the more-specific over-confidence rule (R11a) fires first — its reason is
    reported, not the skew-edge one. Both downgrade, so the verdict is review either
    way; the precedence just fixes the audit-trail reason."""
    v, r, _ = REVIEWER.review(
        _dossier(prob_profit=0.95, vix_level=30.0, premium_source="market_mid", edge_vs_fair=0.20)
    )
    assert v == "review"
    assert r == "elevated_vol_top_bin"


def test_r11b_boundary_strictly_greater_than():
    """edge exactly at R11_SKEW_EDGE_MIN → no fire; just above → fire."""
    at = _dossier(
        prob_profit=0.85,
        vix_level=30.0,
        premium_source="market_mid",
        edge_vs_fair=R11_SKEW_EDGE_MIN,
    )
    above = _dossier(
        prob_profit=0.85,
        vix_level=30.0,
        premium_source="market_mid",
        edge_vs_fair=R11_SKEW_EDGE_MIN + 0.01,
    )
    assert REVIEWER.review(at)[0] == "proceed"
    assert REVIEWER.review(above)[0] == "review"


def test_r11b_payload_is_computed_not_hardcoded():
    """The R11b warning carries THIS candidate's own cvar_5 and its real edge — two
    candidates show two different figures, so the payload tracks the data."""
    _, _, notes_a = REVIEWER.review(
        _dossier(0.85, 30.0, cvar_5=-5000.0, premium_source="market_mid", edge_vs_fair=0.20)
    )
    _, _, notes_b = REVIEWER.review(
        _dossier(0.85, 30.0, cvar_5=-12345.0, premium_source="market_mid", edge_vs_fair=0.37)
    )
    r11b_a = next(n for n in notes_a if "R11b" in n)
    r11b_b = next(n for n in notes_b if "R11b" in n)
    assert "5,000" in r11b_a and "0.20" in r11b_a
    assert "12,345" in r11b_b and "0.37" in r11b_b


def test_r11b_fires_via_build_dossiers():
    """End-to-end through build_dossiers: a non-top-bin row carrying the real-premium
    diagnostic fields downgrades under elevated VIX; the synthetic twin proceeds."""

    class _FakeProvider:
        def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None):
            return ChartContext(
                ticker=ticker,
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=100.0,
                visible_indicators={},
                source="test",
            )

    base = {
        "ticker": "TEST",
        "strike": 100.0,
        "premium": 2.0,
        "ev_dollars": 50.0,
        "iv": 0.25,
        "dte": 30,
        "spot": 100.0,
        "prob_profit": 0.85,  # NOT top bin — isolates R11b from R11a
        "cvar_5": -5000.0,
        "edge_vs_fair": 0.20,
    }
    real = pd.DataFrame([{**base, "premium_source": "market_mid"}])
    armed = build_dossiers(ev_frame=real, provider=_FakeProvider(), top_n=1, vix_level=30.0)
    assert armed[0].verdict == "review"
    assert armed[0].verdict_reason == "elevated_vol_skew_edge"

    synth = pd.DataFrame([{**base, "premium_source": "synthetic_bsm"}])
    proceed = build_dossiers(ev_frame=synth, provider=_FakeProvider(), top_n=1, vix_level=30.0)
    assert proceed[0].verdict == "proceed"


# ---------------------------------------------------------------------------
# build_candidate_dossiers VIX-threading FAIL-SAFE (wheel_runner side).
#
# WheelRunner.build_candidate_dossiers threads the PIT market-wide VIX *level*
# into the dossier reviewer so R11 fires live on the ranking path
# (engine/wheel_runner.py, ~L3357):
#
#     vix_level: float | None = None
#     try:
#         if self.connector is not None and hasattr(self.connector, "get_vix_regime"):
#             _v = self.connector.get_vix_regime(as_of).get("vix")
#             vix_level = float(_v) if _v is not None else None
#     except Exception:        # noqa: BLE001 — VIX is advisory; never fail the rank
#         vix_level = None
#
# The broad ``except`` is the fail-safe: VIX is advisory, so ANY connector
# failure must degrade R11 to a no-op (``vix_level=None``) and never propagate
# out of the live rank. The dossier-side tests above pin R11 given a vix_level;
# these pin the *source* of that vix_level — so a future refactor that narrows
# or drops the ``except`` (or breaks the threading) re-introduces a crash on the
# live ranking path, or silently kills R11, and trips CI either way.
# (PR #308 worklog, backlog theme ②.)
# ---------------------------------------------------------------------------

# One R11-eligible row: ev>threshold so R1/R5 reach "proceed", spot==chart price
# so R3 doesn't skip, prob_profit>R11_TOP_BIN_PROB so R11 *can* fire.
_R11_ELIGIBLE_ROW = {
    "ticker": "TEST",
    "strike": 100.0,
    "premium": 2.0,
    "ev_dollars": 50.0,
    "iv": 0.25,
    "dte": 30,
    "spot": 100.0,
    "prob_profit": 0.95,
    "cvar_5": -5000.0,
}


class _CleanChartProvider:
    """Returns a clean chart so R2 (missing) / R3 (mismatch) don't pre-empt R11."""

    def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None):
        return ChartContext(
            ticker=ticker,
            timeframe="1D",
            captured_at=datetime(2026, 4, 25, 12, 0, 0),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            visible_indicators={},
            source="test",
        )


class _RaisingVix:
    def get_vix_regime(self, as_of=None):  # noqa: ARG002
        raise RuntimeError("VIX feed down")


class _NoneVix:
    def get_vix_regime(self, as_of=None):  # noqa: ARG002
        return None  # None.get("vix") -> AttributeError -> caught


class _VixlessDictVix:
    def get_vix_regime(self, as_of=None):  # noqa: ARG002
        return {}  # .get("vix") is None -> vix_level None, no exception


class _NonNumericVix:
    def get_vix_regime(self, as_of=None):  # noqa: ARG002
        return {"vix": "not-a-number"}  # float(...) -> ValueError -> caught


class _NoVixMethod:
    """A connector that doesn't implement get_vix_regime at all."""


class _WorkingVix:
    def __init__(self, vix):
        self._vix = vix

    def get_vix_regime(self, as_of=None):  # noqa: ARG002
        return {"vix": self._vix}


def _runner_with_vix(monkeypatch, connector):
    """A WheelRunner whose ranker is stubbed to one R11-eligible row and whose
    connector is ``connector``. Isolates the build_candidate_dossiers VIX-
    threading fail-safe from the (separately tested) ranker + data layer:
    ``_connector`` is set directly so the lazy real connector is never built.
    """
    runner = WheelRunner()
    runner._connector = connector
    ev = pd.DataFrame([_R11_ELIGIBLE_ROW])
    monkeypatch.setattr(runner, "rank_candidates_by_ev", lambda *a, **k: ev)
    return runner


def _build_one(runner):
    """Drive build_candidate_dossiers over the single stubbed row; return its dossier.

    A propagating exception here fails the test outright — which is the point for
    the degradation cases below.
    """
    dossiers = runner.build_candidate_dossiers(
        tickers=["TEST"],
        top_n=1,
        min_ev_dollars=-1e9,
        chart_provider=_CleanChartProvider(),
    )
    assert len(dossiers) == 1
    return dossiers[0]


def test_failsafe_get_vix_regime_raises_degrades_to_noop(monkeypatch):
    """A raising VIX connector must NOT propagate: vix_level degrades to None and
    R11 is dormant, so the rank completes and the candidate keeps its R5 verdict."""
    d = _build_one(_runner_with_vix(monkeypatch, _RaisingVix()))
    assert d.vix_level is None
    assert d.verdict == "proceed"
    assert d.verdict_reason != "elevated_vol_top_bin"


def test_failsafe_get_vix_regime_returns_none_degrades_to_noop(monkeypatch):
    """connector.get_vix_regime() -> None makes ``.get('vix')`` raise
    AttributeError; the fail-safe must catch it and degrade to a no-op."""
    d = _build_one(_runner_with_vix(monkeypatch, _NoneVix()))
    assert d.vix_level is None
    assert d.verdict == "proceed"


def test_failsafe_vixless_dict_degrades_to_noop(monkeypatch):
    """A dict with no 'vix' key -> ``_v is None`` -> vix_level None, no exception."""
    d = _build_one(_runner_with_vix(monkeypatch, _VixlessDictVix()))
    assert d.vix_level is None
    assert d.verdict == "proceed"


def test_failsafe_non_numeric_vix_degrades_to_noop(monkeypatch):
    """A non-numeric 'vix' -> ``float(...)`` ValueError -> caught -> no-op."""
    d = _build_one(_runner_with_vix(monkeypatch, _NonNumericVix()))
    assert d.vix_level is None
    assert d.verdict == "proceed"


def test_failsafe_connector_without_get_vix_regime_degrades_to_noop(monkeypatch):
    """A connector lacking get_vix_regime -> ``hasattr`` False -> threading skipped."""
    d = _build_one(_runner_with_vix(monkeypatch, _NoVixMethod()))
    assert d.vix_level is None
    assert d.verdict == "proceed"


def test_failsafe_teeth_working_elevated_vix_fires_r11(monkeypatch):
    """ANTI-VACUITY teeth: with a *working* connector returning an elevated VIX,
    the SAME harness reaches R11 and downgrades proceed->review. This proves the
    degradation cases above are dormant because the connector failed — not because
    R11 was unreachable in this fixture. If the threading silently dropped
    vix_level to None always, THIS test fails."""
    d = _build_one(_runner_with_vix(monkeypatch, _WorkingVix(30.0)))
    assert d.vix_level == 30.0
    assert d.verdict == "review"
    assert d.verdict_reason == "elevated_vol_top_bin"


def test_working_low_vix_threads_through_but_r11_holds(monkeypatch):
    """A working connector below the threshold threads the real level through, but
    R11's own gate keeps it silent — distinguishes 'threading works' from
    'R11 over-fires'."""
    d = _build_one(_runner_with_vix(monkeypatch, _WorkingVix(15.0)))
    assert d.vix_level == 15.0
    assert d.verdict == "proceed"
