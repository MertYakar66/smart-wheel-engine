"""W1/W2 re-verification pins (heavy-verify 2026-06-29, #436 independent re-run).

These pin two *durable* properties surfaced by the independent re-verification of
the merged #436 W1/W2 campaign — properties not asserted by the existing
``tests/test_w1_data_wiring.py`` / ``tests/test_w2_output_realism.py`` pins, and
which stay green *after* the #439 split-scale fix (PR #455) lands:

1. The authoritative corporate-action split ground-truth the OHLCV
   split-adjustment (and the #455 remediation) both depend on. The existing
   split-scale ``xfail`` pins the *symptom* (the 2026-03-23 discontinuity); this
   pins the *root-cause data* (``get_corporate_actions`` reports BKNG 25:1 eff
   2026-04-06, CVNA 5:1 eff 2026-05-08), so a corp-action data regression that
   silently changed a ratio/effective-date can never pass unnoticed.

2. The IV-validity gate ``(3.0, 10000]`` is scoped to *implied* vol only — the
   realized-vol columns (``volatility_30/60/90/260d``) are correctly served
   below 3% for calm names rather than being force-floored. The published W1
   counted only the two implied columns (2,062,702 cells); this pins that the
   scoping is deliberate, not an accident.

See ``docs/HEAVY_VERIFY_2026-06-29_DATA_WIRING_RELIABILITY.md``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector

_IV_GATE_LO = 3.0  # exclusive lower bound of the implied-IV validity band (percent)


@pytest.fixture(scope="module")
def conn() -> MarketDataConnector:
    return MarketDataConnector()


# ---------------------------------------------------------------------------
# 1. Corporate-action split ground-truth (root cause of D-W1-1; durable)
# ---------------------------------------------------------------------------
# Hand-verified against the real-world splits and the committed
# sp500_corporate_actions.csv — NOT copied from any engine computation.
_EXPECTED_SPLITS = {
    "BKNG": (25.0, "2026-04-06"),
    "CVNA": (5.0, "2026-05-08"),
    "NFLX": (10.0, "2025-11-17"),
}


@pytest.mark.parametrize("ticker", sorted(_EXPECTED_SPLITS))
def test_corp_action_split_ground_truth(conn: MarketDataConnector, ticker: str) -> None:
    """``get_corporate_actions`` serves the authoritative most-recent material
    Stock Split ratio + effective date the split-adjustment logic relies on."""
    exp_ratio, exp_eff = _EXPECTED_SPLITS[ticker]
    ca = conn.get_corporate_actions(ticker)
    splits = ca[
        ca["action_type"].astype(str).str.contains("Split", na=False) & (ca["ratio"] > 1.5)
    ].sort_values("effective_date")
    assert not splits.empty, f"{ticker}: no material stock split served"
    last = splits.iloc[-1]
    assert float(last["ratio"]) == exp_ratio, (
        f"{ticker}: split ratio {last['ratio']} != authoritative {exp_ratio}"
    )
    assert pd.Timestamp(last["effective_date"]).date() == pd.Timestamp(exp_eff).date(), (
        f"{ticker}: split eff_date {last['effective_date']} != {exp_eff}"
    )


def test_split_effective_dates_postdate_the_2026_03_23_splice(conn: MarketDataConnector) -> None:
    """Diagnosis pin: BKNG/CVNA split *effective dates* are strictly AFTER the
    2026-03-23 OHLCV pull boundary — proving the discontinuity is a splice
    artifact, not a true effective-date adjustment. Stays green post-fix
    (corp-action effective dates are unaffected by the OHLCV regeneration)."""
    boundary = pd.Timestamp("2026-03-23")
    for ticker in ("BKNG", "CVNA"):
        ca = conn.get_corporate_actions(ticker)
        splits = ca[
            ca["action_type"].astype(str).str.contains("Split", na=False) & (ca["ratio"] > 1.5)
        ]
        eff = pd.Timestamp(splits.sort_values("effective_date").iloc[-1]["effective_date"])
        assert eff > boundary, (
            f"{ticker} split eff {eff.date()} not after the {boundary.date()} splice"
        )


# ---------------------------------------------------------------------------
# 2. IV-gate scoping: implied gated, realized NOT floored (durable)
# ---------------------------------------------------------------------------
# Calm names whose realized vol legitimately dips below 3% while their implied
# IV stays in the gated band. Hand-verified mins (EA implied 3.127 / realized
# 1.945; HOLX implied 3.218 / realized 2.316).
@pytest.mark.parametrize("ticker", ["EA", "HOLX"])
def test_iv_gate_scoped_to_implied_not_realized(conn: MarketDataConnector, ticker: str) -> None:
    iv = conn.get_iv_history(ticker)
    assert not iv.empty, f"{ticker}: no IV history served"

    implied = iv[["hist_put_imp_vol", "hist_call_imp_vol"]].apply(pd.to_numeric, errors="coerce")
    realized = iv[["volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d"]].apply(
        pd.to_numeric, errors="coerce"
    )
    implied_min = float(implied.min().min())
    realized_min = float(realized.min().min())

    # implied IV is gated: every served implied cell stays strictly above 3.0
    assert implied_min > _IV_GATE_LO, (
        f"{ticker}: implied IV min {implied_min} violates the (3.0, …] gate"
    )
    # realized vol is NOT floored: at least one realized cell is legitimately < 3.0
    assert realized_min < _IV_GATE_LO, (
        f"{ticker}: realized vol min {realized_min} — expected a legitimately-low (<3%) "
        "value proving the implied-IV gate is not erroneously applied to realized vol"
    )
    # and realized vol is never negative
    assert (realized.fillna(0.0) >= 0).all().all(), f"{ticker}: negative realized vol served"
