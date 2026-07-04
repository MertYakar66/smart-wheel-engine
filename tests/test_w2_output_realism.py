"""W2 — engine output-realism pins (heavy-verify 2026-06-27, #436, Mac terminal).

Validation-only. Pins that the ranker's served outputs are realistic at a calm
and a stress ``as_of``: all finite, probabilities in [0, 1], served IV in the
clean decimal band, premium a sane fraction of spot, and a 25-delta short put
solving to Greeks that honour ``docs/GREEKS_UNIT_CONTRACT.md`` (delta in [-1, 0],
gamma>=0, vega>=0).

A ``universe_limit`` keeps each rank fast enough for the per-PR lane. See
``docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md`` §W2 and
``scripts/audit_output_realism.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from engine.option_pricer import black_scholes_all_greeks
from engine.wheel_runner import WheelRunner

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def runner() -> WheelRunner:
    return WheelRunner()


# A calm and a stress as_of; universe_limit bounds runtime.
@pytest.fixture(scope="module", params=["2022-06-16", "2024-01-16"])
def ranked(runner: WheelRunner, request) -> tuple[str, pd.DataFrame]:
    as_of = request.param
    df = runner.rank_candidates_by_ev(
        as_of=as_of,
        top_n=200,
        min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
        universe_limit=80,
    )
    return as_of, df


def test_outputs_are_finite(ranked) -> None:
    as_of, df = ranked
    assert len(df) > 0, f"no candidates at {as_of}"
    for col in ["spot", "strike", "premium", "iv", "ev_dollars", "prob_profit", "prob_assignment"]:
        v = pd.to_numeric(df[col], errors="coerce")
        assert np.isfinite(v).all(), f"{col} has non-finite values at {as_of}"


def test_probabilities_in_unit_interval(ranked) -> None:
    as_of, df = ranked
    for col in ["prob_profit", "prob_assignment"]:
        v = pd.to_numeric(df[col], errors="coerce")
        assert (v >= 0).all() and (v <= 1).all(), f"{col} outside [0,1] at {as_of}"


def test_served_iv_in_decimal_band(ranked) -> None:
    """Ranker IV is decimal; the connector PERCENT floor (3.0) ⇒ decimal > 0.03."""
    as_of, df = ranked
    iv = pd.to_numeric(df["iv"], errors="coerce")
    assert (iv > 0.03).all(), f"served IV <= 0.03 at {as_of}: {iv[iv <= 0.03].tolist()}"
    assert (iv <= 5.0).all(), f"served IV > 500% at {as_of}: {iv[iv > 5.0].tolist()}"


def test_premium_is_sane_fraction_of_spot(ranked) -> None:
    as_of, df = ranked
    ratio = pd.to_numeric(df["premium"], errors="coerce") / pd.to_numeric(
        df["spot"], errors="coerce"
    )
    assert (ratio > 0).all() and (ratio < 0.5).all(), f"premium/spot out of band at {as_of}"


def test_short_put_greeks_honour_contract(ranked) -> None:
    """25-delta short put: delta∈[-1,0], gamma>=0, vega>=0 (GREEKS_UNIT_CONTRACT)."""
    as_of, df = ranked
    rfr = WheelRunner().connector.get_risk_free_rate(as_of, "rate_3m")
    checked = 0
    for _, row in df.iterrows():
        S, K = float(row["spot"]), float(row["strike"])
        T, sigma = float(row["dte"]) / 365.0, float(row["iv"])
        if not (S > 0 and K > 0 and T > 0 and sigma > 0):
            continue
        g = black_scholes_all_greeks(S, K, T, rfr, sigma, "put", include_second_order=False)
        assert -1.0 <= g["delta"] <= 0.0, f"{row['ticker']} put delta {g['delta']} outside [-1,0]"
        assert g["gamma"] >= -1e-9, f"{row['ticker']} gamma < 0"
        assert g["vega"] >= -1e-9, f"{row['ticker']} vega < 0"
        checked += 1
    assert checked > 0, f"no evaluable greeks at {as_of}"
