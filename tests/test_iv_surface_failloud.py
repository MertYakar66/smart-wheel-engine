"""Tests for the A2 fail-loud SVI-surface contract (DECISIONS.md D9).

Pins the contract decided in ROADMAP A2: the SVI surface tooling is wired in
with a **fail-loud** missing-data guard — an uncovered ticker raises
``SurfaceDataUnavailable`` rather than silently quoting a fabricated flat IV.
Covers the guard (``require_surface``), the empty-input raise on
``create_empirical_surface``, and the diagnostic's pure core
(``scripts/diagnose_iv_surface.py``). A populated-surface regression confirms
the additive guard did not change normal behaviour.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine.volatility_surface import (  # noqa: E402
    SurfaceDataUnavailable,
    VolatilitySurface,
    create_empirical_surface,
    require_surface,
)

from diagnose_iv_surface import (  # noqa: E402 — sys.path injection above
    build_surface_from_atm_term_structure,
    summarize_surface,
)

_AS_OF = date(2026, 1, 2)
_TERM = {30: 0.25, 60: 0.24, 90: 0.23}  # ATM IV term structure (decimal)


def _empty_surface() -> VolatilitySurface:
    return VolatilitySurface(
        as_of_date=_AS_OF, underlying="EMPTY", forward_prices={}, svi_params={}
    )


class TestRequireSurface:
    def test_raises_on_empty(self) -> None:
        with pytest.raises(SurfaceDataUnavailable):
            require_surface(_empty_surface(), "EMPTY")

    def test_returns_populated_unchanged(self) -> None:
        surf = build_surface_from_atm_term_structure(_TERM, 100.0, _AS_OF, "AAPL")
        assert require_surface(surf, "AAPL") is surf


class TestCreateEmpiricalSurface:
    def test_raises_on_empty_term_structure(self) -> None:
        # create_empirical_surface itself fails loud on no input (pre-existing).
        with pytest.raises(ValueError):
            create_empirical_surface(
                iv_by_tenor={}, as_of_date=_AS_OF, underlying="X", spot=100.0, expiries=[]
            )


class TestDiagnosticCore:
    def test_build_raises_on_empty(self) -> None:
        with pytest.raises(SurfaceDataUnavailable):
            build_surface_from_atm_term_structure({}, 100.0, _AS_OF, "NOCOV")

    def test_build_from_synthetic_term_structure(self) -> None:
        surf = build_surface_from_atm_term_structure(_TERM, 100.0, _AS_OF, "AAPL")
        assert len(surf.svi_params) == len(_TERM)

    def test_atm_iv_honoured(self) -> None:
        # create_empirical_surface honours the ATM IV per expiry "exactly".
        surf = build_surface_from_atm_term_structure(_TERM, 100.0, _AS_OF, "AAPL")
        for dte, atm in _TERM.items():
            expiry = _AS_OF + timedelta(days=dte)
            _, atm_iv, _ = surf.get_skew(expiry)
            assert atm_iv == pytest.approx(atm, abs=0.01)

    def test_equity_put_skew_present(self) -> None:
        # Default equity skew: 25Δ put IV >= ATM IV (puts richer).
        surf = build_surface_from_atm_term_structure(_TERM, 100.0, _AS_OF, "AAPL")
        rows = summarize_surface(surf)
        assert len(rows) == len(_TERM)
        for row in rows:
            assert row["iv_25d_put"] >= row["atm_iv"] - 1e-9

    def test_summarize_shape(self) -> None:
        surf = build_surface_from_atm_term_structure(_TERM, 100.0, _AS_OF, "AAPL")
        rows = summarize_surface(surf)
        assert {"expiry", "dte", "atm_iv", "iv_25d_put", "iv_25d_call", "put_skew"} <= set(rows[0])
