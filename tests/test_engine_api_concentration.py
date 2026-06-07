"""Live-path proof for the armed R9/R10 concentration caps via
``/api/concentration_preview``.

``engine.wheel_runner.make_live_book_tracker`` arms R9 (sector, 25% NAV) +
R10 (single-name, 10% NAV), and ``consume_into_live_book`` is the §2-safe
one-call production wire — but until this endpoint landed those caps had
**zero callers outside ``tests/``** (heavy-verify 2026-05-31 Cat-A; #154 /
#343): the armed caps protected nothing on any operator path. This file pins
that the ``/api/concentration_preview`` builder makes the caps fire on the
live path, and that it preserves the §2 contract (EV-ranker-only candidates,
refuse-only caps, no negative-EV rescue).

The ranker is faked (a ``WheelRunner`` subclass returning a synthetic
concentrated batch) so the test is deterministic and data-free — but
``consume_into_live_book`` / ``make_live_book_tracker`` / ``open_short_put``
and the real ``portfolio_risk_gates`` cap math run unmocked end to end, so
the assertions exercise the genuine production cap path, not a stub.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner
from engine_api import build_concentration_preview

ENTRY = date(2026, 1, 5)

# Canonical ranker-row fields the token hash + open_short_put read
# (ticker / strike / premium / dte / ev_dollars / iv / distribution_source,
# plus optional prob_profit). Anything else is ignored by the wire.
_BASE_ROW = {
    "premium": 1.5,
    "dte": 35,
    "iv": 0.30,
    "distribution_source": "empirical",
    "prob_profit": 0.70,
}


def _row(ticker: str, strike: float, ev_dollars: float, **over) -> dict:
    return {**_BASE_ROW, "ticker": ticker, "strike": strike, "ev_dollars": ev_dollars, **over}


class _FakeRankerRunner(WheelRunner):
    """A WheelRunner whose ranker output is injected, with no connector.

    Overriding ``rank_candidates_by_ev`` lets us drive the REAL
    ``consume_into_live_book`` → ``consume_into_tracker`` →
    ``make_live_book_tracker`` → ``open_short_put`` chain with a synthetic
    batch. ``connector`` is ``None`` — the armed caps are notional/NAV-based
    and resolve GICS sectors from the bundled DEFAULT_SECTOR_MAP, exactly as
    ``tests/test_production_tracker_caps.py`` relies on.
    """

    def __init__(self, rows: list[dict]):
        super().__init__()
        self._rows = rows

    @property
    def connector(self):  # type: ignore[override]
        return None

    def rank_candidates_by_ev(self, **kwargs):  # noqa: D102 - injected output
        return pd.DataFrame(self._rows)


def _preview(rows: list[dict], *, initial_capital: float = 100_000.0, top_n: int = 20):
    return build_concentration_preview(
        _FakeRankerRunner(rows),
        entry_date=ENTRY,
        initial_capital=initial_capital,
        rank_kwargs={"top_n": top_n, "min_ev_dollars": 0.0},
        top_n_to_consume=top_n,
    )


def test_caps_fire_on_the_live_path():
    """The preview ADMITS a diversified prefix and REFUSES the
    over-concentrated names — proving R9 (sector) + R10 (single-name) fire on
    the operator path, not just in the tracker unit tests.

    Batch (NAV $100k):
      AAPL/MSFT/NVDA @ $80 strike = $8k = 8% NAV each (Information Tech)
        -> sector climbs 8% -> 16% -> 24% (< 25%), each name 8% (< 10%): OPEN
      JPM @ $200 = $20k = 20% NAV (Financials): single-name 20% > 10% -> R10
      AMD @ $80 = 4th IT name -> sector would hit 32% > 25% -> R9
    """
    rows = [
        _row("AAPL", 80.0, 40.0),
        _row("MSFT", 80.0, 38.0),
        _row("NVDA", 80.0, 36.0),
        _row("JPM", 200.0, 30.0),
        _row("AMD", 80.0, 20.0),
    ]
    out = _preview(rows)

    assert out["consumed"] == 5
    assert out["opened"] == 3
    assert out["refused"] == 2
    assert set(out["book"]["positions"]) == {"AAPL", "MSFT", "NVDA"}

    reason_by_ticker = {r["ticker"]: r["reason"] for r in out["refusals"]}
    assert reason_by_ticker["JPM"] == "single_name_breach"
    assert reason_by_ticker["AMD"] == "sector_cap_breach"

    # Per-row outcome shape carries the verdict back to the operator.
    by_ticker = {o["ticker"]: o for o in out["outcomes"]}
    assert by_ticker["AAPL"]["opened"] is True
    assert by_ticker["AAPL"]["refusalReason"] is None
    assert by_ticker["JPM"]["opened"] is False
    assert by_ticker["JPM"]["refusalReason"] == "tracker_rejected"


def test_negative_ev_is_refused_not_rescued():
    """§2 no-rescue: a non-positive-EV row is refused at the D16 launch gate
    inside ``consume_ranker_row`` (token issuance) — the wire never rescues a
    negative-EV candidate, even though the caps themselves are refuse-only.
    ``ev_authority_refused`` is distinct from a cap breach."""
    out = _preview([_row("AAPL", 50.0, -5.0)])

    assert out["opened"] == 0
    assert out["refused"] == 1
    assert out["book"]["positions"] == []
    assert out["outcomes"][0]["refusalReason"] == "ev_authority_refused"


def test_metadata_and_caps_surfaced():
    """The response surfaces the armed cap percentages, the authority label,
    and the §2/§3 framing so the surface can never be mistaken for an order
    path."""
    out = _preview([_row("AAPL", 50.0, 20.0)])

    assert out["authority"] == "ev_ranked_concentration_gated"
    assert out["surface"] == "concentration_preview"
    assert out["caps"] == {"sector_cap_pct": 0.25, "single_name_cap_pct": 0.10}
    assert out["entry_date"] == "2026-01-05"
    assert out["initial_capital"] == 100_000.0
    # The note states the non-negotiables: ephemeral, refuse-only, no routing.
    note = out["note"].lower()
    assert "no orders are routed" in note
    assert "refuse" in note


def test_diversified_book_all_open_positive_control():
    """Positive control: small positions under both caps all open — the caps
    refuse concentration, not trading."""
    rows = [
        _row("AAPL", 50.0, 20.0),  # 5% NAV, Info Tech
        _row("JPM", 50.0, 19.0),  # 5% NAV, Financials
        _row("XOM", 50.0, 18.0),  # 5% NAV, Energy
    ]
    out = _preview(rows)
    assert out["opened"] == 3
    assert out["refused"] == 0
    assert set(out["book"]["positions"]) == {"AAPL", "JPM", "XOM"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
