"""Invariant pins for the production-armed concentration caps (R9/R10).

heavy-verify 2026-05-31 Category A / #154 follow-up. The D17 concentration
caps were dormant on every path (bundled behind ``require_ev_authority``).
The structural fix decouples them into per-cap ``enforce_*`` flags that are
evaluable WITHOUT ``require_ev_authority`` or a D16 token. The library default
leaves them OFF (research-safe, matching the ``require_ev_authority``
convention); the canonical production constructor
``engine.wheel_runner.make_live_book_tracker`` ARMS R9 (sector) + R10
(single-name). These tests pin that production-arming so a future change
cannot silently drop it — and that the library default is unchanged.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.wheel_runner import make_live_book_tracker
from engine.wheel_tracker import WheelTracker

ENTRY = date(2026, 1, 5)
EXPIRY = date(2026, 2, 9)

# The reject-audit dict shape (D16/D17). A future change must not silently
# alter it — downstream audit consumers (HT-D, _ev_authority_log readers) rely
# on exactly these keys being present on every reject entry.
_REJECT_COMMON_KEYS = {
    "action",
    "ticker",
    "token",
    "current_ev_dollars",
    "nav",
    "nav_source",
    "reason",
}


def test_production_factory_arms_only_r9_r10():
    """Production tracker arms R9+R10, leaves delta/Kelly off and stays
    token-free (decoupled from require_ev_authority)."""
    t = make_live_book_tracker(initial_capital=100_000.0)
    assert t.enforce_sector_cap is True
    assert t.enforce_single_name_cap is True
    assert t.enforce_delta_cap is False  # deferred until re-calibrated
    assert t.enforce_kelly_cap is False
    assert t.require_ev_authority is False  # caps run WITHOUT the D16 token


def test_production_refuses_single_name_over_10pct():
    """R10: a single short put worth >10% NAV is refused on the production
    path — token-free — with the canonical reject-audit shape."""
    t = make_live_book_tracker(initial_capital=100_000.0)
    # $200 strike => $20,000 notional = 20% of $100k NAV > 10% single-name cap.
    opened = t.open_short_put("AAPL", 200.0, 2.0, ENTRY, EXPIRY, 0.30)
    assert opened is False
    assert "AAPL" not in t.positions
    assert t._ev_authority_log, "a reject entry must be logged"
    entry = t._ev_authority_log[-1]
    assert entry["reason"] == "single_name_breach"
    assert entry["action"] == "reject"
    assert entry["ticker"] == "AAPL"
    assert entry["token"] is None  # token-free path (no require_ev_authority)
    # reject-audit dict shape unchanged:
    assert _REJECT_COMMON_KEYS.issubset(entry.keys())


def test_production_refuses_sector_over_25pct():
    """R9: a book that pushes one GICS sector >25% NAV is refused — even when
    every individual name is under the 10% single-name floor."""
    t = make_live_book_tracker(initial_capital=100_000.0)
    # Three Information-Technology puts at $80 strike = $8k each = 8% NAV each
    # (< 10% single-name) -> sector reaches 24% (< 25%), all open.
    for sym in ("AAPL", "MSFT", "NVDA"):
        assert t.open_short_put(sym, 80.0, 1.5, ENTRY, EXPIRY, 0.30) is True, sym
    # A 4th IT name (8%) would push the sector to 32% > 25% -> refused by R9.
    opened = t.open_short_put("AMD", 80.0, 1.5, ENTRY, EXPIRY, 0.30)
    assert opened is False
    assert "AMD" not in t.positions
    entry = t._ev_authority_log[-1]
    assert entry["reason"] == "sector_cap_breach"
    assert _REJECT_COMMON_KEYS.issubset(entry.keys())


def test_production_allows_a_diversified_book():
    """Positive control: small, diversified positions (each < both caps) open
    fine on the production path — the caps refuse concentration, not trading."""
    t = make_live_book_tracker(initial_capital=100_000.0)
    # $50 strike = $5k = 5% NAV each, different sectors -> well within caps.
    assert t.open_short_put("AAPL", 50.0, 1.0, ENTRY, EXPIRY, 0.30) is True
    assert t.open_short_put("JPM", 50.0, 1.0, ENTRY, EXPIRY, 0.30) is True
    assert "AAPL" in t.positions and "JPM" in t.positions


def test_library_default_is_unchanged_caps_off():
    """Backwards-compat invariant: the plain ``WheelTracker(...)`` default
    does NOT enforce the caps (research/test path unchanged). The exact open
    that the production tracker refuses must succeed on the default tracker."""
    t = WheelTracker(initial_capital=100_000.0)
    assert t.enforce_sector_cap is False
    assert t.enforce_single_name_cap is False
    # The 20%-NAV single-name open refused in production opens here:
    assert t.open_short_put("AAPL", 200.0, 2.0, ENTRY, EXPIRY, 0.30) is True
    assert "AAPL" in t.positions


def test_strict_mode_still_arms_all_four():
    """require_ev_authority=True continues to arm all four D17 gates
    regardless of the enforce_* flags (existing token-gated behaviour)."""
    t = WheelTracker(initial_capital=100_000.0, require_ev_authority=True)
    for gate in ("sector", "single_name", "delta", "kelly"):
        assert t._d17_gate_enabled(gate) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
