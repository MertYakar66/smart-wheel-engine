"""Parameter-binding tests for the EV-authority token (brain-audit M1).

D16 introduced a single-use token to gate ``open_short_put`` /
``open_covered_call`` on a prior EV ranking. Before this fix the token was
an unbound bearer token: a token issued for AAPL/180/dte32 would successfully
gate an open for ZZZQ/50 (cross-ticker), and a put-issued token would gate
an ``open_covered_call`` (cross-leg).

These tests lock in the binding semantics introduced in the brain-audit M1
fix: the token is now bound to (ticker, strike, derived-dte, side) at
issuance via ``WheelTracker._ev_authority_payloads``. The SHA-256 token hash
is *not* changed; token strings are byte-identical across versions.

Rejection policy:
- ``token_param_mismatch`` — any bound field diverges; token RETAINED.
- ``unbound_token`` — token in set but no payload (fail-closed).

Both reasons are registered in ``_VALID_SHAPES`` in
``tests/test_ev_authority_log_schema.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from engine.wheel_tracker import PositionState, WheelPosition, WheelTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strict_tracker(initial_capital: float = 10_000_000) -> WheelTracker:
    return WheelTracker(initial_capital=initial_capital, require_ev_authority=True)


def _ev_row(**overrides) -> dict:
    """Canonical EV row.  entry 2026-03-20, expiry 2026-04-21 → dte=32."""
    row = {
        "ticker": "AAPL",
        "strike": 180.0,
        "premium": 2.50,
        "dte": 32,
        "ev_dollars": 25.0,
        "prob_profit": 0.72,
        "distribution_source": "empirical_non_overlapping",
        "iv": 0.25,
    }
    row.update(overrides)
    return row


ENTRY = date(2026, 3, 20)
EXPIRY = date(2026, 4, 21)  # 32 days from ENTRY


def _open_put(t: WheelTracker, **overrides) -> bool:
    """Call open_short_put with default matching params."""
    kwargs: dict = {
        "ticker": "AAPL",
        "strike": 180.0,
        "premium": 2.50,
        "entry_date": ENTRY,
        "expiration_date": EXPIRY,
        "iv": 0.25,
    }
    kwargs.update(overrides)
    return t.open_short_put(**kwargs)  # type: ignore[arg-type]


def _reject_reasons(t: WheelTracker) -> list[str]:
    return [e.get("reason", "") for e in t._ev_authority_log if e.get("action") == "reject"]


# ---------------------------------------------------------------------------
# Mismatch refusals
# ---------------------------------------------------------------------------


class TestMismatchRefusals:
    """Each test verifies one dimension of the binding."""

    def test_mismatched_ticker_refused(self):
        """A token issued for AAPL must NOT gate an open for MSFT."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())

        ok = _open_put(t, ticker="MSFT", ev_authority_token=token, current_ev_dollars=25.0)

        assert ok is False
        assert "MSFT" not in t.positions
        reasons = _reject_reasons(t)
        assert "token_param_mismatch" in reasons
        mismatch_entry = next(
            e for e in t._ev_authority_log if e.get("reason") == "token_param_mismatch"
        )
        assert "ticker" in mismatch_entry["mismatched_fields"]
        # Token is RETAINED — the genuine AAPL open can still use it.
        assert token in t._ev_authority_tokens

    def test_mismatched_strike_refused(self):
        """Strike 185 on a 180-issued token must be refused."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())  # strike=180

        ok = _open_put(t, strike=185.0, ev_authority_token=token, current_ev_dollars=25.0)

        assert ok is False
        reasons = _reject_reasons(t)
        assert "token_param_mismatch" in reasons
        mismatch_entry = next(
            e for e in t._ev_authority_log if e.get("reason") == "token_param_mismatch"
        )
        assert "strike" in mismatch_entry["mismatched_fields"]
        assert token in t._ev_authority_tokens

    def test_mismatched_expiration_refused(self):
        """An expiration 7 days later yields a different derived dte → mismatch."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())  # dte=32

        wrong_expiry = EXPIRY + timedelta(days=7)  # derived dte = 39
        ok = _open_put(
            t, expiration_date=wrong_expiry, ev_authority_token=token, current_ev_dollars=25.0
        )

        assert ok is False
        reasons = _reject_reasons(t)
        assert "token_param_mismatch" in reasons
        mismatch_entry = next(
            e for e in t._ev_authority_log if e.get("reason") == "token_param_mismatch"
        )
        assert "dte" in mismatch_entry["mismatched_fields"]
        assert token in t._ev_authority_tokens

    def test_cross_leg_side_refused(self):
        """A put-issued token must NOT gate open_covered_call (cross-leg hole)."""
        t = _strict_tracker()
        # Pre-load a STOCK_OWNED position so the call-leg can proceed if it passes the gate.
        t.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            state=PositionState.STOCK_OWNED,
            entry_date=date(2026, 3, 1),
            stock_shares=100,
            stock_basis=180.0,
        )
        # Issue a side-bound put token.
        token = t.issue_ev_authority_token(_ev_row(), side="put")

        ok = t.open_covered_call(
            ticker="AAPL",
            strike=180.0,
            premium=2.50,
            entry_date=ENTRY,
            expiration_date=EXPIRY,
            iv=0.25,
            ev_authority_token=token,
            current_ev_dollars=25.0,
        )

        assert ok is False
        reasons = _reject_reasons(t)
        assert "token_param_mismatch" in reasons
        mismatch_entry = next(
            e for e in t._ev_authority_log if e.get("reason") == "token_param_mismatch"
        )
        assert "side" in mismatch_entry["mismatched_fields"]
        # Token retained — the genuine put open can still use it.
        assert token in t._ev_authority_tokens


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestExactMatchOpens:
    def test_exact_match_opens(self):
        """Full param match → position opens, token discarded, payload removed."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())

        ok = _open_put(t, ev_authority_token=token, current_ev_dollars=25.0)

        assert ok is True
        assert "AAPL" in t.positions
        assert token not in t._ev_authority_tokens
        assert token not in t._ev_authority_payloads

    def test_single_use_still_enforced_after_binding(self):
        """After a successful exact-match consume, a retry with identical params
        must fail with ``unknown_token`` — binding did not weaken single-use."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())

        ok1 = _open_put(t, ev_authority_token=token, current_ev_dollars=25.0)
        assert ok1 is True

        # Pop position so the ticker slot is free for the retry.
        t.positions.pop("AAPL", None)

        ok2 = _open_put(t, ev_authority_token=token, current_ev_dollars=25.0)
        assert ok2 is False
        reasons = _reject_reasons(t)
        assert "unknown_token" in reasons

    def test_mismatch_retains_token_then_exact_match_succeeds(self):
        """The corrected d6 probe-B/C sequence:
        1. ZZZQ open with AAPL token → refused (token_param_mismatch), token retained.
        2. AAPL open with the same token → succeeds.
        (Pre-fix: step 1 returned True and step 2 returned False because
        the token was consumed by the ZZZQ open.)
        """
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())

        # Step 1: cross-ticker open — must be refused, token retained.
        ok_wrong = _open_put(t, ticker="ZZZQ", ev_authority_token=token, current_ev_dollars=25.0)
        assert ok_wrong is False
        assert token in t._ev_authority_tokens

        # Step 2: correct open — must succeed.
        ok_right = _open_put(t, ev_authority_token=token, current_ev_dollars=25.0)
        assert ok_right is True
        assert "AAPL" in t.positions
        assert token not in t._ev_authority_tokens


# ---------------------------------------------------------------------------
# Unbound-token (fail-closed)
# ---------------------------------------------------------------------------


class TestUnboundToken:
    def test_unbound_token_refused_fail_closed(self):
        """A token injected directly into ``_ev_authority_tokens`` with no payload
        must be refused with ``unbound_token`` and the token must be retained.

        Rationale: a token whose issuing candidate cannot be verified cannot prove
        EV authority.  The legacy-rebuild path in ``from_dict`` means genuine old
        snapshots never reach this path.
        """
        t = _strict_tracker()
        fake_token = "a" * 64  # 64-hex-char string, not a real token
        t._ev_authority_tokens.add(fake_token)
        # No matching payload entry — _ev_authority_payloads is empty.

        ok = _open_put(t, ev_authority_token=fake_token, current_ev_dollars=25.0)

        assert ok is False
        reasons = _reject_reasons(t)
        assert "unbound_token" in reasons
        # Token retained.
        assert fake_token in t._ev_authority_tokens


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_legacy_snapshot_rebinds_from_log(self):
        """A snapshot whose ``ev_authority_payloads`` key is absent (pre-fix)
        but whose audit log carries the issue rows must rebind on from_dict.
        After rebinding: exact-match consume succeeds AND mismatched-ticker
        consume is refused with token_param_mismatch.
        """
        # Build a real snapshot.
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())  # side='' (legacy)
        d = t.to_dict()
        # Simulate a pre-fix snapshot by stripping the new key.
        d.pop("ev_authority_payloads", None)

        # Reload from the legacy snapshot.
        t2 = WheelTracker.from_dict(d)

        # Mismatched ticker → refused with token_param_mismatch (not unbound_token).
        ok_wrong = _open_put(t2, ticker="MSFT", ev_authority_token=token, current_ev_dollars=25.0)
        assert ok_wrong is False
        assert "token_param_mismatch" in _reject_reasons(t2)
        assert token in t2._ev_authority_tokens

        # Exact match → succeeds (token was rebuilt, not just accepted naively).
        ok_right = _open_put(t2, ev_authority_token=token, current_ev_dollars=25.0)
        assert ok_right is True

    def test_payload_binding_survives_persistence_round_trip(self):
        """issue → to_dict → from_dict → mismatched-ticker consume refused on reload."""
        t = _strict_tracker()
        token = t.issue_ev_authority_token(_ev_row())

        t2 = WheelTracker.from_dict(t.to_dict())

        ok = _open_put(t2, ticker="MSFT", ev_authority_token=token, current_ev_dollars=25.0)
        assert ok is False
        assert "token_param_mismatch" in _reject_reasons(t2)


# ---------------------------------------------------------------------------
# Schema-closure integration
# ---------------------------------------------------------------------------


class TestSchemaShapes:
    """Drive the real tracker paths that emit the two new reasons and assert
    that the schema validator from test_ev_authority_log_schema accepts them."""

    def test_token_param_mismatch_shape_validates(self):
        from tests.test_ev_authority_log_schema import _validate_entry

        t = _strict_tracker()
        t.issue_ev_authority_token(_ev_row())
        token = list(t._ev_authority_tokens)[0]

        _open_put(t, ticker="MSFT", ev_authority_token=token, current_ev_dollars=25.0)

        mismatch_entries = [
            e for e in t._ev_authority_log if e.get("reason") == "token_param_mismatch"
        ]
        assert mismatch_entries, "No token_param_mismatch entry found"
        _validate_entry(mismatch_entries[0])

    def test_unbound_token_shape_validates(self):
        from tests.test_ev_authority_log_schema import _validate_entry

        t = _strict_tracker()
        fake = "b" * 64
        t._ev_authority_tokens.add(fake)

        _open_put(t, ev_authority_token=fake, current_ev_dollars=25.0)

        unbound_entries = [e for e in t._ev_authority_log if e.get("reason") == "unbound_token"]
        assert unbound_entries, "No unbound_token entry found"
        _validate_entry(unbound_entries[0])


# ---------------------------------------------------------------------------
# Non-strict mode untouched
# ---------------------------------------------------------------------------


class TestNonStrictModeUntouched:
    def test_non_strict_mode_ignores_binding(self):
        """require_ev_authority=False: open succeeds with no token, no params — research
        path is byte-for-byte unchanged."""
        t = WheelTracker(initial_capital=100_000)  # non-strict
        ok = _open_put(t)
        assert ok is True
        assert "AAPL" in t.positions


# ---------------------------------------------------------------------------
# consume_ranker_row explicit-expiration escape hatch (BLOCKER resolution)
# ---------------------------------------------------------------------------


class TestConsumeRankerRowEscapeHatch:
    """Validate that consume_ranker_row with an explicit expiration_date that
    diverges from entry_date + row['dte'] still opens successfully (the
    calendar-exact control escape hatch documented in the method docstring).

    Pre-fix, this would self-refuse with token_param_mismatch because the
    token payload had dte=row['dte'] but the open's derived dte differed.
    Post-fix, consume_ranker_row overwrites dte with the derived dte before
    issuing, so issue == consume exactly.
    """

    def test_explicit_expiration_escape_hatch_succeeds(self):
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner()
        ev_df = runner.rank_candidates_by_ev(
            tickers=["AAPL"],
            top_n=1,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        if ev_df.empty:
            pytest.skip("No AAPL row from connector — skipping escape-hatch test")

        ev_row = ev_df.iloc[0].to_dict()
        if float(ev_row.get("ev_dollars", 0.0) or 0.0) <= 0.0:
            ev_row["ev_dollars"] = 25.0

        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        entry = date(2026, 4, 14)
        # 67-day explicit expiration — diverges from row['dte'] (≈35).
        explicit_expiry = entry + timedelta(days=67)

        ok = t.consume_ranker_row(ev_row, entry_date=entry, expiration_date=explicit_expiry)

        # Must succeed (escape hatch preserved).
        assert ok is True, (
            f"consume_ranker_row self-refused with explicit 67-day expiry. "
            f"Log: {t._ev_authority_log}"
        )
        assert str(ev_row["ticker"]) in t.positions

    def test_explicit_expiration_mismatch_pinned_in_payload(self):
        """After consume_ranker_row with explicit expiry, the token payload's
        dte equals the ACTUAL derived dte, not row['dte']."""
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        row = {
            "ticker": "AAPL",
            "strike": 180.0,
            "premium": 2.50,
            "dte": 35,  # row says 35
            "ev_dollars": 25.0,
            "prob_profit": 0.72,
            "distribution_source": "empirical_non_overlapping",
            "iv": 0.25,
        }
        entry = date(2026, 4, 14)
        explicit_expiry = entry + timedelta(days=67)  # derived dte = 67

        # Issue is about to happen — capture token before it's consumed.
        # We'll inspect the payload after the call (it will be cleaned up on
        # success, so we peek via a subclass hook instead — easier: just look at
        # the log's issue entry and verify the dte in the canonical row).
        ok = t.consume_ranker_row(row, entry_date=entry, expiration_date=explicit_expiry)
        assert ok is True

        issue_entries = [e for e in t._ev_authority_log if e.get("action") == "issue"]
        assert issue_entries, "No issue entry in log"
        assert issue_entries[0]["row"]["dte"] == 67, (
            f"Expected payload dte=67 (derived from explicit expiry), "
            f"got {issue_entries[0]['row']['dte']}"
        )
