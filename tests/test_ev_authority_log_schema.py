"""Schema-closure regression for ``WheelTracker._ev_authority_log``.

The companion to ``test_authority_hardening.py``: where that file
asserts *behaviour* (refuse on non-positive EV, retain on stale-EV,
both legs gated, etc.), this file asserts *shape*.

D16 introduced five distinct entry shapes the audit log can carry.
Individual behaviour tests assert specific fields per scenario but
none of them enforce **closure** — i.e. "any entry the tracker writes
matches exactly one of these five shapes, with no unexpected extras
and no missing required keys." Without closure, three failure modes
are silent:

1. A future patch adds a new ``action`` value or ``reason`` variant
   that no downstream consumer (audit script, dashboard, persisted
   replay) is prepared to parse.
2. A key gets renamed (``token`` → ``token_hash``) and tests still
   pass because no behavioural assertion touched the old name.
3. PR #128 persists ``_ev_authority_log`` to disk via
   ``to_dict`` / ``from_dict``; a shape drift would corrupt the
   on-disk JSON, and the regression would only surface the next
   time someone tries to ``load`` a tracker.

These tests exercise every code path that writes to the log and
verify the resulting entries against an explicit schema. If you add
a new entry shape, update ``_VALID_SHAPES`` below and the tests will
re-pass; if you forget, this file will fail.

See `DECISIONS.md` D16 for the contract and rationale.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.wheel_tracker import EVAuthorityRefused, WheelTracker

# ----------------------------------------------------------------------
# Schema definition — the five valid shapes for a single log entry.
#
# Each shape is (required_keys, optional_keys). A valid entry has
# every required key and only required+optional keys (no extras).
# ----------------------------------------------------------------------

# Common keys
_ACTION = "action"

# Shape 1: action="issue" — token minted for a positive-EV row.
_SHAPE_ISSUE = (
    {_ACTION, "token", "row"},
    set(),
)

# Shape 2: action="refuse_issue" — non-positive EV refused at issue.
_SHAPE_REFUSE_ISSUE = (
    {_ACTION, "reason", "row"},
    set(),
)

# Shape 3: action="consume" — successful single-use consume.
_SHAPE_CONSUME = (
    {_ACTION, "token", "ticker", "current_ev_dollars"},
    set(),
)

# Shape 4a: action="reject", reason="unknown_token" — no/unknown token.
_SHAPE_REJECT_UNKNOWN = (
    {_ACTION, "reason", "ticker"},
    set(),
)

# Shape 4b: action="reject", reason="missing_current_ev_dollars".
# Token is known at this point so it's carried.
_SHAPE_REJECT_MISSING_EV = (
    {_ACTION, "reason", "ticker", "token"},
    set(),
)

# Shape 4c: action="reject", reason="stale_ev" — fire-time EV non-positive.
# Carries token AND the rejected current_ev_dollars for the audit trail.
_SHAPE_REJECT_STALE = (
    {_ACTION, "reason", "ticker", "token", "current_ev_dollars"},
    set(),
)

# D17 (#154 C4 Phase 2) — four new shapes for the portfolio-risk
# hard-blocks. All carry the post-D16 fields (token, current_ev_dollars)
# plus the gate-specific details bag and the live-NAV fingerprint
# (nav + nav_source) so an audit consumer can grep by nav_source to
# spot static-fallback vs live-mark-to-market gate runs.

# Shape 5a: action="reject", reason="nav_exhausted" — pre-gate floor.
_SHAPE_REJECT_NAV_EXHAUSTED = (
    {
        _ACTION,
        "reason",
        "ticker",
        "token",
        "current_ev_dollars",
        "nav",
        "nav_source",
        "min_nav_for_trading",
    },
    set(),
)

# Shape 5b: action="reject", reason="sector_cap_breach".
_SHAPE_REJECT_SECTOR = (
    {
        _ACTION,
        "reason",
        "ticker",
        "token",
        "current_ev_dollars",
        "nav",
        "nav_source",
        "sector",
        "post_open_sector_pct",
        "sector_limit",
        "narrative",
    },
    set(),
)

# Shape 5c: action="reject", reason="portfolio_delta_breach".
_SHAPE_REJECT_DELTA = (
    {
        _ACTION,
        "reason",
        "ticker",
        "token",
        "current_ev_dollars",
        "nav",
        "nav_source",
        "current_portfolio_delta_dollars",
        "post_open_delta_dollars",
        "delta_cap_dollars",
    },
    set(),
)

# Shape 5d: action="reject", reason="kelly_size_exceeded".
#
# Coverage note (#166 B1): this shape is verified by the static
# schema in this file only — no end-to-end tracker integration test
# in TestD17HardBlocks exercises the Kelly gate. The gate is
# preemptively reserved for a future multi-contract position path
# (see engine/portfolio_risk_gates.check_kelly_size's docstring and
# #166 B3); under WheelTracker.open_short_put's current single-
# contract emission it is structurally unreachable at realistic NAVs.
# When the multi-contract WheelPosition path lands, add a tracker
# integration test mirroring the sector / delta cases in
# tests/test_authority_hardening.TestD17HardBlocks and remove this
# note.
_SHAPE_REJECT_KELLY = (
    {
        _ACTION,
        "reason",
        "ticker",
        "token",
        "current_ev_dollars",
        "nav",
        "nav_source",
        "margin_required",
        "kelly_recommended_max",
        "kelly_fraction",
    },
    set(),
)

# Shape 5e: action="reject", reason="single_name_breach". Tighter
# per-underlying floor that sits beneath the sector cap. Bounds
# F4-style idiosyncratic-drawdown damage.
_SHAPE_REJECT_SINGLE_NAME = (
    {
        _ACTION,
        "reason",
        "ticker",
        "token",
        "current_ev_dollars",
        "nav",
        "nav_source",
        "symbol",
        "current_name_notional",
        "post_open_name_notional",
        "post_open_name_pct",
        "name_limit_pct",
    },
    set(),
)

_VALID_SHAPES: dict[tuple[str, str | None], tuple[set[str], set[str]]] = {
    ("issue", None): _SHAPE_ISSUE,
    ("refuse_issue", "non_positive_ev"): _SHAPE_REFUSE_ISSUE,
    ("consume", None): _SHAPE_CONSUME,
    ("reject", "unknown_token"): _SHAPE_REJECT_UNKNOWN,
    ("reject", "missing_current_ev_dollars"): _SHAPE_REJECT_MISSING_EV,
    ("reject", "stale_ev"): _SHAPE_REJECT_STALE,
    # D17 hard-block reject shapes (#154 C4 Phase 2)
    ("reject", "nav_exhausted"): _SHAPE_REJECT_NAV_EXHAUSTED,
    ("reject", "sector_cap_breach"): _SHAPE_REJECT_SECTOR,
    ("reject", "portfolio_delta_breach"): _SHAPE_REJECT_DELTA,
    ("reject", "kelly_size_exceeded"): _SHAPE_REJECT_KELLY,
    # F4 damage-bounding addition
    ("reject", "single_name_breach"): _SHAPE_REJECT_SINGLE_NAME,
}


def _classify(entry: dict) -> tuple[str, str | None]:
    """Map an entry to its (action, reason) key in ``_VALID_SHAPES``."""
    action = entry.get(_ACTION)
    reason = entry.get("reason") if action in {"refuse_issue", "reject"} else None
    return action, reason


def _validate_entry(entry: dict) -> None:
    """Assert a single log entry matches one of the five shapes."""
    key = _classify(entry)
    assert key in _VALID_SHAPES, (
        f"Unknown audit-log entry classification {key!r}; "
        f"if you added a new (action, reason) pair, update "
        f"_VALID_SHAPES in this file. Entry: {entry!r}"
    )
    required, optional = _VALID_SHAPES[key]
    keys = set(entry.keys())
    missing = required - keys
    extra = keys - required - optional
    assert not missing, f"Entry shape {key!r} is missing required keys {sorted(missing)}: {entry!r}"
    assert not extra, (
        f"Entry shape {key!r} has unexpected extra keys {sorted(extra)}; "
        f"if you added a new field, decide whether it belongs in the "
        f"required or optional set of _VALID_SHAPES[{key!r}]. "
        f"Entry: {entry!r}"
    )


def _validate_log(tracker: WheelTracker) -> None:
    """Assert every entry in the tracker's audit log is well-shaped."""
    for i, entry in enumerate(tracker._ev_authority_log):
        try:
            _validate_entry(entry)
        except AssertionError as e:
            raise AssertionError(f"Audit-log entry [{i}] invalid: {e}") from e


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _ev_row(**overrides) -> dict:
    """Canonical-shaped EV row. Overrides win."""
    row = {
        "ticker": "TEST",
        "strike": 95.0,
        "premium": 1.20,
        "dte": 35,
        "ev_dollars": 25.0,
        "prob_profit": 0.72,
        "distribution_source": "empirical_non_overlapping",
    }
    row.update(overrides)
    return row


def _open_args(**overrides) -> dict:
    """Standard kwargs for ``open_short_put``. Overrides win."""
    base = {
        "ticker": "TEST",
        "strike": 95.0,
        "premium": 1.20,
        "entry_date": date(2026, 4, 14),
        "expiration_date": date(2026, 5, 19),
        "iv": 0.25,
    }
    base.update(overrides)
    return base


# ======================================================================
# 1. Each individual code path produces a well-shaped entry.
# ======================================================================
class TestPerPathShape:
    def test_issue_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        t.issue_ev_authority_token(_ev_row())
        assert len(t._ev_authority_log) == 1
        e = t._ev_authority_log[0]
        assert e["action"] == "issue"
        _validate_entry(e)

    def test_refuse_issue_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        with pytest.raises(EVAuthorityRefused):
            t.issue_ev_authority_token(_ev_row(ev_dollars=-5.0))
        assert len(t._ev_authority_log) == 1
        e = t._ev_authority_log[0]
        assert e["action"] == "refuse_issue"
        assert e["reason"] == "non_positive_ev"
        _validate_entry(e)

    def test_consume_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=25.0,
        )
        assert ok is True
        # log: issue, consume
        consume = [e for e in t._ev_authority_log if e.get("action") == "consume"]
        assert len(consume) == 1
        _validate_entry(consume[0])

    def test_reject_unknown_token_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        # No token issued — supplying a bad one hits unknown_token.
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token="0" * 64,
            current_ev_dollars=25.0,
        )
        assert ok is False
        rejects = [e for e in t._ev_authority_log if e.get("action") == "reject"]
        assert len(rejects) == 1
        assert rejects[0]["reason"] == "unknown_token"
        _validate_entry(rejects[0])

    def test_reject_unknown_token_shape_when_none(self):
        """Passing ``ev_authority_token=None`` in strict mode also
        logs ``unknown_token`` (no separate ``none_token`` shape)."""
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=None,
            current_ev_dollars=25.0,
        )
        assert ok is False
        rejects = [e for e in t._ev_authority_log if e.get("action") == "reject"]
        assert len(rejects) == 1
        assert rejects[0]["reason"] == "unknown_token"
        _validate_entry(rejects[0])

    def test_reject_missing_current_ev_dollars_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=None,
        )
        assert ok is False
        rejects = [e for e in t._ev_authority_log if e.get("action") == "reject"]
        assert len(rejects) == 1
        assert rejects[0]["reason"] == "missing_current_ev_dollars"
        _validate_entry(rejects[0])

    def test_reject_stale_ev_shape(self):
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=-1.0,
        )
        assert ok is False
        rejects = [e for e in t._ev_authority_log if e.get("action") == "reject"]
        assert len(rejects) == 1
        assert rejects[0]["reason"] == "stale_ev"
        _validate_entry(rejects[0])


# ======================================================================
# 2. The closure property — every entry, across mixed sequences,
#    matches one of the five shapes.
# ======================================================================
class TestLogClosure:
    def test_mixed_sequence_closure(self):
        """A realistic mixed sequence of operations writes only
        well-shaped entries — no key drift, no unexpected actions."""
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)

        # issue + immediate refuse_issue
        with pytest.raises(EVAuthorityRefused):
            t.issue_ev_authority_token(_ev_row(ticker="BAD", ev_dollars=-1.0))
        token_a = t.issue_ev_authority_token(_ev_row(ticker="AAA"))

        # unknown_token reject
        t.open_short_put(
            **_open_args(ticker="UNK"),
            ev_authority_token="deadbeef" * 8,
            current_ev_dollars=10.0,
        )

        # missing_current_ev_dollars reject (token retained)
        t.open_short_put(
            **_open_args(ticker="AAA"),
            ev_authority_token=token_a,
            current_ev_dollars=None,
        )

        # stale_ev reject (token retained)
        t.open_short_put(
            **_open_args(ticker="AAA"),
            ev_authority_token=token_a,
            current_ev_dollars=-0.5,
        )

        # successful consume on same token (retained twice, now fired)
        ok = t.open_short_put(
            **_open_args(ticker="AAA"),
            ev_authority_token=token_a,
            current_ev_dollars=12.0,
        )
        assert ok is True

        # closure: every entry is one of the five shapes
        _validate_log(t)

        # spot-checks on the witnessed action vocabulary
        actions = [e["action"] for e in t._ev_authority_log]
        assert "refuse_issue" in actions
        assert "issue" in actions
        assert "reject" in actions
        assert "consume" in actions

        # spot-check the per-reason coverage
        reject_reasons = {e.get("reason") for e in t._ev_authority_log if e["action"] == "reject"}
        assert reject_reasons == {
            "unknown_token",
            "missing_current_ev_dollars",
            "stale_ev",
        }

    def test_closure_survives_persistence_round_trip(self):
        """The persisted log shape (PR #128) is the same as the
        in-memory shape. If the schema drifts, ``from_dict`` would
        reconstruct entries that fail validation."""
        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=25.0,
        )

        back = WheelTracker.from_dict(t.to_dict())
        # Same entries, same shapes — and the schema validator must
        # accept the reloaded log unchanged.
        assert back._ev_authority_log == t._ev_authority_log
        _validate_log(back)

    def test_covered_call_path_closure(self):
        """D16 brought ``open_covered_call`` under the same gate.
        Both reject and consume entries it writes must match the
        same shapes the put leg uses."""
        from engine.wheel_tracker import PositionState, WheelPosition

        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        t.positions["AAA"] = WheelPosition(
            ticker="AAA",
            state=PositionState.STOCK_OWNED,
            entry_date=date(2026, 4, 1),
            stock_shares=100,
            stock_basis=180.0,
        )

        # No token → unknown_token reject from the call leg.
        t.open_covered_call(
            ticker="AAA",
            strike=190.0,
            premium=1.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.22,
            ev_authority_token=None,
            current_ev_dollars=18.0,
        )

        # Valid token + positive EV → successful call-leg consume.
        token = t.issue_ev_authority_token(
            _ev_row(ticker="AAA", strike=190.0, premium=1.50, ev_dollars=18.0)
        )
        ok = t.open_covered_call(
            ticker="AAA",
            strike=190.0,
            premium=1.50,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 16),
            iv=0.22,
            ev_authority_token=token,
            current_ev_dollars=18.0,
        )
        assert ok is True

        _validate_log(t)


# ======================================================================
# 3. The validator itself must reject known-bad shapes.
#    (Meta-test: a faulty validator that accepts everything would
#    silently let the closure tests pass.)
# ======================================================================
class TestValidatorRejectsBadShapes:
    def test_validator_rejects_unknown_action(self):
        with pytest.raises(AssertionError, match="Unknown audit-log entry"):
            _validate_entry({"action": "ship_to_broker", "ticker": "AAA"})

    def test_validator_rejects_missing_required_key(self):
        # action=issue requires token + row; this entry is missing both.
        with pytest.raises(AssertionError, match="missing required keys"):
            _validate_entry({"action": "issue"})

    def test_validator_rejects_extra_key(self):
        # action=consume is well-formed below except for the extra
        # ``mood`` field. Closure must fail on the extra.
        with pytest.raises(AssertionError, match="unexpected extra keys"):
            _validate_entry(
                {
                    "action": "consume",
                    "token": "abc",
                    "ticker": "AAA",
                    "current_ev_dollars": 12.0,
                    "mood": "confident",
                }
            )

    def test_validator_rejects_unknown_reject_reason(self):
        with pytest.raises(AssertionError, match="Unknown audit-log entry"):
            _validate_entry(
                {
                    "action": "reject",
                    "reason": "vibes_off",
                    "ticker": "AAA",
                }
            )


# ======================================================================
# 4. D17 (#154 C4 Phase 2) — schema for portfolio-risk hard-block
#    reject entries. Each path through the tracker's
#    _evaluate_d17_hard_blocks emits an entry that must match one of
#    the four new D17 shapes registered in _VALID_SHAPES.
# ======================================================================
class TestD17EntryShapes:
    def test_nav_exhausted_shape(self):
        """min_nav_for_trading=$1M; tracker capital $100k → nav < floor."""
        t = WheelTracker(
            initial_capital=100_000,
            require_ev_authority=True,
            min_nav_for_trading=1_000_000.0,
        )
        token = t.issue_ev_authority_token(_ev_row())
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=25.0,
            prob_profit=0.72,
        )
        assert ok is False
        rejects = [
            e
            for e in t._ev_authority_log
            if e.get("action") == "reject" and e.get("reason") == "nav_exhausted"
        ]
        assert len(rejects) == 1
        _validate_entry(rejects[0])

    def test_portfolio_delta_breach_shape(self):
        """At $100k NAV the delta cap is $300; ATM short put has
        delta-dollars ~$4750. Should trip the delta gate (it sorts
        before Kelly in the orchestration)."""
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        ok = t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=25.0,
            prob_profit=0.72,
        )
        assert ok is False
        delta_rejects = [
            e
            for e in t._ev_authority_log
            if e.get("action") == "reject" and e.get("reason") == "portfolio_delta_breach"
        ]
        assert len(delta_rejects) == 1
        _validate_entry(delta_rejects[0])

    def test_sector_cap_breach_shape(self):
        """Force the sector gate to fire by pre-loading the held
        list with same-sector positions. Easiest path: open one
        AAPL short put at $10M NAV (delta passes), then try a
        second AAPL position that would breach sector. Actually
        simpler: instantiate a tracker with a 0.001% sector cap
        override... not supported. Use the gate-direct path
        instead: pre-load self.positions with an AAPL position
        whose strike makes the sector cap tight."""
        from engine.wheel_tracker import PositionState, WheelPosition

        t = WheelTracker(initial_capital=10_000_000, require_ev_authority=True)
        # Pre-load $2M of held AAPL (one $20k put × 100 contracts
        # equivalent is impractical; use the persistence layer
        # injection to set a giant held position).
        t.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            state=PositionState.SHORT_PUT,
            entry_date=date(2026, 4, 1),
            put_strike=2_400_000.0,  # nonsense strike for test setup
            put_premium=1.0,
            put_entry_date=date(2026, 4, 1),
            put_dte_at_entry=30,
            put_entry_iv=0.25,
            put_expiration_date=date(2026, 5, 1),
        )
        token = t.issue_ev_authority_token(
            _ev_row(ticker="MSFT", strike=350.0, premium=2.5, ev_dollars=25.0)
        )
        ok = t.open_short_put(
            ticker="MSFT",
            strike=350.0,
            premium=2.5,
            entry_date=date(2026, 4, 14),
            expiration_date=date(2026, 5, 19),
            iv=0.25,
            ev_authority_token=token,
            current_ev_dollars=25.0,
            prob_profit=0.72,
        )
        assert ok is False
        sector_rejects = [
            e
            for e in t._ev_authority_log
            if e.get("action") == "reject" and e.get("reason") == "sector_cap_breach"
        ]
        assert len(sector_rejects) == 1
        _validate_entry(sector_rejects[0])

    def test_kelly_size_exceeded_shape_via_direct_log_check(self):
        """Validate the kelly_size_exceeded shape through the audit
        log even when delta-breach fires first in the same call.
        We can't easily isolate Kelly at the tracker level, but the
        SHAPE is what this test pins — so we synthesise a kelly
        reject entry directly and validate it."""
        synthesised = {
            "action": "reject",
            "reason": "kelly_size_exceeded",
            "ticker": "TEST",
            "token": "0" * 64,
            "current_ev_dollars": 25.0,
            "nav": 10_000.0,
            "nav_source": "static_fallback",
            "margin_required": 8_000.0,
            "kelly_recommended_max": 5_000.0,
            "kelly_fraction": 0.5,
        }
        _validate_entry(synthesised)

    def test_d17_static_fallback_source_in_audit(self):
        """Connector=None → nav_source='static_fallback' on every
        D17 reject entry."""
        t = WheelTracker(initial_capital=100_000, require_ev_authority=True)
        token = t.issue_ev_authority_token(_ev_row())
        t.open_short_put(
            **_open_args(),
            ev_authority_token=token,
            current_ev_dollars=25.0,
            prob_profit=0.72,
        )
        d17_rejects = [
            e
            for e in t._ev_authority_log
            if e.get("action") == "reject"
            and e.get("reason")
            in {
                "nav_exhausted",
                "sector_cap_breach",
                "portfolio_delta_breach",
                "kelly_size_exceeded",
            }
        ]
        assert len(d17_rejects) >= 1
        for entry in d17_rejects:
            assert entry["nav_source"] == "static_fallback"
            assert entry["nav"] == 100_000

    def test_validator_rejects_d17_entry_with_unknown_reason(self):
        """Meta-test: the validator's closure check catches a
        D17-shaped entry with a fabricated reason."""
        with pytest.raises(AssertionError, match="Unknown audit-log entry"):
            _validate_entry(
                {
                    "action": "reject",
                    "reason": "gamma_overrun",  # not a registered D17 reason
                    "ticker": "TEST",
                    "token": "0" * 64,
                    "current_ev_dollars": 25.0,
                    "nav": 100_000,
                    "nav_source": "static_fallback",
                }
            )

    def test_validator_rejects_d17_entry_missing_nav_source(self):
        """nav_source is required on every D17 reject shape; a
        forgotten field surfaces here."""
        with pytest.raises(AssertionError, match="missing required keys"):
            _validate_entry(
                {
                    "action": "reject",
                    "reason": "nav_exhausted",
                    "ticker": "TEST",
                    "token": "0" * 64,
                    "current_ev_dollars": 25.0,
                    "nav": 100_000,
                    # nav_source omitted
                    "min_nav_for_trading": 1_000_000.0,
                }
            )
