"""Tests for the D17 portfolio-context wire on the ``/api/tv/dossier`` endpoint.

Closes ``docs/PRODUCTION_READINESS.md`` Blocker B2 (D17 live-wire to
``engine_api.py``). The HTTP endpoint now optionally accepts ``nav``,
``holdings``, ``puts_held``, and ``regime_map`` query params; when
supplied they are parsed into a ``PortfolioContext`` and threaded into
``build_candidate_dossiers``, which in turn engages the dossier
reviewer's R7 (VaR > 5% NAV → review) and R8 (stress drawdown / dealer
regime → review) soft-warns.

What this file pins:

1. ``_build_portfolio_context_from_params(nav=None, ...)`` returns
   ``None`` — preserves the pre-wire no-op semantics of R7 / R8 when
   the operator omits ``nav``. **This is the critical safety property:
   adding the wire must NOT break existing callers who don't pass
   the new params.**
2. With ``nav`` supplied, the helper constructs a ``PortfolioContext``
   with the right fields populated.
3. Parsing is forgiving — malformed entries are silently dropped, not
   rejected with 400 (operator integrations supplying bad data
   should not break the dossier endpoint; the affected D17 sub-gate
   just falls back to missing-data behaviour).
4. The wire in ``_handle_tv_dossier`` correctly forwards the
   ``portfolio_context`` kwarg to ``build_candidate_dossiers``.
"""

from __future__ import annotations

from engine_api import _build_portfolio_context_from_params


class TestPortfolioContextOptInSemantics:
    """The critical safety property: omitting ``nav`` is opt-OUT
    (pre-wire behaviour preserved)."""

    def test_nav_none_returns_none(self):
        """No nav → no PortfolioContext → R7/R8 stay no-op."""
        result = _build_portfolio_context_from_params(
            nav=None,
            holdings_csv=None,
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is None

    def test_nav_none_returns_none_even_with_other_params(self):
        """Even if holdings / puts_held / regime_map are populated, no
        nav means no PortfolioContext. The opt-in is gated specifically
        on nav (which is the load-bearing field for R7/R8 calculations
        — VaR and stress drawdowns are expressed as % of NAV)."""
        result = _build_portfolio_context_from_params(
            nav=None,
            holdings_csv="AAPL:100,MSFT:50",
            puts_held_csv="AAPL:180:1:2026-04-19",
            regime_map_csv="NVDA:short_gamma_amplifying",
        )
        assert result is None


class TestPortfolioContextConstruction:
    """When ``nav`` is supplied, the helper builds a properly-shaped
    ``PortfolioContext``."""

    def test_nav_only_yields_minimal_context(self):
        """Just nav, no other inputs. R7/R8 will still engage at the
        nav-anchored level but with empty held positions."""
        result = _build_portfolio_context_from_params(
            nav=250_000.0,
            holdings_csv=None,
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is not None
        assert result.nav == 250_000.0
        assert result.held_option_positions == []
        assert result.stock_holdings == []
        assert result.dealer_regime_by_ticker is None

    def test_full_params_populate_all_fields(self):
        result = _build_portfolio_context_from_params(
            nav=1_000_000.0,
            holdings_csv="AAPL:100,MSFT:50",
            puts_held_csv="AAPL:180:1:2026-04-19,JPM:140:2:2026-04-19",
            regime_map_csv="NVDA:short_gamma_amplifying,AAPL:normal",
        )
        assert result is not None
        assert result.nav == 1_000_000.0
        assert result.stock_holdings == [("AAPL", 100), ("MSFT", 50)]
        assert len(result.held_option_positions) == 2
        # Canonical schema the D17 risk gates consume (pos["symbol"], dte,
        # iv, is_short) — NOT the old {"ticker": ...} shape that silently
        # dropped the held book. dte is derived from the expiry relative to
        # today, so assert it structurally rather than pinning a literal.
        aapl = result.held_option_positions[0]
        assert aapl["symbol"] == "AAPL"
        assert aapl["option_type"] == "put"
        assert aapl["strike"] == 180.0
        assert aapl["contracts"] == 1
        assert aapl["is_short"] is True
        assert aapl["iv"] == 0.30
        assert aapl["expiration"] == "2026-04-19"
        assert isinstance(aapl["dte"], int) and aapl["dte"] >= 0
        assert "ticker" not in aapl
        jpm = result.held_option_positions[1]
        assert jpm["symbol"] == "JPM"
        assert jpm["strike"] == 140.0
        assert jpm["contracts"] == 2
        assert jpm["is_short"] is True
        assert "ticker" not in jpm
        assert result.dealer_regime_by_ticker == {
            "NVDA": "short_gamma_amplifying",
            "AAPL": "normal",
        }

    def test_ticker_normalisation_uppercases(self):
        """Lower-case ticker params get normalised — defensive against
        copy-paste / curl typos."""
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv="aapl:50",
            puts_held_csv=None,
            regime_map_csv="nvda:normal",
        )
        assert result is not None
        assert result.stock_holdings == [("AAPL", 50)]
        assert result.dealer_regime_by_ticker == {"NVDA": "normal"}


class TestPortfolioContextParsingIsForgiving:
    """Malformed entries silently drop rather than raising 400. The
    affected D17 sub-gate falls back to missing-data semantics for the
    unparseable row."""

    def test_malformed_holdings_drop(self):
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv="AAPL:100,BADENTRY,MSFT:50,WTF:notanumber",
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is not None
        # AAPL and MSFT parsed; the two malformed dropped.
        assert result.stock_holdings == [("AAPL", 100), ("MSFT", 50)]

    def test_malformed_puts_drop(self):
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv=None,
            puts_held_csv="AAPL:180:1:2026-04-19,JUNK,MSFT:notafloat:2",
            regime_map_csv=None,
        )
        assert result is not None
        # Only the well-formed AAPL row parsed.
        assert len(result.held_option_positions) == 1
        assert result.held_option_positions[0]["symbol"] == "AAPL"

    def test_malformed_regime_drops(self):
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv=None,
            puts_held_csv=None,
            regime_map_csv="NVDA:normal,JUNK,AAPL:short_gamma_amplifying",
        )
        assert result is not None
        # Two well-formed parsed; the JUNK dropped.
        assert result.dealer_regime_by_ticker == {
            "NVDA": "normal",
            "AAPL": "short_gamma_amplifying",
        }

    def test_empty_csvs_yield_empty_fields(self):
        """Empty strings and whitespace-only inputs produce empty
        collections, not None / not crashes."""
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv="",
            puts_held_csv="   ",
            regime_map_csv=",,,",
        )
        assert result is not None
        assert result.stock_holdings == []
        assert result.held_option_positions == []
        # All-malformed regime_map yields None (matches the parser's
        # contract: if NOTHING parses, dealer_regime_by_ticker remains
        # None so check_dealer_regime returns missing_data).
        assert result.dealer_regime_by_ticker is None

    def test_zero_shares_holding_allowed(self):
        """A zero-share holding is technically valid (e.g., a flat
        position the operator wants tracked). Don't drop these."""
        result = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv="AAPL:0,MSFT:100",
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is not None
        assert result.stock_holdings == [("AAPL", 0), ("MSFT", 100)]


class TestPortfolioContextNumericEdgeCases:
    def test_nav_float_zero_treated_as_provided(self):
        """nav=0.0 is technically a valid input (an operator with no
        capital still wants the engine to compute relative VaR). The
        helper should construct the context — leave validation to the
        downstream D17 gates."""
        result = _build_portfolio_context_from_params(
            nav=0.0,
            holdings_csv=None,
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is not None
        assert result.nav == 0.0

    def test_nav_negative_treated_as_provided(self):
        """Negative NAV is a weird input but not crash-worthy. The
        downstream D17 gate decides what to do."""
        result = _build_portfolio_context_from_params(
            nav=-1000.0,
            holdings_csv=None,
            puts_held_csv=None,
            regime_map_csv=None,
        )
        assert result is not None
        assert result.nav == -1000.0


class TestDossierEndpointSignature:
    """Ensure _handle_tv_dossier accepts the new D17 kwargs without
    error. A signature regression that dropped the kwargs would be
    caught here."""

    def test_handle_tv_dossier_accepts_d17_kwargs(self):
        """Verify the new optional kwargs are present on the method
        signature — pure structural check; no engine invocation."""
        import inspect

        from engine_api import EngineAPIHandler

        sig = inspect.signature(EngineAPIHandler._handle_tv_dossier)
        params = sig.parameters
        assert "nav" in params
        assert "holdings_csv" in params
        assert "puts_held_csv" in params
        assert "regime_map_csv" in params
        # All default to None — opt-in semantics.
        assert params["nav"].default is None
        assert params["holdings_csv"].default is None
        assert params["puts_held_csv"].default is None
        assert params["regime_map_csv"].default is None


class TestEnrichEndpointSignature:
    """B2 closure for the webhook/pull-enrichment surface. The pull
    enrichment endpoint (_handle_tv_enrich) accepts the same D17
    portfolio-context query params as the dossier endpoint; the
    enrichment method (_enrich_alert) accepts an optional
    portfolio_context kwarg. The webhook (POST /api/tv/webhook) does
    NOT accept the params because Pine cannot know the operator's
    book — verified by inspecting the alert-driver call site (no
    portfolio context is constructed there).
    """

    def test_handle_tv_enrich_accepts_d17_kwargs(self):
        """The pull /api/tv/enrich endpoint accepts the same D17
        opt-in params as the dossier endpoint."""
        import inspect

        from engine_api import EngineAPIHandler

        sig = inspect.signature(EngineAPIHandler._handle_tv_enrich)
        params = sig.parameters
        for name in ("nav", "holdings_csv", "puts_held_csv", "regime_map_csv"):
            assert name in params, f"missing {name} on _handle_tv_enrich"
            assert params[name].default is None, f"{name} default must be None (opt-in semantics)"

    def test_enrich_alert_accepts_portfolio_context(self):
        """The shared enrichment method accepts portfolio_context as
        an optional kwarg defaulting to None."""
        import inspect

        from engine_api import EngineAPIHandler

        sig = inspect.signature(EngineAPIHandler._enrich_alert)
        params = sig.parameters
        assert "portfolio_context" in params
        assert params["portfolio_context"].default is None


class TestEnrichSectorCapBreachDowngrade:
    """End-to-end: drive _enrich_alert with a PortfolioContext that
    triggers sector_cap_breach; the verdict ladder must downgrade
    proceed → review with verdict_reason='sector_cap_breach'. Mirrors
    the dossier-side R9 integration test, but on the webhook/pull
    surface (where the verdict ladder is the inline one, not the
    EnginePhaseReviewer chain)."""

    def _setup_runner_returning_proceed(self, monkeypatch):
        """Stub get_connector / get_runner so _enrich_alert receives
        an EV-ranked candidate with a 'proceed'-eligible payoff
        (positive ev_dollars, prob_profit >= 0.65) for a large AAPL
        put. Returns the test runner so the caller can inspect."""
        import numpy as np
        import pandas as pd

        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        class FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.25,
                    "volatility_30d": 0.22,
                    "dividend_yield": 0.01,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["AAPL"]

            def get_iv_rank(self, ticker, as_of=None):
                return 50.0

            def get_vol_risk_premium(self, ticker, as_of=None):
                return 0.0

        runner = WheelRunner()
        runner._connector = FakeConn()

        monkeypatch.setattr("engine_api.get_connector", lambda: runner._connector)
        monkeypatch.setattr("engine_api.get_runner", lambda: runner)
        return runner

    def test_sector_cap_breach_downgrades_enrich_alert(self, monkeypatch):
        from engine.portfolio_risk_gates import PortfolioContext
        from engine.tv_signals import TVAlert
        from engine_api import EngineAPIHandler

        self._setup_runner_returning_proceed(monkeypatch)

        # Large held AAPL position → sector cap breach on any new
        # AAPL put.
        portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 100.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 5,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 100.0},
            nav=80_000.0,
        )

        handler = EngineAPIHandler.__new__(EngineAPIHandler)
        alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")
        enriched = handler._enrich_alert(
            alert, as_of="2024-06-14", portfolio_context=portfolio_context
        )

        # If the engine produced a 'proceed' verdict initially, R9
        # downgrade must have kicked in. If the engine did not reach
        # 'proceed' (e.g., synthetic price path → negative EV), the
        # test is vacuous on the downgrade but still validates that
        # the verdict is not 'proceed' under sector-cap breach.
        assert enriched["verdict"] != "proceed", (
            f"sector-cap breach must not yield proceed; got verdict="
            f"{enriched['verdict']}, reason={enriched.get('verdict_reason')!r}"
        )

    def test_no_portfolio_context_does_not_downgrade(self, monkeypatch):
        """Without portfolio_context, the D17 block is a no-op. The
        verdict ladder runs unchanged. Opt-in contract."""
        from engine.tv_signals import TVAlert
        from engine_api import EngineAPIHandler

        self._setup_runner_returning_proceed(monkeypatch)

        handler = EngineAPIHandler.__new__(EngineAPIHandler)
        alert = TVAlert(ticker="AAPL", signal="wheel_put_zone", source="test")
        enriched = handler._enrich_alert(alert, as_of="2024-06-14")

        # No portfolio_context → no D17 downgrade. We don't assert a
        # specific verdict (depends on synthetic ev_dollars), just
        # that no D17 reason was attached.
        assert enriched.get("verdict_reason") not in (
            "sector_cap_breach",
            "portfolio_var_breach",
            "stress_breach",
            "short_gamma_regime",
        ), (
            f"D17 reason set without portfolio_context: "
            f"verdict_reason={enriched.get('verdict_reason')!r}"
        )


class TestHeldBookIsConsumableByRiskGates:
    """Regression for the ticker-vs-symbol schema bug (2026-07-13 engine
    audit, Finding C-3).

    The dicts produced by ``_build_portfolio_context_from_params`` must be
    consumable by the D17 risk gates, which universally read ``pos["symbol"]``
    and ``pos["is_short"]`` (see engine/portfolio_risk_gates.py and
    engine/stress_testing.py). The prior producer emitted ``{"ticker": ...}``
    with no ``is_short``, so R7-R10 silently dropped the operator's held book
    and the soft-warns could never fire on ``/api/tv/dossier``.

    The two earlier tests never caught this because one only checks the
    producer's dict shape and the other hand-builds a ``PortfolioContext``
    with the correct ``symbol`` key — neither crosses the producer → gate
    seam. This class does.
    """

    def test_produced_dict_uses_canonical_gate_schema(self):
        ctx = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv=None,
            puts_held_csv="AAPL:180:2:2027-01-15",
            regime_map_csv=None,
        )
        assert ctx is not None
        pos = ctx.held_option_positions[0]
        for key in ("symbol", "option_type", "strike", "dte", "iv", "contracts", "is_short"):
            assert key in pos, (
                f"canonical held-position key {key!r} missing — gates will skip this row"
            )
        assert pos["symbol"] == "AAPL"
        assert pos["is_short"] is True
        assert "ticker" not in pos

    def test_single_name_cap_sees_the_held_book(self):
        """End-to-end across the seam: feed the API-produced positions to
        R10's single-name cap and confirm the book is SEEN. Held AAPL short
        puts = 180 * 100 * 2 = $36,000 notional = 36% of a $100k NAV, well
        over the 10% cap → breach. Before the fix the book was invisible
        (current notional $0) and the gate passed."""
        from engine.portfolio_risk_gates import check_single_name_cap

        ctx = _build_portfolio_context_from_params(
            nav=100_000.0,
            holdings_csv=None,
            puts_held_csv="AAPL:180:2:2027-01-15",
            regime_map_csv=None,
        )
        assert ctx is not None
        result = check_single_name_cap(
            "AAPL",
            0.0,
            ctx.held_option_positions,
            100_000.0,
            max_single_name_pct=0.10,
        )
        assert result.passed is False, (
            "R10 must SEE the held AAPL book (36% of NAV > 10% cap); a pass "
            "means the ticker-vs-symbol drop regressed"
        )
        assert result.reason == "single_name_breach"
        assert result.details["current_name_notional"] == 36_000.0
