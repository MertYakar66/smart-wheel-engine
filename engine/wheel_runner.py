"""
Wheel Strategy Runner — Main Orchestrator

Ties together all engine modules with real Bloomberg data to provide
a functional, end-to-end wheel strategy analysis pipeline.

Usage:
    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()

    # Screen universe for wheel candidates
    candidates = runner.screen_candidates()

    # Analyze a specific ticker
    analysis = runner.analyze_ticker("AAPL")

    # Score strangle entry timing
    timing = runner.score_strangle_entry("AAPL")

    # Full portfolio analysis
    report = runner.portfolio_report(["AAPL", "MSFT", "JPM"])
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TickerAnalysis:
    """Complete analysis of a ticker for wheel suitability."""

    ticker: str
    spot_price: float = 0.0

    # Fundamentals
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    beta: float = 0.0
    dividend_yield: float = 0.0
    sector: str = ""
    credit_rating: str = ""

    # Volatility
    iv_30d: float = 0.0
    rv_30d: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    vol_risk_premium: float = 0.0

    # Events
    days_to_earnings: int | None = None
    days_to_ex_div: int | None = None
    next_earnings_date: date | None = None
    next_div_date: date | None = None
    next_div_amount: float = 0.0

    # Strangle timing
    strangle_score: float = 0.0
    strangle_phase: str = ""
    strangle_recommendation: str = ""

    # Risk
    risk_free_rate: float = 0.0
    vix_level: float = 0.0

    # Wheel suitability
    wheel_score: float = 0.0  # 0-100 composite
    wheel_recommendation: str = ""

    def summary(self) -> str:
        lines = [
            f"=== {self.ticker} Wheel Analysis ===",
            f"Price: ${self.spot_price:.2f} | Sector: {self.sector}",
            f"Mkt Cap: ${self.market_cap / 1e9:.1f}B | P/E: {self.pe_ratio:.1f} | Beta: {self.beta:.2f}",
            "",
            "Volatility:",
            f"  IV(30d): {self.iv_30d:.1f}% | RV(30d): {self.rv_30d:.1f}%",
            f"  IV Rank: {self.iv_rank:.0f} | IV Pctl: {self.iv_percentile:.0f}",
            f"  Vol Premium: {self.vol_risk_premium:+.1f}%",
            "",
            "Events:",
            f"  Next Earnings: {self.next_earnings_date} ({self.days_to_earnings}d)"
            if self.days_to_earnings
            else "  Next Earnings: N/A",
            f"  Next Ex-Div: {self.next_div_date} (${self.next_div_amount:.3f})"
            if self.next_div_date
            else "  Next Ex-Div: N/A",
            "",
            f"Strangle Timing: {self.strangle_score:.0f}/100 ({self.strangle_phase}) → {self.strangle_recommendation}",
            f"Wheel Score: {self.wheel_score:.0f}/100 → {self.wheel_recommendation}",
            f"Risk-Free Rate: {self.risk_free_rate:.2%} | VIX: {self.vix_level:.1f}",
        ]
        return "\n".join(lines)


class WheelRunner:
    """
    Main orchestrator for the Smart Wheel Engine.

    Connects Bloomberg data to all engine modules and provides
    high-level analysis methods for wheel strategy decisions.
    """

    def __init__(self, data_dir: str | Path = "data/bloomberg"):
        self.data_dir = Path(data_dir)
        self._connector = None
        self._calendar = None
        self._strangle_engine = None
        # AUDIT-VIII P2.1: per-ticker HMM-regime cache so we do not
        # re-fit the 4-state Gaussian HMM on every /api/candidates hit.
        # Keyed by ``(ticker, tail_hash)`` where ``tail_hash`` is a
        # cheap fingerprint of the last 504 log-returns — this
        # invalidates automatically when new bars arrive or when the
        # PIT cutoff changes (different history → different hash).
        self._hmm_regime_cache: dict[tuple[str, int], float] = {}

    @property
    def connector(self):
        """Lazy-load the data connector — provider selected by SWE_DATA_PROVIDER."""
        if self._connector is None:
            import os
            provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()
            if provider == "theta":
                from engine.theta_connector import ThetaConnector
                self._connector = ThetaConnector(str(self.data_dir))
            else:
                from engine.data_connector import MarketDataConnector
                self._connector = MarketDataConnector(str(self.data_dir))
        return self._connector

    @property
    def strangle_engine(self):
        """Lazy-load strangle timing engine with IV overlay."""
        if self._strangle_engine is None:
            try:
                from engine.strangle_timing import StrangleTimingWithIV

                self._strangle_engine = StrangleTimingWithIV(data_connector=self.connector)
            except (ImportError, Exception):
                from engine.strangle_timing import StrangleTimingEngine

                self._strangle_engine = StrangleTimingEngine()
        return self._strangle_engine

    def get_calendar(
        self,
        tickers: list[str] | None = None,
        years: list[int] | None = None,
    ):
        """Get event calendar populated with Bloomberg data."""
        if self._calendar is None:
            from engine.data_integration import build_calendar_from_bloomberg

            if years is None:
                years = [date.today().year - 1, date.today().year]
            self._calendar = build_calendar_from_bloomberg(
                tickers=tickers, years=years, data_dir=str(self.data_dir)
            )
        return self._calendar

    def analyze_ticker(self, ticker: str, as_of: str | None = None) -> TickerAnalysis:
        """
        Complete wheel suitability analysis for a single ticker.

        Combines fundamentals, volatility, events, and strangle timing.
        """
        analysis = TickerAnalysis(ticker=ticker)
        conn = self.connector

        # --- Fundamentals ---
        fundamentals = conn.get_fundamentals(ticker)
        has_data = fundamentals is not None
        if fundamentals:
            analysis.market_cap = fundamentals.get("market_cap", 0) or 0
            analysis.pe_ratio = fundamentals.get("pe_ratio", 0) or 0
            analysis.beta = fundamentals.get("beta", 0) or 0
            analysis.dividend_yield = fundamentals.get("dividend_yield", 0) or 0
            analysis.sector = fundamentals.get("sector", "")
            analysis.iv_30d = fundamentals.get("implied_vol_atm", 0) or 0
            analysis.rv_30d = fundamentals.get("volatility_30d", 0) or 0

        # Credit risk
        credit = conn.get_credit_risk(ticker)
        if credit:
            analysis.credit_rating = credit.get("rtg_sp_lt_lc_issuer_credit", "")

        # --- Spot price ---
        ohlcv = conn.get_ohlcv(ticker)
        if not ohlcv.empty:
            analysis.spot_price = float(ohlcv["close"].iloc[-1])

        # --- IV rank & percentile ---
        try:
            analysis.iv_rank = conn.get_iv_rank(ticker, as_of)
            analysis.iv_percentile = conn.get_iv_percentile(ticker, as_of)
            analysis.vol_risk_premium = conn.get_vol_risk_premium(ticker, as_of)
        except Exception:
            pass

        # --- Events ---
        try:
            next_earn = conn.get_next_earnings(ticker, as_of)
            if next_earn:
                earn_ts = next_earn.get("announcement_date")
                if earn_ts is not None:
                    analysis.next_earnings_date = (
                        earn_ts.date() if hasattr(earn_ts, "date") else earn_ts
                    )
                if analysis.next_earnings_date:
                    today = date.fromisoformat(as_of) if as_of else date.today()
                    analysis.days_to_earnings = (analysis.next_earnings_date - today).days

            next_div = conn.get_next_dividend(ticker, as_of)
            if next_div:
                div_ts = next_div.get("ex_date")
                if div_ts is not None:
                    analysis.next_div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
                analysis.next_div_amount = next_div.get("dividend_amount", 0) or 0
                if analysis.next_div_date:
                    today = date.fromisoformat(as_of) if as_of else date.today()
                    analysis.days_to_ex_div = (analysis.next_div_date - today).days
        except Exception:
            pass

        # --- Risk-free rate ---
        try:
            from engine.data_integration import get_current_risk_free_rate

            analysis.risk_free_rate = get_current_risk_free_rate(as_of, data_dir=str(self.data_dir))
        except Exception:
            analysis.risk_free_rate = 0.05

        # --- VIX ---
        try:
            vix_data = conn.get_vix_regime(as_of)
            if vix_data:
                analysis.vix_level = vix_data.get("vix", 0)
        except Exception:
            pass

        # --- Strangle timing ---
        try:
            score = None
            # Try IV-enhanced scoring first, fall back to basic OHLCV scoring
            if hasattr(self.strangle_engine, "score_entry_with_iv"):
                try:
                    score = self.strangle_engine.score_entry_with_iv(ticker, as_of)
                except Exception:
                    pass  # Fall through to basic scoring

            if score is None and not ohlcv.empty and len(ohlcv) >= 100:
                from engine.strangle_timing import StrangleTimingEngine

                basic_engine = StrangleTimingEngine()
                score = basic_engine.score_entry(ohlcv)

            if score:
                analysis.strangle_score = score.total_score
                analysis.strangle_phase = score.regime.phase.value if score.regime else "unknown"
                analysis.strangle_recommendation = score.recommendation
        except Exception:
            pass

        # --- Composite wheel score ---
        # Return 0 for unknown tickers with no data
        if not has_data and analysis.spot_price == 0:
            analysis.wheel_score = 0.0
            analysis.wheel_recommendation = "no_data"
            return analysis

        analysis.wheel_score = self._compute_wheel_score(analysis)
        analysis.wheel_recommendation = (
            "strong_candidate"
            if analysis.wheel_score >= 75
            else "moderate"
            if analysis.wheel_score >= 55
            else "weak"
            if analysis.wheel_score >= 35
            else "avoid"
        )

        return analysis

    def _compute_wheel_score(self, a: TickerAnalysis) -> float:
        """
        Compute composite wheel suitability score (0-100).

        Weights:
        - IV environment (30%): high IV rank + positive vol premium
        - Fundamentals (20%): reasonable P/E, good credit, stable business
        - Event safety (15%): not too close to earnings
        - Strangle timing (20%): Layer 1+2 entry score
        - Liquidity/size (15%): sufficient market cap and volume
        """
        # IV score (0-100)
        iv_score = min(100, a.iv_rank * 100) if a.iv_rank > 0 else 30
        if a.vol_risk_premium > 5:
            iv_score = min(100, iv_score + 15)
        elif a.vol_risk_premium < -5:
            iv_score = max(0, iv_score - 20)

        # Fundamental score (0-100)
        fund_score = 50.0
        if 5 < a.pe_ratio < 30:
            fund_score += 15
        elif a.pe_ratio > 50 or a.pe_ratio < 0:
            fund_score -= 15
        if 0.3 < a.beta < 1.5:
            fund_score += 10
        elif a.beta > 2.0:
            fund_score -= 20
        if a.dividend_yield > 1:
            fund_score += 10
        if a.credit_rating and a.credit_rating[0] in ("A", "B"):
            fund_score += 10
        fund_score = max(0, min(100, fund_score))

        # Event safety (0-100)
        event_score = 80.0
        if a.days_to_earnings is not None:
            if a.days_to_earnings < 5:
                event_score = 10.0  # Too close
            elif a.days_to_earnings < 14:
                event_score = 40.0
            elif a.days_to_earnings < 30:
                event_score = 70.0
        # Ex-div proximity is less dangerous
        if a.days_to_ex_div is not None and a.days_to_ex_div < 3:
            event_score = max(event_score - 15, 0)

        # Strangle timing score (already 0-100)
        timing_score = a.strangle_score

        # Liquidity score (0-100)
        liquidity_score = 50.0
        if a.market_cap > 100e9:
            liquidity_score = 90.0
        elif a.market_cap > 20e9:
            liquidity_score = 75.0
        elif a.market_cap > 5e9:
            liquidity_score = 60.0
        elif a.market_cap > 0:
            liquidity_score = 35.0

        # Weighted composite
        total = (
            iv_score * 0.30
            + fund_score * 0.20
            + event_score * 0.15
            + timing_score * 0.20
            + liquidity_score * 0.15
        )

        return total

    def screen_candidates(
        self,
        min_wheel_score: float = 50.0,
        min_market_cap: float = 5e9,
        max_beta: float = 2.0,
        min_iv_rank: float = 0.3,
        sectors: list[str] | None = None,
        exclude_near_earnings_days: int = 7,
        top_n: int = 20,
        as_of: str | None = None,
    ) -> pd.DataFrame:
        """
        Screen the full S&P 500 universe for wheel candidates.

        Returns top candidates sorted by wheel score.
        """
        conn = self.connector

        # Start with fundamental screen.
        # AUDIT-VIII P1.3: pass ``as_of`` so the IV-rank sub-filter is
        # PIT-safe in backtests.
        try:
            universe = conn.screen_universe(
                min_market_cap=min_market_cap,
                max_beta=max_beta,
                sectors=sectors,
                min_iv_rank=min_iv_rank,
                as_of=as_of,
            )
        except Exception:
            universe = pd.DataFrame({"ticker": conn.get_universe()})

        if universe.empty:
            return pd.DataFrame()

        tickers = universe["ticker"].tolist() if "ticker" in universe.columns else []

        # Analyze each candidate
        results = []
        for ticker in tickers[:100]:  # Cap at 100 for performance
            try:
                analysis = self.analyze_ticker(ticker, as_of)

                # Apply filters
                if analysis.days_to_earnings is not None:
                    if 0 < analysis.days_to_earnings < exclude_near_earnings_days:
                        continue

                if analysis.wheel_score >= min_wheel_score:
                    results.append(
                        {
                            "ticker": ticker,
                            "wheel_score": analysis.wheel_score,
                            "recommendation": analysis.wheel_recommendation,
                            "spot": analysis.spot_price,
                            "iv_30d": analysis.iv_30d,
                            "rv_30d": analysis.rv_30d,
                            "iv_rank": analysis.iv_rank,
                            "vol_premium": analysis.vol_risk_premium,
                            "pe_ratio": analysis.pe_ratio,
                            "beta": analysis.beta,
                            "div_yield": analysis.dividend_yield,
                            "sector": analysis.sector,
                            "credit_rating": analysis.credit_rating,
                            "days_to_earnings": analysis.days_to_earnings,
                            "strangle_score": analysis.strangle_score,
                            "strangle_phase": analysis.strangle_phase,
                            "mkt_cap_B": analysis.market_cap / 1e9,
                        }
                    )
            except Exception:
                continue

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("wheel_score", ascending=False).head(top_n)
        return df

    # ------------------------------------------------------------------
    # EV-based ranking (audit upgrade)
    # ------------------------------------------------------------------
    def rank_candidates_by_ev(
        self,
        tickers: list[str] | None = None,
        dte_target: int = 35,
        delta_target: float = 0.25,
        contracts: int = 1,
        top_n: int = 20,
        min_ev_dollars: float = 0.0,
        as_of: str | None = None,
        include_diagnostic_fields: bool = True,
        use_event_gate: bool = True,
        earnings_buffer_days: int = 5,
        macro_buffer_days: int = 1,
        use_dealer_positioning: bool = False,
        dealer_assumption: str = "long_calls_short_puts",
        min_history_days: int = 504,
        enforce_history_gate: bool = True,
        enforce_chain_quality_gate: bool = True,
    ) -> pd.DataFrame:
        """Rank tickers by **probabilistic expected value** for a short-put wheel entry.

        This is the audit-grade replacement for ``screen_candidates``. The
        old ranker used a heuristic composite score; this one uses
        :class:`engine.ev_engine.EVEngine` with a PIT-safe empirical
        forward-return distribution pulled from the ticker's OHLCV history.

        For each ticker:
          1. Pull OHLCV up to ``as_of`` from the connector.
          2. Pull ATM IV (``volatility_30d`` fallback to Bloomberg ATM IV).
          3. Solve BSM delta to find the strike corresponding to
             ``delta_target`` (e.g. 0.25 = 25-delta put).
          4. Compute a fair BSM premium as a synthetic quote (flagged as
             synthetic in the output so traders know to check the real
             chain).
          5. Build a :class:`ShortOptionTrade`, sample an empirical
             forward distribution via
             :func:`engine.forward_distribution.best_available_forward_distribution`
             for ``dte_target`` days, and evaluate.
          6. Drop candidates with ``days_to_earnings < 5``.
          7. Return the top N sorted by ``ev_per_day``, with full EV
             diagnostics attached.

        Args:
            tickers: Explicit ticker list. When ``None`` the full universe
                is used (capped at 100 for performance parity with the
                legacy screen).
            dte_target: Target DTE for the synthetic trade.
            delta_target: Target put delta (positive; the sign is handled
                internally).
            contracts: Number of contracts per candidate.
            top_n: Number of top candidates to return.
            min_ev_dollars: Hard filter — drop any trade with ``ev_dollars``
                below this threshold.
            as_of: PIT cutoff date string (YYYY-MM-DD). ``None`` means now.
            include_diagnostic_fields: Include CVaR, Omega, fair value, etc.

        Returns:
            DataFrame sorted by ``ev_per_day`` descending, or empty.
        """
        from datetime import timedelta

        from scipy.optimize import brentq
        from scipy.stats import norm

        from engine.dealer_positioning import (
            DealerAssumption,
            DealerPositioningAnalyzer,
        )
        from engine.event_gate import EventGate, ScheduledEvent
        from engine.ev_engine import EVEngine, ShortOptionTrade
        from engine.forward_distribution import best_available_forward_distribution
        from engine.option_pricer import black_scholes_price

        conn = self.connector
        # Build a per-run event gate from the connector's earnings +
        # (optional) macro calendar. When use_event_gate=False the EV
        # engine falls back to the soft days_to_earnings skip below.
        event_gate: EventGate | None = None
        if use_event_gate:
            event_gate = EventGate(
                earnings_buffer_days=earnings_buffer_days,
                macro_buffer_days=macro_buffer_days,
            )
        ev_eng = EVEngine(event_gate=event_gate)

        # Optional dealer positioning analyzer. Off by default; when
        # enabled we pull the option chain per ticker and feed a
        # MarketStructure into EVEngine.evaluate alongside the other
        # regime multipliers. Chain fetch failures degrade gracefully
        # to market_structure=None (candidate still ranks).
        dealer_analyzer: DealerPositioningAnalyzer | None = None
        if use_dealer_positioning:
            try:
                assumption_enum = DealerAssumption(dealer_assumption)
            except ValueError:
                assumption_enum = DealerAssumption.LONG_CALLS_SHORT_PUTS
            dealer_analyzer = DealerPositioningAnalyzer(assumption=assumption_enum)

        if tickers is None:
            tickers = conn.get_universe()[:100]

        rows: list[dict] = []
        T = max(dte_target, 1) / 365.0

        for ticker in tickers:
            try:
                ohlcv = conn.get_ohlcv(ticker)
            except Exception:
                continue
            if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
                continue

            # Respect PIT cutoff on OHLCV.
            if as_of is not None:
                try:
                    cutoff = pd.Timestamp(as_of)
                    ohlcv = ohlcv.loc[ohlcv.index <= cutoff]
                except Exception:
                    pass
            if ohlcv.empty:
                continue

            # AUDIT-V P0.2: Historical data integrity gate.
            # Survivorship bias protection in the live path. We refuse
            # to rank a ticker whose OHLCV history is shorter than
            # ``min_history_days`` because the empirical forward-return
            # distribution it produces is statistically unreliable and
            # the ticker was likely backfilled into the universe (i.e.
            # survived long enough to be in today's SP500). Callers can
            # disable via enforce_history_gate=False for research paths.
            if enforce_history_gate and len(ohlcv) < min_history_days:
                continue

            spot = float(ohlcv["close"].iloc[-1])
            if spot <= 0:
                continue

            # Get ATM IV and fundamentals.
            # AUDIT-VIII P0.1: Bloomberg fundamentals CSV reports IV and
            # volatility in PERCENT (e.g. ``26.15`` means 26.15% annualized).
            # Earlier code treated the raw value as a decimal, then rejected
            # it as ``iv > 5`` (degenerate), which caused every candidate
            # to be dropped — the EV ranker silently returned zero rows.
            # We normalize to a decimal by dividing by 100 when the raw
            # value is clearly a percentage (>3) and guard NaN/None.
            fundamentals = conn.get_fundamentals(ticker) or {}
            iv_raw = fundamentals.get("implied_vol_atm")
            if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
                iv_raw = fundamentals.get("volatility_30d")
            try:
                iv = float(iv_raw) if iv_raw is not None else 0.0
            except (TypeError, ValueError):
                iv = 0.0
            if np.isnan(iv) or iv <= 0:
                continue
            # Normalise percent -> decimal. A sigma of 3.0 (= 300%) is an
            # extreme upper bound for any real equity; anything above is
            # virtually certainly a percent representation.
            if iv > 3.0:
                iv = iv / 100.0
            if iv <= 0 or iv > 5:
                continue  # still degenerate after normalisation

            dividend_yield_raw = fundamentals.get("dividend_yield", 0.0) or 0.0
            try:
                dividend_yield = float(dividend_yield_raw)
            except (TypeError, ValueError):
                dividend_yield = 0.0
            # Bloomberg reports dividend yield in percent too (e.g. ``0.5``
            # means 0.5% if raw is already small, but Bloomberg sends ``1.5``
            # for a 1.5% yield). Normalise.
            if dividend_yield > 1.0:
                dividend_yield = dividend_yield / 100.0
            if np.isnan(dividend_yield) or dividend_yield < 0:
                dividend_yield = 0.0

            # AUDIT-VIII P0.2: ``MarketDataConnector.get_risk_free_rate``
            # returns the raw treasury CSV value which is in PERCENT
            # (e.g. ``4.333`` means 4.333%). Passing it straight into BSM
            # as if it were a decimal breaks delta/premium math and
            # produced zero tradeable candidates. Normalise defensively.
            risk_free_rate = 0.05
            try:
                rf_raw = conn.get_risk_free_rate(as_of)
                if rf_raw is None or (isinstance(rf_raw, float) and np.isnan(rf_raw)):
                    risk_free_rate = 0.05
                else:
                    rf_val = float(rf_raw)
                    if rf_val > 1.0:
                        rf_val = rf_val / 100.0
                    # Sanity clamp — reject absurd values (we'd rather
                    # fall back to 5% than corrupt downstream math).
                    if 0.0 <= rf_val <= 0.25:
                        risk_free_rate = rf_val
            except Exception:
                pass

            # Event exclusion
            # Two layers:
            #   (a) Soft skip on days_to_earnings < earnings_buffer_days
            #       (kept for backwards compat with callers that opt out
            #       of the hard event gate).
            #   (b) Hard event lockout via EventGate, populated per-ticker
            #       below. When event_gate is active the EV engine will
            #       short-circuit the candidate and return an EVResult
            #       with event_lockout_reason set.
            today_date = date.fromisoformat(as_of) if as_of else date.today()
            trade_start_d = today_date
            trade_end_d = today_date + timedelta(days=dte_target)
            try:
                next_earn = conn.get_next_earnings(ticker, as_of)
                days_to_earn = None
                if next_earn:
                    earn_ts = next_earn.get("announcement_date")
                    if earn_ts is not None:
                        earn_d = earn_ts.date() if hasattr(earn_ts, "date") else earn_ts
                        days_to_earn = (earn_d - today_date).days
                        # Register on the per-run event gate so the EV
                        # engine can pre-emptively block.
                        if event_gate is not None and earn_d is not None:
                            event_gate.add_event(
                                ScheduledEvent(
                                    ticker=ticker,
                                    kind="earnings",
                                    event_date=earn_d,
                                )
                            )
                if (
                    event_gate is None
                    and days_to_earn is not None
                    and 0 <= days_to_earn < earnings_buffer_days
                ):
                    continue
            except Exception:
                days_to_earn = None

            # Solve for the strike that gives the target put delta
            # Put delta = e^{-qT} * (N(d1) - 1); target is -delta_target.
            def put_delta_err(K: float) -> float:
                if K <= 0:
                    return 1.0
                d1 = (np.log(spot / K) + (risk_free_rate - dividend_yield + 0.5 * iv**2) * T) / (
                    iv * np.sqrt(T)
                )
                put_delta = np.exp(-dividend_yield * T) * (norm.cdf(d1) - 1.0)
                return put_delta + delta_target  # target: -delta_target

            # Reasonable strike bracket: 50% OTM to 5% OTM
            try:
                strike = brentq(put_delta_err, spot * 0.5, spot * 0.99, xtol=1e-2)
            except (ValueError, RuntimeError):
                continue
            # Round to nearest $0.50 for realism
            strike = round(strike * 2) / 2
            if strike <= 0 or strike >= spot:
                continue

            # Synthetic fair-value premium (mid). Real chains will differ.
            premium = black_scholes_price(
                S=spot,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=iv,
                option_type="put",
                q=dividend_yield,
            )
            if premium <= 0.05:
                continue  # premium too thin to trade

            # Approximate a bid/ask from the synthetic mid (10% spread proxy).
            bid = premium * 0.95
            ask = premium * 1.05

            # Pull the PIT-safe forward distribution
            fwd_rets, method = best_available_forward_distribution(
                ohlcv,
                horizon_days=dte_target,
                as_of=as_of,
            )

            # HMM regime multiplier — compute from the ticker's own
            # OHLCV log-returns. The HMM fit is cheap (~50 ms per
            # ticker for 504 daily returns) but adds up across a
            # 100-ticker universe. AUDIT-VIII P2.1: cache the result
            # keyed by a fingerprint of the tail — same tail, same
            # regime multiplier. Hash is deliberately cheap (length +
            # first / last / mid log-return rounded to 1e-6) so
            # collisions are astronomical for real price histories
            # but the key is trivially computable. Failure of any
            # sub-step degrades cleanly to 1.0.
            hmm_regime_mult = 1.0
            try:
                from engine.regime_hmm import GaussianHMM

                log_rets = np.diff(np.log(ohlcv["close"].values))
                if len(log_rets) >= 200:
                    tail = log_rets[-504:]
                    fp = (
                        len(tail),
                        round(float(tail[0]), 6),
                        round(float(tail[len(tail) // 2]), 6),
                        round(float(tail[-1]), 6),
                    )
                    cache_key = (ticker, hash(fp))
                    cached = self._hmm_regime_cache.get(cache_key)
                    if cached is not None:
                        hmm_regime_mult = cached
                    else:
                        hmm = GaussianHMM(n_states=4, n_iter=20, random_state=42)
                        hmm.fit(tail)
                        probs = hmm.predict_proba(tail)
                        hmm_regime_mult = float(
                            hmm.position_multiplier(probs[-1])
                        )
                        self._hmm_regime_cache[cache_key] = hmm_regime_mult
            except Exception:
                hmm_regime_mult = 1.0

            trade = ShortOptionTrade(
                option_type="put",
                underlying=ticker,
                spot=spot,
                strike=strike,
                premium=premium,
                dte=dte_target,
                iv=iv,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                contracts=contracts,
                bid=bid,
                ask=ask,
                open_interest=1000,  # unknown; mid-liquid assumption
                regime_multiplier=hmm_regime_mult,
            )

            # Optional dealer positioning: pull the option chain for
            # the target expiry and build a MarketStructure to feed
            # into the EV evaluation. Every failure mode degrades to
            # market_structure=None (candidate still ranks on pure EV).
            #
            # AUDIT-V P0.3: when a chain IS fetched (for dealer
            # positioning or elsewhere), we run the
            # DataQualityFramework options-consistency check first and
            # HARD-SKIP the ticker if it reports any ERROR-severity
            # issue (bid>ask, IV out of range, expired options). The
            # check is cheap and the consequences of ignoring crossed
            # markets in a live path are severe.
            market_structure = None
            chain_quality_blocked_reason = ""
            if dealer_analyzer is not None:
                try:
                    chain_df = None
                    if hasattr(conn, "get_options"):
                        chain_df = conn.get_options(ticker)
                    elif hasattr(conn, "get_option_chain"):
                        chain_df = conn.get_option_chain(ticker)
                    if chain_df is not None and len(chain_df) > 0:
                        # Coerce column names + filter to the nearest
                        # expiry around the target DTE.
                        cdf = chain_df.copy()
                        cdf.columns = [c.lower() for c in cdf.columns]
                        if "expiration" in cdf.columns:
                            cdf["expiration"] = pd.to_datetime(
                                cdf["expiration"], errors="coerce"
                            )
                            target_expiry_ts = pd.Timestamp(
                                trade_start_d + timedelta(days=dte_target)
                            )
                            cdf["_dte_gap"] = (
                                cdf["expiration"] - target_expiry_ts
                            ).abs()
                            # Pick the single closest expiry
                            best_expiry = cdf.sort_values("_dte_gap")[
                                "expiration"
                            ].iloc[0]
                            cdf = cdf[cdf["expiration"] == best_expiry].copy()
                            cdf = cdf.drop(columns=["_dte_gap"])
                            expiry_date = (
                                best_expiry.date()
                                if hasattr(best_expiry, "date")
                                else best_expiry
                            )
                        else:
                            expiry_date = trade_end_d

                        if enforce_chain_quality_gate and len(cdf) > 0:
                            try:
                                from data.quality import (
                                    DataQualityFramework,
                                    Severity,
                                )

                                qf = DataQualityFramework()
                                q_cdf = cdf.copy()
                                if "date" not in q_cdf.columns:
                                    q_cdf["date"] = pd.Timestamp(trade_start_d)
                                issues = qf._check_options_consistency(q_cdf)
                                critical = [
                                    i for i in issues
                                    if i.severity in (Severity.ERROR, Severity.CRITICAL)
                                ]
                                if critical:
                                    chain_quality_blocked_reason = (
                                        f"chain_quality:{critical[0].message[:80]}"
                                    )
                            except Exception:
                                # Quality check failure must never crash
                                # the ranker itself — degrade to "unknown
                                # quality" and let the trade proceed.
                                pass

                        if chain_quality_blocked_reason:
                            # Hard skip this ticker entirely
                            continue

                        if len(cdf) > 0:
                            market_structure = dealer_analyzer.analyze(
                                chain=cdf,
                                spot=spot,
                                expiry=expiry_date,
                                ticker=ticker,
                                dividend_yield=dividend_yield,
                            )
                except Exception:
                    # Graceful degrade — dealer positioning is optional
                    market_structure = None

            res = ev_eng.evaluate(
                trade,
                forward_log_returns=fwd_rets,
                trade_start=trade_start_d,
                trade_end=trade_end_d,
                market_structure=market_structure,
            )
            # Event-gate short-circuit: drop blocked candidates entirely.
            if res.event_lockout_reason:
                continue
            if res.ev_dollars < min_ev_dollars:
                continue

            row: dict = {
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "premium": round(premium, 3),
                "dte": dte_target,
                "iv": round(iv, 4),
                "ev_dollars": round(res.ev_dollars, 2),
                "ev_per_day": round(res.ev_per_day, 3),
                "prob_profit": round(res.prob_profit, 4),
                "prob_assignment": round(res.prob_assignment, 4),
                "days_to_earnings": days_to_earn,
                "distribution_source": method,
            }
            if include_diagnostic_fields:
                row.update(
                    {
                        "cvar_5": round(res.cvar_5, 2),
                        "cvar_99_evt": (
                            round(res.cvar_99_evt, 2)
                            if not np.isnan(res.cvar_99_evt)
                            else None
                        ),
                        "tail_xi": (
                            round(res.tail_xi, 4)
                            if not np.isnan(res.tail_xi)
                            else None
                        ),
                        "heavy_tail": bool(res.heavy_tail),
                        "omega_ratio": round(res.omega_ratio, 3),
                        "fair_value": round(res.fair_value, 3),
                        "edge_vs_fair": round(res.edge_vs_fair, 2),
                        "breakeven_move_pct": round(res.breakeven_move_pct, 4),
                        "total_transaction_cost": round(res.total_transaction_cost, 2),
                        "skew_pnl": round(res.skew_pnl, 3),
                        # Dealer positioning diagnostics (all None when
                        # use_dealer_positioning=False or the chain was
                        # unavailable).
                        "dealer_regime": res.dealer_regime or None,
                        "dealer_multiplier": round(res.dealer_multiplier, 4),
                        "gex_total": (
                            round(res.gex_total, 0)
                            if not np.isnan(res.gex_total)
                            else None
                        ),
                        "gamma_flip_distance_pct": (
                            round(res.gamma_flip_distance_pct, 4)
                            if not np.isnan(res.gamma_flip_distance_pct)
                            else None
                        ),
                        "nearest_put_wall_strike": (
                            round(res.nearest_put_wall_strike, 2)
                            if not np.isnan(res.nearest_put_wall_strike)
                            else None
                        ),
                        "nearest_call_wall_strike": (
                            round(res.nearest_call_wall_strike, 2)
                            if not np.isnan(res.nearest_call_wall_strike)
                            else None
                        ),
                    }
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ev_per_day", ascending=False).head(top_n)
        return df

    # ------------------------------------------------------------------
    # Mode B: EV ranking + TradingView chart context dossier
    # ------------------------------------------------------------------
    def build_candidate_dossiers(
        self,
        tickers: list[str] | None = None,
        dte_target: int = 35,
        delta_target: float = 0.25,
        contracts: int = 1,
        top_n: int = 10,
        min_ev_dollars: float = 0.0,
        as_of: str | None = None,
        chart_provider=None,
        chart_timeframe: str = "1D",
        reviewer=None,
        use_event_gate: bool = True,
        earnings_buffer_days: int = 5,
        macro_buffer_days: int = 1,
    ) -> list:
        """Engine-first Mode B: rank by EV, then attach TradingView charts.

        This is the canonical workflow for the Claude-terminal-driven
        TradingView integration. The engine ranks candidates *first*
        using :meth:`rank_candidates_by_ev`, then for the top N we
        attach a chart context via a :class:`ChartContextProvider`
        (typically a filesystem provider reading screenshots dropped by
        the terminal's own browser tooling) and run a
        :class:`ChartReviewer` that can DOWNGRADE a trade based on
        visual context but cannot upgrade a negative-EV trade.

        Args:
            tickers: Optional explicit ticker list.
            dte_target / delta_target / contracts / min_ev_dollars:
                Forwarded to :meth:`rank_candidates_by_ev`.
            top_n: Only the top N ranked candidates get chart contexts
                attached — cheap optimisation since chart capture is
                expensive.
            as_of: PIT cutoff.
            chart_provider: A :class:`ChartContextProvider` instance.
                Defaults to the filesystem provider under
                ``screenshots/``.
            chart_timeframe: TradingView timeframe (``"1D"`` default).
            reviewer: Optional :class:`ChartReviewer`; defaults to
                :class:`EnginePhaseReviewer`.
            use_event_gate / earnings_buffer_days / macro_buffer_days:
                Forwarded to :meth:`rank_candidates_by_ev`.

        Returns:
            List of :class:`CandidateDossier` with full EV + chart +
            verdict, sorted by the underlying EV ranking.
        """
        from engine.candidate_dossier import EnginePhaseReviewer, build_dossiers
        from engine.tradingview_bridge import build_default_provider

        ev_df = self.rank_candidates_by_ev(
            tickers=tickers,
            dte_target=dte_target,
            delta_target=delta_target,
            contracts=contracts,
            top_n=max(top_n, 20),  # rank a wider pool, attach charts to top_n
            min_ev_dollars=min_ev_dollars,
            as_of=as_of,
            include_diagnostic_fields=True,
            use_event_gate=use_event_gate,
            earnings_buffer_days=earnings_buffer_days,
            macro_buffer_days=macro_buffer_days,
        )

        if ev_df is None or len(ev_df) == 0:
            return []

        provider = chart_provider or build_default_provider()
        chart_reviewer = reviewer or EnginePhaseReviewer()

        return build_dossiers(
            ev_frame=ev_df,
            provider=provider,
            reviewer=chart_reviewer,
            timeframe=chart_timeframe,  # type: ignore[arg-type]
            top_n=top_n,
        )

    def portfolio_report(
        self,
        tickers: list[str],
        as_of: str | None = None,
    ) -> dict:
        """
        Generate a portfolio-level analysis report.

        Args:
            tickers: List of tickers in the portfolio
            as_of: Analysis date

        Returns:
            Dict with portfolio-level metrics and per-ticker analysis
        """
        from engine.data_integration import get_current_risk_free_rate

        analyses = {}
        for ticker in tickers:
            try:
                analyses[ticker] = self.analyze_ticker(ticker, as_of)
            except Exception:
                continue

        if not analyses:
            return {"error": "No valid analyses"}

        # Aggregate metrics
        avg_iv_rank = np.mean([a.iv_rank for a in analyses.values()])
        avg_beta = np.mean([a.beta for a in analyses.values() if a.beta > 0])
        avg_wheel_score = np.mean([a.wheel_score for a in analyses.values()])
        total_mkt_cap = sum(a.market_cap for a in analyses.values())

        # Sector allocation
        sector_counts: dict[str, int] = {}
        for a in analyses.values():
            s = a.sector or "Unknown"
            sector_counts[s] = sector_counts.get(s, 0) + 1

        # Upcoming events
        upcoming_events = []
        for ticker, a in analyses.items():
            if a.days_to_earnings is not None and 0 < a.days_to_earnings <= 30:
                upcoming_events.append(
                    {
                        "ticker": ticker,
                        "event": "earnings",
                        "date": str(a.next_earnings_date),
                        "days": a.days_to_earnings,
                    }
                )
            if a.days_to_ex_div is not None and 0 < a.days_to_ex_div <= 30:
                upcoming_events.append(
                    {
                        "ticker": ticker,
                        "event": "ex_div",
                        "date": str(a.next_div_date),
                        "days": a.days_to_ex_div,
                        "amount": a.next_div_amount,
                    }
                )

        upcoming_events.sort(key=lambda x: x["days"])

        risk_free = get_current_risk_free_rate(as_of, data_dir=str(self.data_dir))

        return {
            "as_of": as_of or str(date.today()),
            "tickers": tickers,
            "n_positions": len(analyses),
            "avg_iv_rank": avg_iv_rank,
            "avg_beta": avg_beta,
            "avg_wheel_score": avg_wheel_score,
            "total_mkt_cap_B": total_mkt_cap / 1e9,
            "risk_free_rate": risk_free,
            "sector_allocation": sector_counts,
            "upcoming_events": upcoming_events,
            "per_ticker": {t: a.summary() for t, a in analyses.items()},
        }
