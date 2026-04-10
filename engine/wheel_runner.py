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

    @property
    def connector(self):
        """Lazy-load the data connector."""
        if self._connector is None:
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

        # Start with fundamental screen
        try:
            universe = conn.screen_universe(
                min_market_cap=min_market_cap,
                max_beta=max_beta,
                sectors=sectors,
                min_iv_rank=min_iv_rank,
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
