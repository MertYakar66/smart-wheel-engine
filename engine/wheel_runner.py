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
    timing = runner.strangle_engine.score_entry_with_iv("AAPL")

    # Full portfolio analysis
    report = runner.portfolio_report(["AAPL", "MSFT", "JPM"])
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


# Above this many DP cells (items × capacity) the exact knapsack is
# skipped for a greedy ROC fill — keeps ``select_book`` responsive for
# very large accounts. Realistic wheel accounts ($25k–$2M) stay far
# below it, so the exact path is what runs in practice.
_KNAPSACK_MAX_CELLS = 60_000_000


def _solve_book_knapsack(weights: list[int], values: list[float], capacity: int) -> list[int]:
    """0/1 knapsack — maximise total value under an integer capacity.

    Returns the indices (into ``weights`` / ``values``) of the selected
    items. Exact dynamic program; the caller bounds ``capacity`` via the
    collateral unit so the table stays tractable. Items that cannot fit
    (``weight <= 0`` or ``weight > capacity``) are skipped cleanly.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return []
    dp = [0.0] * (capacity + 1)
    keep = [bytearray(capacity + 1) for _ in range(n)]
    for i in range(n):
        wi = weights[i]
        vi = values[i]
        if wi <= 0 or wi > capacity:
            continue
        ki = keep[i]
        # Descending w so dp[w - wi] still holds the pre-item value
        # (the standard 1-D 0/1-knapsack ordering).
        for w in range(capacity, wi - 1, -1):
            cand = dp[w - wi] + vi
            if cand > dp[w]:
                dp[w] = cand
                ki[w] = 1
    selected: list[int] = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if w >= 0 and keep[i][w]:
            selected.append(i)
            w -= weights[i]
    selected.reverse()
    return selected


# ----------------------------------------------------------------------
# Column schema for WheelRunner.rank_covered_calls_by_ev
# ----------------------------------------------------------------------
# Pinned at module scope (mirrors wheel_tracker._ROLL_COLUMNS) so the
# empty-result path returns a same-shaped zero-row DataFrame and tests can
# assert the schema without running the ranker. The diagnostic block is
# appended only when ``include_diagnostic_fields=True``.
_CC_RANK_CORE_COLUMNS = [
    "ticker",
    "spot",
    "strike",
    "premium",
    "dte",
    "new_expiry",
    "target_delta",
    "iv",
    "contracts",
    "ev_dollars",
    "ev_per_day",
    "prob_profit",
    "prob_assignment",
    "days_to_earnings",
    "days_to_ex_div",
    "distribution_source",
]
_CC_RANK_DIAGNOSTIC_COLUMNS = [
    "cvar_5",
    "cvar_99_evt",
    "tail_xi",
    "heavy_tail",
    "omega_ratio",
    "fair_value",
    "edge_vs_fair",
    "breakeven_move_pct",
    "prob_touch",
    "total_transaction_cost",
    "skew_pnl",
    "expected_dividend",
    "regime_multiplier",
]


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
        # PIT cutoff changes (different history → different hash). The
        # cached value is ``(regime_multiplier, regime_label)``.
        self._hmm_regime_cache: dict[tuple[str, int], tuple[float, str]] = {}

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
        use_dealer_positioning: bool = True,
        use_skew_dynamics: bool = True,
        use_news_sentiment: bool = True,
        use_credit_regime: bool = True,
        dealer_assumption: str = "long_calls_short_puts",
        min_history_days: int = 504,
        enforce_history_gate: bool = True,
        enforce_chain_quality_gate: bool = True,
        universe_limit: int | None = None,
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
                is ranked; pass ``universe_limit`` to cap the scan.
            dte_target: Target DTE for the synthetic trade.
            delta_target: Target put delta (positive; the sign is handled
                internally).
            contracts: Number of contracts per candidate.
            top_n: Number of top candidates to return.
            min_ev_dollars: Hard filter — drop any trade with ``ev_dollars``
                below this threshold.
            as_of: PIT cutoff date string (YYYY-MM-DD). ``None`` means now.
            include_diagnostic_fields: Include CVaR, Omega, fair value, etc.
            universe_limit: When ``tickers`` is ``None``, cap the scanned
                universe to the first N names. ``None`` (default) ranks
                the entire universe.

        Returns:
            DataFrame sorted by ``ev_per_day`` descending, or empty.
            Always carries the capital-efficiency columns ``collateral``
            (``strike × 100 × contracts``) and ``roc``
            (``ev_dollars / collateral``); :meth:`select_book` consumes
            both to fit a book under an account-size budget.

            The returned frame's ``.attrs["drops"]`` carries a
            diagnostic list of dicts -- one per candidate gated out
            before it could become a row -- each
            ``{"ticker", "gate", "reason"}``. ``gate`` is one of
            ``data``, ``history``, ``event``, ``strike``, ``premium``,
            ``chain_quality`` or ``ev_threshold``. Pure observability:
            survivor rows are unaffected and no extra
            ``EVEngine.evaluate`` call is made to populate it.
        """
        from datetime import timedelta

        from scipy.optimize import brentq
        from scipy.stats import norm

        from engine.dealer_positioning import (
            DealerAssumption,
            DealerPositioningAnalyzer,
        )
        from engine.ev_engine import EVEngine, ShortOptionTrade
        from engine.event_gate import EventGate, ScheduledEvent
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

        # News sentiment reader — shared across tickers, cached for 5m.
        news_reader = None
        if use_news_sentiment:
            try:
                from engine.news_sentiment import NewsSentimentReader

                news_reader = NewsSentimentReader()
            except Exception:
                news_reader = None

        # Credit-regime multiplier (HY OAS stressed/crisis → soft de-rank).
        # Fetched once per run, applied uniformly to every candidate.
        credit_mult = 1.0
        credit_regime = "unknown"
        if use_credit_regime:
            try:
                from engine.external_data.fred_adapter import FREDAdapter

                fa = FREDAdapter()
                cr = fa.credit_regime(as_of=as_of)
                credit_regime = cr.get("regime", "unknown")
                if credit_regime == "crisis":
                    credit_mult = 0.80
                elif credit_regime == "stressed":
                    credit_mult = 0.92
            except Exception:
                credit_mult = 1.0

        if tickers is None:
            tickers = conn.get_universe()
            if universe_limit is not None and universe_limit > 0:
                tickers = tickers[:universe_limit]

        rows: list[dict] = []
        # Diagnostic drop log: one dict per candidate gated out before
        # it could become a row, exposed on the returned frame's
        # ``.attrs["drops"]``. Pure observability -- it captures what was
        # already being discarded; see CLAUDE.md section 2.
        drops: list[dict] = []
        T = max(dte_target, 1) / 365.0

        for ticker in tickers:
            try:
                ohlcv = conn.get_ohlcv(ticker)
            except Exception:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "OHLCV fetch raised",
                    }
                )
                continue
            if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "no OHLCV data (empty or missing 'close')",
                    }
                )
                continue

            # Respect PIT cutoff on OHLCV.
            if as_of is not None:
                try:
                    cutoff = pd.Timestamp(as_of)
                    ohlcv = ohlcv.loc[ohlcv.index <= cutoff]
                except Exception:
                    pass
            if ohlcv.empty:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "no OHLCV history at or before as_of",
                    }
                )
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
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "history",
                        "reason": f"history {len(ohlcv)}d < required {min_history_days}d",
                    }
                )
                continue

            spot = float(ohlcv["close"].iloc[-1])
            if spot <= 0:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "non-positive spot price",
                    }
                )
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
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "IV missing or non-positive",
                    }
                )
                continue
            # Normalise percent -> decimal. A sigma of 3.0 (= 300%) is an
            # extreme upper bound for any real equity; anything above is
            # virtually certainly a percent representation.
            if iv > 3.0:
                iv = iv / 100.0
            if iv <= 0 or iv > 5:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "data",
                        "reason": "IV degenerate after percent normalisation",
                    }
                )
                continue  # still degenerate after normalisation

            dividend_yield_raw = fundamentals.get("dividend_yield", 0.0) or 0.0
            try:
                dividend_yield = float(dividend_yield_raw)
            except (TypeError, ValueError):
                dividend_yield = 0.0
            # AUDIT-IX: ``MarketDataConnector.get_fundamentals`` returns
            # ``dividend_yield`` straight from the Bloomberg CSV column
            # ``eqy_dvd_yld_12m``, which is stored in PERCENT (e.g. ``2.04``
            # means 2.04%; the column's median is ~2.0 and max ~9.8). The
            # earlier ``if dividend_yield > 1.0`` guard divided only values
            # above 1.0, so every sub-1%-yield name — 92 of the 410 priced
            # names, most mega-cap tech — skipped normalisation and reached
            # BSM as a whole-number decimal (``0.87`` -> an 87% dividend
            # yield), corrupting the delta->strike solve and the synthetic
            # premium. The column is uniformly percent, so divide
            # unconditionally; absurd results fall back to "no dividend".
            if not np.isfinite(dividend_yield) or dividend_yield < 0.0:
                dividend_yield = 0.0
            else:
                dividend_yield /= 100.0
                if dividend_yield > 0.30:  # >30% is a data error, not a yield
                    dividend_yield = 0.0

            # ``MarketDataConnector.get_risk_free_rate`` normalises the
            # treasury value internally and returns a DECIMAL rate (e.g.
            # ``0.0433``), so the ``rf_val > 1.0`` check below is redundant
            # defence-in-depth; the ``0.0 <= rf_val <= 0.25`` clamp is the
            # effective guard and falls back to 5% on anything absurd.
            # AUDIT-IX: the connector's own normaliser still uses a ``> 1``
            # heuristic that would mis-handle a genuine sub-1% treasury
            # rate — latent only (rates ~3-5% today) and caught by the
            # clamp here; tracked as a follow-up in the AUDIT-IX PR.
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
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "event",
                            "reason": (
                                f"earnings in {days_to_earn}d < buffer {earnings_buffer_days}d"
                            ),
                        }
                    )
                    continue
            except Exception:
                days_to_earn = None

            # Solve for the strike that gives the target put delta
            # Put delta = e^{-qT} * (N(d1) - 1); target is -delta_target.
            def put_delta_err(
                K: float,
                spot=spot,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                iv=iv,
                T=T,
                delta_target=delta_target,
            ) -> float:
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
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "strike",
                        "reason": "delta-to-strike solve did not converge",
                    }
                )
                continue
            # Round to nearest $0.50 for realism
            strike = round(strike * 2) / 2
            if strike <= 0 or strike >= spot:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "strike",
                        "reason": "solved strike out of range (<=0 or >=spot)",
                    }
                )
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
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "premium",
                        "reason": "synthetic premium too thin (<=$0.05)",
                    }
                )
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
            # Companion regime label for hmm_regime_mult. "unknown" when
            # the HMM does not run (short history) or fails -- never a
            # fabricated regime; mirrors credit_regime's "unknown".
            hmm_regime = "unknown"
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
                        hmm_regime_mult, hmm_regime = cached
                    else:
                        hmm = GaussianHMM(n_states=4, n_iter=20, random_state=42)
                        hmm.fit(tail)
                        probs = hmm.predict_proba(tail)
                        hmm_regime_mult = float(hmm.position_multiplier(probs[-1]))
                        # Label is the argmax state -- a pure read of the
                        # same posterior, in its own try so it can never
                        # perturb the already-computed multiplier.
                        try:
                            hmm_regime = hmm.fit_result.state_labels[int(np.argmax(probs[-1]))]
                        except Exception:
                            hmm_regime = "unknown"
                        self._hmm_regime_cache[cache_key] = (hmm_regime_mult, hmm_regime)
            except Exception:
                hmm_regime_mult = 1.0
                hmm_regime = "unknown"

            # Fetch the chain once and use it for (a) open interest at our
            # strike, (b) 25Δ put / ATM / 25Δ call for skew signals, and
            # (c) dealer-positioning MarketStructure. A single fetch avoids
            # hammering the Terminal and keeps the snapshot internally
            # consistent across signals.
            chain_df = None
            try:
                if hasattr(conn, "get_options"):
                    chain_df = conn.get_options(ticker)
                elif hasattr(conn, "get_option_chain"):
                    chain_df = conn.get_option_chain(ticker)
            except Exception:
                chain_df = None

            # Raw-chain integrity gate (runs regardless of dealer positioning).
            # Crossed markets (bid > ask), negative volume, invalid IV, or
            # expired contracts in the snapshot are data-source bugs, not
            # single-row noise — block the whole ticker when any CRITICAL or
            # ERROR issue is present on the raw chain, before any per-row
            # pre-clean can silently suppress the signal.
            if enforce_chain_quality_gate and chain_df is not None and len(chain_df) > 0:
                try:
                    from data.quality import DataQualityFramework, Severity

                    raw_cdf = chain_df.copy()
                    raw_cdf.columns = [c.lower() for c in raw_cdf.columns]
                    if "date" not in raw_cdf.columns:
                        raw_cdf["date"] = pd.Timestamp(trade_start_d)
                    raw_issues = DataQualityFramework()._check_options_consistency(raw_cdf)
                    critical_raw = [
                        i for i in raw_issues if i.severity in (Severity.ERROR, Severity.CRITICAL)
                    ]
                    if critical_raw:
                        logger.warning(
                            "%s: chain quality gate blocked ticker — %s",
                            ticker,
                            critical_raw[0].message[:100],
                        )
                        drops.append(
                            {
                                "ticker": ticker,
                                "gate": "chain_quality",
                                "reason": f"chain quality: {critical_raw[0].message[:100]}",
                            }
                        )
                        continue
                except Exception:
                    # Quality framework missing / import failure → fall back
                    # to the downstream pre-clean path; do not block trades
                    # on infrastructure bugs.
                    pass

            # Look up OI at our target strike from the chain when possible
            strike_oi = 1000  # mid-liquid fallback
            if chain_df is not None and len(chain_df) > 0:
                try:
                    cdf_lc = chain_df.copy()
                    cdf_lc.columns = [c.lower() for c in cdf_lc.columns]
                    if {"strike", "right", "open_interest"}.issubset(cdf_lc.columns):
                        puts_only = cdf_lc[cdf_lc["right"].astype(str).str.lower() == "put"]
                        if not puts_only.empty:
                            puts_only = puts_only.copy()
                            puts_only["_gap"] = (puts_only["strike"] - strike).abs()
                            row_oi = puts_only.sort_values("_gap").iloc[0]["open_interest"]
                            if pd.notna(row_oi) and float(row_oi) > 0:
                                strike_oi = int(float(row_oi))
                except Exception:
                    pass

            # Skew multiplier: steepening put skew is a risk-off signal.
            # skew_slope(iv_25d_put, iv_atm, iv_25d_call) returns
            # (iv_25d_put - iv_25d_call) / iv_atm. Larger = steeper put skew.
            # Map slope -> multiplier in [0.85, 1.08]. Positive slope
            # (normal risk-off) cuts multiplier; negative slope (call-skew
            # risk-on, rare in equities) boosts it slightly.
            skew_mult = 1.0
            skew_diag: dict = {}
            if use_skew_dynamics and chain_df is not None and len(chain_df) > 0:
                try:
                    from engine.skew_dynamics import skew_slope

                    cdf_lc = chain_df.copy()
                    cdf_lc.columns = [c.lower() for c in cdf_lc.columns]
                    if {"delta", "iv", "right"}.issubset(cdf_lc.columns):
                        cdf_lc = cdf_lc.dropna(subset=["delta", "iv"])
                        puts_s = cdf_lc[cdf_lc["right"].astype(str).str.lower() == "put"].copy()
                        calls_s = cdf_lc[cdf_lc["right"].astype(str).str.lower() == "call"].copy()
                        if not puts_s.empty and not calls_s.empty:
                            puts_s["_gp"] = (puts_s["delta"] - (-0.25)).abs()
                            puts_a = puts_s.copy()
                            puts_a["_ga"] = (puts_a["delta"] - (-0.50)).abs()
                            calls_s["_gc"] = (calls_s["delta"] - 0.25).abs()
                            iv_25p = float(puts_s.sort_values("_gp").iloc[0]["iv"])
                            iv_atm_chain = float(puts_a.sort_values("_ga").iloc[0]["iv"])
                            iv_25c = float(calls_s.sort_values("_gc").iloc[0]["iv"])
                            if all(0 < v <= 3.0 for v in (iv_25p, iv_atm_chain, iv_25c)):
                                skew_diag = skew_slope(iv_25p, iv_atm_chain, iv_25c)
                                slope = skew_diag["skew_slope"]
                                # Anchor: slope=0 -> 1.0, slope=+0.20 -> 0.90,
                                # slope=-0.10 -> 1.05. Clamp to [0.85, 1.08].
                                skew_mult = float(np.clip(1.0 - 0.5 * slope, 0.85, 1.08))
                except Exception:
                    skew_mult = 1.0

            # News sentiment multiplier (per-ticker).
            news_mult = 1.0
            news_sentiment = 0.0
            news_n_articles = 0
            if news_reader is not None:
                try:
                    news_mult = float(news_reader.sentiment_multiplier(ticker, as_of=as_of))
                    ns = news_reader.get_ticker_sentiment(ticker, as_of=as_of)
                    news_sentiment = float(ns.get("sentiment", 0.0))
                    news_n_articles = int(ns.get("n_articles", 0))
                except Exception:
                    news_mult = 1.0

            combined_regime_mult = float(hmm_regime_mult * skew_mult * news_mult * credit_mult)

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
                open_interest=strike_oi,
                regime_multiplier=combined_regime_mult,
            )

            # Dealer positioning uses the already-fetched chain_df.
            # Every failure mode degrades to market_structure=None
            # (candidate still ranks on pure EV).
            market_structure = None
            chain_quality_blocked_reason = ""
            if dealer_analyzer is not None:
                try:
                    if chain_df is not None and len(chain_df) > 0:
                        # Coerce column names + filter to the nearest
                        # expiry around the target DTE.
                        cdf = chain_df.copy()
                        cdf.columns = [c.lower() for c in cdf.columns]
                        if "expiration" in cdf.columns:
                            cdf["expiration"] = pd.to_datetime(cdf["expiration"], errors="coerce")
                            target_expiry_ts = pd.Timestamp(
                                trade_start_d + timedelta(days=dte_target)
                            )
                            cdf["_dte_gap"] = (cdf["expiration"] - target_expiry_ts).abs()
                            # Pick the single closest expiry
                            best_expiry = cdf.sort_values("_dte_gap")["expiration"].iloc[0]
                            cdf = cdf[cdf["expiration"] == best_expiry].copy()
                            cdf = cdf.drop(columns=["_dte_gap"])
                            expiry_date = (
                                best_expiry.date() if hasattr(best_expiry, "date") else best_expiry
                            )
                        else:
                            expiry_date = trade_end_d

                        # Pre-clean: after-hours snapshots have stale NBBO
                        # quotes with crossed markets (bid > ask) and stale
                        # contracts with bid = ask = 0. These are ROW-LEVEL
                        # issues, not ticker-level ones. Drop the bad rows
                        # so the quality gate has a clean surface to check
                        # against — otherwise we hard-skip the whole ticker
                        # when market is closed.
                        if {"bid", "ask"}.issubset(cdf.columns):
                            bid_n = pd.to_numeric(cdf["bid"], errors="coerce")
                            ask_n = pd.to_numeric(cdf["ask"], errors="coerce")
                            valid = bid_n.notna() & ask_n.notna()
                            keep = ~valid | ((bid_n <= ask_n) & ~((bid_n == 0) & (ask_n == 0)))
                            cdf = cdf[keep].copy()
                        if {"iv"}.issubset(cdf.columns):
                            iv_n = pd.to_numeric(cdf["iv"], errors="coerce")
                            cdf = cdf[iv_n.isna() | ((iv_n >= 0) & (iv_n <= 5.0))].copy()

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
                                    i
                                    for i in issues
                                    if i.severity in (Severity.ERROR, Severity.CRITICAL)
                                ]
                                if critical:
                                    chain_quality_blocked_reason = (
                                        f"chain_quality:{critical[0].message[:80]}"
                                    )
                            except Exception:
                                pass

                        # When the chain has quality issues we DROP the
                        # dealer-positioning overlay for this ticker but
                        # still let the EV ranker rank it on synthetic
                        # premium + forward distribution. Dealer
                        # positioning is a multiplier, not a gate — a
                        # noisy chain should not invalidate an otherwise
                        # good candidate. The blocked reason is exposed
                        # in the output row so callers can audit.
                        if chain_quality_blocked_reason:
                            logger.debug(
                                "%s: %s — skipping dealer overlay, ranking continues",
                                ticker,
                                chain_quality_blocked_reason,
                            )
                            market_structure = None
                        elif len(cdf) > 0:
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
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "event",
                        "reason": str(res.event_lockout_reason),
                    }
                )
                continue
            if res.ev_dollars < min_ev_dollars:
                drops.append(
                    {
                        "ticker": ticker,
                        "gate": "ev_threshold",
                        "reason": (
                            f"ev_dollars {res.ev_dollars:.2f} < min_ev_dollars {min_ev_dollars:.2f}"
                        ),
                    }
                )
                continue

            # Capital-efficiency fields. A cash-secured put reserves
            # ``strike × 100 × contracts`` of collateral; ROC is the
            # forward EV per dollar of that collateral. These are core
            # (not diagnostic) — a capital-constrained trader needs them
            # to rank, and ``select_book`` consumes them. Computed purely
            # from ``res.ev_dollars`` (post-``EVEngine.evaluate``), so
            # they re-present the EV authority's output, never rescue it.
            collateral = strike * 100.0 * contracts
            roc = (res.ev_dollars / collateral) if collateral > 0 else 0.0

            row: dict = {
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "premium": round(premium, 3),
                "dte": dte_target,
                "iv": round(iv, 4),
                "ev_dollars": round(res.ev_dollars, 2),
                "ev_per_day": round(res.ev_per_day, 3),
                "collateral": round(collateral, 2),
                "roc": round(roc, 6),
                "prob_profit": round(res.prob_profit, 4),
                "prob_assignment": round(res.prob_assignment, 4),
                "days_to_earnings": days_to_earn,
                "distribution_source": method,
            }
            if include_diagnostic_fields:
                row.update(
                    {
                        # EV before the regime overlays: res.mean_pnl is
                        # the mean scenario P&L the engine computes as
                        # ``ev_raw`` (ev_engine.py), pre regime multiplier.
                        # ev_dollars is the post-multiplier value.
                        "ev_raw": round(res.mean_pnl, 2),
                        "cvar_5": round(res.cvar_5, 2),
                        "cvar_99_evt": (
                            round(res.cvar_99_evt, 2) if not np.isnan(res.cvar_99_evt) else None
                        ),
                        "tail_xi": (round(res.tail_xi, 4) if not np.isnan(res.tail_xi) else None),
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
                            round(res.gex_total, 0) if not np.isnan(res.gex_total) else None
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
                        # Skew-dynamics diagnostics (populated when
                        # use_skew_dynamics=True and chain has 25Δ points)
                        "skew_slope": round(skew_diag["skew_slope"], 4) if skew_diag else None,
                        "put_skew": round(skew_diag["put_skew"], 4) if skew_diag else None,
                        "risk_reversal": round(skew_diag["risk_reversal"], 4)
                        if skew_diag
                        else None,
                        "skew_multiplier": round(skew_mult, 4),
                        "hmm_multiplier": round(hmm_regime_mult, 4),
                        "hmm_regime": hmm_regime,
                        "news_multiplier": round(news_mult, 4),
                        "news_sentiment": round(news_sentiment, 4),
                        "news_n_articles": news_n_articles,
                        "credit_multiplier": round(credit_mult, 4),
                        "credit_regime": credit_regime,
                        "strike_open_interest": strike_oi,
                        "chain_quality_warning": chain_quality_blocked_reason or None,
                    }
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ev_per_day", ascending=False).head(top_n)
        # Diagnostic drop log -- attached after the sort/head so it rides
        # on the exact frame returned (empty or not). Survivor rows are
        # untouched; see CLAUDE.md section 2.
        df.attrs["drops"] = drops
        return df

    # ------------------------------------------------------------------
    # Account-aware book selection (S4 follow-up)
    # ------------------------------------------------------------------
    def select_book(
        self,
        account_size: float,
        tickers: list[str] | None = None,
        *,
        ranking: pd.DataFrame | None = None,
        max_weight_per_name: float | None = None,
        min_roc: float = 0.0,
        collateral_unit: float = 50.0,
        **rank_kwargs,
    ) -> pd.DataFrame:
        """Fit a cash-secured-put book under an account-size budget.

        S4 logged that :meth:`rank_candidates_by_ev` is capital-blind:
        it returns the same ranking for a $50k account and a $5M one,
        front-loads the most expensive names, and offers no helper to
        answer "which names fit under budget X". This is that helper.

        It is a **pure post-processor** of the ranker output and is
        §2-safe: it never calls :class:`~engine.ev_engine.EVEngine`
        itself. Every candidate it considers has already been through
        ``EVEngine.evaluate`` inside :meth:`rank_candidates_by_ev`; this
        method only *subsets* that output to maximise total forward EV
        under the collateral constraint. It cannot rescue a
        negative-EV candidate — those are filtered out of the pool
        before selection — and it cannot change any candidate's EV.

        The selection is a 0/1 knapsack: each ticker is either in the
        book (one entry, ``contracts`` as ranked) or out, each reserving
        ``collateral`` dollars, maximising ``Σ ev_dollars`` subject to
        ``Σ collateral ≤ account_size``. Solved exactly by dynamic
        program; for very large accounts it degrades to a greedy ROC
        fill (see ``_KNAPSACK_MAX_CELLS``).

        Args:
            account_size: Hard collateral budget in dollars.
            tickers: Forwarded to :meth:`rank_candidates_by_ev` when
                ``ranking`` is not supplied.
            ranking: A precomputed :meth:`rank_candidates_by_ev` frame.
                When given, no ranking is run (and no ``EVEngine``
                call is made) — the frame is used as-is. Must carry the
                ``collateral`` and ``ev_dollars`` columns.
            max_weight_per_name: Optional concentration cap as a
                fraction of ``account_size`` (e.g. ``0.25`` → no single
                name may reserve more than 25% of the account). Names
                exceeding it are dropped from the pool before selection.
            min_roc: Drop candidates whose ``roc`` is below this.
            collateral_unit: Granularity (dollars) the knapsack capacity
                is discretised to. Defaults to ``50`` — the natural
                granularity of a $0.50-rounded strike × 100. Must be
                positive (it is used as a divisor).
            **rank_kwargs: Forwarded to :meth:`rank_candidates_by_ev`
                when ``ranking`` is not supplied. ``top_n`` defaults to
                effectively unlimited here so the budget is fit against
                the whole candidate set, not the ranker's display slice.

        Returns:
            The selected book as a DataFrame (subset of the ranking
            rows, sorted by ``ev_per_day`` descending). ``.attrs``
            carries ``account_size``, ``total_collateral``,
            ``total_ev_dollars``, ``cash_remaining``, ``n_positions``,
            ``capital_utilization`` and ``selection_method``. Empty
            when nothing fits.
        """
        import math

        if collateral_unit <= 0:
            raise ValueError(
                f"select_book: collateral_unit must be positive, got "
                f"{collateral_unit!r}. It discretises the knapsack capacity "
                f"and is used as a divisor."
            )

        empty_attrs = {
            "account_size": float(account_size),
            "total_collateral": 0.0,
            "total_ev_dollars": 0.0,
            "cash_remaining": float(max(account_size, 0.0)),
            "n_positions": 0,
            "capital_utilization": 0.0,
            "selection_method": "none",
        }

        def _empty() -> pd.DataFrame:
            out = pd.DataFrame()
            out.attrs.update(empty_attrs)
            return out

        if account_size <= 0:
            return _empty()

        if ranking is None:
            # The book is fit against the *whole* feasible candidate set,
            # not a display slice. rank_candidates_by_ev defaults top_n to
            # 10, which would silently truncate the pool to the 10 highest
            # ev_per_day names — and the budget-optimal book for a small
            # account routinely includes cheaper names ranked below that.
            # Default top_n wide open here; an explicit caller value still
            # wins.
            rank_kwargs.setdefault("top_n", 10**9)
            ranking = self.rank_candidates_by_ev(tickers=tickers, **rank_kwargs)

        if ranking is None or len(ranking) == 0:
            return _empty()

        missing = {"collateral", "ev_dollars"} - set(ranking.columns)
        if missing:
            raise ValueError(
                f"select_book: ranking frame is missing required column(s) "
                f"{sorted(missing)}. Pass a frame from rank_candidates_by_ev "
                f"(which always emits 'collateral' and 'ev_dollars')."
            )

        # Candidate pool: only positive-EV names can enter a book — a
        # negative-EV trade never improves Σ EV, and including it would
        # be the §2 violation this helper must not commit. Also enforce
        # the budget, the ROC floor and the optional concentration cap.
        pool = ranking[
            (ranking["ev_dollars"] > 0)
            & (ranking["collateral"] > 0)
            & (ranking["collateral"] <= account_size)
        ].copy()
        if "roc" in pool.columns and min_roc > 0:
            pool = pool[pool["roc"] >= min_roc]
        if max_weight_per_name is not None:
            name_cap = account_size * max_weight_per_name
            pool = pool[pool["collateral"] <= name_cap]

        if len(pool) == 0:
            return _empty()

        pool = pool.reset_index(drop=True)
        collateral = pool["collateral"].astype(float).tolist()
        ev = pool["ev_dollars"].astype(float).tolist()

        capacity = int(account_size // collateral_unit)
        weights = [max(1, math.ceil(c / collateral_unit)) for c in collateral]

        if capacity * len(pool) > _KNAPSACK_MAX_CELLS:
            # Greedy ROC fill — large-account degradation path.
            order = sorted(
                range(len(pool)),
                key=lambda i: ev[i] / collateral[i],
                reverse=True,
            )
            chosen: list[int] = []
            spent = 0.0
            for i in order:
                if spent + collateral[i] <= account_size:
                    chosen.append(i)
                    spent += collateral[i]
            method = "greedy_roc"
        else:
            chosen = _solve_book_knapsack(weights, ev, capacity)
            method = "exact_knapsack"

        if not chosen:
            return _empty()

        book = pool.iloc[chosen].copy()
        if "ev_per_day" in book.columns:
            book = book.sort_values("ev_per_day", ascending=False)
        book = book.reset_index(drop=True)

        total_collateral = float(book["collateral"].sum())
        total_ev = float(book["ev_dollars"].sum())
        book.attrs.update(
            {
                "account_size": float(account_size),
                "total_collateral": round(total_collateral, 2),
                "total_ev_dollars": round(total_ev, 2),
                "cash_remaining": round(account_size - total_collateral, 2),
                "n_positions": len(book),
                "capital_utilization": round(total_collateral / account_size, 4),
                "selection_method": method,
            }
        )
        return book

    # ------------------------------------------------------------------
    # Covered-call entry ranking (issue #118 P1 — S8 follow-up)
    # ------------------------------------------------------------------
    def rank_covered_calls_by_ev(
        self,
        ticker: str,
        shares_held: int = 100,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        as_of: str | None = None,
        min_ev_dollars: float = 0.0,
        top_n: int = 20,
        include_diagnostic_fields: bool = True,
        use_event_gate: bool = True,
        earnings_buffer_days: int = 5,
        macro_buffer_days: int = 1,
        min_history_days: int = 504,
        enforce_history_gate: bool = True,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> pd.DataFrame:
        """Rank covered-call **entry** candidates for a held stock by forward EV.

        S8 logged that :meth:`engine.wheel_tracker.WheelTracker.open_covered_call`
        takes a raw ``strike`` / ``premium`` with **no EV evaluation** — the
        covered-call entry sits outside the EV authority (CLAUDE.md §2). This
        is the entry parallel of
        :meth:`~engine.wheel_tracker.WheelTracker.suggest_call_rolls` (the
        roll): given a held stock position, it enumerates a
        ``(DTE × delta)`` grid of candidate covered calls and ranks them by
        the forward EV of the **short-call leg**, every candidate scored
        through :meth:`engine.ev_engine.EVEngine.evaluate`.

        It mirrors :meth:`rank_candidates_by_ev` (the put-entry ranker) for
        the data plumbing — PIT-safe OHLCV, percent→decimal IV/dividend
        normalisation, the history gate, the event lockout, the
        ``.attrs["drops"]`` diagnostic — and :meth:`suggest_call_rolls` for
        the call-leg EV pattern (``ShortOptionTrade(option_type="call")``,
        :func:`~engine.wheel_tracker._solve_call_strike`, the empirical
        forward distribution).

        Scope — this ranks the **option leg only**: the forward EV of
        *being short the call*. The stock leg's P&L (basis vs an
        assigned/called-away price) is separate position accounting handled
        by :class:`~engine.wheel_tracker.WheelTracker`; it does not belong in
        an option-EV ranking and is deliberately not blended in here.

        For each ``(DTE, delta)`` pair:
          1. Solve the BSM call strike at ``delta`` (OTM, above spot).
          2. Round to the nearest $0.50 and price a synthetic BSM mid
             premium (real chains will differ — check the live chain).
          3. Build a :class:`ShortOptionTrade` with ``option_type="call"``,
             sized to ``contracts = shares_held // 100``.
          4. Pull a PIT-safe empirical forward distribution for ``DTE`` and
             call :meth:`EVEngine.evaluate`.
          5. Drop event-gate-blocked candidates and any with
             ``ev_dollars < min_ev_dollars``.
          6. Return the survivors, sorted by ``ev_per_day`` descending.

        Args:
            ticker: The held stock.
            shares_held: Shares of ``ticker`` currently owned. The covered
                call is sized to the largest whole-contract count the
                holding supports (``shares_held // 100``); a value below
                100 raises :class:`ValueError` — you cannot write a covered
                call without 100 shares to cover it.
            target_dtes: Candidate days-to-expiry for the new call.
            target_deltas: Candidate call deltas (positive; OTM). Each
                ``(DTE, delta)`` pair is one candidate.
            as_of: PIT cutoff date ``YYYY-MM-DD``. ``None`` means now.
            min_ev_dollars: Hard EV floor. Candidates with
                ``ev_dollars`` below this are dropped — the ranker **ranks,
                never rescues**: with the default ``0.0`` a negative-EV
                covered call never surfaces as tradeable.
            top_n: Number of top candidates to return.
            include_diagnostic_fields: Append CVaR, Omega, fair value,
                tail and other diagnostics (see ``_CC_RANK_DIAGNOSTIC_COLUMNS``).
            use_event_gate: Hard-block candidates whose holding window
                touches the ticker's next earnings (downgrade-only — it can
                only remove a candidate, never rescue one).
            earnings_buffer_days, macro_buffer_days: Event-gate buffers.
            min_history_days, enforce_history_gate: Survivorship/quality
                gate on OHLCV length (mirrors :meth:`rank_candidates_by_ev`).
            risk_free_rate: Decimal rate. ``None`` resolves it from the
                connector; an explicit value outside ``[0, 0.25]`` raises.
            dividend_yield: Decimal annual yield. ``None`` resolves it from
                the connector's fundamentals (percent column, normalised).

        Returns:
            DataFrame sorted by ``ev_per_day`` descending — one row per
            surviving ``(strike, DTE)`` candidate — with the columns of
            ``_CC_RANK_CORE_COLUMNS`` (+ ``_CC_RANK_DIAGNOSTIC_COLUMNS``
            when ``include_diagnostic_fields``). Empty but correctly
            shaped when nothing survives.

            ``.attrs["drops"]`` carries one ``{"ticker", "gate", "reason"}``
            dict per gated-out candidate — ``gate`` is one of ``data``,
            ``history``, ``strike``, ``premium``, ``event`` or
            ``ev_threshold``. Pure observability; survivor rows are
            unaffected and no extra :meth:`EVEngine.evaluate` call is made.

        §2 invariant:
            Every candidate's EV comes from a direct
            :meth:`EVEngine.evaluate` call on a properly-constructed
            ``ShortOptionTrade``. Strike enumeration, the synthetic premium
            and the gates only decide *which* candidates to score and
            which to drop — they never compute or adjust EV. There is no
            side-channel that lifts a candidate's EV.
        """
        from datetime import timedelta

        from engine.ev_engine import EVEngine, ShortOptionTrade
        from engine.event_gate import EventGate, ScheduledEvent
        from engine.forward_distribution import best_available_forward_distribution
        from engine.option_pricer import black_scholes_price
        from engine.wheel_tracker import _solve_call_strike

        # A covered call needs 100 shares per contract to be "covered".
        contracts = int(shares_held) // 100
        if contracts < 1:
            raise ValueError(
                f"rank_covered_calls_by_ev: writing a covered call requires "
                f">=100 shares to cover one contract; got shares_held={shares_held}."
            )
        # An explicit risk-free rate must be a sane decimal; None means
        # "resolve from the connector" below.
        if risk_free_rate is not None and not (0.0 <= risk_free_rate <= 0.25):
            raise ValueError(f"risk_free_rate {risk_free_rate} outside [0, 0.25]")

        conn = self.connector
        # Diagnostic drop log — one dict per gated-out candidate, exposed
        # on the returned frame's ``.attrs["drops"]``. See CLAUDE.md §2.
        drops: list[dict] = []

        cols = list(_CC_RANK_CORE_COLUMNS)
        if include_diagnostic_fields:
            cols = cols + _CC_RANK_DIAGNOSTIC_COLUMNS

        def _empty() -> pd.DataFrame:
            df = pd.DataFrame(columns=cols)
            df.attrs["drops"] = drops
            return df

        # ---- OHLCV + PIT cutoff ----
        try:
            ohlcv = conn.get_ohlcv(ticker)
        except Exception:
            drops.append({"ticker": ticker, "gate": "data", "reason": "OHLCV fetch raised"})
            return _empty()
        if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
            drops.append(
                {
                    "ticker": ticker,
                    "gate": "data",
                    "reason": "no OHLCV data (empty or missing 'close')",
                }
            )
            return _empty()
        if as_of is not None:
            try:
                ohlcv = ohlcv.loc[ohlcv.index <= pd.Timestamp(as_of)]
            except Exception:
                pass
        if ohlcv.empty:
            drops.append(
                {"ticker": ticker, "gate": "data", "reason": "no OHLCV history at or before as_of"}
            )
            return _empty()
        # Survivorship / distribution-reliability gate — mirrors
        # rank_candidates_by_ev: an empirical forward distribution from a
        # short history is statistically unreliable.
        if enforce_history_gate and len(ohlcv) < min_history_days:
            drops.append(
                {
                    "ticker": ticker,
                    "gate": "history",
                    "reason": f"history {len(ohlcv)}d < required {min_history_days}d",
                }
            )
            return _empty()

        spot = float(ohlcv["close"].iloc[-1])
        if spot <= 0:
            drops.append({"ticker": ticker, "gate": "data", "reason": "non-positive spot price"})
            return _empty()

        # ---- IV: percent→decimal normalisation (mirrors rank_candidates_by_ev) ----
        fundamentals = conn.get_fundamentals(ticker) or {}
        iv_raw = fundamentals.get("implied_vol_atm")
        if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
            iv_raw = fundamentals.get("volatility_30d")
        try:
            iv = float(iv_raw) if iv_raw is not None else 0.0
        except (TypeError, ValueError):
            iv = 0.0
        if np.isnan(iv) or iv <= 0:
            drops.append({"ticker": ticker, "gate": "data", "reason": "IV missing or non-positive"})
            return _empty()
        if iv > 3.0:
            iv = iv / 100.0
        if iv <= 0 or iv > 5:
            drops.append(
                {
                    "ticker": ticker,
                    "gate": "data",
                    "reason": "IV degenerate after percent normalisation",
                }
            )
            return _empty()

        # ---- dividend yield (decimal) ----
        if dividend_yield is None:
            dy_raw = fundamentals.get("dividend_yield", 0.0) or 0.0
            try:
                div_q = float(dy_raw)
            except (TypeError, ValueError):
                div_q = 0.0
            if not np.isfinite(div_q) or div_q < 0.0:
                div_q = 0.0
            else:
                # Bloomberg's eqy_dvd_yld_12m is in PERCENT (AUDIT-IX).
                div_q /= 100.0
                if div_q > 0.30:  # >30% is a data error, not a yield
                    div_q = 0.0
        else:
            div_q = float(dividend_yield)
            if not np.isfinite(div_q) or div_q < 0.0:
                div_q = 0.0

        # ---- risk-free rate ----
        if risk_free_rate is not None:
            rf = float(risk_free_rate)
        else:
            rf = 0.05
            try:
                rf_raw = conn.get_risk_free_rate(as_of)
                if rf_raw is not None and not (isinstance(rf_raw, float) and np.isnan(rf_raw)):
                    rf_val = float(rf_raw)
                    if rf_val > 1.0:
                        rf_val = rf_val / 100.0
                    if 0.0 <= rf_val <= 0.25:
                        rf = rf_val
            except Exception:
                pass

        today_date = date.fromisoformat(as_of) if as_of else date.today()

        # ---- event gate: register the ticker's next earnings ----
        # The gate blocks (zeroes EV on) any candidate whose holding window
        # touches earnings; per-candidate trade_start/trade_end are passed
        # to evaluate() so a short-DTE candidate can clear while a longer
        # one is blocked. days_to_earnings is surfaced regardless.
        event_gate: EventGate | None = None
        if use_event_gate:
            event_gate = EventGate(
                earnings_buffer_days=earnings_buffer_days,
                macro_buffer_days=macro_buffer_days,
            )
        days_to_earn: int | None = None
        try:
            next_earn = conn.get_next_earnings(ticker, as_of)
            if next_earn:
                earn_ts = next_earn.get("announcement_date")
                if earn_ts is not None:
                    earn_d = earn_ts.date() if hasattr(earn_ts, "date") else earn_ts
                    days_to_earn = (earn_d - today_date).days
                    if event_gate is not None:
                        event_gate.add_event(
                            ScheduledEvent(ticker=ticker, kind="earnings", event_date=earn_d)
                        )
        except Exception:
            days_to_earn = None

        ev_eng = EVEngine(event_gate=event_gate)

        # ---- ex-dividend early-assignment input (covered-call-specific) ----
        # A short call ITM into ex-div is a rational early-exercise target;
        # EVEngine adds the dividend to the expected loss when
        # days_to_ex_div <= dte. Optional + fully defensive — any failure
        # degrades to "no ex-div in the holding window".
        days_to_ex_div: int | None = None
        expected_dividend = 0.0
        try:
            if hasattr(conn, "get_next_dividend"):
                next_div = conn.get_next_dividend(ticker, as_of)
                if next_div:
                    div_ts = next_div.get("ex_date")
                    amt = next_div.get("dividend_amount", 0.0) or 0.0
                    if div_ts is not None:
                        div_d = div_ts.date() if hasattr(div_ts, "date") else div_ts
                        d2x = (div_d - today_date).days
                        amt_f = float(amt)
                        if d2x >= 0 and np.isfinite(amt_f) and amt_f > 0:
                            days_to_ex_div = d2x
                            expected_dividend = amt_f
        except Exception:
            days_to_ex_div = None
            expected_dividend = 0.0

        # ---- forward-distribution cache (one fetch per distinct DTE) ----
        fwd_cache: dict[int, tuple] = {}

        def _fwd_for(horizon: int) -> tuple:
            if horizon in fwd_cache:
                return fwd_cache[horizon]
            try:
                arr, method = best_available_forward_distribution(
                    ohlcv, horizon_days=int(horizon), as_of=as_of
                )
            except Exception:
                arr, method = None, "lognormal_fallback"
            fwd_cache[horizon] = (arr, method)
            return arr, method

        # ---- enumerate the (DTE × delta) covered-call grid ----
        rows: list[dict] = []
        for new_dte in target_dtes:
            if new_dte <= 0:
                continue
            T = max(new_dte, 1) / 365.0
            new_expiry = today_date + timedelta(days=int(new_dte))
            for tgt_delta in target_deltas:
                strike_raw = _solve_call_strike(
                    spot=spot, T=T, r=rf, q=div_q, iv=iv, target_delta=tgt_delta
                )
                if strike_raw is None:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"delta-to-strike solve did not converge "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                strike = round(strike_raw * 2) / 2  # nearest $0.50
                if strike <= spot:
                    # A covered call is sold OTM — strike must sit above spot.
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"solved strike {strike} <= spot {spot:.2f} "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                premium = black_scholes_price(
                    S=spot, K=strike, T=T, r=rf, sigma=iv, option_type="call", q=div_q
                )
                if premium <= 0.05:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "premium",
                            "reason": (
                                f"synthetic premium too thin (<=$0.05) "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue

                fwd_rets, method = _fwd_for(int(new_dte))
                trade = ShortOptionTrade(
                    option_type="call",
                    underlying=ticker,
                    spot=spot,
                    strike=float(strike),
                    premium=premium,
                    dte=int(new_dte),
                    iv=iv,
                    risk_free_rate=rf,
                    dividend_yield=div_q,
                    contracts=contracts,
                    bid=premium * 0.95,
                    ask=premium * 1.05,
                    open_interest=1000,
                    regime_multiplier=1.0,
                    days_to_ex_div=days_to_ex_div,
                    expected_dividend=expected_dividend,
                )
                res = ev_eng.evaluate(
                    trade,
                    forward_log_returns=fwd_rets,
                    trade_start=today_date,
                    trade_end=new_expiry,
                )
                # Event-gate short-circuit: a blocked candidate has its
                # ev_dollars zeroed — drop it, never rank it.
                if res.event_lockout_reason:
                    drops.append(
                        {"ticker": ticker, "gate": "event", "reason": str(res.event_lockout_reason)}
                    )
                    continue
                # Ranks, never rescues: a covered call below the EV floor
                # does not surface as tradeable.
                if res.ev_dollars < min_ev_dollars:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "ev_threshold",
                            "reason": (
                                f"ev_dollars {res.ev_dollars:.2f} < "
                                f"min_ev_dollars {min_ev_dollars:.2f} "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue

                row: dict = {
                    "ticker": ticker,
                    "spot": round(spot, 2),
                    "strike": strike,
                    "premium": round(premium, 3),
                    "dte": int(new_dte),
                    "new_expiry": new_expiry,
                    "target_delta": tgt_delta,
                    "iv": round(iv, 4),
                    "contracts": contracts,
                    "ev_dollars": round(res.ev_dollars, 2),
                    "ev_per_day": round(res.ev_per_day, 3),
                    "prob_profit": round(res.prob_profit, 4),
                    "prob_assignment": round(res.prob_assignment, 4),
                    "days_to_earnings": days_to_earn,
                    "days_to_ex_div": days_to_ex_div,
                    "distribution_source": method,
                }
                if include_diagnostic_fields:
                    row.update(
                        {
                            "cvar_5": round(res.cvar_5, 2),
                            "cvar_99_evt": (
                                round(res.cvar_99_evt, 2) if not np.isnan(res.cvar_99_evt) else None
                            ),
                            "tail_xi": (
                                round(res.tail_xi, 4) if not np.isnan(res.tail_xi) else None
                            ),
                            "heavy_tail": bool(res.heavy_tail),
                            "omega_ratio": round(res.omega_ratio, 3),
                            "fair_value": round(res.fair_value, 3),
                            "edge_vs_fair": round(res.edge_vs_fair, 2),
                            "breakeven_move_pct": round(res.breakeven_move_pct, 4),
                            "prob_touch": round(res.prob_touch, 4),
                            "total_transaction_cost": round(res.total_transaction_cost, 2),
                            "skew_pnl": round(res.skew_pnl, 3),
                            "expected_dividend": round(expected_dividend, 4),
                            "regime_multiplier": round(res.regime_multiplier, 4),
                        }
                    )
                rows.append(row)

        if not rows:
            return _empty()
        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("ev_per_day", ascending=False).head(top_n).reset_index(drop=True)
        # Drop log attached after sort/head so it rides on the exact frame
        # returned; survivor rows are untouched (CLAUDE.md §2).
        df.attrs["drops"] = drops
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
        universe_limit: int | None = None,
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
            use_event_gate / earnings_buffer_days / macro_buffer_days /
                universe_limit: Forwarded to :meth:`rank_candidates_by_ev`.

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
            universe_limit=universe_limit,
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
