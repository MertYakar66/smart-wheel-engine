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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from engine.risk_manager import DEFAULT_SECTOR_MAP

if TYPE_CHECKING:
    from engine.wheel_tracker import WheelTracker

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


def _resolve_pit_atm_iv(conn, ticker: str, as_of: str | None) -> float | None:
    """Best-effort point-in-time ATM IV from the connector, normalised to decimal.

    Mirrors :meth:`engine.wheel_tracker.WheelTracker._connector_atm_iv`
    for use by the rankers, which historically used
    ``conn.get_fundamentals(ticker)["implied_vol_atm"]`` — a snapshot
    column with no date axis. That meant a ranker run with
    ``as_of="2026-03-05"`` priced strikes against today's IV, not the
    IV that was actually quoted on 2026-03-05 — distorting EV for any
    name whose IV had since drifted (S23 F3: AVGO 42.96% snapshot vs
    49.5% PIT on 2026-03-05, a 15% relative error in the IV input to
    the BSM strike-solve and synthetic-premium computation).

    Returns the composite ``(hist_put_imp_vol + hist_call_imp_vol) / 2``
    from the most recent row of ``get_iv_history(ticker, end_date=as_of)``,
    normalised to a decimal — or ``None`` when no connector is attached,
    the connector lacks ``get_iv_history`` (e.g. ``ThetaConnector``), the
    lookup raises, or the result is empty / unusable. ``None`` signals
    the caller to fall back to the legacy snapshot fundamentals path —
    which preserves connectors and tests that stub only
    ``get_fundamentals``.
    """
    if conn is None or not hasattr(conn, "get_iv_history"):
        return None
    try:
        hist = conn.get_iv_history(ticker, end_date=as_of)
    except Exception:
        return None
    if hist is None or len(hist) == 0:
        return None
    cols = [c for c in ("hist_put_imp_vol", "hist_call_imp_vol") if c in hist.columns]
    if not cols:
        return None
    try:
        row = hist.iloc[-1]
        vals = [float(row[c]) for c in cols if pd.notna(row[c])]
    except (KeyError, TypeError, ValueError):
        return None
    if not vals:
        return None
    iv = sum(vals) / len(vals)
    # Bloomberg vol_iv columns are in PERCENT (e.g. 30.79). Anything
    # above 3.0 (= 300%) is virtually certainly a percent representation;
    # below is already decimal. Mirrors the legacy snapshot path's
    # heuristic in the three rankers.
    if iv > 3.0:
        iv = iv / 100.0
    if not (0.0 < iv <= 5.0):
        return None
    return iv


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
    # Raw P&L distribution spread (pre regime/dealer multipliers) — mirrors
    # the put-side ranker row; without these the row dict's pnl_p25/50/75
    # keys are silently dropped by `pd.DataFrame(rows, columns=cols)` below.
    "pnl_p25",
    "pnl_p50",
    "pnl_p75",
    "prob_profit",
    "prob_assignment",
    "days_to_earnings",
    "days_to_ex_div",
    "distribution_source",
    "sector",  # S31 F2 / F6 closer — GICS sector per DEFAULT_SECTOR_MAP
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

# ----------------------------------------------------------------------
# Column schema for WheelRunner.rank_strangles_by_ev
# ----------------------------------------------------------------------
# A strangle is two short legs. ``ev_dollars`` is the *composed* EV
# (put leg + call leg) — expected value is linear, so the sum is exact.
# The per-leg risk columns (``put_*`` / ``call_*``) are reported
# separately and are deliberately NOT summed: the strangle payoff is
# nonlinear in the shared underlying path, so a summed CVaR / prob would
# be wrong. A joint combined-risk metric is a documented follow-up.
_STRANGLE_RANK_CORE_COLUMNS = [
    "ticker",
    "spot",
    "dte",
    "expiry",
    "target_delta",
    "iv",
    "contracts",
    "put_strike",
    "call_strike",
    "put_premium",
    "call_premium",
    "total_premium",
    "ev_dollars",
    "put_ev_dollars",
    "call_ev_dollars",
    "lower_breakeven",
    "upper_breakeven",
    "days_to_earnings",
    "timing_score",
    "timing_recommendation",
    "distribution_source",
    "sector",  # S31 F2 / F6 closer — GICS sector per DEFAULT_SECTOR_MAP
]
_STRANGLE_RANK_DIAGNOSTIC_COLUMNS = [
    "put_prob_profit",
    "call_prob_profit",
    "put_prob_assignment",
    "call_prob_assignment",
    "put_cvar_5",
    "call_cvar_5",
    "put_edge_vs_fair",
    "call_edge_vs_fair",
    "total_transaction_cost",
    "timing_phase",
]


def _attach_drops_summary(frame: pd.DataFrame, drops: list[dict]) -> pd.DataFrame:
    """Attach the drops list AND a trader-facing roll-up summary to a
    ranker output frame.

    Both attributes ride on ``frame.attrs`` so survivor rows are
    untouched (CLAUDE.md §2). The summary closes S31 F1/F4
    discoverability: a caller scanning the frame sees at a glance
    "20 dropped — 17 event, 3 history" without iterating the full
    drops list.

    - ``attrs["drops"]``: the existing per-candidate list of
      ``{"ticker", "gate", "reason"}`` dicts (or
      ``{"new_dte", "target_delta", "gate", "reason"}`` for
      :meth:`WheelTracker.suggest_rolls` siblings).
    - ``attrs["drops_summary"]``: ``{"total_dropped": int, "by_gate":
      {gate: count}}``. Gate-set matches whatever the per-call drops
      taxonomy used (no new gate labels invented here).

    Used by every ranker return path in this module so the shape is
    consistent across ``rank_candidates_by_ev``,
    ``rank_covered_calls_by_ev``, ``rank_strangles_by_ev``, and the
    multi-grid ``select_book`` aggregator. The same helper is defined
    in :mod:`engine.wheel_tracker` for ``suggest_rolls`` /
    ``suggest_call_rolls`` (small duplication to keep the
    decision-layer modules import-independent).
    """
    from collections import Counter

    frame.attrs["drops"] = drops
    frame.attrs["drops_summary"] = {
        "total_dropped": len(drops),
        "by_gate": dict(Counter(d["gate"] for d in drops)),
    }
    return frame


def make_live_book_tracker(
    initial_capital: float = 100_000.0,
    connector: Any | None = None,
    **kwargs: Any,
) -> "WheelTracker":
    """Canonical production / live-book ``WheelTracker`` constructor.

    The library default (``WheelTracker(...)``) leaves the D17 concentration
    caps OFF — research-safe, matching the ``require_ev_authority`` convention
    (default off; production sets it on). Production / live books MUST be
    constructed through this factory, which ARMS the refusal-only
    concentration caps — **R9 sector (25% NAV)** and **R10 single-name
    (10% NAV)** — so "off in the library, on in production" is enforced in
    code, not promised in prose (heavy-verify 2026-05-31 Category A; #154
    follow-up). ``tests/test_production_tracker_caps.py`` pins that a tracker
    from this factory refuses an over-concentrated book, so the production
    arming is an invariant a future change cannot silently drop.

    §2-safe: the caps only REFUSE an over-concentrated open; they never touch
    ``ev_raw`` / ``ev_dollars`` / ``prob_profit``. ``enforce_delta_cap`` /
    ``enforce_kelly_cap`` stay OFF (deferred) even here: the delta cap's
    $300/$100k-NAV calibration would refuse essentially every post-assignment
    wheel book until re-calibrated. Pass ``require_ev_authority=True`` via
    ``kwargs`` to additionally arm the D16 token gate + delta/Kelly.
    """
    from engine.wheel_tracker import WheelTracker

    return WheelTracker(
        initial_capital=initial_capital,
        connector=connector,
        enforce_sector_cap=True,
        enforce_single_name_cap=True,
        **kwargs,
    )


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
        self._hmm_regime_cache: dict[tuple[str, int], tuple[float, str, bool]] = {}

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

    def analyze_ticker(
        self,
        ticker: str,
        as_of: str | None = None,
        *,
        max_as_of_staleness_days: int = 30,
    ) -> TickerAnalysis:
        """
        Complete wheel suitability analysis for a single ticker.

        Combines fundamentals, volatility, events, and strangle timing.

        S33 audit holdover: spot price now respects ``as_of`` (PIT
        filtered to ``<= as_of``) and refuses to silently substitute
        when ``as_of`` is more than ``max_as_of_staleness_days``
        beyond the latest available OHLCV row. On a stale-as_of
        rejection, ``spot_price`` stays at its default 0.0 -- the
        same "no data available" signal callers already handle.
        Mirrors the gate applied to ``rank_candidates_by_ev`` /
        ``rank_covered_calls_by_ev`` / ``rank_strangles_by_ev``
        (PRs #215 / #220) for surface consistency. Default 30 days
        permits normal weekend / holiday / refresh-cycle gaps.
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

        # --- Spot price (PIT + staleness gate; S33 audit holdover) ---
        ohlcv = conn.get_ohlcv(ticker)
        if not ohlcv.empty:
            # PIT filter: respect as_of by trimming OHLCV to <= as_of.
            if as_of is not None:
                try:
                    cutoff = pd.Timestamp(as_of)
                    ohlcv_pit = ohlcv.loc[ohlcv.index <= cutoff]
                    # Staleness gate: refuse to substitute if as_of is
                    # more than max_as_of_staleness_days beyond the
                    # latest filtered row. Leaves spot_price at the
                    # 0.0 default (the "no data available" signal).
                    if not ohlcv_pit.empty:
                        gap_days = (cutoff - ohlcv_pit.index.max()).days
                        if gap_days <= max_as_of_staleness_days:
                            analysis.spot_price = float(ohlcv_pit["close"].iloc[-1])
                        else:
                            logger.warning(
                                "analyze_ticker(%s, as_of=%s): %d days beyond "
                                "latest OHLCV (%s); spot_price left at default 0.0",
                                ticker,
                                as_of,
                                gap_days,
                                ohlcv_pit.index.max().date().isoformat(),
                            )
                except Exception:
                    # PIT-filter cast failed -- fall back to live behavior.
                    analysis.spot_price = float(ohlcv["close"].iloc[-1])
            else:
                # No as_of -> live behaviour: latest close.
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
        max_as_of_staleness_days: int = 30,
        refuse_stale_live: bool = False,
    ) -> pd.DataFrame:
        """Rank tickers by **probabilistic expected value** for a short-put wheel entry.

        This is the audit-grade replacement for ``screen_candidates``. The
        old ranker used a heuristic composite score; this one uses
        :class:`engine.ev_engine.EVEngine` with a PIT-safe empirical
        forward-return distribution pulled from the ticker's OHLCV history.

        For each ticker:
          1. Pull OHLCV up to ``as_of`` from the connector.
          2. Pull ATM IV at ``as_of`` — :func:`_resolve_pit_atm_iv` reads
             the most recent row of ``conn.get_iv_history(end_date=as_of)``
             (S23 F3); falls back to ``fundamentals['implied_vol_atm']``
             (then ``volatility_30d``) only when no PIT history is
             available, e.g. on a connector without ``get_iv_history``.
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
                When ``None`` the spot is the latest available close. If that
                close lags ``date.today()`` by more than
                ``max_as_of_staleness_days`` (e.g. a stale data cache or a
                missed refresh), the engine logs a one-time warning rather than
                silently pricing off a back-dated spot (D-2 fix, 2026-06-15);
                set ``refuse_stale_live=True`` to hard-drop such candidates.
                Every row carries ``spot_date`` — the trading date the spot is
                actually priced from — so the staleness is never silent.
            refuse_stale_live: When ``as_of is None`` and the latest close is
                more than ``max_as_of_staleness_days`` behind ``date.today()``,
                drop the candidate with a ``gate="data"`` reason instead of
                warning-and-ranking. Default ``False`` so backtests that run
                ``as_of=None`` against committed (necessarily back-dated) data
                are unaffected; arm it for real-money live operation.
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
        from engine.forward_distribution import (
            best_available_forward_distribution,
            realized_vol_widened_log_returns,
            realized_vol_widening_factor,
        )
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
        # Warn at most once per call when the live (as_of=None) spot is stale —
        # avoids per-ticker log spam across a 100-name universe.
        warned_stale_live = False

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

            # S32 F3 closer: refuse a future as_of when the actual data
            # ends more than `max_as_of_staleness_days` before. Before
            # this gate the engine silently used the latest available
            # close as the "current spot" for any future as_of (e.g.
            # querying as_of=2030-01-01 against 2026-03-20 data returned
            # a row with 2026 spot, NO warning) -- a D11 "no silent
            # substitution" violation. The default 30-day tolerance
            # allows the normal weekend/holiday/refresh-cycle gap; a
            # year-out as_of correctly drops.
            if as_of is not None:
                try:
                    cutoff = pd.Timestamp(as_of)
                    actual_last = ohlcv.index.max()
                    gap_days = (cutoff - actual_last).days
                    if gap_days > max_as_of_staleness_days:
                        drops.append(
                            {
                                "ticker": ticker,
                                "gate": "data",
                                "reason": (
                                    f"as_of {as_of} is {gap_days} days beyond "
                                    f"latest data ({actual_last.date().isoformat()}); "
                                    f"max_as_of_staleness_days={max_as_of_staleness_days}"
                                ),
                            }
                        )
                        continue
                except Exception:
                    pass

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

            # spot_date is the actual trading date the spot is priced from
            # (iloc[-1] after any as_of PIT trim). Surfaced on every row so an
            # operator never has to assume "spot == today".
            spot_date = ohlcv.index[-1]

            # Live-mode spot staleness (as_of is None). The as_of-provided gate
            # above only fires when as_of is set, so on the live path the engine
            # would otherwise price off the latest close as "today's" spot even
            # when that close is months old (stale cache / missed refresh) — the
            # same D11 "no silent substitution" the as_of gate prevents, but
            # live (adversarial-review D-2, 2026-06-15). Default: warn once and
            # rank (non-breaking — backtests run as_of=None against committed
            # data). refuse_stale_live=True hard-drops stale-live candidates for
            # real trading, where pricing off a back-dated spot is unacceptable.
            if as_of is None:
                try:
                    live_gap_days = (pd.Timestamp(date.today()) - pd.Timestamp(spot_date)).days
                except Exception:
                    live_gap_days = 0
                if live_gap_days > max_as_of_staleness_days:
                    if refuse_stale_live:
                        drops.append(
                            {
                                "ticker": ticker,
                                "gate": "data",
                                "reason": (
                                    f"live spot is {live_gap_days} days stale "
                                    f"(latest data {spot_date.date().isoformat()}, "
                                    f"today {date.today().isoformat()}); "
                                    f"refuse_stale_live=True, "
                                    f"max_as_of_staleness_days={max_as_of_staleness_days}"
                                ),
                            }
                        )
                        continue
                    if not warned_stale_live:
                        logger.warning(
                            "rank_candidates_by_ev: live spot is %d days stale "
                            "(latest OHLCV %s vs today %s) — pricing off a back-dated "
                            "close. Refresh data, pass an explicit as_of, or set "
                            "refuse_stale_live=True to drop stale-live candidates.",
                            live_gap_days,
                            spot_date.date().isoformat(),
                            date.today().isoformat(),
                        )
                        warned_stale_live = True

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
            # S23 F3 fix: prefer the connector's as-of IV via
            # ``get_iv_history`` over ``fundamentals['implied_vol_atm']``,
            # which is a single snapshot with no date column and silently
            # mis-prices any historical ``as_of`` run. The fundamentals
            # path stays as a fallback for connectors / test stubs that
            # don't expose ``get_iv_history``.
            #
            # AUDIT-VIII P0.1: Bloomberg fundamentals CSV reports IV and
            # volatility in PERCENT (e.g. ``26.15`` means 26.15% annualized).
            # Earlier code treated the raw value as a decimal, then rejected
            # it as ``iv > 5`` (degenerate), which caused every candidate
            # to be dropped — the EV ranker silently returned zero rows.
            # We normalize to a decimal by dividing by 100 when the raw
            # value is clearly a percentage (>3) and guard NaN/None.
            fundamentals = conn.get_fundamentals(ticker) or {}
            iv = _resolve_pit_atm_iv(conn, ticker, as_of)
            if iv is None:
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
                # S23 F1 — symmetric back-buffer. The gate's
                # _event_touches_window arithmetic is symmetric and its
                # reason string says ±{buf}d, but get_next_earnings only
                # ever returns future events. Also pull the most recent
                # PAST earnings within the back-buffer and register it
                # so the gate can fire on a trade opened immediately
                # post-earnings (the IV-crush window the gate's
                # docstring explicitly cites as motivation). Defensive
                # hasattr() so connectors / test stubs without the new
                # method continue working with the legacy behavior.
                if event_gate is not None and hasattr(conn, "get_recent_earnings"):
                    recent_earn = conn.get_recent_earnings(
                        ticker, as_of, lookback_days=earnings_buffer_days
                    )
                    if recent_earn:
                        recent_ts = recent_earn.get("announcement_date")
                        if recent_ts is not None:
                            recent_d = recent_ts.date() if hasattr(recent_ts, "date") else recent_ts
                            event_gate.add_event(
                                ScheduledEvent(
                                    ticker=ticker,
                                    kind="earnings",
                                    event_date=recent_d,
                                )
                            )
                if event_gate is None:
                    # Soft fallback (use_event_gate=False) — also
                    # symmetric. Forward branch was the original
                    # behavior; back branch is the S23 F1 fix.
                    if days_to_earn is not None and 0 <= days_to_earn < earnings_buffer_days:
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
                    if hasattr(conn, "get_recent_earnings"):
                        recent_earn = conn.get_recent_earnings(
                            ticker, as_of, lookback_days=earnings_buffer_days
                        )
                        if recent_earn:
                            r_ts = recent_earn.get("announcement_date")
                            if r_ts is not None:
                                r_d = r_ts.date() if hasattr(r_ts, "date") else r_ts
                                d_since = (today_date - r_d).days
                                if 0 <= d_since < earnings_buffer_days:
                                    drops.append(
                                        {
                                            "ticker": ticker,
                                            "gate": "event",
                                            "reason": (
                                                f"earnings was {d_since}d ago "
                                                f"< buffer {earnings_buffer_days}d"
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
            # Provenance: this branch is the only one that constructs `premium`
            # today; a future market-mid path would set `"market_mid"` instead.
            # Consumers use this to tell a real quote from a synthetic one --
            # and, consequently, to know that ``edge_vs_fair`` is structurally
            # zero on the synthetic path (premium and fair both come from the
            # same BSM call in ev_engine.evaluate).
            premium_source = "synthetic_bsm"
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
            # Audit flag: True when the regime multiplier is trustworthy — a
            # clean EM convergence, or the neutral no-fit path (short history →
            # multiplier stays 1.0, no adjustment). False when the HMM WAS
            # fitted but did NOT converge within n_iter, or the fit errored:
            # the multiplier is still applied (changing that would shift
            # baselines) but should be read as low-confidence. Closes the
            # "silent non-convergence" gap (adversarial-review Dim-2, 2026-06-15)
            # — converged was computed in HMMFit and then discarded here.
            hmm_converged = True
            # S33 F4 closer: realized vol / mean over the 252d window
            # the HMM saw at fit time. The "crisis" label by itself means
            # "high-vol regime" regardless of return direction (S33
            # documented Feb 2026 labeling as 'crisis' with positive
            # annualized return) -- a trader assuming "crisis = crashing"
            # mis-anchors. Surfacing the underlying vol + return lets
            # them disambiguate without re-fitting the HMM. NaN when the
            # HMM did not run (history too short or fit failed).
            hmm_realized_vol_252d_ann = float("nan")
            hmm_realized_return_252d_ann = float("nan")
            try:
                from engine.regime_hmm import GaussianHMM

                log_rets = np.diff(np.log(ohlcv["close"].values))
                if len(log_rets) >= 200:
                    tail = log_rets[-504:]
                    # Realized vol / mean from the most recent 252 days
                    # of the same window the HMM fitted to. Computed
                    # outside the cache lookup so both fresh and cached
                    # paths emit consistent disambiguation stats.
                    if len(tail) >= 252:
                        tail_252 = tail[-252:]
                        hmm_realized_vol_252d_ann = float(np.std(tail_252) * np.sqrt(252))
                        hmm_realized_return_252d_ann = float(np.mean(tail_252) * 252)
                    fp = (
                        len(tail),
                        round(float(tail[0]), 6),
                        round(float(tail[len(tail) // 2]), 6),
                        round(float(tail[-1]), 6),
                    )
                    cache_key = (ticker, hash(fp))
                    cached = self._hmm_regime_cache.get(cache_key)
                    if cached is not None:
                        hmm_regime_mult, hmm_regime, hmm_converged = cached
                    else:
                        hmm = GaussianHMM(n_states=4, n_iter=20, random_state=42)
                        hmm.fit(tail)
                        hmm_converged = bool(getattr(hmm.fit_result, "converged", False))
                        probs = hmm.predict_proba(tail)
                        hmm_regime_mult = float(hmm.position_multiplier(probs[-1]))
                        # Label is the argmax state -- a pure read of the
                        # same posterior, in its own try so it can never
                        # perturb the already-computed multiplier.
                        try:
                            hmm_regime = hmm.fit_result.state_labels[int(np.argmax(probs[-1]))]
                        except Exception:
                            hmm_regime = "unknown"
                        self._hmm_regime_cache[cache_key] = (
                            hmm_regime_mult,
                            hmm_regime,
                            hmm_converged,
                        )
            except Exception:
                hmm_regime_mult = 1.0
                hmm_regime = "unknown"
                hmm_converged = False

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
            oi_source = "fallback"
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
                                oi_source = "chain"
                except Exception:
                    pass

            # Skew multiplier: steepening put skew is a risk-off signal.
            # skew_slope(iv_25d_put, iv_atm, iv_25d_call) returns
            # (iv_25d_put - iv_25d_call) / iv_atm. Larger = steeper put skew.
            # Map slope -> multiplier in [0.85, 1.08]. Positive slope
            # (normal risk-off) cuts multiplier; negative slope (call-skew
            # risk-on, rare in equities) boosts it slightly.
            #
            # ``skew_source`` provenance (S29 Fix #1, mirrors S1B's
            # ``oi_source`` / ``premium_source``): tells a trader whether
            # ``skew_multiplier`` reflects measured chain data
            # (``"chain"``) or just the unmeasured-default identity
            # (``"unavailable"``). On the Bloomberg connector
            # ``chain_df`` is always ``None`` (no ``get_options`` /
            # ``get_option_chain`` method), so this column is the
            # honest signal that ``skew_multiplier=1.0`` means
            # "not measured" rather than "measured neutral". S29
            # confirmed the block is uniformly dormant on Bloomberg.
            skew_mult = 1.0
            skew_diag: dict = {}
            skew_source = "unavailable"
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
                                skew_source = "chain"
                except Exception:
                    skew_mult = 1.0
                    # Leave skew_source = "unavailable"; the calc failed.

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
                                # PIT: anchor dealer time-to-expiry to as_of, not
                                # wall-clock now() (D-review E4). None = live.
                                as_of=pd.Timestamp(as_of) if as_of else None,
                            )
                except Exception:
                    # Graceful degrade — dealer positioning is optional
                    market_structure = None

            # F4 follow-up — realized-vol-ratio widening. When the
            # ticker's 30d realized vol is materially elevated vs its
            # 1y baseline (rv30/rv252 >= 1.30), widen the forward
            # distribution's std by a gentle factor (max 1.15) before
            # the EV engine evaluates. Captures vol-clustering. No-op
            # on calm regimes (86% of dates). Sign- and mean-preserving.
            # Replaces the rolled-back Fix B1+C (HMM-multiplier
            # widening that inverted S27 ρ — see
            # docs/F4_TAIL_RISK_DIAGNOSTIC.md §10). The new signal
            # has 2.07x lift on tail-realized dates (vs HMM's ~1.3x)
            # and fires on 14% of dates (vs HMM's 98%) — calibrated
            # to preserve S27 ρ. Does NOT close named F4 cases
            # (COST 2022-04 had rv30/rv252 = 0.96, below threshold —
            # the named cases are fundamentally unpredictable; the
            # R10 single-name cap is the damage-bounding mechanism).
            fwd_rets_widened = realized_vol_widened_log_returns(
                fwd_rets,
                ohlcv,
                as_of=as_of,
            )
            tail_widening_factor = realized_vol_widening_factor(
                ohlcv,
                as_of=as_of,
            )
            res = ev_eng.evaluate(
                trade,
                forward_log_returns=fwd_rets_widened,
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
                # Raw P&L distribution spread (pre regime/dealer
                # multipliers). Headline fields so the operator reads
                # the verdict as a distribution, not a point estimate.
                # NaN-safe via None: small-distribution paths and the
                # event-lockout short-circuit both leave these as nan
                # on EVResult.
                "pnl_p25": (round(res.pnl_p25, 2) if not np.isnan(res.pnl_p25) else None),
                "pnl_p50": (round(res.pnl_p50, 2) if not np.isnan(res.pnl_p50) else None),
                "pnl_p75": (round(res.pnl_p75, 2) if not np.isnan(res.pnl_p75) else None),
                "collateral": round(collateral, 2),
                "roc": round(roc, 6),
                "prob_profit": round(res.prob_profit, 4),
                "prob_assignment": round(res.prob_assignment, 4),
                "days_to_earnings": days_to_earn,
                # Provenance: the trading date the spot (and thus the strike
                # solve, synthetic premium, and EV) is priced from. On an
                # as_of run this is the last bar <= as_of; on a live (as_of=
                # None) run it is the latest available close, which may lag
                # "today" — see the live-staleness gate above (D-2 fix).
                "spot_date": spot_date.date().isoformat(),
                # Provenance must match what the engine ACTUALLY computed EV
                # from. The engine reports the generic "empirical" when it
                # consumed our forward returns — surface the specific sampler
                # (`method`) in that case for detail. But when the cascade
                # returned "none" and the engine fell back to the IV lognormal,
                # report the engine's real source ("lognormal_fallback"), never
                # the stale "none" the cascade label would otherwise leak into
                # the row and the EV-authority token hash.
                "distribution_source": (
                    method
                    if res.metadata.get("distribution_source") == "empirical"
                    else res.metadata.get("distribution_source", method)
                ),
                # S31 F2 / F6 closer: GICS sector for the underlying.
                # Same source the sector_cap gate uses
                # (engine.portfolio_risk_gates.check_sector_cap →
                # engine.risk_manager.SectorExposureManager) so the
                # trader sees the SAME sector label the gate would
                # aggregate by. Closes the F6 trader-mental-model gap
                # where "tech" colloquially crosses three GICS sectors
                # (Info Tech / Cons Disc / Comm Svcs) and the ranker
                # output offered no per-row sector tag for
                # post-hoc verification.
                "sector": DEFAULT_SECTOR_MAP.get(ticker, "Unknown"),
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
                        # Provenance: "chain" means skew was measured from
                        # the per-strike option chain; "unavailable" means
                        # the connector has no chain access OR the chain
                        # lacked the 25Δ points the slope calc needs. On
                        # Bloomberg-CSV path this is always "unavailable"
                        # (no get_options method); on Theta it reflects
                        # per-call outcome. Closes S29 Fix #1 — separates
                        # "measured neutral skew" from "not measured at
                        # all" in the diagnostic row.
                        "skew_source": skew_source,
                        "hmm_multiplier": round(hmm_regime_mult, 4),
                        "hmm_regime": hmm_regime,
                        # False ⇒ the HMM was fitted but did not converge (or
                        # the fit errored) — hmm_multiplier was still applied
                        # but is low-confidence. True on clean fit or the
                        # neutral no-fit path. Audit-only; does not gate.
                        "hmm_converged": hmm_converged,
                        # F4 follow-up: realized-vol-ratio widening
                        # factor (1.00 = no widening, > 1.0 = vol-
                        # cluster regime fired). Audit signal for the
                        # post-rollback widening mechanism. See
                        # engine.forward_distribution.realized_vol_widening_factor.
                        "tail_widening_factor": round(tail_widening_factor, 4),
                        # S33 F4 disambiguation: realized vol + mean
                        # over the same 252d window the HMM fitted to.
                        # The "crisis" label alone means "high-vol
                        # regime regardless of direction" -- these
                        # two columns let the trader see WHY the HMM
                        # called crisis (genuinely crashing vs
                        # high-vol-with-positive-trend). NaN when the
                        # HMM did not run (short history / fit fail).
                        "hmm_realized_vol_252d_ann": (
                            round(hmm_realized_vol_252d_ann, 4)
                            if not np.isnan(hmm_realized_vol_252d_ann)
                            else None
                        ),
                        "hmm_realized_return_252d_ann": (
                            round(hmm_realized_return_252d_ann, 4)
                            if not np.isnan(hmm_realized_return_252d_ann)
                            else None
                        ),
                        "news_multiplier": round(news_mult, 4),
                        "news_sentiment": round(news_sentiment, 4),
                        "news_n_articles": news_n_articles,
                        "credit_multiplier": round(credit_mult, 4),
                        "credit_regime": credit_regime,
                        # S31 F9 closer: surface the engine's FINAL regime
                        # multiplier (= ev_dollars / ev_raw), the scalar
                        # that actually scaled the EV. Differs from the
                        # input combined_regime_mult (hmm × skew × news ×
                        # credit) by the engine's clamp to [0.0, 1.25],
                        # heavy_tail_penalty if heavy_tail, and dealer_mult.
                        # Mirrors the pattern at line 2315 (the strangle
                        # ranker's emit) for consistency.
                        "regime_multiplier": round(res.regime_multiplier, 4),
                        "strike_open_interest": strike_oi,
                        # Provenance: "chain" means OI came from the real chain;
                        # "fallback" means the 1000 placeholder was used. On the
                        # bloomberg-CSV path the connector exposes no chain so
                        # this is always "fallback"; on a real-chain path it
                        # reflects the per-strike lookup outcome.
                        "oi_source": oi_source,
                        # Provenance: "synthetic_bsm" means `premium` was
                        # constructed from BSM at the engine boundary (the
                        # current bloomberg-CSV path); "market_mid" would
                        # indicate a real quote. When this is "synthetic_bsm",
                        # `edge_vs_fair` is structurally zero (the engine prices
                        # `fair` from the same BSM inputs).
                        "premium_source": premium_source,
                        "chain_quality_warning": chain_quality_blocked_reason or None,
                    }
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ev_per_day", ascending=False).head(top_n)
        # Diagnostic drop log + S31 F1/F4 discoverability summary —
        # attached after the sort/head so it rides on the exact frame
        # returned (empty or not). Survivor rows are untouched; see
        # CLAUDE.md section 2.
        return _attach_drops_summary(df, drops)

    # ------------------------------------------------------------------
    # Single-ticker surface exploration (investor-scenario 2 follow-up)
    # ------------------------------------------------------------------
    def explore_ticker(
        self,
        ticker: str,
        deltas: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35),
        dtes: tuple[int, ...] = (21, 28, 35, 42, 49, 60),
        contracts: int = 1,
        as_of: str | None = None,
        include_diagnostic_fields: bool = True,
        **rank_kwargs,
    ) -> pd.DataFrame:
        """Return a (delta, DTE) grid of short-put candidates for one ticker.

        Where :meth:`rank_candidates_by_ev` solves for **one** strike per
        ticker via brentq on a single ``delta_target`` and emits one row,
        this method evaluates every (delta, DTE) cell in the grid and
        returns one row per cell. The strike-solve, premium construction,
        forward-distribution pull, and EV evaluation are reused
        cell-by-cell via :meth:`rank_candidates_by_ev` itself — every row
        is produced by the same authoritative EV path (§2-safe, no
        parallel implementation).

        Investor-scenario 2 logged that the screener emits a single
        ``(delta=0.25, DTE=35)`` cell per ticker, while a direct grid
        scan shows 14 of 48 cells with strictly higher pre-multiplier EV
        than the screener's pick. This method surfaces the same grid
        through the production EV path so investors (or an AI
        explanation layer) can compare strikes/DTEs after the screener
        has surfaced a candidate.

        Args:
            ticker: Single ticker to explore. For multi-ticker screening
                use :meth:`rank_candidates_by_ev` directly; this method
                is the per-ticker follow-up.
            deltas: Tuple of target put deltas (positive, OTM). Each
                element becomes ``delta_target`` for one inner ranker call.
            dtes: Tuple of target DTEs in days. Each element becomes
                ``dte_target`` for one inner ranker call.
            contracts: Number of contracts per cell (constant across grid).
            as_of: PIT cutoff date string (same semantics as
                :meth:`rank_candidates_by_ev`).
            include_diagnostic_fields: Forwarded to the inner ranker.
            **rank_kwargs: Additional kwargs forwarded to the inner
                ranker (e.g. ``use_dealer_positioning=False`` to skip
                chain-fetch overhead, ``enforce_history_gate=False`` for
                a short-history research path).

        Returns:
            DataFrame with one row per (delta, DTE) cell, sorted
            descending by ``ev_dollars``. The output schema is the
            inner ranker's schema with an additional ``delta_target``
            column (inserted after ``ticker``) carrying the cell's
            target delta. Cells the inner ranker drops (history gate,
            strike-solver failure, event lockout, etc.) are absent from
            the returned frame; the union of all inner-ranker drops is
            exposed on ``df.attrs["drops"]`` with each entry tagged by
            its ``delta_target`` and ``dte_target``.
        """
        rows: list[pd.DataFrame] = []
        all_drops: list[dict] = []
        for d in deltas:
            for t in dtes:
                sub = self.rank_candidates_by_ev(
                    tickers=[ticker],
                    delta_target=float(d),
                    dte_target=int(t),
                    contracts=contracts,
                    top_n=1,
                    # The grid's purpose is to surface every cell, so we
                    # disable the EV floor here. Callers that want to
                    # filter post-hoc can apply ``df[df.ev_dollars > X]``.
                    min_ev_dollars=-1e9,
                    as_of=as_of,
                    include_diagnostic_fields=include_diagnostic_fields,
                    **rank_kwargs,
                )
                if sub is not None and len(sub) > 0:
                    sub = sub.copy()
                    # Insert delta_target right after ticker so the grid
                    # frame reads naturally left-to-right.
                    sub.insert(1, "delta_target", float(d))
                    rows.append(sub)
                inner_drops = sub.attrs.get("drops", []) if sub is not None else []
                for drop in inner_drops:
                    all_drops.append({**drop, "delta_target": float(d), "dte_target": int(t)})
        if not rows:
            out = pd.DataFrame()
        else:
            out = pd.concat(rows, ignore_index=True)
            out = out.sort_values("ev_dollars", ascending=False).reset_index(drop=True)
        return _attach_drops_summary(out, all_drops)

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
        max_as_of_staleness_days: int = 30,
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
            return _attach_drops_summary(df, drops)

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

        # S33 F3 follow-up: mirror the as_of-beyond-data gate from
        # rank_candidates_by_ev (PR #215). Without it the CC ranker
        # silently substituted the latest available close as "current
        # spot" for any future as_of — same D11 violation, same fix
        # shape.
        if as_of is not None:
            try:
                cutoff = pd.Timestamp(as_of)
                actual_last = ohlcv.index.max()
                gap_days = (cutoff - actual_last).days
                if gap_days > max_as_of_staleness_days:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "data",
                            "reason": (
                                f"as_of {as_of} is {gap_days} days beyond "
                                f"latest data ({actual_last.date().isoformat()}); "
                                f"max_as_of_staleness_days={max_as_of_staleness_days}"
                            ),
                        }
                    )
                    return _empty()
            except Exception:
                pass

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

        # ---- IV: PIT-first via get_iv_history, fallback to fundamentals snapshot ----
        # S23 F3 fix: same as rank_candidates_by_ev.
        fundamentals = conn.get_fundamentals(ticker) or {}
        iv = _resolve_pit_atm_iv(conn, ticker, as_of)
        if iv is None:
            iv_raw = fundamentals.get("implied_vol_atm")
            if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
                iv_raw = fundamentals.get("volatility_30d")
            try:
                iv = float(iv_raw) if iv_raw is not None else 0.0
            except (TypeError, ValueError):
                iv = 0.0
            if np.isnan(iv) or iv <= 0:
                drops.append(
                    {"ticker": ticker, "gate": "data", "reason": "IV missing or non-positive"}
                )
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
            # S23 F1 — symmetric back-buffer (matches rank_candidates_by_ev).
            # Pull the most recent past earnings within the back-buffer
            # so the gate can fire on a trade opened immediately
            # post-earnings (IV-crush window). Defensive hasattr() for
            # legacy connectors.
            if event_gate is not None and hasattr(conn, "get_recent_earnings"):
                recent_earn = conn.get_recent_earnings(
                    ticker, as_of, lookback_days=earnings_buffer_days
                )
                if recent_earn:
                    r_ts = recent_earn.get("announcement_date")
                    if r_ts is not None:
                        r_d = r_ts.date() if hasattr(r_ts, "date") else r_ts
                        event_gate.add_event(
                            ScheduledEvent(ticker=ticker, kind="earnings", event_date=r_d)
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
                    # Raw P&L distribution spread (pre multipliers); see
                    # rank_candidates_by_ev row above for rationale.
                    "pnl_p25": (round(res.pnl_p25, 2) if not np.isnan(res.pnl_p25) else None),
                    "pnl_p50": (round(res.pnl_p50, 2) if not np.isnan(res.pnl_p50) else None),
                    "pnl_p75": (round(res.pnl_p75, 2) if not np.isnan(res.pnl_p75) else None),
                    "prob_profit": round(res.prob_profit, 4),
                    "prob_assignment": round(res.prob_assignment, 4),
                    "days_to_earnings": days_to_earn,
                    "days_to_ex_div": days_to_ex_div,
                    "distribution_source": method,
                    "sector": DEFAULT_SECTOR_MAP.get(ticker, "Unknown"),
                }
                if include_diagnostic_fields:
                    # Mirror the EVEngine dividend gate at
                    # engine/ev_engine.py:355-361: the early-exercise
                    # penalty fires only when option_type=="call"
                    # (always true here) AND days_to_ex_div <= dte AND
                    # expected_dividend > 0. When the gate would NOT
                    # fire (e.g. ex-div falls outside the holding
                    # window), the diagnostic column reads 0.0 — not
                    # the upstream amount — so a trader inspecting the
                    # row doesn't conclude the engine is factoring an
                    # ex-div it isn't. Closes S28 Fix #1.
                    applied_dividend = (
                        expected_dividend
                        if (
                            days_to_ex_div is not None
                            and days_to_ex_div <= int(new_dte)
                            and expected_dividend > 0
                        )
                        else 0.0
                    )
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
                            "expected_dividend": round(applied_dividend, 4),
                            "regime_multiplier": round(res.regime_multiplier, 4),
                        }
                    )
                rows.append(row)

        if not rows:
            return _empty()
        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("ev_per_day", ascending=False).head(top_n).reset_index(drop=True)
        # Drop log + S31 F1/F4 summary attached after sort/head so it
        # rides on the exact frame returned; survivor rows are
        # untouched (CLAUDE.md §2).
        return _attach_drops_summary(df, drops)

    # ------------------------------------------------------------------
    # Strangle EV ranking (issue #118 P1 — S14 follow-up)
    # ------------------------------------------------------------------
    def rank_strangles_by_ev(
        self,
        ticker: str,
        contracts: int = 1,
        *,
        target_dtes: tuple[int, ...] = (21, 35, 49, 63),
        target_deltas: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15),
        as_of: str | None = None,
        min_ev_dollars: float = 0.0,
        top_n: int = 20,
        include_diagnostic_fields: bool = True,
        use_event_gate: bool = True,
        use_timing_gate: bool = True,
        earnings_buffer_days: int = 5,
        macro_buffer_days: int = 1,
        min_history_days: int = 504,
        enforce_history_gate: bool = True,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        max_as_of_staleness_days: int = 30,
    ) -> pd.DataFrame:
        """Rank short-strangle candidates for a ticker by composed forward EV.

        S14 found the strangle path (:mod:`engine.strangle_timing`)
        produces only a *timing score* — it never yields a tradeable
        candidate (strikes + premium) and never touches
        :class:`~engine.ev_engine.EVEngine`. CLAUDE.md §2 held for
        strangles only because that path never produced a tradeable
        candidate. Issue #118 P1. This method closes that gap: it
        enumerates a ``(DTE × delta)`` grid of short strangles (short OTM
        put + short OTM call) and EV-ranks them, bringing §4's
        timing-gated strategy under the EV authority.

        **A strangle is two short legs.** Each candidate is scored as two
        independent :meth:`EVEngine.evaluate` calls — the put leg as a
        ``ShortOptionTrade(option_type="put")`` and the call leg as a
        ``ShortOptionTrade(option_type="call")`` — both over the *same*
        empirical ``forward_log_returns`` (the same underlying path).
        There is no blended-strangle side channel and no ``StrangleTrade``
        shortcut: the EV authority sees two ordinary short options.

        **Composed EV is additive.** ``ev_dollars`` is
        ``put.ev_dollars + call.ev_dollars``. Expected value is linear,
        so this sum is exact *regardless of how the legs co-move* — it
        needs no joint distribution. This is the headline metric and the
        ranking key.

        **Risk metrics are NOT additive.** The strangle's payoff is
        nonlinear in the shared underlying path (it loses on a large move
        in *either* direction), so summing the two legs' ``cvar_5`` /
        ``prob_profit`` / ``prob_assignment`` would be wrong. They are
        reported per-leg, explicitly labelled ``put_*`` / ``call_*``,
        each a real :class:`~engine.ev_engine.EVResult` field — never a
        fabricated sum. A *joint* combined-risk metric (per-path: did the
        underlying finish between the two strikes) would need per-path
        P&L, which :class:`EVResult` does not expose; it is a documented
        follow-up. ``lower_breakeven`` / ``upper_breakeven`` *are*
        surfaced as a combined view — those are exact contract algebra
        (``put_strike − credit`` / ``call_strike + credit``), not path
        statistics, so there is no fabrication risk.

        **§2 — the floor is on the composed EV.** ``min_ev_dollars``
        (default ``0.0``) drops any candidate whose
        ``put.ev_dollars + call.ev_dollars`` is below it. Neither leg's
        EV in isolation admits a candidate: a +$520 put leg paired with a
        −$540 call leg is a −$20 strangle and is dropped. It **ranks,
        never rescues** — a negative composed-EV strangle never surfaces
        as tradeable.

        **The timing gate (§4) is downgrade-only.** When
        ``use_timing_gate`` is set, :class:`StrangleTimingEngine` scores
        the ticker; an ``avoid`` recommendation drops the whole ticker
        (logged in ``.attrs["drops"]`` with gate ``timing``) before any
        EV ranking. The timing score is a pure pre-filter — it can only
        *remove* a ticker, never lift a candidate's EV or rescue a
        negative composed-EV strangle. ``timing_score`` /
        ``timing_recommendation`` / ``timing_phase`` are surfaced for
        transparency but feed nothing into ``ev_dollars``.

        Args:
            ticker: The underlying to scan.
            contracts: Contracts per leg (the put and call are sized
                equally — a 1-contract strangle is one short put + one
                short call).
            target_dtes: Candidate days-to-expiry (both legs share an
                expiry).
            target_deltas: Candidate deltas. Each is applied
                *symmetrically* — a 0.25 candidate is a 25-delta put plus
                a 25-delta call. One ``(DTE, delta)`` pair = one strangle.
            as_of: PIT cutoff date ``YYYY-MM-DD``. ``None`` means now.
            min_ev_dollars: Hard floor on the **composed** EV.
            top_n: Number of top candidates to return.
            include_diagnostic_fields: Append the per-leg risk block.
            use_event_gate: Hard-block candidates whose holding window
                touches the ticker's next earnings (downgrade-only).
            use_timing_gate: Apply the §4 strangle-timing pre-filter
                (downgrade-only — an ``avoid`` verdict drops the ticker).
            earnings_buffer_days, macro_buffer_days: Event-gate buffers.
            min_history_days, enforce_history_gate: Survivorship/quality
                gate on OHLCV length.
            risk_free_rate: Decimal rate. ``None`` resolves from the
                connector; an explicit value outside ``[0, 0.25]`` raises.
            dividend_yield: Decimal annual yield. ``None`` resolves from
                the connector's fundamentals.

        Returns:
            DataFrame sorted by composed ``ev_dollars`` descending — one
            row per surviving ``(DTE, delta)`` strangle — with the
            columns of ``_STRANGLE_RANK_CORE_COLUMNS`` (+
            ``_STRANGLE_RANK_DIAGNOSTIC_COLUMNS`` when
            ``include_diagnostic_fields``). Empty but correctly shaped
            when nothing survives.

            ``.attrs["drops"]`` carries one ``{"ticker", "gate",
            "reason"}`` dict per gated-out candidate — ``gate`` is one of
            ``data``, ``history``, ``timing``, ``strike``, ``premium``,
            ``event`` or ``ev_threshold``.
        """
        from datetime import timedelta

        from engine.ev_engine import EVEngine, ShortOptionTrade
        from engine.event_gate import EventGate, ScheduledEvent
        from engine.forward_distribution import best_available_forward_distribution
        from engine.option_pricer import black_scholes_price
        from engine.wheel_tracker import _solve_call_strike, _solve_put_strike

        if risk_free_rate is not None and not (0.0 <= risk_free_rate <= 0.25):
            raise ValueError(f"risk_free_rate {risk_free_rate} outside [0, 0.25]")
        contracts = max(int(contracts), 1)

        conn = self.connector
        drops: list[dict] = []

        cols = list(_STRANGLE_RANK_CORE_COLUMNS)
        if include_diagnostic_fields:
            cols = cols + _STRANGLE_RANK_DIAGNOSTIC_COLUMNS

        def _empty() -> pd.DataFrame:
            df = pd.DataFrame(columns=cols)
            return _attach_drops_summary(df, drops)

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

        # S33 F3 follow-up: mirror the as_of-beyond-data gate from
        # rank_candidates_by_ev (PR #215). Without it the strangle
        # ranker silently substituted the latest available close as
        # "current spot" for any future as_of — same D11 violation,
        # same fix shape.
        if as_of is not None:
            try:
                cutoff = pd.Timestamp(as_of)
                actual_last = ohlcv.index.max()
                gap_days = (cutoff - actual_last).days
                if gap_days > max_as_of_staleness_days:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "data",
                            "reason": (
                                f"as_of {as_of} is {gap_days} days beyond "
                                f"latest data ({actual_last.date().isoformat()}); "
                                f"max_as_of_staleness_days={max_as_of_staleness_days}"
                            ),
                        }
                    )
                    return _empty()
            except Exception:
                pass

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

        # ---- IV: PIT-first via get_iv_history, fallback to fundamentals snapshot ----
        # S23 F3 fix: same as rank_candidates_by_ev.
        fundamentals = conn.get_fundamentals(ticker) or {}
        iv = _resolve_pit_atm_iv(conn, ticker, as_of)
        if iv is None:
            iv_raw = fundamentals.get("implied_vol_atm")
            if iv_raw is None or (isinstance(iv_raw, float) and np.isnan(iv_raw)):
                iv_raw = fundamentals.get("volatility_30d")
            try:
                iv = float(iv_raw) if iv_raw is not None else 0.0
            except (TypeError, ValueError):
                iv = 0.0
            if np.isnan(iv) or iv <= 0:
                drops.append(
                    {"ticker": ticker, "gate": "data", "reason": "IV missing or non-positive"}
                )
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
                div_q /= 100.0  # Bloomberg eqy_dvd_yld_12m is percent (AUDIT-IX)
                if div_q > 0.30:
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

        # ---- §4 strangle-timing pre-filter (downgrade-only) ----
        # Always computed (the timing_* columns are informational); only
        # *gates* when use_timing_gate is set. An 'avoid' verdict drops
        # the whole ticker before any EV ranking — it can never lift EV.
        timing_score: float | None = None
        timing_recommendation = "unknown"
        timing_phase = "unknown"
        try:
            from engine.strangle_timing import StrangleTimingEngine

            timing = StrangleTimingEngine().score_entry(ohlcv)
            timing_score = round(float(timing.total_score), 2)
            timing_recommendation = str(timing.recommendation)
            if timing.regime is not None:
                timing_phase = str(timing.regime.phase.value)
        except Exception:
            timing_score = None
            timing_recommendation = "unknown"
            timing_phase = "unknown"
        if use_timing_gate and timing_recommendation == "avoid":
            drops.append(
                {
                    "ticker": ticker,
                    "gate": "timing",
                    "reason": f"strangle timing recommendation 'avoid' (score {timing_score})",
                }
            )
            return _empty()

        # ---- event gate: register the ticker's next earnings ----
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
            # S23 F1 — symmetric back-buffer (matches rank_candidates_by_ev).
            # Pull the most recent past earnings within the back-buffer
            # so the gate can fire on a trade opened immediately
            # post-earnings (IV-crush window). Defensive hasattr() for
            # legacy connectors.
            if event_gate is not None and hasattr(conn, "get_recent_earnings"):
                recent_earn = conn.get_recent_earnings(
                    ticker, as_of, lookback_days=earnings_buffer_days
                )
                if recent_earn:
                    r_ts = recent_earn.get("announcement_date")
                    if r_ts is not None:
                        r_d = r_ts.date() if hasattr(r_ts, "date") else r_ts
                        event_gate.add_event(
                            ScheduledEvent(ticker=ticker, kind="earnings", event_date=r_d)
                        )
        except Exception:
            days_to_earn = None

        ev_eng = EVEngine(event_gate=event_gate)

        # ---- forward-distribution cache (one fetch per distinct DTE) ----
        # Both legs of a candidate share the SAME forward path — that is
        # what makes the composed EV an honest sum over one distribution.
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

        # ---- enumerate the (DTE × delta) strangle grid ----
        rows: list[dict] = []
        for new_dte in target_dtes:
            if new_dte <= 0:
                continue
            T = max(new_dte, 1) / 365.0
            expiry = today_date + timedelta(days=int(new_dte))
            fwd_rets, method = _fwd_for(int(new_dte))
            for tgt_delta in target_deltas:
                # Put leg — OTM, strike below spot.
                put_strike_raw = _solve_put_strike(
                    spot=spot, T=T, r=rf, q=div_q, iv=iv, target_delta=tgt_delta
                )
                if put_strike_raw is None:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"put delta-to-strike solve did not converge "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                put_strike = round(put_strike_raw * 2) / 2
                if put_strike <= 0 or put_strike >= spot:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"put strike {put_strike} not OTM vs spot {spot:.2f} "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                # Call leg — OTM, strike above spot.
                call_strike_raw = _solve_call_strike(
                    spot=spot, T=T, r=rf, q=div_q, iv=iv, target_delta=tgt_delta
                )
                if call_strike_raw is None:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"call delta-to-strike solve did not converge "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                call_strike = round(call_strike_raw * 2) / 2
                if call_strike <= spot:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "strike",
                            "reason": (
                                f"call strike {call_strike} not OTM vs spot {spot:.2f} "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue

                put_premium = black_scholes_price(
                    S=spot, K=put_strike, T=T, r=rf, sigma=iv, option_type="put", q=div_q
                )
                if put_premium <= 0.05:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "premium",
                            "reason": (
                                f"put premium too thin (<=$0.05) (dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue
                call_premium = black_scholes_price(
                    S=spot, K=call_strike, T=T, r=rf, sigma=iv, option_type="call", q=div_q
                )
                if call_premium <= 0.05:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "premium",
                            "reason": (
                                f"call premium too thin (<=$0.05) "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue

                # Two short legs → two EVEngine.evaluate calls, same path.
                put_trade = ShortOptionTrade(
                    option_type="put",
                    underlying=ticker,
                    spot=spot,
                    strike=float(put_strike),
                    premium=put_premium,
                    dte=int(new_dte),
                    iv=iv,
                    risk_free_rate=rf,
                    dividend_yield=div_q,
                    contracts=contracts,
                    bid=put_premium * 0.95,
                    ask=put_premium * 1.05,
                    open_interest=1000,
                    regime_multiplier=1.0,
                )
                call_trade = ShortOptionTrade(
                    option_type="call",
                    underlying=ticker,
                    spot=spot,
                    strike=float(call_strike),
                    premium=call_premium,
                    dte=int(new_dte),
                    iv=iv,
                    risk_free_rate=rf,
                    dividend_yield=div_q,
                    contracts=contracts,
                    bid=call_premium * 0.95,
                    ask=call_premium * 1.05,
                    open_interest=1000,
                    regime_multiplier=1.0,
                )
                put_res = ev_eng.evaluate(
                    put_trade,
                    forward_log_returns=fwd_rets,
                    trade_start=today_date,
                    trade_end=expiry,
                )
                call_res = ev_eng.evaluate(
                    call_trade,
                    forward_log_returns=fwd_rets,
                    trade_start=today_date,
                    trade_end=expiry,
                )
                # Event-gate short-circuit: either leg blocked → drop.
                if put_res.event_lockout_reason or call_res.event_lockout_reason:
                    reason = put_res.event_lockout_reason or call_res.event_lockout_reason
                    drops.append({"ticker": ticker, "gate": "event", "reason": str(reason)})
                    continue

                # Composed strangle EV — additive by linearity of
                # expectation. The floor is on this composed value, so
                # neither leg in isolation can admit a candidate.
                composed_ev = put_res.ev_dollars + call_res.ev_dollars
                if composed_ev < min_ev_dollars:
                    drops.append(
                        {
                            "ticker": ticker,
                            "gate": "ev_threshold",
                            "reason": (
                                f"composed ev_dollars {composed_ev:.2f} < "
                                f"min_ev_dollars {min_ev_dollars:.2f} "
                                f"(dte={new_dte}, delta={tgt_delta})"
                            ),
                        }
                    )
                    continue

                total_premium = put_premium + call_premium
                row: dict = {
                    "ticker": ticker,
                    "spot": round(spot, 2),
                    "dte": int(new_dte),
                    "expiry": expiry,
                    "target_delta": tgt_delta,
                    "iv": round(iv, 4),
                    "contracts": contracts,
                    "put_strike": put_strike,
                    "call_strike": call_strike,
                    "put_premium": round(put_premium, 3),
                    "call_premium": round(call_premium, 3),
                    "total_premium": round(total_premium, 3),
                    "ev_dollars": round(composed_ev, 2),
                    "put_ev_dollars": round(put_res.ev_dollars, 2),
                    "call_ev_dollars": round(call_res.ev_dollars, 2),
                    # Breakevens — exact contract algebra: the short
                    # strangle profits if spot finishes between these.
                    "lower_breakeven": round(put_strike - total_premium, 2),
                    "upper_breakeven": round(call_strike + total_premium, 2),
                    "days_to_earnings": days_to_earn,
                    "timing_score": timing_score,
                    "timing_recommendation": timing_recommendation,
                    "distribution_source": method,
                    "sector": DEFAULT_SECTOR_MAP.get(ticker, "Unknown"),
                }
                if include_diagnostic_fields:
                    # Per-leg risk — explicitly labelled, NOT summed (the
                    # strangle payoff is nonlinear in the shared path).
                    row.update(
                        {
                            "put_prob_profit": round(put_res.prob_profit, 4),
                            "call_prob_profit": round(call_res.prob_profit, 4),
                            "put_prob_assignment": round(put_res.prob_assignment, 4),
                            "call_prob_assignment": round(call_res.prob_assignment, 4),
                            "put_cvar_5": round(put_res.cvar_5, 2),
                            "call_cvar_5": round(call_res.cvar_5, 2),
                            "put_edge_vs_fair": round(put_res.edge_vs_fair, 2),
                            "call_edge_vs_fair": round(call_res.edge_vs_fair, 2),
                            # Transaction cost IS additive (a deterministic
                            # cost, not a path statistic) — both legs paid.
                            "total_transaction_cost": round(
                                put_res.total_transaction_cost + call_res.total_transaction_cost,
                                2,
                            ),
                            "timing_phase": timing_phase,
                        }
                    )
                rows.append(row)

        if not rows:
            return _empty()
        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("ev_dollars", ascending=False).head(top_n).reset_index(drop=True)
        # Drop log + S31 F1/F4 summary attached after sort/head so it
        # rides on the exact frame returned; survivor rows are
        # untouched (CLAUDE.md §2).
        return _attach_drops_summary(df, drops)

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
        portfolio_context: Any | None = None,
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
            portfolio_context: Optional
                :class:`engine.portfolio_risk_gates.PortfolioContext`
                threaded into :func:`engine.candidate_dossier.build_dossiers`
                and attached to every :class:`CandidateDossier` built
                in this pass. When set, the dossier reviewer's D17
                soft-warns fire live: **R7** (portfolio VaR_95 >
                ``max_var_pct`` × NAV) and **R8** (stress drawdown
                > 8% NAV OR underlying in ``short_gamma_amplifying``
                regime) can downgrade a ``"proceed"`` to ``"review"``
                against the held book.

                When ``None`` (the default), R7 and R8 are no-ops —
                soft-warns do not fire on absent evidence (Q3 of the
                #154 C4 design checkpoint; matches D11's
                "no silent substitution" principle). Construct the
                context via
                :meth:`engine.wheel_tracker.WheelTracker.portfolio_context_snapshot`
                when you have a live tracker. The HTTP API
                (:mod:`engine_api`) does **not** supply one — the
                endpoint is stateless and has no held-book reference,
                so R7/R8 remain dormant on ``/api/tv/dossier`` by
                design.

                Closes C3 from
                ``docs/END_TO_END_REVIEW_2026_05_25.md`` by exposing
                the parameter the underlying
                :func:`~engine.candidate_dossier.build_dossiers`
                already accepted.

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

        # Thread the PIT market-wide VIX *level* into the dossier reviewer so
        # R11 (elevated-vol top-bin size-down, heavy-verify I11) fires live.
        # Best-effort: any failure (no connector / no VIX data / future as_of)
        # degrades to None, which makes R11 a no-op — never blocks ranking.
        vix_level: float | None = None
        try:
            if self.connector is not None and hasattr(self.connector, "get_vix_regime"):
                _v = self.connector.get_vix_regime(as_of).get("vix")
                vix_level = float(_v) if _v is not None else None
        except Exception:  # noqa: BLE001 — VIX is advisory; never fail the rank
            vix_level = None

        return build_dossiers(
            ev_frame=ev_df,
            provider=provider,
            reviewer=chart_reviewer,
            timeframe=chart_timeframe,  # type: ignore[arg-type]
            top_n=top_n,
            portfolio_context=portfolio_context,
            vix_level=vix_level,
        )

    # ------------------------------------------------------------------
    # Production wire: rank → consume_ranker_row over the top-N
    # ------------------------------------------------------------------
    def consume_into_tracker(
        self,
        tracker: "WheelTracker",
        entry_date: date,
        *,
        rank_kwargs: dict[str, Any] | None = None,
        top_n_to_consume: int | None = None,
        expiration_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """End-to-end production wire: rank → ``consume_ranker_row``.

        Loops :meth:`engine.wheel_tracker.WheelTracker.consume_ranker_row`
        over the top-N rows of :meth:`rank_candidates_by_ev`, capturing
        per-row outcomes. The wire is:

        ``rank_candidates_by_ev`` →
        :meth:`~engine.wheel_tracker.WheelTracker.issue_ev_authority_token`
        (D16 launch gate refuses non-positive ``ev_dollars``) →
        :meth:`~engine.wheel_tracker.WheelTracker.open_short_put`
        with ``current_ev_dollars`` (D16 fresh-EV stale-check + D17
        portfolio-risk hard-blocks when ``require_ev_authority=True``).

        Refusals at any stage (D16 ``EVAuthorityRefused``, D17
        sector-cap / portfolio-delta / Kelly hard-block, stale-EV,
        duplicate position, insufficient cash, NAV exhaustion) are
        caught and recorded rather than raised — the helper is
        loop-safe so a single bad row does not abort the campaign.
        Callers inspect the returned outcomes to know what fired and
        what refused.

        Closes C4 from ``docs/END_TO_END_REVIEW_2026_05_25.md`` and
        TERMINAL_A_AUDIT.md cross-cutting #4: until this method
        landed, the rank-to-tracker chain was the operator's
        responsibility to wire row-by-row. D16 / D17 hardening was a
        contract for *direct* tracker callers (tests today), not the
        ranker chain operators run.

        Args:
            tracker: The :class:`~engine.wheel_tracker.WheelTracker`
                positions land in. For production use this should be
                constructed with ``require_ev_authority=True`` so the
                D16 token + D17 hard-blocks gate the fire path; the
                helper itself does not enforce that — it surfaces
                refusals if they happen but does not require the
                caller to use strict mode.
            entry_date: Trade entry date passed through to
                :meth:`~engine.wheel_tracker.WheelTracker.consume_ranker_row`.
            rank_kwargs: Keyword arguments forwarded to
                :meth:`rank_candidates_by_ev`. Use to control the
                ranker (``dte_target``, ``delta_target``, ``as_of``,
                ``universe_limit``, ``tickers`` subset, etc.).
                ``top_n`` is overridden — see ``top_n_to_consume``
                below. ``include_diagnostic_fields`` is forced True
                because the token-hash canonicalisation reads
                ``distribution_source``.
            top_n_to_consume: How many of the ranker's top rows to
                feed into ``consume_ranker_row``. Defaults to the
                ranker's own ``top_n`` (default 20). Bounded by the
                ranker's actual output size.
            expiration_date: Optional explicit expiration; passed
                through to ``consume_ranker_row``. When ``None``,
                ``consume_ranker_row`` defaults to
                ``entry_date + row['dte']``.

        Returns:
            A list of per-row outcome dicts, in ranking order:

            ``{
                "ticker": str,
                "ev_dollars": float,
                "opened": bool,           # tracker accepted + position landed
                "refusal_reason": str | None,  # None on success
            }``

            Possible ``refusal_reason`` values:

            * ``"ev_authority_refused"`` — ``ev_dollars <= 0`` at
              token issuance (D16); should not happen with the
              default ``min_ev_dollars=0.0`` filter on the ranker,
              but is caught for callers who relax the floor.
            * ``"tracker_rejected"`` — :meth:`open_short_put`
              returned False (stale-EV, D17 hard-block, duplicate
              ticker, insufficient cash, or NAV exhaustion). Inspect
              ``tracker._ev_authority_log`` for the structured reason.
            * ``"unexpected_exception"`` — any other exception
              during the per-row consume. The exception string is
              appended to the reason for triage.

            Empty list when the ranker returns no rows.
        """
        from engine.wheel_tracker import EVAuthorityRefused

        rank_kwargs = dict(rank_kwargs or {})
        # Force the schema flag the token-hash needs.
        rank_kwargs["include_diagnostic_fields"] = True

        df = self.rank_candidates_by_ev(**rank_kwargs)
        if df is None or len(df) == 0:
            return []

        if top_n_to_consume is not None:
            df = df.head(int(top_n_to_consume))

        outcomes: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            ticker = str(row_dict.get("ticker", ""))
            ev_dollars = float(row_dict.get("ev_dollars", 0.0) or 0.0)
            outcome: dict[str, Any] = {
                "ticker": ticker,
                "ev_dollars": ev_dollars,
                "opened": False,
                "refusal_reason": None,
            }
            try:
                opened = tracker.consume_ranker_row(
                    row_dict, entry_date=entry_date, expiration_date=expiration_date
                )
            except EVAuthorityRefused:
                # D16 launch-gate refusal — ev_dollars <= 0 at issuance.
                # Token never issued; position not opened. Audit log on
                # the tracker records ``action="refuse_issue"``.
                outcome["refusal_reason"] = "ev_authority_refused"
            except Exception as exc:  # pragma: no cover — defensive
                outcome["refusal_reason"] = f"unexpected_exception:{exc!r}"
            else:
                outcome["opened"] = bool(opened)
                if not opened:
                    # ``consume_ranker_row`` returned False — the tracker
                    # logged the reason on ``_ev_authority_log``
                    # (e.g. stale_ev, sector_cap, portfolio_delta,
                    # kelly_blocked, duplicate ticker, insufficient cash).
                    outcome["refusal_reason"] = "tracker_rejected"
            outcomes.append(outcome)

        return outcomes

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
