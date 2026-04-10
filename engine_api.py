"""
Smart Wheel Engine API Server

HTTP API that serves engine data to the Next.js dashboard.
Run with: python engine_api.py

Endpoints:
  GET /api/status          - Engine health check
  GET /api/candidates      - Top wheel trade candidates
  GET /api/analyze/AAPL    - Full ticker analysis
  GET /api/portfolio       - Portfolio report for given tickers
  GET /api/regime          - Current market regime
  GET /api/calendar        - Upcoming events
  GET /api/committee       - Run investment committee on a trade
  GET /api/screen          - Screen universe with filters
"""

import json
import sys
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from engine.data_connector import MarketDataConnector
from engine.wheel_runner import WheelRunner


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


# Global engine instance (lazy-loaded)
_runner = None
_connector = None


def get_runner():
    global _runner
    if _runner is None:
        _runner = WheelRunner()
    return _runner


def get_connector():
    global _connector
    if _connector is None:
        _connector = MarketDataConnector()
    return _connector


class EngineAPIHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the engine API."""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, cls=NumpyEncoder).encode())

    def _send_error(self, message, status=500):
        self._send_json({"error": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        # Helper to get single param
        def param(key, default=None):
            return params.get(key, [default])[0]

        try:
            if path == "/api/status":
                self._handle_status()
            elif path == "/api/candidates":
                self._handle_candidates(param("limit", "15"), param("min_score", "50"))
            elif path.startswith("/api/analyze/"):
                ticker = path.split("/")[-1].upper()
                self._handle_analyze(ticker, param("as_of"))
            elif path == "/api/portfolio":
                tickers = param("tickers", "AAPL,MSFT,JPM")
                self._handle_portfolio(tickers.split(","), param("as_of"))
            elif path == "/api/regime":
                self._handle_regime(param("ticker", "SPY"))
            elif path == "/api/calendar":
                self._handle_calendar(param("ticker"), param("days", "30"))
            elif path == "/api/screen":
                self._handle_screen(params)
            elif path == "/api/committee":
                self._handle_committee(param("ticker", "AAPL"))
            elif path == "/api/vix":
                self._handle_vix()
            elif path == "/api/fundamentals":
                ticker = param("ticker", "AAPL")
                self._handle_fundamentals(ticker)
            elif path == "/api/universe":
                self._handle_universe()
            elif path.startswith("/api/chart/"):
                chart_type = path.split("/")[-1]
                ticker = param("ticker", "AAPL")
                days = param("days", "120")
                self._handle_chart(chart_type, ticker, int(days))
            elif path == "/api/strangle":
                ticker = param("ticker", "AAPL")
                self._handle_strangle(ticker)
            elif path == "/api/iv_history":
                ticker = param("ticker", "AAPL")
                days = param("days", "252")
                self._handle_iv_history(ticker, int(days))
            else:
                self._send_error(f"Unknown endpoint: {path}", 404)
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    def _handle_status(self):
        conn = get_connector()
        universe = conn.get_universe()
        vix = conn.get_vix_regime()
        self._send_json(
            {
                "status": "connected",
                "engine": "smart-wheel-engine",
                "version": "2.0",
                "universe_size": len(universe),
                "vix": vix.get("vix", 0) if vix else 0,
                "data_dir": str(conn._data_dir),
            }
        )

    def _handle_candidates(self, limit, min_score):
        runner = get_runner()
        df = runner.screen_candidates(
            min_wheel_score=float(min_score),
            top_n=int(limit),
        )
        if df.empty:
            self._send_json({"trades": [], "count": 0})
            return

        trades = []
        for _, row in df.iterrows():
            trades.append(
                {
                    "ticker": row["ticker"],
                    "strategy": "short_put",
                    "strike": 0,  # Would need chain data
                    "expiration": "",
                    "premium": 0,
                    "probability": 0,
                    "expectedPnL": 0,
                    "maxLoss": 0,
                    "iv": row.get("iv_30d", 0),
                    "delta": 0,
                    "score": row["wheel_score"],
                    "wheelScore": row["wheel_score"],
                    "recommendation": row["recommendation"],
                    "strangleScore": row.get("strangle_score", 0),
                    "stranglePhase": row.get("strangle_phase", ""),
                    "sector": row.get("sector", ""),
                    "peRatio": row.get("pe_ratio", 0),
                    "beta": row.get("beta", 0),
                    "divYield": row.get("div_yield", 0),
                    "volPremium": row.get("vol_premium", 0),
                    "ivRank": row.get("iv_rank", 0),
                    "mktCapB": row.get("mkt_cap_B", 0),
                    "daysToEarnings": row.get("days_to_earnings"),
                    "creditRating": row.get("credit_rating", ""),
                }
            )

        self._send_json({"trades": trades, "count": len(trades)})

    def _handle_analyze(self, ticker, as_of):
        runner = get_runner()
        a = runner.analyze_ticker(ticker, as_of)
        self._send_json(
            {
                "ticker": a.ticker,
                "spotPrice": a.spot_price,
                "marketCap": a.market_cap,
                "peRatio": a.pe_ratio,
                "beta": a.beta,
                "dividendYield": a.dividend_yield,
                "sector": a.sector,
                "creditRating": a.credit_rating,
                "iv30d": a.iv_30d,
                "rv30d": a.rv_30d,
                "ivRank": a.iv_rank,
                "ivPercentile": a.iv_percentile,
                "volRiskPremium": a.vol_risk_premium,
                "daysToEarnings": a.days_to_earnings,
                "daysToExDiv": a.days_to_ex_div,
                "nextEarningsDate": a.next_earnings_date,
                "nextDivDate": a.next_div_date,
                "nextDivAmount": a.next_div_amount,
                "strangleScore": a.strangle_score,
                "stranglePhase": a.strangle_phase,
                "strangleRecommendation": a.strangle_recommendation,
                "riskFreeRate": a.risk_free_rate,
                "vixLevel": a.vix_level,
                "wheelScore": a.wheel_score,
                "wheelRecommendation": a.wheel_recommendation,
            }
        )

    def _handle_portfolio(self, tickers, as_of):
        runner = get_runner()
        report = runner.portfolio_report([t.strip().upper() for t in tickers], as_of)
        self._send_json(report)

    def _handle_regime(self, ticker):
        conn = get_connector()
        vix_data = conn.get_vix_regime()
        conn.get_iv_history(ticker)

        regime = "NEUTRAL"
        vix = 0
        if vix_data:
            vix = vix_data.get("vix", 0)
            if vix > 30:
                regime = "HIGH_VOL"
            elif vix > 20:
                regime = "ELEVATED"
            elif vix < 15:
                regime = "LOW_VOL"

        self._send_json(
            {
                "regime": regime,
                "vix": vix,
                "vixPercentile": vix_data.get("vix_percentile", 0) if vix_data else 0,
                "contango": vix_data.get("contango", False) if vix_data else False,
                "trendScore": 0,
                "confidence": 0.7,
            }
        )

    def _handle_calendar(self, ticker, days):
        from engine.data_integration import (
            load_earnings_from_bloomberg,
        )
        from engine.event_calendar import EventCalendarBuilder

        events = []

        # Macro events
        builder = EventCalendarBuilder()
        year = date.today().year
        for evt in builder.generate_fomc_dates(year):
            days_until = (evt.event_date - date.today()).days
            if 0 <= days_until <= int(days):
                events.append(
                    {
                        "eventId": f"fomc_{evt.event_date}",
                        "eventType": "fomc",
                        "ticker": None,
                        "eventDate": evt.event_date.isoformat(),
                        "description": evt.description,
                        "daysUntil": days_until,
                    }
                )

        # Earnings (if ticker specified)
        if ticker:
            earnings = load_earnings_from_bloomberg(tickers=[ticker.upper()])
            for e in earnings:
                days_until = (e.event_date - date.today()).days
                if 0 <= days_until <= int(days):
                    events.append(
                        {
                            "eventId": f"earn_{e.symbol}_{e.event_date}",
                            "eventType": "earnings",
                            "ticker": e.symbol,
                            "eventDate": e.event_date.isoformat(),
                            "description": e.description,
                            "daysUntil": days_until,
                        }
                    )

        events.sort(key=lambda x: x["daysUntil"])
        self._send_json({"events": events})

    def _handle_screen(self, params):
        def p(key, default=None):
            return params.get(key, [default])[0]

        runner = get_runner()
        df = runner.screen_candidates(
            min_wheel_score=float(p("min_score", "50")),
            min_market_cap=float(p("min_mkt_cap", "5000000000")),
            max_beta=float(p("max_beta", "2.0")),
            top_n=int(p("limit", "20")),
        )

        results = []
        if not df.empty:
            results = df.to_dict(orient="records")

        self._send_json({"results": results, "count": len(results)})

    def _handle_committee(self, ticker):
        from advisors import CommitteeEngine, create_sample_input, format_committee_report

        committee = CommitteeEngine(parallel=False)
        sample = create_sample_input()
        # Override ticker
        sample.candidate_trade.ticker = ticker.upper()
        result = committee.evaluate(sample)

        advisor_summaries = []
        for r in result.advisor_responses:
            advisor_summaries.append(
                {
                    "name": r.advisor_name,
                    "judgment": r.judgment.value,
                    "summary": r.judgment_summary,
                    "keyReasons": r.key_reasons[:3],
                    "confidence": r.confidence.value,
                }
            )

        self._send_json(
            {
                "ticker": ticker,
                "judgment": result.committee_judgment.value,
                "reasoning": result.committee_reasoning,
                "confidence": result.committee_confidence.value,
                "approvals": result.approval_count,
                "rejections": result.rejection_count,
                "neutrals": result.neutral_count,
                "advisors": advisor_summaries,
                "risksUnresolved": result.unresolved_risks[:4],
                "requiredActions": result.required_before_trade[:4],
                "report": format_committee_report(result),
            }
        )

    def _handle_vix(self):
        conn = get_connector()
        vix = conn.get_vix_regime()
        self._send_json(vix or {"vix": 0})

    def _handle_fundamentals(self, ticker):
        conn = get_connector()
        fund = conn.get_fundamentals(ticker.upper())
        self._send_json(fund or {"error": "Not found"})

    def _handle_universe(self):
        conn = get_connector()
        universe = conn.get_universe()
        self._send_json({"tickers": universe, "count": len(universe)})

    def _handle_chart(self, chart_type, ticker, days):
        """Serve chart data: OHLCV + technical indicators as JSON arrays."""
        from src.features.technical import TechnicalFeatures

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)

        if ohlcv.empty:
            self._send_json({"error": f"No OHLCV data for {ticker}", "data": []})
            return

        tech = TechnicalFeatures()
        df = ohlcv.tail(days).copy()
        df["date_str"] = df.index.strftime("%Y-%m-%d")

        if chart_type == "bollinger":
            upper, middle, lower = tech.bollinger_bands(ohlcv["close"], 20, 2.0)
            data = []
            for idx in df.index:
                data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": round(float(df.loc[idx, "close"]), 2),
                        "open": round(float(df.loc[idx, "open"]), 2),
                        "high": round(float(df.loc[idx, "high"]), 2),
                        "low": round(float(df.loc[idx, "low"]), 2),
                        "upper": round(float(upper.loc[idx]), 2)
                        if idx in upper.index and not np.isnan(upper.loc[idx])
                        else None,
                        "middle": round(float(middle.loc[idx]), 2)
                        if idx in middle.index and not np.isnan(middle.loc[idx])
                        else None,
                        "lower": round(float(lower.loc[idx]), 2)
                        if idx in lower.index and not np.isnan(lower.loc[idx])
                        else None,
                        "volume": int(df.loc[idx, "volume"])
                        if not np.isnan(df.loc[idx, "volume"])
                        else 0,
                    }
                )
            bb_pos = tech.bollinger_position(ohlcv["close"])
            self._send_json(
                {
                    "ticker": ticker,
                    "chart_type": "bollinger",
                    "data": data,
                    "current_pct_b": round(float(bb_pos.iloc[-1]), 3)
                    if not np.isnan(bb_pos.iloc[-1])
                    else 0.5,
                }
            )

        elif chart_type == "rsi":
            rsi_14 = tech.rsi(ohlcv["close"], 14)
            rsi_2 = tech.rsi(ohlcv["close"], 2)
            data = []
            for idx in df.index:
                data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": round(float(df.loc[idx, "close"]), 2),
                        "rsi_14": round(float(rsi_14.loc[idx]), 1)
                        if idx in rsi_14.index and not np.isnan(rsi_14.loc[idx])
                        else None,
                        "rsi_2": round(float(rsi_2.loc[idx]), 1)
                        if idx in rsi_2.index and not np.isnan(rsi_2.loc[idx])
                        else None,
                    }
                )
            self._send_json(
                {
                    "ticker": ticker,
                    "chart_type": "rsi",
                    "data": data,
                    "current_rsi_14": round(float(rsi_14.iloc[-1]), 1)
                    if not np.isnan(rsi_14.iloc[-1])
                    else 50,
                }
            )

        elif chart_type == "atr":
            atr_14 = tech.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
            atr_pct = tech.atr_percent(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
            data = []
            for idx in df.index:
                data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": round(float(df.loc[idx, "close"]), 2),
                        "atr": round(float(atr_14.loc[idx]), 2)
                        if idx in atr_14.index and not np.isnan(atr_14.loc[idx])
                        else None,
                        "atr_pct": round(float(atr_pct.loc[idx]), 2)
                        if idx in atr_pct.index and not np.isnan(atr_pct.loc[idx])
                        else None,
                    }
                )
            self._send_json({"ticker": ticker, "chart_type": "atr", "data": data})

        elif chart_type == "ohlcv":
            data = []
            sma_20 = tech.sma(ohlcv["close"], 20)
            sma_50 = tech.sma(ohlcv["close"], 50)
            for idx in df.index:
                data.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": round(float(df.loc[idx, "open"]), 2),
                        "high": round(float(df.loc[idx, "high"]), 2),
                        "low": round(float(df.loc[idx, "low"]), 2),
                        "close": round(float(df.loc[idx, "close"]), 2),
                        "volume": int(df.loc[idx, "volume"])
                        if not np.isnan(df.loc[idx, "volume"])
                        else 0,
                        "sma20": round(float(sma_20.loc[idx]), 2)
                        if idx in sma_20.index and not np.isnan(sma_20.loc[idx])
                        else None,
                        "sma50": round(float(sma_50.loc[idx]), 2)
                        if idx in sma_50.index and not np.isnan(sma_50.loc[idx])
                        else None,
                    }
                )
            self._send_json({"ticker": ticker, "chart_type": "ohlcv", "data": data})

        elif chart_type == "strangle":
            # Strangle timing score history
            from engine.strangle_timing import StrangleTimingEngine

            engine = StrangleTimingEngine()
            hist = engine.compute_historical_scores(ohlcv, lookback_required=100)
            data = []
            if not hist.empty:
                for _, row in hist.tail(days).iterrows():
                    data.append(
                        {
                            "date": str(row.get("date", "")),
                            "score": round(float(row["score"]), 1),
                            "phase": row.get("phase", ""),
                            "bb_score": round(float(row.get("bb_score", 0)), 1),
                            "atr_score": round(float(row.get("atr_score", 0)), 1),
                            "rsi_score": round(float(row.get("rsi_score", 0)), 1),
                            "trend_score": round(float(row.get("trend_score", 0)), 1),
                            "range_score": round(float(row.get("range_score", 0)), 1),
                        }
                    )
            self._send_json({"ticker": ticker, "chart_type": "strangle", "data": data})

        else:
            self._send_error(f"Unknown chart type: {chart_type}", 400)

    def _handle_strangle(self, ticker):
        """Full strangle timing analysis for a ticker."""
        from engine.strangle_timing import StrangleTimingEngine

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)
        if ohlcv.empty or len(ohlcv) < 100:
            self._send_json({"error": "Insufficient data", "ticker": ticker})
            return

        engine = StrangleTimingEngine()
        score = engine.score_entry(ohlcv)
        regime = engine.classify_regime(ohlcv)

        self._send_json(
            {
                "ticker": ticker,
                "score": round(float(score.total_score), 1),
                "recommendation": score.recommendation,
                "phase": regime.phase.value,
                "confidence": round(float(regime.confidence), 2),
                "components": {
                    "bollinger": {
                        "score": round(float(score.bollinger_score), 1),
                        "state": regime.bollinger_state,
                    },
                    "atr": {"score": round(float(score.atr_score), 1), "state": regime.atr_state},
                    "rsi": {
                        "score": round(float(score.rsi_score), 1),
                        "state": regime.rsi_state,
                        "value": round(float(regime.rsi_14), 1),
                    },
                    "trend": {
                        "score": round(float(score.trend_score), 1),
                        "state": regime.trend_state,
                    },
                    "range": {
                        "score": round(float(score.range_score), 1),
                        "state": regime.range_state,
                    },
                },
                "metrics": {
                    "bb_width_pctl": round(float(regime.bb_width_percentile), 1),
                    "bb_pct_b": round(float(regime.bb_pct_b), 3),
                    "atr_pctl": round(float(regime.atr_percentile), 1),
                    "atr_slope": round(float(regime.atr_slope), 4),
                    "rsi_14": round(float(regime.rsi_14), 1),
                    "rsi_2": round(float(regime.rsi_2), 1),
                    "ma_slope": round(float(regime.ma_slope_20), 4),
                },
                "warnings": {
                    "compression": score.compression_warning,
                    "expansion": score.expansion_active,
                    "strong_trend": score.strong_trend_warning,
                },
            }
        )

    def _handle_iv_history(self, ticker, days):
        """IV vs RV history for a ticker."""
        conn = get_connector()
        iv_df = conn.get_iv_history(ticker)
        if iv_df.empty:
            self._send_json({"error": "No IV data", "ticker": ticker, "data": []})
            return

        data = []
        for idx in iv_df.tail(days).index:
            row = iv_df.loc[idx]
            data.append(
                {
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "put_iv": round(float(row.get("hist_put_imp_vol", 0)), 2)
                    if not np.isnan(row.get("hist_put_imp_vol", 0))
                    else None,
                    "call_iv": round(float(row.get("hist_call_imp_vol", 0)), 2)
                    if not np.isnan(row.get("hist_call_imp_vol", 0))
                    else None,
                    "rv_30d": round(float(row.get("volatility_30d", 0)), 2)
                    if not np.isnan(row.get("volatility_30d", 0))
                    else None,
                    "rv_60d": round(float(row.get("volatility_60d", 0)), 2)
                    if not np.isnan(row.get("volatility_60d", 0))
                    else None,
                }
            )
        self._send_json({"ticker": ticker, "data": data})

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[Engine API] {args[0]}")


def main():
    port = 8787
    server = HTTPServer(("0.0.0.0", port), EngineAPIHandler)
    print(f"Smart Wheel Engine API running on http://localhost:{port}")
    print("Endpoints:")
    print("  GET /api/status")
    print("  GET /api/candidates?limit=15&min_score=50")
    print("  GET /api/analyze/AAPL")
    print("  GET /api/portfolio?tickers=AAPL,MSFT,JPM")
    print("  GET /api/regime")
    print("  GET /api/calendar?ticker=AAPL&days=30")
    print("  GET /api/screen?min_score=60&limit=20")
    print("  GET /api/committee?ticker=NVDA")
    print("  GET /api/vix")
    print("  GET /api/fundamentals?ticker=AAPL")
    print("  GET /api/universe")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
