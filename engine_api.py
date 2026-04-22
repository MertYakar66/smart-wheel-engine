"""
Smart Wheel Engine API Server

HTTP API that serves engine data to the Next.js dashboard.
Run with: python engine_api.py

Endpoints:
  GET /api/status                          - Engine health check
  GET /api/candidates?limit=15&min_score=50 - Top wheel trade candidates
  GET /api/analyze/AAPL                   - Full ticker analysis
  GET /api/portfolio?tickers=AAPL,MSFT    - Portfolio report
  GET /api/regime?ticker=SPY              - Current market regime
  GET /api/calendar?ticker=AAPL&days=30   - Upcoming events
  GET /api/screen?min_score=60&limit=20   - Screen universe with filters
  GET /api/committee?ticker=NVDA          - Run investment committee
  GET /api/vix                            - VIX regime
  GET /api/fundamentals?ticker=AAPL       - Fundamental data
  GET /api/universe                       - Universe of tracked tickers
  GET /api/chart/bollinger?ticker=AAPL    - Bollinger bands chart
  GET /api/chart/rsi?ticker=AAPL          - RSI chart
  GET /api/chart/atr?ticker=AAPL          - ATR chart
  GET /api/chart/ohlcv?ticker=AAPL        - OHLCV + SMA chart
  GET /api/chart/strangle?ticker=AAPL     - Strangle timing history
  GET /api/strangle?ticker=AAPL           - Current strangle timing analysis
  GET /api/iv_history?ticker=AAPL&days=252 - IV vs RV history
  GET /api/payoff?ticker=AAPL&strategy=csp - Payoff diagram
  GET /api/expected_move?ticker=AAPL&dte=45 - Expected move bands
  GET /api/strikes?ticker=AAPL&strategy=csp - Strike recommendations
  GET /api/memo?ticker=AAPL               - AI trade memo (72B model)
  GET /api/summary?ticker=AAPL            - Quick AI summary (32B model)
  GET /api/ollama_status                  - Check Ollama/model availability

TradingView bridge:
  GET  /api/tv/signal?ticker=AAPL         - Canonical TV-parity signal for ticker
  GET  /api/tv/scan?limit=25&zone=wheel_put - Screen universe for TV zones
  GET  /api/tv/enrich?ticker=AAPL&signal=wheel_put_zone - Enriched decision
  GET  /api/tv/alerts?limit=50            - Recent webhook alerts (ring buffer)
  GET  /api/tv/ranked?limit=20&dte=35&delta=0.25 - EV-ranked candidates (audit-II)
  GET  /api/tv/dossier?top_n=10&timeframe=1D - EV + TV screenshot dossier (Mode B)
  GET  /api/tv/dealer_positioning?ticker=AAPL&dte=35 - Dealer GEX / walls / regime (audit V)
  POST /api/tv/webhook                    - Ingest TradingView Pine alert (JSON)
"""

import hashlib
import hmac
import json
import sys
import time
import traceback
from collections import OrderedDict
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from engine.data_connector import MarketDataConnector
from engine.wheel_runner import WheelRunner


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types and NaN in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return [
                None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
                for x in obj.tolist()
            ]
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


# Global engine instance (lazy-loaded)
_runner = None
_connector = None

# In-memory ring buffer of ingested TradingView webhook alerts. The bridge is
# designed for a single-user private deployment, so we intentionally avoid
# adding a database dependency — the buffer is rebuilt on each server start
# and served back through /api/tv/alerts for the dashboard to display.
_TV_ALERT_LOG: list[dict] = []
_TV_ALERT_LOG_MAX = 200

# In-memory news story buffer. Populated by the /api/news/ingest endpoint
# which the orchestrator calls after running the news pipeline. Stories are
# served back through /api/news for the dashboard and committee.
_NEWS_BUFFER: list[dict] = []
_NEWS_BUFFER_MAX = 100

# AUDIT: replay-protection nonce cache. We key by the SHA-256 of the raw body
# AND the incoming signature header so identical payloads with different
# HMACs — e.g. an attacker flipping fields — cannot collide. The cache is
# bounded (LRU) so we don't accumulate memory on long runs. Any alert seen
# within _TV_WEBHOOK_MAX_AGE_SEC that matches a cached digest is rejected.
_TV_WEBHOOK_MAX_AGE_SEC = 300  # 5 minutes
_TV_SEEN_NONCES: "OrderedDict[str, float]" = OrderedDict()
_TV_SEEN_NONCES_MAX = 1024


def _tv_seen_register(digest: str, now: float) -> bool:
    """Check-and-set whether ``digest`` has been seen in the last window.

    Returns True if this is a *new* nonce (proceed). Returns False if we have
    already processed a payload with this digest, which means we should reject
    it as a replay attempt.
    """
    # Purge stale entries at the head of the LRU
    cutoff = now - _TV_WEBHOOK_MAX_AGE_SEC
    while _TV_SEEN_NONCES and next(iter(_TV_SEEN_NONCES.values())) < cutoff:
        _TV_SEEN_NONCES.popitem(last=False)

    if digest in _TV_SEEN_NONCES:
        return False

    _TV_SEEN_NONCES[digest] = now
    while len(_TV_SEEN_NONCES) > _TV_SEEN_NONCES_MAX:
        _TV_SEEN_NONCES.popitem(last=False)
    return True


def _tv_verify_hmac(body_bytes: bytes, provided_sig: str, secret: str) -> bool:
    """Constant-time HMAC-SHA256 verification.

    ``provided_sig`` may be of the form ``sha256=<hex>`` or a bare hex digest
    to stay compatible with arbitrary webhook intermediaries. We never raise
    on decoding errors — any parse failure returns False.
    """
    if not secret or not provided_sig:
        return False
    expected = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    # Accept either "sha256=<hex>" or bare hex
    got = provided_sig.strip()
    if got.lower().startswith("sha256="):
        got = got.split("=", 1)[1]
    return hmac.compare_digest(expected, got)


def get_runner():
    global _runner
    if _runner is None:
        _runner = WheelRunner()
    return _runner


def get_connector():
    global _connector
    if _connector is None:
        import os
        provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()
        if provider == "theta":
            from engine.theta_connector import ThetaConnector
            _connector = ThetaConnector()
            logger.info("Data provider: ThetaData v3 (live)")
        else:
            _connector = MarketDataConnector()
            logger.info("Data provider: Bloomberg CSV")
    return _connector


def _sanitize_nans(obj):
    """Recursively replace NaN/Inf with None in nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


class EngineAPIHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the engine API."""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(_sanitize_nans(data), cls=NumpyEncoder).encode())

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
                self._handle_candidates(
                    limit=param("limit", "15"),
                    min_score=param("min_score", "50"),
                    dte=param("dte", "35"),
                    delta=param("delta", "0.25"),
                    min_ev=param("min_ev", "0"),
                    as_of=param("as_of"),
                )
            elif path == "/api/analyze" or path.startswith("/api/analyze/"):
                ticker = path.split("/")[-1].upper() if "/" in path[len("/api/analyze") :] else ""
                if not ticker:
                    ticker = param("ticker", "AAPL") or "AAPL"
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
                self._handle_committee(param("ticker"))
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
            elif path == "/api/payoff":
                ticker = param("ticker", "AAPL")
                self._handle_payoff(
                    ticker,
                    param("strategy", "csp"),
                    param("strike"),
                    param("premium"),
                    param("dte", "45"),
                )
            elif path == "/api/expected_move":
                ticker = param("ticker", "AAPL")
                self._handle_expected_move(ticker, param("dte", "45"))
            elif path == "/api/strikes":
                ticker = param("ticker", "AAPL")
                self._handle_strikes(ticker, param("strategy", "csp"), param("dte", "45"))
            elif path == "/api/strangle":
                ticker = param("ticker", "AAPL")
                self._handle_strangle(ticker)
            elif path == "/api/iv_history":
                ticker = param("ticker", "AAPL")
                days = param("days", "252")
                self._handle_iv_history(ticker, int(days))
            elif path == "/api/memo":
                ticker = param("ticker", "AAPL")
                self._handle_memo(ticker, param("as_of"))
            elif path == "/api/summary":
                ticker = param("ticker", "AAPL")
                self._handle_summary(ticker)
            elif path == "/api/ollama_status":
                self._handle_ollama_status()
            elif path == "/api/tv/signal":
                ticker = (param("ticker", "") or "").upper()
                self._handle_tv_signal(ticker, param("as_of"))
            elif path == "/api/tv/scan":
                self._handle_tv_scan(
                    param("limit", "25"),
                    param("phase"),
                    param("zone"),
                )
            elif path == "/api/tv/enrich":
                ticker = (param("ticker", "") or "").upper()
                self._handle_tv_enrich(
                    ticker,
                    param("signal", "wheel_put_zone"),
                    param("as_of"),
                )
            elif path == "/api/tv/alerts":
                self._handle_tv_alerts(int(param("limit", "50")))
            elif path == "/api/tv/ranked":
                self._handle_tv_ranked(
                    limit=int(param("limit", "20")),
                    dte_target=int(param("dte", "35")),
                    delta_target=float(param("delta", "0.25")),
                    min_ev_dollars=float(param("min_ev", "0")),
                    as_of=param("as_of"),
                    tickers_csv=param("tickers"),
                )
            elif path == "/api/tv/dossier":
                self._handle_tv_dossier(
                    top_n=int(param("top_n", "10")),
                    dte_target=int(param("dte", "35")),
                    delta_target=float(param("delta", "0.25")),
                    min_ev_dollars=float(param("min_ev", "0")),
                    as_of=param("as_of"),
                    tickers_csv=param("tickers"),
                    timeframe=param("timeframe", "1D") or "1D",
                    screenshots_dir=param("screenshots_dir", "screenshots") or "screenshots",
                )
            elif path == "/api/tv/dealer_positioning":
                self._handle_tv_dealer_positioning(
                    ticker=(param("ticker", "") or "").upper(),
                    dte_target=int(param("dte", "35")),
                    assumption=param("assumption", "long_calls_short_puts") or "long_calls_short_puts",
                )
            elif path == "/api/news":
                self._handle_news(limit=int(param("limit", "20")))
            else:
                self._send_error(f"Unknown endpoint: {path}", 404)
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    def do_POST(self):
        """Handle POST requests — currently only TradingView webhook alerts."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            # Hard cap body size to prevent memory-exhaustion DoS. TradingView
            # alert bodies are <1 KB in practice; 16 KB is generous.
            if length > 16 * 1024:
                self._send_error("payload too large", 413)
                return
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError as exc:
                self._send_error(f"Invalid JSON: {exc}", 400)
                return
            if not isinstance(payload, dict):
                self._send_error("payload must be a JSON object", 400)
                return

            if path == "/api/tv/webhook":
                # Pass raw body + signature header through to the webhook so
                # HMAC verification operates on the exact bytes TradingView
                # signed (JSON re-serialization would change whitespace).
                sig_header = (
                    self.headers.get("X-Signature")
                    or self.headers.get("X-Signature-256")
                    or self.headers.get("X-Hub-Signature-256")
                    or ""
                )
                self._handle_tv_webhook(payload, raw_body=raw, signature_header=sig_header)
            elif path == "/api/news/ingest":
                self._handle_news_ingest(payload)
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

    def _handle_candidates(
        self,
        limit="15",
        min_score="50",
        dte="35",
        delta="0.25",
        min_ev="0",
        as_of=None,
    ):
        """EV-authoritative candidates endpoint (audit-V P0.1).

        Replaces the legacy heuristic path that called
        ``runner.screen_candidates`` and rebuilt BSM math inside the
        API layer. The new handler is the single authoritative entry
        point for tradeable candidates — every row carries native EV
        diagnostics (ev_dollars, ev_per_day, prob_profit,
        prob_assignment, cvar_5, distribution_source, dealer_regime)
        and the legacy dashboard field names (expectedPnL, wheelScore,
        etc.) are mapped from those EV values so existing UI code
        keeps working.

        Query parameters:
            limit      top-N to return (default 15)
            min_score  backward-compat heuristic filter — post-applied
                       if the underlying frame carries wheel_score; does
                       not gate EV ranking (default 50)
            dte        target days-to-expiry (default 35)
            delta      target put delta (default 0.25)
            min_ev     hard EV threshold in dollars (default 0)
            as_of      optional PIT cutoff YYYY-MM-DD

        Returns:
            {trades: [...], count: N, authority: "ev_ranked",
             engine_version: "ev_engine_2026_04_14"}
        """
        try:
            limit_int = max(1, int(limit or 15))
            dte_int = int(dte or 35)
            delta_f = float(delta or 0.25)
            min_ev_f = float(min_ev or 0)
            min_score_f = float(min_score or 0)
        except (TypeError, ValueError):
            limit_int, dte_int, delta_f, min_ev_f, min_score_f = 15, 35, 0.25, 0.0, 0.0

        runner = get_runner()
        try:
            df = runner.rank_candidates_by_ev(
                dte_target=dte_int,
                delta_target=delta_f,
                top_n=limit_int,
                min_ev_dollars=min_ev_f,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_error(f"rank_candidates_by_ev failed: {exc}", 500)
            return

        if df is None or df.empty:
            self._send_json(
                {
                    "trades": [],
                    "count": 0,
                    "authority": "ev_ranked",
                    "engine_version": "ev_engine_2026_04_14",
                }
            )
            return

        trades = []
        for _, row in df.iterrows():
            spot = float(row.get("spot", 0) or 0)
            strike = float(row.get("strike", 0) or 0)
            premium = float(row.get("premium", 0) or 0)
            ev_dollars = float(row.get("ev_dollars", 0) or 0)
            prob_profit = float(row.get("prob_profit", 0) or 0)
            # Backward-compat heuristic filter only if wheel_score present
            wheel_score = row.get("wheel_score")
            if wheel_score is not None:
                try:
                    if float(wheel_score) < min_score_f:
                        continue
                except (TypeError, ValueError):
                    pass

            trades.append(
                {
                    # EV-native (authoritative) fields
                    "ticker": row.get("ticker"),
                    "strategy": "short_put",
                    "spot": round(spot, 2),
                    "strike": round(strike, 2),
                    "premium": round(premium, 3),
                    "dte": dte_int,
                    "iv": float(row.get("iv", 0) or 0),
                    "evDollars": round(ev_dollars, 2),
                    "evPerDay": round(float(row.get("ev_per_day", 0) or 0), 3),
                    "probProfit": round(prob_profit, 4),
                    "probAssignment": round(float(row.get("prob_assignment", 0) or 0), 4),
                    "cvar5": row.get("cvar_5"),
                    "cvar99Evt": row.get("cvar_99_evt"),
                    "tailXi": row.get("tail_xi"),
                    "heavyTail": bool(row.get("heavy_tail", False)),
                    "omegaRatio": row.get("omega_ratio"),
                    "fairValue": row.get("fair_value"),
                    "edgeVsFair": row.get("edge_vs_fair"),
                    "breakevenMovePct": row.get("breakeven_move_pct"),
                    "distributionSource": row.get("distribution_source"),
                    "dealerRegime": row.get("dealer_regime"),
                    "dealerMultiplier": row.get("dealer_multiplier"),
                    "gexTotal": row.get("gex_total"),
                    "gammaFlipDistancePct": row.get("gamma_flip_distance_pct"),
                    "nearestPutWallStrike": row.get("nearest_put_wall_strike"),
                    "nearestCallWallStrike": row.get("nearest_call_wall_strike"),
                    "daysToEarnings": row.get("days_to_earnings"),
                    # Backward-compat aliases (dashboard expects these)
                    "spotPrice": round(spot, 2),
                    "expectedPnL": round(ev_dollars, 2),  # was heuristic — now EV
                    "probability": round(prob_profit * 100, 1),
                    "maxLoss": round((strike - premium) * 100, 2) if strike > 0 else 0,
                    "score": row.get("wheel_score", round(prob_profit * 100, 1)),
                    "wheelScore": row.get("wheel_score"),
                    "recommendation": (
                        "proceed"
                        if ev_dollars >= 10 and prob_profit >= 0.65
                        else "review"
                        if ev_dollars > 0
                        else "skip"
                    ),
                    "expiration": "",
                }
            )

        self._send_json(
            {
                "trades": trades,
                "count": len(trades),
                "authority": "ev_ranked",
                "engine_version": "ev_engine_2026_04_14",
                "params": {
                    "limit": limit_int,
                    "dte": dte_int,
                    "delta": delta_f,
                    "min_ev": min_ev_f,
                    "min_score": min_score_f,
                    "as_of": as_of,
                },
            }
        )

    def _handle_analyze(self, ticker, as_of):
        if not ticker or not ticker.strip():
            self._send_error("ticker parameter is required", 400)
            return
        ticker = ticker.strip().upper()
        runner = get_runner()
        conn = get_connector()
        # Validate ticker exists in universe
        ohlcv = conn.get_ohlcv(ticker)
        if ohlcv.empty:
            self._send_json(
                {"error": f"Ticker '{ticker}' not found in data universe"},
                404,
            )
            return
        a = runner.analyze_ticker(ticker, as_of)
        # AUDIT FIX: wheelScore and strangleScore come from heuristic
        # modules (screen_candidates / StrangleTimingEngine), NOT from the
        # EV engine. The response now carries an explicit authority
        # contract so callers cannot mistake these scores for EV-backed
        # rankings. For tradeable EV decisions use /api/candidates.
        self._send_json(
            {
                "ticker": a.ticker,
                "authority": "heuristic_diagnostic",
                "tradeable_endpoint": "/api/candidates",
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
                "strangleAuthority": "heuristic_diagnostic",
                "riskFreeRate": a.risk_free_rate,
                "vixLevel": a.vix_level,
                "wheelScore": a.wheel_score,
                "wheelRecommendation": a.wheel_recommendation,
                "wheelScoreAuthority": "heuristic_diagnostic",
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
        """Legacy heuristic screen (RESEARCH-ONLY — not trade authority).

        AUDIT-V P0.1b: This endpoint still runs the legacy wheel_score
        heuristic — fundamental/liquidity filters, not probabilistic
        EV. It is kept for research introspection and for debugging
        the difference between the heuristic and EV-authoritative
        rankings, but every response is flagged ``authority:
        heuristic_research_only`` so the dashboard NEVER routes a
        trade through it.

        Callers who want tradeable candidates MUST use
        ``/api/candidates`` (EV-authoritative) or ``/api/tv/ranked``.
        """
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

        self._send_json(
            {
                "results": results,
                "count": len(results),
                "authority": "heuristic_research_only",
                "warning": (
                    "This endpoint returns the legacy heuristic wheel_score "
                    "ranking for research only. For tradeable candidates use "
                    "/api/candidates (EV-authoritative)."
                ),
                "tradeable_endpoint": "/api/candidates",
            }
        )

    def _handle_committee(self, ticker):
        from advisors import CommitteeEngine, format_committee_report
        from advisors.schema import (
            AdvisorInput,
            CandidateTrade,
            MarketContext,
            PortfolioContext,
            RegimeType,
            TradeType,
        )

        if not ticker or not ticker.strip():
            self._send_error("ticker parameter is required", 400)
            return
        ticker = ticker.strip().upper()
        conn = get_connector()
        runner = get_runner()

        # Get real data for this ticker
        analysis = runner.analyze_ticker(ticker)
        conn.get_fundamentals(ticker) or {}
        vix_data = conn.get_vix_regime() or {}

        spot = analysis.spot_price or 100
        iv = analysis.iv_30d or 25
        iv_decimal = iv / 100 if iv > 1 else iv

        # AUDIT-VIII P1.4: anchor the committee's candidate trade on the
        # EV ranker when available. Previously the handler constructed
        # a synthetic short put with a hardcoded strike (spot * 0.92),
        # hardcoded 0.04 risk-free rate, and an "expected_value" field
        # computed as a return-over-capital ratio — none of which
        # matched the EV engine's definitions. That produced a committee
        # verdict completely disconnected from the authoritative path,
        # which the UI could easily mistake for an EV-backed decision.
        # Now: when the EV ranker returns a row for this ticker, we
        # use its strike, premium, delta target, EV dollars, and
        # assignment probability. The synthetic BSM fallback is kept
        # only for tickers the EV ranker cannot price.
        ev_row = None
        ev_path_available = False
        try:
            ev_df = runner.rank_candidates_by_ev(
                tickers=[ticker],
                dte_target=45,
                delta_target=0.30,
                top_n=1,
                min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
                enforce_history_gate=False,
            )
            if ev_df is not None and len(ev_df) > 0:
                ev_row = ev_df.iloc[0].to_dict()
                ev_path_available = True
        except Exception:
            ev_row = None

        dte = 45
        if ev_row is not None:
            strike = float(ev_row["strike"])
            premium = float(ev_row["premium"])
            dte = int(ev_row.get("dte", 45))
            p_otm = float(ev_row.get("prob_profit", 0.70))
            # Put delta at the actual EV strike (signed negative).
            from scipy.stats import norm as _norm

            T = dte / 365
            if iv_decimal > 0 and T > 0:
                d1 = (
                    np.log(spot / strike) + (0.04 + 0.5 * iv_decimal**2) * T
                ) / (iv_decimal * np.sqrt(T))
                delta = float(-_norm.cdf(-d1))
            else:
                delta = -0.30
            ev_dollars = float(ev_row.get("ev_dollars", premium * 100 * p_otm))
        else:
            # Fallback synthetic BSM trade — only fires when the EV
            # ranker cannot price the ticker (e.g. missing OHLCV /
            # fundamentals). The response is still labelled
            # authority="heuristic_diagnostic" to prevent the UI from
            # treating it as EV-backed.
            from scipy.stats import norm as _norm

            strike = round(spot * 0.92, 0)
            T = dte / 365
            if iv_decimal > 0 and T > 0:
                d1 = (np.log(spot / strike) + (0.04 + 0.5 * iv_decimal**2) * T) / (
                    iv_decimal * np.sqrt(T)
                )
                d2 = d1 - iv_decimal * np.sqrt(T)
                premium = float(
                    strike * np.exp(-0.04 * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)
                )
                premium = max(0.01, premium)
                delta = float(-_norm.cdf(-d1))
                p_otm = float(_norm.cdf(d2))
            else:
                premium = 1.0
                delta = -0.30
                p_otm = 0.70
            ev_dollars = p_otm * premium * 100

        ev = ev_dollars  # committee schema label, dollar-valued

        # Build realistic AdvisorInput
        trade = CandidateTrade(
            ticker=ticker,
            trade_type=TradeType.SHORT_PUT,
            strike=strike,
            expiration_date="",
            dte=dte,
            delta=delta,
            premium=round(premium, 2),
            contracts=1,
            expected_value=round(ev, 2),
            p_otm=round(p_otm, 2),
            p_profit=round(p_otm * 0.95, 2),
            iv_rank=analysis.iv_rank * 100 if analysis.iv_rank < 1 else analysis.iv_rank,
            iv_percentile=analysis.iv_percentile * 100
            if analysis.iv_percentile < 1
            else analysis.iv_percentile,
            theta=round(premium / dte, 4),
            gamma=0.02,
            vega=round(premium * 0.1, 4),
            underlying_price=spot,
            earnings_before_expiry=analysis.days_to_earnings is not None
            and 0 < (analysis.days_to_earnings or 999) < dte,
        )

        portfolio = PortfolioContext(
            positions=[],
            total_equity=150000.0,
            cash_available=50000.0,
            buying_power=100000.0,
            sector_allocation={"Technology": 30, "Healthcare": 20, "Financials": 20, "Other": 30},
            top_5_concentration=50.0,
            portfolio_beta=1.0,
            portfolio_delta=0.5,
            max_drawdown_30d=-5.0,
            var_95=3.0,
            open_positions_count=3,
            total_premium_at_risk=5000.0,
            total_margin_used=20000.0,
        )

        vix_level = vix_data.get("vix", 20)
        regime = RegimeType.HIGH_VOL if vix_level > 25 else RegimeType.NORMAL
        market = MarketContext(
            regime=regime,
            vix=vix_level,
            vix_percentile=vix_data.get("vix_percentile", 50),
            spy_price=spot,
            spy_50ma=spot * 0.98,
            spy_200ma=spot * 0.95,
            fed_funds_rate=0.045,
            treasury_10y=0.042,
        )

        advisor_input = AdvisorInput(
            candidate_trade=trade,
            portfolio=portfolio,
            market=market,
            request_id=f"api_{ticker}",
        )

        committee = CommitteeEngine(parallel=False)
        result = committee.evaluate(advisor_input)

        # Deduplicate keyReasons across advisors so that boilerplate
        # templates (e.g. "Probability profile: …") shared between advisors
        # don't appear twice in the committee output. We keep the first
        # advisor's copy and drop the duplicate from later advisors.
        seen_reasons: set[str] = set()
        advisor_summaries = []
        for r in result.advisor_responses:
            unique_reasons = []
            for reason in r.key_reasons:
                key = reason.strip().lower()
                if key and key not in seen_reasons:
                    seen_reasons.add(key)
                    unique_reasons.append(reason)
            # Guarantee each advisor keeps at least 2 reasons — if dedup
            # stripped too many, fall back to a name-tagged fallback so the
            # reason is still unique across the committee.
            while len(unique_reasons) < 2:
                fallback = (
                    f"[{r.advisor_name}] {r.judgment.value.replace('_', ' ').title()} "
                    f"based on {r.judgment_summary[:80]}"
                )
                if fallback.strip().lower() not in seen_reasons:
                    seen_reasons.add(fallback.strip().lower())
                    unique_reasons.append(fallback)
                else:
                    break
            advisor_summaries.append(
                {
                    "name": r.advisor_name,
                    "judgment": r.judgment.value,
                    "summary": r.judgment_summary,
                    "keyReasons": unique_reasons[:3],
                    "criticalQuestions": r.critical_questions[:3],
                    "hiddenRisks": r.hidden_risks[:3],
                    "confidence": r.confidence.value,
                }
            )

        self._send_json(
            {
                "ticker": ticker,
                # AUDIT-VIII P1.4: explicit authority contract. The
                # committee is a narrative / risk-overlay layer, not
                # the tradeable authority. Callers that want a
                # tradeable decision must route through the EV ranker
                # at ``tradeable_endpoint``.
                "authority": "heuristic_diagnostic",
                "tradeable_endpoint": "/api/candidates",
                "ev_anchored": bool(ev_path_available),
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
                "trade": {
                    "ticker": trade.ticker,
                    "strategy": "short_put",
                    "strike": trade.strike,
                    "spot": spot,
                    "spotPrice": spot,
                    "dte": trade.dte,
                    "delta": trade.delta,
                    "premium": trade.premium,
                    "expectedValue": trade.expected_value,
                    "pOtm": trade.p_otm,
                    "ivRank": trade.iv_rank,
                    "theta": trade.theta,
                    "vega": trade.vega,
                    "gamma": trade.gamma,
                    "contracts": trade.contracts,
                    "earningsBeforeExpiry": trade.earnings_before_expiry,
                },
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

    def _handle_payoff(self, ticker, strategy, strike_str, premium_str, dte_str):
        """Generate payoff diagram data."""
        from engine.payoff_engine import compute_payoff

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)
        spot = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 100.0

        # Auto-estimate strike and premium if not provided
        fund = conn.get_fundamentals(ticker)
        iv = fund.get("implied_vol_atm", 25) if fund else 25

        if strike_str and strike_str != "None":
            strike = float(strike_str)
        else:
            strike = round(spot * 0.95 if strategy == "csp" else spot * 1.05, 0)

        if premium_str and premium_str != "None":
            premium = float(premium_str)
        else:
            # Rough BSM estimate
            iv_dec = iv / 100 if iv > 1 else iv
            T = int(dte_str) / 365
            from scipy.stats import norm as _norm

            d1 = (np.log(spot / strike) + (0.04 + 0.5 * iv_dec**2) * T) / (iv_dec * np.sqrt(T))
            d2 = d1 - iv_dec * np.sqrt(T)
            if strategy == "csp":
                premium = float(strike * np.exp(-0.04 * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1))
            else:
                premium = float(spot * _norm.cdf(d1) - strike * np.exp(-0.04 * T) * _norm.cdf(d2))
            premium = max(0.01, premium)

        # Round premium once so every downstream field stays consistent.
        premium = round(premium, 2)
        breakeven = round((strike - premium) if strategy == "csp" else (spot - premium), 2)
        data = compute_payoff(spot, strike, premium, strategy, breakeven=breakeven)

        self._send_json(
            {
                "ticker": ticker,
                "strategy": strategy,
                "spot": spot,
                "spotPrice": spot,
                "strike": strike,
                "premium": premium,
                "breakeven": breakeven,
                "maxProfit": round(premium * 100, 2),
                "maxLoss": round((strike - premium) * 100 if strategy == "csp" else 0, 2),
                "data": data,
            }
        )

    def _handle_expected_move(self, ticker, dte_str):
        """Compute expected move bands."""
        from engine.payoff_engine import compute_expected_move

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)
        spot = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 100.0

        fund = conn.get_fundamentals(ticker)
        iv = fund.get("implied_vol_atm", 25) if fund else 25

        result = compute_expected_move(spot, iv, int(dte_str))
        self._send_json(result)

    def _handle_strikes(self, ticker, strategy, dte_str):
        """Recommend optimal strikes."""
        from engine.data_integration import get_current_risk_free_rate
        from engine.payoff_engine import recommend_strikes

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)
        spot = float(ohlcv["close"].iloc[-1]) if not ohlcv.empty else 100.0

        fund = conn.get_fundamentals(ticker)
        iv = fund.get("implied_vol_atm", 25) if fund else 25
        rf = get_current_risk_free_rate()

        candidates = recommend_strikes(ticker, spot, iv, int(dte_str), rf, strategy)
        # AUDIT FIX: recommend_strikes uses BSM payoff math (delta/OTM
        # probability), NOT EV. These are diagnostic strike ranges — they
        # do not represent a ranked tradeable recommendation.
        self._send_json(
            {
                "ticker": ticker,
                "authority": "heuristic_diagnostic",
                "tradeable_endpoint": "/api/candidates",
                "note": (
                    "Strike ranges come from BSM payoff math, not EV. "
                    "For EV-ranked strikes use /api/candidates."
                ),
                "strategy": strategy,
                "spot": round(spot, 2),
                "iv": round(iv, 1),
                "dte": int(dte_str),
                "riskFreeRate": round(rf, 4),
                "recommendations": candidates,
            }
        )

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
            # Only compute scores for the requested window so long histories
            # don't blow up into an O(N) loop across years of data.
            hist = engine.compute_historical_scores(ohlcv, lookback_required=100, last_n=days)
            data = []
            if not hist.empty:
                for _, row in hist.tail(days).iterrows():
                    data.append(
                        {
                            "date": str(row.get("date", ""))[:10],
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

        # AUDIT FIX: StrangleTimingEngine is a heuristic scoring module
        # (Bollinger/ATR/RSI/trend/range composite). It does NOT compute
        # probabilistic EV. Response now carries an authority contract.
        self._send_json(
            {
                "ticker": ticker,
                "authority": "heuristic_diagnostic",
                "tradeable_endpoint": "/api/candidates",
                "note": (
                    "Strangle timing is a heuristic regime/phase signal. "
                    "For EV-ranked strangle candidates use /api/candidates "
                    "or /api/tv/ranked with a strangle delta pair."
                ),
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

    # ------------------------------------------------------------------
    # TradingView bridge handlers
    # ------------------------------------------------------------------

    def _handle_tv_signal(self, ticker: str, as_of):
        """Return the canonical TV-parity signal for a single ticker.

        This endpoint is the "pull" side of the bridge: TradingView can be
        polled from an external poller, or a power user can paste the JSON
        into their own Pine script overlay, and the engine always replies
        with the same shape that ``tv_signals.compute_tv_signal`` produces.
        """
        from engine.tv_signals import compute_tv_signal

        if not ticker:
            self._send_error("ticker parameter is required", 400)
            return

        conn = get_connector()
        ohlcv = conn.get_ohlcv(ticker)
        if ohlcv is None or ohlcv.empty:
            self._send_json({"error": f"Ticker '{ticker}' not found"}, 404)
            return

        # Optional IV overlay from fundamentals (not required)
        iv_rank = None
        vol_risk_premium = None
        try:
            iv_rank = conn.get_iv_rank(ticker, as_of)
        except Exception:
            iv_rank = None
        try:
            vol_risk_premium = conn.get_vol_risk_premium(ticker, as_of)
        except Exception:
            vol_risk_premium = None

        signal = compute_tv_signal(
            ohlcv,
            ticker=ticker,
            as_of=as_of,
            iv_rank=iv_rank,
            vol_risk_premium=vol_risk_premium,
        )
        self._send_json(signal.to_dict())

    def _handle_tv_scan(self, limit, phase_filter, zone_filter):
        """Scan the universe and return tickers matching a TV signal filter.

        Query parameters
        ----------------
        limit : int
            Maximum number of tickers to return (default 25).
        phase : str, optional
            Restrict results to the given phase (e.g. ``post_expansion``).
        zone : str, optional
            One of ``wheel_put``, ``covered_call``, ``strangle``.

        Implementation notes (audit-V P0.1b)
        ------------------------------------
        Candidate pool is now EV-ranked via
        :meth:`WheelRunner.rank_candidates_by_ev`, NOT the legacy
        heuristic ``screen_candidates``. This unifies decision
        authority with ``/api/candidates`` so there is no path where
        a heuristic wheel_score can surface names that EV would
        reject.

        The TV signal layer itself remains purely *visual / context* —
        it attaches regime phase and zone flags to the EV-ranked pool
        so traders can eyeball the chart state, but it never
        upgrades or overrides the EV ranking.
        """
        from engine.tv_signals import compute_tv_signal

        try:
            limit_int = max(1, int(limit or 25))
        except ValueError:
            limit_int = 25

        runner = get_runner()
        conn = get_connector()

        # EV-authoritative pool — no heuristic wheel_score filter.
        # We pull a deeper bench than the final limit so the phase/zone
        # filter below has room to work.
        try:
            df = runner.rank_candidates_by_ev(
                dte_target=35,
                delta_target=0.25,
                top_n=max(limit_int * 4, 60),
                min_ev_dollars=0.0,  # strict: positive-EV only
                include_diagnostic_fields=True,
            )
        except Exception:
            traceback.print_exc()
            df = None

        if df is None or df.empty:
            self._send_json(
                {
                    "signals": [],
                    "count": 0,
                    "authority": "ev_ranked",
                    "layer": "visual_context_only",
                }
            )
            return

        results = []
        for _, row in df.iterrows():
            tkr = row.get("ticker")
            if not tkr:
                continue
            ohlcv = conn.get_ohlcv(tkr)
            if ohlcv is None or ohlcv.empty:
                continue
            signal = compute_tv_signal(
                ohlcv,
                ticker=tkr,
                iv_rank=row.get("iv_rank"),
            )
            if not signal.ok:
                continue

            if phase_filter and signal.phase != phase_filter:
                continue
            if zone_filter:
                zmap = {
                    "wheel_put": signal.wheel_put_zone,
                    "covered_call": signal.covered_call_zone,
                    "strangle": signal.strangle_zone,
                }
                if not zmap.get(zone_filter, False):
                    continue

            results.append(
                {
                    "ticker": tkr,
                    "phase": signal.phase,
                    "signal_action": signal.signal_action,
                    "wheel_put_zone": signal.wheel_put_zone,
                    "covered_call_zone": signal.covered_call_zone,
                    "strangle_zone": signal.strangle_zone,
                    "avoid_zone": signal.avoid_zone,
                    "bb_width_pctl": round(signal.bb_width_pctl, 1),
                    "rsi_14": round(signal.rsi_14, 1),
                    "close": signal.close,
                    # EV-native ranking signals from the authoritative path
                    "ev_dollars": float(row.get("ev_dollars", 0) or 0),
                    "ev_per_day": float(row.get("ev_per_day", 0) or 0),
                    "prob_profit": float(row.get("prob_profit", 0) or 0),
                    "distribution_source": row.get("distribution_source"),
                }
            )
            if len(results) >= limit_int:
                break

        self._send_json(
            {
                "signals": results,
                "count": len(results),
                "authority": "ev_ranked",
                "layer": "visual_context_only",
                "engine_version": "ev_engine_2026_04_14",
            }
        )

    def _handle_tv_webhook(
        self,
        payload: dict,
        raw_body: bytes = b"",
        signature_header: str = "",
    ):
        """Ingest a TradingView Pine Script webhook alert.

        The webhook is the **push** side of the bridge. Pine fires an alert
        with a JSON payload matching
        ``tradingview/alert_payload_schema.json``; this handler validates
        it, enriches it with options/fundamentals data, stores it in an
        in-memory ring buffer (``_TV_ALERT_LOG``) that ``/api/tv/alerts``
        serves back, and returns the enriched decision so Pine's alert
        dialog can show a confirmation toast.

        Security hardening (AUDIT)
        --------------------------
        Three layers of defence are applied, in order:

        1. **HMAC-SHA256 body signature** (preferred). When
           ``TV_WEBHOOK_HMAC_SECRET`` is set, the handler requires a
           matching hex digest in the ``X-Signature`` /
           ``X-Signature-256`` / ``X-Hub-Signature-256`` header, verified
           in constant time via :func:`hmac.compare_digest`. This protects
           against tampering with the body in transit.

        2. **Plain shared secret** (legacy, lower strength). When
           ``TV_WEBHOOK_SECRET`` is set, the handler requires
           ``payload["secret"]`` to match using a constant-time compare.
           TradingView cannot send arbitrary headers, so for pure Pine
           alerts this is the only option. Callers who can send headers
           should prefer HMAC.

        3. **Timestamp freshness + nonce-replay guard**. Every accepted
           payload is tagged with a SHA-256 digest of the raw body plus the
           received signature header. If the same digest is seen again
           within ``_TV_WEBHOOK_MAX_AGE_SEC`` the alert is rejected with
           409. Additionally, if the payload contains a ``timestamp`` field
           that parses as a POSIX seconds or ISO 8601 string, it must be
           within the freshness window to be accepted.

        When none of the env vars above are set the handler accepts all
        alerts (the intended behaviour for local development on a
        loopback-only socket).
        """
        import os
        from datetime import datetime, timezone

        from engine.tv_signals import TVAlert

        alert = TVAlert.parse(payload)
        if not alert.is_valid():
            self._send_error("alert payload missing ticker or signal", 400)
            return

        hmac_secret = os.environ.get("TV_WEBHOOK_HMAC_SECRET", "")
        plain_secret = os.environ.get("TV_WEBHOOK_SECRET", "")

        # Layer 1: HMAC verification (preferred).
        if hmac_secret:
            if not _tv_verify_hmac(raw_body, signature_header, hmac_secret):
                self._send_error("unauthorized (hmac)", 401)
                return

        # Layer 2: Plain shared secret (in-body). Constant-time compare.
        if plain_secret:
            if not hmac.compare_digest(
                (alert.secret or "").encode("utf-8"), plain_secret.encode("utf-8")
            ):
                self._send_error("unauthorized", 401)
                return

        now = time.time()

        # Layer 3a: Timestamp freshness (if the payload includes one).
        ts_val = alert.timestamp
        if ts_val:
            ts_parsed: float | None = None
            try:
                if isinstance(ts_val, (int, float)):
                    # TradingView {{time}} is epoch ms — accept either.
                    ts_parsed = float(ts_val) / 1000.0 if ts_val > 1e12 else float(ts_val)
                elif isinstance(ts_val, str):
                    # Numeric string?
                    try:
                        num = float(ts_val)
                        ts_parsed = num / 1000.0 if num > 1e12 else num
                    except ValueError:
                        # ISO 8601
                        cleaned = ts_val.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(cleaned)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts_parsed = dt.timestamp()
            except Exception:
                ts_parsed = None

            if ts_parsed is not None:
                age = now - ts_parsed
                if abs(age) > _TV_WEBHOOK_MAX_AGE_SEC:
                    self._send_error(
                        f"alert outside freshness window (age={int(age)}s)", 400
                    )
                    return

        # Layer 3b: Nonce / replay guard.
        digest = hashlib.sha256(raw_body + b"|" + signature_header.encode("utf-8")).hexdigest()
        if not _tv_seen_register(digest, now):
            self._send_error("duplicate alert (replay blocked)", 409)
            return

        enriched = self._enrich_alert(alert)

        # Store in ring buffer
        _TV_ALERT_LOG.append(enriched)
        if len(_TV_ALERT_LOG) > _TV_ALERT_LOG_MAX:
            del _TV_ALERT_LOG[0 : len(_TV_ALERT_LOG) - _TV_ALERT_LOG_MAX]

        self._send_json({"accepted": True, "enriched": enriched})

    def _handle_tv_enrich(self, ticker: str, signal_name: str, as_of):
        """Pull-style enrichment: build the same decision object the
        webhook would, but for an arbitrary ticker/signal combination.

        Useful for UI callers that want the enriched view without wiring
        up a Pine alert first.
        """
        from engine.tv_signals import TVAlert

        if not ticker:
            self._send_error("ticker parameter is required", 400)
            return

        alert = TVAlert(ticker=ticker, signal=signal_name, source="api")
        enriched = self._enrich_alert(alert, as_of=as_of)
        self._send_json(enriched)

    def _handle_tv_alerts(self, limit: int):
        """Return the most recent alerts held in the in-memory log."""
        limit = max(1, min(limit or 50, _TV_ALERT_LOG_MAX))
        items = list(reversed(_TV_ALERT_LOG[-limit:]))
        self._send_json({"alerts": items, "count": len(items)})

    def _handle_tv_dealer_positioning(
        self,
        ticker: str,
        dte_target: int,
        assumption: str,
    ):
        """Return aggregated dealer positioning (GEX, walls, regime) for a ticker.

        Query params:
            ticker      required — underlying symbol
            dte         target DTE (default 35); picks the option chain
                        expiry closest to today+dte
            assumption  dealer-direction convention
                        (long_calls_short_puts | short_both)

        Returns a :class:`MarketStructure.to_dict` payload including
        aggregate GEX / DEX / vanna / charm, per-strike exposures,
        top-3 call & put gamma walls, nearest walls above/below spot,
        zero-gamma flip level, pinning zones, regime label, and
        confidence. When the chain is unavailable the endpoint returns
        HTTP 404 with a structured reason.
        """
        from datetime import date, timedelta

        if not ticker:
            self._send_error("ticker parameter is required", 400)
            return

        from engine.dealer_positioning import (
            DealerAssumption,
            DealerPositioningAnalyzer,
        )

        try:
            assumption_enum = DealerAssumption(assumption)
        except ValueError:
            assumption_enum = DealerAssumption.LONG_CALLS_SHORT_PUTS

        conn = get_connector()
        chain_df = None
        try:
            if hasattr(conn, "get_options"):
                chain_df = conn.get_options(ticker)
            elif hasattr(conn, "get_option_chain"):
                chain_df = conn.get_option_chain(ticker)
        except Exception as exc:
            traceback.print_exc()
            self._send_error(f"chain fetch failed: {exc}", 500)
            return

        if chain_df is None or len(chain_df) == 0:
            self._send_error("option chain unavailable for ticker", 404)
            return

        # Pick expiry closest to today + dte_target
        cdf = chain_df.copy()
        cdf.columns = [c.lower() for c in cdf.columns]
        if "expiration" in cdf.columns:
            import pandas as pd

            cdf["expiration"] = pd.to_datetime(cdf["expiration"], errors="coerce")
            target_ts = pd.Timestamp(date.today() + timedelta(days=dte_target))
            cdf["_dte_gap"] = (cdf["expiration"] - target_ts).abs()
            best = cdf.sort_values("_dte_gap")["expiration"].iloc[0]
            cdf = cdf[cdf["expiration"] == best].drop(columns=["_dte_gap"])
            expiry_d = best.date() if hasattr(best, "date") else best
        else:
            expiry_d = date.today() + timedelta(days=dte_target)

        # Get spot
        spot = 0.0
        try:
            ohlcv = conn.get_ohlcv(ticker)
            if ohlcv is not None and len(ohlcv) > 0:
                close_col = "close" if "close" in ohlcv.columns else "Close"
                spot = float(ohlcv[close_col].iloc[-1])
        except Exception:
            pass
        if spot <= 0 and "underlying_price" in cdf.columns:
            spot = float(cdf["underlying_price"].dropna().iloc[0])

        if spot <= 0:
            self._send_error("spot price unavailable", 404)
            return

        analyzer = DealerPositioningAnalyzer(assumption=assumption_enum)
        ms = analyzer.analyze(
            chain=cdf,
            spot=spot,
            expiry=expiry_d,
            ticker=ticker,
        )
        self._send_json(ms.to_dict())

    def _handle_tv_dossier(
        self,
        top_n: int,
        dte_target: int,
        delta_target: float,
        min_ev_dollars: float,
        as_of,
        tickers_csv,
        timeframe: str,
        screenshots_dir: str,
    ):
        """Mode B dossier endpoint.

        Runs the engine-first workflow end-to-end:
          1. Rank candidates by EV.
          2. For the top N, load a TradingView screenshot via a
             filesystem-based :class:`FilesystemChartProvider` pointed
             at ``screenshots_dir``. The terminal is expected to drop
             screenshots at ``<screenshots_dir>/<TICKER>/<TIMEFRAME>.png``.
          3. Run the :class:`EnginePhaseReviewer` to produce a verdict
             per candidate.
          4. Return the full dossier JSON for the dashboard candidate
             table + detail drawer.

        Query params:
            top_n           top-N candidates to attach charts to (default 10)
            dte             target DTE (default 35)
            delta           target put delta (default 0.25)
            min_ev          hard EV filter (default 0)
            as_of           optional PIT cutoff YYYY-MM-DD
            tickers         optional comma-separated subset
            timeframe       TradingView timeframe for screenshots (default 1D)
            screenshots_dir filesystem provider base directory
        """
        from engine.tradingview_bridge import FilesystemChartProvider

        runner = get_runner()
        tickers = None
        if tickers_csv:
            tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]

        provider = FilesystemChartProvider(base_dir=screenshots_dir)

        try:
            dossiers = runner.build_candidate_dossiers(
                tickers=tickers,
                dte_target=dte_target,
                delta_target=delta_target,
                top_n=max(1, top_n),
                min_ev_dollars=min_ev_dollars,
                as_of=as_of,
                chart_provider=provider,
                chart_timeframe=timeframe,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_error(f"build_candidate_dossiers failed: {exc}", 500)
            return

        records = [d.to_dict() for d in dossiers]
        counts = {
            "proceed": sum(1 for d in dossiers if d.verdict == "proceed"),
            "review": sum(1 for d in dossiers if d.verdict == "review"),
            "skip": sum(1 for d in dossiers if d.verdict == "skip"),
            "blocked": sum(1 for d in dossiers if d.verdict == "blocked"),
        }
        self._send_json(
            {
                "dossiers": records,
                "count": len(records),
                "verdict_counts": counts,
                "params": {
                    "top_n": top_n,
                    "dte_target": dte_target,
                    "delta_target": delta_target,
                    "min_ev_dollars": min_ev_dollars,
                    "as_of": as_of,
                    "tickers": tickers,
                    "timeframe": timeframe,
                    "screenshots_dir": screenshots_dir,
                },
                "engine_version": "ev_engine_2026_04_14",
            }
        )

    def _handle_tv_ranked(
        self,
        limit: int,
        dte_target: int,
        delta_target: float,
        min_ev_dollars: float,
        as_of,
        tickers_csv,
    ):
        """EV-ranked candidates endpoint for the dashboard candidate table.

        This is the audit-II upgrade: replaces the legacy /api/candidates
        heuristic score with probabilistic expected value per day via
        :meth:`WheelRunner.rank_candidates_by_ev`. The endpoint returns
        JSON that the Next.js dashboard can render directly as a table,
        one row per candidate, pre-filtered by the event lockout gate
        and sorted by ``ev_per_day`` descending.

        Query params:
            limit        top N candidates to return (default 20)
            dte          target days-to-expiry for the synthetic trade (default 35)
            delta        target put delta (positive; default 0.25)
            min_ev       hard threshold on ev_dollars (default 0)
            as_of        optional PIT cutoff YYYY-MM-DD
            tickers      optional comma-separated ticker subset
        """
        runner = get_runner()
        tickers = None
        if tickers_csv:
            tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]

        try:
            df = runner.rank_candidates_by_ev(
                tickers=tickers,
                dte_target=dte_target,
                delta_target=delta_target,
                top_n=max(1, limit),
                min_ev_dollars=min_ev_dollars,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send_error(f"rank_candidates_by_ev failed: {exc}", 500)
            return

        records = df.to_dict(orient="records") if hasattr(df, "to_dict") else []
        payload = {
            "candidates": records,
            "count": len(records),
            "params": {
                "limit": limit,
                "dte_target": dte_target,
                "delta_target": delta_target,
                "min_ev_dollars": min_ev_dollars,
                "as_of": as_of,
                "tickers": tickers,
            },
            "engine_version": "ev_engine_2026_04_14",
        }
        self._send_json(payload)

    # ------------------------------------------------------------------

    def _enrich_alert(self, alert, as_of=None):
        """Build the enriched decision object for a TV alert.

        This is the decision layer of the integration: after Pine tells us
        "post-expansion stabilization on MU," we want to attach IV rank,
        expected move, preferred delta zone, event risk, and a ranked
        verdict so the trader does not have to re-query five endpoints.
        """
        from engine.tv_signals import compute_tv_signal

        conn = get_connector()
        runner = get_runner()

        ohlcv = conn.get_ohlcv(alert.ticker)
        if ohlcv is None or ohlcv.empty:
            return {
                "ticker": alert.ticker,
                "signal": alert.signal,
                "accepted": False,
                "reason": "ticker_not_in_universe",
            }

        iv_rank = None
        vrp = None
        try:
            iv_rank = conn.get_iv_rank(alert.ticker, as_of)
        except Exception:
            pass
        try:
            vrp = conn.get_vol_risk_premium(alert.ticker, as_of)
        except Exception:
            pass

        sig = compute_tv_signal(
            ohlcv,
            ticker=alert.ticker,
            as_of=as_of,
            iv_rank=iv_rank,
            vol_risk_premium=vrp,
        )

        # Run the wheel analysis to get events + scoring
        try:
            analysis = runner.analyze_ticker(alert.ticker, as_of)
            wheel_score = float(analysis.wheel_score)
            wheel_reco = analysis.wheel_recommendation
            days_to_earnings = analysis.days_to_earnings
            sector = analysis.sector
        except Exception:
            wheel_score = 0.0
            wheel_reco = "unknown"
            days_to_earnings = None
            sector = ""

        # Verdict is EV-AUTHORITATIVE (audit fix 2026-04-14).
        # Previously used a heuristic `wheel_score >= 60` rule here, which
        # produced "proceed" verdicts stored in the ring buffer that were
        # never validated against EV. That's a silent authority leak: any
        # user reading `/api/tv/alerts` would see verdicts indistinguishable
        # from EV-backed ones.
        # Now: run the EV ranker on this single ticker and use ev_dollars +
        # prob_profit as the authority. wheel_score is kept only as a
        # supplementary diagnostic, never as decision authority.
        agrees = sig.signal_action == alert.signal or getattr(
            sig, alert.signal.replace(" ", "_"), False
        )
        ev_dollars = 0.0
        ev_per_day = 0.0
        prob_profit = 0.0
        prob_assignment = 0.0
        verdict_authority = "ev_ranked"
        verdict_reason = ""
        try:
            # Preferred DTE is deterministic based on phase; the ranker
            # will use this as the target DTE for the synthetic trade.
            preferred_dte_tmp = 31 if sig.phase == "post_expansion" else 45
            preferred_delta_tmp = {
                "wheel_put_zone": 0.20,
                "covered_call_zone": 0.22,
                "strangle_zone": 0.18,
            }.get(alert.signal, 0.22)
            ev_df = runner.rank_candidates_by_ev(
                tickers=[alert.ticker],
                dte_target=preferred_dte_tmp,
                delta_target=preferred_delta_tmp,
                top_n=1,
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
            if ev_df is not None and len(ev_df) > 0:
                r0 = ev_df.iloc[0]
                ev_dollars = float(r0.get("ev_dollars", 0) or 0)
                ev_per_day = float(r0.get("ev_per_day", 0) or 0)
                prob_profit = float(r0.get("prob_profit", 0) or 0)
                prob_assignment = float(r0.get("prob_assignment", 0) or 0)
        except Exception:
            verdict_authority = "ev_unavailable"
            verdict_reason = "ev_computation_failed"

        # Hard event gate (second line of defense — EV should already have
        # blocked). This remains as a belt-and-suspenders skip.
        if days_to_earnings is not None and 0 <= days_to_earnings < 5:
            verdict = "skip"
            verdict_reason = "earnings_within_5d"
        elif verdict_authority != "ev_ranked":
            verdict = "review"
            verdict_reason = verdict_reason or "ev_engine_unreachable"
        elif ev_dollars < 0:
            verdict = "skip"
            verdict_reason = "negative_ev"
        elif ev_dollars >= 10 and prob_profit >= 0.65 and agrees:
            verdict = "proceed"
            verdict_reason = "ev_above_threshold_and_chart_agrees"
        elif ev_dollars > 0:
            verdict = "review"
            verdict_reason = "positive_but_low_ev" if ev_dollars < 10 else "chart_disagrees"
        else:
            verdict = "skip"
            verdict_reason = "ev_zero_or_below"

        # Preferred expiry / delta suggestion — deterministic heuristic
        preferred_dte = 31 if sig.phase == "post_expansion" else 45
        preferred_delta = {
            "wheel_put_zone": (0.18, 0.22),
            "covered_call_zone": (0.20, 0.25),
            "strangle_zone": (0.15, 0.20),
        }.get(alert.signal, (0.20, 0.25))

        return {
            "ticker": alert.ticker,
            "signal": alert.signal,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "authority": verdict_authority,
            "ev_dollars": round(ev_dollars, 2),
            "ev_per_day": round(ev_per_day, 3),
            "prob_profit": round(prob_profit, 4),
            "prob_assignment": round(prob_assignment, 4),
            "pine_agrees": agrees,
            "phase": sig.phase,
            "bollinger_state": sig.bollinger_state,
            "atr_state": sig.atr_state,
            "rsi_14": round(sig.rsi_14, 1),
            "trend_state": sig.trend_state,
            "range_state": sig.range_state,
            "wheel_put_zone": sig.wheel_put_zone,
            "covered_call_zone": sig.covered_call_zone,
            "strangle_zone": sig.strangle_zone,
            "wheel_score": round(wheel_score, 1),
            "wheel_recommendation": wheel_reco,
            "iv_rank": round(float(iv_rank), 1) if iv_rank is not None else None,
            "vol_risk_premium": round(float(vrp), 1) if vrp is not None else None,
            "days_to_earnings": days_to_earnings,
            "sector": sector,
            "close": sig.close,
            "preferred_dte": preferred_dte,
            "preferred_delta_range": list(preferred_delta),
            "alert_source": alert.source or "webhook",
            "alert_timestamp": alert.timestamp,
        }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # News endpoints
    # ------------------------------------------------------------------
    def _handle_news(self, limit: int):
        """Serve the most recent news stories from the in-memory buffer.

        Stories are ingested by ``POST /api/news/ingest`` (called by
        ``scripts/orchestrate.py`` after running the news pipeline)
        and served back here for the dashboard + committee.
        """
        limit = max(1, min(limit, _NEWS_BUFFER_MAX))
        items = list(reversed(_NEWS_BUFFER[-limit:]))
        self._send_json({"stories": items, "count": len(items)})

    def _handle_news_ingest(self, payload: dict):
        """Ingest news stories from the orchestrator / news pipeline.

        Expects ``{"stories": [...]}`` where each story is a dict with
        at minimum ``title`` and ``summary``. Optional fields:
        ``tickers``, ``impact``, ``source``, ``timestamp``, ``url``.

        Stories are appended to the in-memory ring buffer and served
        via ``GET /api/news``. The endpoint also extracts ticker
        mentions and checks them against the event gate.
        """
        stories = payload.get("stories", [])
        if not isinstance(stories, list):
            self._send_error("stories must be a list", 400)
            return

        ingested = 0
        for story in stories:
            if not isinstance(story, dict):
                continue
            if not story.get("title") and not story.get("summary"):
                continue
            story.setdefault(
                "ingested_at", datetime.now(timezone.utc).isoformat()
            )
            story.setdefault("source", "pipeline")
            _NEWS_BUFFER.append(story)
            ingested += 1

        # Trim buffer
        while len(_NEWS_BUFFER) > _NEWS_BUFFER_MAX:
            _NEWS_BUFFER.pop(0)

        self._send_json({"ingested": ingested, "buffer_size": len(_NEWS_BUFFER)})

    def _handle_memo(self, ticker, as_of):
        """Generate AI trade memo for a ticker."""
        from engine.trade_memo import MemoGenerator

        gen = MemoGenerator()
        result = gen.generate_memo(ticker, as_of)
        self._send_json(result)

    def _handle_summary(self, ticker):
        """Generate quick AI summary for a ticker."""
        from engine.trade_memo import MemoGenerator

        gen = MemoGenerator()
        summary = gen.generate_quick_summary(ticker)
        self._send_json({"ticker": ticker, "summary": summary})

    def _handle_ollama_status(self):
        """Check Ollama availability and models."""
        from engine.trade_memo import _check_ollama

        status = _check_ollama()
        self._send_json(status)

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[Engine API] {args[0]}")


def main():
    port = 8787
    # ThreadingHTTPServer spawns one thread per request so a slow committee
    # or memo call can't block the 5+ parallel fetches the dashboard fires
    # when a trader switches tickers.
    server = ThreadingHTTPServer(("0.0.0.0", port), EngineAPIHandler)
    server.daemon_threads = True
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
    print("  GET /api/chart/bollinger?ticker=AAPL&days=120")
    print("  GET /api/chart/rsi?ticker=AAPL")
    print("  GET /api/chart/atr?ticker=AAPL")
    print("  GET /api/chart/ohlcv?ticker=AAPL")
    print("  GET /api/chart/strangle?ticker=AAPL")
    print("  GET /api/strangle?ticker=AAPL")
    print("  GET /api/iv_history?ticker=AAPL&days=252")
    print("  GET /api/payoff?ticker=AAPL&strategy=csp&dte=45")
    print("  GET /api/expected_move?ticker=AAPL&dte=45")
    print("  GET /api/strikes?ticker=AAPL&strategy=csp&dte=45")
    print("  GET /api/memo?ticker=AAPL")
    print("  GET /api/summary?ticker=AAPL")
    print("  GET /api/ollama_status")
    print("  GET  /api/tv/signal?ticker=AAPL")
    print("  GET  /api/tv/scan?limit=25&zone=wheel_put")
    print("  GET  /api/tv/enrich?ticker=AAPL&signal=wheel_put_zone")
    print("  GET  /api/tv/alerts?limit=50")
    print("  POST /api/tv/webhook")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
