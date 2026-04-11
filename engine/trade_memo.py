"""
AI Trade Memo Generator

Generates institutional-quality trade memos by combining:
- Quantitative analysis from the engine (fundamentals, IV, Greeks, timing)
- Investment committee verdict (Buffett, Munger, Simons, Taleb)
- Strangle timing assessment
- Market context (VIX, regime, events)

Uses dual-model Ollama routing:
- 72B model (qwen2.5:72b) for serious analysis: trade memos, deep research
- 32B model (qwen2.5:32b) for quick tasks: summaries, news extraction

Usage:
    from engine.trade_memo import MemoGenerator

    gen = MemoGenerator()
    memo = gen.generate_memo("AAPL")
    print(memo["memo"])
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

logger = logging.getLogger(__name__)

# Model configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
HEAVY_MODEL = os.environ.get("OLLAMA_HEAVY_MODEL", "qwen2.5:72b")
FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:32b")


def _call_ollama(prompt: str, model: str | None = None, system: str = "") -> str:
    """Call Ollama API and return the response text."""
    import urllib.error
    import urllib.request

    model = model or HEAVY_MODEL
    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2000,
        },
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")
    except urllib.error.URLError as e:
        logger.warning(f"Ollama unavailable ({model}): {e}")
        return ""
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return ""


def _check_ollama() -> dict:
    """Check which Ollama models are available."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return {
                "available": True,
                "models": models,
                "heavy_available": HEAVY_MODEL in models,
                "fast_available": FAST_MODEL in models,
            }
    except Exception:
        return {"available": False, "models": [], "heavy_available": False, "fast_available": False}


class MemoGenerator:
    """
    Generate institutional-quality trade memos using AI + engine data.

    Combines quantitative analysis, committee verdict, and AI narrative
    into a single actionable document.
    """

    SYSTEM_PROMPT = """You are a senior quantitative analyst at a top-tier options trading desk.
You write concise, precise trade memos for portfolio managers. Your writing style:
- Lead with the verdict (trade or don't trade)
- Back every statement with data
- Be direct — no hedging language or filler
- Flag risks explicitly with severity
- Use dollar amounts and percentages, not vague terms
- Structure: Verdict → Thesis → Risk Assessment → Trade Parameters → Decision Factors"""

    def __init__(
        self,
        heavy_model: str | None = None,
        fast_model: str | None = None,
    ):
        self.heavy_model = heavy_model or HEAVY_MODEL
        self.fast_model = fast_model or FAST_MODEL

    def generate_memo(self, ticker: str, as_of: str | None = None) -> dict:
        """
        Generate a complete trade memo for a ticker.

        Gathers all engine data, runs the committee, and asks the AI
        to synthesize everything into a readable memo.

        Returns dict with: ticker, memo, analysis, committee, model_used
        """
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner()

        # 1. Get quantitative analysis
        analysis = runner.analyze_ticker(ticker, as_of)

        # 2. Get committee verdict
        committee_data = self._run_committee(ticker, analysis)

        # 3. Get strangle timing detail
        strangle_data = self._get_strangle_detail(ticker, runner)

        # 4. Build the data package for the AI
        data_package = self._build_data_package(analysis, committee_data, strangle_data)

        # 5. Generate the memo with AI
        prompt = self._build_memo_prompt(ticker, data_package)
        memo_text = _call_ollama(prompt, model=self.heavy_model, system=self.SYSTEM_PROMPT)

        # If AI is unavailable, generate a structured memo from data
        if not memo_text:
            memo_text = self._generate_fallback_memo(ticker, data_package)

        return {
            "ticker": ticker,
            "date": as_of or date.today().isoformat(),
            "memo": memo_text,
            "analysis": {
                "spotPrice": analysis.spot_price,
                "sector": analysis.sector,
                "marketCap": analysis.market_cap,
                "peRatio": analysis.pe_ratio,
                "beta": analysis.beta,
                "iv30d": analysis.iv_30d,
                "rv30d": analysis.rv_30d,
                "ivRank": analysis.iv_rank,
                "volRiskPremium": analysis.vol_risk_premium,
                "wheelScore": analysis.wheel_score,
                "strangleScore": analysis.strangle_score,
                "stranglePhase": analysis.strangle_phase,
            },
            "committee": committee_data,
            "model_used": self.heavy_model,
        }

    def generate_quick_summary(self, ticker: str) -> str:
        """
        Generate a 2-3 sentence summary using the fast model.

        For quick lookups during trading — not a full memo.
        """
        from engine.wheel_runner import WheelRunner

        runner = WheelRunner()
        analysis = runner.analyze_ticker(ticker)

        prompt = (
            f"Give a 2-3 sentence trading summary for {ticker}:\n"
            f"Price: ${analysis.spot_price:.2f}, Sector: {analysis.sector}\n"
            f"IV: {analysis.iv_30d:.1f}%, RV: {analysis.rv_30d:.1f}%, "
            f"Vol Premium: {analysis.vol_risk_premium:+.1f}%\n"
            f"Wheel Score: {analysis.wheel_score:.0f}/100, "
            f"Strangle Timing: {analysis.strangle_score:.0f}/100 ({analysis.strangle_phase})\n"
            f"Beta: {analysis.beta:.2f}, P/E: {analysis.pe_ratio:.1f}\n"
            f"Days to earnings: {analysis.days_to_earnings or 'N/A'}\n"
            f"Focus on: Is this a good wheel/strangle candidate right now? Why or why not?"
        )

        result = _call_ollama(prompt, model=self.fast_model)
        if not result:
            return (
                f"{ticker} at ${analysis.spot_price:.2f}. "
                f"Wheel score {analysis.wheel_score:.0f}/100 "
                f"({'strong' if analysis.wheel_score >= 75 else 'moderate' if analysis.wheel_score >= 55 else 'weak'}). "
                f"IV {analysis.iv_30d:.1f}% vs RV {analysis.rv_30d:.1f}% "
                f"({'premium rich' if analysis.vol_risk_premium > 3 else 'fair' if analysis.vol_risk_premium > -3 else 'premium cheap'})."
            )
        return result

    def _run_committee(self, ticker: str, analysis) -> dict:
        """Run the investment committee and return structured results."""
        try:
            import numpy as np
            from scipy.stats import norm

            from advisors import CommitteeEngine
            from advisors.schema import (
                AdvisorInput,
                CandidateTrade,
                MarketContext,
                PortfolioContext,
                RegimeType,
                TradeType,
            )

            spot = analysis.spot_price or 100
            iv = analysis.iv_30d or 25
            iv_dec = iv / 100 if iv > 1 else iv
            strike = round(spot * 0.92, 0)
            dte = 45
            T = dte / 365

            if iv_dec > 0 and T > 0:
                d1 = (np.log(spot / strike) + (0.04 + 0.5 * iv_dec**2) * T) / (iv_dec * np.sqrt(T))
                d2 = d1 - iv_dec * np.sqrt(T)
                premium = max(
                    0.01, float(strike * np.exp(-0.04 * T) * norm.cdf(-d2) - spot * norm.cdf(-d1))
                )
                delta = float(-norm.cdf(-d1))
                p_otm = float(norm.cdf(d2))
            else:
                premium, delta, p_otm = 1.0, -0.30, 0.70

            ev = p_otm * premium * 100 / (strike * 100) * (365 / dte) * 100

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
                total_equity=150000,
                cash_available=50000,
                buying_power=100000,
                sector_allocation={
                    "Technology": 30,
                    "Healthcare": 20,
                    "Financials": 20,
                    "Other": 30,
                },
                top_5_concentration=50,
                portfolio_beta=1.0,
                portfolio_delta=0.5,
                max_drawdown_30d=-5,
                var_95=3,
                open_positions_count=3,
                total_premium_at_risk=5000,
                total_margin_used=20000,
            )

            market = MarketContext(
                regime=RegimeType.HIGH_VOL if (analysis.vix_level or 0) > 25 else RegimeType.NORMAL,
                vix=analysis.vix_level or 20,
                vix_percentile=50,
                spy_price=spot,
                spy_50ma=spot * 0.98,
                spy_200ma=spot * 0.95,
                fed_funds_rate=0.045,
                treasury_10y=0.042,
            )

            committee = CommitteeEngine(parallel=False)
            result = committee.evaluate(
                AdvisorInput(
                    candidate_trade=trade,
                    portfolio=portfolio,
                    market=market,
                    request_id=f"memo_{ticker}",
                )
            )

            return {
                "judgment": result.committee_judgment.value,
                "confidence": result.committee_confidence.value,
                "reasoning": result.committee_reasoning,
                "advisors": [
                    {
                        "name": r.advisor_name,
                        "judgment": r.judgment.value,
                        "summary": r.judgment_summary,
                    }
                    for r in result.advisor_responses
                ],
                "risks": result.unresolved_risks[:4],
                "actions": result.required_before_trade[:4],
            }
        except Exception as e:
            logger.warning(f"Committee failed: {e}")
            return {
                "judgment": "error",
                "confidence": "low",
                "reasoning": str(e),
                "advisors": [],
                "risks": [],
                "actions": [],
            }

    def _get_strangle_detail(self, ticker: str, runner) -> dict:
        """Get detailed strangle timing analysis."""
        try:
            from engine.strangle_timing import StrangleTimingEngine

            conn = runner.connector
            ohlcv = conn.get_ohlcv(ticker)
            if ohlcv.empty or len(ohlcv) < 100:
                return {}

            engine = StrangleTimingEngine()
            score = engine.score_entry(ohlcv)
            regime = engine.classify_regime(ohlcv)

            return {
                "score": round(float(score.total_score), 1),
                "recommendation": score.recommendation,
                "phase": regime.phase.value,
                "components": {
                    "bollinger": round(float(score.bollinger_score), 1),
                    "atr": round(float(score.atr_score), 1),
                    "rsi": round(float(score.rsi_score), 1),
                    "trend": round(float(score.trend_score), 1),
                    "range": round(float(score.range_score), 1),
                },
                "rsi_14": round(float(regime.rsi_14), 1),
                "bb_pct_b": round(float(regime.bb_pct_b), 3),
                "warnings": {
                    "compression": score.compression_warning,
                    "expansion": score.expansion_active,
                    "trend": score.strong_trend_warning,
                },
            }
        except Exception as e:
            logger.warning(f"Strangle detail failed: {e}")
            return {}

    def _build_data_package(self, analysis, committee: dict, strangle: dict) -> str:
        """Build a structured data summary for the AI prompt."""
        lines = [
            f"TICKER: {analysis.ticker}",
            f"PRICE: ${analysis.spot_price:.2f}",
            f"SECTOR: {analysis.sector}",
            f"MARKET CAP: ${analysis.market_cap / 1e9:.1f}B",
            f"P/E: {analysis.pe_ratio:.1f}",
            f"BETA: {analysis.beta:.2f}",
            f"CREDIT RATING: {analysis.credit_rating}",
            "",
            "--- VOLATILITY ---",
            f"IV (30d): {analysis.iv_30d:.1f}%",
            f"RV (30d): {analysis.rv_30d:.1f}%",
            f"IV RANK: {analysis.iv_rank:.0f}",
            f"VOL RISK PREMIUM: {analysis.vol_risk_premium:+.1f}%",
            "",
            "--- EVENTS ---",
            f"DAYS TO EARNINGS: {analysis.days_to_earnings or 'N/A'}",
            f"NEXT EX-DIV: {analysis.next_div_date or 'N/A'}",
            f"VIX: {analysis.vix_level:.1f}",
            f"RISK-FREE RATE: {analysis.risk_free_rate:.2%}",
            "",
            "--- WHEEL SCORE ---",
            f"WHEEL SCORE: {analysis.wheel_score:.0f}/100",
            f"RECOMMENDATION: {analysis.wheel_recommendation}",
        ]

        if strangle:
            lines.extend(
                [
                    "",
                    "--- STRANGLE TIMING ---",
                    f"ENTRY SCORE: {strangle.get('score', 0)}/100",
                    f"PHASE: {strangle.get('phase', 'unknown')}",
                    f"RECOMMENDATION: {strangle.get('recommendation', 'N/A')}",
                    f"COMPONENTS: BB={strangle.get('components', {}).get('bollinger', 0)} "
                    f"ATR={strangle.get('components', {}).get('atr', 0)} "
                    f"RSI={strangle.get('components', {}).get('rsi', 0)} "
                    f"TREND={strangle.get('components', {}).get('trend', 0)} "
                    f"RANGE={strangle.get('components', {}).get('range', 0)}",
                    f"RSI(14): {strangle.get('rsi_14', 'N/A')}",
                    f"BB %B: {strangle.get('bb_pct_b', 'N/A')}",
                ]
            )
            warnings = strangle.get("warnings", {})
            if any(warnings.values()):
                active = [k for k, v in warnings.items() if v]
                lines.append(f"WARNINGS: {', '.join(active)}")

        if committee.get("advisors"):
            lines.extend(
                [
                    "",
                    "--- INVESTMENT COMMITTEE ---",
                    f"VERDICT: {committee['judgment'].upper()}",
                    f"CONFIDENCE: {committee['confidence']}",
                    f"REASONING: {committee['reasoning']}",
                ]
            )
            for adv in committee["advisors"]:
                lines.append(f"  {adv['name']}: {adv['judgment'].upper()} — {adv['summary']}")
            if committee.get("risks"):
                lines.append("UNRESOLVED RISKS:")
                for risk in committee["risks"]:
                    lines.append(f"  - {risk}")

        return "\n".join(lines)

    def _build_memo_prompt(self, ticker: str, data_package: str) -> str:
        """Build the prompt for the AI to generate the memo."""
        return f"""Write a trade memo for {ticker} based on this data:

{data_package}

Structure your memo as:

## VERDICT
[1 sentence: TRADE or DO NOT TRADE, with conviction level]

## THESIS
[2-3 sentences: Why this is or isn't an attractive wheel/strangle candidate right now]

## RISK ASSESSMENT
[Bullet points: Top 3-4 risks, each with severity (HIGH/MEDIUM/LOW)]

## TRADE PARAMETERS (if recommending trade)
[Suggested strategy, strike, DTE, position size reasoning]

## DECISION FACTORS
[What would change this recommendation — both upgrade and downgrade triggers]

Be specific. Use the numbers provided. Do not invent data not in the package."""

    def _generate_fallback_memo(self, ticker: str, data_package: str) -> str:
        """Generate a structured memo without AI when Ollama is unavailable."""
        data_package.split("\n")

        # Extract key values
        memo = f"""## TRADE MEMO: {ticker}
Generated: {date.today().isoformat()} | Model: Fallback (No AI)

{data_package}

---
NOTE: This is a data-only memo. Start Ollama with qwen2.5:72b for AI-generated
narrative analysis. Run: ollama pull qwen2.5:72b && ollama serve
"""
        return memo
