"""
TradingView Signal Bridge

Deterministic signal computation layer that mirrors the logic embedded in
the companion Pine Script indicator (``tradingview/smart_wheel_signals.pine``).

Architecture principle
----------------------
TradingView's role in this project is the **visual cockpit**: charting,
custom indicator overlays, alert delivery, and discretionary review.
The engine's role is the **decision layer**: options math, scoring,
historical labeling, portfolio intelligence, and committee reasoning.

This module is the bridge between the two. It exposes a single function,
:func:`compute_tv_signal`, that takes a ticker's OHLCV history and returns
a canonical ``TVSignal`` struct containing:

* Raw technical features (bb width percentile, RSI, ATR percentile, trend,
  range distance)
* Qualitative states matching the Pine Script state machine
* A composite volatility-lifecycle phase classification
* Zone flags (``wheel_put_zone``, ``covered_call_zone``, ``strangle_zone``)
* An aggregate ``signal_action`` hint (``short_premium_ok``, ``avoid``, ...)

The same features are then re-implemented in Pine v5 on the chart. Because
both sides consume the same primitives (close, Bollinger bands,
ATR, RSI, SMA), they produce identical labels for the same bar — which is
what makes the engine and the chart agree.

Non-goals
---------
This module does **not**:
* Make trading decisions (``wheel_runner`` owns composite scoring).
* Look at options chains (``engine_api`` enriches signals at webhook time).
* Classify news / sentiment.

Keep this module tight, deterministic, and side-effect free so it can be
called inside HTTP handlers and Pine-side parity tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.features.technical import TechnicalFeatures

# ---------------------------------------------------------------------------
# Tunable parameters (kept in sync with the Pine Script counterpart)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "bb_window": 20,
    "bb_std": 2.0,
    "atr_window": 14,
    "rsi_window": 14,
    "trend_ma_window": 20,
    "range_lookback": 20,
    "percentile_lookback": 252,  # ~1 trading year
    "slope_lookback": 5,
}

# Thresholds — identical constants live at the top of the Pine Script
# indicator so state labels map 1:1 between engine and chart.
BB_WIDTH_NARROW_PCTL = 20.0
BB_WIDTH_WIDE_PCTL = 60.0
BB_WIDTH_EXPAND_SLOPE = 0.001
BB_WIDTH_CONTRACT_SLOPE = -0.0005

ATR_LOW_PCTL = 25.0
ATR_HIGH_PCTL = 60.0
ATR_RISING_SLOPE = 0.05
ATR_DECLINING_SLOPE = -0.03
ATR_FLAT_ABS = 0.03

RSI_EXTREME_OB = 80.0
RSI_OB = 70.0
RSI_OS = 30.0
RSI_EXTREME_OS = 20.0

TREND_STRONG_SLOPE = 0.02
TREND_WEAK_SLOPE = 0.005
TREND_FLAT_PRICE_GAP = 0.02

RANGE_NEAR_EDGE = 0.01


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TVSignal:
    """Canonical TradingView-parity signal for a single bar.

    Every field in this struct has a 1:1 counterpart in the Pine Script
    indicator. The engine and Pine must agree on these values for a given
    ticker and as-of date; parity tests live under ``tests/test_tv_signals.py``.
    """

    ticker: str
    as_of: Optional[str] = None
    bar_count: int = 0
    ok: bool = False
    reason: str = ""

    # --- Raw features ----------------------------------------------------
    close: float = 0.0
    sma_20: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_width_pctl: float = 0.0
    bb_pct_b: float = 0.5
    bb_width_slope: float = 0.0
    atr_14: float = 0.0
    atr_pctl: float = 0.0
    atr_slope: float = 0.0
    rsi_14: float = 50.0
    ma_slope: float = 0.0
    price_vs_ma: float = 0.0
    dist_to_high_20: float = 0.0
    dist_to_low_20: float = 0.0

    # --- Qualitative states ---------------------------------------------
    bollinger_state: str = "unknown"
    atr_state: str = "unknown"
    rsi_state: str = "unknown"
    trend_state: str = "unknown"
    range_state: str = "unknown"

    # --- Composite classification ---------------------------------------
    phase: str = "unknown"
    confidence: float = 0.0

    # --- Zone flags (drive Pine labels and alert conditions) ------------
    wheel_put_zone: bool = False
    covered_call_zone: bool = False
    strangle_zone: bool = False
    avoid_zone: bool = False

    # --- Aggregate hint (string consumed by the webhook enricher) -------
    signal_action: str = "neutral"

    def to_dict(self) -> dict:
        """Return a JSON-safe dict representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


_tech = TechnicalFeatures()


def _safe_pctl(series: pd.Series, value: float) -> float:
    """Return the percentile rank of ``value`` within the historical window."""
    hist = series.dropna()
    if len(hist) == 0:
        return 50.0
    return float((hist < value).mean() * 100.0)


def _bollinger_state(width_pctl: float, slope: float) -> str:
    if width_pctl < BB_WIDTH_NARROW_PCTL:
        return "narrow"
    if slope > BB_WIDTH_EXPAND_SLOPE:
        return "expanding"
    if width_pctl > BB_WIDTH_WIDE_PCTL and slope < BB_WIDTH_CONTRACT_SLOPE:
        return "wide_contracting"
    if width_pctl > BB_WIDTH_WIDE_PCTL:
        return "wide_flat"
    return "normal"


def _atr_state(pctl: float, slope: float) -> str:
    if pctl < ATR_LOW_PCTL:
        return "low"
    if slope > ATR_RISING_SLOPE:
        return "rising"
    if pctl > ATR_HIGH_PCTL and abs(slope) < ATR_FLAT_ABS:
        return "elevated_flat"
    if slope < ATR_DECLINING_SLOPE:
        return "declining"
    return "normal"


def _rsi_state(rsi: float) -> str:
    if rsi >= RSI_EXTREME_OB:
        return "extreme_ob"
    if rsi >= RSI_OB:
        return "overbought"
    if rsi <= RSI_EXTREME_OS:
        return "extreme_os"
    if rsi <= RSI_OS:
        return "oversold"
    return "neutral"


def _trend_state(ma_slope: float, price_vs_ma: float) -> str:
    if abs(ma_slope) > TREND_STRONG_SLOPE and abs(price_vs_ma) < TREND_FLAT_PRICE_GAP:
        return "strong_up" if ma_slope > 0 else "strong_down"
    if abs(ma_slope) > TREND_WEAK_SLOPE:
        return "weak"
    return "flat"


def _range_state(dist_high: float, dist_low: float, pct_b: float) -> str:
    if pct_b > 1.0 or pct_b < 0.0:
        return "beyond_band"
    if dist_high < RANGE_NEAR_EDGE:
        return "near_high"
    if dist_low < RANGE_NEAR_EDGE:
        return "near_low"
    return "mid"


def _classify_phase(
    bb_state: str, atr_state: str, trend_state: str
) -> tuple[str, float]:
    """Map component states to the volatility lifecycle phase.

    This mirrors ``StrangleTimingEngine._classify_phase`` semantics so that
    the TV overlay and the engine agree on regime names. Returns
    ``(phase, confidence)`` where confidence is 0.0-1.0.
    """
    # Compression: narrow bands + low ATR
    if bb_state == "narrow" and atr_state in ("low", "declining"):
        return "compression", 0.85

    # Expansion: bands widening or ATR rising
    if bb_state == "expanding" or atr_state == "rising":
        return "expansion", 0.80

    # Post-expansion: wide bands but flattening / ATR elevated flat or declining
    if bb_state in ("wide_flat", "wide_contracting") and atr_state in (
        "elevated_flat",
        "declining",
        "normal",
    ):
        return "post_expansion", 0.85

    # Trend: strong directional MA slope overrides symmetric-vol classification
    if trend_state in ("strong_up", "strong_down"):
        return "trend", 0.75

    # Default
    return "normal", 0.50


def _resolve_zones(
    phase: str,
    trend_state: str,
    rsi_state: str,
    range_state: str,
    vol_risk_premium: Optional[float],
    iv_rank: Optional[float],
) -> tuple[bool, bool, bool, bool, str]:
    """Derive actionable zone flags from the classified state.

    Returns
    -------
    tuple
        ``(wheel_put_zone, covered_call_zone, strangle_zone, avoid_zone,
        signal_action)``

    The decision table below encodes the team's published wheel/strangle
    playbook. It is intentionally conservative: when any single component
    screams "avoid" (e.g. compression, extreme overbought while at band
    edge), the function returns the avoid flag even if another component
    would approve entry.
    """
    # Hard avoid: compression has no premium to sell
    if phase == "compression":
        return False, False, False, True, "avoid_compression"

    # Hard avoid: active expansion — wait for stabilization
    if phase == "expansion":
        return False, False, False, True, "avoid_expansion"

    # Hard avoid: strong trend → symmetric strangles get run over
    if phase == "trend":
        # But a strong up-trend is fine for covered calls on existing stock
        if trend_state == "strong_up":
            return False, True, False, False, "trend_covered_call_ok"
        return False, False, False, True, "avoid_trend"

    # Post-expansion is the sweet spot
    if phase == "post_expansion":
        wheel_put = rsi_state in ("neutral", "oversold", "extreme_os") and (
            range_state in ("mid", "near_low")
        )
        covered_call = rsi_state in ("neutral", "overbought", "extreme_ob") and (
            range_state in ("mid", "near_high")
        )
        strangle = rsi_state == "neutral" and range_state == "mid"

        # Optional IV overlay: require IV rank > 30 if supplied
        if iv_rank is not None and iv_rank < 30:
            # Low IV rank → premium too thin for systematic entry
            return False, False, False, True, "avoid_low_iv"

        # Optional vol risk premium check
        if vol_risk_premium is not None and vol_risk_premium < -5:
            return False, False, False, True, "avoid_neg_vrp"

        if strangle:
            return wheel_put, covered_call, True, False, "short_premium_ok"
        if wheel_put and not covered_call:
            return True, False, False, False, "wheel_put_zone"
        if covered_call and not wheel_put:
            return False, True, False, False, "covered_call_zone"
        if wheel_put and covered_call:
            return True, True, False, False, "wheel_both_sides"

    # Normal/unknown: neutral, no action
    return False, False, False, False, "neutral"


def compute_tv_signal(
    df: pd.DataFrame,
    *,
    ticker: str,
    as_of: Optional[str] = None,
    iv_rank: Optional[float] = None,
    vol_risk_premium: Optional[float] = None,
    params: Optional[dict] = None,
) -> TVSignal:
    """Compute the canonical TV-parity signal for the latest bar of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV frame. Must contain ``open``, ``high``, ``low``, ``close``
        columns. The function uses the last bar as "today".
    ticker : str
        Ticker symbol (used for logging and round-tripping).
    as_of : str, optional
        ISO date of the evaluation bar. Purely informational — the function
        always uses the last row of ``df``.
    iv_rank : float, optional
        Externally supplied IV rank (0-100). When provided, the zone
        resolver refuses to fire entry signals if IV rank is below 30.
    vol_risk_premium : float, optional
        ``iv_30d - rv_30d`` in percent. When supplied, negative values
        suppress short-premium signals.
    params : dict, optional
        Override for the default tunable parameters. Any key present in
        :data:`DEFAULTS` is accepted.

    Returns
    -------
    TVSignal
        A fully populated signal struct. When the input frame is too short
        for reliable percentile calculations the struct returns
        ``ok=False`` with a ``reason`` describing why.
    """
    p = {**DEFAULTS, **(params or {})}

    signal = TVSignal(ticker=ticker, as_of=as_of, bar_count=len(df))

    if df is None or df.empty:
        signal.reason = "empty_dataframe"
        return signal

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        signal.reason = f"missing_columns:{sorted(required_cols - set(df.columns))}"
        return signal

    min_bars = max(p["percentile_lookback"], p["bb_window"] + p["slope_lookback"])
    if len(df) < min_bars:
        signal.reason = f"insufficient_history:{len(df)}<{min_bars}"
        return signal

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # --- Bollinger -------------------------------------------------------
    upper, middle, lower = _tech.bollinger_bands(close, p["bb_window"], p["bb_std"])
    bb_width = (upper - lower) / middle
    width_current = float(bb_width.iloc[-1])
    width_hist = bb_width.dropna().tail(p["percentile_lookback"])
    width_pctl = _safe_pctl(width_hist, width_current)

    bb_width_slope = 0.0
    if len(bb_width) >= p["slope_lookback"]:
        bb_width_slope = float(
            (bb_width.iloc[-1] - bb_width.iloc[-p["slope_lookback"]])
            / p["slope_lookback"]
        )

    upper_last = float(upper.iloc[-1])
    lower_last = float(lower.iloc[-1])
    pct_b = 0.5
    if upper_last - lower_last > 0:
        pct_b = float((close.iloc[-1] - lower_last) / (upper_last - lower_last))

    bb_state = _bollinger_state(width_pctl, bb_width_slope)

    # --- ATR -------------------------------------------------------------
    atr = _tech.atr(high, low, close, p["atr_window"])
    atr_current = float(atr.iloc[-1])
    atr_hist = atr.dropna().tail(p["percentile_lookback"])
    atr_pctl = _safe_pctl(atr_hist, atr_current)

    atr_slope = 0.0
    if len(atr) >= p["slope_lookback"] and float(atr.iloc[-p["slope_lookback"]]) > 0:
        atr_slope = float(
            (atr.iloc[-1] - atr.iloc[-p["slope_lookback"]])
            / atr.iloc[-p["slope_lookback"]]
        )

    atr_state = _atr_state(atr_pctl, atr_slope)

    # --- RSI -------------------------------------------------------------
    rsi_series = _tech.rsi(close, p["rsi_window"])
    rsi_14 = float(rsi_series.iloc[-1])
    rsi_state = _rsi_state(rsi_14)

    # --- Trend (SMA20) ---------------------------------------------------
    sma_20 = _tech.sma(close, p["trend_ma_window"])
    ma_slope = 0.0
    if len(sma_20) >= p["slope_lookback"] and float(sma_20.iloc[-p["slope_lookback"]]) > 0:
        ma_slope = float(
            (sma_20.iloc[-1] - sma_20.iloc[-p["slope_lookback"]])
            / sma_20.iloc[-p["slope_lookback"]]
        )

    price_vs_ma = 0.0
    if float(sma_20.iloc[-1]) > 0:
        price_vs_ma = float((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])

    trend_state = _trend_state(ma_slope, price_vs_ma)

    # --- Range -----------------------------------------------------------
    rolling_high = high.rolling(p["range_lookback"]).max()
    rolling_low = low.rolling(p["range_lookback"]).min()
    dist_high = float((rolling_high.iloc[-1] - close.iloc[-1]) / close.iloc[-1])
    dist_low = float((close.iloc[-1] - rolling_low.iloc[-1]) / close.iloc[-1])
    range_state = _range_state(dist_high, dist_low, pct_b)

    # --- Phase -----------------------------------------------------------
    phase, confidence = _classify_phase(bb_state, atr_state, trend_state)

    # --- Zones -----------------------------------------------------------
    wheel_put, covered_call, strangle, avoid, action = _resolve_zones(
        phase=phase,
        trend_state=trend_state,
        rsi_state=rsi_state,
        range_state=range_state,
        vol_risk_premium=vol_risk_premium,
        iv_rank=iv_rank,
    )

    # --- Populate struct -------------------------------------------------
    signal.ok = True
    signal.close = float(close.iloc[-1])
    signal.sma_20 = float(sma_20.iloc[-1]) if not np.isnan(sma_20.iloc[-1]) else 0.0
    signal.bb_upper = upper_last
    signal.bb_lower = lower_last
    signal.bb_width = width_current
    signal.bb_width_pctl = width_pctl
    signal.bb_pct_b = pct_b
    signal.bb_width_slope = bb_width_slope
    signal.atr_14 = atr_current
    signal.atr_pctl = atr_pctl
    signal.atr_slope = atr_slope
    signal.rsi_14 = rsi_14
    signal.ma_slope = ma_slope
    signal.price_vs_ma = price_vs_ma
    signal.dist_to_high_20 = dist_high
    signal.dist_to_low_20 = dist_low
    signal.bollinger_state = bb_state
    signal.atr_state = atr_state
    signal.rsi_state = rsi_state
    signal.trend_state = trend_state
    signal.range_state = range_state
    signal.phase = phase
    signal.confidence = confidence
    signal.wheel_put_zone = wheel_put
    signal.covered_call_zone = covered_call
    signal.strangle_zone = strangle
    signal.avoid_zone = avoid
    signal.signal_action = action

    return signal


# ---------------------------------------------------------------------------
# Webhook alert ingestion
# ---------------------------------------------------------------------------


@dataclass
class TVAlert:
    """Normalized TradingView webhook alert.

    Pine Script alerts can send arbitrary JSON via ``alert_message``.
    This module accepts the canonical schema documented in
    ``tradingview/alert_payload_schema.json``:

    .. code-block:: json

        {
            "ticker": "MU",
            "signal": "wheel_put_zone",
            "price": 82.45,
            "timeframe": "1D",
            "source": "smart_wheel_signals_v1",
            "timestamp": "2026-04-13T14:32:00Z",
            "phase": "post_expansion",
            "bb_width_pctl": 72,
            "rsi": 44.8,
            "secret": "<shared>"
        }

    Missing fields are tolerated; unknown fields are preserved under
    :attr:`extras`.
    """

    ticker: str
    signal: str
    price: float = 0.0
    timeframe: str = ""
    source: str = ""
    timestamp: str = ""
    phase: str = ""
    bb_width_pctl: float = 0.0
    rsi: float = 0.0
    secret: str = ""
    extras: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: dict) -> "TVAlert":
        """Build a TVAlert from a raw webhook JSON body.

        Unknown keys are captured into ``extras`` so downstream consumers
        can log them without losing information.
        """
        known = {
            "ticker",
            "signal",
            "price",
            "timeframe",
            "source",
            "timestamp",
            "phase",
            "bb_width_pctl",
            "rsi",
            "secret",
        }
        extras = {k: v for k, v in payload.items() if k not in known}
        return cls(
            ticker=str(payload.get("ticker", "")).upper(),
            signal=str(payload.get("signal", "")),
            price=float(payload.get("price", 0) or 0),
            timeframe=str(payload.get("timeframe", "")),
            source=str(payload.get("source", "")),
            timestamp=str(payload.get("timestamp", "")),
            phase=str(payload.get("phase", "")),
            bb_width_pctl=float(payload.get("bb_width_pctl", 0) or 0),
            rsi=float(payload.get("rsi", 0) or 0),
            secret=str(payload.get("secret", "")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def is_valid(self) -> bool:
        """A minimal schema check before enrichment."""
        return bool(self.ticker) and bool(self.signal)
