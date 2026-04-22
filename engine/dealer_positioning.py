"""
Market-structure dealer-positioning analyzer.

Why this module exists
----------------------
Raw long-vs-short positioning counts are structurally useless because
every long is matched by a short at the transaction level. What
actually moves option pricing and P&L is **dealer hedging flow** —
once dealers are long or short gamma, they are forced to trade the
underlying in the opposite direction to stay delta-neutral, and those
forced flows dampen or amplify realized volatility relative to
implied.

The standard literature approach (SpotGamma, SqueezeMetrics, Kai
Volatility) estimates dealer positioning under the assumption that
retail is long calls and short puts (net), which means dealers are
net short calls and net long puts. The alternative SpotGamma
assumption used here — LONG_CALLS_SHORT_PUTS — is the opposite and is
the most widely cited convention in the published literature. We
default to it, make it explicit, and expose the alternative
:class:`DealerAssumption.SHORT_BOTH` for ablation studies.

What this module computes
-------------------------
Given an option chain (per-strike ``strike``, ``option_type``,
``open_interest``, ``implied_vol``, optional ``delta``/``gamma``) and
a spot price, the analyzer produces a :class:`MarketStructure`
dataclass containing:

* ``gex_total``     — dollar gamma exposure per 1% move (SpotGamma-style)
* ``dex_total``     — dollar delta exposure
* ``vanna_total``   — aggregate vanna (for observability)
* ``charm_total``   — aggregate charm (for observability)
* ``per_strike``    — per-strike breakdown for diagnostics
* ``call_walls``    — top-N strikes by positive net gamma-dollar
* ``put_walls``     — top-N strikes by negative net gamma-dollar
* ``nearest_call_wall`` / ``nearest_put_wall`` — above / below spot
* ``flip_level``    — price at which total GEX crosses zero
* ``flip_distance_pct`` — spot-distance to flip as % of spot
* ``pinning_zones`` — strikes within ±5% spot with high OI × gamma
* ``regime``        — "long_gamma_dampening" | "short_gamma_amplifying" |
                      "near_flip" | "neutral"

Plus a single helper :func:`dealer_regime_multiplier` that maps the
regime into a scalar in [0.70, 1.05] for the EV engine to consume.

Hard guardrails
---------------
* Dealer positioning is a **multiplier**, never a decider. It cannot
  rescue a negative-EV trade. This is enforced at the EV engine level
  (the short-circuit on ``ev_raw < 0`` runs before any regime
  multiplication).
* The long-gamma boost is capped at 1.05; the short-gamma cut can go
  as low as 0.70. Asymmetric by design — we stay conservative.
* Missing / degenerate chains degrade gracefully: the analyzer returns
  a :class:`MarketStructure` with ``regime="neutral"`` and
  ``confidence=0.0`` rather than crashing.

Units
-----
GEX is reported in **dollars per 1% underlying move**, matching
SpotGamma convention. A GEX of +1e9 means the aggregate dealer book
will buy/sell $1bn of underlying per 1% move. Positive GEX ⇒ dealers
are long gamma ⇒ they sell into rallies and buy into dips ⇒ realized
vol is dampened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from .option_pricer import black_scholes_all_greeks

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Enums + dataclasses
# ----------------------------------------------------------------------
class DealerAssumption(str, Enum):
    """Assumed direction of dealers relative to retail.

    LONG_CALLS_SHORT_PUTS is the SpotGamma convention:
    dealers are LONG calls (retail buys them from market makers) and
    SHORT puts (retail sells them via CSPs / covered calls). Under this
    convention, positive total GEX means dealers are LONG gamma ⇒
    hedging flows *dampen* realized vol.

    SHORT_BOTH models a market where dealers are net sellers of both
    legs — the sign of the GEX contribution from calls flips. This
    produces the inverted story and is useful for ablation.
    """

    LONG_CALLS_SHORT_PUTS = "long_calls_short_puts"
    SHORT_BOTH = "short_both"


@dataclass
class PerStrikeExposure:
    """Per-strike contribution to the aggregate dealer book."""

    strike: float
    call_oi: int
    put_oi: int
    call_gamma: float  # per-share gamma (from BSM)
    put_gamma: float
    call_delta: float
    put_delta: float
    # Dollar exposures under the chosen dealer assumption, per 1% move.
    call_gex: float
    put_gex: float
    net_gex: float
    net_dex: float
    net_vanna: float
    net_charm: float


@dataclass
class GammaWall:
    """A strike that acts as a gamma concentration point."""

    strike: float
    distance_pct: float  # signed: positive if above spot
    net_gex: float
    side: Literal["call", "put"]


@dataclass
class MarketStructure:
    """Aggregated dealer-positioning view of an option chain."""

    ticker: str
    as_of: datetime
    spot: float
    expiry: date
    assumption: DealerAssumption

    # Aggregates
    gex_total: float = 0.0
    dex_total: float = 0.0
    vanna_total: float = 0.0  # observability only — not used by multiplier
    charm_total: float = 0.0  # observability only — not used by multiplier

    # Structure
    per_strike: list[PerStrikeExposure] = field(default_factory=list)
    call_walls: list[GammaWall] = field(default_factory=list)
    put_walls: list[GammaWall] = field(default_factory=list)
    nearest_call_wall: GammaWall | None = None
    nearest_put_wall: GammaWall | None = None

    flip_level: float | None = None
    flip_distance_pct: float | None = None
    pinning_zones: list[float] = field(default_factory=list)

    # Regime label + confidence in [0, 1]
    regime: Literal[
        "long_gamma_dampening",
        "short_gamma_amplifying",
        "near_flip",
        "neutral",
    ] = "neutral"
    confidence: float = 0.0

    # Diagnostic
    n_strikes: int = 0
    n_calls: int = 0
    n_puts: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        """JSON-safe dictionary for the API layer."""
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "spot": self.spot,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "assumption": self.assumption.value,
            "gex_total": self.gex_total,
            "dex_total": self.dex_total,
            "vanna_total": self.vanna_total,
            "charm_total": self.charm_total,
            "regime": self.regime,
            "confidence": self.confidence,
            "flip_level": self.flip_level,
            "flip_distance_pct": self.flip_distance_pct,
            "pinning_zones": list(self.pinning_zones),
            "nearest_call_wall": (
                _wall_to_dict(self.nearest_call_wall)
                if self.nearest_call_wall
                else None
            ),
            "nearest_put_wall": (
                _wall_to_dict(self.nearest_put_wall)
                if self.nearest_put_wall
                else None
            ),
            "call_walls": [_wall_to_dict(w) for w in self.call_walls],
            "put_walls": [_wall_to_dict(w) for w in self.put_walls],
            "per_strike": [
                {
                    "strike": p.strike,
                    "call_oi": p.call_oi,
                    "put_oi": p.put_oi,
                    "net_gex": p.net_gex,
                    "net_dex": p.net_dex,
                }
                for p in self.per_strike
            ],
            "n_strikes": self.n_strikes,
            "n_calls": self.n_calls,
            "n_puts": self.n_puts,
            "notes": self.notes,
        }


def _wall_to_dict(w: GammaWall) -> dict:
    return {
        "strike": w.strike,
        "distance_pct": w.distance_pct,
        "net_gex": w.net_gex,
        "side": w.side,
    }


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------
class DealerPositioningAnalyzer:
    """Compute aggregated dealer positioning from an option chain.

    Designed as a pure-function analyzer: no I/O, no caching, no
    hidden state. Pass in a DataFrame, get back a :class:`MarketStructure`.

    Parameters
    ----------
    assumption:
        Dealer direction convention. Default
        :attr:`DealerAssumption.LONG_CALLS_SHORT_PUTS` (SpotGamma).
    risk_free_rate:
        Used when Greeks need to be recomputed from IV + spot.
    flip_neighborhood_pct:
        Distance-to-flip threshold (as % of spot) below which the
        regime is classified as ``near_flip`` regardless of the sign
        of total GEX. Default 1%.
    near_wall_pct:
        How close spot must be to a gamma wall (as % of spot) for the
        wall to be considered "nearest". Walls farther than this are
        still listed in ``call_walls`` / ``put_walls`` but not
        promoted to ``nearest_*_wall``.
    top_walls:
        How many walls to return on each side.
    pin_window_pct:
        Pinning-zone window around spot.
    """

    def __init__(
        self,
        assumption: DealerAssumption = DealerAssumption.LONG_CALLS_SHORT_PUTS,
        risk_free_rate: float = 0.05,
        flip_neighborhood_pct: float = 0.01,
        near_wall_pct: float = 0.05,
        top_walls: int = 3,
        pin_window_pct: float = 0.05,
    ) -> None:
        self.assumption = assumption
        self.risk_free_rate = risk_free_rate
        self.flip_neighborhood_pct = flip_neighborhood_pct
        self.near_wall_pct = near_wall_pct
        self.top_walls = top_walls
        self.pin_window_pct = pin_window_pct

    # ------------------------------------------------------------------
    def analyze(
        self,
        chain: pd.DataFrame,
        spot: float,
        expiry: date,
        ticker: str = "",
        dividend_yield: float = 0.0,
        as_of: datetime | None = None,
    ) -> MarketStructure:
        """Produce a :class:`MarketStructure` from a single-expiry chain.

        ``chain`` must contain at minimum:
          * ``strike`` (float)
          * ``option_type`` ('C'/'P' or 'call'/'put')
          * ``open_interest`` (int)
          * ``implied_vol`` (float, decimal)

        Optional columns (used when present, otherwise reconstructed):
          * ``delta``, ``gamma`` — reused if finite; otherwise recomputed
          * ``expiration`` — used only for the returned ``expiry`` if
            ``expiry`` argument is missing
        """
        ms = MarketStructure(
            ticker=ticker or str(chain.get("ticker", pd.Series(["?"])).iloc[0]) if len(chain) else ticker,
            as_of=as_of or datetime.now(timezone.utc).replace(tzinfo=None),
            spot=float(spot),
            expiry=expiry,
            assumption=self.assumption,
        )

        if chain is None or len(chain) == 0 or spot <= 0:
            ms.regime = "neutral"
            ms.confidence = 0.0
            ms.notes = "empty_chain_or_zero_spot"
            return ms

        # Normalise columns
        df = chain.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"strike", "option_type", "open_interest", "implied_vol"}
        missing = required - set(df.columns)
        if missing:
            ms.notes = f"missing_columns:{','.join(sorted(missing))}"
            ms.regime = "neutral"
            return ms

        # Drop rows with invalid IV or non-positive strikes
        df = df[
            (df["strike"] > 0)
            & df["implied_vol"].notna()
            & (df["implied_vol"] > 0)
            & (df["implied_vol"] < 5.0)
            & df["open_interest"].notna()
        ].copy()
        if df.empty:
            ms.notes = "no_valid_rows"
            ms.regime = "neutral"
            return ms

        # Normalise option_type to 'C' / 'P'
        df["option_type"] = (
            df["option_type"].astype(str).str.upper().str.strip().str[0]
        )
        df = df[df["option_type"].isin(["C", "P"])]
        if df.empty:
            ms.notes = "no_call_or_put_rows"
            ms.regime = "neutral"
            return ms

        # Compute time to expiry in years
        T = max((expiry - datetime.now(timezone.utc).replace(tzinfo=None).date()).days, 1) / 365.0

        # Per-strike aggregation: group strikes and collect call + put rows.
        per_strike = self._per_strike_exposures(df, spot, T, dividend_yield)
        ms.per_strike = per_strike
        ms.n_strikes = len(per_strike)
        ms.n_calls = int((df["option_type"] == "C").sum())
        ms.n_puts = int((df["option_type"] == "P").sum())

        if not per_strike:
            ms.notes = "no_per_strike_after_aggregation"
            ms.regime = "neutral"
            return ms

        # Aggregates
        ms.gex_total = float(sum(p.net_gex for p in per_strike))
        ms.dex_total = float(sum(p.net_dex for p in per_strike))
        ms.vanna_total = float(sum(p.net_vanna for p in per_strike))
        ms.charm_total = float(sum(p.net_charm for p in per_strike))

        # Walls
        call_walls, put_walls = self._find_walls(per_strike, spot)
        ms.call_walls = call_walls
        ms.put_walls = put_walls
        ms.nearest_call_wall = self._nearest_wall(call_walls, spot, above=True)
        ms.nearest_put_wall = self._nearest_wall(put_walls, spot, above=False)

        # Flip level
        ms.flip_level = self._solve_flip_level(df, spot, T, dividend_yield)
        if ms.flip_level is not None and spot > 0:
            ms.flip_distance_pct = (ms.flip_level - spot) / spot
        else:
            ms.flip_distance_pct = None

        # Pinning zones
        ms.pinning_zones = self._detect_pinning_zones(per_strike, spot)

        # Regime classification + confidence
        ms.regime = self._classify_regime(
            ms.gex_total, ms.flip_distance_pct
        )
        ms.confidence = self._regime_confidence(per_strike, ms.regime, ms.gex_total)

        return ms

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _per_strike_exposures(
        self,
        df: pd.DataFrame,
        spot: float,
        T: float,
        q: float,
    ) -> list[PerStrikeExposure]:
        """Build per-strike exposure records under the chosen assumption."""
        call_sign, put_sign = self._signs()
        out: list[PerStrikeExposure] = []
        # One-pass per strike — iterate unique strikes sorted ascending
        for strike, sub in df.groupby("strike"):
            strike_f = float(strike)
            # Split calls / puts within this strike
            calls = sub[sub["option_type"] == "C"]
            puts = sub[sub["option_type"] == "P"]

            call_oi = int(calls["open_interest"].sum()) if len(calls) else 0
            put_oi = int(puts["open_interest"].sum()) if len(puts) else 0

            call_gamma, call_delta, call_vanna, call_charm = self._greek_row(
                calls, "call", strike_f, spot, T, q
            )
            put_gamma, put_delta, put_vanna, put_charm = self._greek_row(
                puts, "put", strike_f, spot, T, q
            )

            # Dollar-per-1%-move convention:
            # gamma is d²P/dS², so dollar GEX per 1% spot move is
            #    0.5 * gamma * (0.01 * spot)² * 100 * OI
            # but the industry convention is the simpler linear form:
            #    gamma * OI * 100 * spot² * 0.01
            # which expresses dealer *delta* change per 1% move (what they
            # must actually hedge). We use the simpler convention.
            call_gex = call_sign * call_gamma * call_oi * 100 * spot * spot * 0.01
            put_gex = put_sign * put_gamma * put_oi * 100 * spot * spot * 0.01

            net_dex = (
                call_sign * call_delta * call_oi * 100 * spot
                + put_sign * put_delta * put_oi * 100 * spot
            )
            net_vanna = (
                call_sign * call_vanna * call_oi * 100
                + put_sign * put_vanna * put_oi * 100
            )
            net_charm = (
                call_sign * call_charm * call_oi * 100
                + put_sign * put_charm * put_oi * 100
            )

            out.append(
                PerStrikeExposure(
                    strike=strike_f,
                    call_oi=call_oi,
                    put_oi=put_oi,
                    call_gamma=call_gamma,
                    put_gamma=put_gamma,
                    call_delta=call_delta,
                    put_delta=put_delta,
                    call_gex=call_gex,
                    put_gex=put_gex,
                    net_gex=call_gex + put_gex,
                    net_dex=net_dex,
                    net_vanna=net_vanna,
                    net_charm=net_charm,
                )
            )

        out.sort(key=lambda p: p.strike)
        return out

    def _signs(self) -> tuple[int, int]:
        """Return (call_sign, put_sign) for the dealer assumption.

        LONG_CALLS_SHORT_PUTS ⇒ (+1, -1).
        SHORT_BOTH            ⇒ (-1, -1).
        """
        if self.assumption == DealerAssumption.LONG_CALLS_SHORT_PUTS:
            return 1, -1
        if self.assumption == DealerAssumption.SHORT_BOTH:
            return -1, -1
        return 1, -1  # fall-through

    def _greek_row(
        self,
        rows: pd.DataFrame,
        option_type: Literal["call", "put"],
        strike: float,
        spot: float,
        T: float,
        q: float,
    ) -> tuple[float, float, float, float]:
        """Return (gamma, delta, vanna, charm) averaged across the rows.

        We reuse stored delta/gamma when finite; we ALWAYS recompute
        vanna/charm from BSM because Bloomberg only stores first-order
        Greeks. When stored Greeks are missing we reconstruct
        everything via :func:`black_scholes_all_greeks`.
        """
        if len(rows) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Pick one representative row (usually there's only one per
        # strike × type anyway). Average IV across rows if multiple.
        iv = float(rows["implied_vol"].mean())
        if iv <= 0 or iv >= 5.0:
            return 0.0, 0.0, 0.0, 0.0

        stored_gamma = None
        stored_delta = None
        if "gamma" in rows.columns and rows["gamma"].notna().any():
            stored_gamma = float(rows["gamma"].dropna().iloc[0])
        if "delta" in rows.columns and rows["delta"].notna().any():
            stored_delta = float(rows["delta"].dropna().iloc[0])

        try:
            greeks = black_scholes_all_greeks(
                S=spot,
                K=strike,
                T=max(T, 1e-6),
                r=self.risk_free_rate,
                sigma=iv,
                option_type=option_type,
                q=q,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("BSM greeks failed for strike %s: %s", strike, exc)
            return (
                stored_gamma or 0.0,
                stored_delta or 0.0,
                0.0,
                0.0,
            )

        # Prefer stored first-order Greeks when available and finite;
        # fall back to BSM. Second-order Greeks always from BSM.
        gamma = stored_gamma if stored_gamma and np.isfinite(stored_gamma) else greeks["gamma"]
        delta = stored_delta if stored_delta is not None and np.isfinite(stored_delta) else greeks["delta"]
        vanna = float(greeks.get("vanna", 0.0) or 0.0)
        charm = float(greeks.get("charm", 0.0) or 0.0)
        return float(gamma), float(delta), vanna, charm

    # ------------------------------------------------------------------
    def _find_walls(
        self,
        per_strike: list[PerStrikeExposure],
        spot: float,
    ) -> tuple[list[GammaWall], list[GammaWall]]:
        """Find the top call and put gamma walls.

        Call walls = strikes with the largest POSITIVE net_gex (dealers
        long gamma concentration above spot → pinning resistance).
        Put walls  = strikes with the largest NEGATIVE net_gex (dealers
        short gamma concentration → gamma cliff / support).
        """
        calls = [p for p in per_strike if p.net_gex > 0]
        puts = [p for p in per_strike if p.net_gex < 0]
        calls.sort(key=lambda p: p.net_gex, reverse=True)
        puts.sort(key=lambda p: p.net_gex)

        def to_wall(p: PerStrikeExposure, side: Literal["call", "put"]) -> GammaWall:
            return GammaWall(
                strike=p.strike,
                distance_pct=(p.strike - spot) / spot if spot > 0 else 0.0,
                net_gex=p.net_gex,
                side=side,
            )

        top_calls = [to_wall(p, "call") for p in calls[: self.top_walls]]
        top_puts = [to_wall(p, "put") for p in puts[: self.top_walls]]
        return top_calls, top_puts

    def _nearest_wall(
        self,
        walls: list[GammaWall],
        spot: float,
        above: bool,
    ) -> GammaWall | None:
        """Return the nearest wall above / below spot within ``near_wall_pct``."""
        if not walls:
            return None
        filtered = [
            w
            for w in walls
            if (w.strike >= spot if above else w.strike <= spot)
            and abs(w.distance_pct) <= self.near_wall_pct
        ]
        if not filtered:
            return None
        filtered.sort(key=lambda w: abs(w.distance_pct))
        return filtered[0]

    # ------------------------------------------------------------------
    def _solve_flip_level(
        self,
        df: pd.DataFrame,
        spot: float,
        T: float,
        q: float,
    ) -> float | None:
        """Find the spot at which total GEX crosses zero via bisection.

        The search range is [0.7*spot, 1.3*spot] in 60 steps for the
        sign-change scan. When no sign change is found we return None
        (meaning the regime is unambiguously long or short gamma
        across the entire investigated range).
        """
        call_sign, put_sign = self._signs()

        def total_gex(s: float) -> float:
            total = 0.0
            for _, row in df.iterrows():
                iv = float(row["implied_vol"])
                if iv <= 0 or iv >= 5.0:
                    continue
                K = float(row["strike"])
                oi = float(row.get("open_interest", 0) or 0)
                opt_type = "call" if row["option_type"] == "C" else "put"
                try:
                    g = black_scholes_all_greeks(
                        S=s, K=K, T=max(T, 1e-6), r=self.risk_free_rate,
                        sigma=iv, option_type=opt_type, q=q,
                    )
                except Exception:
                    continue
                gamma = float(g.get("gamma", 0.0) or 0.0)
                sign = call_sign if opt_type == "call" else put_sign
                total += sign * gamma * oi * 100 * s * s * 0.01
            return total

        # Scan for a sign change
        lo = 0.7 * spot
        hi = 1.3 * spot
        n = 30
        xs = np.linspace(lo, hi, n)
        ys = np.array([total_gex(float(x)) for x in xs])

        sign_changes = np.where(np.diff(np.sign(ys)) != 0)[0]
        if len(sign_changes) == 0:
            return None

        # Pick the sign change closest to the current spot
        best_idx = sign_changes[
            np.argmin(np.abs(xs[sign_changes] - spot))
        ]
        a, b = xs[best_idx], xs[best_idx + 1]
        fa, fb = ys[best_idx], ys[best_idx + 1]

        # Linear interpolation within the bracket
        if fb == fa:
            return float((a + b) / 2)
        flip = a + (0 - fa) * (b - a) / (fb - fa)
        return float(flip)

    # ------------------------------------------------------------------
    def _detect_pinning_zones(
        self,
        per_strike: list[PerStrikeExposure],
        spot: float,
    ) -> list[float]:
        """Return strikes within the pin window with top-quartile OI*|gamma|."""
        window = self.pin_window_pct * spot
        nearby = [p for p in per_strike if abs(p.strike - spot) <= window]
        if not nearby:
            return []
        scores = [
            (p.strike, (p.call_oi + p.put_oi) * (abs(p.call_gamma) + abs(p.put_gamma)))
            for p in nearby
        ]
        if not scores:
            return []
        score_vals = np.array([s[1] for s in scores])
        if score_vals.max() <= 0:
            return []
        threshold = np.percentile(score_vals, 75)
        return [s[0] for s in scores if s[1] >= threshold and s[1] > 0]

    # ------------------------------------------------------------------
    def _classify_regime(
        self,
        gex_total: float,
        flip_distance_pct: float | None,
    ) -> Literal[
        "long_gamma_dampening",
        "short_gamma_amplifying",
        "near_flip",
        "neutral",
    ]:
        """Classify the regime from total GEX and flip proximity."""
        # Near-flip dominates: even a big GEX is uncertain if we're
        # within a 1% band of the flip level (regime can flip intraday).
        if (
            flip_distance_pct is not None
            and abs(flip_distance_pct) < self.flip_neighborhood_pct
        ):
            return "near_flip"

        if gex_total > 0:
            return "long_gamma_dampening"
        if gex_total < 0:
            return "short_gamma_amplifying"
        return "neutral"

    def _regime_confidence(
        self,
        per_strike: list[PerStrikeExposure],
        regime: str,
        gex_total: float,
    ) -> float:
        """Return a 0..1 confidence score for the regime label.

        Based on the ratio of same-sign GEX concentration to total
        absolute GEX. A chain where 95% of the GEX comes from strikes
        of the dominant sign gets confidence 0.95.
        """
        if regime == "neutral" or not per_strike:
            return 0.0
        if regime == "near_flip":
            return 0.50  # moderate by definition — regime is transitional
        sign = 1 if gex_total > 0 else -1
        total_abs = sum(abs(p.net_gex) for p in per_strike)
        if total_abs <= 0:
            return 0.0
        aligned = sum(abs(p.net_gex) for p in per_strike if np.sign(p.net_gex) == sign)
        return float(min(1.0, aligned / total_abs))


# ----------------------------------------------------------------------
# Regime → multiplier
# ----------------------------------------------------------------------
def dealer_regime_multiplier(ms: MarketStructure | None) -> float:
    """Return a scalar in [0.70, 1.05] for the EV engine.

    Asymmetric by design:
      * long_gamma_dampening: up to 1.05 (small boost — dealers dampen
        realized vol, premium sellers benefit)
      * short_gamma_amplifying: down to 0.70 (meaningful cut — dealers
        amplify moves, breach risk rises)
      * near_flip: 0.85 (regime is transitional, uncertain)
      * neutral / missing: 1.00 (no adjustment)

    Confidence scales the distance from 1.0. Low-confidence regimes
    move less.
    """
    if ms is None:
        return 1.0
    conf = max(0.0, min(1.0, float(ms.confidence)))
    if ms.regime == "long_gamma_dampening":
        # 1.00 → 1.05 linearly in confidence
        return 1.0 + 0.05 * conf
    if ms.regime == "short_gamma_amplifying":
        # 1.00 → 0.70 linearly in confidence (30% cut at full confidence)
        return 1.0 - 0.30 * conf
    if ms.regime == "near_flip":
        # Flat 0.85 — near-flip is a warning regardless of confidence
        return 0.85
    return 1.0
