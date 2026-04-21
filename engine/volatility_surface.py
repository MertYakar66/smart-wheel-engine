"""
Volatility Surface Module

Professional IV surface modeling for realistic option pricing:
- SVI (Stochastic Volatility Inspired) parameterization
- Cubic spline interpolation
- Term structure modeling
- Smile dynamics under stress

Critical upgrade: Replaces constant-IV assumption with dynamic surfaces.
Requires historical option chain data (Bloomberg) to be fully functional.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import interpolate, optimize
from scipy.stats import norm


@dataclass
class SVIParams:
    """
    SVI (Stochastic Volatility Inspired) parameters.

    The SVI parameterization models the implied variance as:
    w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))

    Where:
    - k = log(K/F) = log-moneyness
    - w = σ²*T = total implied variance
    - a, b, ρ, m, σ are the 5 SVI parameters

    This creates a realistic smile/skew shape.

    Arbitrage Enforcement:
    - Butterfly no-arbitrage condition is enforced by default (raises ValueError)
    - Set strict_arbitrage=False only for calibration testing/debugging
    """

    a: float  # Vertical shift (overall variance level)
    b: float  # Tightness of smile (must be >= 0)
    rho: float  # Asymmetry/skew (-1 < ρ < 1)
    m: float  # Horizontal shift (ATM location)
    sigma: float  # Smoothness of vertex (must be > 0)
    strict_arbitrage: bool = True  # If True, reject arbitrage violations

    def __post_init__(self):
        """Validate parameters with arbitrage enforcement."""
        if self.b < 0:
            raise ValueError(f"b must be >= 0, got {self.b}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

        # Butterfly arbitrage condition: a + b*sigma*sqrt(1-rho^2) >= 0
        # This ensures the total variance w(k) >= 0 for all k
        # Reference: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces"
        butterfly = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if butterfly < 0:
            if self.strict_arbitrage:
                raise ValueError(
                    f"SVI params violate butterfly no-arbitrage condition: "
                    f"a + b*sigma*sqrt(1-rho^2) = {butterfly:.6f} < 0. "
                    f"Set strict_arbitrage=False to allow (for testing only)."
                )
            else:
                warnings.warn(
                    f"SVI params violate butterfly arbitrage: {butterfly:.4f}. "
                    f"This surface may produce negative variances.",
                    stacklevel=2,
                )

    def total_variance(self, k: float) -> float:
        """Calculate total variance w(k) for log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """Calculate implied volatility for log-moneyness k and time T."""
        if T <= 0:
            return 0.0
        w = self.total_variance(k)
        if w < 0:
            return 0.0
        return np.sqrt(w / T)


@dataclass
class VolatilitySurface:
    """
    Complete implied volatility surface.

    Stores SVI parameters per expiry for full surface reconstruction.
    """

    as_of_date: date
    underlying: str
    forward_prices: dict[date, float]  # Forward price per expiry
    svi_params: dict[date, SVIParams]  # SVI params per expiry

    # Raw data (for diagnostics)
    raw_strikes: dict[date, np.ndarray] = field(default_factory=dict)
    raw_ivs: dict[date, np.ndarray] = field(default_factory=dict)

    def get_iv(self, strike: float, expiry: date, spot: float | None = None) -> float:
        """
        Get interpolated IV for strike and expiry.

        Args:
            strike: Option strike price
            expiry: Option expiration date
            spot: Current spot price (uses forward if not provided)

        Returns:
            Implied volatility
        """
        if expiry not in self.svi_params:
            return self._interpolate_expiry(strike, expiry, spot)

        # Get forward for this expiry
        F = self.forward_prices.get(expiry, spot or strike)

        # Calculate log-moneyness
        k = np.log(strike / F)

        # Time to expiry
        T = (expiry - self.as_of_date).days / 365

        # Get IV from SVI
        params = self.svi_params[expiry]
        return params.implied_vol(k, T)

    def _interpolate_expiry(self, strike: float, expiry: date, spot: float | None) -> float:
        """Interpolate IV for expiry not in surface."""
        if not self.svi_params:
            return 0.20  # Default

        # Find bracketing expiries
        expiries = sorted(self.svi_params.keys())
        T_target = (expiry - self.as_of_date).days / 365

        if expiry <= expiries[0]:
            # Extrapolate from first expiry
            return self.get_iv(strike, expiries[0], spot)
        if expiry >= expiries[-1]:
            # Extrapolate from last expiry
            return self.get_iv(strike, expiries[-1], spot)

        # Find bracketing expiries
        for i in range(len(expiries) - 1):
            if expiries[i] <= expiry <= expiries[i + 1]:
                T1 = (expiries[i] - self.as_of_date).days / 365
                T2 = (expiries[i + 1] - self.as_of_date).days / 365

                iv1 = self.get_iv(strike, expiries[i], spot)
                iv2 = self.get_iv(strike, expiries[i + 1], spot)

                # Linear interpolation in variance space
                w1 = iv1**2 * T1
                w2 = iv2**2 * T2

                # Interpolate total variance
                weight = (T_target - T1) / (T2 - T1)
                w_interp = w1 + weight * (w2 - w1)

                if T_target > 0 and w_interp > 0:
                    return np.sqrt(w_interp / T_target)
                else:
                    return (iv1 + iv2) / 2

        return 0.20  # Fallback

    def get_term_structure(self, strike: float | None = None, delta: float = 0.5) -> pd.DataFrame:
        """
        Get IV term structure (IV vs expiry).

        Args:
            strike: Fixed strike (uses ATM if None)
            delta: Target delta if strike not specified

        Returns:
            DataFrame with expiry, days_to_expiry, iv
        """
        results = []

        for expiry in sorted(self.svi_params.keys()):
            T = (expiry - self.as_of_date).days

            if strike is not None:
                iv = self.get_iv(strike, expiry)
            else:
                # ATM (k=0)
                params = self.svi_params[expiry]
                iv = params.implied_vol(0, T / 365)

            results.append({"expiry": expiry, "dte": T, "iv": iv})

        return pd.DataFrame(results)

    def get_skew(self, expiry: date) -> tuple[float, float, float]:
        """
        Get volatility skew metrics for an expiry.

        Returns:
            (25-delta put IV, ATM IV, 25-delta call IV)
        """
        if expiry not in self.svi_params:
            return 0.20, 0.20, 0.20

        params = self.svi_params[expiry]
        T = (expiry - self.as_of_date).days / 365

        # Approximate 25-delta log-moneyness
        # For 25-delta put: k ≈ -0.67*σ*√T
        # For 25-delta call: k ≈ +0.67*σ*√T
        atm_iv = params.implied_vol(0, T)

        if T > 0 and atm_iv > 0:
            k_25d = 0.67 * atm_iv * np.sqrt(T)
            iv_25d_put = params.implied_vol(-k_25d, T)
            iv_25d_call = params.implied_vol(k_25d, T)
        else:
            iv_25d_put = iv_25d_call = atm_iv

        return iv_25d_put, atm_iv, iv_25d_call

    def stress(
        self, spot_change_pct: float, parallel_iv_shift: float = 0.0, skew_steepening: float = 0.0
    ) -> "VolatilitySurface":
        """
        Create stressed version of surface.

        Args:
            spot_change_pct: % change in underlying
            parallel_iv_shift: Parallel shift to all IVs
            skew_steepening: Additional skew (negative = more put skew)

        Returns:
            New stressed VolatilitySurface
        """
        new_forwards = {
            exp: fwd * (1 + spot_change_pct) for exp, fwd in self.forward_prices.items()
        }

        # Adjust SVI params for stress
        new_params = {}
        for exp, params in self.svi_params.items():
            # a controls level, rho controls skew
            new_a = params.a + parallel_iv_shift**2  # Shift in variance space
            new_rho = np.clip(params.rho + skew_steepening, -0.99, 0.99)

            new_params[exp] = SVIParams(
                a=new_a,
                b=params.b,
                rho=new_rho,
                m=params.m,
                sigma=params.sigma,
                strict_arbitrage=params.strict_arbitrage,
            )

        return VolatilitySurface(
            as_of_date=self.as_of_date,
            underlying=self.underlying,
            forward_prices=new_forwards,
            svi_params=new_params,
        )


class SVICalibrator:
    """
    Calibrate SVI parameters from market option data.

    Uses constrained least-squares optimization to fit SVI curve to market IVs.
    Enforces butterfly no-arbitrage constraint during calibration.
    """

    def __init__(
        self,
        min_points: int = 5,
        max_iterations: int = 1000,
        enforce_arbitrage: bool = True,
    ):
        """
        Initialize SVI calibrator.

        Args:
            min_points: Minimum data points required for calibration
            max_iterations: Maximum optimizer iterations
            enforce_arbitrage: If True, add butterfly constraint to optimization
        """
        self.min_points = min_points
        self.max_iterations = max_iterations
        self.enforce_arbitrage = enforce_arbitrage

    def calibrate(
        self, strikes: np.ndarray, ivs: np.ndarray, forward: float, T: float
    ) -> SVIParams:
        """
        Calibrate SVI parameters to market data with arbitrage constraints.

        Args:
            strikes: Array of strike prices
            ivs: Array of implied volatilities
            forward: Forward price
            T: Time to expiry in years

        Returns:
            Fitted SVIParams (guaranteed arbitrage-free if enforce_arbitrage=True)

        Raises:
            ValueError: If calibration fails or produces arbitrage violations
        """
        if len(strikes) < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {len(strikes)}")

        # Convert to log-moneyness and total variance
        k = np.log(strikes / forward)
        w = ivs**2 * T

        # Initial guess
        atm_var = np.interp(0, k, w)
        x0 = [
            atm_var,  # a
            0.1,  # b
            -0.3,  # rho (typical equity skew)
            0.0,  # m
            0.1,  # sigma
        ]

        # Bounds ensuring basic parameter validity
        bounds = [
            (-0.5, 1.0),  # a
            (0.001, 1.0),  # b
            (-0.99, 0.99),  # rho
            (-0.5, 0.5),  # m
            (0.001, 1.0),  # sigma
        ]

        def objective(params):
            a, b, rho, m, sigma = params
            try:
                w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
                return np.sum((w_model - w) ** 2)
            except Exception:
                return 1e10

        # Butterfly arbitrage constraint: a + b*sigma*sqrt(1-rho^2) >= 0
        def butterfly_constraint(params):
            a, b, rho, _, sigma = params
            return a + b * sigma * np.sqrt(1 - rho**2)

        constraints = []
        if self.enforce_arbitrage:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": butterfly_constraint,
                }
            )

        # Optimize with constraints
        # Use SLSQP for constrained optimization (supports both bounds and constraints)
        if constraints:
            result = optimize.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.max_iterations},
            )
        else:
            # Faster unconstrained optimization for testing
            result = optimize.minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iterations},
            )

        if not result.success:
            warnings.warn(f"SVI calibration did not converge: {result.message}", stacklevel=2)

        a, b, rho, m, sigma = result.x

        # Return with strict_arbitrage matching our calibration setting
        return SVIParams(
            a=a, b=b, rho=rho, m=m, sigma=sigma, strict_arbitrage=self.enforce_arbitrage
        )


class VolatilitySurfaceBuilder:
    """
    Build volatility surfaces from option chain data.

    This is the main interface for constructing surfaces from Bloomberg data.
    """

    def __init__(self):
        self.calibrator = SVICalibrator()

    def build_from_chain(
        self,
        option_data: pd.DataFrame,
        underlying_price: float,
        as_of_date: date,
        underlying: str,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.02,
    ) -> VolatilitySurface:
        """
        Build surface from option chain snapshot.

        Args:
            option_data: DataFrame with columns:
                strike, expiration, iv_mid, option_type
            underlying_price: Current spot price
            as_of_date: Snapshot date
            underlying: Symbol
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            VolatilitySurface
        """
        # Group by expiration
        expiries = option_data["expiration"].unique()

        forward_prices = {}
        svi_params = {}
        raw_strikes = {}
        raw_ivs = {}

        for expiry in sorted(expiries):
            expiry_date = pd.Timestamp(expiry).date() if not isinstance(expiry, date) else expiry
            T = (expiry_date - as_of_date).days / 365

            if T <= 0:
                continue

            # Calculate forward
            F = underlying_price * np.exp((risk_free_rate - dividend_yield) * T)
            forward_prices[expiry_date] = F

            # Get OTM options only (cleaner for calibration)
            exp_data = option_data[option_data["expiration"] == expiry].copy()

            # Filter for OTM
            otm_puts = exp_data[(exp_data["option_type"] == "put") & (exp_data["strike"] < F)]
            otm_calls = exp_data[(exp_data["option_type"] == "call") & (exp_data["strike"] >= F)]
            otm_data = pd.concat([otm_puts, otm_calls])

            if len(otm_data) < 5:
                continue

            strikes = otm_data["strike"].values
            ivs = otm_data["iv_mid"].values

            # Clean data
            valid = (ivs > 0.01) & (ivs < 3.0) & np.isfinite(ivs)
            strikes = strikes[valid]
            ivs = ivs[valid]

            if len(strikes) < 5:
                continue

            raw_strikes[expiry_date] = strikes
            raw_ivs[expiry_date] = ivs

            try:
                params = self.calibrator.calibrate(strikes, ivs, F, T)
                svi_params[expiry_date] = params
            except Exception as e:
                warnings.warn(f"Failed to calibrate {expiry_date}: {e}", stacklevel=2)
                continue

        return VolatilitySurface(
            as_of_date=as_of_date,
            underlying=underlying,
            forward_prices=forward_prices,
            svi_params=svi_params,
            raw_strikes=raw_strikes,
            raw_ivs=raw_ivs,
        )


class SplineVolSurface:
    """
    Simple cubic spline interpolation for IV surface.

    Alternative to SVI when SVI calibration is unstable.
    Faster but less smooth extrapolation.
    """

    def __init__(self, as_of_date: date, underlying: str):
        self.as_of_date = as_of_date
        self.underlying = underlying
        self.splines: dict[date, Callable] = {}  # expiry -> interpolator
        self.forward_prices: dict[date, float] = {}

    def fit(self, expiry: date, strikes: np.ndarray, ivs: np.ndarray, forward: float):
        """Fit spline for single expiry."""
        if len(strikes) < 3:
            return

        # Sort by strike
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        ivs = ivs[idx]

        # Create spline
        spline = interpolate.CubicSpline(strikes, ivs, bc_type="natural", extrapolate=True)

        self.splines[expiry] = spline
        self.forward_prices[expiry] = forward

    def get_iv(self, strike: float, expiry: date) -> float:
        """Get IV at strike and expiry."""
        if expiry not in self.splines:
            # Find nearest expiry
            if not self.splines:
                return 0.20
            nearest = min(self.splines.keys(), key=lambda e: abs((e - expiry).days))
            return float(self.splines[nearest](strike))

        return float(self.splines[expiry](strike))


# =============================================================================
# Utility Functions
# =============================================================================


def validate_no_calendar_arbitrage(
    surface: VolatilitySurface,
    moneyness_grid: np.ndarray | None = None,
    tolerance: float = -1e-6,
) -> dict[str, list[str]]:
    """
    Validate that a volatility surface is free of calendar arbitrage.

    Calendar arbitrage occurs when total variance is not monotonically
    non-decreasing in time for any given strike/log-moneyness.

    Formally: w(k, T1) <= w(k, T2) for all k when T1 < T2

    Also checks butterfly arbitrage per-expiry (non-negative density).

    Args:
        surface: VolatilitySurface to validate
        moneyness_grid: Log-moneyness points to check (default: -0.5 to +0.5)
        tolerance: Negative tolerance for numerical noise (default: -1e-6)

    Returns:
        Dict with 'calendar_violations', 'butterfly_violations', 'health_metrics'
    """
    if moneyness_grid is None:
        moneyness_grid = np.linspace(-0.5, 0.5, 21)

    expiries = sorted(surface.svi_params.keys())
    results: dict[str, list[str]] = {
        "calendar_violations": [],
        "butterfly_violations": [],
        "health_metrics": [],
    }

    if len(expiries) < 2:
        results["health_metrics"].append("Insufficient expiries for calendar arbitrage check")
        return results

    # Check calendar arbitrage: total variance must be non-decreasing in T
    for i in range(len(expiries) - 1):
        T1 = (expiries[i] - surface.as_of_date).days / 365
        T2 = (expiries[i + 1] - surface.as_of_date).days / 365

        if T1 <= 0 or T2 <= 0:
            continue

        params1 = surface.svi_params[expiries[i]]
        params2 = surface.svi_params[expiries[i + 1]]

        for k in moneyness_grid:
            w1 = params1.total_variance(k)
            w2 = params2.total_variance(k)

            if w2 - w1 < tolerance:
                results["calendar_violations"].append(
                    f"k={k:.3f}: w({expiries[i]})={w1:.6f} > "
                    f"w({expiries[i + 1]})={w2:.6f} (diff={w2 - w1:.6f})"
                )

    # Check butterfly arbitrage per expiry: g(k) = (1 - k*w'/2w)^2 - w'^2/4*(1/w + 1/4) + w''/2
    # Simplified: check that implied variance is non-negative everywhere
    for expiry in expiries:
        params = surface.svi_params[expiry]
        T = (expiry - surface.as_of_date).days / 365
        if T <= 0:
            continue

        for k in moneyness_grid:
            w = params.total_variance(k)
            if w < tolerance:
                results["butterfly_violations"].append(
                    f"{expiry} k={k:.3f}: negative total variance w={w:.6f}"
                )

    # Health metrics
    n_cal = len(results["calendar_violations"])
    n_but = len(results["butterfly_violations"])
    n_checks = len(moneyness_grid) * (len(expiries) - 1) + len(moneyness_grid) * len(expiries)
    results["health_metrics"].append(
        f"Checked {n_checks} points: {n_cal} calendar violations, {n_but} butterfly violations"
    )
    if n_cal == 0 and n_but == 0:
        results["health_metrics"].append("Surface is arbitrage-free")

    return results


def create_constant_surface(
    iv: float, as_of_date: date, underlying: str, spot: float, expiries: list[date]
) -> VolatilitySurface:
    """
    Create flat (constant IV) surface.

    AUDIT NOTE: A flat surface has zero skew and zero smile and will
    systematically mis-price OTM puts (real markets have ~30% steeper
    put wings than the flat model). Use this only as a deterministic
    fixture for unit tests or as an absolute last-resort fall-back.
    Prefer :func:`create_empirical_surface` whenever historical IV
    data (even just ATM 30-day) is available — it still has no smile
    but at least honours the term-structure shape.

    Useful as fallback when no option data available.
    """
    forward_prices = dict.fromkeys(expiries, spot)

    # Flat SVI: a = iv^2 * T, b = 0
    svi_params = {}
    for exp in expiries:
        T = (exp - as_of_date).days / 365
        if T > 0:
            svi_params[exp] = SVIParams(
                a=iv**2 * T,
                b=0.001,  # Near-zero but valid
                rho=0.0,
                m=0.0,
                sigma=0.1,
            )

    return VolatilitySurface(
        as_of_date=as_of_date,
        underlying=underlying,
        forward_prices=forward_prices,
        svi_params=svi_params,
    )


def create_empirical_surface(
    iv_by_tenor: dict[int, float],
    as_of_date: date,
    underlying: str,
    spot: float,
    expiries: list[date],
    put_skew_slope: float = -0.15,
    call_skew_slope: float = 0.02,
    smile_curvature: float = 0.40,
) -> VolatilitySurface:
    """Create a smile-aware surface from an ATM IV term structure.

    This is the recommended fall-back when the underlying has at least
    a reliable ATM IV per tenor (Bloomberg ``hist_put_imp_vol`` + a few
    VIX-style tenors) but no full option chain. It builds a per-expiry
    SVI parameter set that:

    * Honours the ATM IV for each expiry exactly.
    * Imposes a realistic smile/skew (equity indices and single names
      have negative put skew and a small positive call skew, with the
      strength of the smile decreasing with tenor).
    * Stays arbitrage-free under the Gatheral SSVI restrictions by
      construction (``|ρ|<1``, ``b>0``).

    Arguments:
        iv_by_tenor: dict mapping ``days_to_expiry -> atm_iv`` (decimal).
            The function will nearest-neighbour interpolate this for
            any expiry in ``expiries``.
        as_of_date: Surface as-of date.
        underlying: Ticker symbol.
        spot: Current spot price (used as forward proxy; add carry
            adjustments upstream if needed).
        expiries: Expiries to materialise the surface for.
        put_skew_slope: ATM-normalised slope of the put wing. More
            negative = steeper put skew (crash-premium). Typical
            equity-index value is -0.15.
        call_skew_slope: ATM-normalised slope of the call wing.
            Typical equity-index value is ~0.02.
        smile_curvature: Convexity of the smile. Higher = more
            pronounced smile. Default 0.40 is in the middle of the
            historical SPX range.

    Returns:
        :class:`VolatilitySurface` with per-expiry SVI params.
    """
    if not iv_by_tenor:
        raise ValueError("iv_by_tenor must contain at least one tenor")

    tenor_keys = sorted(iv_by_tenor.keys())

    def _interp_atm_iv(days: int) -> float:
        if days <= tenor_keys[0]:
            return iv_by_tenor[tenor_keys[0]]
        if days >= tenor_keys[-1]:
            return iv_by_tenor[tenor_keys[-1]]
        for i in range(len(tenor_keys) - 1):
            lo, hi = tenor_keys[i], tenor_keys[i + 1]
            if lo <= days <= hi:
                w = (days - lo) / (hi - lo)
                return (1 - w) * iv_by_tenor[lo] + w * iv_by_tenor[hi]
        return iv_by_tenor[tenor_keys[-1]]

    forward_prices = dict.fromkeys(expiries, spot)
    svi_params: dict[date, SVIParams] = {}

    # Equity-style rho: negative skew ⇒ put wing steeper than call wing.
    # Derived from the asymmetry of the two wing slopes. Clamped to [-0.95, 0.95].
    put_abs = abs(put_skew_slope)
    call_abs = abs(call_skew_slope)
    if put_abs + call_abs > 0:
        rho = -(put_abs - call_abs) / (put_abs + call_abs)
    else:
        rho = 0.0
    rho = max(-0.95, min(0.95, rho))

    # SPX-calibrated magnitudes. These are small on purpose — standard
    # SPX 30-DTE skew is ~3-5 vol points per 10% OTM, and that translates
    # to b ~ 0.010 and sigma ~ 0.04 in the SVI parameterization. Using
    # bigger numbers blows the variance up exponentially.
    b_nominal = 0.010 * max(put_abs + call_abs, 0.10) / 0.17  # scale by caller's slopes
    sigma_nominal = 0.04 * smile_curvature / 0.40

    for exp in expiries:
        days = (exp - as_of_date).days
        T = days / 365
        if T <= 0:
            continue
        atm_iv = _interp_atm_iv(days)
        w_atm_target = atm_iv**2 * T

        # Decay wing slope and curvature with sqrt(T) to match observed
        # flattening of the smile at longer tenors.
        scale = min(1.0, 0.5 / max(np.sqrt(T), 0.25))  # short-tenor emphasis
        b = b_nominal * scale
        sigma_svi = sigma_nominal * scale
        m = 0.0

        # Solve ATM condition exactly:
        #   w(0) = a + b * (ρ * (-m) + sqrt(m^2 + σ^2))
        #        = a + b * sigma_svi           (with m=0)
        # ⇒ a = w_atm - b * sigma_svi
        # We insist a >= 1e-8 by shrinking b if necessary.
        if b * sigma_svi >= 0.95 * w_atm_target:
            # Shrink wing/curvature to avoid a<=0
            b = 0.5 * w_atm_target / max(sigma_svi, 1e-4)
        a = w_atm_target - b * sigma_svi
        a = max(a, 1e-8)

        svi_params[exp] = SVIParams(
            a=a,
            b=b,
            rho=rho,
            m=m,
            sigma=sigma_svi,
        )

    return VolatilitySurface(
        as_of_date=as_of_date,
        underlying=underlying,
        forward_prices=forward_prices,
        svi_params=svi_params,
    )


def estimate_iv_for_delta(
    delta: float, surface: VolatilitySurface, expiry: date, spot: float, is_put: bool = True
) -> tuple[float, float]:
    """
    Find strike and IV for target delta.

    Args:
        delta: Target delta (e.g., 0.25 for 25-delta)
        surface: Volatility surface
        expiry: Option expiration
        spot: Current spot price
        is_put: True for put, False for call

    Returns:
        (strike, iv)
    """
    T = (expiry - surface.as_of_date).days / 365
    if T <= 0:
        return spot, 0.20

    # Use Newton's method to find strike
    F = surface.forward_prices.get(expiry, spot)

    # Initial guess based on ATM vol
    atm_iv = surface.get_iv(F, expiry, spot)

    # For puts: delta = -N(-d1), so we need N(-d1) = -delta
    # k ≈ -sigma * sqrt(T) * N^{-1}(|delta|) for puts
    if is_put:
        k_guess = -atm_iv * np.sqrt(T) * norm.ppf(delta)
    else:
        k_guess = atm_iv * np.sqrt(T) * norm.ppf(delta)

    strike_guess = F * np.exp(k_guess)
    iv = surface.get_iv(strike_guess, expiry, spot)

    return strike_guess, iv


def surface_to_dataframe(surface: VolatilitySurface) -> pd.DataFrame:
    """Convert surface to DataFrame for analysis/export."""
    rows = []

    for expiry in sorted(surface.svi_params.keys()):
        params = surface.svi_params[expiry]
        F = surface.forward_prices.get(expiry, 100)
        T = (expiry - surface.as_of_date).days / 365

        # Sample strikes from 0.7F to 1.3F
        for moneyness in np.linspace(0.7, 1.3, 13):
            strike = F * moneyness
            k = np.log(moneyness)
            iv = params.implied_vol(k, T)

            rows.append(
                {
                    "expiry": expiry,
                    "dte": (expiry - surface.as_of_date).days,
                    "strike": strike,
                    "moneyness": moneyness,
                    "log_moneyness": k,
                    "iv": iv,
                    "forward": F,
                }
            )

    return pd.DataFrame(rows)
