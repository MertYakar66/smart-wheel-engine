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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from scipy import optimize, interpolate
from scipy.stats import norm
import warnings


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
    """
    a: float  # Vertical shift (overall variance level)
    b: float  # Tightness of smile (must be >= 0)
    rho: float  # Asymmetry/skew (-1 < ρ < 1)
    m: float  # Horizontal shift (ATM location)
    sigma: float  # Smoothness of vertex (must be > 0)

    def __post_init__(self):
        """Validate parameters."""
        if self.b < 0:
            raise ValueError(f"b must be >= 0, got {self.b}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        # Butterfly arbitrage condition: a + b*sigma*sqrt(1-rho^2) >= 0
        butterfly = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if butterfly < 0:
            warnings.warn(f"SVI params may violate butterfly arbitrage: {butterfly:.4f}")

    def total_variance(self, k: float) -> float:
        """Calculate total variance w(k) for log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) +
            np.sqrt((k - self.m)**2 + self.sigma**2)
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
    forward_prices: Dict[date, float]  # Forward price per expiry
    svi_params: Dict[date, SVIParams]  # SVI params per expiry

    # Raw data (for diagnostics)
    raw_strikes: Dict[date, np.ndarray] = field(default_factory=dict)
    raw_ivs: Dict[date, np.ndarray] = field(default_factory=dict)

    def get_iv(
        self,
        strike: float,
        expiry: date,
        spot: Optional[float] = None
    ) -> float:
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

    def _interpolate_expiry(
        self,
        strike: float,
        expiry: date,
        spot: Optional[float]
    ) -> float:
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

    def get_term_structure(
        self,
        strike: Optional[float] = None,
        delta: float = 0.5
    ) -> pd.DataFrame:
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

            results.append({
                'expiry': expiry,
                'dte': T,
                'iv': iv
            })

        return pd.DataFrame(results)

    def get_skew(self, expiry: date) -> Tuple[float, float, float]:
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
        self,
        spot_change_pct: float,
        parallel_iv_shift: float = 0.0,
        skew_steepening: float = 0.0
    ) -> 'VolatilitySurface':
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
            exp: fwd * (1 + spot_change_pct)
            for exp, fwd in self.forward_prices.items()
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
                sigma=params.sigma
            )

        return VolatilitySurface(
            as_of_date=self.as_of_date,
            underlying=self.underlying,
            forward_prices=new_forwards,
            svi_params=new_params
        )


class SVICalibrator:
    """
    Calibrate SVI parameters from market option data.

    Uses least-squares optimization to fit SVI curve to market IVs.
    """

    def __init__(
        self,
        min_points: int = 5,
        max_iterations: int = 1000
    ):
        self.min_points = min_points
        self.max_iterations = max_iterations

    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        T: float
    ) -> SVIParams:
        """
        Calibrate SVI parameters to market data.

        Args:
            strikes: Array of strike prices
            ivs: Array of implied volatilities
            forward: Forward price
            T: Time to expiry in years

        Returns:
            Fitted SVIParams
        """
        if len(strikes) < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {len(strikes)}")

        # Convert to log-moneyness and total variance
        k = np.log(strikes / forward)
        w = ivs**2 * T

        # Initial guess
        atm_var = np.interp(0, k, w)
        x0 = [
            atm_var,      # a
            0.1,          # b
            -0.3,         # rho (typical equity skew)
            0.0,          # m
            0.1           # sigma
        ]

        # Bounds
        bounds = [
            (-0.5, 1.0),      # a
            (0.001, 1.0),     # b
            (-0.99, 0.99),    # rho
            (-0.5, 0.5),      # m
            (0.001, 1.0)      # sigma
        ]

        def objective(params):
            a, b, rho, m, sigma = params
            try:
                w_model = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
                return np.sum((w_model - w)**2)
            except Exception:
                return 1e10

        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )

        if not result.success:
            warnings.warn(f"SVI calibration did not converge: {result.message}")

        a, b, rho, m, sigma = result.x
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


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
        dividend_yield: float = 0.02
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
        expiries = option_data['expiration'].unique()

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
            exp_data = option_data[option_data['expiration'] == expiry].copy()

            # Filter for OTM
            otm_puts = exp_data[
                (exp_data['option_type'] == 'put') &
                (exp_data['strike'] < F)
            ]
            otm_calls = exp_data[
                (exp_data['option_type'] == 'call') &
                (exp_data['strike'] >= F)
            ]
            otm_data = pd.concat([otm_puts, otm_calls])

            if len(otm_data) < 5:
                continue

            strikes = otm_data['strike'].values
            ivs = otm_data['iv_mid'].values

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
                warnings.warn(f"Failed to calibrate {expiry_date}: {e}")
                continue

        return VolatilitySurface(
            as_of_date=as_of_date,
            underlying=underlying,
            forward_prices=forward_prices,
            svi_params=svi_params,
            raw_strikes=raw_strikes,
            raw_ivs=raw_ivs
        )


class SplineVolSurface:
    """
    Simple cubic spline interpolation for IV surface.

    Alternative to SVI when SVI calibration is unstable.
    Faster but less smooth extrapolation.
    """

    def __init__(
        self,
        as_of_date: date,
        underlying: str
    ):
        self.as_of_date = as_of_date
        self.underlying = underlying
        self.splines: Dict[date, Callable] = {}  # expiry -> interpolator
        self.forward_prices: Dict[date, float] = {}

    def fit(
        self,
        expiry: date,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float
    ):
        """Fit spline for single expiry."""
        if len(strikes) < 3:
            return

        # Sort by strike
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        ivs = ivs[idx]

        # Create spline
        spline = interpolate.CubicSpline(
            strikes, ivs,
            bc_type='natural',
            extrapolate=True
        )

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

def create_constant_surface(
    iv: float,
    as_of_date: date,
    underlying: str,
    spot: float,
    expiries: List[date]
) -> VolatilitySurface:
    """
    Create flat (constant IV) surface.

    Useful as fallback when no option data available.
    This is what the backtest currently uses.
    """
    forward_prices = {exp: spot for exp in expiries}

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
                sigma=0.1
            )

    return VolatilitySurface(
        as_of_date=as_of_date,
        underlying=underlying,
        forward_prices=forward_prices,
        svi_params=svi_params
    )


def estimate_iv_for_delta(
    delta: float,
    surface: VolatilitySurface,
    expiry: date,
    spot: float,
    is_put: bool = True
) -> Tuple[float, float]:
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

            rows.append({
                'expiry': expiry,
                'dte': (expiry - surface.as_of_date).days,
                'strike': strike,
                'moneyness': moneyness,
                'log_moneyness': k,
                'iv': iv,
                'forward': F
            })

    return pd.DataFrame(rows)
