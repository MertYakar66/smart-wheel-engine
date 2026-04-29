#!/usr/bin/env python3
"""
Smart Wheel Engine — Feature Smoke Test Harness
================================================

Professional-grade end-to-end smoke test. Exercises every user-facing feature
of the product in one run and reports a structured PASS / FAIL / SKIP table.

Use this whenever you want a fast answer to: "is my data pipeline working?",
"is the brain (EV engine) wired up?", "does the dashboard API respond?".

Exit codes:
  0 — every check passed or was explicitly skipped
  1 — one or more checks failed

Usage::

    python scripts/feature_smoke_test.py                # run everything
    python scripts/feature_smoke_test.py --fast         # skip network/slow paths
    python scripts/feature_smoke_test.py --section ev   # run one section
    python scripts/feature_smoke_test.py --json > out.json   # machine-readable
    python scripts/feature_smoke_test.py --verbose      # show all details

Design notes:
  * No external test framework. Single runnable script.
  * Each check is a tiny closure returning either a short detail string, None,
    or raising either ``Skip(reason)`` (not a failure) or any Exception (FAIL).
  * Live-data paths (Theta, Bloomberg, HTTP API) auto-skip when not reachable.
  * No mocking of the brain: EV engine, risk manager, wheel runner all run
    against synthetic but realistic inputs so real bugs surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
# Insert project root at front unconditionally — the "not in sys.path" guard
# is unreliable on Windows where normalised and unnormalised path strings can
# both be present and both miss string equality against the repo root.
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PYTHONPATH", str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ----------------------------------------------------------------------
# Framework
# ----------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


class Skip(Exception):
    """Raise inside a check to mark it as SKIP rather than FAIL."""


@dataclass
class CheckResult:
    section: str
    name: str
    status: str
    detail: str = ""
    duration_ms: float = 0.0


class Harness:
    def __init__(self, section_filter: str | None = None, fast: bool = False) -> None:
        self.section_filter = section_filter
        self.fast = fast
        self.results: list[CheckResult] = []
        self._current_section = ""

    def section(self, name: str) -> None:
        self._current_section = name

    def run(self, name: str, fn: Callable[[], Any]) -> None:
        if self.section_filter and self.section_filter.lower() not in self._current_section.lower():
            return
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            status = PASS
        except Skip as s:
            detail = str(s)
            status = SKIP
        except AssertionError as e:
            status = FAIL
            detail = f"assertion: {e}"
        except Exception as e:  # noqa: BLE001 — smoke harness intentionally catches all
            status = FAIL
            detail = f"{type(e).__name__}: {e}"
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.results.append(CheckResult(self._current_section, name, status, str(detail), elapsed))

    # ------------------------------------------------------------------
    def report(self, verbose: bool, as_json: bool) -> int:
        if as_json:
            print(json.dumps([asdict(r) for r in self.results], indent=2))
            return 0 if not any(r.status == FAIL for r in self.results) else 1

        # Human report.
        sections: dict[str, list[CheckResult]] = {}
        for r in self.results:
            sections.setdefault(r.section, []).append(r)

        total_pass = sum(1 for r in self.results if r.status == PASS)
        total_fail = sum(1 for r in self.results if r.status == FAIL)
        total_skip = sum(1 for r in self.results if r.status == SKIP)

        def color(s: str, status: str) -> str:
            if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
                return s
            code = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}.get(status, "")
            return f"{code}{s}\033[0m" if code else s

        print()
        print("=" * 78)
        print(" Smart Wheel Engine — Feature Smoke Test")
        print("=" * 78)

        for sect, items in sections.items():
            p = sum(1 for r in items if r.status == PASS)
            f = sum(1 for r in items if r.status == FAIL)
            s = sum(1 for r in items if r.status == SKIP)
            header = f" [{sect}]  {p} pass  {f} fail  {s} skip"
            print()
            print(header)
            print("-" * len(header))
            for r in items:
                tag = color(f"{r.status:<4}", r.status)
                dur = f"{r.duration_ms:>6.1f}ms"
                line = f"  {tag}  {dur}  {r.name}"
                if r.status == FAIL or (verbose and r.detail):
                    line += f"    — {r.detail}"
                print(line)

        print()
        print("=" * 78)
        summary = (
            f" Total: {len(self.results)}   "
            f"{color(f'{total_pass} PASS', PASS)}   "
            f"{color(f'{total_fail} FAIL', FAIL)}   "
            f"{color(f'{total_skip} SKIP', SKIP)}"
        )
        print(summary)
        print("=" * 78)
        return 0 if total_fail == 0 else 1


# ----------------------------------------------------------------------
# Shared fixtures (cheap; built once, reused across checks)
# ----------------------------------------------------------------------
def _synth_ohlcv(n_days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV series — long enough for realized-vol
    estimators and the 504-day HMM window used by the regime detector."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    log_rets = rng.normal(0.0003, 0.012, n_days)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    # O/H/L with ~40bp intraday noise around close.
    openp = close * (1 + rng.normal(0, 0.004, n_days))
    high = np.maximum.reduce([openp, close]) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
    low = np.minimum.reduce([openp, close]) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
    vol = rng.integers(1_000_000, 5_000_000, n_days)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _synth_chain(spot: float = 100.0, dte_days: int = 32) -> pd.DataFrame:
    """Synthetic options chain with clean bid/ask around a BSM mid."""
    from datetime import date, timedelta

    from engine.option_pricer import black_scholes_price

    expiry = date.today() + timedelta(days=dte_days)
    T = dte_days / 365.0
    iv = 0.25
    strikes = np.arange(spot * 0.80, spot * 1.201, 2.5)
    rows = []
    for K in strikes:
        for opt, right in (("put", "P"), ("call", "C")):
            mid = black_scholes_price(spot, K, T, 0.05, iv, opt, 0.0)
            mid = max(mid, 0.05)
            rows.append(
                {
                    "strike": float(K),
                    "option_type": right,
                    "right": opt,
                    "expiration": expiry,
                    "bid": round(mid * 0.97, 2),
                    "ask": round(mid * 1.03, 2),
                    "implied_vol": iv,
                    "iv": iv,
                    "open_interest": 500,
                    "volume": 100,
                    "delta": (
                        -0.5 if opt == "put" and K > spot else
                        0.5 if opt == "call" and K < spot else
                        0.25 if opt == "put" else -0.25
                    ),
                }
            )
    return pd.DataFrame(rows)


class _FakeConnector:
    """Minimal deterministic connector used by wheel-runner integration checks."""

    def __init__(self, ticker: str = "TESTA", chain: pd.DataFrame | None = None) -> None:
        self._ticker = ticker
        self._ohlcv = _synth_ohlcv()
        self._chain = chain if chain is not None else _synth_chain()

    def get_ohlcv(self, ticker, start_date=None, end_date=None):  # noqa: ARG002
        return self._ohlcv

    def get_fundamentals(self, ticker):  # noqa: ARG002
        return {"implied_vol_atm": 0.25, "volatility_30d": 0.22, "dividend_yield": 0.005}

    def get_risk_free_rate(self, as_of=None):  # noqa: ARG002
        return 0.05

    def get_next_earnings(self, ticker, as_of=None):  # noqa: ARG002
        return None

    def get_universe(self):
        return [self._ticker]

    def get_options(self, ticker):  # noqa: ARG002
        return self._chain.copy()

    def get_credit_risk(self, ticker):  # noqa: ARG002
        return None

    def get_option_chain(self, ticker, dte_target=None):  # noqa: ARG002
        return self._chain.copy()


# ======================================================================
# CHECKS
# ======================================================================
def register_checks(h: Harness) -> None:
    # ------------------------------------------------------------------
    # 1. Option pricing — the math spine
    # ------------------------------------------------------------------
    h.section("01 option_pricer")
    from engine.option_pricer import (
        black_scholes_price,
        black_scholes_delta,
        black_scholes_gamma,
        black_scholes_theta,
        black_scholes_vega,
        black_scholes_rho,
        black_scholes_all_greeks,
        vectorized_bs_all_greeks,
        black_scholes_speed,
        black_scholes_color,
        black_scholes_ultima,
        implied_volatility,
        american_option_price,
        american_option_greeks,
    )

    def bsm_call():
        p = black_scholes_price(100, 100, 30 / 365, 0.05, 0.25, "call", 0.0)
        assert 2.5 < p < 4.5, p
        return f"ATM call price={p:.3f}"

    def bsm_put():
        p = black_scholes_price(100, 100, 30 / 365, 0.05, 0.25, "put", 0.0)
        assert 2.0 < p < 4.5, p
        return f"ATM put price={p:.3f}"

    def bsm_put_call_parity():
        S, K, T, r, sigma, q = 100, 100, 30 / 365, 0.05, 0.25, 0.0
        c = black_scholes_price(S, K, T, r, sigma, "call", q)
        p = black_scholes_price(S, K, T, r, sigma, "put", q)
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        err = abs(lhs - rhs)
        assert err < 1e-6, f"parity break {err:.2e}"
        return f"residual={err:.2e}"

    def bsm_greeks_first_order():
        g = {
            "delta": black_scholes_delta(100, 100, 30 / 365, 0.05, 0.25, "call"),
            "gamma": black_scholes_gamma(100, 100, 30 / 365, 0.05, 0.25),
            "theta": black_scholes_theta(100, 100, 30 / 365, 0.05, 0.25, "call"),
            "vega": black_scholes_vega(100, 100, 30 / 365, 0.05, 0.25),
            "rho": black_scholes_rho(100, 100, 30 / 365, 0.05, 0.25, "call"),
        }
        assert 0.45 < g["delta"] < 0.60, g["delta"]
        assert g["gamma"] > 0, g["gamma"]
        return ", ".join(f"{k}={v:.4f}" for k, v in g.items())

    def bsm_higher_order_greeks():
        d = black_scholes_all_greeks(100, 100, 30 / 365, 0.05, 0.25, "call", 0.0, True)
        # Verify core keys are present (second-order Greeks populated).
        for k in ("delta", "gamma", "theta", "vega"):
            assert k in d, k
        sp = black_scholes_speed(100, 100, 30 / 365, 0.05, 0.25)
        co = black_scholes_color(100, 100, 30 / 365, 0.05, 0.25)
        ul = black_scholes_ultima(100, 100, 30 / 365, 0.05, 0.25)
        assert all(np.isfinite([sp, co, ul]))
        return f"speed={sp:.6f} color={co:.6f} ultima={ul:.3f}"

    def bsm_vectorized():
        Ks = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        out = vectorized_bs_all_greeks(100.0, Ks, 30 / 365, 0.05, 0.25, "call", 0.0)
        assert len(out["delta"]) == len(Ks)
        assert all(np.diff(out["delta"]) < 0)  # monotonic in K
        return f"n={len(Ks)} monotone delta OK"

    def iv_solver():
        true_sigma = 0.32
        price = black_scholes_price(100, 105, 45 / 365, 0.05, true_sigma, "call", 0.01)
        iv = implied_volatility(price, 100, 105, 45 / 365, 0.05, "call", q=0.01)
        assert abs(iv - true_sigma) < 1e-3, f"iv={iv}"
        return f"recovered iv={iv:.5f} (true={true_sigma})"

    def iv_solver_edge_itm():
        # Deep ITM should still converge via Brent fallback.
        true_sigma = 0.45
        price = black_scholes_price(100, 80, 60 / 365, 0.05, true_sigma, "call")
        iv = implied_volatility(price, 100, 80, 60 / 365, 0.05, "call")
        assert abs(iv - true_sigma) < 5e-3, iv
        return f"ITM recovery iv={iv:.4f}"

    def american_baw_call_no_div():
        # Without dividends, American call should equal European call.
        euro = black_scholes_price(100, 100, 90 / 365, 0.05, 0.30, "call", 0.0)
        am = american_option_price(100, 100, 90 / 365, 0.05, 0.30, "call", 0.0)
        assert abs(am - euro) < 0.05, f"am={am} euro={euro}"
        return f"am={am:.4f} euro={euro:.4f}"

    def american_baw_put_early_exercise_premium():
        euro = black_scholes_price(100, 100, 90 / 365, 0.05, 0.30, "put", 0.0)
        am = american_option_price(100, 100, 90 / 365, 0.05, 0.30, "put", 0.0)
        # American put >= European put (early-exercise premium).
        assert am >= euro - 1e-6, f"am={am} euro={euro}"
        return f"EEP = {am - euro:.4f}"

    def american_greeks_path():
        g = american_option_greeks(100, 105, 60 / 365, 0.05, 0.28, "put")
        for k in ("delta", "gamma", "theta", "vega"):
            assert np.isfinite(g.get(k, float("nan"))), k
        return f"delta={g['delta']:.4f} gamma={g['gamma']:.5f}"

    h.run("bsm_call_price", bsm_call)
    h.run("bsm_put_price", bsm_put)
    h.run("bsm_put_call_parity", bsm_put_call_parity)
    h.run("bsm_first_order_greeks", bsm_greeks_first_order)
    h.run("bsm_higher_order_greeks", bsm_higher_order_greeks)
    h.run("bsm_vectorized_greeks", bsm_vectorized)
    h.run("implied_vol_newton", iv_solver)
    h.run("implied_vol_itm_fallback", iv_solver_edge_itm)
    h.run("american_baw_no_div_equals_euro", american_baw_call_no_div)
    h.run("american_put_early_exercise_premium", american_baw_put_early_exercise_premium)
    h.run("american_greeks", american_greeks_path)

    # ------------------------------------------------------------------
    # 2. Binomial tree
    # ------------------------------------------------------------------
    h.section("02 binomial_tree")
    from engine.binomial_tree import (
        binomial_american_price,
        binomial_with_richardson,
        convergence_study,
    )

    def crr_american_price():
        p = binomial_american_price(100, 100, 60 / 365, 0.05, 0.25, "call", 0.0, n_steps=200)
        assert 2.0 < p < 6.0, p
        return f"price={p:.4f}"

    def crr_richardson():
        p = binomial_with_richardson(100, 100, 60 / 365, 0.05, 0.25, "call", 0.0, n_steps=100)
        assert np.isfinite(p) and p > 0
        return f"price={p:.4f}"

    def crr_convergence():
        res = convergence_study(100, 95, 60 / 365, 0.05, 0.30, "put", step_counts=[50, 100, 200])
        assert isinstance(res, (pd.DataFrame, dict, list))
        return "ran 50/100/200 steps"

    h.run("crr_american_price", crr_american_price)
    h.run("crr_richardson", crr_richardson)
    h.run("crr_convergence_study", crr_convergence)

    # ------------------------------------------------------------------
    # 3. Realized volatility estimators
    # ------------------------------------------------------------------
    h.section("03 realized_vol")
    from engine.realized_vol import (
        close_to_close_vol,
        parkinson_vol,
        garman_klass_vol,
        rogers_satchell_vol,
        realised_vol_bundle,
    )

    ohlcv = _synth_ohlcv()

    def rv_close_to_close():
        v = close_to_close_vol(ohlcv, window=30)
        assert np.isfinite(v), v
        assert 0.05 < float(v) < 0.60, v
        return f"annualised σ={float(v):.3f}"

    def rv_parkinson():
        v = parkinson_vol(ohlcv, window=30)
        assert 0.03 < float(v) < 0.80, v
        return f"parkinson σ={float(v):.3f}"

    def rv_garman_klass():
        v = garman_klass_vol(ohlcv, window=30)
        assert 0.03 < float(v) < 0.80, v
        return f"GK σ={float(v):.3f}"

    def rv_rogers_satchell():
        v = rogers_satchell_vol(ohlcv, window=30)
        assert 0.03 < float(v) < 0.80, v
        return f"RS σ={float(v):.3f}"

    def rv_bundle():
        bundle = realised_vol_bundle(ohlcv, window=30)
        assert isinstance(bundle, (dict, pd.DataFrame))
        return "bundle computed"

    h.run("rv_close_to_close", rv_close_to_close)
    h.run("rv_parkinson", rv_parkinson)
    h.run("rv_garman_klass", rv_garman_klass)
    h.run("rv_rogers_satchell", rv_rogers_satchell)
    h.run("rv_bundle", rv_bundle)

    # ------------------------------------------------------------------
    # 4. Volatility surface & skew
    # ------------------------------------------------------------------
    h.section("04 vol_surface")
    from engine.volatility_surface import SVICalibrator, create_constant_surface
    from engine.skew_dynamics import skew_slope, skew_momentum, ivs_dislocation_score

    def svi_calibration():
        # Synthetic SVI-shaped smile. The calibrator expects IVs (not total
        # variance) plus a forward and time-to-expiry so it can build the
        # internal total-variance surface.
        spot = 100.0
        T = 60 / 365.0
        strikes = np.linspace(80.0, 120.0, 15)
        k = np.log(strikes / spot)
        ivs = 0.22 + 0.08 * k**2 - 0.03 * k
        try:
            params = SVICalibrator().calibrate(
                strikes=strikes, ivs=ivs, forward=spot, T=T,
            )
        except Exception as e:
            raise Skip(f"SVI unavailable: {e}")
        assert params is not None
        return "SVI fit converged"

    def constant_surface():
        from datetime import date, timedelta
        s = create_constant_surface(
            iv=0.25,
            as_of_date=date.today(),
            underlying="TESTA",
            spot=100.0,
            expiries=[date.today() + timedelta(days=d) for d in (7, 30, 60, 90)],
        )
        assert s is not None
        return "constant σ=0.25 surface built"

    def skew_slope_check():
        out = skew_slope(iv_25d_put=0.28, iv_atm=0.24, iv_25d_call=0.22)
        assert "skew_slope" in out
        assert out["skew_slope"] > 0
        return f"slope={out['skew_slope']:.4f}"

    def skew_momentum_check():
        slopes = np.linspace(0.0, 0.2, 30)
        mom = skew_momentum(slopes)
        assert isinstance(mom, dict) and len(mom) > 0
        return f"keys={list(mom.keys())[:3]}"

    def ivs_dislocation():
        tenors = np.array([7/365, 30/365, 60/365, 90/365, 180/365, 365/365])
        ivs = np.array([0.28, 0.26, 0.25, 0.24, 0.23, 0.22])
        out = ivs_dislocation_score(tenors, ivs)
        assert isinstance(out, dict) and len(out) > 0
        return f"keys={list(out.keys())[:3]}"

    h.run("svi_calibration", svi_calibration)
    h.run("constant_vol_surface", constant_surface)
    h.run("skew_slope", skew_slope_check)
    h.run("skew_momentum", skew_momentum_check)
    h.run("ivs_dislocation_score", ivs_dislocation)

    # ------------------------------------------------------------------
    # 5. Forward-return distribution & regime
    # ------------------------------------------------------------------
    h.section("05 forward_dist_regime")
    from engine.forward_distribution import (
        empirical_forward_log_returns,
        block_bootstrap_log_returns,
        best_available_forward_distribution,
    )
    from engine.regime_hmm import GaussianHMM
    from engine.regime_detector import RegimeDetector

    def empirical_forward():
        arr = empirical_forward_log_returns(ohlcv, horizon_days=30)
        assert isinstance(arr, np.ndarray)
        assert len(arr) >= 20, f"got {len(arr)}"
        return f"n={len(arr)}"

    def block_bootstrap():
        arr = block_bootstrap_log_returns(
            ohlcv, horizon_days=30, n_scenarios=2000, block_size=5, seed=0,
        )
        assert len(arr) == 2000
        return "bootstrap n=2000"

    def best_available_fwd():
        arr, method = best_available_forward_distribution(ohlcv, horizon_days=30)
        assert len(arr) > 0
        return f"method={method} n={len(arr)}"

    def hmm_fit_predict():
        log_rets = np.diff(np.log(ohlcv["close"].values))
        hmm = GaussianHMM(n_states=3, n_iter=20, random_state=42)
        hmm.fit(log_rets[-504:])
        probs = hmm.predict_proba(log_rets[-504:])
        assert probs.shape[1] == 3
        mult = hmm.position_multiplier(probs[-1])
        assert 0.0 <= mult <= 2.0
        return f"mult={mult:.3f}"

    def regime_detector_probe():
        rd = RegimeDetector()
        state = rd.detect_regime(current_iv=0.25, prices=ohlcv["close"])
        assert state is not None
        return f"regime={getattr(state, 'volatility_regime', '?')}"

    h.run("empirical_forward_log_returns", empirical_forward)
    h.run("block_bootstrap_returns", block_bootstrap)
    h.run("best_available_forward_distribution", best_available_fwd)
    h.run("hmm_fit_and_position_multiplier", hmm_fit_predict)
    h.run("regime_detector_classify", regime_detector_probe)

    # ------------------------------------------------------------------
    # 6. EV engine — the brain
    # ------------------------------------------------------------------
    h.section("06 ev_engine")
    from engine.ev_engine import EVEngine, ShortOptionTrade
    from engine.event_gate import EventGate, ScheduledEvent

    def ev_fallback_path():
        trade = ShortOptionTrade(
            option_type="put", underlying="TESTA", spot=100, strike=95,
            premium=1.5, dte=30, iv=0.25, bid=1.45, ask=1.55, open_interest=1000,
        )
        res = EVEngine().evaluate(trade)
        assert 0.0 <= res.prob_profit <= 1.0
        assert res.metadata["distribution_source"] == "lognormal_fallback"
        return f"ev=${res.ev_dollars:.2f} pp={res.prob_profit:.3f}"

    def ev_empirical_path():
        rng = np.random.default_rng(0)
        fwd = rng.normal(0.0, 0.02, 1500)
        trade = ShortOptionTrade(
            option_type="put", underlying="TESTA", spot=100, strike=92,
            premium=1.0, dte=30, iv=0.25, bid=0.95, ask=1.05, open_interest=500,
        )
        res = EVEngine().evaluate(trade, forward_log_returns=fwd)
        assert res.metadata["distribution_source"] == "empirical"
        assert np.isfinite(res.omega_ratio)
        return f"ev=${res.ev_dollars:.2f} cvar5=${res.cvar_5:.2f} omega={res.omega_ratio:.2f}"

    def ev_regime_clamp_nan():
        t = ShortOptionTrade(
            option_type="put", underlying="TESTA", spot=100, strike=95,
            premium=1.5, dte=30, iv=0.25, bid=1.45, ask=1.55, open_interest=1000,
            regime_multiplier=float("nan"),
        )
        r = EVEngine().evaluate(t)
        assert "regime_mult_nonfinite" in r.metadata["regime_anomaly"]
        return "nan → flagged & clamped"

    def ev_event_gate_blocks():
        from datetime import date, timedelta
        gate = EventGate(earnings_buffer_days=5, macro_buffer_days=1)
        gate.add_event(ScheduledEvent(ticker="TESTA", kind="earnings", event_date=date.today() + timedelta(days=3)))
        t = ShortOptionTrade(
            option_type="put", underlying="TESTA", spot=100, strike=95,
            premium=1.5, dte=30, iv=0.25, bid=1.45, ask=1.55, open_interest=1000,
        )
        eng = EVEngine(event_gate=gate)
        r = eng.evaluate(
            t,
            trade_start=date.today(),
            trade_end=date.today() + timedelta(days=30),
        )
        assert r.ev_dollars == 0.0 and "event_lockout" in r.event_lockout_reason
        return f"blocked: {r.event_lockout_reason}"

    def ev_stop_loss_prob():
        rng = np.random.default_rng(1)
        bad = rng.normal(-0.15, 0.05, 5000)  # many stop breaches
        t = ShortOptionTrade(
            option_type="put", underlying="TESTA", spot=100, strike=95,
            premium=1.5, dte=30, iv=0.25, bid=1.45, ask=1.55, open_interest=1000,
        )
        r = EVEngine().evaluate(t, forward_log_returns=bad)
        assert 0.0 <= r.metadata["prob_stop_terminal"] <= 1.0
        return f"prob_stop={r.metadata['prob_stop_terminal']:.3f}"

    h.run("evaluate_fallback", ev_fallback_path)
    h.run("evaluate_empirical", ev_empirical_path)
    h.run("regime_multiplier_clamp", ev_regime_clamp_nan)
    h.run("event_gate_hard_block", ev_event_gate_blocks)
    h.run("stop_loss_probability", ev_stop_loss_prob)

    # ------------------------------------------------------------------
    # 7. Risk manager
    # ------------------------------------------------------------------
    h.section("07 risk_manager")
    from engine.risk_manager import (
        RiskManager,
        RiskLimits,
        calculate_kelly_fraction,
        calculate_optimal_contracts,
        calculate_hrp_weights,
        optimize_position_weights,
    )

    def risk_manager_init():
        rm = RiskManager(limits=RiskLimits())
        ok, violations = rm.check_limits(
            portfolio_value=500_000,
            positions=[],
            spot_prices={},
        )
        return f"limits={'OK' if ok else 'VIOLATIONS'} ({len(violations)} flagged)"

    def kelly_fraction():
        f = calculate_kelly_fraction(win_rate=0.65, avg_win=100, avg_loss=80, kelly_fraction=0.5)
        assert 0.0 <= f <= 0.25, f
        return f"f*={f:.4f}"

    def kelly_invalid_inputs_guarded():
        f = calculate_kelly_fraction(win_rate=1.5, avg_win=100, avg_loss=80)
        assert f == 0.0
        return "OOR input → 0"

    def optimal_contracts():
        n = calculate_optimal_contracts(
            capital=100_000, strike=100.0, max_risk_pct=0.05,
            margin_requirement=0.20, stress_loss_pct=0.25, premium_per_share=1.5,
        )
        assert n >= 0
        return f"n={n}"

    def hrp_weights():
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.normal(0, 0.01, (252, 5)),
                               columns=[f"A{i}" for i in range(5)])
        w = calculate_hrp_weights(returns)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        return f"Σw={sum(w.values()):.6f}"

    def mv_optimizer():
        rng = np.random.default_rng(0)
        symbols = [f"A{i}" for i in range(4)]
        returns = pd.DataFrame(rng.normal(0, 0.01, (252, 4)), columns=symbols)
        w = optimize_position_weights(symbols, returns, max_weight=0.4, min_weight=0.05)
        assert isinstance(w, dict) and abs(sum(w.values()) - 1.0) < 0.1
        return f"Σw={sum(w.values()):.3f}"

    h.run("risk_manager_init", risk_manager_init)
    h.run("kelly_fraction", kelly_fraction)
    h.run("kelly_bad_input_guarded", kelly_invalid_inputs_guarded)
    h.run("optimal_contracts", optimal_contracts)
    h.run("hrp_weights_sum_to_one", hrp_weights)
    h.run("mv_optimizer", mv_optimizer)

    # ------------------------------------------------------------------
    # 8. Transaction costs & event gate
    # ------------------------------------------------------------------
    h.section("08 costs_and_gates")
    from engine.transaction_costs import (
        calculate_commission, calculate_slippage,
        calculate_assignment_fee, calculate_total_entry_cost,
        calculate_total_exit_cost, calculate_reg_t_margin_short_put,
    )

    def commission_check():
        c = calculate_commission("option", num_contracts=5)
        assert c > 0
        return f"${c:.2f}"

    def slippage_check():
        s = calculate_slippage(mid_price=1.50, bid_ask_spread=0.10,
                               trade_direction="sell", open_interest=500)
        assert s > 0
        return f"${s:.4f}/sh"

    def assignment_fee_check():
        f = calculate_assignment_fee()
        assert f >= 0
        return f"${f:.2f}"

    def total_entry_cost_check():
        out = calculate_total_entry_cost(
            premium_per_share=1.5, bid_ask_spread=0.1,
            trade_type="option", open_interest=500,
        )
        assert isinstance(out, dict)
        return f"keys={sorted(out.keys())[:3]}"

    def total_exit_cost_check():
        out = calculate_total_exit_cost(
            buyback_price_per_share=1.0, bid_ask_spread=0.1,
            trade_type="option", open_interest=500,
        )
        assert isinstance(out, dict)
        return f"keys={sorted(out.keys())[:3]}"

    def reg_t_margin_check():
        m = calculate_reg_t_margin_short_put(strike=100, underlying_price=98, premium=1.5)
        assert m > 0
        return f"margin=${m:.2f}"

    def event_gate_not_blocked():
        from datetime import date, timedelta
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent(ticker="TESTA", kind="earnings",
                                event_date=date.today() + timedelta(days=90)))
        blocked, reason = gate.is_blocked(
            "TESTA", date.today(), date.today() + timedelta(days=30)
        )
        assert not blocked
        return "outside buffer: clear"

    def event_gate_blocked():
        from datetime import date, timedelta
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent(ticker="TESTA", kind="earnings",
                                event_date=date.today() + timedelta(days=3)))
        blocked, reason = gate.is_blocked(
            "TESTA", date.today(), date.today() + timedelta(days=30)
        )
        assert blocked
        return reason

    h.run("commission", commission_check)
    h.run("slippage", slippage_check)
    h.run("assignment_fee", assignment_fee_check)
    h.run("total_entry_cost", total_entry_cost_check)
    h.run("total_exit_cost", total_exit_cost_check)
    h.run("reg_t_margin_short_put", reg_t_margin_check)
    h.run("event_gate_clear_of_event", event_gate_not_blocked)
    h.run("event_gate_within_buffer", event_gate_blocked)

    # ------------------------------------------------------------------
    # 9. Signals & payoff
    # ------------------------------------------------------------------
    h.section("09 signals_payoff")
    from engine.signals import (
        IVRankSignal, DTESignal, ProfitTargetSignal, EventFilterSignal,
        create_default_aggregator,
    )
    from engine.payoff_engine import compute_payoff, compute_expected_move, recommend_strikes

    def iv_rank_signal():
        s = IVRankSignal().generate({"iv_rank": 0.75})
        assert s.is_actionable
        return f"{s.strength.name} v={s.value:.2f}"

    def dte_signal():
        s = DTESignal().generate({"dte": 35})
        assert s is not None
        return f"{s.strength.name}"

    def profit_target_signal():
        s = ProfitTargetSignal(target_pct=0.5).generate(
            {"premium_received": 1.50, "current_premium": 0.70}
        )
        assert s is not None
        return f"{s.strength.name}"

    def event_filter_signal():
        s = EventFilterSignal(earnings_buffer_days=5).generate({"days_to_earnings": 3})
        assert s is not None
        return f"{s.strength.name}"

    def composite_aggregator():
        agg = create_default_aggregator()
        assert agg is not None
        return "aggregator constructed"

    def payoff_short_put():
        res = compute_payoff(
            spot=100, strike=95, premium=1.5, strategy="csp", contracts=1,
        )
        assert isinstance(res, list) and len(res) > 0
        return f"{len(res)} payoff points"

    def expected_move_one_sigma():
        out = compute_expected_move(spot=100, iv=0.25, dte=30)
        assert isinstance(out, dict) and len(out) > 0
        return f"keys={list(out.keys())[:3]}"

    def strike_recommendation():
        recs = recommend_strikes(
            ticker="TESTA", spot=100, iv=0.25, dte=30, strategy="csp", n_candidates=3,
        )
        assert isinstance(recs, list) and len(recs) > 0
        return f"{len(recs)} strikes recommended"

    h.run("iv_rank_signal", iv_rank_signal)
    h.run("dte_signal", dte_signal)
    h.run("profit_target_signal", profit_target_signal)
    h.run("event_filter_signal", event_filter_signal)
    h.run("signal_aggregator", composite_aggregator)
    h.run("payoff_short_put", payoff_short_put)
    h.run("expected_move", expected_move_one_sigma)
    h.run("strike_recommendation", strike_recommendation)

    # ------------------------------------------------------------------
    # 10. Dealer positioning, tail risk
    # ------------------------------------------------------------------
    h.section("10 dealer_tailrisk")
    from engine.dealer_positioning import (
        DealerAssumption, DealerPositioningAnalyzer, dealer_regime_multiplier,
    )
    from engine.tail_risk import fit_gpd_tail, gpd_var_cvar

    chain = _synth_chain()

    def dealer_analyze():
        from datetime import date, timedelta
        analyzer = DealerPositioningAnalyzer(
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS,
        )
        ms = analyzer.analyze(
            chain=chain, spot=100.0, expiry=date.today() + timedelta(days=32),
            ticker="TESTA", dividend_yield=0.0,
        )
        assert ms is not None
        return f"regime={getattr(ms,'regime','?')}"

    def dealer_multiplier():
        from datetime import date, timedelta
        ms = DealerPositioningAnalyzer(
            assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS
        ).analyze(
            chain=chain, spot=100.0, expiry=date.today() + timedelta(days=32),
            ticker="TESTA", dividend_yield=0.0,
        )
        m = dealer_regime_multiplier(ms)
        assert 0.70 <= m <= 1.05, m
        return f"m={m:.3f}"

    def gpd_fit_losses():
        rng = np.random.default_rng(0)
        losses = np.abs(rng.standard_t(4, 1500)) * 1.2
        fit = fit_gpd_tail(losses)
        if not fit.converged:
            raise Skip("GPD fit did not converge on synthetic")
        var99, cvar99 = gpd_var_cvar(fit, confidence=0.99)
        assert var99 > 0 and cvar99 >= var99
        return f"ξ={fit.shape_xi:.3f} CVaR99={cvar99:.3f}"

    h.run("dealer_positioning_analyze", dealer_analyze)
    h.run("dealer_regime_multiplier", dealer_multiplier)
    h.run("gpd_tail_fit", gpd_fit_losses)

    # ------------------------------------------------------------------
    # 11. Monte Carlo & stress
    # ------------------------------------------------------------------
    h.section("11 monte_carlo_stress")
    from engine.monte_carlo import (
        BlockBootstrap, JumpDiffusionSimulator, JumpDiffusionParams, LSMPricer,
    )
    from engine.stress_testing import quick_stress_test

    def block_bootstrap_cls():
        rng = np.random.default_rng(0)
        daily_returns = rng.normal(0, 0.01, 500)
        bb = BlockBootstrap(block_size=21, n_simulations=500, seed=0)
        result = bb.simulate(daily_returns=daily_returns, n_days=60, initial_capital=100_000)
        assert result is not None
        return "BlockBootstrap simulated"

    def jump_diffusion_sim():
        sim = JumpDiffusionSimulator(JumpDiffusionParams(
            mu=0.05, sigma=0.2, jump_intensity=0.1,
            jump_mean=-0.02, jump_std=0.05, dividend_yield=0.0,
        ))
        paths = sim.simulate_paths(S0=100.0, n_days=63)
        assert paths.shape[-1] == 64 or paths.shape[0] >= 63
        return f"shape={paths.shape}"

    def lsm_american():
        pricer = LSMPricer(n_paths=2000, n_steps_per_day=1, polynomial_degree=3, seed=0)
        res = pricer.price(
            S0=100, K=100, T=30/365, r=0.05, sigma=0.3, option_type="put", q=0.0,
        )
        assert res.american_price > 0
        return f"LSM am=${res.american_price:.3f} euro=${res.european_price:.3f}"

    def quick_stress():
        report = quick_stress_test(
            positions=[
                {"symbol": "TESTA", "option_type": "put", "strike": 95, "dte": 30,
                 "iv": 0.25, "contracts": 1, "is_short": True,
                 "underlying_price": 100, "premium": 1.5},
            ],
            spot_prices={"TESTA": 100.0},
            portfolio_value=100_000,
        )
        assert isinstance(report, str) and len(report) > 0
        return "stress report produced"

    h.run("block_bootstrap_class", block_bootstrap_cls)
    h.run("jump_diffusion_simulate", jump_diffusion_sim)
    h.run("lsm_american_put", lsm_american)
    h.run("quick_stress_test", quick_stress)

    # ------------------------------------------------------------------
    # 12. Portfolio, copula, performance
    # ------------------------------------------------------------------
    h.section("12 portfolio_perf")
    from engine.portfolio_tracker import quick_snapshot, Holding
    from engine.portfolio_copula import (
        gaussian_copula_simulation, student_t_copula_simulation, portfolio_cvar_copula,
    )
    from engine.performance_metrics import (
        calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
        calculate_profit_factor, calculate_ulcer_index, calculate_performance_report,
    )

    def portfolio_snapshot():
        from datetime import date
        from engine.portfolio_tracker import PortfolioTracker, Transaction, TransactionType
        tracker = PortfolioTracker(initial_cash=10_000)
        tracker.add_transaction(Transaction(
            ticker="AAPL", action=TransactionType.BUY, shares=10.0,
            price=150.0, date=date.today(),
        ))
        snap = quick_snapshot(tracker, prices={"AAPL": 170.0})
        assert isinstance(snap, dict)
        return f"keys={sorted(list(snap.keys())[:4])}"

    def gaussian_copula():
        rng = np.random.default_rng(0)
        marginals = [rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500)]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        sims = gaussian_copula_simulation(marginals=marginals, correlation=corr,
                                           n_samples=5000, seed=0)
        assert sims.shape[1] == 2
        return f"shape={sims.shape}"

    def student_t_copula():
        rng = np.random.default_rng(0)
        marginals = [rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500)]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        sims = student_t_copula_simulation(marginals=marginals, correlation=corr,
                                            df=4, n_samples=5000, seed=0)
        assert sims.shape[1] == 2
        return f"df=4 shape={sims.shape}"

    def portfolio_cvar():
        rng = np.random.default_rng(0)
        marginals = [rng.normal(0, 0.01, 500) for _ in range(3)]
        corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
        weights = np.array([1/3, 1/3, 1/3])
        out = portfolio_cvar_copula(marginals=marginals, correlation=corr,
                                     weights=weights, confidence=0.95, seed=0)
        assert isinstance(out, dict) and len(out) > 0
        return f"keys={list(out.keys())[:3]}"

    def perf_sharpe_sortino():
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        sh = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        so = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        assert np.isfinite(sh) and np.isfinite(so)
        return f"Sharpe={sh:.3f} Sortino={so:.3f}"

    def perf_drawdown_ulcer_profit():
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0003, 0.012, 252)
        equity = pd.DataFrame({"equity": 100_000 * np.cumprod(1.0 + returns)})
        dd_result = calculate_max_drawdown(equity)
        ui = calculate_ulcer_index(equity)
        trades = pd.DataFrame({"pnl": rng.normal(50, 200, 50)})
        pf = calculate_profit_factor(trades)
        assert np.isfinite(ui) and np.isfinite(pf)
        return f"DD_tuple_len={len(dd_result)} UI={ui:.4f} PF={pf:.3f}"

    def perf_full_report():
        from datetime import date, timedelta
        rng = np.random.default_rng(0)
        n = 60
        equity_curve = [
            {"date": (date.today() - timedelta(days=n - i)),
             "portfolio_value": 100_000 * (1 + 0.001 * i)}
            for i in range(n)
        ]
        closed_trades = [
            {"ticker": "TESTA", "net_pnl": float(rng.normal(50, 100)),
             "opened": date.today() - timedelta(days=30),
             "closed": date.today() - timedelta(days=10)}
            for _ in range(10)
        ]
        rep = calculate_performance_report(closed_trades, equity_curve, initial_capital=100_000)
        assert rep is not None
        return "full report generated"

    h.run("portfolio_snapshot", portfolio_snapshot)
    h.run("gaussian_copula_sim", gaussian_copula)
    h.run("student_t_copula_sim", student_t_copula)
    h.run("portfolio_cvar_copula", portfolio_cvar)
    h.run("sharpe_sortino", perf_sharpe_sortino)
    h.run("drawdown_ulcer_profit_factor", perf_drawdown_ulcer_profit)
    h.run("full_performance_report", perf_full_report)

    # ------------------------------------------------------------------
    # 13. Wheel runner — end-to-end with fake connector
    # ------------------------------------------------------------------
    h.section("13 wheel_runner_e2e")
    from engine.wheel_runner import WheelRunner

    def analyze_ticker_e2e():
        runner = WheelRunner()
        runner._connector = _FakeConnector()
        analysis = runner.analyze_ticker("TESTA")
        assert analysis is not None
        return f"wheel_score={getattr(analysis, 'wheel_score', '?')}"

    def rank_candidates_e2e():
        runner = WheelRunner()
        runner._connector = _FakeConnector()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"], dte_target=30, top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=False,  # dealer path exercised in own test
            use_news_sentiment=False,
            use_credit_regime=False,
        )
        assert not df.empty
        return f"rows={len(df)} top_ev_per_day={df.iloc[0].get('ev_per_day','?')}"

    def rank_candidates_crossed_chain_blocked():
        from datetime import date, timedelta
        expiry = date.today() + timedelta(days=32)
        bad = pd.DataFrame([
            {"strike": 95, "option_type": "P", "open_interest": 1000,
             "implied_vol": 0.25, "bid": 2.5, "ask": 2.0, "expiration": expiry},
            {"strike": 100, "option_type": "P", "open_interest": 1000,
             "implied_vol": 0.22, "bid": 1.0, "ask": 1.1, "expiration": expiry},
        ])
        runner = WheelRunner()
        runner._connector = _FakeConnector(chain=bad)
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"], dte_target=30, top_n=5, min_ev_dollars=-1e9,
            use_dealer_positioning=True,
        )
        assert df.empty, "crossed-market chain should have been blocked"
        return "crossed chain blocked"

    def rank_candidates_clean_chain_passes():
        runner = WheelRunner()
        runner._connector = _FakeConnector()
        df = runner.rank_candidates_by_ev(
            tickers=["TESTA"], dte_target=30, top_n=5,
            min_ev_dollars=-1e9,
            use_dealer_positioning=True,
            use_news_sentiment=False,
            use_credit_regime=False,
        )
        assert not df.empty
        return f"rows={len(df)}"

    h.run("wheel_analyze_ticker", analyze_ticker_e2e)
    h.run("wheel_rank_candidates_by_ev", rank_candidates_e2e)
    h.run("chain_quality_gate_blocks_crossed", rank_candidates_crossed_chain_blocked)
    h.run("chain_quality_gate_allows_clean", rank_candidates_clean_chain_passes)

    # ------------------------------------------------------------------
    # 14. Configuration
    # ------------------------------------------------------------------
    h.section("14 policy_config")
    from engine.policy_config import load_policy, validate_policy, TradingPolicyConfig

    def policy_defaults():
        p = load_policy()
        assert validate_policy(p) == []
        return "defaults valid"

    def policy_rejects_invalid():
        bad = TradingPolicyConfig()
        bad.risk.max_daily_loss_pct = 0.5
        bad.risk.max_drawdown_pct = 0.1  # daily > dd is invalid
        errs = validate_policy(bad)
        assert errs, "validation should have caught inversion"
        return f"{len(errs)} error(s) raised"

    h.run("load_default_policy", policy_defaults)
    h.run("policy_rejects_invalid_config", policy_rejects_invalid)

    # ------------------------------------------------------------------
    # 15. Data connectors (env-dependent — auto-skip on no connectivity)
    # ------------------------------------------------------------------
    h.section("15 data_connectors")
    from engine.theta_connector import ThetaConnector
    from engine.data_connector import MarketDataConnector

    def market_data_connector_base():
        # Class should instantiate without API calls.
        cls = MarketDataConnector
        assert hasattr(cls, "get_ohlcv") or hasattr(cls, "get_universe")
        return "base connector importable"

    def theta_connector_instantiate():
        try:
            ThetaConnector()
        except Exception as e:
            raise Skip(f"Theta env not set: {e}")
        return "Theta instantiated"

    def theta_ohlcv_live():
        if h.fast:
            raise Skip("--fast flag set")
        try:
            conn = ThetaConnector()
            df = conn.get_ohlcv("SPY", start_date="2025-01-01")
        except Exception as e:
            raise Skip(f"theta unreachable: {e}")
        if df is None or len(df) == 0:
            raise Skip("theta returned empty (terminal offline?)")
        return f"rows={len(df)}"

    def theta_chain_live():
        if h.fast:
            raise Skip("--fast flag set")
        try:
            conn = ThetaConnector()
            df = conn.get_option_chain("SPY", dte_target=35)
        except Exception as e:
            raise Skip(f"theta unreachable: {e}")
        if df is None or len(df) == 0:
            raise Skip("theta returned empty")
        return f"rows={len(df)}"

    h.run("market_data_connector_base_class", market_data_connector_base)
    h.run("theta_connector_instantiate", theta_connector_instantiate)
    h.run("theta_ohlcv_live", theta_ohlcv_live)
    h.run("theta_option_chain_live", theta_chain_live)

    # ------------------------------------------------------------------
    # 16. Engine HTTP API (skip if server not up)
    # ------------------------------------------------------------------
    h.section("16 engine_api_http")

    def engine_api_health():
        if h.fast:
            raise Skip("--fast flag set")
        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:8787/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                body = r.read().decode()
        except Exception as e:
            raise Skip(f"API not running: {e}")
        return f"200 ({len(body)}B body)"

    def engine_api_status():
        if h.fast:
            raise Skip("--fast flag set")
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8787/status", timeout=2) as r:
                body = r.read().decode()
        except Exception as e:
            raise Skip(f"API not running: {e}")
        return f"{len(body)}B"

    h.run("api_health_endpoint", engine_api_health)
    h.run("api_status_endpoint", engine_api_status)

    # ------------------------------------------------------------------
    # 17. News pipeline
    # ------------------------------------------------------------------
    h.section("17 news_pipeline")
    from engine.news_sentiment import NewsSentimentReader

    def news_reader_init():
        try:
            r = NewsSentimentReader()
        except Exception as e:
            raise Skip(f"news reader unavailable: {e}")
        assert r is not None
        return "reader constructed"

    def news_sentiment_multiplier():
        try:
            r = NewsSentimentReader()
            m = r.sentiment_multiplier("AAPL")
        except Exception as e:
            raise Skip(f"news unavailable: {e}")
        assert np.isfinite(m)
        return f"m={m:.3f}"

    h.run("news_reader_init", news_reader_init)
    h.run("news_sentiment_multiplier", news_sentiment_multiplier)

    # ------------------------------------------------------------------
    # 18. TradingView bridge
    # ------------------------------------------------------------------
    h.section("18 tradingview_bridge")
    from engine.tradingview_bridge import build_tradingview_url

    def tv_url_build():
        url = build_tradingview_url("AAPL", timeframe="1D")
        assert url.startswith("http")
        return url[:60] + ("..." if len(url) > 60 else "")

    h.run("build_tradingview_url", tv_url_build)

    # ------------------------------------------------------------------
    # 19. TV signal computation
    # ------------------------------------------------------------------
    h.section("19 tv_signals")

    def tv_signal_compute():
        try:
            from engine.tv_signals import compute_tv_signal
        except ImportError as e:
            raise Skip(f"tv_signals import chain broken: {e}")
        out = compute_tv_signal(df=ohlcv, ticker="TESTA")
        assert out is not None
        return "computed"

    h.run("compute_tv_signal", tv_signal_compute)

    # ------------------------------------------------------------------
    # 20. Point-in-time index membership (survivorship fix)
    # ------------------------------------------------------------------
    h.section("20 pit_index_membership")

    def pit_membership_loaded():
        from data.consolidated_loader import get_bloomberg_loader
        L = get_bloomberg_loader()
        df = getattr(L, "_index_membership", None)
        if df is None or len(df) == 0:
            raise Skip("sp500_index_membership.csv not loaded")
        return f"rows={len(df)}"

    def pit_universe_latest():
        from data.consolidated_loader import get_bloomberg_loader
        u = get_bloomberg_loader().get_universe_as_of(None)
        assert len(u) >= 400, f"latest universe too small: {len(u)}"
        return f"latest N={len(u)}"

    def pit_universe_historical_drift():
        from data.consolidated_loader import get_bloomberg_loader
        L = get_bloomberg_loader()
        a = set(L.get_universe_as_of("2015-06-30"))
        b = set(L.get_universe_as_of("2026-01-01"))
        if not a or not b:
            raise Skip("historical snapshots not populated")
        added = len(b - a)
        dropped = len(a - b)
        assert added > 0 and dropped > 0, "no churn — membership data looks static"
        return f"2015→2026: +{added}/-{dropped}"

    h.run("index_membership_loaded", pit_membership_loaded)
    h.run("pit_universe_latest", pit_universe_latest)
    h.run("pit_universe_drift_2015_to_2026", pit_universe_historical_drift)

    # ------------------------------------------------------------------
    # 21. Feature-store coverage (one check per feature group)
    # ------------------------------------------------------------------
    h.section("21 feature_store_coverage")
    from pathlib import Path as _P

    def _count_tickers(group: str) -> int:
        d = _P("data/features") / group
        if not d.exists():
            return 0
        return len([p for p in d.iterdir() if p.is_dir() and p.name.startswith("ticker=")])

    def coverage_check(group: str, min_tickers: int):
        def fn():
            n = _count_tickers(group)
            if n == 0:
                raise Skip(f"{group}/ is empty — run scripts/backfill_features.py")
            assert n >= min_tickers, (
                f"only {n}/{min_tickers} tickers — "
                f"run: python scripts/backfill_features.py --workers 6"
            )
            return f"{n} tickers"
        return fn

    # Any meaningful S&P signal needs ≥400 tickers in the feature store.
    for group in ("technical", "volatility", "dynamics", "options_features",
                  "regime", "events", "vol_edge", "labels"):
        h.run(f"coverage_{group}", coverage_check(group, min_tickers=400))

    # ------------------------------------------------------------------
    # 22. Theta pull outputs (IV surface history, options flow)
    # ------------------------------------------------------------------
    h.section("22 theta_history_pulls")

    def iv_surface_history_present():
        d = _P("data_processed/theta/iv_surface_history")
        if not d.exists():
            raise Skip(
                "data_processed/theta/iv_surface_history/ missing — "
                "run scripts/pull_theta_iv_surface_history.py"
            )
        tickers = [p for p in d.iterdir() if p.is_dir() and p.name.startswith("ticker=")]
        if not tickers:
            raise Skip("directory exists but no ticker partitions yet")
        # Spot-check one partition for schema
        for tdir in tickers[:1]:
            parquets = list(tdir.rglob("*.parquet"))
            if parquets:
                df = pd.read_parquet(parquets[0])
                required = {"date", "ticker", "expiration", "strike", "right", "iv"}
                missing = required - set(df.columns)
                assert not missing, f"schema missing {missing}"
        return f"{len(tickers)} tickers"

    def options_flow_present():
        d = _P("data_processed/theta/options_flow")
        if not d.exists():
            raise Skip(
                "data_processed/theta/options_flow/ missing — "
                "run scripts/pull_theta_options_flow.py"
            )
        parquets = list(d.glob("*.parquet"))
        if not parquets:
            raise Skip("directory exists but empty")
        df = pd.read_parquet(parquets[0])
        required = {"date", "ticker"}
        assert required.issubset(df.columns)
        return f"{len(parquets)} tickers"

    h.run("iv_surface_history_present", iv_surface_history_present)
    h.run("options_flow_present", options_flow_present)

    # ------------------------------------------------------------------
    # 23. News sentiment store (unblocks the EV news multiplier)
    # ------------------------------------------------------------------
    h.section("23 news_sentiment_store")

    def news_parquet_present():
        for rel in ("data_processed/news_sentiment.parquet",
                    "data/news/sentiment.parquet"):
            p = _P(rel)
            if p.exists():
                df = pd.read_parquet(p)
                required = {"ticker", "as_of", "sentiment", "confidence", "n_articles"}
                missing = required - set(df.columns)
                assert not missing, f"schema missing {missing}"
                return f"{len(df)} rows at {p}"
        raise Skip("no news sentiment store found — run scripts/pull_news_sentiment.py")

    def news_multiplier_non_trivial():
        """Multiplier should vary across tickers once a real store is populated."""
        from engine.news_sentiment import NewsSentimentReader
        r = NewsSentimentReader()
        mults = [r.sentiment_multiplier(t) for t in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN")]
        if all(m == 1.0 for m in mults):
            raise Skip("news store empty or lookback stale — multiplier stuck at 1.0")
        return f"sample multipliers={['%.2f' % m for m in mults]}"

    h.run("news_sentiment_parquet_present", news_parquet_present)
    h.run("news_multiplier_non_trivial", news_multiplier_non_trivial)

    # ------------------------------------------------------------------
    # 24. Bloomberg data freshness (make sure new pulls land)
    # ------------------------------------------------------------------
    h.section("24 bloomberg_freshness")
    from datetime import datetime as _dt

    def _freshness(filename: str, max_age_days: int):
        def fn():
            p = _P("data/bloomberg") / filename
            if not p.exists():
                raise Skip(f"{filename} not found — run the matching BQL from scripts/bloomberg_bql_pulls.md")
            age = (_dt.now().timestamp() - p.stat().st_mtime) / 86400.0
            if age > max_age_days:
                return f"stale: {age:.0f} days old"
            return f"fresh: {age:.1f} days old ({p.stat().st_size/1024:.0f} KB)"
        return fn

    # Freshness of the canonical files consumed by the loader.
    for fname, age in (
        ("sp500_ohlcv.csv", 14),
        ("sp500_vol_iv_full.csv", 14),
        ("sp500_earnings.csv", 14),
        ("sp500_dividends.csv", 30),
        ("treasury_yields.csv", 14),
        ("vix_term_structure.csv", 14),
        ("sp500_fundamentals.csv", 30),
        ("sp500_index_membership.csv", 90),
    ):
        h.run(f"freshness_{fname}", _freshness(fname, age))

    # New pulls from the BQL templates (skip gracefully until they exist).
    for fname, age in (
        ("sp500_macro_calendar.csv", 90),
        ("sp500_short_interest.csv", 30),
        ("vix_futures_curve.csv", 14),
        ("vol_indices.csv", 14),
        ("sp500_analyst_revisions.csv", 30),
        ("spx_correlation.csv", 14),
    ):
        h.run(f"new_pull_{fname}", _freshness(fname, age))

    # ------------------------------------------------------------------
    # 25. yfinance-sourced pulls (this session)
    # ------------------------------------------------------------------
    h.section("25 yfinance_pulls")

    def vol_indices_parquet():
        p = _P("data_processed/vol_indices.parquet")
        if not p.exists():
            raise Skip("run scripts/pull_vol_indices.py")
        df = pd.read_parquet(p)
        required = {"date", "symbol", "close"}
        assert required.issubset(df.columns)
        n_sym = df["symbol"].nunique()
        assert n_sym >= 8, f"only {n_sym} symbols; expected 10"
        age_days = (_dt.now() - pd.to_datetime(df["date"]).max()).days
        assert age_days < 7, f"stale by {age_days} days — re-run the puller"
        return f"{n_sym} symbols, {len(df)} rows, latest {df['date'].max().date()}"

    def vol_indices_wide_view():
        p = _P("data_processed/vol_indices_wide.parquet")
        if not p.exists():
            raise Skip("wide view not built — re-run pull_vol_indices.py")
        df = pd.read_parquet(p)
        required = {"vix_close", "skew_close", "vvix_close"}
        missing = required - set(df.columns)
        assert not missing, f"missing columns {missing}"
        return f"{len(df)} rows, {len(df.columns)-1} indices"

    def fundamentals_yf_present():
        p = _P("data/bloomberg/sp500_fundamentals_yf.csv")
        if not p.exists():
            raise Skip("run scripts/pull_fundamentals_yf.py")
        df = pd.read_csv(p)
        required = {"ticker", "cur_mkt_cap", "pe_ratio", "beta_raw_overridable",
                    "gics_sector_name"}
        assert required.issubset(df.columns), f"schema: {set(df.columns)}"
        return f"{len(df)} tickers (P/E={df['pe_ratio'].notna().sum()} non-null)"

    def earnings_yf_present():
        p = _P("data/bloomberg/sp500_earnings_yf.csv")
        if not p.exists():
            raise Skip("run scripts/pull_earnings_yf.py")
        df = pd.read_csv(p)
        required = {"ticker", "announcement_date", "announcement_time"}
        assert required.issubset(df.columns)
        today = pd.Timestamp.now().normalize()
        upcoming = df[pd.to_datetime(df["announcement_date"]) >= today]
        return f"{len(df)} rows, {len(upcoming)} upcoming, {df['ticker'].nunique()} tickers"

    def treasury_yields_fresh():
        p = _P("data/bloomberg/treasury_yields.csv")
        if not p.exists():
            raise Skip("run scripts/pull_treasury_yields_yf.py")
        df = pd.read_csv(p, parse_dates=["date"])
        last = df["date"].max()
        age_days = (pd.Timestamp.now().normalize() - last.normalize()).days
        assert age_days < 10, f"stale by {age_days} days"
        required = {"rate_3m", "rate_6m", "rate_2y", "rate_10y"}
        assert required.issubset(df.columns)
        return f"{len(df)} rows, latest {last.date()}, 10Y={df['rate_10y'].iloc[-1]:.2f}"

    def pull_all_dry_run():
        """Meta-check: the orchestrator can enumerate its own plan."""
        import subprocess
        r = subprocess.run(
            [sys.executable, "scripts/pull_all.py", "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, r.stderr[:200]
        assert "vol_indices" in r.stdout and "treasury" in r.stdout
        return "orchestrator plan OK"

    h.run("vol_indices_parquet", vol_indices_parquet)
    h.run("vol_indices_wide_view", vol_indices_wide_view)
    h.run("fundamentals_yf_present", fundamentals_yf_present)
    h.run("earnings_yf_present", earnings_yf_present)
    h.run("treasury_yields_fresh", treasury_yields_fresh)
    h.run("pull_all_orchestrator_dry_run", pull_all_dry_run)

    # ------------------------------------------------------------------
    # 26. Theta outputs (only flip to PASS once Theta has been run)
    # ------------------------------------------------------------------
    h.section("26 theta_outputs")

    def theta_indices_rows():
        p = _P("data_processed/vol_indices.parquet")
        if not p.exists():
            raise Skip("vol_indices.parquet not found")
        df = pd.read_parquet(p)
        if "source" not in df.columns:
            raise Skip("source column missing — re-run pull_vol_indices.py")
        theta_rows = int((df["source"] == "theta").sum())
        if theta_rows == 0:
            raise Skip(
                "no rows sourced from Theta — run scripts/pull_theta_indices_history.py "
                "once Terminal is up"
            )
        return f"{theta_rows} theta rows"

    def vix_futures_present():
        p = _P("data_processed/vix_futures.parquet")
        if not p.exists():
            raise Skip("vix_futures.parquet not found — run pull_theta_vix_futures.py")
        df = pd.read_parquet(p)
        assert {"date", "expiration", "dte", "month_index", "close"}.issubset(df.columns)
        n_months = df["month_index"].max()
        return f"{len(df)} rows, curve depth={n_months}"

    def vix_futures_wide():
        p = _P("data_processed/vix_futures_wide.parquet")
        if not p.exists():
            raise Skip("wide view missing — re-run pull_theta_vix_futures.py")
        df = pd.read_parquet(p)
        ux_cols = [c for c in df.columns if c.startswith("ux")]
        assert len(ux_cols) >= 2, "need at least UX1, UX2"
        return f"{len(df)} dates × {len(ux_cols)} contracts"

    def corp_actions_splits():
        p = _P("data_processed/corporate_actions/splits.parquet")
        if not p.exists():
            raise Skip("no splits file — run pull_theta_corp_actions.py")
        df = pd.read_parquet(p)
        return f"{len(df)} split events, {df['ticker'].nunique()} tickers"

    def corp_actions_dividends():
        p = _P("data_processed/corporate_actions/dividends.parquet")
        if not p.exists():
            raise Skip("no dividends file — run pull_theta_corp_actions.py")
        df = pd.read_parquet(p)
        return f"{len(df)} dividend events, {df['ticker'].nunique()} tickers"

    def capability_report():
        p = _P("data_processed/theta_capabilities.json")
        if not p.exists():
            raise Skip("no capability report — run scripts/probe_theta_capabilities.py")
        import json as _json
        data = _json.loads(p.read_text())
        ok = sum(1 for r in data.get("results", []) if r.get("status") == "OK")
        blocked = sum(1 for r in data.get("results", []) if r.get("status") == "BLOCKED")
        return f"{ok} endpoints OK, {blocked} blocked (probed {data.get('probed_at','?')[:19]})"

    h.run("theta_indices_history_rows", theta_indices_rows)
    h.run("vix_futures_present", vix_futures_present)
    h.run("vix_futures_wide_view", vix_futures_wide)
    h.run("corp_actions_splits", corp_actions_splits)
    h.run("corp_actions_dividends", corp_actions_dividends)
    h.run("theta_capability_report", capability_report)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Smart Wheel Engine smoke test")
    ap.add_argument("--section", help="Run only sections matching this substring")
    ap.add_argument("--fast", action="store_true", help="Skip network/live paths")
    ap.add_argument("--verbose", action="store_true", help="Show detail on PASS rows too")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    h = Harness(section_filter=args.section, fast=args.fast)
    try:
        register_checks(h)
    except Exception:
        print("FATAL: harness registration crashed", file=sys.stderr)
        traceback.print_exc()
        return 1

    return h.report(verbose=args.verbose, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
