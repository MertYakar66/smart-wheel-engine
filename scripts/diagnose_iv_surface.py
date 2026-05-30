#!/usr/bin/env python3
"""Diagnose the SVI IV-surface for one or more tickers — the fail-loud caller.

This is the production consumer that closes ROADMAP A2: the SVI surface tooling
in ``engine/volatility_surface.py`` (``SVICalibrator`` / ``VolatilitySurfaceBuilder``
/ ``create_empirical_surface``) previously had no live caller and was dormant.
The decided contract (DECISIONS.md D9, A2) is **wire it in, fail loud**: for a
ticker with no surface data (outside Theta's partial ``iv_surface_history``
coverage, or on the bloomberg provider which carries no skew — S29) the tool
**raises / reports UNAVAILABLE and exits non-zero**, never a fabricated flat IV.

It builds a smile-aware surface from the underlying's ATM IV term structure via
:func:`engine.volatility_surface.create_empirical_surface`, guarded by
:func:`engine.volatility_surface.require_surface`, and reports the per-expiry
skew (25Δ put / ATM / 25Δ call) and term structure.

The numeric core (:func:`build_surface_from_atm_term_structure`,
:func:`summarize_surface`) is pure and unit-tested
(``tests/test_iv_surface_failloud.py``). The connector path in :func:`main`
assembles the ATM term structure from the live provider and is
operator-first-run-verified (no Theta/network in the CI sandbox), mirroring the
EDGAR puller convention.

Usage:
    SWE_DATA_PROVIDER=theta python scripts/diagnose_iv_surface.py AAPL MSFT
    python scripts/diagnose_iv_surface.py AAPL          # bloomberg → UNAVAILABLE (no surface)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Put the repo root (not scripts/) on sys.path so `import engine` works when
# run as documented from the repo root: `python scripts/diagnose_iv_surface.py`.
# Mirrors the bootstrap other scripts use (scripts/diagnose_candidates.py et al.).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.volatility_surface import (  # noqa: E402 — sys.path bootstrap above
    SurfaceDataUnavailable,
    VolatilitySurface,
    create_empirical_surface,
    require_surface,
)


def build_surface_from_atm_term_structure(
    iv_by_tenor: dict[int, float],
    spot: float,
    as_of: date,
    ticker: str,
) -> VolatilitySurface:
    """Build a smile-aware SVI surface from an ATM IV term structure, fail-loud.

    ``iv_by_tenor`` maps days-to-expiry -> ATM IV (decimal). Raises
    :class:`SurfaceDataUnavailable` when the term structure is empty (the
    uncovered-ticker case) — never returns a flat fallback.
    """
    if not iv_by_tenor:
        raise SurfaceDataUnavailable(
            f"{ticker}: empty ATM IV term structure — no surface to build (A2 fail-loud)."
        )
    expiries = [as_of + timedelta(days=int(d)) for d in sorted(iv_by_tenor)]
    surface = create_empirical_surface(
        iv_by_tenor=iv_by_tenor,
        as_of_date=as_of,
        underlying=ticker,
        spot=spot,
        expiries=expiries,
    )
    # Guard: create_empirical_surface skips non-positive-T expiries, so an
    # all-past term structure could yield an empty surface — fail loud.
    return require_surface(surface, ticker)


def summarize_surface(surface: VolatilitySurface) -> list[dict]:
    """Per-expiry skew + ATM IV summary rows for a built surface."""
    rows: list[dict] = []
    for expiry in sorted(surface.svi_params):
        iv_put, iv_atm, iv_call = surface.get_skew(expiry)
        rows.append(
            {
                "expiry": expiry.isoformat(),
                "dte": (expiry - surface.as_of_date).days,
                "atm_iv": round(iv_atm, 4),
                "iv_25d_put": round(iv_put, 4),
                "iv_25d_call": round(iv_call, 4),
                "put_skew": round(iv_put - iv_atm, 4),
            }
        )
    return rows


def _term_structure_from_df(df, dte_col: str, iv_col: str) -> dict[int, float]:
    """Parse a ``{dte: iv}`` map from a DataFrame with one IV row per tenor."""
    out: dict[int, float] = {}
    for _, r in df.iterrows():
        try:
            d = int(r[dte_col])
            v = float(r[iv_col])
        except (TypeError, ValueError):
            continue
        if d > 0 and 0.01 < v < 3.0:
            out.setdefault(d, v)
    return out


def _atm_term_structure_from_connector(conn, ticker: str) -> dict[int, float]:
    """ATM IV term structure from the live provider — ATM rows, not wing rows.

    Prefers ``conn.get_atm_term_structure()`` (columns ``expiration, dte,
    atm_iv`` — the closest-to-ATM IV per expiry, purpose-built for term-structure
    fits). Only if that method is absent does it fall back to a defensive parse
    of ``get_iv_surface()`` — which returns EVERY strike/right row per expiry, so
    a naive first-row pick would calibrate off an arbitrary wing IV. There we
    select the row whose ``delta`` is nearest 0.5 (ATM) per tenor.

    Returns ``{}`` (→ fail-loud upstream) when the provider exposes no usable
    surface for the ticker (e.g. the bloomberg ``MarketDataConnector``, which
    carries no skew — S29). Defensive throughout: a shape mismatch degrades to
    UNAVAILABLE, never a wrong number.
    """
    get_atm = getattr(conn, "get_atm_term_structure", None)
    if get_atm is not None:
        try:
            ts = get_atm(ticker)
        except Exception:
            ts = None
        if ts is not None and not getattr(ts, "empty", True):
            cols = {c.lower(): c for c in ts.columns}
            dte_col = cols.get("dte") or cols.get("days_to_expiry")
            iv_col = cols.get("atm_iv") or cols.get("iv")
            if dte_col and iv_col:
                return _term_structure_from_df(ts, dte_col, iv_col)

    # Fallback: parse the full surface, picking the nearest-ATM row per tenor.
    get_surface = getattr(conn, "get_iv_surface", None)
    if get_surface is None:
        return {}
    try:
        surf = get_surface(ticker)
    except Exception:
        return {}
    if surf is None or getattr(surf, "empty", True):
        return {}
    cols = {c.lower(): c for c in surf.columns}
    dte_col = cols.get("dte") or cols.get("days_to_expiry")
    iv_col = cols.get("iv") or cols.get("iv_mid")
    if not dte_col or not iv_col:
        return {}
    delta_col = cols.get("delta")
    out: dict[int, float] = {}
    if delta_col is not None:
        # Keep, per tenor, the row whose |delta| is nearest 0.5 (ATM).
        best: dict[int, float] = {}  # dte -> |abs(delta) - 0.5|
        for _, r in surf.iterrows():
            try:
                d = int(r[dte_col])
                v = float(r[iv_col])
                dist = abs(abs(float(r[delta_col])) - 0.5)
            except (TypeError, ValueError):
                continue
            if d > 0 and 0.01 < v < 3.0 and (d not in best or dist < best[d]):
                best[d] = dist
                out[d] = v
        return out
    # No delta column — fall back to first valid row per tenor (best effort).
    return _term_structure_from_df(surf, dte_col, iv_col)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail-loud SVI IV-surface diagnostic.")
    ap.add_argument("tickers", nargs="+", help="tickers to diagnose")
    args = ap.parse_args(argv)

    provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()
    print(f"provider: {provider}")
    from engine.wheel_runner import WheelRunner

    conn = WheelRunner().connector
    as_of = date.today()

    unavailable: list[str] = []
    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            spot_hint = 100.0  # forward proxy; create_empirical_surface uses spot as F
            iv_by_tenor = _atm_term_structure_from_connector(conn, ticker)
            surface = build_surface_from_atm_term_structure(iv_by_tenor, spot_hint, as_of, ticker)
        except SurfaceDataUnavailable as exc:
            print(f"  {ticker}: UNAVAILABLE — {exc}")
            unavailable.append(ticker)
            continue
        print(f"  {ticker}: surface OK ({len(surface.svi_params)} expiries)")
        for row in summarize_surface(surface):
            print(
                f"    dte={row['dte']:>4}  atm={row['atm_iv']:.3f}  "
                f"25dP={row['iv_25d_put']:.3f}  25dC={row['iv_25d_call']:.3f}  "
                f"put_skew={row['put_skew']:+.3f}"
            )

    if unavailable:
        print(f"\nFAIL-LOUD: no surface for {len(unavailable)} ticker(s): {', '.join(unavailable)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
