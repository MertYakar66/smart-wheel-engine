"""
ThetaData Terminal health probe.

Exercises every v3 endpoint the engine uses and prints a one-line
pass/fail per check so you can confirm the Terminal and subscription
are set up correctly before running the bulk backfill.

Usage
-----
    python scripts/theta_health_check.py

Exits 0 on full green, 1 if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.theta_connector import ThetaConnector  # noqa: E402


def _fmt(ok: bool, label: str, detail: str = "") -> str:
    mark = "OK  " if ok else "FAIL"
    return f"[{mark}] {label:<35} {detail}"


def main() -> int:
    conn = ThetaConnector()
    failures = 0

    alive = conn.is_terminal_alive()
    print(_fmt(alive, "Terminal reachable", "127.0.0.1:25503"))
    if not alive:
        print("\nStart Terminal with:  java -jar ThetaTerminalv3.jar <email> <password>")
        return 1

    # Expirations
    try:
        exp = conn._nearest_expiration("SPY", dte_target=35)
        print(_fmt(exp is not None, "SPY expirations", f"nearest~35DTE: {exp}"))
        failures += 0 if exp else 1
    except Exception as e:
        print(_fmt(False, "SPY expirations", repr(e)))
        failures += 1

    # Chain snapshot
    chain = pd.DataFrame()
    try:
        chain = conn.get_option_chain("SPY", dte_target=35)
        has_iv = "iv" in chain.columns
        has_delta = "delta" in chain.columns
        has_bid = "bid" in chain.columns
        ok = not chain.empty and has_iv and has_delta
        detail = (
            f"rows={len(chain)} iv={has_iv} delta={has_delta} bid={has_bid}"
        )
        print(_fmt(ok, "Option chain snapshot (greeks+quote+OI)", detail))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Option chain snapshot", repr(e)))
        failures += 1

    # OI present
    try:
        has_oi = "open_interest" in chain.columns and not chain.empty
        print(_fmt(has_oi, "Open interest merged", f"present={has_oi}"))
        failures += 0 if has_oi else 1
    except Exception:
        pass

    # IV surface (full)
    try:
        surf = conn.get_iv_surface("SPY", max_expirations=4)
        ok = not surf.empty
        print(_fmt(ok, "Full IV surface", f"rows={len(surf)}"))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Full IV surface", repr(e)))
        failures += 1

    # VIX family
    try:
        fam = conn.get_vix_family()
        detail = ", ".join(f"{k}={v:.2f}" for k, v in fam.items())
        print(_fmt(bool(fam), "VIX family snapshot", detail))
        failures += 0 if fam else 1
    except Exception as e:
        print(_fmt(False, "VIX family snapshot", repr(e)))
        failures += 1

    # Stock EOD — Theta first, fall back to Bloomberg CSV (engine parent class)
    try:
        # Direct Theta call (avoid automatic fallback so we can distinguish)
        raw = conn._fetch(
            "/v3/stock/history/eod",
            {
                "symbol": "SPY",
                "start_date": "20240101",
                "end_date": pd.Timestamp.now().strftime("%Y%m%d"),
            },
        )
        from engine.data_connector import MarketDataConnector
        bloomberg = MarketDataConnector(str(conn._data_dir)).get_ohlcv("SPY", start_date="2024-01-01")
        theta_ok = not raw.empty
        bb_ok = bloomberg is not None and not bloomberg.empty
        ok = theta_ok or bb_ok  # engine works as long as ONE path works
        src = "theta" if theta_ok else ("bloomberg-fallback" if bb_ok else "none")
        rows = len(raw) if theta_ok else (len(bloomberg) if bb_ok else 0)
        print(_fmt(ok, "Stock EOD OHLCV", f"source={src} rows={rows}"))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Stock EOD OHLCV", repr(e)))
        failures += 1

    # Stock intraday — Theta-only (Bloomberg CSV is daily only)
    try:
        df = conn.get_stock_intraday("SPY", interval="5m")
        ok = not df.empty
        tier_msg = "needs Stocks tier" if not ok else f"rows={len(df)}"
        print(_fmt(ok, "Stock intraday bars (5m)", tier_msg))
        # Intraday is optional — nice-to-have, not required
        if not ok:
            print("       (optional — needed only for GK/YZ realised vol; engine uses daily RV otherwise)")
    except Exception as e:
        print(_fmt(False, "Stock intraday bars", repr(e)))

    # IV rank — Theta live first, Bloomberg CSV fallback
    try:
        rank = conn.get_iv_rank("SPY")
        ok = 0.0 <= rank <= 1.0
        if ok:
            print(_fmt(True, "IV rank (live 1Y)", f"SPY={rank:.3f}"))
        else:
            # Check Bloomberg CSV fallback
            from engine.data_connector import MarketDataConnector
            bb_rank = MarketDataConnector(str(conn._data_dir)).get_iv_rank("SPY")
            bb_ok = 0.0 <= bb_rank <= 1.0 if bb_rank == bb_rank else False
            if bb_ok:
                print(_fmt(True, "IV rank", f"source=bloomberg-fallback SPY={bb_rank:.3f}"))
            else:
                print(_fmt(False, "IV rank", "SPY=nan in both theta & bloomberg"))
                failures += 1
    except Exception as e:
        print(_fmt(False, "IV rank", repr(e)))
        failures += 1

    print()
    if failures == 0:
        print("All required endpoints healthy (fallbacks engaged where needed).")
        print("Safe to run:  python -m scripts.theta_backfill all")
        return 0
    else:
        print(
            f"{failures} required check(s) failed. The engine cannot run without a\n"
            "working price source — check Bloomberg CSVs exist under data/bloomberg/."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
