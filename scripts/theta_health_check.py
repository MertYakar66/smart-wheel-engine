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
    try:
        chain = conn.get_option_chain("SPY", dte_target=35)
        ok = not chain.empty and "iv" in chain.columns
        detail = f"rows={len(chain)}  cols={list(chain.columns)[:8]}"
        print(_fmt(ok, "Option chain snapshot (greeks+quote+OI)", detail))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Option chain snapshot", repr(e)))
        failures += 1

    # OI present
    try:
        has_oi = "open_interest" in chain.columns
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

    # Stock EOD
    try:
        df = conn.get_ohlcv("SPY", start_date="2024-01-01")
        ok = not df.empty and "close" in df.columns
        print(_fmt(ok, "Stock EOD OHLCV", f"rows={len(df)}"))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Stock EOD OHLCV", repr(e)))
        failures += 1

    # Stock intraday
    try:
        df = conn.get_stock_intraday("SPY", interval="5m")
        ok = not df.empty
        print(_fmt(ok, "Stock intraday bars (5m)", f"rows={len(df)}"))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "Stock intraday bars", repr(e)))
        failures += 1

    # IV rank
    try:
        rank = conn.get_iv_rank("SPY")
        ok = 0.0 <= rank <= 1.0
        print(_fmt(ok, "IV rank (live 1Y)", f"SPY={rank:.3f}"))
        failures += 0 if ok else 1
    except Exception as e:
        print(_fmt(False, "IV rank", repr(e)))
        failures += 1

    print()
    if failures == 0:
        print("All endpoints healthy — safe to run 'python -m scripts.theta_backfill all'")
        return 0
    else:
        print(f"{failures} check(s) failed. Verify subscription tier covers the failed endpoints.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
