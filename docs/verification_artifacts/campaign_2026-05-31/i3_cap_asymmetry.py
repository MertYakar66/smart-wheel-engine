"""I3-B: CONCENTRATION-CAP ASYMMETRY.

Two sub-questions:
  (B1) select_book's max_weight_per_name cap -- does it actually bind, and how
       does its semantics differ from the D17 single-name cap? Demonstrate.
  (B2) S24 finding: check_sector_cap counts OPTION-premium notional only, while
       check_portfolio_delta includes ASSIGNED STOCK delta. Construct a book
       where the same position is invisible to the sector cap but visible to the
       delta cap -- the asymmetry a senior trader must know.

READ-ONLY.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = L.ROOT


def main() -> None:
    print("=" * 78)
    print("I3-B  CONCENTRATION-CAP ASYMMETRY")
    print("=" * 78)

    # ---- B1: select_book max_weight_per_name semantics -------------------
    print("\n--- B1: select_book max_weight_per_name cap ---")
    from engine.wheel_runner import WheelRunner

    # Use a real ranked snapshot as the `ranking=` frame so no re-rank / no
    # EVEngine call is needed. select_book is a pure post-processor.
    snap = pd.read_parquet(os.path.join(HERE, "snapshots", "put_2021-06-01.parquet"))
    rk = snap[snap["ev_dollars"] > 0].copy()
    # select_book requires 'collateral' and 'ev_dollars'; snapshot has both.
    print(f"  ranking pool: {len(rk)} positive-EV rows, "
          f"collateral range ${rk['collateral'].min():,.0f}..${rk['collateral'].max():,.0f}")

    wr = WheelRunner()
    account = 200_000.0  # small account so the cap can bind
    for cap in (None, 0.25, 0.10):
        book = wr.select_book(account_size=account, ranking=rk.copy(),
                              max_weight_per_name=cap)
        attrs = dict(book.attrs)
        if len(book):
            max_name = (book["collateral"] / account).max()
        else:
            max_name = 0.0
        print(f"  cap={cap}: n_positions={attrs.get('n_positions')} "
              f"total_collateral=${attrs.get('total_collateral'):,.0f} "
              f"max_single_name_weight={max_name:.1%} "
              f"method={attrs.get('selection_method')}")
        if cap is not None:
            dropped = rk[rk["collateral"] > account * cap]
            print(f"        names dropped pre-selection for exceeding "
                  f"{cap:.0%}*account=${account * cap:,.0f}: "
                  f"{list(dropped['ticker'])}")

    print("\n  SEMANTICS NOTE: select_book.max_weight_per_name filters on a "
          "SINGLE contract's\n  COLLATERAL (strike*100) vs account_size. Because "
          "select_book emits at\n  most ONE entry per name (0/1 knapsack), the cap "
          "can only ever DROP a\n  too-expensive single name; it never aggregates "
          "across multiple positions\n  in the same name (there can't be any). It "
          "is also OFF by default (None).")

    # ---- B2: sector-cap (option-premium-only) vs delta-cap (incl stock) --
    print("\n--- B2: sector-cap excludes assigned stock; delta-cap includes it ---")
    from engine.portfolio_risk_gates import (
        check_portfolio_delta,
        check_sector_cap,
    )

    # Construct a book: a wheel that got ASSIGNED. The original short put on
    # ticker X converted to STOCK_OWNED (no option leg). take_snapshot would
    # then put X in stock_holdings, NOT option_positions. So:
    #   - check_sector_cap sees option_positions only -> X invisible.
    #   - check_portfolio_delta adds stock_holdings delta -> X fully counted.
    nav = 1_000_000.0
    # Held: 1000 shares of AAPL (assigned from a prior wheel) @ ~$150 = $150k
    # stock delta, PLUS one live short put on AAPL @ strike 150.
    stock_holdings = [("AAPL", 1000)]  # assigned shares
    held_option_positions = [
        {"symbol": "AAPL", "option_type": "put", "strike": 150.0, "dte": 35,
         "iv": 0.30, "contracts": 1, "is_short": True},
    ]
    spot_prices = {"AAPL": 150.0}
    candidate = {"symbol": "AAPL", "option_type": "put", "strike": 150.0,
                 "dte": 35, "iv": 0.30, "contracts": 1, "is_short": True}

    # Sector cap: held option notional for AAPL's sector. proposed = one put.
    sec = check_sector_cap(
        symbol="AAPL",
        proposed_notional=150.0 * 100.0,  # one contract
        held_option_positions=held_option_positions,
        nav=nav,
    )
    print(f"  check_sector_cap (sees ONLY option_positions, no stock):")
    print(f"     sector={sec.details.get('sector')} "
          f"post_open_sector_pct={sec.details.get('post_open_sector_pct'):.4%} "
          f"passed={sec.passed}")
    print(f"     -> the 1000 assigned AAPL shares (${stock_holdings[0][1] * 150:,.0f} "
          f"= {1000 * 150 / nav:.1%} NAV) are INVISIBLE to this gate.")

    dlt = check_portfolio_delta(
        held_option_positions=held_option_positions,
        spot_prices=spot_prices,
        candidate_option=candidate,
        stock_holdings=stock_holdings,
        nav=nav,
    )
    print(f"\n  check_portfolio_delta (adds stock_holdings delta-dollars):")
    print(f"     portfolio_delta_dollars (or current/post): {dlt.details} passed={dlt.passed}")
    # Show the stock contribution explicitly.
    stock_delta = sum(sh * spot_prices.get(t, 0.0) for t, sh in stock_holdings)
    print(f"     stock-leg delta-dollars alone = 1000 * $150 = ${stock_delta:,.0f}")
    print(f"     -> the SAME assigned shares the sector cap ignored fully "
          f"count here.")

    # Make the asymmetry bite: a book that PASSES sector cap (low option
    # notional) but would FAIL delta cap purely due to assigned stock.
    print("\n  --- asymmetry that BITES: pass sector, fail delta on stock alone ---")
    # No options at all; just a big pile of assigned stock in one name.
    big_stock = [("AAPL", 3000)]  # $450k of delta
    sec2 = check_sector_cap("AAPL", proposed_notional=0.0,
                            held_option_positions=[], nav=nav)
    dlt2 = check_portfolio_delta(
        held_option_positions=[], spot_prices={"AAPL": 150.0},
        candidate_option={}, stock_holdings=big_stock, nav=nav,
    )
    cap_dollars = 300.0 * (nav / 100_000.0)
    print(f"     3000 assigned AAPL shares = ${3000 * 150:,.0f} stock delta "
          f"({3000 * 150 / nav:.0%} NAV)")
    print(f"     sector cap: post_open_pct={sec2.details.get('post_open_sector_pct'):.1%} "
          f"passed={sec2.passed}  (option notional = $0, so it PASSES)")
    print(f"     delta cap: post_open_delta=${dlt2.details.get('post_open_delta_dollars', 'n/a')} "
          f"limit=${cap_dollars:,.0f} passed={dlt2.passed}")
    print(f"     -> CONFIRMS asymmetry: sector-cap blind to assigned stock; "
          f"delta-cap is not.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
