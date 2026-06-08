"""Tests for the Phase-4 exact-fill ledger reconstruction
(scripts/ibkr_flex_ledger.py). The real Flex CSVs are gitignored (operator
account data), so these exercise the pure components: the Open/Close-driven
long/short StockBook, the FX builder, and the two-file boundary dedup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_flex_ledger", _REPO / "scripts" / "ibkr_flex_ledger.py"
)
flex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(flex)


# --------------------------------------------------------- StockBook long
def test_stockbook_long_roundtrip_realizes_gain():
    b = flex.StockBook()
    b.event(100, 100.0, 0.0, "O", "2025-01-02")  # open long 100 @ 100
    b.event(-100, 110.0, 0.0, "C", "2025-02-02")  # close long 100 @ 110
    assert abs(b.net_qty) < 1e-9
    assert len(b.closes) == 1
    entry, exit_, realized, _comm = b.closes[0]
    assert entry == "2025-01-02" and exit_ == "2025-02-02"
    assert realized == pytest.approx(1000.0)


# --------------------------------------------------------- StockBook short
def test_stockbook_short_roundtrip_realizes_gain():
    b = flex.StockBook()
    b.event(-100, 100.0, 0.0, "O", "2025-03-01")  # open short 100 @ 100
    b.event(100, 90.0, 0.0, "C", "2025-03-15")  # cover short 100 @ 90
    assert abs(b.net_qty) < 1e-9
    assert len(b.closes) == 1
    _e, _x, realized, _c = b.closes[0]
    assert realized == pytest.approx(1000.0)  # bought back cheaper


# ------------------------------------------------ StockBook ACAT seed + sale
def test_stockbook_acat_seed_sale():
    # 40 CLS in at basis 3,346.38 (83.66/sh) then sold 40 @ 208.37
    b = flex.StockBook(seed_qty=40, seed_basis=3346.38, seed_date="2025-04-03")
    b.event(-40, 208.37, 0.0, "C", "2025-06-25")
    assert abs(b.net_qty) < 1e-9
    _e, _x, realized, _c = b.closes[0]
    assert realized == pytest.approx((208.37 - 3346.38 / 40) * 40, abs=0.01)


def test_stockbook_ends_long_after_open():
    b = flex.StockBook()
    b.event(100, 50.0, 0.0, "O", "2025-01-01")
    assert b.net_qty == pytest.approx(100.0)
    assert b.held_basis == pytest.approx(5000.0)


# ------------------------------------------------------------- helpers
def test_iso8_and_days():
    assert flex.iso8("20250718;103247") == "2025-07-18"
    assert flex.days_between("2025-07-18", "2025-08-18") == 31


def test_underlying():
    assert flex.underlying("CLS   250718C00210000") == "CLS"
    assert flex.underlying("CLS") == "CLS"


# --------------------------------------------------- two-file boundary dedup
def _write_csv(path, rows):
    header = (
        "AssetClass,Symbol,Strike,DateTime,Put/Call,Proceeds,IBCommission,"
        "Open/CloseIndicator,Buy/Sell,CurrencyPrimary,Expiry,Quantity,TradePrice,OrigTradeID"
    )
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_fills_drops_boundary_overlap(tmp_path):
    # File A ends at 20260310;100903; B repeats that boundary fill + adds a later one.
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    boundary = "STK,CLS,,20260310;100903,,1000,-1,C,SELL,USD,,-100,269.54,"
    _write_csv(a, ["STK,NVDA,,20250401;120000,,5000,-1,O,SELL,USD,,-100,50,", boundary])
    _write_csv(b, [boundary, "STK,AMD,,20260401;120000,,9000,-1,O,SELL,USD,,-100,90,"])
    fills, na, nb, dropped, a_max = flex.load_fills(str(a), str(b))
    assert a_max == "20260310;100903"
    assert dropped == 1  # the duplicated boundary row from B
    assert len(fills) == na + (nb - dropped) == 3
    syms = sorted(r["Symbol"] for r in fills)
    assert syms == ["AMD", "CLS", "NVDA"]  # boundary CLS counted once
