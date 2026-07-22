"""Tests for the Trades-tab data path: Flex-XML ingest + trades_view aggregation.

Guards the two properties that make per-ticker options P&L trustworthy:
  1. **No trade lost or double-counted.** IBKR fills identical legs of one order
     as byte-identical <Trade> rows (no tradeID in this query) — the occurrence
     ordinal must keep them distinct, and a re-ingest of the same XML must be
     idempotent.
  2. **Currency-correct aggregation.** Realized P&L sums per ticker in native
     currency, and the USD-equivalent uses per-trade FX so a dual-listed name
     (CLS: NYSE-USD + TSX-CAD) is flagged mixed and totalled correctly.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.ibkr_portfolio_adapter import trades_view  # noqa: E402
from scripts.ibkr_trades_ingest import ingest, parse_flex_xml  # noqa: E402

# Two identical CLS option sell-to-opens in one order (byte-identical rows),
# a CLS option expiry (Ep, kept premium), a CLS stock buy, and a CAD-listed
# CLS option (dual-listing) — the exact shapes that break naive dedup / FX.
_XML = """<FlexQueryResponse>
 <FlexStatements><FlexStatement><Trades>
  <Trade assetCategory="OPT" symbol="CLS   260821C00310000" putCall="C" strike="310"
         expiry="20260821" buySell="SELL" quantity="-1" tradePrice="2.0" proceeds="200"
         ibCommission="-1.05" ibCommissionCurrency="USD" fifoPnlRealized="0"
         openCloseIndicator="O" dateTime="20260401;100000" notes=""/>
  <Trade assetCategory="OPT" symbol="CLS   260821C00310000" putCall="C" strike="310"
         expiry="20260821" buySell="SELL" quantity="-1" tradePrice="2.0" proceeds="200"
         ibCommission="-1.05" ibCommissionCurrency="USD" fifoPnlRealized="0"
         openCloseIndicator="O" dateTime="20260401;100000" notes=""/>
  <Trade assetCategory="OPT" symbol="CLS   260410C00300000" putCall="C" strike="300"
         expiry="20260410" buySell="BUY" quantity="1" tradePrice="0" proceeds="0"
         ibCommission="0" ibCommissionCurrency="USD" fifoPnlRealized="400"
         openCloseIndicator="C" dateTime="20260410;160000" notes="Ep"/>
  <Trade assetCategory="STK" symbol="CLS" buySell="BUY" quantity="100" tradePrice="300"
         proceeds="-30000" ibCommission="-1" ibCommissionCurrency="USD" fifoPnlRealized="0"
         openCloseIndicator="O" dateTime="20260405;120000" notes=""/>
  <Trade assetCategory="OPT" symbol="CLS   260501C00320000" putCall="C" strike="320"
         expiry="20260501" buySell="BUY" quantity="1" tradePrice="1.0" proceeds="100"
         ibCommission="-1" ibCommissionCurrency="CAD" fifoPnlRealized="-100"
         openCloseIndicator="C" dateTime="20260415;100000" notes=""/>
 </Trades></FlexStatement></FlexStatements>
</FlexQueryResponse>"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "flex.xml"
    p.write_text(_XML, encoding="utf-8")
    return p


def test_identical_fills_preserved(tmp_path):
    trades = parse_flex_xml(_write(tmp_path))
    assert len(trades) == 5  # nothing collapsed
    assert len({t["id"] for t in trades}) == 5  # ids all unique
    cls_shorts = [t for t in trades if t["strike"] == 310.0]
    assert len(cls_shorts) == 2  # both identical sell-to-opens survive


def test_option_fields_and_underlying(tmp_path):
    trades = parse_flex_xml(_write(tmp_path))
    opt = next(t for t in trades if t["strike"] == 310.0)
    assert opt["symbol"] == "CLS"  # underlying from the OCC symbol
    assert opt["sec_type"] == "OPT"
    assert opt["right"] == "C"
    assert opt["expiry"] == "2026-08-21"  # YYYYMMDD -> ISO
    assert opt["multiplier"] == 100
    exp = next(t for t in trades if t["code"] == "Ep")
    assert exp["realized_pnl"] == 400.0  # expiry keeps premium


def test_ingest_merge_idempotent(tmp_path):
    xml = _write(tmp_path)
    out = tmp_path / "trades.json"
    doc1 = ingest(xml)
    out.write_text(__import__("json").dumps(doc1), encoding="utf-8")
    doc2 = ingest(xml, merge_into=out)  # re-ingest same pull
    assert len(doc2["trades"]) == len(doc1["trades"]) == 5  # no growth
    assert doc1["coverage"]["from"] == "2026-04-01"
    assert doc1["coverage"]["to"] == "2026-04-15"


def test_trades_view_currency_correct(tmp_path):
    doc = ingest(_write(tmp_path))
    view = trades_view(doc, fx_rates={"CAD": 0.5})  # exaggerated rate for clarity
    cls = next(t for t in view["tickers"] if t["symbol"] == "CLS")
    assert cls["mixed_currency"] is True  # USD + CAD legs
    assert cls["currency"] == "USD"  # dominant
    # native options P&L: +400 (USD expiry) + (-100 CAD) summed raw = 300
    assert cls["opt_pnl"] == 300.0
    # USD-equivalent: +400*1.0 + (-100)*0.5 = 350
    assert cls["opt_pnl_usd"] == 350.0
    assert cls["stk_pnl"] == 0.0
    assert cls["opt_count"] == 4
    assert cls["stk_count"] == 1
    # premium collected = credit from the two sell-to-opens (200+200)
    assert cls["premium_collected"] == 400.0


def test_trades_view_totals(tmp_path):
    doc = ingest(_write(tmp_path))
    view = trades_view(doc, fx_rates={"CAD": 0.5})
    assert view["totals"]["trade_count"] == 5
    assert view["totals"]["ticker_count"] == 1
    assert view["source"] == "ibkr_flex"
