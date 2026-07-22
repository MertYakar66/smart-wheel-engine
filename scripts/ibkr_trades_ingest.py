"""IBKR Flex **Trades** XML -> normalized ``trades.json`` (read-only, observational).

Parses an IBKR Flex Web Service *Activity / Trades* statement (the raw XML saved
by the dashboard's Flex fetch) into the flat, documented trade schema the
dashboard's Trades tab consumes. Every execution — buys, sells, **expiries
(`Ep`)** and **assignments (`A`)** — is preserved with its FIFO realized P&L, so
per-ticker options gain/loss is accurate (the connector feed misses expiries;
Flex does not — see ``docs/DASHBOARD_TERMINAL.md`` §3.3).

**Scope contract (CLAUDE.md §2/§3).** Strictly read-only and observational.
Imports nothing from the decision trio, never ranks a candidate, never issues an
EV token. Output lands in the gitignored ``data_processed/ibkr`` runtime dir;
real account data is never committed.

**Merge-safe.** Each trade gets a stable ``id`` (hash of the immutable execution
fields) so re-ingesting an overlapping/wider Flex pull (e.g. a since-inception
range) dedupes rather than double-counts — call :func:`ingest` with
``merge_into`` pointing at an existing ``trades.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _f(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_dt(raw: str) -> tuple[str | None, str | None]:
    """``"20260407;104909"`` -> ``("2026-04-07T10:49:09", "2026-04-07")``."""
    if not raw:
        return None, None
    raw = raw.replace(",", ";").strip()
    date_part, _, time_part = raw.partition(";")
    date_part = date_part.strip()
    if len(date_part) != 8 or not date_part.isdigit():
        return None, None
    d = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
    tp = time_part.strip().replace(":", "")
    if len(tp) == 6 and tp.isdigit():
        return f"{d}T{tp[:2]}:{tp[2:4]}:{tp[4:6]}", d
    return f"{d}T00:00:00", d


def _fmt_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _trade_key(a: dict) -> str:
    """Base content key. This Flex query emits no ``tradeID``, and IBKR fills
    two identical legs of one order as byte-identical ``<Trade>`` rows (e.g. a
    2-lot sold as two qty=-1 rows). So the key alone is NOT unique — the caller
    appends a per-key occurrence ordinal (stable in Flex's chronological order)
    to keep identical fills distinct AND dedupe cleanly across overlapping pulls."""
    key = "|".join(
        str(a.get(k, ""))
        for k in (
            "dateTime",
            "symbol",
            "buySell",
            "quantity",
            "tradePrice",
            "proceeds",
            "expiry",
            "strike",
        )
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_trade(el: ET.Element) -> dict:
    a = el.attrib
    dt, date = _fmt_dt(a.get("dateTime", ""))
    sec_type = (a.get("assetCategory") or "").upper()
    symbol_raw = (a.get("symbol") or "").strip()
    underlying = symbol_raw.split()[0].upper() if symbol_raw else ""
    right = (a.get("putCall") or "").upper() or None
    return {
        "_key": _trade_key(a),
        "datetime": dt,
        "date": date,
        "symbol": underlying,
        "occ_symbol": symbol_raw if sec_type == "OPT" else None,
        "sec_type": sec_type or "OTHER",
        "right": right,
        "strike": _f(a.get("strike")),
        "expiry": _fmt_date(a.get("expiry")),
        "side": (a.get("buySell") or "").upper() or None,
        "qty": _f(a.get("quantity")),
        "price": _f(a.get("tradePrice")),
        "proceeds": _f(a.get("proceeds")),
        "commission": _f(a.get("ibCommission")),
        "realized_pnl": _f(a.get("fifoPnlRealized")) or 0.0,
        "currency": (a.get("ibCommissionCurrency") or "").upper() or None,
        "open_close": (a.get("openCloseIndicator") or "").upper() or None,
        "code": (a.get("notes") or a.get("code") or "").strip() or None,
        "multiplier": 100 if sec_type == "OPT" else 1,
    }


def parse_flex_xml(xml_path: str | Path) -> list[dict]:
    root = ET.fromstring(Path(xml_path).read_text(encoding="utf-8"))
    out: list[dict] = []
    occ: dict[str, int] = {}
    for el in root.findall(".//Trade"):
        rec = parse_trade(el)
        base = rec.pop("_key")
        n = occ.get(base, 0)
        occ[base] = n + 1
        # occurrence ordinal disambiguates identical fills and stays stable
        # across overlapping pulls (Flex order is chronological).
        rec["id"] = hashlib.sha1(f"{base}:{n}".encode()).hexdigest()[:16]
        out.append(rec)
    return out


def ingest(
    xml_path: str | Path,
    *,
    merge_into: str | Path | None = None,
    source: str = "ibkr_flex",
    query: str | None = None,
) -> dict:
    trades = parse_flex_xml(xml_path)
    by_id: dict[str, dict] = {}
    if merge_into and Path(merge_into).exists():
        for t in json.loads(Path(merge_into).read_text(encoding="utf-8")).get("trades", []):
            by_id[t["id"]] = t
    for t in trades:
        by_id[t["id"]] = t  # new pull wins on collision (same execution)
    merged = sorted(by_id.values(), key=lambda t: t.get("datetime") or "")
    dates = [t["date"] for t in merged if t.get("date")]
    return {
        "schema_version": 1,
        "source": source,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coverage": {
            "from": min(dates) if dates else None,
            "to": max(dates) if dates else None,
            "query": query,
            "count": len(merged),
        },
        "note": (
            "Normalized IBKR Flex Trades (buys/sells/expiries[Ep]/assignments[A]) "
            "by scripts/ibkr_trades_ingest.py. realized_pnl = fifoPnlRealized in the "
            "trade's native currency. Observational (CLAUDE.md §2/§3)."
        ),
        "trades": merged,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="raw Flex Trades XML")
    ap.add_argument("--out", required=True, help="destination trades.json")
    ap.add_argument("--merge-into", default=None, help="existing trades.json to merge/dedupe into")
    ap.add_argument("--query", default=None, help="provenance label for the Flex query/period")
    args = ap.parse_args(argv)
    doc = ingest(args.xml, merge_into=args.merge_into or args.out, query=args.query)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cov = doc["coverage"]
    print(f"wrote {out}  trades={cov['count']}  range={cov['from']}..{cov['to']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
