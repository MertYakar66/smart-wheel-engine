#!/usr/bin/env python3
"""Dashboard terminal — one-command live refresh of the read-only IBKR portfolio
viewer's data files. Companion to ``docs/DASHBOARD_TERMINAL.md``.

READ-ONLY / observational (CLAUDE.md sec 2/3): never places or modifies orders,
never imports the decision trio (ev_engine / wheel_runner / candidate_dossier),
never commits account data. The connector *account* pulls are agent-time MCP
tools, so the agent saves those three JSONs first; this script does the
deterministic remainder — regenerate the snapshot, rebuild the Flex ledger
(incl. option expiries), sync the equity-curve live point, and verify.

Data dir = ``$SWE_IBKR_DATA_DIR`` (the engine's source of truth; gitignored).
Backs up every file before overwriting (``_backup/``) and never writes the Flex
token anywhere.

Usage:
  # 1) Agent saves the 3 read-only connector JSONs into <dir>:
  #      get_account_summary->summary.json  get_account_balances->balances.json
  #      get_account_positions->positions.json
  # 2) Then:
  python scripts/dashboard_refresh.py all      --inputs <dir>
  python scripts/dashboard_refresh.py snapshot --inputs <dir>
  python scripts/dashboard_refresh.py ledger
  python scripts/dashboard_refresh.py verify
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
for _p in (str(_ROOT), str(_SCRIPTS)):  # so sibling scripts + engine import on direct invocation
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ibkr_flex_ledger as flex  # noqa: E402
import ibkr_live_snapshot as live  # noqa: E402


def data_dir() -> Path:
    return Path(os.environ.get("SWE_IBKR_DATA_DIR") or (_ROOT / "data_processed" / "ibkr"))


def _read(p: Path | str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _backup(p: Path) -> None:
    if p.exists():
        bk = p.parent / "_backup"
        bk.mkdir(exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(p, bk / f"{p.stem}.{ts}{p.suffix}")


# ---------------------------------------------------------------- snapshot
def do_snapshot(inputs: Path) -> dict:
    D = data_dir()
    snap = live.build_snapshot(
        _read(inputs / "summary.json"),
        _read(inputs / "balances.json"),
        _read(inputs / "positions.json"),
        reference=live.load_reference(),
        include_day_change=False,  # gated until reconciled vs the IBKR app
    )
    sp = D / "portfolio_snapshot.json"
    _backup(sp)
    sp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    _sync_curve(D, snap)
    print(
        f"[snapshot] netLiq={snap['account']['net_liquidation']:.0f} "
        f"legs={len(snap['positions'])} as_of={snap['as_of']}"
    )
    return snap


def _spy_closes(days: int = 120) -> dict[str, float] | None:
    """EOD SPY closes keyed ``YYYY-MM-DD`` via the local read-only IB Gateway.

    Best-effort: returns None (never raises) when the Gateway is down/logged
    out — the curve sync then writes ``spy: null`` for new days instead of
    carrying a stale value forward (PR #403 review F9: the old carry-forward
    drew a fake flat benchmark for as long as the freeze lasted).
    Read-only contract: ``readonly=True`` + a historical-data read only.
    """
    try:
        from ib_insync import IB, Stock

        ib = IB()
        ib.connect("127.0.0.1", 4001, clientId=23, readonly=True, timeout=10)
        try:
            c = Stock("SPY", "SMART", "USD")
            ib.qualifyContracts(c)
            bars = ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr=f"{days} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            out = {b.date.isoformat(): float(b.close) for b in bars if b.close and b.close > 0}
            return out or None
        finally:
            ib.disconnect()
    except Exception:
        return None


def _close_on_or_before(closes: dict[str, float], iso: str) -> float | None:
    """Nearest trading-day close at or before *iso* (point dates can be
    non-trading days only in pathological data; normally exact hits)."""
    if iso in closes:
        return closes[iso]
    prior = [d for d in closes if d <= iso]
    return closes[max(prior)] if prior else None


def _derive_spy(pts: list[dict], iso: str, closes: dict[str, float]) -> int | None:
    """Indexed-benchmark value for *iso*: walk back to the last point holding
    a real numeric ``spy`` + a date covered by the closes window, then scale
    by the SPY close ratio. None when the chain cannot be derived honestly."""
    for p in reversed(pts):
        v, d = p.get("spy"), p.get("date")
        if isinstance(v, (int, float)) and v > 0 and isinstance(d, str):
            px_anchor = _close_on_or_before(closes, d)
            px_t = _close_on_or_before(closes, iso)
            if px_anchor and px_t:
                return round(v * px_t / px_anchor)
            return None
    return None


def _repair_spy_tail(pts: list[dict], closes: dict[str, float]) -> int:
    """Re-derive the trailing run of carried-forward (byte-identical) ``spy``
    values from real SPY closes, anchored at the run's first point (the last
    REAL value). Heals the freeze the old ``_sync_curve`` carry-forward wrote;
    idempotent once values differ day to day. Returns points repaired."""
    if len(pts) < 2:
        return 0
    r = len(pts) - 1
    while r > 0 and pts[r].get("spy") is not None and pts[r].get("spy") == pts[r - 1].get("spy"):
        r -= 1
    repaired = 0
    anchor = pts[r]
    av, ad = anchor.get("spy"), anchor.get("date")
    if not (isinstance(av, (int, float)) and av > 0 and isinstance(ad, str)):
        return 0
    px_a = _close_on_or_before(closes, ad)
    if not px_a:
        return 0
    for p in pts[r + 1 :]:
        d = p.get("date")
        if not isinstance(d, str):
            continue
        px = _close_on_or_before(closes, d)
        if px:
            new_v = round(av * px / px_a)
            if p.get("spy") != new_v:
                p["spy"] = new_v
                repaired += 1
    return repaired


def _sync_curve(D: Path, snap: dict) -> None:
    hp = D / "portfolio_history.json"
    if not hp.exists():
        return
    hist = _read(hp)
    pts = hist.get("points") or []
    if not pts:
        return
    nav = round(float(snap["account"]["net_liquidation"]))
    iso = snap["as_of"][:10]
    dt = datetime.fromisoformat(iso)
    label = f"{dt.strftime('%b')} {dt.day}"  # cross-platform "Mon D"
    same_day = pts[-1].get("date") == iso

    # Real benchmark extension: derive today's indexed SPY value from actual
    # closes (and heal any previously-frozen tail) instead of copying the
    # prior point forever. No Gateway -> spy is null for a NEW day (the chart
    # renders a gap), and a same-day refresh keeps the morning's real value.
    closes = _spy_closes()
    healed = _repair_spy_tail(pts, closes) if closes else 0
    if closes:
        spy_val = _derive_spy(pts[:-1] if same_day else pts, iso, closes)
    else:
        spy_val = None
    if spy_val is None and same_day:
        prev_v = pts[-1].get("spy")
        spy_val = prev_v if isinstance(prev_v, (int, float)) else None

    new = {
        "label": label,
        "date": iso,
        "port": nav,
        "spy": spy_val,
        "premium": pts[-1].get("premium", 0) if same_day else 0,
    }
    pts[-1] = new if same_day else pts[-1]
    if not same_day:
        pts.append(new)
    hist["as_of"] = snap["as_of"]
    _backup(hp)
    hp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(
        f"[curve] live point {label} port={nav} spy={spy_val if spy_val is not None else 'null'}"
        f"{f' (healed {healed} frozen benchmark points)' if healed else ''}"
        f"{' [no Gateway - benchmark gap]' if closes is None and not same_day else ''}"
    )


def do_spy_repair(closes_file: str | None = None) -> None:
    """Standalone benchmark repair: heal the frozen carried-forward SPY tail
    in the canonical history from real closes, without touching the
    snapshot/ledger. Backs up first; engine re-reads the file per request.

    Closes come from the local IB Gateway (Mode 2) or, when the Gateway is
    logged out (its daily habit — runbook §3.2), from ``--closes <file>``: a
    ``{"YYYY-MM-DD": close}`` JSON saved agent-time from the cloud connector's
    ``get_price_history`` (Mode 1) — the same two-mode split as the snapshot.
    """
    D = data_dir()
    hp = D / "portfolio_history.json"
    if not hp.exists():
        raise SystemExit(f"[spy-repair] no history at {hp}")
    if closes_file:
        raw = _read(closes_file)
        closes = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and v > 0}
    else:
        closes = _spy_closes()
    if not closes:
        raise SystemExit(
            "[spy-repair] no closes available - IB Gateway down and no --closes file "
            "(pull SPY daily history via the connector and pass it)"
        )
    hist = _read(hp)
    pts = hist.get("points") or []
    healed = _repair_spy_tail(pts, closes)
    if healed:
        _backup(hp)
        hp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"[spy-repair] healed {healed} frozen benchmark points ({len(closes)} closes loaded)")


# ---------------------------------------------------------------- ledger (Flex)
_SEND = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/"
    "SendRequest?t={t}&q={q}&v=3"
)
_COLS = [
    "DateTime",
    "AssetClass",
    "Symbol",
    "Strike",
    "Expiry",
    "Put/Call",
    "Quantity",
    "TradePrice",
    "Proceeds",
    "IBCommission",
    "CurrencyPrimary",
    "Open/CloseIndicator",
]
_MAP = {
    "DateTime": "dateTime",
    "AssetClass": "assetCategory",
    "Symbol": "symbol",
    "Strike": "strike",
    "Expiry": "expiry",
    "Put/Call": "putCall",
    "Quantity": "quantity",
    "TradePrice": "tradePrice",
    "Proceeds": "proceeds",
    "IBCommission": "ibCommission",
    "CurrencyPrimary": "ibCommissionCurrency",
    "Open/CloseIndicator": "openCloseIndicator",
}


def _http(url: str) -> str:
    with urllib.request.urlopen(url, timeout=90) as r:  # noqa: S310 (fixed IBKR https host)
        return r.read().decode("utf-8", "replace")


def _flex_fetch(token: str, query_id: str) -> str:
    r = ET.fromstring(_http(_SEND.format(t=token, q=query_id)))
    if r.findtext("Status") != "Success":
        raise SystemExit(
            f"[flex] SendRequest failed: {r.findtext('ErrorCode')} {r.findtext('ErrorMessage')}"
        )
    ref, base = r.findtext("ReferenceCode"), r.findtext("Url")
    get = f"{base}?t={token}&q={ref}&v=3"
    for _ in range(12):
        xml = _http(get)
        if xml.lstrip().startswith("<FlexQueryResponse"):
            return xml
        time.sleep(5)  # statement generation in progress
    raise SystemExit("[flex] statement not ready after retries")


def _xml_to_csv(xml: str, csv_a: Path, csv_b: Path) -> int:
    trades = ET.fromstring(xml).findall(".//Trade")
    with csv_a.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for t in trades:
            w.writerow({c: (t.get(_MAP[c]) or "") for c in _COLS})
    with csv_b.open("w", newline="", encoding="utf-8") as fh:  # header-only dedup partner
        csv.DictWriter(fh, fieldnames=_COLS).writeheader()
    return len(trades)


def do_ledger() -> None:
    D = data_dir()
    creds = _read(D / "flex_credentials.json")
    tmp = Path(os.environ.get("TEMP") or "/tmp") / "swe_dash_flex"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        xml = _flex_fetch(str(creds["token"]), str(creds["query_id"]))
        a, b = tmp / "a.csv", tmp / "b.csv"
        n = _xml_to_csv(xml, a, b)
        _backup(D / "wheel_ledger.json")
        _backup(D / "portfolio_history.json")
        flex.AS_OF = date.today().strftime("%Y%m%d")  # re-run keys to today
        closed = flex.build(str(a), str(b), str(D))
        ytd = sum(c["net_pnl"] for c in closed if c["exit_date"][:4] == str(date.today().year))
        wins = sum(1 for c in closed if c["net_pnl"] > 0)
        print(
            f"[ledger] trades={n} cycles={len(closed)} "
            f"winRate={wins / len(closed):.3f} realizedYTD={ytd:,.0f}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # scrub account-trade temp


# ---------------------------------------------------------------- verify
def do_verify() -> None:
    from engine import ibkr_portfolio_adapter as A  # heavy import; only here

    D = str(data_dir())
    p = A.build_all(D)
    s = p["summary"]
    pos = p["positions"]
    legs = pos.get("legs") or pos.get("holdings") or []  # robust to pre-/post-all-legs engine
    print(
        f"[verify] netLiq={s['netLiq']} realizedYtd={s['realizedYtd']} "
        f"premium30d={s['premium30d']} winRate={s['winRate']} "
        f"unrealized={s['unrealizedPnl']} legs={len(legs)} "
        f"source={A.provenance(A.load_snapshot(D))} asOf={s['asOf']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["all", "snapshot", "ledger", "verify", "spy-repair"])
    ap.add_argument("--inputs", help="dir with summary.json / balances.json / positions.json")
    ap.add_argument(
        "--closes",
        help="spy-repair only: JSON file {YYYY-MM-DD: close} from an agent-time connector pull (fallback when the Gateway is logged out)",
    )
    a = ap.parse_args(argv)
    if a.cmd == "spy-repair":
        do_spy_repair(a.closes)
        return 0
    if a.cmd in ("all", "snapshot"):
        if not a.inputs:
            ap.error("--inputs is required for 'all' / 'snapshot'")
        do_snapshot(Path(a.inputs))
    if a.cmd in ("all", "ledger"):
        do_ledger()
    if a.cmd in ("all", "verify"):
        do_verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
