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
    new = {
        "label": label,
        "date": iso,
        "port": nav,
        "spy": pts[-1].get("spy"),
        "premium": pts[-1].get("premium", 0) if same_day else 0,
    }
    pts[-1] = new if same_day else pts[-1]
    if not same_day:
        pts.append(new)
    hist["as_of"] = snap["as_of"]
    _backup(hp)
    hp.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"[curve] live point {label} port={nav}")


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
    ap.add_argument("cmd", choices=["all", "snapshot", "ledger", "verify"])
    ap.add_argument("--inputs", help="dir with summary.json / balances.json / positions.json")
    a = ap.parse_args(argv)
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
