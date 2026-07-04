#!/usr/bin/env python3
"""Repair the 2026-03-23 OHLCV + lockstep dividend split-scale splice (#439, D-W1-1).

BKNG (25:1, eff 2026-04-06) and CVNA (5:1, eff 2026-05-08) carry the split
adjustment only on the recent (>= 2026-03-23) pull slice; the older history is
at the un-split scale, so the committed series has a one-day ~/25 (BKNG) / ~/5
(CVNA) scale break at the 2026-03-23 pull boundary that poisons any
realized-vol / return / forward-distribution computation crossing it.

Fix = back-adjust the *full* pre-splice history onto the split-adjusted
(frontier) scale, the frame the engine already assumes
(engine/data_connector.py:909 "OHLCV is Bloomberg SPLIT-adjusted"):

  OHLCV (data/bloomberg/sp500_ohlcv.csv):
      rows with date < 2026-03-23 AND ticker in {BKNG UW Equity, CVNA UN Equity}
      -> open/high/low/close /= factor ; volume *= factor   (factor 25 / 5)
  DIVIDENDS (data/bloomberg/sp500_dividends.csv):
      BKNG UW rows -> dividend_amount /= 25                  (CVNA pays none)
      preserves dividend YIELD, keeping ex-div early-assignment economics intact
      (10.5 -> 0.42, the real post-split per-share level).

Properties: streaming (only target rows change, all else byte-identical),
CRLF-preserving, Decimal-exact (÷5 and ÷25 always terminate -> no rounding),
idempotency-guarded (refuses to run twice).

Split factors are the true corporate-action ratios (sp500_corporate_actions.csv),
NOT the raw boundary ratios (which fold in a genuine weekend move).

NOTE: the regression snapshots (s27/s32/s34/s35) pin the connector data SHA and
must be re-baselined by the regression/#402 lane against this corrected data;
this script does not touch them. See the #439 PR for the handoff.
"""

from __future__ import annotations

import sys
from decimal import Decimal

OHLCV = "data/bloomberg/sp500_ohlcv.csv"
DIVID = "data/bloomberg/sp500_dividends.csv"
BOUNDARY = "2026-03-23"  # rows strictly BEFORE this (the un-split history) get adjusted

# ohlcv ticker -> integer split factor (divide pre-splice prices, multiply volume)
OHLCV_FACTORS: dict[str, int] = {"BKNG UW Equity": 25, "CVNA UN Equity": 5}
# dividends ticker (no " Equity" suffix) -> per-share divisor
DIVID_FACTORS: dict[str, int] = {"BKNG UW": 25}

OHLCV_HEADER = "date,ticker,open,high,low,close,volume"
DIVID_HEADER = "declared_date,ex_date,record_date,payable_date,dividend_amount,dividend_frequency,dividend_type,ticker"


def _fmt(d: Decimal) -> str:
    """Canonical float-style string: exact, trailing zeros trimmed, keep one dp."""
    s = format(d.normalize(), "f")
    if "." not in s:
        s += ".0"
    return s


def _split_line(line: str) -> tuple[str, str]:
    """Return (content, line_ending) preserving CRLF/LF exactly."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _rewrite(path: str, header: str, transform, guard_ok) -> int:
    with open(path, encoding="utf-8", newline="") as fin:
        lines = fin.readlines()
    content0, _ = _split_line(lines[0])
    if content0 != header:
        print(f"!! {path}: unexpected header {content0!r}", file=sys.stderr)
        return -1
    if not guard_ok(lines):
        print(f"!! {path}: idempotency guard tripped — already fixed? aborting.", file=sys.stderr)
        return -1
    out = [lines[0]]
    changed = 0
    for line in lines[1:]:
        content, eol = _split_line(line)
        new = transform(content)
        if new is None:
            out.append(line)
        else:
            out.append(new + eol)
            changed += 1
    with open(path, "w", encoding="utf-8", newline="") as fout:
        fout.writelines(out)
    return changed


def _ohlcv_transform(content: str):
    p = content.split(",")
    if len(p) != 7:
        return None
    date, ticker, o, h, lo, c, v = p
    if ticker not in OHLCV_FACTORS or date >= BOUNDARY:
        return None
    f = Decimal(OHLCV_FACTORS[ticker])
    no, nh, nl, nc = (_fmt(Decimal(x) / f) for x in (o, h, lo, c))
    nv = _fmt(Decimal(v) * f)
    return f"{date},{ticker},{no},{nh},{nl},{nc},{nv}"


def _ohlcv_guard(lines: list[str]) -> bool:
    # BKNG 2026-03-20 must still be un-split (close col > 1000) before we run
    for line in lines:
        if line.startswith("2026-03-20,BKNG UW Equity,"):
            close = float(_split_line(line)[0].split(",")[5])
            return close > 1000.0
    return True


def _divid_transform(content: str):
    p = content.split(",")
    if len(p) != 8:
        return None
    ticker = p[7]
    if ticker not in DIVID_FACTORS:
        return None
    p[4] = _fmt(Decimal(p[4]) / Decimal(DIVID_FACTORS[ticker]))
    return ",".join(p)


def _divid_guard(lines: list[str]) -> bool:
    for line in lines:
        if line.rstrip("\r\n").endswith(",BKNG UW"):
            amt = float(_split_line(line)[0].split(",")[4])
            return amt > 1.0  # un-split per-share dividends are >> 1
    return True


def main() -> int:
    n_o = _rewrite(OHLCV, OHLCV_HEADER, _ohlcv_transform, _ohlcv_guard)
    if n_o < 0:
        return 2
    n_d = _rewrite(DIVID, DIVID_HEADER, _divid_transform, _divid_guard)
    if n_d < 0:
        return 2
    print(f"OHLCV rows back-adjusted: {n_o} (BKNG+CVNA pre-{BOUNDARY})")
    print(f"BKNG dividend rows /25:   {n_d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
