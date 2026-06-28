"""#402 root-cause attribution — does the #363 IV-sentinel gate touch the
s27/s32/s34 regression windows? (Windows main, 2026-06-28; diagnostic only.)

#402 reports snapshot drift (ev_mean / spearman_p) vs the #338 baseline and
names the **#363 IV-sentinel connector gate** (NULLs served vol_iv outside the
``(_IV_LOW_FLOOR, _DEEP_IV_SENTINEL_FLOOR] == (3.0, 10000]`` band, see
``engine/data_connector.py:_clean_vol_iv_inplace``) as the prime suspect.

This characterizes the gate's FOOTPRINT on the exact windows the three drifting
reproducers consume: s27/s32 = UNIVERSE_24 over 2022-2024, s34 = UNIVERSE_100
over 2024. If the gate NULLs zero in-window cells, it cannot move those
reproducers' metrics via the PIT ``get_iv_history`` path → the prime suspect is
refuted for the vol_iv path. (The #369/#430 ``_clean_served_iv`` fundamentals-
fallback path is a separate, rarely-fired branch for well-covered names.)

Read-only; no engine/data edits. Tickers are normalized exactly as the connector
does on load (``normalize_ticker`` over the raw Bloomberg "AAPL UW" forms) — the
gate runs post-normalization, so the scan must too.

Usage: SWE_DATA_PROVIDER=bloomberg python .../diag_s402_iv_gate_footprint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass

import pandas as pd  # noqa: E402

from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100  # noqa: E402
from engine.data_connector import MarketDataConnector, normalize_ticker  # noqa: E402

VOL_IV = WORKTREE / "data" / "bloomberg" / "sp500_vol_iv_full.csv"
LOW = MarketDataConnector._IV_LOW_FLOOR  # 3.0
HIGH = MarketDataConnector._DEEP_IV_SENTINEL_FLOOR  # 10_000.0
IV_COLS = MarketDataConnector._DEEP_IV_COLS  # ("hist_put_imp_vol", "hist_call_imp_vol")

WINDOWS = [
    ("s27/s32 (UNIVERSE_24, 2022-2024)", set(UNIVERSE_24), "2022-01-03", "2024-12-31"),
    ("s34 (UNIVERSE_100, 2024)", set(UNIVERSE_100), "2024-01-02", "2024-12-31"),
]


def main() -> int:
    df = pd.read_csv(VOL_IV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tn"] = df["ticker"].map({t: normalize_ticker(t) for t in df["ticker"].dropna().unique()})
    cols = [c for c in IV_COLS if c in df.columns]

    out: dict = {
        "gate_band": {"low_floor": LOW, "high_floor": HIGH},
        "vol_iv_rows": int(len(df)),
        "vol_iv_cols": cols,
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "windows": {},
    }
    print(f"vol_iv: {len(df)} rows {out['date_range']} cols={cols}")
    for label, uni, lo, hi in WINDOWS:
        sub = df[df["tn"].isin(uni) & df["date"].between(lo, hi)]
        matched = sorted(set(sub["tn"]) & uni)
        w: dict = {"rows": int(len(sub)), "tickers_matched": len(matched), "tickers_total": len(uni)}
        print(f"\n== {label}: {len(sub)} rows, {len(matched)}/{len(uni)} tickers ==")
        for c in cols:
            v = pd.to_numeric(sub[c], errors="coerce")
            nn = v.notna()
            low = (v <= LOW) & nn
            high = (v > HIGH) & nn
            gated = low | high
            w[c] = {
                "nonnull": int(nn.sum()),
                "gated": int(gated.sum()),
                "gated_pct": round(100 * gated.sum() / max(1, nn.sum()), 6),
                "low_le_floor": int(low.sum()),
                "high_gt_floor": int(high.sum()),
            }
            print(f"  {c}: nonnull={int(nn.sum())} gated={int(gated.sum())} "
                  f"({w[c]['gated_pct']}%)  low<=3={int(low.sum())} high>1e4={int(high.sum())}")
        out["windows"][label] = w

    total_gated = sum(
        w[c]["gated"] for w in out["windows"].values() for c in cols if c in w
    )
    out["verdict"] = (
        "REFUTED — #363 gate NULLs zero in-window vol_iv cells; it cannot move "
        "s27/s32/s34 metrics via the PIT path"
        if total_gated == 0
        else f"gate touches {total_gated} in-window cells — quantify metric impact"
    )
    print("\nVERDICT:", out["verdict"])
    (Path(__file__).parent / "iv_gate_footprint.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
