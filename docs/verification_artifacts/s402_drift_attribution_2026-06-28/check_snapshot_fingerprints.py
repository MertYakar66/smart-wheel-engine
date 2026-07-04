"""#402 decisive check — do the committed s27/s32/s34 snapshot fingerprints match
current data EXACTLY? (Windows main, 2026-06-28; diagnostic only.)

Stronger and far faster than re-running the multi-hour backtests: the regression
snapshots embed the full input fingerprint, the engine is deterministic, and the
last EV-moving merge that regenerated them was #433 (everything since is the
gitignored option-premium rail, absent here → synthetic-BSM path → a no-op on
these reproducers). So if the committed fingerprint equals the live data
fingerprint, the reproducers produce the committed metrics BY CONSTRUCTION — an
exact identity, not a within-tolerance pass. All-MATCH ⇒ #402's drift (measured
vs the #338 baseline) is resolved/stale.

Read-only. Usage: SWE_DATA_PROVIDER=bloomberg python .../check_snapshot_fingerprints.py
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

from backtests.regression._common import (  # noqa: E402
    connector_data_sha256,
    ohlcv_sha256,
    treasury_sha256,
    vol_iv_sha256,
)

SNAPSHOTS = ("s27_ivpit_24t_100k", "s32_friction_24t_1m", "s34_universe_100t_1m")
SNAP_DIR = WORKTREE / "backtests" / "regression" / "snapshots"


def main() -> int:
    cur = {
        "data_csv_sha256": ohlcv_sha256(),
        "vol_iv_sha256": vol_iv_sha256(),
        "treasury_sha256": treasury_sha256(),
    }
    cur_cd = connector_data_sha256()
    out: dict = {"snapshots": {}, "all_match": True}
    for sid in SNAPSHOTS:
        fp = json.loads((SNAP_DIR / f"{sid}.json").read_text()).get("fingerprint", {})
        rec: dict = {}
        for k, c in cur.items():
            rec[k] = "MATCH" if fp.get(k) == c else "MISMATCH"
        scd = fp.get("connector_data_sha256", {})
        diffs = (
            [k for k in set(scd) | set(cur_cd) if scd.get(k) != cur_cd.get(k)]
            if isinstance(scd, dict)
            else ["<snapshot has no connector_data dict>"]
        )
        rec["connector_data_sha256"] = "MATCH ALL" if not diffs else f"DIFFS: {diffs}"
        rec["n_connector_files"] = len(cur_cd)
        if "MISMATCH" in rec.values() or diffs:
            out["all_match"] = False
        out["snapshots"][sid] = rec
        print(f"{sid}: {rec}")
    out["verdict"] = (
        "ALL-MATCH — reproducers produce committed metrics by construction; #402 stale"
        if out["all_match"]
        else "MISMATCH — a fingerprinted input changed; investigate the named file(s)"
    )
    print("\nVERDICT:", out["verdict"])
    (Path(__file__).parent / "fingerprint_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
