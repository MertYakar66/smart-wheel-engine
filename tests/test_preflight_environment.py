"""Preflight environment invariants — fail fast on a stale / wrong-provenance tree.

This guard automates the two session-start environment checks that CLAUDE.md §4
tells every agent to do by hand — and that, when skipped, have repeatedly cost
this project cycles:

1. **Silent provider selection** (CLAUDE.md §4.1). The data provider is chosen
   from ``SWE_DATA_PROVIDER`` at runtime; a silent switch away from the expected
   ``MarketDataConnector`` is a known recurring bug source. This pins (and logs)
   the resolved provider.
2. **Stale tree / wrong clone.** Reading code+data from an older clone instead of
   current ``main`` produced the "OHLCV is 79 days stale" premise and a
   fingerprint false-positive in earlier work. A loud, actionable failure when
   the bundled OHLCV ends before the pinned frontier catches that instantly.

Design contract (so the guard helps, never hinders):
* **Fast** — instantiate the runner; read only the OHLCV ``date`` column.
* **Deterministic** — a pinned frontier constant, not ``date.today()``.
* **Self-skipping** — skips when the provider is intentionally non-default
  (``SWE_DATA_PROVIDER=theta``) or when the bundled data is simply absent.
* **Messages diagnose** — they say what is wrong AND what to do.

Deliberately NOT included (kept additive + non-flaky):
* fingerprint/_FILES completeness — already guarded by
  ``tests/test_data_integrity_bloomberg.py::test_fingerprint_pins_every_connector_file``
  (no duplication);
* a trio-vs-``origin/main`` provenance diff — needs network + would false-fail on
  any legitimate decision-layer branch; a flaky preflight hinders, so it is out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path("data/bloomberg")
_OHLCV = DATA_DIR / "sp500_ohlcv.csv"
HAS_BLOOMBERG_DATA = _OHLCV.exists()
_SNAPSHOT_BDP = DATA_DIR / "broad_pull" / "per_name" / "sp500_snapshot_bdp.csv"

# Pinned data frontier — the most-recent OHLCV bar current ``main`` is expected
# to carry. BUMP THIS in the same commit as every data refresh
# (docs/DATA_POLICY.md §5). A tree whose OHLCV ends before this is almost
# certainly stale or the wrong clone — which is exactly what this guard catches.
EXPECTED_FRONTIER = pd.Timestamp("2026-06-04")

# Pinned knowledge date of the broad-pull per-name snapshot whose
# ``next_earnings_dt`` column feeds the live earnings lockout (the D3-1
# forward-calendar overlay in ``engine/data_connector.py``). BUMP THIS in the
# same commit as every broad-pull snapshot refresh. Deterministic on purpose
# (this file's contract forbids ``date.today()``): it catches a stale tree /
# wrong clone, while wall-clock DECAY of the calendar is covered by the
# opt-in live preflight (``SWE_LIVE_PREFLIGHT=1`` in
# tests/test_earnings_calendar_overlay.py) and a runtime connector warning.
EXPECTED_EARNINGS_CALENDAR_ASOF = pd.Timestamp("2026-06-18")


@pytest.mark.skipif(
    os.environ.get("SWE_DATA_PROVIDER", "bloomberg").strip().lower() == "theta",
    reason="SWE_DATA_PROVIDER=theta selected intentionally — default-provider preflight N/A",
)
def test_default_provider_is_market_data_connector():
    """The default / ``bloomberg`` provider must resolve to ``MarketDataConnector``.

    A silent switch to another connector is a recurring bug source (CLAUDE.md
    §4.1: "Always log which provider was actually selected"). The resolved
    provider is printed so it appears in the run log either way.
    """
    from engine.wheel_runner import WheelRunner

    provider = os.environ.get("SWE_DATA_PROVIDER", "(unset -> bloomberg)")
    conn_type = type(WheelRunner().connector).__name__
    print(f"[preflight] SWE_DATA_PROVIDER={provider} -> connector={conn_type}")
    assert conn_type == "MarketDataConnector", (
        f"Expected MarketDataConnector for the bloomberg/default provider, got "
        f"{conn_type!r} (SWE_DATA_PROVIDER={provider}). Silent provider selection is a "
        "recurring bug — set SWE_DATA_PROVIDER explicitly and confirm the data layer "
        "(CLAUDE.md §4.1)."
    )


@pytest.mark.skipif(
    not HAS_BLOOMBERG_DATA,
    reason="no bundled data/bloomberg/sp500_ohlcv.csv — nothing to date-check",
)
def test_bundled_ohlcv_reaches_expected_frontier():
    """The bundled OHLCV must reach the pinned frontier.

    A tree ending earlier is almost certainly STALE or the wrong clone — the
    class of mistake that produced the "79 days stale" premise and a fingerprint
    false-positive when code/data were read from an older clone instead of main.
    """
    dates = pd.to_datetime(pd.read_csv(_OHLCV, usecols=["date"])["date"], errors="coerce")
    data_max = dates.max()
    assert pd.notna(data_max), f"OHLCV {_OHLCV} has no parseable dates — corrupt or wrong file."
    assert data_max >= EXPECTED_FRONTIER, (
        f"OHLCV ends {str(data_max)[:10]}, expected >= {EXPECTED_FRONTIER.date()} — you may be "
        "on a STALE tree or the wrong clone. Verify current main (`git fetch`; `git log -1`). "
        "If this is an intentional refresh that moved the frontier, BUMP EXPECTED_FRONTIER in "
        "this file in the same commit (docs/DATA_POLICY.md §5)."
    )


@pytest.mark.skipif(
    not _SNAPSHOT_BDP.exists(),
    reason="no bundled broad_pull snapshot — nothing to date-check",
)
def test_bundled_earnings_calendar_reaches_expected_asof():
    """The broad-pull snapshot feeding the live earnings lockout must carry
    the pinned knowledge date.

    An earlier ``asof`` means a stale tree / wrong clone — and unlike most
    stale data, a stale earnings calendar fails OPEN: its forward dates fall
    behind the query date and simply stop registering on the event gate, so
    the lockout silently reverts to the ~8 % coverage that was finding D3-1.
    """
    asof = pd.to_datetime(pd.read_csv(_SNAPSHOT_BDP, usecols=["asof"])["asof"], errors="coerce")
    asof_max = asof.max()
    assert pd.notna(asof_max), f"{_SNAPSHOT_BDP} has no parseable asof — corrupt or wrong file."
    assert asof_max >= EXPECTED_EARNINGS_CALENDAR_ASOF, (
        f"broad-pull snapshot asof is {str(asof_max)[:10]}, expected >= "
        f"{EXPECTED_EARNINGS_CALENDAR_ASOF.date()} — stale tree or wrong clone. If this is an "
        "intentional refresh, BUMP EXPECTED_EARNINGS_CALENDAR_ASOF in this file in the same "
        "commit. NOTE: this pin cannot catch wall-clock decay — before live use, run the "
        "opt-in preflight (SWE_LIVE_PREFLIGHT=1 pytest "
        "tests/test_earnings_calendar_overlay.py) which bounds the snapshot's age."
    )
