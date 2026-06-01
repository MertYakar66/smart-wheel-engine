"""Pin the Theta option-history puller's correctness fixes.

Two bugs found 2026-06-01 that these tests prevent from regressing:

1. **History-floor clamp.** ``option/history/eod`` returns EMPTY for the WHOLE
   range if ``start_date`` precedes the STANDARD tier's 2016-01-01 data floor.
   The puller's "expiry - 210 days" start lands in 2015 for early-2016
   expirations, which silently zeroed every early-2016 pull. The fix clamps the
   start to ``_THETA_HISTORY_FLOOR``.

2. **Atomic partition writes.** Resume is file-exists (+ size>0) based, so a
   crash mid-write must not leave a partial ``data.parquet`` that the next run
   skips as "done". The write goes to a tmp file then atomically renames.
"""
import pandas as pd

import scripts.pull_theta_option_history as m


class _RecordingConn:
    """Captures the params passed to ``_fetch``; returns a tiny EOD frame."""

    def __init__(self):
        self.calls = []

    def _fetch(self, path, params):
        self.calls.append((path, dict(params)))
        if "eod" in path:
            return pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "expiration": [params["expiration"]],
                    "strike": [100.0],
                    "right": ["put"],
                    "created": ["2016-01-04"],
                    "close": [1.0],
                }
            )
        return pd.DataFrame()  # no OI


def _eod_start(conn):
    return [p for path, p in conn.calls if "eod" in path][0]["start_date"]


def test_start_clamped_to_floor_for_early_2016():
    # 20160115 - 210d ~ 2015-06-19, BELOW the floor -> must clamp to 2016-01-01.
    conn = _RecordingConn()
    m._fetch_expiration_bulk(conn, "AAPL", "20160115", include_oi=False)
    assert _eod_start(conn) == "20160101"


def test_start_not_clamped_when_above_floor():
    conn = _RecordingConn()
    m._fetch_expiration_bulk(conn, "AAPL", "20200117", include_oi=False)
    expected = (pd.Timestamp("20200117") - pd.Timedelta(days=210)).strftime("%Y%m%d")
    assert _eod_start(conn) == expected
    assert expected > "20160101"  # i.e. genuinely unclamped


def test_lookback_days_sets_the_window():
    # 90-day lookback from a 2024 expiry (well above the floor) -> start = exp-90d.
    conn = _RecordingConn()
    m._fetch_expiration_bulk(conn, "AAPL", "20240119", include_oi=False, lookback_days=90)
    expected = (pd.Timestamp("20240119") - pd.Timedelta(days=90)).strftime("%Y%m%d")
    assert _eod_start(conn) == expected


def test_write_partition_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "OUT_ROOT", tmp_path)
    frame = pd.DataFrame(
        {"strike": [100.0, 105.0], "right": ["put", "call"], "close": [1.0, 2.0]}
    )
    rows, contracts = m._write_partition("AAPL", "20240119", [frame])
    part = tmp_path / "ticker=AAPL" / "expiration=20240119" / "data.parquet"
    assert part.exists() and part.stat().st_size > 0
    assert not (part.parent / "data.parquet.tmp").exists()  # no partial left behind
    assert rows == 2 and contracts == 2
    assert m._partition_exists("AAPL", "20240119")


def test_cadence_filters():
    # monthly = 3rd Friday only (the load-bearing _is_third_friday gate).
    assert m._is_third_friday(pd.Timestamp("2024-01-19"))      # 3rd Fri Jan 2024
    assert not m._is_third_friday(pd.Timestamp("2024-01-05"))  # 1st Fri
    assert not m._is_third_friday(pd.Timestamp("2024-01-26"))  # 4th Fri
    # weekly = all Friday expirations (weekday()==4); drops Mon-Thu 0DTE dailies.
    assert pd.Timestamp("2024-01-05").weekday() == 4           # kept (Friday)
    assert pd.Timestamp("2024-01-17").weekday() != 4           # dropped (Wed 0DTE)
