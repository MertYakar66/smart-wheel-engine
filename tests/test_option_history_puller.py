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

import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

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
    frame = pd.DataFrame({"strike": [100.0, 105.0], "right": ["put", "call"], "close": [1.0, 2.0]})
    rows, contracts = m._write_partition("AAPL", "20240119", [frame])
    part = tmp_path / "ticker=AAPL" / "expiration=20240119" / "data.parquet"
    assert part.exists() and part.stat().st_size > 0
    # No tmp litter left behind — the finally clears the (now per-writer-unique)
    # tmp whether or not the rename happened.
    assert not list(part.parent.glob("*.tmp"))
    assert rows == 2 and contracts == 2
    assert m._partition_exists("AAPL", "20240119")


def test_tmp_partition_path_is_unique_per_writer(tmp_path):
    # The race fix: two writers targeting the SAME partition dir must get
    # distinct tmp paths so neither clobbers the other's in-flight file before
    # the atomic rename.
    a = m._tmp_partition_path(tmp_path)
    b = m._tmp_partition_path(tmp_path)
    assert a != b
    assert a.parent == tmp_path and a.name.startswith("data.parquet.") and a.suffix == ".tmp"


def test_write_partition_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    # If the parquet write blows up mid-flight, the finally must clear the tmp so
    # a unique-named partial can't accumulate across crashes.
    monkeypatch.setattr(m, "OUT_ROOT", tmp_path)

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m.pd.DataFrame, "to_parquet", _boom)
    frame = pd.DataFrame({"strike": [100.0], "right": ["put"], "close": [1.0]})
    with pytest.raises(OSError):
        m._write_partition("AAPL", "20240119", [frame])
    outdir = tmp_path / "ticker=AAPL" / "expiration=20240119"
    assert not list(outdir.glob("*.tmp"))  # no leftover partial
    assert not (outdir / "data.parquet").exists()  # and nothing renamed into place


def test_cadence_filters():
    # monthly = 3rd Friday only (the load-bearing _is_third_friday gate).
    assert m._is_third_friday(pd.Timestamp("2024-01-19"))  # 3rd Fri Jan 2024
    assert not m._is_third_friday(pd.Timestamp("2024-01-05"))  # 1st Fri
    assert not m._is_third_friday(pd.Timestamp("2024-01-26"))  # 4th Fri
    # weekly = all Friday expirations (weekday()==4); drops Mon-Thu 0DTE dailies.
    assert pd.Timestamp("2024-01-05").weekday() == 4  # kept (Friday)
    assert pd.Timestamp("2024-01-17").weekday() != 4  # dropped (Wed 0DTE)


# ----------------------------------------------------------------------
# CMD 6 — Bloomberg per-name panel puller (scripts/_bbg_panel.py) hardening.
# Homed in this registered puller-correctness file: _bbg_panel.py has no
# dedicated test, and the TESTING.md taxonomy gate (out of CMD ownership)
# forbids a new unlisted test file. xbbg is not installed and the pull_*.py
# scripts run a real pull at import, so we stub xbbg and load _bbg_panel alone.
# ----------------------------------------------------------------------


def _load_bbg_panel():
    """Import scripts._bbg_panel with a stubbed xbbg (not installed in CI)."""
    fake = sys.modules.get("xbbg")
    if not isinstance(fake, types.ModuleType) or not hasattr(fake, "blp"):
        fake = types.ModuleType("xbbg")
        fake.blp = MagicMock()
        sys.modules["xbbg"] = fake
    import scripts._bbg_panel as bbg  # noqa: PLC0415

    return bbg


def _cfg(bbg, **over):
    kw = dict(
        out_name="x.csv",
        fields=["PX_LAST"],
        field_map={"PX_LAST": "v"},
        out_cols=["date", "ticker", "v"],
        start_date_full="2020-01-01",
        end_date="2020-12-31",
        chunk_size=2,
    )
    kw.update(over)
    return bbg.PanelConfig(**kw)


def test_bbg_pull_window_fails_loudly_on_chunk_error():
    """Bug 1: a per-chunk bdh exception must NOT be silently dropped — a chunk
    failure raises so a reduced universe can't masquerade as a clean pull.
    Pre-fix _pull_window printed ERROR + returned survivors/None (no raise)."""
    bbg = _load_bbg_panel()
    bbg.blp.bdh = MagicMock(side_effect=RuntimeError("bad/renamed ticker"))
    tickers = ["A Equity", "B Equity", "C Equity"]  # 2 chunks of size 2
    with pytest.raises(RuntimeError, match="failed to pull"):
        bbg._pull_window(_cfg(bbg), tickers, "2020-01-01", "2020-12-31")


def test_bbg_pull_window_ok_when_all_chunks_succeed():
    """Control: no chunk error → returns the concatenated frame, no raise."""
    bbg = _load_bbg_panel()

    def _bdh(tickers, flds, start_date, end_date, **kw):
        return pd.DataFrame(
            [
                {"date": "2020-06-01", "ticker": t, "field": "PX_LAST", "value": 10.0}
                for t in tickers
            ]
        )

    bbg.blp.bdh = MagicMock(side_effect=_bdh)
    out = bbg._pull_window(
        _cfg(bbg), ["A Equity", "B Equity", "C Equity"], "2020-01-01", "2020-12-31"
    )
    assert out is not None
    assert set(out["ticker"]) == {"A Equity", "B Equity", "C Equity"}


def test_bbg_resolve_end_date_advances_stale_literal():
    """Bug 2: a stale hardcoded end_date (no SWE_PULL_END override) advances to
    today so a plain refresh always plans a forward window; an explicit override
    wins; a fresh literal is unchanged. Pre-fix _resolve_end_date did not exist."""
    bbg = _load_bbg_panel()
    ed, advanced = bbg._resolve_end_date("2020-06-04", None, today="2026-07-15")
    assert ed == "2026-07-15" and advanced is True
    ed2, adv2 = bbg._resolve_end_date("2020-06-04", "2026-07-01", today="2026-07-15")
    assert ed2 == "2026-07-01" and adv2 is False  # explicit override wins
    ed3, adv3 = bbg._resolve_end_date("2026-07-20", None, today="2026-07-15")
    assert ed3 == "2026-07-20" and adv3 is False  # fresh literal unchanged
