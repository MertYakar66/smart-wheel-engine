"""R7 — deep-IV sentinel nulling on the assembled vol_iv read.

The deep vol_iv panels carry a corrupt implied-vol sentinel ~134217.7
(≈ 2**27/1000), 1994-95 + a few delisted names. The deep-read intake nulls the
implied-vol columns above ``_DEEP_IV_SENTINEL_FLOOR`` (10,000) while KEEPING the
row, and preserves real distressed-name extremes (500-1196%). Gated on
``SWE_DEEP_TEST_DATA`` — skips in CI (deep panels not committed).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector

_DEEP = os.environ.get("SWE_DEEP_TEST_DATA")

deep_data = pytest.mark.skipif(
    not (_DEEP and (Path(_DEEP) / "deep" / "sp500_vol_iv_full__1994_2012.csv.gz").exists()),
    reason="set SWE_DEEP_TEST_DATA to a dir with deep/ vol_iv slices to run the sentinel test",
)

_IV_COLS = ("hist_put_imp_vol", "hist_call_imp_vol")


@deep_data
def test_sentinel_nulled_real_extremes_kept():
    conn = MarketDataConnector(_DEEP, deep_history=True)
    df = conn._load("vol_iv")
    assert not df.empty
    in_band = 0
    for c in _IV_COLS:
        col = pd.to_numeric(df[c], errors="coerce")
        assert (col > MarketDataConnector._DEEP_IV_SENTINEL_FLOOR).sum() == 0, (
            f"{c} still carries sentinel-magnitude values"
        )
        # Real distressed-name implied vols in (500, 1000] must survive.
        in_band += int(col.between(500, 1000, inclusive="neither").sum())
    assert in_band > 0, "over-nulled — real high implied vols (500-1000%) were removed"


@deep_data
def test_sentinel_row_kept_with_nan_iv():
    """A sentinel row is KEPT (date+ticker present) with the IV nulled to NaN —
    not dropped — so its realized-vol columns remain usable."""
    from engine.data_connector import normalize_ticker

    raw = pd.read_csv(
        Path(_DEEP) / "deep" / "sp500_vol_iv_full__1994_2012.csv.gz",
        compression="gzip",
    )
    hit = raw[pd.to_numeric(raw["hist_put_imp_vol"], errors="coerce") > 10000].iloc[0]
    tkr = normalize_ticker(str(hit["ticker"]))
    dt = pd.Timestamp(hit["date"])

    iv = MarketDataConnector(_DEEP, deep_history=True).get_iv_history(tkr)
    assert dt in iv.index, "sentinel row was dropped instead of nulled"
    assert pd.isna(iv.loc[dt, "hist_put_imp_vol"]), "sentinel IV not nulled"
