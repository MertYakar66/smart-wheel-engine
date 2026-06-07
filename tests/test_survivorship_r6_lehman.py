"""R6 — the survivorship proof: a delisted name's loss flows into realized P&L.

docs/DATA_LAYER_DEEP_READ_DESIGN.md Part B / §B3. A 2008 survivorship backtest
(deep_history=True) where Lehman (PIT key ``LEHMQ``) crashes toward zero and
delists. The point of the proof is that the loss is REALIZED at the delisting
price — not silently dropped, which is what a plain on/after spot lookup does
once the name stops trading.

Gated on ``SWE_DEEP_TEST_DATA`` (needs the deep + delisted panels + membership,
which are not committed) — skips in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_DEEP = os.environ.get("SWE_DEEP_TEST_DATA")

deep_data = pytest.mark.skipif(
    not (
        _DEEP
        and (Path(_DEEP) / "deep" / "sp500_ohlcv__delisted.csv.gz").exists()
        and (Path(_DEEP) / "sp500_index_membership.csv").exists()
    ),
    reason="set SWE_DEEP_TEST_DATA to a dir with deep/ slices + membership to run the R6 proof",
)


@deep_data
def test_survivorship_backtest_realizes_lehman_loss():
    """Curated 2008 backtest: Lehman flows through the §2 ranker and its realized
    P&L is computed (non-NaN) and reflects the crash — no silent drop."""
    from backtests.survivorship import run_survivorship_backtest

    out = run_survivorship_backtest(
        capital=1_000_000,
        start="2008-08-01",
        end="2008-12-31",
        data_dir=_DEEP,
        tickers=["LEHMQ", "WAMUQ", "AAPL", "XOM", "JPM"],
        top_n=5,
        dte_target=35,
    )
    rl = out["rank_log"]
    assert not rl.empty, "no ranked rows produced"

    # (1) The delisted name flowed through the EV ranker (survivorship plumbing).
    assert "LEHMQ" in set(rl["ticker"]), "Lehman never appeared in the ranked frame"
    leh = rl[rl["ticker"] == "LEHMQ"]

    # (2) No silent drop: every Lehman ranked row has a realized P&L computed.
    assert leh["realized_pnl"].notna().all(), "Lehman rows have NaN realized_pnl (silent drop)"

    # (3) At least one Lehman put expires AFTER the delisting and realizes a real
    #     loss at the delisting price (not ~0, not dropped).
    delisted_rows = leh[leh["delisted_at_expiry"]]
    assert len(delisted_rows) > 0, "no Lehman row expired post-delisting"
    assert delisted_rows["realized_pnl"].min() < -500.0, (
        "Lehman's post-delisting put should realize a real loss, not ~0"
    )
