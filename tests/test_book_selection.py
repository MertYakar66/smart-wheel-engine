"""Tests for the S4 follow-up — capital-aware book selection.

Two surfaces:

1. The ``collateral`` / ``roc`` columns emitted by
   :meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev`.
2. :meth:`engine.wheel_runner.WheelRunner.select_book` — its
   skip-and-fill walk, the ``max_pct_per_name`` concentration cap, the
   budget invariant, the empty-book / empty-input cases, input
   validation, and the §2 regression that ``select_book`` performs
   **zero** ``EVEngine.evaluate`` calls (it is a presentation layer
   over the ranker's already-evaluated output).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.wheel_runner import WheelRunner

# ----------------------------------------------------------------------
# Synthetic candidate set — keeps the select_book tests deterministic
# and free of any data-on-disk dependency. collateral / roc / ev_dollars
# are mutually consistent (roc == ev_dollars / collateral), as a real
# rank_candidates_by_ev row is.
# ----------------------------------------------------------------------
_SYNTH = pd.DataFrame(
    {
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "BIG"],
        "collateral": [12_000.0, 20_000.0, 30_000.0, 9_000.0, 8_000.0, 80_000.0],
        "roc": [0.0500, 0.0400, 0.0300, 0.0250, 0.0200, 0.0150],
        "ev_dollars": [600.0, 800.0, 900.0, 225.0, 160.0, 1_200.0],
        "ev_per_day": [18.0, 35.0, 40.0, 12.0, 9.0, 60.0],
    }
)


def _ranked() -> pd.DataFrame:
    """Return the synthetic candidates ev_per_day-sorted — like real
    rank_candidates_by_ev output — so select_book's internal re-sort is
    genuinely exercised. A fresh copy each call.
    """
    return _SYNTH.sort_values("ev_per_day", ascending=False).reset_index(drop=True)


@pytest.fixture
def runner() -> WheelRunner:
    """A WheelRunner — construction is cheap (the connector is lazy)."""
    return WheelRunner()


# ======================================================================
# 1. The collateral / roc columns on real ranker output
# ======================================================================
class TestCollateralRocColumns:
    def test_columns_present_and_correct_on_real_run(self, runner, monkeypatch):
        """rank_candidates_by_ev emits collateral and roc, and both hold
        their defining formulas on a real Bloomberg-data run."""
        monkeypatch.setenv("SWE_DATA_PROVIDER", "bloomberg")
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "CF", "KO", "PG"],
            as_of="2026-03-20",
            top_n=20,
            min_ev_dollars=-1e9,
            # Multipliers off — collateral/roc are pure functions of
            # strike/ev_dollars/contracts; this keeps the test fast and
            # free of news / chain / FRED file dependencies.
            use_dealer_positioning=False,
            use_news_sentiment=False,
            use_credit_regime=False,
            use_skew_dynamics=False,
        )
        assert not df.empty, "real ranker run returned no rows — data window moved?"
        assert "collateral" in df.columns
        assert "roc" in df.columns
        assert df["collateral"].notna().all()
        assert df["roc"].notna().all()
        assert (df["collateral"] > 0).all()
        # contracts defaults to 1 — collateral == strike x 100.
        assert (df["collateral"] == (df["strike"] * 100.0).round(2)).all()
        # roc == ev_dollars / collateral.
        assert (df["roc"] == (df["ev_dollars"] / df["collateral"]).round(6)).all()


# ======================================================================
# 2. select_book — the skip-and-fill book builder
# ======================================================================
class TestSelectBook:
    def test_skip_and_fill_roc_order(self, runner):
        """Default roc order, $50k. roc desc: AAA BBB CCC DDD EEE BIG.
        AAA(12k)+BBB(20k) leave 18k; CCC(30k) does NOT fit and is
        skipped; the walk CONTINUES and picks up DDD(9k)+EEE(8k)."""
        book = runner.select_book(_ranked(), account_size=50_000.0)
        assert list(book["ticker"]) == ["AAA", "BBB", "DDD", "EEE"]
        assert book.attrs["n_positions"] == 4
        assert book.attrs["capital_used"] == pytest.approx(49_000.0)
        assert book.attrs["cash_idle"] == pytest.approx(1_000.0)
        assert book.attrs["book_ev"] == pytest.approx(600 + 800 + 225 + 160)
        assert book.attrs["order_by"] == "roc"

    def test_skip_is_not_a_stop(self, runner):
        """The defining skip-and-fill property: an unaffordable
        mid-walk name (CCC) is skipped, not a hard stop — lower-ranked
        smaller names after it are still selected. A strict-stop walk
        would have yielded only [AAA, BBB]."""
        book = runner.select_book(_ranked(), account_size=50_000.0)
        assert "CCC" not in set(book["ticker"])
        assert {"DDD", "EEE"}.issubset(set(book["ticker"]))

    def test_order_by_ev_per_day_differs(self, runner):
        """order_by reproduces a different walk. ev_per_day desc: BIG
        CCC BBB AAA DDD EEE. BIG(80k) skipped, CCC(30k) then BBB(20k)
        exactly exhaust the $50k."""
        book = runner.select_book(_ranked(), 50_000.0, order_by="ev_per_day")
        assert list(book["ticker"]) == ["CCC", "BBB"]
        assert book.attrs["capital_used"] == pytest.approx(50_000.0)
        assert book.attrs["cash_idle"] == pytest.approx(0.0)

    def test_order_by_ev_dollars(self, runner):
        """order_by='ev_dollars' is accepted and walks absolute EV."""
        book = runner.select_book(_ranked(), 50_000.0, order_by="ev_dollars")
        # ev_dollars desc: BIG1200 CCC900 BBB800 AAA600 DDD225 EEE160.
        # BIG skip; CCC(30k)->rem20k; BBB(20k)->rem0; rest skip.
        assert list(book["ticker"]) == ["CCC", "BBB"]

    def test_budget_never_exceeded(self, runner):
        """For every account size and order_by, selected collateral
        never exceeds the budget and idle cash is non-negative."""
        for acct in (7_000.0, 25_000.0, 50_000.0, 200_000.0):
            for ob in ("roc", "ev_per_day", "ev_dollars"):
                book = runner.select_book(_ranked(), acct, order_by=ob)
                assert book["collateral"].sum() <= acct + 1e-6
                assert book.attrs["capital_used"] <= acct + 1e-6
                assert book.attrs["cash_idle"] >= -1e-6

    def test_max_pct_per_name_cap(self, runner):
        """max_pct_per_name skips any name over the cap regardless of
        remaining budget. cap = $50k x 0.25 = $12.5k → BBB/CCC/BIG are
        excluded even though BBB would otherwise fit."""
        book = runner.select_book(_ranked(), 50_000.0, order_by="roc", max_pct_per_name=0.25)
        assert list(book["ticker"]) == ["AAA", "DDD", "EEE"]
        assert (book["collateral"] <= 12_500.0 + 1e-6).all()

    def test_empty_book_when_nothing_fits(self, runner):
        """An account smaller than the cheapest collateral yields an
        empty book — still a DataFrame, still carrying .attrs."""
        book = runner.select_book(_ranked(), account_size=7_000.0)
        assert isinstance(book, pd.DataFrame)
        assert len(book) == 0
        assert book.attrs["n_positions"] == 0
        assert book.attrs["capital_used"] == 0.0
        assert book.attrs["cash_idle"] == pytest.approx(7_000.0)
        assert book.attrs["book_ev"] == 0.0

    def test_empty_ranked_input(self, runner):
        """An empty ranked frame is handled gracefully, not an error."""
        book = runner.select_book(pd.DataFrame(), account_size=50_000.0)
        assert isinstance(book, pd.DataFrame)
        assert len(book) == 0
        assert book.attrs["n_positions"] == 0
        assert book.attrs["cash_idle"] == pytest.approx(50_000.0)

    def test_summary_attrs_complete(self, runner):
        """Every documented summary key is present on .attrs."""
        book = runner.select_book(_ranked(), 50_000.0, max_pct_per_name=0.5)
        for key in (
            "account_size",
            "order_by",
            "max_pct_per_name",
            "n_positions",
            "capital_used",
            "cash_idle",
            "book_ev",
        ):
            assert key in book.attrs, f"missing .attrs[{key!r}]"
        assert book.attrs["account_size"] == 50_000.0
        assert book.attrs["max_pct_per_name"] == 0.5

    def test_book_preserves_all_columns(self, runner):
        """The book is a row subset — every input column is preserved."""
        ranked = _ranked()
        book = runner.select_book(ranked, 50_000.0)
        assert set(book.columns) == set(ranked.columns)

    def test_input_frame_not_mutated(self, runner):
        """select_book must not mutate the caller's ranked frame."""
        ranked = _ranked()
        before = ranked.copy(deep=True)
        runner.select_book(ranked, 50_000.0, max_pct_per_name=0.25)
        pd.testing.assert_frame_equal(ranked, before)


# ======================================================================
# 3. select_book — input validation
# ======================================================================
class TestSelectBookValidation:
    def test_rejects_non_dataframe(self, runner):
        with pytest.raises(ValueError, match="DataFrame"):
            runner.select_book([1, 2, 3], 50_000.0)

    def test_rejects_non_positive_account_size(self, runner):
        with pytest.raises(ValueError, match="account_size"):
            runner.select_book(_ranked(), 0.0)
        with pytest.raises(ValueError, match="account_size"):
            runner.select_book(_ranked(), -5_000.0)

    def test_rejects_missing_required_column(self, runner):
        bad = pd.DataFrame({"ticker": ["X"], "collateral": [1_000.0]})  # no ev_dollars
        with pytest.raises(ValueError, match="missing required column"):
            runner.select_book(bad, 50_000.0)

    def test_rejects_bad_order_by(self, runner):
        with pytest.raises(ValueError, match="order_by"):
            runner.select_book(_ranked(), 50_000.0, order_by="not_a_column")

    def test_rejects_bad_max_pct_per_name(self, runner):
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="max_pct_per_name"):
                runner.select_book(_ranked(), 50_000.0, max_pct_per_name=bad)


# ======================================================================
# 4. §2 invariant — select_book never re-enters the EV authority
# ======================================================================
class TestSection2Invariant:
    def test_select_book_makes_zero_ev_engine_calls(self, runner):
        """§2 regression: select_book is a presentation layer over the
        ranker's already-evaluated output. It must never call
        EVEngine.evaluate — a nonzero count means a refactor introduced
        a side-channel EV path. Mirrors the call-count spy PR #104 used
        for suggest_rolls, but here the expected count is exactly 0."""
        ranked = _ranked()
        original_evaluate = EVEngine.evaluate

        def _pass_through(self, *args, **kwargs):
            return original_evaluate(self, *args, **kwargs)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            book = runner.select_book(ranked, account_size=50_000.0)
            runner.select_book(ranked, 50_000.0, order_by="ev_per_day")
            runner.select_book(ranked, 50_000.0, max_pct_per_name=0.25)
            runner.select_book(pd.DataFrame(), 50_000.0)

        assert mock_eval.call_count == 0, (
            f"select_book triggered {mock_eval.call_count} EVEngine.evaluate "
            "call(s); it must operate purely on the passed DataFrame (§2)."
        )
        # Sanity: the patched runs still produced a real book.
        assert len(book) > 0
