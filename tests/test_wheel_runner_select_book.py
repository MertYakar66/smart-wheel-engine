"""Tests for the S4 follow-up: capital-efficiency columns on the ranker
output and :meth:`engine.wheel_runner.WheelRunner.select_book`.

Covers:
  * ``collateral`` / ``roc`` are core (always-emitted) ranker columns.
  * ``_solve_book_knapsack`` — the 0/1 knapsack helper — is correct.
  * ``select_book`` fits a book under an account-size budget, beats a
    naive top-down fill, respects a concentration cap, and handles the
    edge cases (zero budget, empty ranking, missing columns).
  * §2 invariant: ``select_book`` is a pure post-processor — it never
    calls ``EVEngine.evaluate`` itself (call-count regression test) and
    cannot place a negative-EV candidate into a book.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.wheel_runner import WheelRunner, _solve_book_knapsack

# Overlays that need a live chain / network are off everywhere here so
# the tests are deterministic and offline.
_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}


# ----------------------------------------------------------------------
# Fake connector — deterministic, no data-on-disk dependency. Each
# ticker gets a distinct spot level so collateral varies across names.
# ----------------------------------------------------------------------
class _FakeConn:
    def __init__(self, tickers: list[str], n: int = 1200) -> None:
        self._idx = pd.date_range("2019-01-01", periods=n, freq="B")
        self._prices: dict[str, np.ndarray] = {}
        for i, t in enumerate(tickers):
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            self._prices[t] = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
        self._tickers = list(tickers)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame({"close": self._prices[ticker]}, index=self._idx)

    def get_fundamentals(self, ticker: str) -> dict:
        return {
            "implied_vol_atm": 28.0,  # IV in PERCENT (Bloomberg contract; #356)
            "volatility_30d": 25.0,
            "dividend_yield": 0.01,
        }

    def get_risk_free_rate(self, as_of=None) -> float:
        return 0.05

    def get_next_earnings(self, ticker: str, as_of=None):
        return None

    def get_universe(self) -> list[str]:
        return list(self._tickers)


def _runner(tickers: list[str]) -> WheelRunner:
    r = WheelRunner()
    r._connector = _FakeConn(tickers)
    return r


def _mk_ranking(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic ranker-shaped frame for select_book unit tests.

    Each row needs at least ticker / collateral / ev_dollars; ev_per_day
    and roc are derived when absent.
    """
    out = []
    for r in rows:
        coll = float(r["collateral"])
        ev = float(r["ev_dollars"])
        out.append(
            {
                "ticker": r["ticker"],
                "collateral": coll,
                "ev_dollars": ev,
                "ev_per_day": r.get("ev_per_day", ev / 18.0),
                "roc": r.get("roc", ev / coll if coll > 0 else 0.0),
            }
        )
    return pd.DataFrame(out)


# ======================================================================
# 1. Ranker capital-efficiency columns
# ======================================================================
class TestRankerCapitalColumns:
    TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    def _rank(self, **extra) -> pd.DataFrame:
        return _runner(self.TICKERS).rank_candidates_by_ev(
            tickers=self.TICKERS,
            top_n=10,
            min_ev_dollars=-1e9,
            **_OFFLINE,
            **extra,
        )

    def test_collateral_and_roc_present(self):
        df = self._rank()
        assert not df.empty
        assert "collateral" in df.columns
        assert "roc" in df.columns

    def test_columns_present_even_without_diagnostics(self):
        """collateral / roc are core fields, not diagnostic — they
        must survive include_diagnostic_fields=False."""
        df = self._rank(include_diagnostic_fields=False)
        assert not df.empty
        assert "collateral" in df.columns
        assert "roc" in df.columns
        # The diagnostic block should genuinely be gone.
        assert "cvar_5" not in df.columns

    def test_collateral_equals_strike_times_100(self):
        df = self._rank()
        for _, row in df.iterrows():
            assert row["collateral"] == pytest.approx(row["strike"] * 100.0, rel=1e-9)

    def test_collateral_scales_with_contracts(self):
        df = self._rank(contracts=3)
        for _, row in df.iterrows():
            assert row["collateral"] == pytest.approx(row["strike"] * 100.0 * 3, rel=1e-9)

    def test_roc_equals_ev_over_collateral(self):
        df = self._rank()
        for _, row in df.iterrows():
            expected = row["ev_dollars"] / row["collateral"]
            assert row["roc"] == pytest.approx(expected, abs=1e-4)


# ======================================================================
# 2. _solve_book_knapsack — the 0/1 knapsack helper
# ======================================================================
class TestSolveBookKnapsack:
    def test_empty_inputs(self):
        assert _solve_book_knapsack([], [], 100) == []

    def test_zero_capacity(self):
        assert _solve_book_knapsack([3], [10.0], 0) == []

    def test_nothing_fits(self):
        assert _solve_book_knapsack([100], [5.0], 50) == []

    def test_picks_optimal_combination(self):
        # capacity 8: {0,2} weights 3+5=8 value 115 beats {0,1}=110.
        sel = _solve_book_knapsack([3, 4, 5], [60.0, 50.0, 55.0], 8)
        assert sorted(sel) == [0, 2]

    def test_single_high_value_item_when_it_alone_wins(self):
        # One fat item worth more than every affordable combination.
        sel = _solve_book_knapsack([10, 4, 4], [100.0, 5.0, 5.0], 10)
        assert sel == [0]

    def test_exact_fit_takes_everything(self):
        sel = _solve_book_knapsack([2, 3, 5], [1.0, 1.0, 1.0], 10)
        assert sorted(sel) == [0, 1, 2]

    def test_returned_weights_never_exceed_capacity(self):
        weights = [7, 11, 13, 5, 9]
        values = [12.0, 20.0, 25.0, 8.0, 15.0]
        cap = 24
        sel = _solve_book_knapsack(weights, values, cap)
        assert sum(weights[i] for i in sel) <= cap

    def test_matches_brute_force_on_a_small_instance(self):
        from itertools import combinations

        weights = [4, 6, 8, 3, 7, 5]
        values = [10.0, 13.0, 18.0, 6.0, 14.0, 11.0]
        cap = 20
        sel = _solve_book_knapsack(weights, values, cap)
        dp_value = sum(values[i] for i in sel)
        best = 0.0
        for k in range(len(weights) + 1):
            for combo in combinations(range(len(weights)), k):
                if sum(weights[i] for i in combo) <= cap:
                    best = max(best, sum(values[i] for i in combo))
        assert dp_value == pytest.approx(best)


# ======================================================================
# 3. select_book — budget fitting on synthetic rankings
# ======================================================================
class TestSelectBook:
    def test_book_never_exceeds_budget(self):
        ranking = _mk_ranking(
            [
                {"ticker": "A", "collateral": 17750.0, "ev_dollars": 190.0},
                {"ticker": "B", "collateral": 13800.0, "ev_dollars": 120.0},
                {"ticker": "C", "collateral": 8950.0, "ev_dollars": 40.0},
                {"ticker": "D", "collateral": 48000.0, "ev_dollars": 300.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=30000.0, ranking=ranking)
        assert book.attrs["total_collateral"] <= 30000.0
        assert book.attrs["cash_remaining"] >= 0.0

    def test_book_maximises_ev_and_beats_top_down_fill(self):
        # S4's headline finding: a top-down ev_per_day walk strands
        # capital; the knapsack does better. A is ranked first by
        # ev_per_day but eats the budget; B+C is the optimal book.
        ranking = _mk_ranking(
            [
                {"ticker": "A", "collateral": 6000.0, "ev_dollars": 60.0, "ev_per_day": 9.0},
                {"ticker": "B", "collateral": 5000.0, "ev_dollars": 55.0, "ev_per_day": 3.0},
                {"ticker": "C", "collateral": 5000.0, "ev_dollars": 55.0, "ev_per_day": 2.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=10000.0, ranking=ranking)

        # Naive top-down fill in ev_per_day order: takes A (6000), then
        # B does not fit (6000+5000 > 10000) -> 1 name, 60 EV.
        topdown_ev = 60.0
        assert book.attrs["total_ev_dollars"] > topdown_ev
        assert set(book["ticker"]) == {"B", "C"}
        assert book.attrs["n_positions"] == 2

    def test_concentration_cap_drops_oversized_names(self):
        ranking = _mk_ranking(
            [
                {"ticker": "BIG", "collateral": 40000.0, "ev_dollars": 500.0},
                {"ticker": "MID", "collateral": 12000.0, "ev_dollars": 90.0},
                {"ticker": "SML", "collateral": 9000.0, "ev_dollars": 70.0},
            ]
        )
        # 25% of 50k = 12.5k cap -> BIG (40k) is ineligible.
        book = _runner(["A"]).select_book(
            account_size=50000.0,
            ranking=ranking,
            max_weight_per_name=0.25,
        )
        assert "BIG" not in set(book["ticker"])
        for c in book["collateral"]:
            assert c <= 0.25 * 50000.0

    def test_min_roc_floor_filters_low_roc_names(self):
        ranking = _mk_ranking(
            [
                {"ticker": "HI", "collateral": 10000.0, "ev_dollars": 200.0},
                {"ticker": "LO", "collateral": 10000.0, "ev_dollars": 5.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=100000.0, ranking=ranking, min_roc=0.001)
        assert "LO" not in set(book["ticker"])
        assert "HI" in set(book["ticker"])

    def test_zero_or_negative_account_returns_empty(self):
        ranking = _mk_ranking([{"ticker": "A", "collateral": 1000.0, "ev_dollars": 50.0}])
        book = _runner(["A"]).select_book(account_size=0.0, ranking=ranking)
        assert book.empty
        assert book.attrs["n_positions"] == 0

    def test_empty_ranking_returns_empty(self):
        book = _runner(["A"]).select_book(account_size=50000.0, ranking=pd.DataFrame())
        assert book.empty
        assert book.attrs["n_positions"] == 0

    def test_missing_collateral_column_raises(self):
        bad = pd.DataFrame({"ticker": ["A"], "ev_dollars": [50.0]})
        with pytest.raises(ValueError, match="collateral"):
            _runner(["A"]).select_book(account_size=50000.0, ranking=bad)

    @pytest.mark.parametrize("bad_unit", [0.0, -50.0, -1.0])
    def test_nonpositive_collateral_unit_raises(self, bad_unit):
        # collateral_unit is a divisor — zero crashes, negatives produce
        # nonsense capacity/weights. Fail fast with a clear ValueError.
        ranking = _mk_ranking([{"ticker": "A", "collateral": 10000.0, "ev_dollars": 100.0}])
        with pytest.raises(ValueError, match="collateral_unit"):
            _runner(["A"]).select_book(
                account_size=50000.0, ranking=ranking, collateral_unit=bad_unit
            )

    def test_attrs_totals_are_internally_consistent(self):
        ranking = _mk_ranking(
            [
                {"ticker": "A", "collateral": 10000.0, "ev_dollars": 100.0},
                {"ticker": "B", "collateral": 15000.0, "ev_dollars": 130.0},
                {"ticker": "C", "collateral": 9000.0, "ev_dollars": 60.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=40000.0, ranking=ranking)
        assert book.attrs["total_collateral"] == pytest.approx(book["collateral"].sum())
        assert book.attrs["total_ev_dollars"] == pytest.approx(book["ev_dollars"].sum())
        assert book.attrs["cash_remaining"] == pytest.approx(40000.0 - book["collateral"].sum())
        assert book.attrs["n_positions"] == len(book)
        assert book.attrs["selection_method"] == "exact_knapsack"

    def test_greedy_fallback_for_huge_account(self, monkeypatch):
        # Force the greedy degradation path with a tiny cell ceiling.
        monkeypatch.setattr("engine.wheel_runner._KNAPSACK_MAX_CELLS", 1)
        ranking = _mk_ranking(
            [
                {"ticker": "A", "collateral": 10000.0, "ev_dollars": 300.0},
                {"ticker": "B", "collateral": 10000.0, "ev_dollars": 50.0},
                {"ticker": "C", "collateral": 10000.0, "ev_dollars": 20.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=20000.0, ranking=ranking)
        assert book.attrs["selection_method"] == "greedy_roc"
        assert book.attrs["total_collateral"] <= 20000.0
        # Greedy ROC order keeps the two highest-ROC names.
        assert set(book["ticker"]) == {"A", "B"}


# ======================================================================
# 4. §2 invariant — select_book is a pure post-processor
# ======================================================================
class TestSelectBookSection2:
    TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    def test_negative_ev_candidate_never_enters_a_book(self):
        """§2: select_book cannot rescue a negative-EV trade. Even with
        a large budget and a generous ROC floor, a negative-EV row is
        filtered out of the pool before selection."""
        ranking = _mk_ranking(
            [
                {"ticker": "GOOD", "collateral": 10000.0, "ev_dollars": 120.0},
                {"ticker": "BAD", "collateral": 10000.0, "ev_dollars": -250.0},
            ]
        )
        book = _runner(["A"]).select_book(account_size=1_000_000.0, ranking=ranking)
        assert "BAD" not in set(book["ticker"])
        assert "GOOD" in set(book["ticker"])

    def test_select_book_with_precomputed_ranking_makes_zero_ev_calls(self):
        """When a ranking frame is supplied, select_book must not touch
        EVEngine at all — it is pure subsetting."""
        ranking = _mk_ranking(
            [
                {"ticker": "A", "collateral": 10000.0, "ev_dollars": 100.0},
                {"ticker": "B", "collateral": 12000.0, "ev_dollars": 80.0},
            ]
        )
        original = EVEngine.evaluate

        def _pass_through(self, *a, **k):
            return original(self, *a, **k)

        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_eval:
            _runner(["A"]).select_book(account_size=50000.0, ranking=ranking)
        assert mock_eval.call_count == 0, (
            f"select_book(ranking=...) called EVEngine.evaluate "
            f"{mock_eval.call_count} times; a pure post-processor must "
            f"call it zero times."
        )

    def test_select_book_adds_no_ev_calls_beyond_the_ranker(self):
        """§2 call-count regression: select_book(ranking=None) runs the
        ranker once and then a knapsack. The knapsack must contribute
        ZERO extra EVEngine.evaluate calls — select_book's evaluate
        count must equal rank_candidates_by_ev's on the same inputs.
        A mismatch means a side-channel EV path was introduced.
        """
        original = EVEngine.evaluate

        def _pass_through(self, *a, **k):
            return original(self, *a, **k)

        rank_kwargs = dict(
            tickers=self.TICKERS,
            top_n=10,
            min_ev_dollars=-1e9,
            **_OFFLINE,
        )

        # Baseline: the ranker alone.
        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_rank:
            _runner(self.TICKERS).rank_candidates_by_ev(**rank_kwargs)
        ranker_calls = mock_rank.call_count
        assert ranker_calls >= 1

        # select_book driving the same ranker internally.
        with patch.object(
            EVEngine, "evaluate", autospec=True, side_effect=_pass_through
        ) as mock_book:
            _runner(self.TICKERS).select_book(account_size=80000.0, **rank_kwargs)
        assert mock_book.call_count == ranker_calls, (
            f"select_book triggered {mock_book.call_count} EVEngine.evaluate "
            f"calls; the ranker alone triggers {ranker_calls}. The knapsack "
            f"must add zero — a mismatch means select_book bypasses or "
            f"re-invokes the EV authority."
        )

    def test_end_to_end_book_under_budget_from_live_ranker(self):
        """select_book(ranking=None) end-to-end against the fake
        connector: the book fits the budget and every name in it is
        positive-EV."""
        book = _runner(self.TICKERS).select_book(
            account_size=60000.0,
            tickers=self.TICKERS,
            top_n=10,
            min_ev_dollars=-1e9,
            **_OFFLINE,
        )
        if not book.empty:
            assert book.attrs["total_collateral"] <= 60000.0
            assert (book["ev_dollars"] > 0).all()

    def test_self_ranking_uses_whole_pool_not_ranker_display_slice(self):
        """When select_book builds the ranking itself it must consider
        the whole candidate set — rank_candidates_by_ev's top_n default
        of 10 would silently truncate the pool and can yield a provably
        suboptimal book. An explicit caller top_n still wins."""
        captured: list[dict] = []
        original = WheelRunner.rank_candidates_by_ev

        def _spy(self, *a, **k):
            captured.append(dict(k))
            return original(self, *a, **k)

        with patch.object(WheelRunner, "rank_candidates_by_ev", autospec=True, side_effect=_spy):
            _runner(self.TICKERS).select_book(
                account_size=80000.0,
                tickers=self.TICKERS,
                min_ev_dollars=-1e9,
                **_OFFLINE,
            )
        assert captured, "rank_candidates_by_ev was not called"
        assert captured[0].get("top_n", 10) > 10, (
            "select_book inherited the ranker's top_n=10 default — the candidate pool is truncated."
        )

        # An explicit top_n passed by the caller must be respected.
        captured.clear()
        with patch.object(WheelRunner, "rank_candidates_by_ev", autospec=True, side_effect=_spy):
            _runner(self.TICKERS).select_book(
                account_size=80000.0,
                tickers=self.TICKERS,
                top_n=3,
                min_ev_dollars=-1e9,
                **_OFFLINE,
            )
        assert captured[0].get("top_n") == 3
