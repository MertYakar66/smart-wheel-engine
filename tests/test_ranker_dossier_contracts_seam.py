"""Audit 2026-07-15 #10: the ranker->dossier contracts seam.

The put ranker now emits ``contracts`` on its row, and the dossier soft-warns
(R7-R10) read it instead of hardcoding 1 contract.
"""

from __future__ import annotations

from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer


def _dict_for(ev_row):
    return EnginePhaseReviewer._build_candidate_dict(
        CandidateDossier(ticker="TEST", ev_row=ev_row)
    )


class TestDossierReadsContracts:
    def test_reads_contracts_from_row(self):
        d = _dict_for({"strike": 100.0, "dte": 30, "iv": 0.3, "contracts": 3})
        assert d["contracts"] == 3

    def test_absent_contracts_defaults_to_one(self):
        d = _dict_for({"strike": 100.0, "dte": 30, "iv": 0.3})
        assert d["contracts"] == 1

    def test_explicit_zero_contracts_stays_zero(self):
        # No ``or 1`` coercion — an explicit 0 must survive (S42 Finding #3).
        d = _dict_for({"strike": 100.0, "dte": 30, "iv": 0.3, "contracts": 0})
        assert d["contracts"] == 0


class TestRankerEmitsContracts:
    def test_put_ranker_row_carries_contracts(self):
        from engine.wheel_runner import WheelRunner

        df = WheelRunner().rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
            contracts=3,
            top_n=10,
            min_ev_dollars=-1e9,
            as_of="2026-05-01",
            include_diagnostic_fields=True,
        )
        assert len(df) > 0, "expected rows at as_of=2026-05-01"
        assert "contracts" in df.columns
        assert (df["contracts"] == 3).all()
