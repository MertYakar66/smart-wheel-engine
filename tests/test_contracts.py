"""Tests for engine/contracts.py interface validation helpers."""

import pandas as pd

from engine.contracts import (
    REQUIRED_GREEK_KEYS,
    REQUIRED_LADDER_COLUMNS,
    PricerContract,
    RiskContract,
    StressTesterContract,
    validate_greeks_output,
    validate_greeks_semantics,
    validate_ladder_output,
)


class TestConstants:
    def test_required_greek_keys(self):
        assert "price" in REQUIRED_GREEK_KEYS
        assert "delta" in REQUIRED_GREEK_KEYS
        assert "gamma" in REQUIRED_GREEK_KEYS
        assert "theta" in REQUIRED_GREEK_KEYS
        assert "vega" in REQUIRED_GREEK_KEYS
        assert "rho" in REQUIRED_GREEK_KEYS
        assert len(REQUIRED_GREEK_KEYS) == 6

    def test_required_ladder_columns(self):
        assert "spot_change" in REQUIRED_LADDER_COLUMNS
        assert "total_pnl" in REQUIRED_LADDER_COLUMNS
        assert "delta_pnl" in REQUIRED_LADDER_COLUMNS


class TestValidateGreeksOutput:
    def test_valid_greeks(self):
        greeks = {
            "price": 5.0,
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.2,
            "rho": 0.01,
        }
        errors = validate_greeks_output(greeks)
        assert errors == []

    def test_missing_keys(self):
        greeks = {"price": 5.0, "delta": 0.5}
        errors = validate_greeks_output(greeks)
        assert len(errors) >= 1
        assert "Missing required keys" in errors[0]

    def test_non_numeric(self):
        greeks = {
            "price": "bad",
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.2,
            "rho": 0.01,
        }
        errors = validate_greeks_output(greeks)
        assert any("numeric" in e for e in errors)

    def test_nan_value(self):
        greeks = {
            "price": float("nan"),
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.2,
            "rho": 0.01,
        }
        errors = validate_greeks_output(greeks)
        assert any("NaN" in e for e in errors)

    def test_not_a_dict(self):
        errors = validate_greeks_output("not a dict")
        assert len(errors) == 1
        assert "Expected dict" in errors[0]

    def test_extra_keys_ok(self):
        greeks = {
            "price": 5.0,
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.2,
            "rho": 0.01,
            "charm": 0.001,
        }
        errors = validate_greeks_output(greeks)
        assert errors == []


class TestProtocols:
    def test_pricer_contract_is_protocol(self):
        assert hasattr(PricerContract, "price")
        assert hasattr(PricerContract, "all_greeks")

    def test_risk_contract_is_protocol(self):
        assert hasattr(RiskContract, "calculate_position_size")
        assert hasattr(RiskContract, "calculate_portfolio_greeks")
        assert hasattr(RiskContract, "calculate_var")

    def test_stress_tester_contract_is_protocol(self):
        assert hasattr(StressTesterContract, "run_scenario")
        assert hasattr(StressTesterContract, "greeks_stress_ladder")


class TestValidateLadderOutput:
    def _good_ladder(self) -> pd.DataFrame:
        return pd.DataFrame({
            "spot_change": [-0.10, 0.0, 0.10],
            "total_pnl": [100.0, 0.0, -100.0],
            "delta_pnl": [80.0, 0.0, -80.0],
            "gamma_pnl": [10.0, 0.0, -10.0],
            "theta_pnl": [5.0, 5.0, 5.0],
            "vega_pnl": [5.0, -5.0, -15.0],
        })

    def test_valid_ladder(self):
        assert validate_ladder_output(self._good_ladder()) == []

    def test_not_a_dataframe(self):
        errors = validate_ladder_output("not a dataframe")
        assert len(errors) == 1
        assert "Expected DataFrame" in errors[0]

    def test_empty_dataframe(self):
        errors = validate_ladder_output(pd.DataFrame())
        assert any("empty" in e for e in errors)

    def test_missing_columns(self):
        df = pd.DataFrame({"spot_change": [0.0], "total_pnl": [0.0]})
        errors = validate_ladder_output(df)
        assert any("Missing required columns" in e for e in errors)

    def test_non_numeric_column(self):
        df = self._good_ladder()
        df["delta_pnl"] = ["a", "b", "c"]
        errors = validate_ladder_output(df)
        assert any("delta_pnl" in e and "numeric" in e for e in errors)


class TestValidateGreeksSemantics:
    def _good(self, **overrides) -> dict:
        base = {"price": 5.0, "delta": 0.5, "gamma": 0.02, "vega": 0.2, "theta": -10.0}
        base.update(overrides)
        return base

    def test_valid_call(self):
        assert validate_greeks_semantics(self._good(delta=0.5), "call") == []

    def test_valid_put(self):
        assert validate_greeks_semantics(self._good(delta=-0.5), "put") == []

    def test_negative_price_flagged(self):
        errors = validate_greeks_semantics(self._good(price=-1.0), "call")
        assert any("price" in e and "negative" in e for e in errors)

    def test_negative_call_delta_flagged(self):
        errors = validate_greeks_semantics(self._good(delta=-0.5), "call")
        assert any("delta" in e for e in errors)

    def test_positive_put_delta_flagged(self):
        errors = validate_greeks_semantics(self._good(delta=0.5), "put")
        assert any("delta" in e for e in errors)

    def test_delta_magnitude_above_one_flagged(self):
        errors = validate_greeks_semantics(self._good(delta=1.5), "call")
        assert any("|delta|" in e for e in errors)

    def test_negative_gamma_flagged(self):
        errors = validate_greeks_semantics(self._good(gamma=-0.01), "call")
        assert any("gamma" in e for e in errors)

    def test_negative_vega_flagged(self):
        errors = validate_greeks_semantics(self._good(vega=-0.5), "call")
        assert any("vega" in e for e in errors)

    def test_huge_theta_with_meaningful_delta_flagged(self):
        # Theta magnitude > 500 suggests daily, not annual, units
        errors = validate_greeks_semantics(
            self._good(theta=-1000.0, delta=0.5), "call",
        )
        assert any("theta" in e and "annual" in e for e in errors)

    def test_huge_theta_with_tiny_delta_not_flagged(self):
        # Sub-1% delta — theta sanity check is bypassed
        errors = validate_greeks_semantics(
            self._good(theta=-1000.0, delta=0.005), "call",
        )
        assert not any("annual" in e for e in errors)

    def test_missing_keys_default_to_zero(self):
        # Empty dict should not crash; values default to 0 → no violations
        assert validate_greeks_semantics({}, "call") == []
