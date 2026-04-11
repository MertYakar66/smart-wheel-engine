"""Tests for engine/contracts.py interface validation helpers."""

from engine.contracts import (
    REQUIRED_GREEK_KEYS,
    REQUIRED_LADDER_COLUMNS,
    PricerContract,
    RiskContract,
    StressTesterContract,
    validate_greeks_output,
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
