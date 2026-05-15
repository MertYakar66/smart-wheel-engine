"""Tests for engine/policy_config.py — load/save/validate trading policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.policy_config import (
    AdvisorPolicyConfig,
    GreeksPolicyConfig,
    RiskPolicyConfig,
    SignalPolicyConfig,
    TradingPolicyConfig,
    load_policy,
    save_policy,
    validate_policy,
)


class TestDefaults:
    def test_default_policy_is_valid(self):
        cfg = TradingPolicyConfig()
        assert validate_policy(cfg) == []

    def test_risk_defaults_in_range(self):
        r = RiskPolicyConfig()
        assert 0.90 <= r.var_confidence <= 0.999
        assert 0.0 < r.max_drawdown_pct < 1.0
        assert 0.0 < r.max_daily_loss_pct < r.max_drawdown_pct

    def test_signal_dte_window_ordered(self):
        s = SignalPolicyConfig()
        assert s.target_dte_min < s.target_dte_ideal < s.target_dte_max

    def test_advisor_weights_present(self):
        a = AdvisorPolicyConfig()
        assert a.committee_weights
        assert all(v > 0 for v in a.committee_weights.values())

    def test_greeks_defaults(self):
        g = GreeksPolicyConfig()
        assert g.theta_annual is True
        assert g.vega_per_vol_point is True
        assert 0.0 < g.decomposition_residual_tolerance <= 1.0


class TestValidate:
    def _good(self) -> TradingPolicyConfig:
        return TradingPolicyConfig()

    def test_var_confidence_too_low(self):
        cfg = self._good()
        cfg.risk.var_confidence = 0.50
        errs = validate_policy(cfg)
        assert any("var_confidence" in e for e in errs)

    def test_var_confidence_too_high(self):
        cfg = self._good()
        cfg.risk.var_confidence = 1.5
        errs = validate_policy(cfg)
        assert any("var_confidence" in e for e in errs)

    def test_max_drawdown_out_of_range(self):
        cfg = self._good()
        cfg.risk.max_drawdown_pct = 0.0
        assert any("max_drawdown_pct" in e for e in validate_policy(cfg))
        cfg.risk.max_drawdown_pct = 1.5
        assert any("max_drawdown_pct" in e for e in validate_policy(cfg))

    def test_daily_loss_must_be_less_than_drawdown(self):
        cfg = self._good()
        cfg.risk.max_daily_loss_pct = 0.99
        cfg.risk.max_drawdown_pct = 0.20
        errs = validate_policy(cfg)
        assert any("max_daily_loss_pct" in e for e in errs)

    def test_concentrated_book_threshold_must_be_positive(self):
        cfg = self._good()
        cfg.risk.concentrated_book_threshold = 0
        errs = validate_policy(cfg)
        assert any("concentrated_book_threshold" in e for e in errs)

    def test_max_var_pct_out_of_range(self):
        cfg = self._good()
        cfg.risk.max_var_pct = 1.5
        errs = validate_policy(cfg)
        assert any("max_var_pct" in e for e in errs)

    def test_iv_rank_ordering(self):
        cfg = self._good()
        cfg.signal.iv_rank_low = 0.6
        cfg.signal.iv_rank_high = 0.5  # inverted
        errs = validate_policy(cfg)
        assert any("iv_rank" in e for e in errs)

    def test_trend_lookback_too_short(self):
        cfg = self._good()
        cfg.signal.trend_lookback_days = 2
        errs = validate_policy(cfg)
        assert any("trend_lookback_days" in e for e in errs)

    def test_profit_target_out_of_range(self):
        cfg = self._good()
        cfg.signal.profit_target_pct = 1.5
        errs = validate_policy(cfg)
        assert any("profit_target_pct" in e for e in errs)

    def test_stop_loss_must_exceed_one(self):
        cfg = self._good()
        cfg.signal.stop_loss_multiplier = 0.5
        errs = validate_policy(cfg)
        assert any("stop_loss_multiplier" in e for e in errs)

    def test_dte_ordering(self):
        cfg = self._good()
        cfg.signal.target_dte_min = 50
        cfg.signal.target_dte_ideal = 35
        cfg.signal.target_dte_max = 45
        errs = validate_policy(cfg)
        assert any("target_dte" in e for e in errs)

    def test_earnings_buffer_negative(self):
        cfg = self._good()
        cfg.signal.earnings_buffer_days = -1
        errs = validate_policy(cfg)
        assert any("earnings_buffer_days" in e for e in errs)

    def test_advisor_empty_weights(self):
        cfg = self._good()
        cfg.advisor.committee_weights = {}
        errs = validate_policy(cfg)
        assert any("committee_weights" in e and "empty" in e for e in errs)

    def test_advisor_negative_weights(self):
        cfg = self._good()
        cfg.advisor.committee_weights = {"regime": -0.5, "vol": 1.0}
        errs = validate_policy(cfg)
        assert any("positive" in e for e in errs)

    def test_calibration_drift_out_of_range(self):
        cfg = self._good()
        cfg.advisor.calibration_drift_threshold = 1.5
        errs = validate_policy(cfg)
        assert any("calibration_drift_threshold" in e for e in errs)

    def test_rebalance_frequency_must_be_positive(self):
        cfg = self._good()
        cfg.advisor.rebalance_frequency_days = 0
        errs = validate_policy(cfg)
        assert any("rebalance_frequency_days" in e for e in errs)

    def test_greeks_residual_tolerance_out_of_range(self):
        cfg = self._good()
        cfg.greeks.decomposition_residual_tolerance = 1.5
        errs = validate_policy(cfg)
        assert any("decomposition_residual_tolerance" in e for e in errs)

    def test_collects_multiple_errors(self):
        cfg = self._good()
        cfg.risk.var_confidence = 0.5
        cfg.signal.profit_target_pct = 1.5
        errs = validate_policy(cfg)
        assert len(errs) >= 2


class TestLoadSave:
    def test_load_with_no_path_returns_defaults(self):
        cfg = load_policy()
        assert isinstance(cfg, TradingPolicyConfig)
        assert validate_policy(cfg) == []

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_policy(str(tmp_path / "does_not_exist.json"))

    def test_load_invalid_json_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            load_policy(str(bad))

    def test_load_partial_overrides_take_effect(self, tmp_path: Path):
        path = tmp_path / "policy.json"
        path.write_text(
            json.dumps(
                {
                    "risk": {"max_drawdown_pct": 0.30, "max_daily_loss_pct": 0.05},
                    "signal": {"profit_target_pct": 0.40},
                }
            )
        )
        cfg = load_policy(str(path))
        assert cfg.risk.max_drawdown_pct == 0.30
        assert cfg.signal.profit_target_pct == 0.40
        # Untouched keys keep defaults
        assert cfg.greeks.theta_annual is True

    def test_load_invalid_policy_raises_value_error(self, tmp_path: Path):
        path = tmp_path / "bad_policy.json"
        path.write_text(
            json.dumps(
                {
                    "risk": {"var_confidence": 0.5},
                }
            )
        )
        with pytest.raises(ValueError, match="Policy validation failed"):
            load_policy(str(path))

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        cfg = TradingPolicyConfig()
        cfg.signal.profit_target_pct = 0.40
        cfg.risk.max_drawdown_pct = 0.25
        path = tmp_path / "out.json"
        save_policy(cfg, str(path))
        cfg2 = load_policy(str(path))
        assert cfg2.signal.profit_target_pct == 0.40
        assert cfg2.risk.max_drawdown_pct == 0.25

    def test_save_writes_indented_json_with_sorted_keys(self, tmp_path: Path):
        path = tmp_path / "out.json"
        save_policy(TradingPolicyConfig(), str(path))
        text = path.read_text()
        # Indented (multi-line) and sorted
        lines = text.strip().split("\n")
        assert len(lines) > 5
        # First top-level key alphabetically should be 'advisor'
        assert '"advisor"' in lines[1]
