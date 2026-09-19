"""
Unit tests for Long/Short Risk-Reward Ratio Calculation Tool.

Feature: Story 5.4.1: Implement Long/Short Risk-Reward Ratio Calculation Tool
Target Module: src/analytics/risk_reward.py
"""

import pytest
from src.analytics.risk_reward import RiskRewardResult, calculate_risk_reward


def _get_ratio(result: RiskRewardResult) -> float:
    """Helper to retrieve the risk-reward ratio supporting 'ratio' or 'risk_reward_ratio' attributes."""
    if hasattr(result, "ratio"):
        return result.ratio
    if hasattr(result, "risk_reward_ratio"):
        return result.risk_reward_ratio
    raise AttributeError("Result object has neither 'ratio' nor 'risk_reward_ratio' attribute.")


class TestRiskRewardCalculationSuccess:
    """Tests verifying successful calculation for valid Long and Short setups."""

    def test_long_position_risk_reward_ac1(self):
        """
        AC1: Given entry=100, SL=95, TP=115 for LONG position,
        When executed, returns risk=5.0, reward=15.0, and risk-reward ratio=3.0.
        """
        result = calculate_risk_reward(
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            position_type="LONG",
        )

        assert isinstance(result, RiskRewardResult)
        assert result.risk == pytest.approx(5.0)
        assert result.reward == pytest.approx(15.0)
        assert _get_ratio(result) == pytest.approx(3.0)

    def test_short_position_risk_reward_ac2(self):
        """
        AC2: Given entry=100, SL=105, TP=85 for SHORT position,
        When executed, returns risk=5.0, reward=15.0, and risk-reward ratio=3.0.
        """
        result = calculate_risk_reward(
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=85.0,
            position_type="SHORT",
        )

        assert isinstance(result, RiskRewardResult)
        assert result.risk == pytest.approx(5.0)
        assert result.reward == pytest.approx(15.0)
        assert _get_ratio(result) == pytest.approx(3.0)

    @pytest.mark.parametrize("pos_type", ["long", "LONG", "Long", "LoNg"])
    def test_long_position_case_insensitivity(self, pos_type: str):
        """Verify position_type string comparison is case-insensitive for LONG."""
        result = calculate_risk_reward(
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            position_type=pos_type,
        )
        assert result.risk == pytest.approx(5.0)
        assert result.reward == pytest.approx(15.0)
        assert _get_ratio(result) == pytest.approx(3.0)

    @pytest.mark.parametrize("pos_type", ["short", "SHORT", "Short", "sHoRt"])
    def test_short_position_case_insensitivity(self, pos_type: str):
        """Verify position_type string comparison is case-insensitive for SHORT."""
        result = calculate_risk_reward(
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=85.0,
            position_type=pos_type,
        )
        assert result.risk == pytest.approx(5.0)
        assert result.reward == pytest.approx(15.0)
        assert _get_ratio(result) == pytest.approx(3.0)

    def test_fractional_prices_precision(self):
        """Verify calculations handle non-integer floating point prices accurately."""
        result = calculate_risk_reward(
            entry_price=150.25,
            stop_loss=148.25,
            take_profit=156.25,
            position_type="LONG",
        )
        assert result.risk == pytest.approx(2.0)
        assert result.reward == pytest.approx(6.0)
        assert _get_ratio(result) == pytest.approx(3.0)


class TestRiskRewardPositionLogicViolations:
    """Tests verifying ValueError raised when price inputs violate position mechanics."""

    @pytest.mark.parametrize(
        ("entry_price", "stop_loss", "take_profit"),
        [
            (100.0, 100.0, 115.0),  # SL == Entry
            (100.0, 105.0, 115.0),  # SL > Entry
            (100.0, 95.0, 100.0),   # TP == Entry
            (100.0, 95.0, 90.0),    # TP < Entry
            (100.0, 110.0, 90.0),   # Both SL above and TP below entry
        ],
        ids=[
            "long_sl_equal_entry",
            "long_sl_above_entry",
            "long_tp_equal_entry",
            "long_tp_below_entry",
            "long_both_inverted",
        ],
    )
    def test_long_position_logic_violations_raise_value_error(
        self, entry_price: float, stop_loss: float, take_profit: float
    ):
        """AC3: Stop loss >= entry or take profit <= entry on LONG must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_risk_reward(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type="LONG",
            )
        assert len(str(exc_info.value).strip()) > 0

    @pytest.mark.parametrize(
        ("entry_price", "stop_loss", "take_profit"),
        [
            (100.0, 100.0, 85.0),   # SL == Entry
            (100.0, 95.0, 85.0),    # SL < Entry
            (100.0, 105.0, 100.0),  # TP == Entry
            (100.0, 105.0, 110.0),  # TP > Entry
            (100.0, 90.0, 110.0),   # Both SL below and TP above entry
        ],
        ids=[
            "short_sl_equal_entry",
            "short_sl_below_entry",
            "short_tp_equal_entry",
            "short_tp_above_entry",
            "short_both_inverted",
        ],
    )
    def test_short_position_logic_violations_raise_value_error(
        self, entry_price: float, stop_loss: float, take_profit: float
    ):
        """AC3: Stop loss <= entry or take profit >= entry on SHORT must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_risk_reward(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type="SHORT",
            )
        assert len(str(exc_info.value).strip()) > 0


class TestRiskRewardBoundaryAndInputValidation:
    """Tests verifying input validation for invalid position types and non-positive prices."""

    @pytest.mark.parametrize(
        ("entry_price", "stop_loss", "take_profit", "pos_type"),
        [
            (0.0, 95.0, 115.0, "LONG"),
            (-100.0, 95.0, 115.0, "LONG"),
            (100.0, 0.0, 115.0, "LONG"),
            (100.0, -95.0, 115.0, "LONG"),
            (100.0, 95.0, 0.0, "LONG"),
            (100.0, 95.0, -115.0, "LONG"),
            (0.0, 105.0, 85.0, "SHORT"),
            (-100.0, 105.0, 85.0, "SHORT"),
            (100.0, 0.0, 85.0, "SHORT"),
            (100.0, 105.0, 0.0, "SHORT"),
        ],
        ids=[
            "long_zero_entry",
            "long_negative_entry",
            "long_zero_sl",
            "long_negative_sl",
            "long_zero_tp",
            "long_negative_tp",
            "short_zero_entry",
            "short_negative_entry",
            "short_zero_sl",
            "short_zero_tp",
        ],
    )
    def test_non_positive_prices_raise_value_error(
        self, entry_price: float, stop_loss: float, take_profit: float, pos_type: str
    ):
        """Prices must be positive numbers; zero or negative values must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_risk_reward(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type=pos_type,
            )
        assert len(str(exc_info.value).strip()) > 0

    @pytest.mark.parametrize(
        "invalid_pos_type",
        ["BUY", "SELL", "HOLD", "INVALID", "", "123"],
    )
    def test_invalid_position_type_raises_value_error(self, invalid_pos_type: str):
        """Unsupported position types must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_risk_reward(
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=115.0,
                position_type=invalid_pos_type,
            )
        assert len(str(exc_info.value).strip()) > 0