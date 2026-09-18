"""
Unit tests for Story 6.1.3: Build Slippage, Commission, and Spread Emulation Models.

Acceptance Criteria:
- Given a trade order with a mid-market price and a spread model,
  When calculating fill prices,
  Then buy orders execute at `mid + (spread / 2)` and sell orders execute at `mid - (spread / 2)`.
- Given an executed order size, market volume, and a slippage model,
  When calculating execution price impact,
  Then the fill price shifts proportionally according to the slippage model's impact rate.
- Given an executed trade notional value and a commission model with a percentage rate and minimum fee,
  When computing transaction costs,
  Then the commission returned equals the greater of the percentage of notional value or the minimum fee.

Target Modules to Test:
- src/execution/cost_models.py
- src/execution/__init__.py
"""

import pytest

from src.execution import (
    CommissionModel,
    OrderSide,
    SlippageModel,
    SpreadModel,
)
from src.execution.cost_models import (
    CommissionModel as DirectCommissionModel,
    OrderSide as DirectOrderSide,
    SlippageModel as DirectSlippageModel,
    SpreadModel as DirectSpreadModel,
)


# ==============================================================================
# Module Packaging and Export Tests
# ==============================================================================


class TestModuleExports:
    """Verify clean public interface and export parity between modules."""

    def test_cost_models_exports_are_available_in_execution_root(self):
        """Root execution namespace must expose all cost model classes and enums."""
        assert DirectOrderSide is OrderSide
        assert DirectSpreadModel is SpreadModel
        assert DirectSlippageModel is SlippageModel
        assert DirectCommissionModel is CommissionModel

    def test_order_side_enum_values(self):
        """OrderSide must define standard BUY and SELL sides."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


# ==============================================================================
# AC1: Spread Model Tests
# ==============================================================================


class TestSpreadModel:
    """
    Acceptance Criteria 1:
    Given a trade order with a mid-market price and a spread model,
    When calculating fill prices,
    Then buy orders execute at `mid + (spread / 2)` and sell orders execute at `mid - (spread / 2)`.
    """

    def test_buy_order_executes_at_mid_plus_half_spread(self):
        mid_price = 100.0
        spread = 2.0
        model = SpreadModel(spread=spread)

        fill_price = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.BUY)

        expected_price = mid_price + (spread / 2.0)
        assert fill_price == pytest.approx(expected_price)
        assert fill_price == pytest.approx(101.0)

    def test_sell_order_executes_at_mid_minus_half_spread(self):
        mid_price = 100.0
        spread = 2.0
        model = SpreadModel(spread=spread)

        fill_price = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.SELL)

        expected_price = mid_price - (spread / 2.0)
        assert fill_price == pytest.approx(expected_price)
        assert fill_price == pytest.approx(99.0)

    def test_spread_difference_between_buy_and_sell_equals_total_spread(self):
        mid_price = 150.50
        spread = 0.30
        model = SpreadModel(spread=spread)

        buy_fill = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.BUY)
        sell_fill = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.SELL)

        assert (buy_fill - sell_fill) == pytest.approx(spread)
        assert ((buy_fill + sell_fill) / 2.0) == pytest.approx(mid_price)

    def test_zero_spread_executes_at_exact_mid_price(self):
        mid_price = 245.75
        model = SpreadModel(spread=0.0)

        buy_fill = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.BUY)
        sell_fill = model.calculate_fill_price(mid_price=mid_price, side=OrderSide.SELL)

        assert buy_fill == pytest.approx(mid_price)
        assert sell_fill == pytest.approx(mid_price)

    @pytest.mark.parametrize(
        ("mid_price", "spread", "expected_buy", "expected_sell"),
        [
            (10.0, 0.02, 10.01, 9.99),
            (0.50, 0.05, 0.525, 0.475),
            (1234.56, 1.20, 1235.16, 1233.96),
            (50.0, 10.0, 55.0, 45.0),
        ],
    )
    def test_spread_calculation_across_price_scales(
        self, mid_price: float, spread: float, expected_buy: float, expected_sell: float
    ):
        model = SpreadModel(spread=spread)

        assert model.calculate_fill_price(mid_price, OrderSide.BUY) == pytest.approx(expected_buy)
        assert model.calculate_fill_price(mid_price, OrderSide.SELL) == pytest.approx(expected_sell)

    @pytest.mark.parametrize("side_str", ["BUY", "buy", "SELL", "sell"])
    def test_supports_case_insensitive_string_order_side(self, side_str: str):
        model = SpreadModel(spread=1.0)
        mid_price = 100.0

        fill_price = model.calculate_fill_price(mid_price=mid_price, side=side_str)

        if side_str.upper() == "BUY":
            assert fill_price == pytest.approx(100.5)
        else:
            assert fill_price == pytest.approx(99.5)

    def test_negative_spread_raises_value_error(self):
        with pytest.raises(ValueError):
            SpreadModel(spread=-0.01)

    def test_non_positive_mid_price_raises_value_error(self):
        model = SpreadModel(spread=0.50)
        with pytest.raises(ValueError):
            model.calculate_fill_price(mid_price=0.0, side=OrderSide.BUY)
        with pytest.raises(ValueError):
            model.calculate_fill_price(mid_price=-10.0, side=OrderSide.SELL)

    def test_invalid_order_side_raises_value_error(self):
        model = SpreadModel(spread=0.50)
        with pytest.raises(ValueError):
            model.calculate_fill_price(mid_price=100.0, side="INVALID_SIDE")


# ==============================================================================
# AC2: Slippage Model Tests
# ==============================================================================


class TestSlippageModel:
    """
    Acceptance Criteria 2:
    Given an executed order size, market volume, and a slippage model,
    When calculating execution price impact,
    Then the fill price shifts proportionally according to the slippage model's impact rate.
    """

    def test_price_impact_is_proportional_to_impact_rate(self):
        order_size = 500.0
        market_volume = 10_000.0
        volume_fraction = order_size / market_volume  # 0.05

        impact_rate_1 = 0.1
        impact_rate_2 = 0.2  # 2x impact rate

        model_1 = SlippageModel(impact_rate=impact_rate_1)
        model_2 = SlippageModel(impact_rate=impact_rate_2)

        impact_1 = model_1.calculate_price_impact(
            order_size=order_size, market_volume=market_volume
        )
        impact_2 = model_2.calculate_price_impact(
            order_size=order_size, market_volume=market_volume
        )

        expected_impact_1 = impact_rate_1 * volume_fraction
        expected_impact_2 = impact_rate_2 * volume_fraction

        assert impact_1 == pytest.approx(expected_impact_1)
        assert impact_2 == pytest.approx(expected_impact_2)
        assert impact_2 == pytest.approx(2.0 * impact_1)

    def test_buy_fill_price_shifts_upward_by_proportional_impact(self):
        base_price = 100.0
        order_size = 1_000.0
        market_volume = 10_000.0  # order is 10% of volume
        impact_rate = 0.05

        model = SlippageModel(impact_rate=impact_rate)

        # Expected shift: base_price * impact_rate * (order_size / market_volume)
        # 100.0 * 0.05 * 0.10 = 0.50
        fill_price = model.calculate_fill_price(
            base_price=base_price,
            order_size=order_size,
            market_volume=market_volume,
            side=OrderSide.BUY,
        )

        assert fill_price == pytest.approx(100.50)

    def test_sell_fill_price_shifts_downward_by_proportional_impact(self):
        base_price = 100.0
        order_size = 1_000.0
        market_volume = 10_000.0
        impact_rate = 0.05

        model = SlippageModel(impact_rate=impact_rate)

        # Sell price worsens downward: 100.0 * (1 - 0.005) = 99.50
        fill_price = model.calculate_fill_price(
            base_price=base_price,
            order_size=order_size,
            market_volume=market_volume,
            side=OrderSide.SELL,
        )

        assert fill_price == pytest.approx(99.50)

    def test_doubling_impact_rate_doubles_fill_price_shift(self):
        base_price = 200.0
        order_size = 250.0
        market_volume = 5_000.0

        model_standard = SlippageModel(impact_rate=0.04)
        model_doubled = SlippageModel(impact_rate=0.08)

        fill_standard = model_standard.calculate_fill_price(
            base_price=base_price,
            order_size=order_size,
            market_volume=market_volume,
            side=OrderSide.BUY,
        )
        fill_doubled = model_doubled.calculate_fill_price(
            base_price=base_price,
            order_size=order_size,
            market_volume=market_volume,
            side=OrderSide.BUY,
        )

        shift_standard = fill_standard - base_price
        shift_doubled = fill_doubled - base_price

        assert shift_doubled == pytest.approx(2.0 * shift_standard)

    def test_zero_order_size_produces_zero_price_shift(self):
        base_price = 150.0
        model = SlippageModel(impact_rate=0.1)

        fill_buy = model.calculate_fill_price(
            base_price=base_price,
            order_size=0.0,
            market_volume=10_000.0,
            side=OrderSide.BUY,
        )
        fill_sell = model.calculate_fill_price(
            base_price=base_price,
            order_size=0.0,
            market_volume=10_000.0,
            side=OrderSide.SELL,
        )

        assert fill_buy == pytest.approx(base_price)
        assert fill_sell == pytest.approx(base_price)

    def test_zero_impact_rate_produces_zero_price_shift(self):
        base_price = 100.0
        model = SlippageModel(impact_rate=0.0)

        fill_buy = model.calculate_fill_price(
            base_price=base_price,
            order_size=2_000.0,
            market_volume=10_000.0,
            side=OrderSide.BUY,
        )

        assert fill_buy == pytest.approx(base_price)

    def test_negative_impact_rate_raises_value_error(self):
        with pytest.raises(ValueError):
            SlippageModel(impact_rate=-0.01)

    def test_negative_order_size_raises_value_error(self):
        model = SlippageModel(impact_rate=0.1)
        with pytest.raises(ValueError):
            model.calculate_price_impact(order_size=-10.0, market_volume=1000.0)
        with pytest.raises(ValueError):
            model.calculate_fill_price(
                base_price=100.0,
                order_size=-10.0,
                market_volume=1000.0,
                side=OrderSide.BUY,
            )

    def test_non_positive_market_volume_raises_value_error(self):
        model = SlippageModel(impact_rate=0.1)
        with pytest.raises(ValueError):
            model.calculate_price_impact(order_size=10.0, market_volume=0.0)
        with pytest.raises(ValueError):
            model.calculate_price_impact(order_size=10.0, market_volume=-500.0)
        with pytest.raises(ValueError):
            model.calculate_fill_price(
                base_price=100.0,
                order_size=10.0,
                market_volume=0.0,
                side=OrderSide.BUY,
            )

    def test_non_positive_base_price_raises_value_error(self):
        model = SlippageModel(impact_rate=0.1)
        with pytest.raises(ValueError):
            model.calculate_fill_price(
                base_price=0.0,
                order_size=10.0,
                market_volume=1000.0,
                side=OrderSide.BUY,
            )
        with pytest.raises(ValueError):
            model.calculate_fill_price(
                base_price=-100.0,
                order_size=10.0,
                market_volume=1000.0,
                side=OrderSide.BUY,
            )


# ==============================================================================
# AC3: Commission Model Tests
# ==============================================================================


class TestCommissionModel:
    """
    Acceptance Criteria 3:
    Given an executed trade notional value and a commission model with a percentage rate and minimum fee,
    When computing transaction costs,
    Then the commission returned equals the greater of the percentage of notional value or the minimum fee.
    """

    def test_returns_percentage_commission_when_greater_than_min_fee(self):
        percentage_rate = 0.001  # 10 bps (0.1%)
        min_fee = 1.50
        model = CommissionModel(percentage_rate=percentage_rate, min_fee=min_fee)

        notional_value = 5_000.0
        # percentage commission = 5_000.0 * 0.001 = 5.00 > 1.50

        commission = model.calculate_commission(notional_value=notional_value)
        assert commission == pytest.approx(5.00)

    def test_returns_minimum_fee_when_greater_than_percentage_commission(self):
        percentage_rate = 0.001  # 10 bps (0.1%)
        min_fee = 2.50
        model = CommissionModel(percentage_rate=percentage_rate, min_fee=min_fee)

        notional_value = 500.0
        # percentage commission = 500.0 * 0.001 = 0.50 < 2.50

        commission = model.calculate_commission(notional_value=notional_value)
        assert commission == pytest.approx(2.50)

    def test_returns_exact_fee_when_percentage_equals_minimum_fee(self):
        percentage_rate = 0.002
        min_fee = 4.00
        model = CommissionModel(percentage_rate=percentage_rate, min_fee=min_fee)

        notional_value = 2_000.0
        # percentage commission = 2_000.0 * 0.002 = 4.00 == 4.00

        commission = model.calculate_commission(notional_value=notional_value)
        assert commission == pytest.approx(4.00)

    def test_supports_zero_minimum_fee(self):
        model = CommissionModel(percentage_rate=0.0005, min_fee=0.0)

        notional_value = 10_000.0
        assert model.calculate_commission(notional_value) == pytest.approx(5.00)

    def test_supports_zero_percentage_rate(self):
        min_fee = 9.99
        model = CommissionModel(percentage_rate=0.0, min_fee=min_fee)

        assert model.calculate_commission(notional_value=1_000_000.0) == pytest.approx(min_fee)
        assert model.calculate_commission(notional_value=0.0) == pytest.approx(min_fee)

    def test_zero_notional_value_returns_minimum_fee(self):
        min_fee = 1.25
        model = CommissionModel(percentage_rate=0.01, min_fee=min_fee)

        commission = model.calculate_commission(notional_value=0.0)
        assert commission == pytest.approx(min_fee)

    @pytest.mark.parametrize(
        ("percentage_rate", "min_fee", "notional", "expected_commission"),
        [
            (0.0005, 1.00, 100.0, 1.00),       # 0.05 < 1.00 -> 1.00
            (0.0005, 1.00, 2_000.0, 1.00),     # 1.00 == 1.00 -> 1.00
            (0.0005, 1.00, 10_000.0, 5.00),    # 5.00 > 1.00 -> 5.00
            (0.0025, 0.00, 4_000.0, 10.00),    # 10.00 > 0.00 -> 10.00
            (0.0, 0.0, 50_000.0, 0.00),        # 0.00 == 0.00 -> 0.00
        ],
    )
    def test_commission_matrix(
        self, percentage_rate: float, min_fee: float, notional: float, expected_commission: float
    ):
        model = CommissionModel(percentage_rate=percentage_rate, min_fee=min_fee)
        assert model.calculate_commission(notional) == pytest.approx(expected_commission)

    def test_negative_percentage_rate_raises_value_error(self):
        with pytest.raises(ValueError):
            CommissionModel(percentage_rate=-0.001, min_fee=1.0)

    def test_negative_minimum_fee_raises_value_error(self):
        with pytest.raises(ValueError):
            CommissionModel(percentage_rate=0.001, min_fee=-1.0)

    def test_negative_notional_value_raises_value_error(self):
        model = CommissionModel(percentage_rate=0.001, min_fee=1.0)
        with pytest.raises(ValueError):
            model.calculate_commission(notional_value=-50.0)


# ==============================================================================
# End-to-End Emulation Pipeline Integration Test
# ==============================================================================


class TestCostModelsIntegration:
    """
    Validates end-to-end integration across Spread, Slippage, and Commission models.
    """

    def test_trade_cost_emulation_pipeline(self):
        # Trade parameters
        mid_market_price = 100.0
        order_size = 1_000.0
        market_volume = 50_000.0  # order is 2% of volume
        spread_amount = 0.10
        slippage_rate = 0.05
        commission_rate = 0.0004  # 4 bps
        min_commission = 2.00

        # Model initializations
        spread_model = SpreadModel(spread=spread_amount)
        slippage_model = SlippageModel(impact_rate=slippage_rate)
        commission_model = CommissionModel(
            percentage_rate=commission_rate, min_fee=min_commission
        )

        # 1. Spread calculation: Buy executes on the ask (mid + spread / 2)
        spread_adjusted_price = spread_model.calculate_fill_price(
            mid_price=mid_market_price, side=OrderSide.BUY
        )
        assert spread_adjusted_price == pytest.approx(100.05)

        # 2. Slippage impact calculation on the spread-adjusted price
        # impact = 0.05 * (1000 / 50000) = 0.001 (0.1%)
        # fill_price = 100.05 * (1 + 0.001) = 100.15005
        final_fill_price = slippage_model.calculate_fill_price(
            base_price=spread_adjusted_price,
            order_size=order_size,
            market_volume=market_volume,
            side=OrderSide.BUY,
        )
        assert final_fill_price == pytest.approx(100.15005)

        # 3. Commission calculation on executed notional
        executed_notional = order_size * final_fill_price
        assert executed_notional == pytest.approx(100_150.05)

        # percentage fee = 100_150.05 * 0.0004 = 40.06002 > min_fee (2.00)
        commission = commission_model.calculate_commission(notional_value=executed_notional)
        assert commission == pytest.approx(40.06002)