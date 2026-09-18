"""
Unit tests for Maintenance Margin Calculator and Liquidation Engine.

Story 6.2.2: Implement Liquidation Engine and Maintenance Margin Calculator
Target Modules:
- src/risk/margin_calculator.py
- src/risk/liquidation_engine.py
"""

from decimal import Decimal
import pytest

from src.risk.margin_calculator import (
    Account,
    MaintenanceMarginCalculator,
    MarginEvaluation,
    Position,
)
from src.risk.liquidation_engine import (
    LiquidationEngine,
    LiquidationEvaluation,
    LiquidationEvent,
    LiquidationSide,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def margin_calculator() -> MaintenanceMarginCalculator:
    """Provides a clean instance of MaintenanceMarginCalculator."""
    return MaintenanceMarginCalculator()


@pytest.fixture
def liquidation_engine(
    margin_calculator: MaintenanceMarginCalculator,
) -> LiquidationEngine:
    """Provides a clean instance of LiquidationEngine configured with the calculator."""
    return LiquidationEngine(margin_calculator=margin_calculator)


@pytest.fixture
def btc_usd_mark_prices() -> dict[str, Decimal]:
    """Standard mark prices mapping for single-asset tests."""
    return {"BTC-USDT": Decimal("50000.00")}


@pytest.fixture
def multi_asset_mark_prices() -> dict[str, Decimal]:
    """Standard mark prices mapping for multi-asset tests."""
    return {
        "BTC-USDT": Decimal("50000.00"),
        "ETH-USDT": Decimal("3000.00"),
        "SOL-USDT": Decimal("100.00"),
    }


# ============================================================================
# Acceptance Criteria 1:
# Given an account with active positions and equity greater than the calculated
# maintenance margin requirement,
# When the liquidation engine evaluates the account,
# Then the account is marked solvent and no liquidation events are generated.
# ============================================================================


def test_liquidation_engine_evaluates_solvent_account_with_no_events(
    liquidation_engine: LiquidationEngine,
    btc_usd_mark_prices: dict[str, Decimal],
):
    """
    Solvent account with equity > MMR requirement must be marked solvent
    and yield zero liquidation events.
    """
    # 1 BTC long @ entry 50,000. Mark = 50,000. Unrealized PnL = 0.
    # Balance = 10,000 -> Equity = 10,000.
    # Maintenance margin rate = 5% (0.05).
    # Maintenance margin requirement = 1 * 50,000 * 0.05 = 2,500.
    # Equity (10,000) > MM (2,500).
    account = Account(
        account_id="ACC_SOLVENT_01",
        balance=Decimal("10000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.05"),
            )
        ],
    )

    evaluation: LiquidationEvaluation = liquidation_engine.evaluate_account(
        account, btc_usd_mark_prices
    )

    assert evaluation.account_id == "ACC_SOLVENT_01"
    assert evaluation.is_solvent is True
    assert evaluation.shortfall == Decimal("0.00")
    assert evaluation.liquidation_events == []
    assert len(evaluation.liquidation_events) == 0


def test_liquidation_engine_evaluates_solvent_account_with_unrealized_profit(
    liquidation_engine: LiquidationEngine,
    btc_usd_mark_prices: dict[str, Decimal],
):
    """
    Account with unrealized profit increases equity well beyond MMR requirement;
    must be marked solvent without liquidation events.
    """
    # Entry 40,000, Mark 50,000 -> PnL = +10,000.
    # Balance = 2,000 -> Equity = 12,000.
    # MM = 1 * 50,000 * 0.10 = 5,000.
    # Equity (12,000) > MM (5,000).
    account = Account(
        account_id="ACC_PROFIT_01",
        balance=Decimal("2000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("40000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    evaluation = liquidation_engine.evaluate_account(account, btc_usd_mark_prices)

    assert evaluation.is_solvent is True
    assert evaluation.shortfall == Decimal("0.00")
    assert evaluation.liquidation_events == []


def test_liquidation_engine_evaluates_exact_threshold_solvency(
    liquidation_engine: LiquidationEngine,
    btc_usd_mark_prices: dict[str, Decimal],
):
    """
    When equity exactly matches maintenance margin requirement,
    the account remains solvent and no liquidation events are created.
    """
    # Position: 1 BTC, Mark = 50,000, MMR = 0.10 -> MM = 5,000.
    # Balance = 5,000, Entry = 50,000 -> Equity = 5,000.
    account = Account(
        account_id="ACC_THRESHOLD_01",
        balance=Decimal("5000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    evaluation = liquidation_engine.evaluate_account(account, btc_usd_mark_prices)

    assert evaluation.is_solvent is True
    assert evaluation.shortfall == Decimal("0.00")
    assert evaluation.liquidation_events == []


def test_liquidation_engine_process_account_returns_empty_when_solvent(
    liquidation_engine: LiquidationEngine,
    btc_usd_mark_prices: dict[str, Decimal],
):
    """
    When process_account is called on a solvent account,
    it returns an empty list of liquidation orders/events.
    """
    account = Account(
        account_id="ACC_SOLVENT_PROCESS",
        balance=Decimal("20000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.05"),
            )
        ],
    )

    events = liquidation_engine.process_account(account, btc_usd_mark_prices)

    assert events == []
    assert len(events) == 0


# ============================================================================
# Acceptance Criteria 2:
# Given an account whose equity falls below the required maintenance margin,
# When the maintenance margin calculator evaluates account solvency,
# Then it flags a margin deficit and returns the shortfall amount.
# ============================================================================


def test_margin_calculator_flags_deficit_and_returns_shortfall(
    margin_calculator: MaintenanceMarginCalculator,
):
    """
    AC 2: When equity falls below maintenance margin, calculator flags deficit
    and returns exact shortfall amount (Shortfall = MM - Equity).
    """
    mark_prices = {"BTC-USDT": Decimal("45000.00")}

    # Entry 50,000, Mark drops to 45,000 -> Unrealized PnL = (45,000 - 50,000) * 1.0 = -5,000.
    # Cash balance = 7,000 -> Equity = 7,000 - 5,000 = 2,000.
    # Maintenance margin rate = 0.10.
    # Maintenance margin = 1.0 * 45,000 * 0.10 = 4,500.
    # Shortfall = 4,500 - 2,000 = 2,500.
    account = Account(
        account_id="ACC_DEFICIT_01",
        balance=Decimal("7000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    result: MarginEvaluation = margin_calculator.evaluate_solvency(
        account, mark_prices
    )

    assert result.account_id == "ACC_DEFICIT_01"
    assert result.is_solvent is False
    assert result.equity == Decimal("2000.00")
    assert result.maintenance_margin == Decimal("4500.00")
    assert result.shortfall == Decimal("2500.00")


def test_margin_calculator_deficit_with_negative_equity(
    margin_calculator: MaintenanceMarginCalculator,
):
    """
    When account equity turns negative, shortfall must correctly include
    the negative equity plus the full maintenance margin requirement.
    """
    mark_prices = {"BTC-USDT": Decimal("30000.00")}

    # Entry 50,000, Mark drops to 30,000 -> PnL = (30,000 - 50,000) * 1.0 = -20,000.
    # Balance = 5,000 -> Equity = 5,000 - 20,000 = -15,000.
    # MM = 1.0 * 30,000 * 0.10 = 3,000.
    # Shortfall = MM - Equity = 3,000 - (-15,000) = 18,000.
    account = Account(
        account_id="ACC_BANKRUPT_01",
        balance=Decimal("5000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    result = margin_calculator.evaluate_solvency(account, mark_prices)

    assert result.is_solvent is False
    assert result.equity == Decimal("-15000.00")
    assert result.maintenance_margin == Decimal("3000.00")
    assert result.shortfall == Decimal("18000.00")


def test_margin_calculator_deficit_short_position(
    margin_calculator: MaintenanceMarginCalculator,
):
    """
    Short positions in adverse market moves reduce equity and trigger shortfall.
    """
    mark_prices = {"ETH-USDT": Decimal("4000.00")}

    # Short position: size = -10 ETH @ entry 3,000. Mark rises to 4,000.
    # PnL = (entry - mark) * abs(size) = (3,000 - 4,000) * 10 = -10,000.
    # Balance = 12,000 -> Equity = 12,000 - 10,000 = 2,000.
    # MM = abs(-10) * 4,000 * 0.08 = 3,200.
    # Shortfall = 3,200 - 2,000 = 1,200.
    account = Account(
        account_id="ACC_SHORT_DEFICIT",
        balance=Decimal("12000.00"),
        positions=[
            Position(
                symbol="ETH-USDT",
                size=Decimal("-10.0"),
                entry_price=Decimal("3000.00"),
                maintenance_margin_rate=Decimal("0.08"),
            )
        ],
    )

    result = margin_calculator.evaluate_solvency(account, mark_prices)

    assert result.is_solvent is False
    assert result.equity == Decimal("2000.00")
    assert result.maintenance_margin == Decimal("3200.00")
    assert result.shortfall == Decimal("1200.00")


def test_margin_calculator_multi_position_aggregation(
    margin_calculator: MaintenanceMarginCalculator,
    multi_asset_mark_prices: dict[str, Decimal],
):
    """
    Maintenance margin and equity calculations must accurately aggregate across multiple positions.
    """
    # BTC Long: size=1.0, entry=50,000, mark=50,000, MMR=0.05 -> PnL=0, MM=2,500.
    # ETH Short: size=-5.0, entry=2,500, mark=3,000, MMR=0.05 -> PnL=-2,500, MM=750.
    # SOL Long: size=20.0, entry=120, mark=100, MMR=0.10 -> PnL=-400, MM=200.
    # Total PnL = 0 - 2,500 - 400 = -2,900.
    # Balance = 5,000 -> Equity = 5,000 - 2,900 = 2,100.
    # Total MM = 2,500 + 750 + 200 = 3,450.
    # Shortfall = 3,450 - 2,100 = 1,350.
    account = Account(
        account_id="ACC_MULTI_01",
        balance=Decimal("5000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.05"),
            ),
            Position(
                symbol="ETH-USDT",
                size=Decimal("-5.0"),
                entry_price=Decimal("2500.00"),
                maintenance_margin_rate=Decimal("0.05"),
            ),
            Position(
                symbol="SOL-USDT",
                size=Decimal("20.0"),
                entry_price=Decimal("120.00"),
                maintenance_margin_rate=Decimal("0.10"),
            ),
        ],
    )

    result = margin_calculator.evaluate_solvency(account, multi_asset_mark_prices)

    assert result.is_solvent is False
    assert result.equity == Decimal("2100.00")
    assert result.maintenance_margin == Decimal("3450.00")
    assert result.shortfall == Decimal("1350.00")


# ============================================================================
# Acceptance Criteria 3:
# Given an account flagged with a margin deficit,
# When the liquidation engine processes the account,
# Then it generates a liquidation event with the required liquidation volume
# and mark price.
# ============================================================================


def test_liquidation_engine_generates_event_for_long_position_deficit(
    liquidation_engine: LiquidationEngine,
):
    """
    AC 3: For an account flagged with a deficit, processing generates a liquidation
    event with side SELL, current mark price, and required liquidation volume.
    """
    mark_price = Decimal("40000.00")
    mark_prices = {"BTC-USDT": mark_price}

    # Position: Long 2.0 BTC @ 50,000, MMR = 0.10.
    # PnL = (40,000 - 50,000) * 2.0 = -20,000.
    # Balance = 15,000 -> Equity = 15,000 - 20,000 = -5,000.
    # MM = 2.0 * 40,000 * 0.10 = 8,000.
    # Shortfall = 8,000 - (-5,000) = 13,000.
    # Since Shortfall (13,000) >= MM of position (8,000), 100% of position must be liquidated:
    # Volume = 2.0.
    account = Account(
        account_id="ACC_LIQ_LONG_01",
        balance=Decimal("15000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("2.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    events: list[LiquidationEvent] = liquidation_engine.process_account(
        account, mark_prices
    )

    assert len(events) == 1
    event = events[0]
    assert event.account_id == "ACC_LIQ_LONG_01"
    assert event.symbol == "BTC-USDT"
    assert event.side == LiquidationSide.SELL
    assert event.mark_price == mark_price
    assert event.volume == Decimal("2.0")


def test_liquidation_engine_generates_event_for_short_position_deficit(
    liquidation_engine: LiquidationEngine,
):
    """
    AC 3: Deficit on a short position generates a BUY liquidation event with the mark price.
    """
    mark_price = Decimal("3500.00")
    mark_prices = {"ETH-USDT": mark_price}

    # Short position: -10.0 ETH @ entry 2,500. Mark = 3,500.
    # PnL = (2,500 - 3,500) * 10 = -10,000.
    # Balance = 7,000 -> Equity = 7,000 - 10,000 = -3,000.
    # MM = 10.0 * 3,500 * 0.10 = 3,500.
    # Shortfall = 3,500 - (-3,000) = 6,500 >= MM (3,500). Full liquidation required.
    account = Account(
        account_id="ACC_LIQ_SHORT_01",
        balance=Decimal("7000.00"),
        positions=[
            Position(
                symbol="ETH-USDT",
                size=Decimal("-10.0"),
                entry_price=Decimal("2500.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    events = liquidation_engine.process_account(account, mark_prices)

    assert len(events) == 1
    event = events[0]
    assert event.account_id == "ACC_LIQ_SHORT_01"
    assert event.symbol == "ETH-USDT"
    assert event.side == LiquidationSide.BUY
    assert event.mark_price == mark_price
    assert event.volume == Decimal("10.0")


def test_liquidation_engine_calculates_required_partial_liquidation_volume(
    liquidation_engine: LiquidationEngine,
):
    """
    When shortfall can be resolved by partial reduction of the position:
    Required volume = Shortfall / (Mark Price * MMR).
    """
    mark_price = Decimal("48000.00")
    mark_prices = {"BTC-USDT": mark_price}

    # Position: Long 1.0 BTC @ entry 50,000, MMR = 0.10.
    # PnL = (48,000 - 50,000) * 1.0 = -2,000.
    # Balance = 5,000 -> Equity = 5,000 - 2,000 = 3,000.
    # MM = 1.0 * 48,000 * 0.10 = 4,800.
    # Shortfall = 4,800 - 3,000 = 1,800.
    # Required Volume = Shortfall / (Mark * MMR) = 1,800 / (48,000 * 0.10) = 1,800 / 4,800 = 0.375 BTC.
    account = Account(
        account_id="ACC_PARTIAL_01",
        balance=Decimal("5000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    events = liquidation_engine.process_account(account, mark_prices)

    assert len(events) == 1
    event = events[0]
    assert event.symbol == "BTC-USDT"
    assert event.mark_price == mark_price
    assert event.volume == Decimal("0.375")
    assert event.side == LiquidationSide.SELL


def test_liquidation_engine_caps_liquidation_volume_to_open_position(
    liquidation_engine: LiquidationEngine,
):
    """
    Partial volume calculation must be bounded by the maximum open position size.
    """
    mark_price = Decimal("10000.00")
    mark_prices = {"BTC-USDT": mark_price}

    # Position: Long 0.5 BTC @ entry 50,000, MMR = 0.10.
    # PnL = (10,000 - 50,000) * 0.5 = -20,000.
    # Balance = 1,000 -> Equity = -19,000.
    # MM = 0.5 * 10,000 * 0.10 = 500.
    # Shortfall = 500 - (-19,000) = 19,500.
    # Uncapped formula gives: 19,500 / 1,000 = 19.5 BTC > 0.5 BTC.
    # Must be capped at total open position: 0.5 BTC.
    account = Account(
        account_id="ACC_CAP_01",
        balance=Decimal("1000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    events = liquidation_engine.process_account(account, mark_prices)

    assert len(events) == 1
    assert events[0].volume == Decimal("0.5")


def test_liquidation_engine_processes_multiple_positions_until_deficit_covered(
    liquidation_engine: LiquidationEngine,
    multi_asset_mark_prices: dict[str, Decimal],
):
    """
    When multiple positions exist, engine generates events across positions
    to cover the cumulative shortfall.
    """
    # BTC: Long 1.0 @ 60,000, Mark=50,000, MMR=0.10 -> PnL = -10,000, MM = 5,000.
    # ETH: Long 10.0 @ 4,000, Mark=3,000, MMR=0.10 -> PnL = -10,000, MM = 3,000.
    # Balance = 12,000 -> Equity = 12,000 - 20,000 = -8,000.
    # Total MM = 8,000.
    # Total Shortfall = 8,000 - (-8,000) = 16,000.
    # Both positions must be liquidated completely.
    account = Account(
        account_id="ACC_MULTI_LIQ_01",
        balance=Decimal("12000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("60000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            ),
            Position(
                symbol="ETH-USDT",
                size=Decimal("10.0"),
                entry_price=Decimal("4000.00"),
                maintenance_margin_rate=Decimal("0.10"),
            ),
        ],
    )

    events = liquidation_engine.process_account(account, multi_asset_mark_prices)

    assert len(events) == 2
    event_symbols = {e.symbol for e in events}
    assert event_symbols == {"BTC-USDT", "ETH-USDT"}

    for event in events:
        assert event.mark_price == multi_asset_mark_prices[event.symbol]
        assert event.volume > Decimal("0.0")
        assert event.side == LiquidationSide.SELL


# ============================================================================
# Edge Cases & Validation Tests
# ============================================================================


def test_margin_calculator_missing_mark_price_raises_error(
    margin_calculator: MaintenanceMarginCalculator,
):
    """
    Evaluating solvency when mark price is unavailable for an active position
    must raise a KeyError.
    """
    incomplete_prices = {"ETH-USDT": Decimal("3000.00")}

    account = Account(
        account_id="ACC_MISSING_PRICE",
        balance=Decimal("5000.00"),
        positions=[
            Position(
                symbol="BTC-USDT",
                size=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                maintenance_margin_rate=Decimal("0.05"),
            )
        ],
    )

    with pytest.raises(KeyError):
        margin_calculator.evaluate_solvency(account, incomplete_prices)


def test_liquidation_engine_missing_mark_price_raises_error(
    liquidation_engine: LiquidationEngine,
):
    """
    Liquidation engine processing an account with missing mark price
    must raise a KeyError.
    """
    incomplete_prices = {"BTC-USDT": Decimal("50000.00")}

    account = Account(
        account_id="ACC_MISSING_PRICE_ENGINE",
        balance=Decimal("1000.00"),
        positions=[
            Position(
                symbol="SOL-USDT",
                size=Decimal("10.0"),
                entry_price=Decimal("150.00"),
                maintenance_margin_rate=Decimal("0.10"),
            )
        ],
    )

    with pytest.raises(KeyError):
        liquidation_engine.process_account(account, incomplete_prices)


def test_zero_positions_account_is_solvent(
    margin_calculator: MaintenanceMarginCalculator,
    liquidation_engine: LiquidationEngine,
):
    """
    An account with zero positions and non-negative balance has 0 MM requirement,
    is solvent, and yields no liquidation events.
    """
    account = Account(
        account_id="ACC_EMPTY",
        balance=Decimal("1000.00"),
        positions=[],
    )

    margin_result = margin_calculator.evaluate_solvency(account, {})
    assert margin_result.is_solvent is True
    assert margin_result.shortfall == Decimal("0.00")
    assert margin_result.maintenance_margin == Decimal("0.00")

    events = liquidation_engine.process_account(account, {})
    assert events == []


def test_invalid_maintenance_margin_rate_raises_error():
    """
    A position initialized with a negative or zero maintenance margin rate
    must raise ValueError.
    """
    with pytest.raises(ValueError):
        Position(
            symbol="BTC-USDT",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            maintenance_margin_rate=Decimal("-0.05"),
        )