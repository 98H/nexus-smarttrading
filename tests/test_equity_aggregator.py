from datetime import datetime, timezone
from typing import List

import pytest

from src.analytics.equity_aggregator import (
    EquityCurveAggregator,
    EquityPoint,
    Trade,
)


@pytest.fixture
def base_timestamp() -> datetime:
    """Provides a consistent base UTC timestamp for test trade sequences."""
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# AC 1: Chronological Sequence Processing & Running Cumulative Metrics
# ============================================================================


def test_processes_chronological_trades_cumulative_pnl_and_equity(
    base_timestamp: datetime,
) -> None:
    """Given an initial cash balance and a chronological sequence of closed trades,

    When processed,
    Then it produces a trade log containing running cumulative PnL and cumulative equity
    at each trade timestamp.
    """
    initial_cash = 50_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    t1_time = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    t2_time = datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    t3_time = datetime(2025, 1, 15, 11, 45, 0, tzinfo=timezone.utc)

    trades: List[Trade] = [
        Trade(trade_id="T-001", timestamp=t1_time, realized_pnl=1_500.0),
        Trade(trade_id="T-002", timestamp=t2_time, realized_pnl=-500.0),
        Trade(trade_id="T-003", timestamp=t3_time, realized_pnl=2_000.0),
    ]

    trade_log: List[EquityPoint] = aggregator.process(trades)

    assert len(trade_log) == 3

    # Step 1: Initial (50,000) + 1,500 = 51,500
    assert trade_log[0].timestamp == t1_time
    assert trade_log[0].realized_pnl == pytest.approx(1_500.0)
    assert trade_log[0].cumulative_pnl == pytest.approx(1_500.0)
    assert trade_log[0].cumulative_equity == pytest.approx(51_500.0)

    # Step 2: 51,500 - 500 = 51,000 (cum_pnl = 1,000)
    assert trade_log[1].timestamp == t2_time
    assert trade_log[1].realized_pnl == pytest.approx(-500.0)
    assert trade_log[1].cumulative_pnl == pytest.approx(1_000.0)
    assert trade_log[1].cumulative_equity == pytest.approx(51_000.0)

    # Step 3: 51,000 + 2,000 = 53,000 (cum_pnl = 3,000)
    assert trade_log[2].timestamp == t3_time
    assert trade_log[2].realized_pnl == pytest.approx(2_000.0)
    assert trade_log[2].cumulative_pnl == pytest.approx(3_000.0)
    assert trade_log[2].cumulative_equity == pytest.approx(53_000.0)


def test_single_trade_processing_matches_timestamp_and_totals(
    base_timestamp: datetime,
) -> None:
    """A sequence of exactly one trade must produce a trade log of length 1

    with correct running cumulative values matching the single trade.
    """
    initial_cash = 10_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    trade = Trade(trade_id="T-SOLO", timestamp=base_timestamp, realized_pnl=-750.0)
    trade_log = aggregator.process([trade])

    assert len(trade_log) == 1
    point = trade_log[0]
    assert point.timestamp == base_timestamp
    assert point.realized_pnl == pytest.approx(-750.0)
    assert point.cumulative_pnl == pytest.approx(-750.0)
    assert point.cumulative_equity == pytest.approx(9_250.0)


# ============================================================================
# AC 2: Empty Trades Sequence Handling (Baseline Point)
# ============================================================================


def test_empty_trades_returns_single_baseline_equity_point() -> None:
    """Given an empty list of trades and an initial cash balance,

    When the aggregator is invoked,
    Then it returns a single baseline equity point representing the initial balance
    with zero cumulative PnL and zero drawdown.
    """
    initial_cash = 25_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    trade_log = aggregator.process([])

    assert len(trade_log) == 1
    baseline = trade_log[0]

    assert isinstance(baseline, EquityPoint)
    assert baseline.timestamp is None
    assert baseline.realized_pnl == pytest.approx(0.0)
    assert baseline.cumulative_pnl == pytest.approx(0.0)
    assert baseline.cumulative_equity == pytest.approx(initial_cash)
    assert baseline.peak_equity == pytest.approx(initial_cash)
    assert baseline.drawdown == pytest.approx(0.0)
    assert baseline.drawdown_pct == pytest.approx(0.0)


# ============================================================================
# AC 3: Drawdown Tracking and Cumulative Peak Metrics
# ============================================================================


def test_drawdown_metrics_when_equity_drops_below_previous_peak(
    base_timestamp: datetime,
) -> None:
    """Given an equity curve experiencing drops below previous peak equity,

    When cumulative metrics are calculated,
    Then each step accurately records peak equity, current drawdown value, and drawdown percentage.
    """
    initial_cash = 10_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    t1 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    t4 = datetime(2025, 1, 15, 13, 0, tzinfo=timezone.utc)
    t5 = datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc)

    trades = [
        # Step 1: Gain +2,000 -> Equity 12,000 (New Peak: 12,000, DD: 0.0, DD%: 0.0)
        Trade(trade_id="T-1", timestamp=t1, realized_pnl=2_000.0),
        # Step 2: Loss -3,000 -> Equity 9,000 (Peak: 12,000, DD: 3,000, DD%: 25.0%)
        Trade(trade_id="T-2", timestamp=t2, realized_pnl=-3_000.0),
        # Step 3: Loss -1,800 -> Equity 7,200 (Peak: 12,000, DD: 4,800, DD%: 40.0%)
        Trade(trade_id="T-3", timestamp=t3, realized_pnl=-1_800.0),
        # Step 4: Recovery +2,400 -> Equity 9,600 (Peak: 12,000, DD: 2,400, DD%: 20.0%)
        Trade(trade_id="T-4", timestamp=t4, realized_pnl=2_400.0),
        # Step 5: Breakout +3,000 -> Equity 12,600 (New Peak: 12,600, DD: 0.0, DD%: 0.0)
        Trade(trade_id="T-5", timestamp=t5, realized_pnl=3_000.0),
    ]

    trade_log = aggregator.process(trades)

    assert len(trade_log) == 5

    # Step 1: New peak established above initial cash
    assert trade_log[0].cumulative_equity == pytest.approx(12_000.0)
    assert trade_log[0].peak_equity == pytest.approx(12_000.0)
    assert trade_log[0].drawdown == pytest.approx(0.0)
    assert trade_log[0].drawdown_pct == pytest.approx(0.0)

    # Step 2: First drawdown step
    assert trade_log[1].cumulative_equity == pytest.approx(9_000.0)
    assert trade_log[1].peak_equity == pytest.approx(12_000.0)
    assert trade_log[1].drawdown == pytest.approx(3_000.0)
    assert trade_log[1].drawdown_pct == pytest.approx(0.25)

    # Step 3: Deepened drawdown step
    assert trade_log[2].cumulative_equity == pytest.approx(7_200.0)
    assert trade_log[2].peak_equity == pytest.approx(12_000.0)
    assert trade_log[2].drawdown == pytest.approx(4_800.0)
    assert trade_log[2].drawdown_pct == pytest.approx(0.40)

    # Step 4: Partial recovery while still below peak
    assert trade_log[3].cumulative_equity == pytest.approx(9_600.0)
    assert trade_log[3].peak_equity == pytest.approx(12_000.0)
    assert trade_log[3].drawdown == pytest.approx(2_400.0)
    assert trade_log[3].drawdown_pct == pytest.approx(0.20)

    # Step 5: Drawdown fully recovered, new all-time peak
    assert trade_log[4].cumulative_equity == pytest.approx(12_600.0)
    assert trade_log[4].peak_equity == pytest.approx(12_600.0)
    assert trade_log[4].drawdown == pytest.approx(0.0)
    assert trade_log[4].drawdown_pct == pytest.approx(0.0)


def test_drawdown_from_initial_cash_with_immediate_losses() -> None:
    """When the first trades are losses, the initial cash balance acts as the baseline peak."""
    initial_cash = 20_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    trades = [
        Trade(
            trade_id="T-1",
            timestamp=datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc),
            realized_pnl=-5_000.0,
        ),
        Trade(
            trade_id="T-2",
            timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            realized_pnl=-1_000.0,
        ),
    ]

    trade_log = aggregator.process(trades)

    # Step 1: 20,000 -> 15,000 (DD: 5,000, DD%: 25.0%)
    assert trade_log[0].peak_equity == pytest.approx(20_000.0)
    assert trade_log[0].cumulative_equity == pytest.approx(15_000.0)
    assert trade_log[0].drawdown == pytest.approx(5_000.0)
    assert trade_log[0].drawdown_pct == pytest.approx(0.25)

    # Step 2: 15,000 -> 14,000 (DD: 6,000, DD%: 30.0%)
    assert trade_log[1].peak_equity == pytest.approx(20_000.0)
    assert trade_log[1].cumulative_equity == pytest.approx(14_000.0)
    assert trade_log[1].drawdown == pytest.approx(6_000.0)
    assert trade_log[1].drawdown_pct == pytest.approx(0.30)


# ============================================================================
# Edge Cases & Validation Rules
# ============================================================================


def test_processes_unsorted_trades_in_chronological_order() -> None:
    """Out-of-order trades must be normalized and processed chronologically."""
    initial_cash = 10_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    t1 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)

    # Input delivered out of chronological order: T3, T1, T2
    unsorted_trades = [
        Trade(trade_id="T-3", timestamp=t3, realized_pnl=300.0),
        Trade(trade_id="T-1", timestamp=t1, realized_pnl=100.0),
        Trade(trade_id="T-2", timestamp=t2, realized_pnl=-200.0),
    ]

    trade_log = aggregator.process(unsorted_trades)

    assert len(trade_log) == 3
    assert trade_log[0].timestamp == t1
    assert trade_log[0].cumulative_equity == pytest.approx(10_100.0)

    assert trade_log[1].timestamp == t2
    assert trade_log[1].cumulative_equity == pytest.approx(9_900.0)

    assert trade_log[2].timestamp == t3
    assert trade_log[2].cumulative_equity == pytest.approx(10_200.0)


def test_multiple_trades_at_identical_timestamp_processed_deterministically() -> None:
    """Multiple trades sharing the exact same timestamp must be processed sequentially."""
    initial_cash = 10_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)
    same_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    trades = [
        Trade(trade_id="T-A", timestamp=same_time, realized_pnl=500.0),
        Trade(trade_id="T-B", timestamp=same_time, realized_pnl=-200.0),
    ]

    trade_log = aggregator.process(trades)

    assert len(trade_log) == 2
    assert trade_log[0].timestamp == same_time
    assert trade_log[0].cumulative_equity == pytest.approx(10_500.0)
    assert trade_log[1].timestamp == same_time
    assert trade_log[1].cumulative_equity == pytest.approx(10_300.0)


def test_zero_pnl_trade_preserves_equity_and_drawdown(
    base_timestamp: datetime,
) -> None:
    """A break-even trade (0.0 PnL) must not distort equity, peak, or drawdown values."""
    initial_cash = 10_000.0
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    t1 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)

    trades = [
        Trade(trade_id="T-1", timestamp=t1, realized_pnl=-1_000.0),
        Trade(trade_id="T-2", timestamp=t2, realized_pnl=0.0),
    ]

    trade_log = aggregator.process(trades)

    assert len(trade_log) == 2
    assert trade_log[1].realized_pnl == pytest.approx(0.0)
    assert trade_log[1].cumulative_equity == pytest.approx(9_000.0)
    assert trade_log[1].peak_equity == pytest.approx(10_000.0)
    assert trade_log[1].drawdown == pytest.approx(1_000.0)
    assert trade_log[1].drawdown_pct == pytest.approx(0.10)


@pytest.mark.parametrize("invalid_cash", [0.0, -1.0, -10_000.0])
def test_non_positive_initial_cash_raises_value_error(invalid_cash: float) -> None:
    """Initial cash balance must be strictly positive to define a valid equity curve."""
    with pytest.raises(ValueError):
        EquityCurveAggregator(initial_cash=invalid_cash)


def test_floating_point_precision_handling(base_timestamp: datetime) -> None:
    """Aggregator must maintain precision and avoid catastrophic cancellation with cents."""
    initial_cash = 100.05
    aggregator = EquityCurveAggregator(initial_cash=initial_cash)

    trades = [
        Trade(
            trade_id="T-1",
            timestamp=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            realized_pnl=0.10,
        ),
        Trade(
            trade_id="T-2",
            timestamp=datetime(2025, 1, 15, 10, 1, tzinfo=timezone.utc),
            realized_pnl=0.20,
        ),
        Trade(
            trade_id="T-3",
            timestamp=datetime(2025, 1, 15, 10, 2, tzinfo=timezone.utc),
            realized_pnl=-0.30,
        ),
    ]

    trade_log = aggregator.process(trades)

    assert trade_log[0].cumulative_equity == pytest.approx(100.15)
    assert trade_log[1].cumulative_equity == pytest.approx(100.35)
    assert trade_log[2].cumulative_equity == pytest.approx(100.05)
    assert trade_log[2].cumulative_pnl == pytest.approx(0.0)