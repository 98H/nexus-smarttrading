"""Trade log and cumulative equity curve aggregator module."""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Trade:
    """Represents a closed trade with its realized profit and loss."""

    trade_id: str
    timestamp: datetime
    realized_pnl: float


@dataclass(frozen=True)
class EquityPoint:
    """Represents an equity point on the cumulative equity curve."""

    timestamp: Optional[datetime]
    realized_pnl: float
    cumulative_pnl: float
    cumulative_equity: float
    peak_equity: float
    drawdown: float
    drawdown_pct: float
    trade_id: Optional[str] = None


class EquityCurveAggregator:
    """Aggregates closed trades into a running trade log with equity and drawdown metrics."""

    def __init__(self, initial_cash: float) -> None:
        if initial_cash <= 0.0:
            raise ValueError(f"initial_cash must be strictly positive, got {initial_cash}")
        self.initial_cash = float(initial_cash)

    def process(self, trades: Iterable[Trade]) -> List[EquityPoint]:
        """Process trade sequences chronologically and generate equity curve points."""
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        if not sorted_trades:
            return [
                EquityPoint(
                    timestamp=None,
                    realized_pnl=0.0,
                    cumulative_pnl=0.0,
                    cumulative_equity=self.initial_cash,
                    peak_equity=self.initial_cash,
                    drawdown=0.0,
                    drawdown_pct=0.0,
                )
            ]

        trade_log: List[EquityPoint] = []
        running_cum_pnl = 0.0
        peak_equity = self.initial_cash

        for trade in sorted_trades:
            running_cum_pnl += trade.realized_pnl
            current_equity = self.initial_cash + running_cum_pnl

            if current_equity > peak_equity:
                peak_equity = current_equity

            drawdown = peak_equity - current_equity
            drawdown_pct = drawdown / peak_equity if peak_equity > 0.0 else 0.0

            trade_log.append(
                EquityPoint(
                    timestamp=trade.timestamp,
                    realized_pnl=trade.realized_pnl,
                    cumulative_pnl=running_cum_pnl,
                    cumulative_equity=current_equity,
                    peak_equity=peak_equity,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                    trade_id=trade.trade_id,
                )
            )

        return trade_log