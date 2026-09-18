"""Signal confirmation filter against adaptive trend states."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.indicators.adaptive_trend import TrendDirection


class SignalDirection(str, Enum):
    """Direction of an entry signal."""

    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(str, Enum):
    """Approval status of a validated signal."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class RawSignal:
    """Incoming raw trade signal prior to trend confirmation."""

    signal_id: str
    direction: SignalDirection
    timestamp: datetime
    price: float


@dataclass
class ConfirmedSignal:
    """Confirmed signal state with approval status and context."""

    raw_signal: RawSignal
    status: SignalStatus
    trend_direction: TrendDirection
    reason: Optional[str] = None


class SignalConfirmationFilter:
    """Filters entry signals against the prevailing adaptive trend direction."""

    def confirm(
        self,
        signal: RawSignal,
        trend_direction: TrendDirection,
    ) -> ConfirmedSignal:
        """Validate an entry signal against the adaptive trend state.

        Args:
            signal: The incoming raw trade signal.
            trend_direction: The current directional state of the adaptive trend filter.

        Returns:
            ConfirmedSignal containing approval or rejection status and reason.

        Raises:
            ValueError: If signal or trend_direction is None or invalid type.
        """
        if signal is None or not isinstance(signal, RawSignal):
            raise ValueError("signal must be a valid RawSignal instance.")
        if trend_direction is None or not isinstance(trend_direction, TrendDirection):
            raise ValueError("trend_direction must be a valid TrendDirection instance.")

        if trend_direction == TrendDirection.FLAT:
            return ConfirmedSignal(
                raw_signal=signal,
                status=SignalStatus.REJECTED,
                trend_direction=trend_direction,
                reason=f"Signal {signal.signal_id} rejected: trend direction is FLAT.",
            )

        if signal.direction == SignalDirection.LONG:
            if trend_direction == TrendDirection.UPWARD:
                return ConfirmedSignal(
                    raw_signal=signal,
                    status=SignalStatus.APPROVED,
                    trend_direction=trend_direction,
                    reason=None,
                )
            return ConfirmedSignal(
                raw_signal=signal,
                status=SignalStatus.REJECTED,
                trend_direction=trend_direction,
                reason=(
                    f"Long signal {signal.signal_id} rejected: "
                    f"conflicts with {trend_direction.value} trend."
                ),
            )

        if signal.direction == SignalDirection.SHORT:
            if trend_direction == TrendDirection.DOWNWARD:
                return ConfirmedSignal(
                    raw_signal=signal,
                    status=SignalStatus.APPROVED,
                    trend_direction=trend_direction,
                    reason=None,
                )
            return ConfirmedSignal(
                raw_signal=signal,
                status=SignalStatus.REJECTED,
                trend_direction=trend_direction,
                reason=(
                    f"Short signal {signal.signal_id} rejected: "
                    f"conflicts with {trend_direction.value} trend."
                ),
            )

        return ConfirmedSignal(
            raw_signal=signal,
            status=SignalStatus.REJECTED,
            trend_direction=trend_direction,
            reason=f"Signal {signal.signal_id} has unrecognized direction: {signal.direction}.",
        )

    def confirm_batch(
        self,
        signals: list[RawSignal],
        trend_directions: list[TrendDirection],
    ) -> list[ConfirmedSignal]:
        """Validate a batch of entry signals against corresponding trend directions.

        Args:
            signals: Sequence of incoming raw trade signals.
            trend_directions: Sequence of corresponding trend directions.

        Returns:
            List of ConfirmedSignal objects.

        Raises:
            ValueError: If arguments are None or have mismatched lengths.
        """
        if signals is None or trend_directions is None:
            raise ValueError("signals and trend_directions cannot be None.")
        if len(signals) != len(trend_directions):
            raise ValueError("signals and trend_directions must have identical lengths.")
        return [self.confirm(sig, td) for sig, td in zip(signals, trend_directions)]