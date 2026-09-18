from datetime import datetime, timedelta, timezone
from typing import List
import pytest

from src.market_data.bar_corrector import BarCorrectionEngine
from src.market_data.models import OHLCVBar, Tick


@pytest.fixture
def bar_timeframe() -> timedelta:
    """Standard timeframe duration for OHLCV bars (1 minute)."""
    return timedelta(minutes=1)


@pytest.fixture
def base_bucket_time() -> datetime:
    """Fixed base bucket timestamp aligned to minute boundary."""
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def watermark_time(base_bucket_time: datetime, bar_timeframe: timedelta) -> datetime:
    """Watermark strictly greater than the base bucket (T_watermark > T_bucket)."""
    return base_bucket_time + (bar_timeframe * 5)


@pytest.fixture
def sample_finalized_bar(base_bucket_time: datetime) -> OHLCVBar:
    """
    Standard finalized OHLCV bar:
    T_bucket = 10:00:00 UTC
    Open: 100.0, High: 105.0, Low: 95.0, Close: 102.0, Volume: 1000.0
    Finalized: True, Corrected: False
    """
    return OHLCVBar(
        symbol="AAPL",
        timestamp=base_bucket_time,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_finalized=True,
        is_corrected=False,
    )


@pytest.fixture
def engine(bar_timeframe: timedelta) -> BarCorrectionEngine:
    """Fresh instance of BarCorrectionEngine."""
    return BarCorrectionEngine(timeframe=bar_timeframe)


def test_late_tick_updates_high_and_marks_corrected(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick with price > bar.high updates high and volume,
    sets is_corrected to True, and emits the corrected bar.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    # Late tick arriving inside [T_bucket, T_bucket + Delta t) with higher price
    late_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=25),
        price=110.0,
        volume=150.0,
    )

    emitted_bar = engine.process_tick(late_tick)

    assert emitted_bar is not None
    assert emitted_bar.symbol == "AAPL"
    assert emitted_bar.timestamp == base_bucket_time
    assert emitted_bar.high == 110.0
    assert emitted_bar.low == 95.0
    assert emitted_bar.volume == 1150.0
    assert emitted_bar.is_corrected is True

    # Ensure internal state is also updated
    stored_bar = engine.get_bar(symbol="AAPL", timestamp=base_bucket_time)
    assert stored_bar == emitted_bar


def test_late_tick_updates_low_and_marks_corrected(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick with price < bar.low updates low and volume,
    sets is_corrected to True, and emits the corrected bar.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    late_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=30),
        price=90.0,
        volume=200.0,
    )

    emitted_bar = engine.process_tick(late_tick)

    assert emitted_bar is not None
    assert emitted_bar.low == 90.0
    assert emitted_bar.high == 105.0
    assert emitted_bar.volume == 1200.0
    assert emitted_bar.is_corrected is True


def test_late_tick_updates_close_when_tick_is_latest_in_bucket(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick occurring at the end of the bucket interval updates Close.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    # Late tick near the bucket end boundary
    late_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=58),
        price=108.0,
        volume=50.0,
    )

    emitted_bar = engine.process_tick(late_tick)

    assert emitted_bar is not None
    assert emitted_bar.close == 108.0
    assert emitted_bar.high == 108.0
    assert emitted_bar.volume == 1050.0
    assert emitted_bar.is_corrected is True


def test_late_tick_preserves_close_when_tick_is_not_latest(
    engine: BarCorrectionEngine,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    When the bucket has historical constituent ticks, a late tick inserted
    prior to the bucket's closing tick should NOT overwrite the Close price.
    """
    initial_bar = OHLCVBar(
        symbol="AAPL",
        timestamp=base_bucket_time,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_finalized=True,
        is_corrected=False,
    )
    engine.add_bar(initial_bar)
    engine.set_watermark(watermark_time)

    # Seed closing tick timestamp at 10:00:55
    closing_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=55),
        price=102.0,
        volume=100.0,
    )
    # Late tick inserted earlier in the bucket at 10:00:15 with non-extreme price
    mid_bucket_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=15),
        price=99.0,
        volume=50.0,
    )

    engine.process_tick(closing_tick)
    corrected_bar = engine.process_tick(mid_bucket_tick)

    assert corrected_bar is not None
    # Close must remain 102.0 because mid_bucket_tick is not the latest in the bucket
    assert corrected_bar.close == 102.0
    assert corrected_bar.volume == 1150.0
    assert corrected_bar.is_corrected is True


def test_boundary_tick_at_exact_bucket_start(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test lower boundary condition: t = T_bucket is inclusive.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    boundary_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time,  # Exactly at T_bucket
        price=106.0,
        volume=10.0,
    )

    emitted_bar = engine.process_tick(boundary_tick)

    assert emitted_bar is not None
    assert emitted_bar.timestamp == base_bucket_time
    assert emitted_bar.high == 106.0
    assert emitted_bar.volume == 1010.0
    assert emitted_bar.is_corrected is True


def test_boundary_tick_just_before_bucket_end(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    bar_timeframe: timedelta,
    watermark_time: datetime,
) -> None:
    """
    Test upper boundary condition: t = T_bucket + Delta t - epsilon belongs to T_bucket.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    just_before_end = base_bucket_time + bar_timeframe - timedelta(microseconds=1)
    boundary_tick = Tick(
        symbol="AAPL",
        timestamp=just_before_end,
        price=94.0,
        volume=25.0,
    )

    emitted_bar = engine.process_tick(boundary_tick)

    assert emitted_bar is not None
    assert emitted_bar.timestamp == base_bucket_time
    assert emitted_bar.low == 94.0
    assert emitted_bar.volume == 1025.0
    assert emitted_bar.is_corrected is True


def test_boundary_tick_at_exact_bucket_end_routes_to_next_bucket(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    bar_timeframe: timedelta,
    watermark_time: datetime,
) -> None:
    """
    Test upper boundary condition: t = T_bucket + Delta t is exclusive for T_bucket
    and belongs to the subsequent bucket.
    """
    engine.add_bar(sample_finalized_bar)
    next_bucket_time = base_bucket_time + bar_timeframe
    next_bar = OHLCVBar(
        symbol="AAPL",
        timestamp=next_bucket_time,
        open=102.0,
        high=103.0,
        low=101.0,
        close=102.5,
        volume=500.0,
        is_finalized=True,
        is_corrected=False,
    )
    engine.add_bar(next_bar)
    engine.set_watermark(watermark_time)

    tick_at_cutoff = Tick(
        symbol="AAPL",
        timestamp=next_bucket_time,  # Exactly T_bucket + Delta t
        price=104.0,
        volume=50.0,
    )

    emitted_bar = engine.process_tick(tick_at_cutoff)

    assert emitted_bar is not None
    assert emitted_bar.timestamp == next_bucket_time
    assert emitted_bar.high == 104.0

    # Ensure original base bucket was untouched
    original_bucket = engine.get_bar(symbol="AAPL", timestamp=base_bucket_time)
    assert original_bucket is not None
    assert original_bucket.is_corrected is False
    assert original_bucket.volume == 1000.0


def test_sequential_late_ticks_accumulate_corrections(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test multiple late-arriving ticks sequentially updating the same historical bucket.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    tick_1 = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=10),
        price=112.0,
        volume=100.0,
    )
    tick_2 = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=20),
        price=91.0,
        volume=200.0,
    )

    first_correction = engine.process_tick(tick_1)
    assert first_correction is not None
    assert first_correction.high == 112.0
    assert first_correction.volume == 1100.0
    assert first_correction.is_corrected is True

    second_correction = engine.process_tick(tick_2)
    assert second_correction is not None
    assert second_correction.high == 112.0
    assert second_correction.low == 91.0
    assert second_correction.volume == 1300.0
    assert second_correction.is_corrected is True


def test_on_time_tick_after_watermark_does_not_trigger_late_correction(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    watermark_time: datetime,
) -> None:
    """
    Ticks arriving at or after the watermark (t >= T_watermark) are current/on-time ticks,
    not late-arriving corrections to finalized historical bars.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    current_tick = Tick(
        symbol="AAPL",
        timestamp=watermark_time + timedelta(seconds=5),
        price=120.0,
        volume=100.0,
    )

    emitted_bar = engine.process_tick(current_tick)

    # Current tick should not emit a corrected finalized bar
    if emitted_bar is not None:
        assert emitted_bar.is_corrected is False

    stored_bar = engine.get_bar(symbol="AAPL", timestamp=sample_finalized_bar.timestamp)
    assert stored_bar is not None
    assert stored_bar.is_corrected is False


def test_multiple_symbols_isolation(
    engine: BarCorrectionEngine,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Late ticks for one symbol must only correct that symbol's bucket and leave other symbols unaffected.
    """
    aapl_bar = OHLCVBar(
        symbol="AAPL",
        timestamp=base_bucket_time,
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=1000.0,
        is_finalized=True,
        is_corrected=False,
    )
    msft_bar = OHLCVBar(
        symbol="MSFT",
        timestamp=base_bucket_time,
        open=300.0,
        high=305.0,
        low=298.0,
        close=302.0,
        volume=2000.0,
        is_finalized=True,
        is_corrected=False,
    )
    engine.add_bar(aapl_bar)
    engine.add_bar(msft_bar)
    engine.set_watermark(watermark_time)

    msft_late_tick = Tick(
        symbol="MSFT",
        timestamp=base_bucket_time + timedelta(seconds=40),
        price=310.0,
        volume=500.0,
    )

    emitted_bar = engine.process_tick(msft_late_tick)

    assert emitted_bar is not None
    assert emitted_bar.symbol == "MSFT"
    assert emitted_bar.high == 310.0
    assert emitted_bar.volume == 2500.0
    assert emitted_bar.is_corrected is True

    # AAPL bar must remain completely unchanged
    stored_aapl = engine.get_bar(symbol="AAPL", timestamp=base_bucket_time)
    assert stored_aapl is not None
    assert stored_aapl.is_corrected is False
    assert stored_aapl.high == 155.0
    assert stored_aapl.volume == 1000.0


def test_listener_callback_emission(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that registering a callback listener receives the emitted corrected bar.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    emitted_bars: List[OHLCVBar] = []
    engine.register_listener(lambda bar: emitted_bars.append(bar))

    late_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=15),
        price=107.0,
        volume=50.0,
    )

    engine.process_tick(late_tick)

    assert len(emitted_bars) == 1
    assert emitted_bars[0].is_corrected is True
    assert emitted_bars[0].high == 107.0
    assert emitted_bars[0].volume == 1050.0


def test_missing_bucket_raises_key_error(
    engine: BarCorrectionEngine,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick attempting to correct a non-existent historical bucket raises KeyError.
    """
    engine.set_watermark(watermark_time)

    tick_for_missing_bucket = Tick(
        symbol="UNKNOWN",
        timestamp=base_bucket_time + timedelta(seconds=10),
        price=100.0,
        volume=10.0,
    )

    with pytest.raises(KeyError):
        engine.process_tick(tick_for_missing_bucket)


def test_invalid_tick_negative_price_raises_value_error(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick with invalid non-positive price raises ValueError.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    invalid_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=10),
        price=-1.0,
        volume=100.0,
    )

    with pytest.raises(ValueError):
        engine.process_tick(invalid_tick)


def test_invalid_tick_negative_volume_raises_value_error(
    engine: BarCorrectionEngine,
    sample_finalized_bar: OHLCVBar,
    base_bucket_time: datetime,
    watermark_time: datetime,
) -> None:
    """
    Test that a late tick with negative volume raises ValueError.
    """
    engine.add_bar(sample_finalized_bar)
    engine.set_watermark(watermark_time)

    invalid_tick = Tick(
        symbol="AAPL",
        timestamp=base_bucket_time + timedelta(seconds=10),
        price=100.0,
        volume=-50.0,
    )

    with pytest.raises(ValueError):
        engine.process_tick(invalid_tick)