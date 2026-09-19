import dataclasses
from typing import Any
import pytest

from src.buffers.ring_buffer import RingBuffer


@dataclasses.dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    volume: int
    timestamp: int


@pytest.fixture
def sample_ticks() -> list[Tick]:
    return [
        Tick(symbol="AAPL", price=150.0 + i, volume=100 * (i + 1), timestamp=1000 + i)
        for i in range(10)
    ]


class TestRingBufferInitialization:
    def test_init_with_valid_capacity(self):
        capacity = 5
        buffer = RingBuffer(capacity=capacity)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert buffer.is_empty is True
        assert buffer.is_full is False

    @pytest.mark.parametrize("invalid_capacity", [0, -1, -10])
    def test_init_with_non_positive_capacity_raises_error(self, invalid_capacity: int):
        with pytest.raises(ValueError):
            RingBuffer(capacity=invalid_capacity)

    @pytest.mark.parametrize("invalid_type", [3.5, "5", None, []])
    def test_init_with_non_integer_capacity_raises_error(self, invalid_type: Any):
        with pytest.raises(TypeError):
            RingBuffer(capacity=invalid_type)


class TestRingBufferUnderCapacity:
    """AC1: Given a RingBuffer initialized with fixed capacity N,

    When fewer than N ticks are appended,
    Then the buffer stores all ticks in FIFO order and reports the correct item count.
    """

    def test_fewer_than_capacity_stores_fifo_and_reports_count(self, sample_ticks: list[Tick]):
        capacity = 5
        buffer = RingBuffer(capacity=capacity)
        ticks_to_append = sample_ticks[:3]

        for i, tick in enumerate(ticks_to_append, start=1):
            buffer.append(tick)
            assert len(buffer) == i
            assert buffer.is_empty is False
            assert buffer.is_full is False

        assert len(buffer) == 3

        # Verify FIFO order
        for idx, expected_tick in enumerate(ticks_to_append):
            assert buffer[idx] == expected_tick

        assert list(buffer) == ticks_to_append
        assert buffer.to_list() == ticks_to_append

    def test_single_element_fifo(self, sample_ticks: list[Tick]):
        buffer = RingBuffer(capacity=3)
        tick = sample_ticks[0]
        buffer.append(tick)

        assert len(buffer) == 1
        assert buffer[0] == tick
        assert buffer[-1] == tick
        assert buffer.oldest == tick
        assert buffer.newest == tick


class TestRingBufferCapacityAndOverwrite:
    """AC2: Given a RingBuffer filled to capacity N,

    When an additional tick is appended,
    Then the oldest tick is overwritten, the length remains N, and the newest tick
    is retrievable in O(1) time.
    """

    def test_append_beyond_capacity_overwrites_oldest_and_maintains_length(
        self, sample_ticks: list[Tick]
    ):
        capacity = 4
        buffer = RingBuffer(capacity=capacity)

        # Fill to exact capacity
        for tick in sample_ticks[:capacity]:
            buffer.append(tick)

        assert len(buffer) == capacity
        assert buffer.is_full is True
        assert buffer.to_list() == sample_ticks[:capacity]

        # Append one additional tick (overwriting index 0)
        extra_tick = sample_ticks[capacity]
        buffer.append(extra_tick)

        assert len(buffer) == capacity
        assert buffer.is_full is True

        # Oldest tick (sample_ticks[0]) must be gone.
        # Order should now be sample_ticks[1:5]
        expected_ticks = sample_ticks[1 : capacity + 1]
        assert buffer.to_list() == expected_ticks
        assert buffer[0] == expected_ticks[0]
        assert buffer.oldest == expected_ticks[0]

    def test_newest_tick_retrieval(self, sample_ticks: list[Tick]):
        capacity = 3
        buffer = RingBuffer(capacity=capacity)

        for tick in sample_ticks[:5]:
            buffer.append(tick)
            # Newest tick must always be retrievable via property or negative indexing
            assert buffer.newest == tick
            assert buffer[-1] == tick

        assert len(buffer) == capacity

    def test_continuous_wraparound_maintains_correct_window(self, sample_ticks: list[Tick]):
        capacity = 3
        buffer = RingBuffer(capacity=capacity)

        # Append all 10 ticks sequentially into capacity 3 buffer
        for tick in sample_ticks:
            buffer.append(tick)

        assert len(buffer) == capacity
        expected_window = sample_ticks[-capacity:]
        assert buffer.to_list() == expected_window
        assert [buffer[i] for i in range(capacity)] == expected_window
        assert buffer.oldest == sample_ticks[-3]
        assert buffer.newest == sample_ticks[-1]


class TestRingBufferChronologicalRetrieval:
    """AC3: Given a populated RingBuffer,

    When retrieving the contents as an array or slice,
    Then the ticks are returned in chronological order from oldest to newest
    without mutating internal pointers.
    """

    def test_retrieve_contents_chronological_order_without_mutating_pointers(
        self, sample_ticks: list[Tick]
    ):
        capacity = 4
        buffer = RingBuffer(capacity=capacity)

        # Overfill buffer: append 6 items into capacity 4
        for tick in sample_ticks[:6]:
            buffer.append(tick)

        expected_order = sample_ticks[2:6]

        # First retrieval
        first_array = buffer.to_list()
        assert first_array == expected_order

        # Internal state/pointers must not have shifted
        assert len(buffer) == capacity
        assert buffer.oldest == expected_order[0]
        assert buffer.newest == expected_order[-1]

        # Second retrieval must produce identical state and contents
        second_array = buffer.to_list()
        assert second_array == expected_order
        assert buffer[0] == expected_order[0]
        assert buffer[-1] == expected_order[-1]

    def test_slice_retrieval_preserves_buffer_state(self, sample_ticks: list[Tick]):
        capacity = 5
        buffer = RingBuffer(capacity=capacity)

        for tick in sample_ticks[:8]:  # Causes wraparound
            buffer.append(tick)

        expected = sample_ticks[3:8]
        assert buffer[:] == expected
        assert buffer[1:4] == expected[1:4]
        assert buffer[::-1] == expected[::-1]

        # Length and boundaries remain unaffected after slicing
        assert len(buffer) == capacity
        assert buffer.oldest == expected[0]
        assert buffer.newest == expected[-1]

    def test_iteration_order_and_idempotence(self, sample_ticks: list[Tick]):
        capacity = 3
        buffer = RingBuffer(capacity=capacity)

        for tick in sample_ticks[:5]:
            buffer.append(tick)

        expected = sample_ticks[2:5]

        # Iterating once
        iterated_1 = [item for item in buffer]
        assert iterated_1 == expected

        # Iterating again should produce same items without consuming or altering pointers
        iterated_2 = [item for item in buffer]
        assert iterated_2 == expected
        assert len(buffer) == capacity


class TestRingBufferEdgeCasesAndExceptions:
    def test_accessing_empty_buffer_raises_index_error(self):
        buffer = RingBuffer(capacity=3)

        assert len(buffer) == 0

        with pytest.raises(IndexError):
            _ = buffer[0]

        with pytest.raises(IndexError):
            _ = buffer[-1]

        with pytest.raises(IndexError):
            _ = buffer.oldest

        with pytest.raises(IndexError):
            _ = buffer.newest

    def test_out_of_bounds_indexing_raises_index_error(self, sample_ticks: list[Tick]):
        buffer = RingBuffer(capacity=5)
        for tick in sample_ticks[:3]:
            buffer.append(tick)

        with pytest.raises(IndexError):
            _ = buffer[3]

        with pytest.raises(IndexError):
            _ = buffer[10]

        with pytest.raises(IndexError):
            _ = buffer[-4]

    def test_negative_indexing_resolution(self, sample_ticks: list[Tick]):
        buffer = RingBuffer(capacity=3)
        for tick in sample_ticks[:5]:  # Window will be sample_ticks[2:5]
            buffer.append(tick)

        assert buffer[-1] == sample_ticks[4]
        assert buffer[-2] == sample_ticks[3]
        assert buffer[-3] == sample_ticks[2]

    def test_clear_resets_buffer_state(self, sample_ticks: list[Tick]):
        buffer = RingBuffer(capacity=3)
        for tick in sample_ticks[:3]:
            buffer.append(tick)

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.is_empty is True
        assert buffer.is_full is False
        assert buffer.to_list() == []

        with pytest.raises(IndexError):
            _ = buffer[0]

        # Re-populating after clear should work seamlessly
        buffer.append(sample_ticks[0])
        assert len(buffer) == 1
        assert buffer[0] == sample_ticks[0]