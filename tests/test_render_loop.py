"""
Unit tests for the double-buffered offscreen render loop feature.

Covers:
- Story 1.1.3: Develop Double-Buffered Offscreen Render Loop
  - Acceptance Criteria 1: DoubleBuffer initialization with distinct front/back buffers,
    buffer swapping mechanics, and subsequent drawing target role assignment.
  - Acceptance Criteria 2: RenderLoop target frame rate execution, strict offscreen
    drawing to back buffer prior to buffer exchange.
"""

from typing import List
from unittest.mock import MagicMock, call, patch
import pytest

from src.rendering.double_buffer import DoubleBuffer, FrameBuffer
from src.rendering.render_loop import RenderLoop


# ============================================================================
# Tests: DoubleBuffer (src/rendering/double_buffer.py)
# ============================================================================

class TestDoubleBuffer:
    """Tests for DoubleBuffer initialization, state encapsulation, and swapping."""

    def test_initialization_distinct_buffers(self):
        """
        AC 1: Given an initialized DoubleBuffer instance with distinct front and
        back frame buffers.
        """
        width, height = 800, 600
        db = DoubleBuffer(width=width, height=height)

        assert db.front_buffer is not None
        assert db.back_buffer is not None
        assert db.front_buffer is not db.back_buffer
        assert isinstance(db.front_buffer, FrameBuffer)
        assert isinstance(db.back_buffer, FrameBuffer)
        assert db.width == width
        assert db.height == height
        assert db.front_buffer.width == width
        assert db.front_buffer.height == height
        assert db.back_buffer.width == width
        assert db.back_buffer.height == height

    def test_initialization_with_explicit_frame_buffers(self):
        """Verify initialization with two explicitly supplied distinct FrameBuffer instances."""
        buf1 = FrameBuffer(width=320, height=240)
        buf2 = FrameBuffer(width=320, height=240)

        db = DoubleBuffer(front_buffer=buf1, back_buffer=buf2)

        assert db.front_buffer is buf1
        assert db.back_buffer is buf2
        assert db.front_buffer is not db.back_buffer

    def test_initialization_rejects_aliased_buffers(self):
        """Front and back buffers must never reference the identical instance."""
        shared_buf = FrameBuffer(width=100, height=100)
        with pytest.raises(ValueError):
            DoubleBuffer(front_buffer=shared_buf, back_buffer=shared_buf)

    @pytest.mark.parametrize(
        "width,height",
        [
            (0, 600),
            (-1, 600),
            (800, 0),
            (800, -10),
            (-100, -100),
        ],
    )
    def test_initialization_rejects_invalid_dimensions(self, width: int, height: int):
        """Dimensions must be strictly positive integers."""
        with pytest.raises(ValueError):
            DoubleBuffer(width=width, height=height)

    def test_initialization_rejects_mismatched_explicit_buffers(self):
        """Supplied front and back buffers must share identical dimensions."""
        buf1 = FrameBuffer(width=640, height=480)
        buf2 = FrameBuffer(width=800, height=600)
        with pytest.raises(ValueError):
            DoubleBuffer(front_buffer=buf1, back_buffer=buf2)

    def test_swap_exchanges_front_and_back_references(self):
        """
        AC 1: When a render frame is completed and swapped,
        Then the current back buffer contents become the active front buffer,
        and the old front buffer becomes the target for subsequent offscreen drawing.
        """
        db = DoubleBuffer(width=64, height=64)
        initial_front = db.front_buffer
        initial_back = db.back_buffer

        db.swap()

        assert db.front_buffer is initial_back
        assert db.back_buffer is initial_front

    def test_consecutive_swaps_toggle_consistently(self):
        """Multiple swaps must alternate back and forth deterministically (ping-pong)."""
        db = DoubleBuffer(width=32, height=32)
        buf_a = db.front_buffer
        buf_b = db.back_buffer

        # Cycle 1
        db.swap()
        assert db.front_buffer is buf_b
        assert db.back_buffer is buf_a

        # Cycle 2
        db.swap()
        assert db.front_buffer is buf_a
        assert db.back_buffer is buf_b

        # Cycle 3
        db.swap()
        assert db.front_buffer is buf_b
        assert db.back_buffer is buf_a

    def test_swap_isolates_content_mutations(self):
        """
        Mutations written to the back buffer must not bleed into front buffer
        prior to swap, and must become visible on the front buffer only post-swap.
        """
        db = DoubleBuffer(width=10, height=10)
        
        # Initially both buffers can be distinguished by unique content
        db.front_buffer.clear(color=0x000000)
        db.back_buffer.clear(color=0xFFFFFF)

        # Offscreen back buffer has white; active front buffer remains black
        assert db.front_buffer.get_pixel(0, 0) == 0x000000
        assert db.back_buffer.get_pixel(0, 0) == 0xFFFFFF

        db.swap()

        # After swap, the white buffer is now front; black buffer is back
        assert db.front_buffer.get_pixel(0, 0) == 0xFFFFFF
        assert db.back_buffer.get_pixel(0, 0) == 0x000000

        # Draw to the new back buffer; front buffer must remain unaffected
        db.back_buffer.set_pixel(0, 0, 0x123456)
        assert db.front_buffer.get_pixel(0, 0) == 0xFFFFFF
        assert db.back_buffer.get_pixel(0, 0) == 0x123456


# ============================================================================
# Tests: RenderLoop (src/rendering/render_loop.py)
# ============================================================================

class TestRenderLoop:
    """Tests for RenderLoop execution, frame rate pacing, and tick dispatch sequence."""

    def test_initialization_valid_parameters(self):
        """RenderLoop initializes with DoubleBuffer and positive target FPS."""
        db = DoubleBuffer(width=100, height=100)
        loop = RenderLoop(double_buffer=db, target_fps=60.0)

        assert loop.double_buffer is db
        assert loop.target_fps == 60.0
        assert loop.frame_duration == pytest.approx(1.0 / 60.0)
        assert loop.is_running is False

    @pytest.mark.parametrize("invalid_fps", [0, 0.0, -1, -60.0])
    def test_initialization_rejects_non_positive_fps(self, invalid_fps: float):
        """Target FPS must be strictly positive."""
        db = DoubleBuffer(width=100, height=100)
        with pytest.raises(ValueError):
            RenderLoop(double_buffer=db, target_fps=invalid_fps)

    def test_initialization_rejects_none_buffer(self):
        """Double buffer cannot be None."""
        with pytest.raises(ValueError):
            RenderLoop(double_buffer=None, target_fps=60.0)

    def test_tick_executes_strictly_against_back_buffer_prior_to_swap(self):
        """
        AC 2: Given an active RenderLoop executing at a target frame rate,
        When a frame tick is dispatched,
        Then offscreen draw operations execute strictly against the back buffer
        prior to buffer exchange.
        """
        db = DoubleBuffer(width=64, height=64)
        call_events: List[str] = []
        observed_target_buffers: List[FrameBuffer] = []
        front_buffer_during_draw: List[FrameBuffer] = []
        back_buffer_during_draw: List[FrameBuffer] = []

        initial_front = db.front_buffer
        initial_back = db.back_buffer

        def offscreen_draw(target_buffer: FrameBuffer):
            call_events.append("draw_executed")
            observed_target_buffers.append(target_buffer)
            front_buffer_during_draw.append(db.front_buffer)
            back_buffer_during_draw.append(db.back_buffer)
            # Write unique marker into back buffer
            target_buffer.set_pixel(0, 0, 0xABCDEF)

        loop = RenderLoop(double_buffer=db, target_fps=60.0, render_callback=offscreen_draw)

        # Dispatch single frame tick
        loop.tick()

        # Verify draw happened exactly once
        assert call_events == ["draw_executed"]

        # Verify draw targeted strictly the back buffer prior to swap
        assert observed_target_buffers[0] is initial_back
        assert observed_target_buffers[0] is not initial_front

        # Verify double buffer had NOT swapped yet while drawing was active
        assert front_buffer_during_draw[0] is initial_front
        assert back_buffer_during_draw[0] is initial_back

        # Verify buffer swap completed after offscreen draw finished
        assert db.front_buffer is initial_back
        assert db.back_buffer is initial_front

        # The painted pixel is now exposed on the active front buffer
        assert db.front_buffer.get_pixel(0, 0) == 0xABCDEF

    def test_tick_sequence_order_via_spies(self):
        """
        Verify the atomic execution order:
        1. Render callback invoked with back_buffer.
        2. DoubleBuffer.swap() invoked.
        """
        db = DoubleBuffer(width=32, height=32)
        mock_renderer = MagicMock()
        mock_swap = MagicMock(wraps=db.swap)
        db.swap = mock_swap

        loop = RenderLoop(double_buffer=db, target_fps=30.0, render_callback=mock_renderer)

        expected_target = db.back_buffer
        loop.tick()

        manager = MagicMock()
        manager.attach_mock(mock_renderer, "render")
        manager.attach_mock(mock_swap, "swap")

        # Rerun to record ordered call sequence in manager
        loop.tick()
        assert manager.mock_calls == [
            call.render(db.back_buffer),
            call.swap(),
        ]

    def test_tick_aborts_swap_on_draw_exception(self):
        """
        If the offscreen draw operation raises an exception,
        the buffer exchange must not execute, preventing corrupted state presentation.
        """
        db = DoubleBuffer(width=32, height=32)
        initial_front = db.front_buffer
        initial_back = db.back_buffer

        def crashing_draw(target_buffer: FrameBuffer):
            raise RuntimeError("Drawing pipeline failed")

        loop = RenderLoop(double_buffer=db, target_fps=60.0, render_callback=crashing_draw)

        with pytest.raises(RuntimeError):
            loop.tick()

        # Swap must NOT have occurred
        assert db.front_buffer is initial_front
        assert db.back_buffer is initial_back

    def test_multiple_sequential_ticks_alternate_buffers(self):
        """
        Verify that successive ticks correctly toggle back buffers and
        isolate drawing across frames.
        """
        db = DoubleBuffer(width=16, height=16)
        loop = RenderLoop(double_buffer=db, target_fps=60.0)

        buf_a = db.front_buffer
        buf_b = db.back_buffer

        rendered_buffers: List[FrameBuffer] = []

        def track_draw(target: FrameBuffer):
            rendered_buffers.append(target)
            target.set_pixel(0, 0, len(rendered_buffers))

        loop.render_callback = track_draw

        # Tick 1: draws to buf_b, then swaps (front=buf_b, back=buf_a)
        loop.tick()
        assert rendered_buffers[-1] is buf_b
        assert db.front_buffer is buf_b
        assert db.back_buffer is buf_a
        assert db.front_buffer.get_pixel(0, 0) == 1

        # Tick 2: draws to buf_a, then swaps (front=buf_a, back=buf_b)
        loop.tick()
        assert rendered_buffers[-1] is buf_a
        assert db.front_buffer is buf_a
        assert db.back_buffer is buf_b
        assert db.front_buffer.get_pixel(0, 0) == 2

        # Tick 3: draws to buf_b, then swaps (front=buf_b, back=buf_a)
        loop.tick()
        assert rendered_buffers[-1] is buf_b
        assert db.front_buffer is buf_b
        assert db.back_buffer is buf_a
        assert db.front_buffer.get_pixel(0, 0) == 3

    @patch("time.sleep")
    @patch("time.perf_counter")
    def test_render_loop_frame_pacing_sleeps_remaining_frame_duration(
        self, mock_perf_counter: MagicMock, mock_sleep: MagicMock
    ):
        """
        Given an active RenderLoop at 60 FPS (frame_duration = 0.016666...s),
        When a frame takes 0.005s to draw,
        Then the loop delays by the remaining delta (0.011666...s) to maintain target frame rate.
        """
        target_fps = 60.0
        frame_duration = 1.0 / target_fps  # ~0.016667s

        # Simulate: frame start=1.0, frame draw end=1.005, post-sleep=1.016667
        mock_perf_counter.side_effect = [1.0, 1.005, 1.016667]

        db = DoubleBuffer(width=10, height=10)
        draw_mock = MagicMock()
        loop = RenderLoop(double_buffer=db, target_fps=target_fps, render_callback=draw_mock)

        loop.step_frame()

        draw_mock.assert_called_once_with(db.front_buffer)  # After step_frame finished swap
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg == pytest.approx(frame_duration - 0.005, rel=1e-3)

    @patch("time.sleep")
    @patch("time.perf_counter")
    def test_render_loop_frame_pacing_skips_sleep_when_work_exceeds_budget(
        self, mock_perf_counter: MagicMock, mock_sleep: MagicMock
    ):
        """If drawing exceeds frame time budget, sleep is not called (or called with 0)."""
        target_fps = 60.0
        # Work took 0.020s, exceeding the 0.016667s frame budget
        mock_perf_counter.side_effect = [1.0, 1.020, 1.020]

        db = DoubleBuffer(width=10, height=10)
        loop = RenderLoop(double_buffer=db, target_fps=target_fps, render_callback=lambda b: None)

        loop.step_frame()

        if mock_sleep.called:
            assert mock_sleep.call_args[0][0] <= 0

    def test_lifecycle_start_stop_flags(self):
        """Verify lifecycle controls toggle is_running state."""
        db = DoubleBuffer(width=10, height=10)
        loop = RenderLoop(double_buffer=db, target_fps=30.0)

        assert loop.is_running is False
        loop.start()
        assert loop.is_running is True
        loop.stop()
        assert loop.is_running is False

    def test_run_executes_requested_frame_count(self):
        """Verify run(max_frames=N) executes strictly N ticks and stops."""
        db = DoubleBuffer(width=10, height=10)
        frame_counter = 0

        def count_frame(buf: FrameBuffer):
            nonlocal frame_counter
            frame_counter += 1

        loop = RenderLoop(double_buffer=db, target_fps=100.0, render_callback=count_frame)

        with patch("time.sleep"):
            loop.run(max_frames=5)

        assert frame_counter == 5
        assert loop.is_running is False