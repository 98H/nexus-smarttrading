"""
Offscreen render loop execution and frame pacing manager.

Orchestrates atomic frame rendering targeting the back buffer of a
DoubleBuffer instance followed by buffer exchange, enforcing target FPS pacing.
"""

import time
from typing import Callable, Optional

from src.rendering.double_buffer import DoubleBuffer, FrameBuffer


class RenderLoop:
    """Manages render frame dispatch, target frame rate pacing, and lifecycle."""

    def __init__(
        self,
        double_buffer: DoubleBuffer,
        target_fps: float,
        render_callback: Optional[Callable[[FrameBuffer], None]] = None,
    ) -> None:
        if double_buffer is None or not isinstance(double_buffer, DoubleBuffer):
            raise ValueError("double_buffer must be a valid DoubleBuffer instance.")
        if (
            not isinstance(target_fps, (int, float))
            or isinstance(target_fps, bool)
            or target_fps <= 0
        ):
            raise ValueError(f"target_fps must be a strictly positive number, got {target_fps}.")

        self._double_buffer = double_buffer
        self._target_fps = float(target_fps)
        self.render_callback = render_callback
        self._is_running = False

    @property
    def double_buffer(self) -> DoubleBuffer:
        return self._double_buffer

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @target_fps.setter
    def target_fps(self, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"target_fps must be a strictly positive number, got {value}.")
        self._target_fps = float(value)

    @property
    def frame_duration(self) -> float:
        return 1.0 / self._target_fps

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Activates the running state flag."""
        self._is_running = True

    def stop(self) -> None:
        """Deactivates the running state flag."""
        self._is_running = False

    def tick(self) -> None:
        """
        Dispatches offscreen rendering strictly to the back buffer prior
        to exchanging buffers.
        """
        if self.render_callback is not None:
            self.render_callback(self._double_buffer.back_buffer)
        self._double_buffer.swap()

    def step_frame(self) -> None:
        """
        Executes a single frame tick and sleeps for any remaining budget
        to pace execution to the target frame rate.
        """
        start_time = time.perf_counter()
        self.tick()
        elapsed = time.perf_counter() - start_time
        remaining = self.frame_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def run(self, max_frames: Optional[int] = None) -> None:
        """
        Runs the render loop continuously or until max_frames ticks have completed.
        """
        self.start()
        frames_rendered = 0
        try:
            while self._is_running:
                if max_frames is not None and frames_rendered >= max_frames:
                    break
                self.step_frame()
                frames_rendered += 1
        finally:
            self.stop()