"""
Double-buffering implementation for offscreen graphics rendering.

Provides distinct FrameBuffer abstractions and DoubleBuffer ping-pong
mechanisms to isolate draw operations from visible presentation buffers.
"""

from typing import List, Optional


class FrameBuffer:
    """Represents an offscreen 2D pixel buffer."""

    def __init__(self, width: int, height: int) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError(f"Width must be a strictly positive integer, got {width}.")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError(f"Height must be a strictly positive integer, got {height}.")

        self._width = width
        self._height = height
        self._pixels: List[int] = [0] * (width * height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def clear(self, color: int = 0) -> None:
        """Fills the entire buffer with the specified color."""
        self._pixels = [color] * (self._width * self._height)

    def get_pixel(self, x: int, y: int) -> int:
        """Returns the color value of the pixel at coordinate (x, y)."""
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds for buffer ({self._width}x{self._height})."
            )
        return self._pixels[y * self._width + x]

    def set_pixel(self, x: int, y: int, color: int) -> None:
        """Sets the color value of the pixel at coordinate (x, y)."""
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(
                f"Pixel ({x}, {y}) out of bounds for buffer ({self._width}x{self._height})."
            )
        self._pixels[y * self._width + x] = color

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameBuffer):
            return NotImplemented
        return (
            self._width == other._width
            and self._height == other._height
            and self._pixels == other._pixels
        )

    def __repr__(self) -> str:
        return f"FrameBuffer(width={self._width}, height={self._height})"


class DoubleBuffer:
    """
    Encapsulates front and back FrameBuffers, managing ping-pong role assignment
    to isolate offscreen rendering from front presentation state.
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        front_buffer: Optional[FrameBuffer] = None,
        back_buffer: Optional[FrameBuffer] = None,
    ) -> None:
        if isinstance(width, FrameBuffer) and isinstance(height, FrameBuffer):
            front_buffer, back_buffer = width, height
            width, height = None, None

        if front_buffer is not None or back_buffer is not None:
            if front_buffer is None or back_buffer is None:
                raise ValueError("Both front_buffer and back_buffer must be provided.")
            if front_buffer is back_buffer:
                raise ValueError("Front and back buffers must not reference the identical instance.")
            if (
                front_buffer.width != back_buffer.width
                or front_buffer.height != back_buffer.height
            ):
                raise ValueError(
                    f"Buffer dimension mismatch: front is ({front_buffer.width}x{front_buffer.height}), "
                    f"back is ({back_buffer.width}x{back_buffer.height})."
                )
            if width is not None and width != front_buffer.width:
                raise ValueError("Explicit width does not match buffer dimensions.")
            if height is not None and height != front_buffer.height:
                raise ValueError("Explicit height does not match buffer dimensions.")

            self._front_buffer = front_buffer
            self._back_buffer = back_buffer
            self._width = front_buffer.width
            self._height = front_buffer.height
        else:
            if width is None or height is None:
                raise ValueError("Dimensions width and height must be provided.")
            if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
                raise ValueError(f"Width must be a strictly positive integer, got {width}.")
            if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
                raise ValueError(f"Height must be a strictly positive integer, got {height}.")

            self._width = width
            self._height = height
            self._front_buffer = FrameBuffer(width=width, height=height)
            self._back_buffer = FrameBuffer(width=width, height=height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def front_buffer(self) -> FrameBuffer:
        return self._front_buffer

    @property
    def back_buffer(self) -> FrameBuffer:
        return self._back_buffer

    def swap(self) -> None:
        """
        Exchanges front and back buffer references. The former back buffer
        becomes active front buffer; former front becomes back target.
        """
        self._front_buffer, self._back_buffer = self._back_buffer, self._front_buffer