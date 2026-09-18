"""3x3 Matrix mathematics module for 2D affine transformations."""

from __future__ import annotations

import math
from typing import Sequence


class Matrix3x3:
    """Immutable 3x3 affine transformation matrix."""

    __slots__ = ("_data",)

    def __init__(self, data: Sequence[Sequence[float]] | None = None) -> None:
        """Initialize the matrix with 3x3 values or default to the identity matrix."""
        if data is None:
            self._data: tuple[tuple[float, float, float], ...] = (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        elif isinstance(data, Matrix3x3):
            self._data = data._data
        else:
            rows = tuple(tuple(float(val) for val in row) for row in data)
            if len(rows) != 3 or any(len(row) != 3 for row in rows):
                raise ValueError("Matrix3x3 data must be a 3x3 sequence of numbers.")
            self._data = rows

    @classmethod
    def identity(cls) -> Matrix3x3:
        """Create a 3x3 identity matrix."""
        return cls((
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ))

    @classmethod
    def translation(cls, tx: float, ty: float) -> Matrix3x3:
        """Create a 3x3 translation matrix."""
        return cls((
            (1.0, 0.0, float(tx)),
            (0.0, 1.0, float(ty)),
            (0.0, 0.0, 1.0),
        ))

    @classmethod
    def scale(cls, sx: float, sy: float) -> Matrix3x3:
        """Create a 3x3 scale matrix."""
        return cls((
            (float(sx), 0.0, 0.0),
            (0.0, float(sy), 0.0),
            (0.0, 0.0, 1.0),
        ))

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform a 2D point (x, y) through the 3x3 matrix."""
        m = self._data
        px = float(x)
        py = float(y)
        out_x = m[0][0] * px + m[0][1] * py + m[0][2]
        out_y = m[1][0] * px + m[1][1] * py + m[1][2]
        out_w = m[2][0] * px + m[2][1] * py + m[2][2]

        if out_w != 1.0 and abs(out_w) > 1e-15:
            return (out_x / out_w, out_y / out_w)
        return (out_x, out_y)

    def determinant(self) -> float:
        """Calculate the determinant of the 3x3 matrix."""
        m = self._data
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    def inverse(self) -> Matrix3x3:
        """Calculate the inverse of this matrix.

        Raises:
            ValueError: If the matrix is singular and cannot be inverted.
        """
        m = self._data
        c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
        c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0])
        c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0]

        det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02

        if math.isclose(det, 0.0, abs_tol=1e-12):
            raise ValueError("Matrix is singular and cannot be inverted.")

        inv_det = 1.0 / det

        c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1])
        c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0]
        c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0])

        c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1]
        c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0])
        c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0]

        return Matrix3x3((
            (c00 * inv_det, c10 * inv_det, c20 * inv_det),
            (c01 * inv_det, c11 * inv_det, c21 * inv_det),
            (c02 * inv_det, c12 * inv_det, c22 * inv_det),
        ))

    def __getitem__(self, key: tuple[int, int] | int) -> float | tuple[float, float, float]:
        """Access matrix element by (row, col) or entire row by index."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Matrix index tuple must be (row, col)")
            row, col = key
            return self._data[row][col]
        if isinstance(key, int):
            return self._data[key]
        raise TypeError(f"Invalid index type: {type(key)}")

    def __matmul__(self, other: Matrix3x3) -> Matrix3x3:
        """Matrix multiplication using the @ operator."""
        if not isinstance(other, Matrix3x3):
            return NotImplemented
        a = self._data
        b = other._data
        return Matrix3x3((
            (
                a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
                a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
            ),
            (
                a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
                a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
            ),
            (
                a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
                a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
                a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
            ),
        ))

    def __eq__(self, other: object) -> bool:
        """Check equality with another Matrix3x3."""
        if not isinstance(other, Matrix3x3):
            return False
        return self._data == other._data

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"Matrix3x3([\n"
            f"  [{self._data[0][0]}, {self._data[0][1]}, {self._data[0][2]}],\n"
            f"  [{self._data[1][0]}, {self._data[1][1]}, {self._data[1][2]}],\n"
            f"  [{self._data[2][0]}, {self._data[2][1]}, {self._data[2][2]}]\n"
            f"])"
        )