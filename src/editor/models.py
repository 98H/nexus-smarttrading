from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True, order=True)
class Position:
    """Represents a 1-indexed line and column position in a document."""

    line: int
    column: int

    def __post_init__(self) -> None:
        if self.line <= 0 or self.column <= 0:
            raise ValueError(
                f"Position line and column must be >= 1, got line={self.line}, column={self.column}"
            )


@dataclass(frozen=True, order=True)
class Range:
    """Represents a half-open coordinate range [start, end) within a document."""

    start: Position
    end: Position

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"Range start ({self.start}) must not be greater than end ({self.end})"
            )

    @property
    def is_empty(self) -> bool:
        """Return True if the range covers zero characters."""
        return self.start == self.end

    @classmethod
    def from_coords(
        cls,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> Range:
        """Factory helper creating a Range from raw integer coordinates."""
        return cls(
            start=Position(line=start_line, column=start_column),
            end=Position(line=end_line, column=end_column),
        )

    def contains(self, position: Position) -> bool:
        """Check if a position falls within this half-open range [start, end)."""
        return self.start <= position < self.end

    def overlaps(self, other: Range) -> bool:
        """Check if this range overlaps with another half-open range."""
        return max(self.start, other.start) < min(self.end, other.end)


class Severity(str, Enum):
    """Severity levels for emitted diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass(frozen=True)
class TextEdit:
    """Represents a single text replacement over a coordinate range."""

    range: Range
    new_text: str


@dataclass
class QuickFix:
    """Represents an automated fix comprised of one or more text edits."""

    title: str
    edits: list[TextEdit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.edits:
            raise ValueError("QuickFix must contain at least one TextEdit")


@dataclass
class Diagnostic:
    """Diagnostic record detailing an error, warning, or hint in source code."""

    range: Range
    severity: Severity
    message: str
    code: str | None = None
    source: str | None = None
    quick_fixes: list[QuickFix] = field(default_factory=list)


@dataclass
class SourceDocument:
    """Represents an in-memory document with text content and URI metadata."""

    text: str
    uri: str | None = None

    @property
    def line_count(self) -> int:
        """Return the number of lines in the document."""
        return len(self.text.splitlines())

    def get_line(self, line: int) -> str:
        """Return the 1-indexed line content without newline characters."""
        lines = self.text.splitlines()
        if line < 1 or line > len(lines):
            raise IndexError(
                f"Line {line} is out of bounds (document has {len(lines)} lines)"
            )
        return lines[line - 1]