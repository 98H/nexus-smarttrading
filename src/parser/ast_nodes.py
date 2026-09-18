"""Abstract Syntax Tree (AST) node definitions and source location primitives."""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class SourceLocation:
    """Represents a 1-based (line, column) coordinate in source code."""

    line: int
    column: int

    def __post_init__(self) -> None:
        if self.line <= 0:
            raise ValueError(f"Line must be greater than zero, got {self.line}")
        if self.column <= 0:
            raise ValueError(f"Column must be greater than zero, got {self.column}")


@dataclass(frozen=True)
class SourceSpan:
    """Represents a contiguous range in source code between start and end locations."""

    start: SourceLocation
    end: SourceLocation

    def __post_init__(self) -> None:
        if self.end.line < self.start.line or (
            self.end.line == self.start.line and self.end.column < self.start.column
        ):
            raise ValueError(
                f"End location ({self.end}) cannot precede start location ({self.start})"
            )


class ASTNode:
    """Base class for all Abstract Syntax Tree nodes."""

    span: SourceSpan

    def __init__(self, span: SourceSpan) -> None:
        self.span = span

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(span={self.span!r})"

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__


@dataclass
class LiteralExpr(ASTNode):
    """Represents a literal constant expression (integer, string, boolean, etc.)."""

    value: Any
    span: SourceSpan
    raw: Optional[str] = None


@dataclass
class IdentifierExpr(ASTNode):
    """Represents a variable or symbol reference expression."""

    name: str
    span: SourceSpan


@dataclass
class BinaryExpr(ASTNode):
    """Represents a binary operator expression (e.g., left + right)."""

    left: ASTNode
    operator: str
    right: ASTNode
    span: SourceSpan


@dataclass
class UnaryExpr(ASTNode):
    """Represents a prefix unary operator expression (e.g., -x, !isValid)."""

    operator: str
    operand: ASTNode
    span: SourceSpan


@dataclass
class CallExpr(ASTNode):
    """Represents a function or method invocation expression."""

    callee: ASTNode
    arguments: List[ASTNode]
    span: SourceSpan


__all__ = [
    "SourceLocation",
    "SourceSpan",
    "ASTNode",
    "LiteralExpr",
    "IdentifierExpr",
    "BinaryExpr",
    "UnaryExpr",
    "CallExpr",
]