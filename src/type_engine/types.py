"""
Core type definitions, AST node abstractions, and symbol table for static type engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Type(Enum):
    """Primitive type enumeration supported by the type engine."""

    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"


class BinaryOperator(Enum):
    """Binary operators supported in expressions."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    AND = "and"
    OR = "or"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class UnaryOperator(Enum):
    """Unary operators supported in expressions."""

    NOT = "not"
    NEG = "-"
    POS = "+"


class ASTNode:
    """Base class for all Abstract Syntax Tree (AST) nodes."""

    pass


@dataclass
class Literal(ASTNode):
    """AST node representing a raw literal value and its explicit type."""

    value: Any
    inferred_type: Type


@dataclass
class Identifier(ASTNode):
    """AST node representing a named symbol/variable reference."""

    name: str


@dataclass
class BinaryOp(ASTNode):
    """AST node representing a binary operation composed of two operands."""

    left: ASTNode
    operator: BinaryOperator
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    """AST node representing a unary operation applied to a single operand."""

    operator: UnaryOperator
    operand: ASTNode


class UndefinedSymbolError(Exception):
    """Raised when an identifier is referenced but not registered in the SymbolTable."""

    pass


class SymbolTable:
    """Stores identifier-to-type bindings for static analysis and scope resolution."""

    def __init__(self, symbols: Optional[Dict[str, Type]] = None) -> None:
        self._symbols: Dict[str, Type] = dict(symbols) if symbols is not None else {}

    def define(self, name: str, symbol_type: Type) -> None:
        """Register or update an identifier's static type binding."""
        self._symbols[name] = symbol_type

    def lookup(self, name: str) -> Type:
        """Resolve the static type bound to the given identifier name."""
        if name not in self._symbols:
            raise UndefinedSymbolError(f"Undefined symbol: '{name}'")
        return self._symbols[name]

    def __contains__(self, name: str) -> bool:
        return name in self._symbols


__all__ = [
    "Type",
    "BinaryOperator",
    "UnaryOperator",
    "ASTNode",
    "Literal",
    "Identifier",
    "BinaryOp",
    "UnaryOp",
    "UndefinedSymbolError",
    "SymbolTable",
]