"""
Static type inference engine, validation logic, and type error definitions.
"""

from typing import Optional

from src.type_engine.types import (
    ASTNode,
    BinaryOp,
    BinaryOperator,
    Identifier,
    Literal,
    SymbolTable,
    Type,
    UnaryOp,
    UnaryOperator,
    UndefinedSymbolError,
)

NUMERIC_TYPES = {Type.INTEGER, Type.FLOAT}


class TypeMismatchError(Exception):
    """Raised when operand types do not conform to operator type rules."""

    def __init__(
        self,
        expected_type: Type,
        actual_type: Type,
        message: Optional[str] = None,
    ) -> None:
        self.expected_type = expected_type
        self.actual_type = actual_type
        if message is None:
            message = f"Type mismatch: expected {expected_type}, got {actual_type}"
        super().__init__(message)


class TypeInferenceEngine:
    """Statically infers expression target types without evaluating runtime values."""

    def infer(
        self,
        node: ASTNode,
        symbol_table: Optional[SymbolTable] = None,
    ) -> Type:
        """Infer the static result type of an expression node."""
        if isinstance(node, Literal):
            return node.inferred_type

        if isinstance(node, Identifier):
            if symbol_table is None:
                raise UndefinedSymbolError(
                    f"Symbol '{node.name}' cannot be resolved: no symbol table provided"
                )
            return symbol_table.lookup(node.name)

        if isinstance(node, UnaryOp):
            return self._infer_unary(node, symbol_table)

        if isinstance(node, BinaryOp):
            return self._infer_binary(node, symbol_table)

        raise TypeError(f"Unsupported AST node type: {type(node).__name__}")

    def _infer_unary(
        self,
        node: UnaryOp,
        symbol_table: Optional[SymbolTable],
    ) -> Type:
        operand_type = self.infer(node.operand, symbol_table)

        if node.operator == UnaryOperator.NOT:
            if operand_type != Type.BOOLEAN:
                raise TypeMismatchError(
                    expected_type=Type.BOOLEAN,
                    actual_type=operand_type,
                )
            return Type.BOOLEAN

        if node.operator in (UnaryOperator.NEG, UnaryOperator.POS):
            if operand_type not in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=Type.INTEGER,
                    actual_type=operand_type,
                )
            return operand_type

        raise ValueError(f"Unsupported unary operator: {node.operator}")

    def _infer_binary(
        self,
        node: BinaryOp,
        symbol_table: Optional[SymbolTable],
    ) -> Type:
        left_type = self.infer(node.left, symbol_table)
        right_type = self.infer(node.right, symbol_table)

        # Addition: String concatenation or numeric addition
        if node.operator == BinaryOperator.ADD:
            if left_type == Type.STRING or right_type == Type.STRING:
                if left_type != Type.STRING:
                    raise TypeMismatchError(
                        expected_type=Type.STRING,
                        actual_type=left_type,
                    )
                if right_type != Type.STRING:
                    raise TypeMismatchError(
                        expected_type=Type.STRING,
                        actual_type=right_type,
                    )
                return Type.STRING

            if left_type in NUMERIC_TYPES and right_type in NUMERIC_TYPES:
                if left_type == Type.FLOAT or right_type == Type.FLOAT:
                    return Type.FLOAT
                return Type.INTEGER

            if left_type not in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=Type.INTEGER,
                    actual_type=left_type,
                )
            raise TypeMismatchError(
                expected_type=Type.INTEGER,
                actual_type=right_type,
            )

        # Subtraction, Multiplication, Modulo: Strict numeric operations
        if node.operator in (BinaryOperator.SUB, BinaryOperator.MUL, BinaryOperator.MOD):
            if left_type not in NUMERIC_TYPES:
                expected = right_type if right_type in NUMERIC_TYPES else Type.INTEGER
                raise TypeMismatchError(
                    expected_type=expected,
                    actual_type=left_type,
                )
            if right_type not in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=left_type,
                    actual_type=right_type,
                )
            if left_type == Type.FLOAT or right_type == Type.FLOAT:
                return Type.FLOAT
            return Type.INTEGER

        # Division: Numeric division statically yields FLOAT without runtime division
        if node.operator == BinaryOperator.DIV:
            if left_type not in NUMERIC_TYPES:
                expected = right_type if right_type in NUMERIC_TYPES else Type.FLOAT
                raise TypeMismatchError(
                    expected_type=expected,
                    actual_type=left_type,
                )
            if right_type not in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=left_type,
                    actual_type=right_type,
                )
            return Type.FLOAT

        # Logical operations: operands must be BOOLEAN
        if node.operator in (BinaryOperator.AND, BinaryOperator.OR):
            if left_type != Type.BOOLEAN:
                raise TypeMismatchError(
                    expected_type=Type.BOOLEAN,
                    actual_type=left_type,
                )
            if right_type != Type.BOOLEAN:
                raise TypeMismatchError(
                    expected_type=Type.BOOLEAN,
                    actual_type=right_type,
                )
            return Type.BOOLEAN

        # Relational comparisons (<, <=, >, >=)
        if node.operator in (
            BinaryOperator.LT,
            BinaryOperator.GT,
            BinaryOperator.LE,
            BinaryOperator.GE,
        ):
            if left_type in NUMERIC_TYPES and right_type in NUMERIC_TYPES:
                return Type.BOOLEAN
            if left_type == Type.STRING and right_type == Type.STRING:
                return Type.BOOLEAN
            if right_type in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=right_type,
                    actual_type=left_type,
                )
            if left_type in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=left_type,
                    actual_type=right_type,
                )
            if right_type == Type.STRING:
                raise TypeMismatchError(
                    expected_type=Type.STRING,
                    actual_type=left_type,
                )
            raise TypeMismatchError(
                expected_type=left_type,
                actual_type=right_type,
            )

        # Equality comparisons (==, !=)
        if node.operator in (BinaryOperator.EQ, BinaryOperator.NE):
            if left_type in NUMERIC_TYPES and right_type in NUMERIC_TYPES:
                return Type.BOOLEAN
            if left_type == right_type:
                return Type.BOOLEAN
            if right_type in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=right_type,
                    actual_type=left_type,
                )
            if left_type in NUMERIC_TYPES:
                raise TypeMismatchError(
                    expected_type=left_type,
                    actual_type=right_type,
                )
            raise TypeMismatchError(
                expected_type=left_type,
                actual_type=right_type,
            )

        raise ValueError(f"Unsupported binary operator: {node.operator}")


class TypeValidator:
    """Validates AST expressions against type rules and symbol definitions."""

    def __init__(self, inference_engine: Optional[TypeInferenceEngine] = None) -> None:
        self._inference_engine = inference_engine or TypeInferenceEngine()

    def validate(
        self,
        node: ASTNode,
        symbol_table: Optional[SymbolTable] = None,
    ) -> Type:
        """Validate an AST expression node and return its statically inferred type."""
        return self._inference_engine.infer(node, symbol_table=symbol_table)


__all__ = [
    "TypeInferenceEngine",
    "TypeValidator",
    "TypeMismatchError",
    "UndefinedSymbolError",
]