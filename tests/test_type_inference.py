"""
Unit tests for Static Type Inference and Validation Engine.

Feature: Build Static Type Inference and Validation Engine
Requirement: Story 3.1.3: Build Static Type Inference and Validation Engine
Target Modules:
    - src/type_engine/types.py
    - src/type_engine/validator.py
"""

import pytest

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
)
from src.type_engine.validator import (
    TypeInferenceEngine,
    TypeMismatchError,
    TypeValidator,
    UndefinedSymbolError,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def inference_engine() -> TypeInferenceEngine:
    """Fixture providing a fresh instance of the TypeInferenceEngine."""
    return TypeInferenceEngine()


@pytest.fixture
def validator() -> TypeValidator:
    """Fixture providing a fresh instance of the TypeValidator."""
    return TypeValidator()


@pytest.fixture
def populated_symbol_table() -> SymbolTable:
    """Fixture providing a pre-configured SymbolTable with primitive identifier types."""
    table = SymbolTable()
    table.define("user_id", Type.INTEGER)
    table.define("username", Type.STRING)
    table.define("account_balance", Type.FLOAT)
    table.define("is_active", Type.BOOLEAN)
    table.define("retry_count", Type.INTEGER)
    return table


# ============================================================================
# Tests: src/type_engine/types.py Data Structures & SymbolTable
# ============================================================================


class TestTypeDefinitions:
    """Unit tests validating core types and AST node abstractions."""

    def test_primitive_type_enumeration_completeness(self):
        """Ensure all required primitive types are defined and distinct."""
        expected_types = {Type.INTEGER, Type.FLOAT, Type.STRING, Type.BOOLEAN}
        assert len(expected_types) == 4
        assert Type.INTEGER != Type.FLOAT
        assert Type.STRING != Type.BOOLEAN

    def test_literal_node_instantiation(self):
        """Ensure Literal nodes store raw value and explicit type annotation."""
        lit = Literal(value=42, inferred_type=Type.INTEGER)
        assert isinstance(lit, ASTNode)
        assert lit.value == 42
        assert lit.inferred_type == Type.INTEGER

    def test_identifier_node_instantiation(self):
        """Ensure Identifier nodes store symbol names correctly."""
        ident = Identifier(name="account_balance")
        assert isinstance(ident, ASTNode)
        assert ident.name == "account_balance"

    def test_binary_op_node_instantiation(self):
        """Ensure BinaryOp nodes correctly compose operands and operators."""
        left = Literal(10, Type.INTEGER)
        right = Literal(20, Type.INTEGER)
        op = BinaryOp(left=left, operator=BinaryOperator.ADD, right=right)
        assert isinstance(op, ASTNode)
        assert op.left == left
        assert op.operator == BinaryOperator.ADD
        assert op.right == right

    def test_unary_op_node_instantiation(self):
        """Ensure UnaryOp nodes store operand and operator."""
        operand = Literal(True, Type.BOOLEAN)
        op = UnaryOp(operator=UnaryOperator.NOT, operand=operand)
        assert isinstance(op, ASTNode)
        assert op.operator == UnaryOperator.NOT
        assert op.operand == operand


class TestSymbolTable:
    """Unit tests validating SymbolTable scope and type mappings."""

    def test_define_and_lookup_symbol(self):
        """Verify symbols can be defined and looked up with exact types."""
        table = SymbolTable()
        table.define("counter", Type.INTEGER)
        assert table.lookup("counter") == Type.INTEGER

    def test_lookup_undefined_symbol_raises_error(self):
        """Verify looking up an undeclared symbol raises UndefinedSymbolError."""
        table = SymbolTable()
        with pytest.raises(UndefinedSymbolError):
            table.lookup("missing_var")

    def test_symbol_table_initialization_with_mapping(self):
        """Verify SymbolTable can be populated via initial mapping dictionary."""
        table = SymbolTable({"x": Type.FLOAT, "flag": Type.BOOLEAN})
        assert table.lookup("x") == Type.FLOAT
        assert table.lookup("flag") == Type.BOOLEAN

    def test_symbol_table_overwrite_definition(self):
        """Verify defining an existing symbol updates its registered type."""
        table = SymbolTable({"var": Type.INTEGER})
        table.define("var", Type.STRING)
        assert table.lookup("var") == Type.STRING


# ============================================================================
# Tests: Acceptance Criterion 1 - Literal Type Inference (No Runtime Eval)
# ============================================================================


class TestStaticTypeInference:
    """Tests for static type inference of matching literal operands."""

    def test_infer_integer_addition(self, inference_engine: TypeInferenceEngine):
        """Given matching integer literals, When inferred, Then return Type.INTEGER."""
        expr = BinaryOp(
            left=Literal(10, Type.INTEGER),
            operator=BinaryOperator.ADD,
            right=Literal(20, Type.INTEGER),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.INTEGER

    def test_infer_float_multiplication(self, inference_engine: TypeInferenceEngine):
        """Given matching float literals, When inferred, Then return Type.FLOAT."""
        expr = BinaryOp(
            left=Literal(3.14, Type.FLOAT),
            operator=BinaryOperator.MUL,
            right=Literal(2.0, Type.FLOAT),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.FLOAT

    def test_infer_string_concatenation(self, inference_engine: TypeInferenceEngine):
        """Given matching string literals, When added, Then return Type.STRING."""
        expr = BinaryOp(
            left=Literal("Hello, ", Type.STRING),
            operator=BinaryOperator.ADD,
            right=Literal("World!", Type.STRING),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.STRING

    def test_infer_boolean_conjunction(self, inference_engine: TypeInferenceEngine):
        """Given matching boolean literals, When AND-ed, Then return Type.BOOLEAN."""
        expr = BinaryOp(
            left=Literal(True, Type.BOOLEAN),
            operator=BinaryOperator.AND,
            right=Literal(False, Type.BOOLEAN),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.BOOLEAN

    def test_infer_relational_comparison_returns_boolean(
        self, inference_engine: TypeInferenceEngine
    ):
        """Given matching numeric literals, When compared, Then return Type.BOOLEAN."""
        expr = BinaryOp(
            left=Literal(100, Type.INTEGER),
            operator=BinaryOperator.GT,
            right=Literal(50, Type.INTEGER),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.BOOLEAN

    def test_infer_equality_comparison_returns_boolean(
        self, inference_engine: TypeInferenceEngine
    ):
        """Given matching string literals, When compared for equality, Then return Type.BOOLEAN."""
        expr = BinaryOp(
            left=Literal("status", Type.STRING),
            operator=BinaryOperator.EQ,
            right=Literal("status", Type.STRING),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.BOOLEAN

    def test_infer_division_by_zero_does_not_evaluate_runtime_error(
        self, inference_engine: TypeInferenceEngine
    ):
        """
        Given a division operation with a divisor of zero,
        When statically inferred,
        Then correctly infer Type.FLOAT without raising ZeroDivisionError at analysis time.
        """
        expr = BinaryOp(
            left=Literal(10, Type.INTEGER),
            operator=BinaryOperator.DIV,
            right=Literal(0, Type.INTEGER),
        )
        # Runtime evaluation would raise ZeroDivisionError; static analysis must not.
        inferred = inference_engine.infer(expr)
        assert inferred == Type.FLOAT

    def test_infer_unary_negation_on_number(self, inference_engine: TypeInferenceEngine):
        """Given a numeric literal, When negated, Then return the matching numeric type."""
        expr = UnaryOp(
            operator=UnaryOperator.NEG,
            operand=Literal(5, Type.INTEGER),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.INTEGER

    def test_infer_unary_not_on_boolean(self, inference_engine: TypeInferenceEngine):
        """Given a boolean literal, When inverted, Then return Type.BOOLEAN."""
        expr = UnaryOp(
            operator=UnaryOperator.NOT,
            operand=Literal(True, Type.BOOLEAN),
        )
        inferred = inference_engine.infer(expr)
        assert inferred == Type.BOOLEAN


# ============================================================================
# Tests: Acceptance Criterion 2 - Type Mismatch Validation Errors
# ============================================================================


class TestTypeValidationErrors:
    """Tests verifying TypeMismatchError is raised with expected and actual types."""

    def test_validate_string_subtraction_integer_raises_mismatch(
        self, validator: TypeValidator
    ):
        """
        Given an expression with String and Integer for subtraction,
        When validated,
        Then raise TypeMismatchError identifying expected numeric type and actual string type.
        """
        expr = BinaryOp(
            left=Literal("invalid_text", Type.STRING),
            operator=BinaryOperator.SUB,
            right=Literal(42, Type.INTEGER),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type in (Type.INTEGER, Type.FLOAT)
        assert err.actual_type == Type.STRING

    def test_validate_integer_subtraction_string_raises_mismatch(
        self, validator: TypeValidator
    ):
        """
        Given an expression subtracting a String from an Integer,
        When validated,
        Then raise TypeMismatchError identifying expected numeric type and actual string type.
        """
        expr = BinaryOp(
            left=Literal(100, Type.INTEGER),
            operator=BinaryOperator.SUB,
            right=Literal("not_a_number", Type.STRING),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type in (Type.INTEGER, Type.FLOAT)
        assert err.actual_type == Type.STRING

    def test_validate_string_multiplication_raises_mismatch(
        self, validator: TypeValidator
    ):
        """Given two strings multiplied together, When validated, Then raise TypeMismatchError."""
        expr = BinaryOp(
            left=Literal("alpha", Type.STRING),
            operator=BinaryOperator.MUL,
            right=Literal("beta", Type.STRING),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type in (Type.INTEGER, Type.FLOAT)
        assert err.actual_type == Type.STRING

    def test_validate_logical_and_with_non_boolean_operands_raises_mismatch(
        self, validator: TypeValidator
    ):
        """Given an integer operand to a logical AND, When validated, Then raise TypeMismatchError."""
        expr = BinaryOp(
            left=Literal(1, Type.INTEGER),
            operator=BinaryOperator.AND,
            right=Literal(True, Type.BOOLEAN),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type == Type.BOOLEAN
        assert err.actual_type == Type.INTEGER

    def test_validate_relational_operator_with_mismatched_types_raises_mismatch(
        self, validator: TypeValidator
    ):
        """Given string and float comparison via <, When validated, Then raise TypeMismatchError."""
        expr = BinaryOp(
            left=Literal("threshold", Type.STRING),
            operator=BinaryOperator.LT,
            right=Literal(10.5, Type.FLOAT),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type == Type.FLOAT
        assert err.actual_type == Type.STRING

    def test_validate_unary_negation_on_string_raises_mismatch(
        self, validator: TypeValidator
    ):
        """Given a unary negation applied to a string, When validated, Then raise TypeMismatchError."""
        expr = UnaryOp(
            operator=UnaryOperator.NEG,
            operand=Literal("unsupported", Type.STRING),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type in (Type.INTEGER, Type.FLOAT)
        assert err.actual_type == Type.STRING

    def test_validate_unary_not_on_numeric_raises_mismatch(
        self, validator: TypeValidator
    ):
        """Given a unary NOT applied to a float, When validated, Then raise TypeMismatchError."""
        expr = UnaryOp(
            operator=UnaryOperator.NOT,
            operand=Literal(0.0, Type.FLOAT),
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(expr)

        err = exc_info.value
        assert err.expected_type == Type.BOOLEAN
        assert err.actual_type == Type.FLOAT


# ============================================================================
# Tests: Acceptance Criterion 3 - Symbol Table Resolution & Composite Ops
# ============================================================================


class TestSymbolTableResolutionAndCompositeValidation:
    """Tests verifying symbol resolution and validation across composite expressions."""

    def test_validate_single_identifier_resolution(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """Given a declared identifier, When validated, Then resolve its predefined type."""
        expr = Identifier("user_id")
        result_type = validator.validate(expr, symbol_table=populated_symbol_table)
        assert result_type == Type.INTEGER

    def test_validate_composite_arithmetic_operations(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """
        Given identifiers 'user_id' (INT) and 'retry_count' (INT),
        When validating ((user_id + retry_count) * 2),
        Then successfully resolve identifiers and return Type.INTEGER.
        """
        # (user_id + retry_count)
        inner_add = BinaryOp(
            left=Identifier("user_id"),
            operator=BinaryOperator.ADD,
            right=Identifier("retry_count"),
        )
        # ((user_id + retry_count) * 2)
        full_expr = BinaryOp(
            left=inner_add,
            operator=BinaryOperator.MUL,
            right=Literal(2, Type.INTEGER),
        )
        result_type = validator.validate(full_expr, symbol_table=populated_symbol_table)
        assert result_type == Type.INTEGER

    def test_validate_composite_relational_and_boolean_expression(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """
        Given identifiers 'account_balance' (FLOAT) and 'is_active' (BOOLEAN),
        When validating ((account_balance > 0.0) and is_active),
        Then validate composite operations and return Type.BOOLEAN.
        """
        # (account_balance > 0.0) -> BOOLEAN
        comparison = BinaryOp(
            left=Identifier("account_balance"),
            operator=BinaryOperator.GT,
            right=Literal(0.0, Type.FLOAT),
        )
        # (comparison and is_active) -> BOOLEAN
        logical = BinaryOp(
            left=comparison,
            operator=BinaryOperator.AND,
            right=Identifier("is_active"),
        )
        result_type = validator.validate(logical, symbol_table=populated_symbol_table)
        assert result_type == Type.BOOLEAN

    def test_validate_composite_string_concatenation_with_identifier(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """
        Given identifier 'username' (STRING),
        When validating ("prefix_" + username),
        Then validate successfully and return Type.STRING.
        """
        expr = BinaryOp(
            left=Literal("prefix_", Type.STRING),
            operator=BinaryOperator.ADD,
            right=Identifier("username"),
        )
        result_type = validator.validate(expr, symbol_table=populated_symbol_table)
        assert result_type == Type.STRING

    def test_validate_composite_expression_with_nested_type_mismatch(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """
        Given a deeply nested expression with an invalid sub-tree:
        (is_active and (username - user_id)),
        When validated,
        Then bubble up TypeMismatchError from the invalid node.
        """
        invalid_sub_expr = BinaryOp(
            left=Identifier("username"),  # STRING
            operator=BinaryOperator.SUB,  # Invalid for STRING
            right=Identifier("user_id"),  # INTEGER
        )
        full_expr = BinaryOp(
            left=Identifier("is_active"),
            operator=BinaryOperator.AND,
            right=invalid_sub_expr,
        )
        with pytest.raises(TypeMismatchError) as exc_info:
            validator.validate(full_expr, symbol_table=populated_symbol_table)

        err = exc_info.value
        assert err.expected_type in (Type.INTEGER, Type.FLOAT)
        assert err.actual_type == Type.STRING

    def test_validate_unregistered_identifier_raises_undefined_symbol_error(
        self, validator: TypeValidator, populated_symbol_table: SymbolTable
    ):
        """
        Given an expression referencing an undefined symbol 'non_existent_var',
        When validated,
        Then raise UndefinedSymbolError.
        """
        expr = BinaryOp(
            left=Identifier("user_id"),
            operator=BinaryOperator.ADD,
            right=Identifier("non_existent_var"),
        )
        with pytest.raises(UndefinedSymbolError):
            validator.validate(expr, symbol_table=populated_symbol_table)

    def test_validate_composite_without_symbol_table_raises_error_for_identifiers(
        self, validator: TypeValidator
    ):
        """
        Given an expression referencing identifiers but no symbol table is passed,
        When validated,
        Then raise UndefinedSymbolError.
        """
        expr = BinaryOp(
            left=Identifier("x"),
            operator=BinaryOperator.ADD,
            right=Literal(1, Type.INTEGER),
        )
        with pytest.raises(UndefinedSymbolError):
            validator.validate(expr, symbol_table=None)