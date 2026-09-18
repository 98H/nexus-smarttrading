"""
Unit tests for Concrete Syntax Tree (CST) to Abstract Syntax Tree (AST) Generator.

Target Modules:
- src/parser/ast_nodes.py
- src/parser/ast_generator.py
- src/parser/__init__.py

Story 3.1.2 Acceptance Criteria:
- Given a stream of parsed concrete syntax nodes or tokens representing an expression.
- When the AST generator processes the concrete syntax tree (CST).
- Then it constructs a normalized Abstract Syntax Tree (AST) stripped of concrete trivia
  (whitespace, punctuation) preserving source location spans.
"""

from typing import Any, List, Optional
import pytest

from src.parser import (
    ASTGenerationError,
    ASTGenerator,
    ASTNode,
    BinaryExpr,
    CallExpr,
    IdentifierExpr,
    LiteralExpr,
    SourceLocation,
    SourceSpan,
    UnaryExpr,
    generate_ast,
)

# CST types may be exposed via src.parser, src.parser.cst_nodes, or src.parser.ast_generator
try:
    from src.parser.cst_nodes import CSTNode, CSTToken, Trivia, TriviaKind
except (ImportError, ModuleNotFoundError):
    try:
        from src.parser import CSTNode, CSTToken, Trivia, TriviaKind
    except (ImportError, ModuleNotFoundError):
        from src.parser.ast_generator import CSTNode, CSTToken, Trivia, TriviaKind


# ============================================================================
# Helpers & CST Builders (No mocks; strictly deterministic CST construction)
# ============================================================================


def make_span(start_line: int, start_col: int, end_line: int, end_col: int) -> SourceSpan:
    """Helper to build a SourceSpan from line/column integers."""
    return SourceSpan(
        start=SourceLocation(line=start_line, column=start_col),
        end=SourceLocation(line=end_line, column=end_col),
    )


def make_trivia(kind: str, text: str, start_line: int, start_col: int, end_line: int, end_col: int) -> Trivia:
    """Helper to build a Trivia node (whitespace, comment, etc.)."""
    span = make_span(start_line, start_col, end_line, end_col)
    try:
        return Trivia(kind=kind, text=text, span=span)
    except TypeError:
        return Trivia(kind=getattr(TriviaKind, kind.upper(), kind), text=text, span=span)


def make_cst_token(
    kind: str,
    value: str,
    span: SourceSpan,
    leading_trivia: Optional[List[Trivia]] = None,
    trailing_trivia: Optional[List[Trivia]] = None,
) -> CSTToken:
    """Helper to instantiate CSTToken handling either `value` or `text` signatures."""
    leading = leading_trivia or []
    trailing = trailing_trivia or []
    try:
        return CSTToken(
            kind=kind,
            value=value,
            span=span,
            leading_trivia=leading,
            trailing_trivia=trailing,
        )
    except TypeError:
        return CSTToken(
            kind=kind,
            text=value,
            span=span,
            leading_trivia=leading,
            trailing_trivia=trailing,
        )


def make_cst_node(kind: str, children: List[Any], span: SourceSpan) -> CSTNode:
    """Helper to instantiate a generic CST composite node."""
    return CSTNode(kind=kind, children=children, span=span)


# ============================================================================
# Test Suite 1: AST Node Hierarchy & Spans (src/parser/ast_nodes.py)
# ============================================================================


class TestASTNodes:
    """Tests for AST node definitions and source location spans."""

    def test_source_location_creation_and_attributes(self):
        loc = SourceLocation(line=10, column=25)
        assert loc.line == 10
        assert loc.column == 25
        assert SourceLocation(line=10, column=25) == loc

    def test_source_location_validation_negative_or_zero(self):
        with pytest.raises(ValueError):
            SourceLocation(line=0, column=1)

        with pytest.raises(ValueError):
            SourceLocation(line=1, column=-1)

    def test_source_span_valid_range_and_equality(self):
        start = SourceLocation(line=1, column=1)
        end = SourceLocation(line=1, column=10)
        span1 = SourceSpan(start=start, end=end)
        span2 = SourceSpan(start=SourceLocation(line=1, column=1), end=SourceLocation(line=1, column=10))

        assert span1 == span2
        assert span1.start == start
        assert span1.end == end

    def test_source_span_invalid_range_raises_value_error(self):
        # End location before start location on same line
        start = SourceLocation(line=2, column=10)
        end = SourceLocation(line=2, column=5)
        with pytest.raises(ValueError):
            SourceSpan(start=start, end=end)

        # End location on earlier line
        start_multi = SourceLocation(line=3, column=1)
        end_multi = SourceLocation(line=2, column=10)
        with pytest.raises(ValueError):
            SourceSpan(start=start_multi, end=end_multi)

    def test_literal_expr_initialization_and_span(self):
        span = make_span(1, 1, 1, 3)
        lit = LiteralExpr(value=42, span=span, raw="42")

        assert isinstance(lit, ASTNode)
        assert lit.value == 42
        assert lit.raw == "42"
        assert lit.span == span

    def test_identifier_expr_initialization_and_span(self):
        span = make_span(1, 5, 1, 10)
        ident = IdentifierExpr(name="total", span=span)

        assert isinstance(ident, ASTNode)
        assert ident.name == "total"
        assert ident.span == span

    def test_binary_expr_initialization_and_span(self):
        span_left = make_span(1, 1, 1, 2)
        span_right = make_span(1, 5, 1, 6)
        span_full = make_span(1, 1, 1, 6)

        left = LiteralExpr(value=1, span=span_left)
        right = LiteralExpr(value=2, span=span_right)
        binary = BinaryExpr(left=left, operator="+", right=right, span=span_full)

        assert isinstance(binary, ASTNode)
        assert binary.operator == "+"
        assert binary.left == left
        assert binary.right == right
        assert binary.span == span_full

    def test_unary_expr_initialization_and_span(self):
        span_op = make_span(1, 1, 1, 3)
        operand = IdentifierExpr(name="x", span=make_span(1, 2, 1, 3))
        unary = UnaryExpr(operator="-", operand=operand, span=span_op)

        assert isinstance(unary, ASTNode)
        assert unary.operator == "-"
        assert unary.operand == operand
        assert unary.span == span_op

    def test_call_expr_initialization_and_span(self):
        span_callee = make_span(1, 1, 1, 4)
        span_call = make_span(1, 1, 1, 12)
        callee = IdentifierExpr(name="func", span=span_callee)
        arg1 = LiteralExpr(value=10, span=make_span(1, 5, 1, 7))
        arg2 = IdentifierExpr(name="y", span=make_span(1, 9, 1, 10))

        call = CallExpr(callee=callee, arguments=[arg1, arg2], span=span_call)

        assert isinstance(call, ASTNode)
        assert call.callee == callee
        assert call.arguments == [arg1, arg2]
        assert call.span == span_call

    def test_ast_node_equality_structural(self):
        span = make_span(1, 1, 1, 5)
        node_a = BinaryExpr(
            left=LiteralExpr(value=1, span=make_span(1, 1, 1, 2)),
            operator="+",
            right=LiteralExpr(value=2, span=make_span(1, 4, 1, 5)),
            span=span,
        )
        node_b = BinaryExpr(
            left=LiteralExpr(value=1, span=make_span(1, 1, 1, 2)),
            operator="+",
            right=LiteralExpr(value=2, span=make_span(1, 4, 1, 5)),
            span=span,
        )
        assert node_a == node_b


# ============================================================================
# Test Suite 2: Module Exports (src/parser/__init__.py)
# ============================================================================


class TestParserModuleExports:
    """Verifies that the parser module exports all public AST generator symbols."""

    def test_expected_symbols_exported_in_package(self):
        import src.parser as parser

        required_symbols = [
            "ASTNode",
            "LiteralExpr",
            "IdentifierExpr",
            "BinaryExpr",
            "UnaryExpr",
            "CallExpr",
            "SourceLocation",
            "SourceSpan",
            "ASTGenerator",
            "generate_ast",
            "ASTGenerationError",
        ]

        for symbol in required_symbols:
            assert hasattr(parser, symbol), f"Missing exported symbol: {symbol}"

    def test_dunder_all_matches_public_api(self):
        import src.parser as parser

        assert hasattr(parser, "__all__")
        all_list = parser.__all__
        assert "ASTGenerator" in all_list
        assert "generate_ast" in all_list
        assert "BinaryExpr" in all_list
        assert "SourceSpan" in all_list


# ============================================================================
# Test Suite 3: CST to AST Generator - Trivia Stripping (src/parser/ast_generator.py)
# ============================================================================


class TestASTGeneratorTriviaStripping:
    """
    Tests ensuring all whitespace and comment trivia are stripped from the AST
    while preserving correct AST node structure.
    """

    def test_generate_literal_with_leading_and_trailing_trivia(self):
        leading = [make_trivia("whitespace", "   ", 1, 1, 1, 4)]
        trailing = [
            make_trivia("whitespace", " ", 1, 6, 1, 7),
            make_trivia("comment", "# comment", 1, 7, 1, 16),
        ]
        token_span = make_span(1, 4, 1, 6)
        cst_token = make_cst_token(
            kind="INTEGER",
            value="42",
            span=token_span,
            leading_trivia=leading,
            trailing_trivia=trailing,
        )
        cst_root = make_cst_node("literal", [cst_token], token_span)

        generator = ASTGenerator()
        ast = generator.generate(cst_root)

        assert isinstance(ast, LiteralExpr)
        assert ast.value == 42
        assert ast.span == token_span
        # AST should NOT contain trivia attributes holding whitespace/comments
        assert not hasattr(ast, "leading_trivia")
        assert not hasattr(ast, "trailing_trivia")

    def test_generate_string_literal_normalized(self):
        token_span = make_span(1, 1, 1, 13)
        cst_token = make_cst_token(kind="STRING", value='"hello world"', span=token_span)
        cst_root = make_cst_node("literal", [cst_token], token_span)

        ast = generate_ast(cst_root)

        assert isinstance(ast, LiteralExpr)
        assert ast.value == "hello world"
        assert ast.span == token_span

    def test_generate_boolean_literal(self):
        token_span = make_span(1, 1, 1, 5)
        cst_token = make_cst_token(kind="BOOLEAN", value="true", span=token_span)
        cst_root = make_cst_node("literal", [cst_token], token_span)

        ast = generate_ast(cst_root)

        assert isinstance(ast, LiteralExpr)
        assert ast.value is True
        assert ast.span == token_span

    def test_generate_identifier_strips_trivia(self):
        leading = [make_trivia("whitespace", "\t", 1, 1, 1, 2)]
        token_span = make_span(1, 2, 1, 7)
        cst_token = make_cst_token(
            kind="IDENTIFIER",
            value="count",
            span=token_span,
            leading_trivia=leading,
        )
        cst_root = make_cst_node("identifier", [cst_token], token_span)

        ast = generate_ast(cst_root)

        assert isinstance(ast, IdentifierExpr)
        assert ast.name == "count"
        assert ast.span == token_span

    def test_generate_binary_expression_strips_whitespace_tokens(self):
        # CST: "10  +   20"
        left_span = make_span(1, 1, 1, 3)
        left_tok = make_cst_token("INTEGER", "10", left_span)
        left_node = make_cst_node("literal", [left_tok], left_span)

        ws1 = make_cst_token("WHITESPACE", "  ", make_span(1, 3, 1, 5))
        op_span = make_span(1, 5, 1, 6)
        op_tok = make_cst_token("PLUS", "+", op_span)
        ws2 = make_cst_token("WHITESPACE", "   ", make_span(1, 6, 1, 9))

        right_span = make_span(1, 9, 1, 11)
        right_tok = make_cst_token("INTEGER", "20", right_span)
        right_node = make_cst_node("literal", [right_tok], right_span)

        full_span = make_span(1, 1, 1, 11)
        # CST children includes concrete whitespace tokens
        cst_root = make_cst_node("binary_expr", [left_node, ws1, op_tok, ws2, right_node], full_span)

        generator = ASTGenerator()
        ast = generator.generate(cst_root)

        assert isinstance(ast, BinaryExpr)
        assert ast.operator == "+"
        assert ast.span == full_span

        assert isinstance(ast.left, LiteralExpr)
        assert ast.left.value == 10
        assert ast.left.span == left_span

        assert isinstance(ast.right, LiteralExpr)
        assert ast.right.value == 20
        assert ast.right.span == right_span


# ============================================================================
# Test Suite 4: CST to AST Generator - Punctuation Stripping (src/parser/ast_generator.py)
# ============================================================================


class TestASTGeneratorPunctuationStripping:
    """
    Tests ensuring syntax punctuation (parentheses, commas, semicolons)
    is stripped and not retained as children/tokens in the generated AST.
    """

    def test_generate_parenthesized_expression_unwraps_parentheses(self):
        # CST: "( 42 )"
        lparen = make_cst_token("LPAREN", "(", make_span(1, 1, 1, 2))
        ws1 = make_cst_token("WHITESPACE", " ", make_span(1, 2, 1, 3))
        inner_span = make_span(1, 3, 1, 5)
        num_tok = make_cst_token("INTEGER", "42", inner_span)
        inner_node = make_cst_node("literal", [num_tok], inner_span)
        ws2 = make_cst_token("WHITESPACE", " ", make_span(1, 5, 1, 6))
        rparen = make_cst_token("RPAREN", ")", make_span(1, 6, 1, 7))

        full_span = make_span(1, 1, 1, 7)
        cst_root = make_cst_node("paren_expr", [lparen, ws1, inner_node, ws2, rparen], full_span)

        ast = generate_ast(cst_root)

        # AST must unwrap the parenthesis node to the normalized inner expression
        # preserving the expression span
        assert isinstance(ast, LiteralExpr)
        assert ast.value == 42
        assert ast.span == full_span

    def test_generate_nested_parentheses_unwraps_multiple_layers(self):
        # CST: "(( x ))"
        inner_ident_span = make_span(1, 4, 1, 5)
        ident_tok = make_cst_token("IDENTIFIER", "x", inner_ident_span)
        ident_node = make_cst_node("identifier", [ident_tok], inner_ident_span)

        inner_paren_span = make_span(1, 2, 1, 7)
        inner_paren = make_cst_node(
            "paren_expr",
            [
                make_cst_token("LPAREN", "(", make_span(1, 2, 1, 3)),
                make_cst_token("WHITESPACE", " ", make_span(1, 3, 1, 4)),
                ident_node,
                make_cst_token("WHITESPACE", " ", make_span(1, 5, 1, 6)),
                make_cst_token("RPAREN", ")", make_span(1, 6, 1, 7)),
            ],
            inner_paren_span,
        )

        outer_paren_span = make_span(1, 1, 1, 8)
        outer_paren = make_cst_node(
            "paren_expr",
            [
                make_cst_token("LPAREN", "(", make_span(1, 1, 1, 2)),
                inner_paren,
                make_cst_token("RPAREN", ")", make_span(1, 7, 1, 8)),
            ],
            outer_paren_span,
        )

        ast = generate_ast(outer_paren)

        assert isinstance(ast, IdentifierExpr)
        assert ast.name == "x"
        assert ast.span == outer_paren_span

    def test_generate_call_expression_strips_parens_and_commas(self):
        # CST: "add(a, b)"
        callee_span = make_span(1, 1, 1, 4)
        callee_tok = make_cst_token("IDENTIFIER", "add", callee_span)
        callee_node = make_cst_node("identifier", [callee_tok], callee_span)

        lparen = make_cst_token("LPAREN", "(", make_span(1, 4, 1, 5))

        arg1_span = make_span(1, 5, 1, 6)
        arg1_tok = make_cst_token("IDENTIFIER", "a", arg1_span)
        arg1_node = make_cst_node("identifier", [arg1_tok], arg1_span)

        comma = make_cst_token("COMMA", ",", make_span(1, 6, 1, 7))
        ws = make_cst_token("WHITESPACE", " ", make_span(1, 7, 1, 8))

        arg2_span = make_span(1, 8, 1, 9)
        arg2_tok = make_cst_token("IDENTIFIER", "b", arg2_span)
        arg2_node = make_cst_node("identifier", [arg2_tok], arg2_span)

        rparen = make_cst_token("RPAREN", ")", make_span(1, 9, 1, 10))

        full_span = make_span(1, 1, 1, 10)
        cst_call = make_cst_node(
            "call_expr",
            [callee_node, lparen, arg1_node, comma, ws, arg2_node, rparen],
            full_span,
        )

        ast = generate_ast(cst_call)

        assert isinstance(ast, CallExpr)
        assert ast.span == full_span
        assert isinstance(ast.callee, IdentifierExpr)
        assert ast.callee.name == "add"
        assert ast.callee.span == callee_span

        # Arguments must contain ONLY the evaluated argument AST expressions
        assert len(ast.arguments) == 2
        assert isinstance(ast.arguments[0], IdentifierExpr)
        assert ast.arguments[0].name == "a"
        assert ast.arguments[0].span == arg1_span

        assert isinstance(ast.arguments[1], IdentifierExpr)
        assert ast.arguments[1].name == "b"
        assert ast.arguments[1].span == arg2_span

    def test_generate_call_expression_with_zero_arguments(self):
        # CST: "run()"
        callee_span = make_span(1, 1, 1, 4)
        callee_tok = make_cst_token("IDENTIFIER", "run", callee_span)
        callee_node = make_cst_node("identifier", [callee_tok], callee_span)

        lparen = make_cst_token("LPAREN", "(", make_span(1, 4, 1, 5))
        ws = make_cst_token("WHITESPACE", "  ", make_span(1, 5, 1, 7))
        rparen = make_cst_token("RPAREN", ")", make_span(1, 7, 1, 8))

        full_span = make_span(1, 1, 1, 8)
        cst_call = make_cst_node("call_expr", [callee_node, lparen, ws, rparen], full_span)

        ast = generate_ast(cst_call)

        assert isinstance(ast, CallExpr)
        assert ast.span == full_span
        assert ast.callee.name == "run"
        assert ast.arguments == []


# ============================================================================
# Test Suite 5: Complex & Nested Expression Tree Construction
# ============================================================================


class TestASTGeneratorComplexExpressions:
    """Tests complex nested structures and precedence tree preservation."""

    def test_generate_unary_expression(self):
        # CST: "!isValid"
        op_span = make_span(1, 1, 1, 2)
        op_tok = make_cst_token("BANG", "!", op_span)

        operand_span = make_span(1, 2, 1, 9)
        operand_tok = make_cst_token("IDENTIFIER", "isValid", operand_span)
        operand_node = make_cst_node("identifier", [operand_tok], operand_span)

        full_span = make_span(1, 1, 1, 9)
        cst_unary = make_cst_node("unary_expr", [op_tok, operand_node], full_span)

        ast = generate_ast(cst_unary)

        assert isinstance(ast, UnaryExpr)
        assert ast.operator == "!"
        assert ast.span == full_span
        assert isinstance(ast.operand, IdentifierExpr)
        assert ast.operand.name == "isValid"
        assert ast.operand.span == operand_span

    def test_generate_nested_binary_expression_with_parentheses(self):
        # CST: "(a + b) * c"
        # 1. Inner (a + b)
        a_span = make_span(1, 2, 1, 3)
        a_tok = make_cst_token("IDENTIFIER", "a", a_span)
        a_node = make_cst_node("identifier", [a_tok], a_span)

        plus_tok = make_cst_token("PLUS", "+", make_span(1, 4, 1, 5))

        b_span = make_span(1, 6, 1, 7)
        b_tok = make_cst_token("IDENTIFIER", "b", b_span)
        b_node = make_cst_node("identifier", [b_tok], b_span)

        inner_bin_span = make_span(1, 2, 1, 7)
        inner_bin = make_cst_node("binary_expr", [a_node, plus_tok, b_node], inner_bin_span)

        # Paren wrapper
        paren_span = make_span(1, 1, 1, 8)
        paren_node = make_cst_node(
            "paren_expr",
            [
                make_cst_token("LPAREN", "(", make_span(1, 1, 1, 2)),
                inner_bin,
                make_cst_token("RPAREN", ")", make_span(1, 7, 1, 8)),
            ],
            paren_span,
        )

        # 2. Outer * c
        mul_tok = make_cst_token("STAR", "*", make_span(1, 9, 1, 10))

        c_span = make_span(1, 11, 1, 12)
        c_tok = make_cst_token("IDENTIFIER", "c", c_span)
        c_node = make_cst_node("identifier", [c_tok], c_span)

        outer_span = make_span(1, 1, 1, 12)
        outer_cst = make_cst_node("binary_expr", [paren_node, mul_tok, c_node], outer_span)

        ast = generate_ast(outer_cst)

        assert isinstance(ast, BinaryExpr)
        assert ast.operator == "*"
        assert ast.span == outer_span

        # Left side is the unwrapped BinaryExpr
        assert isinstance(ast.left, BinaryExpr)
        assert ast.left.operator == "+"
        assert ast.left.span == paren_span
        assert isinstance(ast.left.left, IdentifierExpr)
        assert ast.left.left.name == "a"
        assert isinstance(ast.left.right, IdentifierExpr)
        assert ast.left.right.name == "b"

        # Right side is 'c'
        assert isinstance(ast.right, IdentifierExpr)
        assert ast.right.name == "c"
        assert ast.right.span == c_span

    def test_generate_call_with_binary_and_unary_arguments(self):
        # CST: "compute(-x, 2 * 3)"
        callee_span = make_span(1, 1, 1, 8)
        callee = make_cst_node(
            "identifier",
            [make_cst_token("IDENTIFIER", "compute", callee_span)],
            callee_span,
        )

        # Arg 1: -x
        arg1_operand_span = make_span(1, 10, 1, 11)
        arg1_operand = make_cst_node(
            "identifier",
            [make_cst_token("IDENTIFIER", "x", arg1_operand_span)],
            arg1_operand_span,
        )
        arg1_span = make_span(1, 9, 1, 11)
        arg1 = make_cst_node(
            "unary_expr",
            [make_cst_token("MINUS", "-", make_span(1, 9, 1, 10)), arg1_operand],
            arg1_span,
        )

        # Arg 2: 2 * 3
        arg2_left_span = make_span(1, 13, 1, 14)
        arg2_left = make_cst_node(
            "literal",
            [make_cst_token("INTEGER", "2", arg2_left_span)],
            arg2_left_span,
        )
        arg2_op = make_cst_token("STAR", "*", make_span(1, 15, 1, 16))
        arg2_right_span = make_span(1, 17, 1, 18)
        arg2_right = make_cst_node(
            "literal",
            [make_cst_token("INTEGER", "3", arg2_right_span)],
            arg2_right_span,
        )
        arg2_span = make_span(1, 13, 1, 18)
        arg2 = make_cst_node("binary_expr", [arg2_left, arg2_op, arg2_right], arg2_span)

        total_span = make_span(1, 1, 1, 19)
        cst_call = make_cst_node(
            "call_expr",
            [
                callee,
                make_cst_token("LPAREN", "(", make_span(1, 8, 1, 9)),
                arg1,
                make_cst_token("COMMA", ",", make_span(1, 11, 1, 12)),
                make_cst_token("WHITESPACE", " ", make_span(1, 12, 1, 13)),
                arg2,
                make_cst_token("RPAREN", ")", make_span(1, 18, 1, 19)),
            ],
            total_span,
        )

        ast = generate_ast(cst_call)

        assert isinstance(ast, CallExpr)
        assert ast.span == total_span
        assert len(ast.arguments) == 2

        assert isinstance(ast.arguments[0], UnaryExpr)
        assert ast.arguments[0].operator == "-"
        assert ast.arguments[0].span == arg1_span

        assert isinstance(ast.arguments[1], BinaryExpr)
        assert ast.arguments[1].operator == "*"
        assert ast.arguments[1].span == arg2_span


# ============================================================================
# Test Suite 6: Source Location Span Preservation Edge Cases
# ============================================================================


class TestSourceLocationSpanPreservation:
    """Verifies that multi-line spans, columns, and internal offsets are preserved accurately."""

    def test_multi_line_binary_expression_preserves_line_boundaries(self):
        # Line 1:  foo +
        # Line 2:    bar
        left_span = make_span(1, 1, 1, 4)
        left = make_cst_node(
            "identifier",
            [make_cst_token("IDENTIFIER", "foo", left_span)],
            left_span,
        )
        op_tok = make_cst_token("PLUS", "+", make_span(1, 5, 1, 6))

        right_span = make_span(2, 3, 2, 6)
        right = make_cst_node(
            "identifier",
            [make_cst_token("IDENTIFIER", "bar", right_span)],
            right_span,
        )

        multi_line_span = make_span(1, 1, 2, 6)
        cst_node = make_cst_node("binary_expr", [left, op_tok, right], multi_line_span)

        ast = generate_ast(cst_node)

        assert isinstance(ast, BinaryExpr)
        assert ast.span.start.line == 1
        assert ast.span.start.column == 1
        assert ast.span.end.line == 2
        assert ast.span.end.column == 6
        assert ast.left.span == left_span
        assert ast.right.span == right_span

    def test_deeply_nested_subexpression_spans_do_not_bleed(self):
        span_a = make_span(1, 1, 1, 2)
        span_b = make_span(1, 5, 1, 6)
        span_c = make_span(1, 9, 1, 10)

        node_a = make_cst_node("identifier", [make_cst_token("IDENTIFIER", "a", span_a)], span_a)
        node_b = make_cst_node("identifier", [make_cst_token("IDENTIFIER", "b", span_b)], span_b)
        node_c = make_cst_node("identifier", [make_cst_token("IDENTIFIER", "c", span_c)], span_c)

        inner_span = make_span(1, 1, 1, 6)
        inner_bin = make_cst_node(
            "binary_expr",
            [node_a, make_cst_token("PLUS", "+", make_span(1, 3, 1, 4)), node_b],
            inner_span,
        )

        outer_span = make_span(1, 1, 1, 10)
        outer_bin = make_cst_node(
            "binary_expr",
            [inner_bin, make_cst_token("STAR", "*", make_span(1, 7, 1, 8)), node_c],
            outer_span,
        )

        ast = generate_ast(outer_bin)

        assert ast.span == outer_span
        assert ast.left.span == inner_span
        assert ast.left.left.span == span_a
        assert ast.left.right.span == span_b
        assert ast.right.span == span_c


# ============================================================================
# Test Suite 7: Error Handling & Edge Cases
# ============================================================================


class TestASTGeneratorErrorHandling:
    """Tests exception paths and malformed input handling in ASTGenerator."""

    def test_generate_with_unrecognized_cst_node_raises_error(self):
        span = make_span(1, 1, 1, 5)
        bad_cst = make_cst_node("completely_unknown_node_type", [], span)

        generator = ASTGenerator()
        with pytest.raises((ValueError, ASTGenerationError)):
            generator.generate(bad_cst)

    def test_generate_with_none_input_raises_type_error(self):
        generator = ASTGenerator()
        with pytest.raises((TypeError, ValueError)):
            generator.generate(None)

    def test_generate_with_primitive_type_input_raises_type_error(self):
        generator = ASTGenerator()
        with pytest.raises((TypeError, ValueError)):
            generator.generate("not a cst node")

    def test_generate_binary_expression_missing_operands_raises_error(self):
        # Binary expression missing right operand
        span = make_span(1, 1, 1, 4)
        left = make_cst_node("literal", [make_cst_token("INTEGER", "1", make_span(1, 1, 1, 2))], make_span(1, 1, 1, 2))
        op = make_cst_token("PLUS", "+", make_span(1, 3, 1, 4))
        malformed_binary = make_cst_node("binary_expr", [left, op], span)

        with pytest.raises((ValueError, ASTGenerationError)):
            generate_ast(malformed_binary)

    def test_generate_unary_expression_missing_operand_raises_error(self):
        span = make_span(1, 1, 1, 2)
        op = make_cst_token("MINUS", "-", span)
        malformed_unary = make_cst_node("unary_expr", [op], span)

        with pytest.raises((ValueError, ASTGenerationError)):
            generate_ast(malformed_unary)

    def test_generate_call_expression_missing_callee_raises_error(self):
        span = make_span(1, 1, 1, 3)
        lparen = make_cst_token("LPAREN", "(", make_span(1, 1, 1, 2))
        rparen = make_cst_token("RPAREN", ")", make_span(1, 2, 1, 3))
        malformed_call = make_cst_node("call_expr", [lparen, rparen], span)

        with pytest.raises((ValueError, ASTGenerationError)):
            generate_ast(malformed_call)