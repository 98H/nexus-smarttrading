"""Concrete Syntax Tree (CST) to Abstract Syntax Tree (AST) generator."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from src.parser.ast_nodes import (
    ASTNode,
    BinaryExpr,
    CallExpr,
    IdentifierExpr,
    LiteralExpr,
    SourceSpan,
    UnaryExpr,
)


class ASTGenerationError(ValueError):
    """Raised when Abstract Syntax Tree (AST) generation from CST fails."""


class TriviaKind(str, Enum):
    """Enumeration of trivia categories stripped during AST generation."""

    WHITESPACE = "whitespace"
    COMMENT = "comment"


@dataclass
class Trivia:
    """Concrete syntax trivia element (whitespace, comment)."""

    kind: Any
    text: str
    span: SourceSpan


class CSTToken:
    """Concrete syntax token carrying trivia and lexeme information."""

    def __init__(
        self,
        kind: str,
        value: Optional[str] = None,
        span: Optional[SourceSpan] = None,
        leading_trivia: Optional[List[Trivia]] = None,
        trailing_trivia: Optional[List[Trivia]] = None,
        text: Optional[str] = None,
    ) -> None:
        self.kind = kind
        val = value if value is not None else (text if text is not None else "")
        self.value = val
        self.text = val
        self.span = span  # type: ignore[assignment]
        self.leading_trivia = leading_trivia if leading_trivia is not None else []
        self.trailing_trivia = trailing_trivia if trailing_trivia is not None else []

    def __repr__(self) -> str:
        return (
            f"CSTToken(kind={self.kind!r}, value={self.value!r}, span={self.span!r}, "
            f"leading_trivia={self.leading_trivia!r}, trailing_trivia={self.trailing_trivia!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CSTToken):
            return False
        return (
            self.kind == other.kind
            and self.value == other.value
            and self.span == other.span
            and self.leading_trivia == other.leading_trivia
            and self.trailing_trivia == other.trailing_trivia
        )


@dataclass
class CSTNode:
    """Composite Concrete Syntax Tree node with child tokens or subnodes."""

    kind: str
    children: List[Any]
    span: SourceSpan


def _is_trivia(item: Any) -> bool:
    """Check if a CST item is whitespace or comment trivia."""
    if isinstance(item, Trivia):
        return True
    if isinstance(item, CSTToken):
        kind_str = item.kind.name if isinstance(item.kind, Enum) else str(item.kind)
        return kind_str.upper() in (
            "WHITESPACE",
            "COMMENT",
            "TRIVIA",
            "NEWLINE",
            "LINE_COMMENT",
            "BLOCK_COMMENT",
        )
    return False


def _is_punctuation(item: Any) -> bool:
    """Check if a CST token is syntax punctuation to be stripped in expressions."""
    if isinstance(item, CSTToken):
        kind_str = item.kind.name if isinstance(item.kind, Enum) else str(item.kind)
        return (
            kind_str.upper() in ("LPAREN", "RPAREN", "COMMA", "SEMICOLON", "COLON")
            or item.value in ("(", ")", ",", ";", ":")
        )
    return False


def _is_parenthesis(item: Any) -> bool:
    """Check if a CST token is an enclosing parenthesis."""
    if isinstance(item, CSTToken):
        kind_str = item.kind.name if isinstance(item.kind, Enum) else str(item.kind)
        return kind_str.upper() in ("LPAREN", "RPAREN") or item.value in ("(", ")")
    return False


def _parse_literal_value(kind: str, raw_value: str) -> Any:
    """Convert a literal token's raw string into a normalized Python primitive."""
    kind_upper = kind.upper()
    if kind_upper in ("INTEGER", "INT"):
        return int(raw_value)
    if kind_upper in ("FLOAT", "DOUBLE"):
        return float(raw_value)
    if kind_upper in ("BOOLEAN", "BOOL"):
        lower = raw_value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        raise ASTGenerationError(f"Invalid boolean literal: {raw_value}")
    if kind_upper == "STRING":
        if (
            len(raw_value) >= 2
            and raw_value[0] in ('"', "'")
            and raw_value[-1] == raw_value[0]
        ):
            return raw_value[1:-1]
        return raw_value

    if raw_value.lower() == "true":
        return True
    if raw_value.lower() == "false":
        return False
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    if (
        len(raw_value) >= 2
        and raw_value[0] in ('"', "'")
        and raw_value[-1] == raw_value[0]
    ):
        return raw_value[1:-1]
    return raw_value


class ASTGenerator:
    """Transforms a Concrete Syntax Tree (CST) into a normalized Abstract Syntax Tree (AST)."""

    def generate(self, node: Any) -> ASTNode:
        """Process a CST node or token and return the normalized AST node."""
        if node is None:
            raise TypeError("CST root node cannot be None")
        if isinstance(node, str) or not hasattr(node, "kind") or not hasattr(node, "span"):
            raise TypeError(f"Expected CST node or token, got {type(node).__name__}")

        if isinstance(node, CSTToken):
            return self._generate_from_token(node)

        kind = node.kind.lower()
        if kind in ("literal", "literal_expr"):
            return self._generate_literal(node)
        elif kind in ("identifier", "identifier_expr", "ident"):
            return self._generate_identifier(node)
        elif kind in ("binary_expr", "binary", "bin_expr"):
            return self._generate_binary(node)
        elif kind in ("unary_expr", "unary"):
            return self._generate_unary(node)
        elif kind in ("paren_expr", "parenthesized_expr", "group_expr"):
            return self._generate_paren(node)
        elif kind in ("call_expr", "call"):
            return self._generate_call(node)
        else:
            raise ASTGenerationError(f"Unrecognized CST node kind: {node.kind}")

    def _generate_from_token(self, token: CSTToken) -> ASTNode:
        kind_upper = (
            token.kind.upper() if isinstance(token.kind, str) else str(token.kind).upper()
        )
        if kind_upper == "IDENTIFIER":
            return IdentifierExpr(name=token.value, span=token.span)
        elif kind_upper in ("INTEGER", "INT", "FLOAT", "DOUBLE", "BOOLEAN", "BOOL", "STRING"):
            val = _parse_literal_value(token.kind, token.value)
            return LiteralExpr(value=val, span=token.span, raw=token.value)
        else:
            raise ASTGenerationError(
                f"Cannot generate standalone AST from token kind: {token.kind}"
            )

    def _generate_literal(self, node: CSTNode) -> LiteralExpr:
        non_trivia = [c for c in node.children if not _is_trivia(c)]
        if not non_trivia:
            raise ASTGenerationError("Literal CST node has no token children")
        tok = non_trivia[0]
        if isinstance(tok, CSTToken):
            val = _parse_literal_value(tok.kind, tok.value)
            return LiteralExpr(value=val, span=node.span, raw=tok.value)
        elif isinstance(tok, CSTNode):
            result = self.generate(tok)
            if isinstance(result, LiteralExpr):
                return result
            raise ASTGenerationError(f"Expected literal inner node, got {type(result).__name__}")
        else:
            return LiteralExpr(value=tok, span=node.span, raw=str(tok))

    def _generate_identifier(self, node: CSTNode) -> IdentifierExpr:
        non_trivia = [c for c in node.children if not _is_trivia(c)]
        if not non_trivia:
            raise ASTGenerationError("Identifier CST node has no token children")
        tok = non_trivia[0]
        name = tok.value if hasattr(tok, "value") else str(tok)
        return IdentifierExpr(name=name, span=node.span)

    def _generate_binary(self, node: CSTNode) -> BinaryExpr:
        non_trivia = [c for c in node.children if not _is_trivia(c)]
        if len(non_trivia) < 3 or len(non_trivia) % 2 == 0:
            raise ASTGenerationError(
                "Binary expression requires left operand, operator, and right operand; "
                f"got {len(non_trivia)} children"
            )

        left = self.generate(non_trivia[0])
        for i in range(1, len(non_trivia), 2):
            op_item = non_trivia[i]
            op_str = op_item.value if hasattr(op_item, "value") else str(op_item)
            right = self.generate(non_trivia[i + 1])
            left = BinaryExpr(left=left, operator=op_str, right=right, span=node.span)
        return left

    def _generate_unary(self, node: CSTNode) -> UnaryExpr:
        non_trivia = [c for c in node.children if not _is_trivia(c)]
        if len(non_trivia) < 2:
            raise ASTGenerationError(
                f"Unary expression requires operator and operand; got {len(non_trivia)} children"
            )
        op_item = non_trivia[0]
        op_str = op_item.value if hasattr(op_item, "value") else str(op_item)
        operand = self.generate(non_trivia[1])
        return UnaryExpr(operator=op_str, operand=operand, span=node.span)

    def _generate_paren(self, node: CSTNode) -> ASTNode:
        non_paren = [
            c for c in node.children if not _is_trivia(c) and not _is_parenthesis(c)
        ]
        if len(non_paren) != 1:
            raise ASTGenerationError(
                f"Parenthesized expression expects exactly 1 inner expression, got {len(non_paren)}"
            )
        inner_ast = self.generate(non_paren[0])
        inner_ast.span = node.span
        return inner_ast

    def _generate_call(self, node: CSTNode) -> CallExpr:
        non_trivia = [c for c in node.children if not _is_trivia(c)]
        if not non_trivia:
            raise ASTGenerationError("Call expression missing callee and arguments")

        first = non_trivia[0]
        if isinstance(first, CSTToken) and (
            (isinstance(first.kind, Enum) and first.kind.name.upper() == "LPAREN")
            or str(first.kind).upper() in ("LPAREN", "RPAREN", "COMMA")
            or first.value in ("(", ")", ",")
        ):
            raise ASTGenerationError("Call expression missing callee")

        callee = self.generate(first)
        raw_args = [c for c in non_trivia[1:] if not _is_punctuation(c)]
        arguments = [self.generate(arg) for arg in raw_args]
        return CallExpr(callee=callee, arguments=arguments, span=node.span)


def generate_ast(cst: Any) -> ASTNode:
    """Convenience function to generate an Abstract Syntax Tree (AST) from a CST."""
    return ASTGenerator().generate(cst)


__all__ = [
    "ASTGenerationError",
    "ASTGenerator",
    "CSTNode",
    "CSTToken",
    "Trivia",
    "TriviaKind",
    "generate_ast",
]