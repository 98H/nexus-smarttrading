"""Token definitions for Pine Script v5 lexer."""

from dataclasses import dataclass
from enum import Enum


class TokenType(str, Enum):
    """Enumeration of Pine Script v5 token types."""

    # Special tokens
    EOF = "EOF"
    COMMENT = "COMMENT"
    WHITESPACE = "WHITESPACE"
    NEWLINE = "NEWLINE"

    # Identifiers and keywords
    IDENTIFIER = "IDENTIFIER"
    KEYWORD = "KEYWORD"
    BOOLEAN = "BOOLEAN"

    # Literals
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"

    # Operators
    OPERATOR = "OPERATOR"

    # Delimiters / Punctuation
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    DOT = "DOT"


@dataclass(frozen=True)
class Token:
    """Represents a lexical token with its position in the source code."""

    type: TokenType
    value: str
    line: int
    column: int