"""Lexical analyzer for Pine Script v5 source code."""

from typing import Final

from pine_lexer.tokens import Token, TokenType

KEYWORDS: Final[set[str]] = {
    "indicator",
    "strategy",
    "library",
    "var",
    "varip",
    "if",
    "else",
    "for",
    "to",
    "by",
    "while",
    "switch",
    "returns",
    "break",
    "continue",
    "import",
    "export",
    "type",
    "method",
    "int",
    "float",
    "bool",
    "string",
    "color",
    "line",
    "linefill",
    "label",
    "box",
    "table",
    "series",
    "simple",
    "const",
    "input",
    "and",
    "or",
    "not",
}

BOOLEANS: Final[set[str]] = {
    "true",
    "false",
}

TWO_CHAR_OPERATORS: Final[set[str]] = {
    ":=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "==",
    "!=",
    "<=",
    ">=",
    "=>",
}

SINGLE_CHAR_OPERATORS: Final[set[str]] = {
    "?",
    ":",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "=",
}

DELIMITERS: Final[dict[str, TokenType]] = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
}


class LexerError(Exception):
    """Raised when the lexer encounters an invalid or unrecognized character sequence."""

    def __init__(self, message: str, line: int = 1, column: int = 1) -> None:
        super().__init__(f"{message} at line {line}, column {column}")
        self.message = message
        self.line = line
        self.column = column


class Lexer:
    """Stateful tokenizer for Pine Script v5."""

    def __init__(self) -> None:
        self._source: str = ""
        self._pos: int = 0
        self._line: int = 1
        self._column: int = 1

    def tokenize(self, source: str) -> list[Token]:
        """Tokenize a Pine Script v5 source code string into a list of Tokens."""
        self._source = source
        self._pos = 0
        self._line = 1
        self._column = 1

        tokens: list[Token] = []
        while not self._is_at_end():
            token = self._next_token()
            if token is not None:
                tokens.append(token)

        tokens.append(Token(TokenType.EOF, "", self._line, self._column))
        return tokens

    def _is_at_end(self, offset: int = 0) -> bool:
        return self._pos + offset >= len(self._source)

    def _peek(self, offset: int = 0) -> str:
        idx = self._pos + offset
        if idx >= len(self._source):
            return ""
        return self._source[idx]

    def _advance(self) -> str:
        ch = self._source[self._pos]
        self._pos += 1
        self._column += 1
        return ch

    def _skip_whitespace(self) -> None:
        while not self._is_at_end():
            ch = self._peek()
            if ch in (" ", "\t"):
                self._advance()
            elif ch == "\r":
                if self._peek(1) == "\n":
                    self._pos += 2
                else:
                    self._pos += 1
                self._line += 1
                self._column = 1
            elif ch == "\n":
                self._pos += 1
                self._line += 1
                self._column = 1
            else:
                break

    def _next_token(self) -> Token | None:
        self._skip_whitespace()
        if self._is_at_end():
            return None

        ch = self._peek()

        # Comment
        if ch == "/" and self._peek(1) == "/":
            return self._lex_comment()

        # String literal
        if ch in ('"', "'"):
            return self._lex_string()

        # Numeric literal starting with dot (e.g. .5)
        if ch == "." and self._peek(1).isdigit():
            return self._lex_number()

        # Numeric literal starting with a digit
        if ch.isdigit():
            return self._lex_number()

        # Identifier or Keyword
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ch == "_":
            return self._lex_identifier_or_keyword()

        # Multi-character operators
        two_char = ch + self._peek(1)
        if two_char in TWO_CHAR_OPERATORS:
            line = self._line
            col = self._column
            self._advance()
            self._advance()
            return Token(TokenType.OPERATOR, two_char, line, col)

        # Delimiters
        if ch in DELIMITERS:
            line = self._line
            col = self._column
            self._advance()
            return Token(DELIMITERS[ch], ch, line, col)

        # Single-character operators
        if ch in SINGLE_CHAR_OPERATORS:
            line = self._line
            col = self._column
            self._advance()
            return Token(TokenType.OPERATOR, ch, line, col)

        # Unrecognized character
        err_line = self._line
        err_col = self._column
        raise LexerError(
            f"Unexpected character: {ch!r}",
            line=err_line,
            column=err_col,
        )

    def _lex_comment(self) -> Token:
        start_line = self._line
        start_col = self._column
        start_pos = self._pos

        self._advance()  # consume first '/'
        self._advance()  # consume second '/'

        while not self._is_at_end() and self._peek() not in ("\r", "\n"):
            self._advance()

        val = self._source[start_pos : self._pos]
        return Token(TokenType.COMMENT, val, start_line, start_col)

    def _lex_string(self) -> Token:
        start_line = self._line
        start_col = self._column
        start_pos = self._pos
        quote = self._peek()
        self._advance()  # consume opening quote

        while not self._is_at_end():
            ch = self._peek()
            if ch == "\\":
                self._advance()  # consume backslash
                if not self._is_at_end():
                    if self._peek() == "\n":
                        self._line += 1
                        self._column = 1
                        self._pos += 1
                    else:
                        self._advance()
            elif ch == quote:
                self._advance()  # consume closing quote
                val = self._source[start_pos : self._pos]
                return Token(TokenType.STRING, val, start_line, start_col)
            elif ch in ("\r", "\n"):
                raise LexerError(
                    "Unterminated string literal",
                    line=start_line,
                    column=start_col,
                )
            else:
                self._advance()

        raise LexerError(
            "Unterminated string literal",
            line=start_line,
            column=start_col,
        )

    def _has_exponent_ahead(self) -> bool:
        if self._peek(1).isdigit():
            return True
        if self._peek(1) in ("+", "-") and self._peek(2).isdigit():
            return True
        return False

    def _lex_number(self) -> Token:
        start_line = self._line
        start_col = self._column
        start_pos = self._pos

        if self._peek() == ".":
            self._advance()  # consume '.'
            while self._peek().isdigit():
                self._advance()
            if self._peek() in ("e", "E") and self._has_exponent_ahead():
                self._advance()
                if self._peek() in ("+", "-"):
                    self._advance()
                while self._peek().isdigit():
                    self._advance()
            val = self._source[start_pos : self._pos]
            return Token(TokenType.FLOAT, val, start_line, start_col)

        while self._peek().isdigit():
            self._advance()

        is_float = False
        if self._peek() == "." and self._peek(1) != ".":
            is_float = True
            self._advance()  # consume '.'
            while self._peek().isdigit():
                self._advance()

        if self._peek() in ("e", "E") and self._has_exponent_ahead():
            is_float = True
            self._advance()
            if self._peek() in ("+", "-"):
                self._advance()
            while self._peek().isdigit():
                self._advance()

        val = self._source[start_pos : self._pos]
        token_type = TokenType.FLOAT if is_float else TokenType.INTEGER
        return Token(token_type, val, start_line, start_col)

    def _lex_identifier_or_keyword(self) -> Token:
        start_line = self._line
        start_col = self._column
        start_pos = self._pos

        while not self._is_at_end():
            ch = self._peek()
            if (
                ("a" <= ch <= "z")
                or ("A" <= ch <= "Z")
                or ("0" <= ch <= "9")
                or ch == "_"
            ):
                self._advance()
            else:
                break

        val = self._source[start_pos : self._pos]
        if val in BOOLEANS:
            tok_type = TokenType.BOOLEAN
        elif val in KEYWORDS:
            tok_type = TokenType.KEYWORD
        else:
            tok_type = TokenType.IDENTIFIER

        return Token(tok_type, val, start_line, start_col)


def tokenize(code: str) -> list[Token]:
    """Tokenize a Pine Script v5 source code string into a list of Tokens."""
    return Lexer().tokenize(code)