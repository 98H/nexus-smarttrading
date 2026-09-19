import pytest

from pine_lexer.lexer import Lexer, LexerError, tokenize
from pine_lexer.tokens import Token, TokenType


class TestPineScriptLexerKeywordsAndIdentifiers:
    """Tests for Pine Script v5 keywords, identifiers, and general token ordering."""

    @pytest.mark.parametrize(
        ("keyword", "expected_type"),
        [
            ("indicator", TokenType.KEYWORD),
            ("strategy", TokenType.KEYWORD),
            ("library", TokenType.KEYWORD),
            ("var", TokenType.KEYWORD),
            ("varip", TokenType.KEYWORD),
            ("if", TokenType.KEYWORD),
            ("else", TokenType.KEYWORD),
            ("for", TokenType.KEYWORD),
            ("to", TokenType.KEYWORD),
            ("by", TokenType.KEYWORD),
            ("while", TokenType.KEYWORD),
            ("switch", TokenType.KEYWORD),
            ("returns", TokenType.KEYWORD),
            ("true", TokenType.BOOLEAN),
            ("false", TokenType.BOOLEAN),
        ],
    )
    def test_keywords_recognized(self, keyword: str, expected_type: TokenType):
        tokens = tokenize(keyword)
        # Filter out EOF if the lexer produces an explicit EOF token
        semantic_tokens = [t for t in tokens if t.type != TokenType.EOF]

        assert len(semantic_tokens) == 1
        assert semantic_tokens[0].type == expected_type
        assert semantic_tokens[0].value == keyword
        assert semantic_tokens[0].line == 1
        assert semantic_tokens[0].column == 1

    @pytest.mark.parametrize(
        "ident",
        [
            ("my_var"),
            ("_hidden"),
            ("var123"),
            ("indicator_val"),
            ("smaLength"),
            ("ta"),
        ],
    )
    def test_identifiers_recognized(self, ident: str):
        tokens = tokenize(ident)
        semantic_tokens = [t for t in tokens if t.type != TokenType.EOF]

        assert len(semantic_tokens) == 1
        assert semantic_tokens[0].type == TokenType.IDENTIFIER
        assert semantic_tokens[0].value == ident
        assert semantic_tokens[0].line == 1
        assert semantic_tokens[0].column == 1

    def test_ordered_stream_keyword_identifier_assignment(self):
        code = "var int length = 14"
        tokens = [t for t in tokenize(code) if t.type != TokenType.EOF]

        expected = [
            (TokenType.KEYWORD, "var", 1, 1),
            (TokenType.KEYWORD, "int", 1, 5),
            (TokenType.IDENTIFIER, "length", 1, 9),
            (TokenType.OPERATOR, "=", 1, 16),
            (TokenType.INTEGER, "14", 1, 18),
        ]

        assert len(tokens) == len(expected)
        for actual, (exp_type, exp_val, exp_line, exp_col) in zip(
            tokens, expected, strict=True
        ):
            assert actual.type == exp_type
            assert actual.value == exp_val
            assert actual.line == exp_line
            assert actual.column == exp_col


class TestPineScriptLexerLiterals:
    """Tests for Pine Script v5 numeric, string, and boolean literals."""

    @pytest.mark.parametrize(
        ("code", "expected_type", "expected_val"),
        [
            ("0", TokenType.INTEGER, "0"),
            ("42", TokenType.INTEGER, "42"),
            ("1000000", TokenType.INTEGER, "1000000"),
            ("3.14159", TokenType.FLOAT, "3.14159"),
            (".5", TokenType.FLOAT, ".5"),
            ("10.", TokenType.FLOAT, "10."),
            ("1e-5", TokenType.FLOAT, "1e-5"),
            ("2.5e3", TokenType.FLOAT, "2.5e3"),
            ("1E+6", TokenType.FLOAT, "1E+6"),
        ],
    )
    def test_numeric_literals(
        self, code: str, expected_type: TokenType, expected_val: str
    ):
        tokens = [t for t in tokenize(code) if t.type != TokenType.EOF]

        assert len(tokens) == 1
        token = tokens[0]
        assert token.type == expected_type
        assert token.value == expected_val
        assert token.line == 1
        assert token.column == 1

    @pytest.mark.parametrize(
        ("code", "expected_val"),
        [
            ('"Hello Pine"', '"Hello Pine"'),
            ("'Pine Script'", "'Pine Script'"),
            ('"Escaped \\"quote\\""', '"Escaped \\"quote\\""'),
            ("''", "''"),
            ('""', '""'),
        ],
    )
    def test_string_literals(self, code: str, expected_val: str):
        tokens = [t for t in tokenize(code) if t.type != TokenType.EOF]

        assert len(tokens) == 1
        token = tokens[0]
        assert token.type == TokenType.STRING
        assert token.value == expected_val
        assert token.line == 1
        assert token.column == 1


class TestPineScriptLexerOperatorsAndDelimiters:
    """Tests for Pine Script v5 operators and punctuation."""

    @pytest.mark.parametrize(
        ("op", "expected_val"),
        [
            (":=", ":="),
            ("+=", "+="),
            ("-=", "-="),
            ("*=", "*="),
            ("/=", "/="),
            ("%=", "%="),
            ("==", "=="),
            ("!=", "!="),
            ("<=", "<="),
            (">=", ">="),
            ("=>", "=>"),
            ("?", "?"),
            (":", ":"),
            ("+", "+"),
            ("-", "-"),
            ("*", "*"),
            ("/", "/"),
            ("%", "%"),
            ("<", "<"),
            (">", ">"),
            ("=", "="),
        ],
    )
    def test_operators(self, op: str, expected_val: str):
        tokens = [t for t in tokenize(op) if t.type != TokenType.EOF]

        assert len(tokens) == 1
        assert tokens[0].type == TokenType.OPERATOR
        assert tokens[0].value == expected_val

    @pytest.mark.parametrize(
        ("delim", "expected_type"),
        [
            ("(", TokenType.LPAREN),
            (")", TokenType.RPAREN),
            ("[", TokenType.LBRACKET),
            ("]", TokenType.RBRACKET),
            ("{", TokenType.LBRACE),
            ("}", TokenType.RBRACE),
            (",", TokenType.COMMA),
            (".", TokenType.DOT),
        ],
    )
    def test_delimiters(self, delim: str, expected_type: TokenType):
        tokens = [t for t in tokenize(delim) if t.type != TokenType.EOF]

        assert len(tokens) == 1
        assert tokens[0].type == expected_type
        assert tokens[0].value == delim


class TestPineScriptLexerCommentsAndWhitespace:
    """Tests for handling comments (//) and whitespace without corrupting token sequences."""

    def test_standalone_comment_line_stripped_or_tagged(self):
        code = "// This is a full-line comment\nindicator('Test')"
        tokens = tokenize(code)

        # Comments may either be preserved as TokenType.COMMENT or stripped.
        # If preserved, they must have valid line and column positions.
        semantic_tokens = [
            t
            for t in tokens
            if t.type not in (TokenType.EOF, TokenType.WHITESPACE, TokenType.NEWLINE)
        ]

        if any(t.type == TokenType.COMMENT for t in semantic_tokens):
            comment_token = semantic_tokens[0]
            assert comment_token.type == TokenType.COMMENT
            assert comment_token.value == "// This is a full-line comment"
            assert comment_token.line == 1
            assert comment_token.column == 1
            semantic_tokens = semantic_tokens[1:]

        assert len(semantic_tokens) == 4
        assert semantic_tokens[0].type == TokenType.KEYWORD
        assert semantic_tokens[0].value == "indicator"
        assert semantic_tokens[0].line == 2
        assert semantic_tokens[0].column == 1

        assert semantic_tokens[1].type == TokenType.LPAREN
        assert semantic_tokens[2].type == TokenType.STRING
        assert semantic_tokens[2].value == "'Test'"
        assert semantic_tokens[3].type == TokenType.RPAREN

    def test_trailing_comment_and_trailing_whitespace(self):
        code = "var x = 10 // inline comment   \t\nvar y = 20"
        tokens = tokenize(code)

        semantic_tokens = [
            t
            for t in tokens
            if t.type not in (TokenType.EOF, TokenType.WHITESPACE, TokenType.NEWLINE)
        ]

        # Filter out comments if tagged
        code_tokens = [t for t in semantic_tokens if t.type != TokenType.COMMENT]

        expected = [
            (TokenType.KEYWORD, "var", 1, 1),
            (TokenType.IDENTIFIER, "x", 1, 5),
            (TokenType.OPERATOR, "=", 1, 7),
            (TokenType.INTEGER, "10", 1, 9),
            (TokenType.KEYWORD, "var", 2, 1),
            (TokenType.IDENTIFIER, "y", 2, 5),
            (TokenType.OPERATOR, "=", 2, 7),
            (TokenType.INTEGER, "20", 2, 9),
        ]

        assert len(code_tokens) == len(expected)
        for actual, (exp_type, exp_val, exp_line, exp_col) in zip(
            code_tokens, expected, strict=True
        ):
            assert actual.type == exp_type
            assert actual.value == exp_val
            assert actual.line == exp_line
            assert actual.column == exp_col

    def test_whitespace_only_code(self):
        code = "    \t  \n   \t  \n  "
        tokens = tokenize(code)
        non_trivia = [
            t
            for t in tokens
            if t.type
            not in (
                TokenType.EOF,
                TokenType.WHITESPACE,
                TokenType.NEWLINE,
                TokenType.COMMENT,
            )
        ]
        assert len(non_trivia) == 0

    def test_empty_string_produces_empty_or_eof_only(self):
        tokens = tokenize("")
        semantic_tokens = [t for t in tokens if t.type != TokenType.EOF]
        assert len(semantic_tokens) == 0


class TestPineScriptLexerPositionsAndErrors:
    """Tests for accurate line/column tracking and LexerError handling."""

    def test_multiline_position_tracking(self):
        code = (
            "//@version=5\n"
            "indicator('RSI')\n"
            "len = 14\n"
            "src = close"
        )
        tokens = tokenize(code)
        code_tokens = [
            t
            for t in tokens
            if t.type
            not in (
                TokenType.EOF,
                TokenType.COMMENT,
                TokenType.WHITESPACE,
                TokenType.NEWLINE,
            )
        ]

        expected = [
            (TokenType.KEYWORD, "indicator", 2, 1),
            (TokenType.LPAREN, "(", 2, 10),
            (TokenType.STRING, "'RSI'", 2, 11),
            (TokenType.RPAREN, ")", 2, 16),
            (TokenType.IDENTIFIER, "len", 3, 1),
            (TokenType.OPERATOR, "=", 3, 5),
            (TokenType.INTEGER, "14", 3, 7),
            (TokenType.IDENTIFIER, "src", 4, 1),
            (TokenType.OPERATOR, "=", 4, 5),
            (TokenType.IDENTIFIER, "close", 4, 7),
        ]

        assert len(code_tokens) == len(expected)
        for actual, (exp_type, exp_val, exp_line, exp_col) in zip(
            code_tokens, expected, strict=True
        ):
            assert actual.type == exp_type
            assert actual.value == exp_val
            assert actual.line == exp_line
            assert actual.column == exp_col

    @pytest.mark.parametrize(
        ("code", "err_line", "err_col"),
        [
            ("§", 1, 1),
            ("var x = 1\nvar y = @", 2, 9),
            ("   `", 1, 4),
            ("indicator('test')\n\n\n   ^", 4, 4),
        ],
    )
    def test_unrecognized_character_raises_lexer_error(
        self, code: str, err_line: int, err_col: int
    ):
        with pytest.raises(LexerError) as exc_info:
            tokenize(code)

        err = exc_info.value
        assert err.line == err_line
        assert err.column == err_col

    def test_lexer_class_instance_reusability(self):
        lexer = Lexer()
        tokens1 = [t for t in lexer.tokenize("var x = 1") if t.type != TokenType.EOF]
        tokens2 = [t for t in lexer.tokenize("var y = 2") if t.type != TokenType.EOF]

        assert len(tokens1) == 4
        assert tokens1[1].value == "x"
        assert len(tokens2) == 4
        assert tokens2[1].value == "y"
        assert tokens2[0].line == 1
        assert tokens2[0].column == 1