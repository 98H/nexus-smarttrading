"""
Unit tests for Pine Script Language Server Protocol (LSP).

Tests cover:
- Pine Script built-in catalog: symbol lookup, prefix querying, signature and Markdown documentation.
- LSP Handlers: token/prefix extraction at position, completion items generation, hover responses.
- LSP Server: document buffer management (didOpen, didChange), handling 'textDocument/completion'
  and 'textDocument/hover' JSON-RPC requests conforming to LSP specifications.
"""

import pytest

from src.lsp.catalog import PineScriptCatalog, PineSymbol
from src.lsp.handlers import (
    extract_identifier_at_position,
    extract_prefix_at_position,
    handle_completion,
    handle_hover,
)
from src.lsp.server import PineScriptLSPServer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def catalog() -> PineScriptCatalog:
    """Fixture providing a populated Pine Script catalog instance."""
    return PineScriptCatalog()


@pytest.fixture
def lsp_server() -> PineScriptLSPServer:
    """Fixture providing an initialized Pine Script LSP server instance."""
    return PineScriptLSPServer()


# ============================================================================
# Catalog Unit Tests: src/lsp/catalog.py
# ============================================================================

class TestPineScriptCatalog:
    """Tests for Pine Script built-in catalog data store and lookup logic."""

    def test_catalog_contains_standard_builtin_functions(self, catalog: PineScriptCatalog):
        """Verify standard Pine Script functions exist with appropriate metadata."""
        symbol = catalog.get_symbol("ta.sma")
        assert symbol is not None
        assert isinstance(symbol, PineSymbol)
        assert symbol.name == "ta.sma"
        assert symbol.kind in ("function", "method")
        assert "ta.sma" in symbol.signature
        assert len(symbol.detail) > 0
        assert "```pine" in symbol.documentation or "**" in symbol.documentation

    def test_catalog_contains_standard_builtin_variables(self, catalog: PineScriptCatalog):
        """Verify built-in variables like close, open, volume are registered."""
        close_sym = catalog.get_symbol("close")
        assert close_sym is not None
        assert close_sym.name == "close"
        assert close_sym.kind == "variable"
        assert "series float" in close_sym.detail or "series float" in close_sym.signature
        assert len(close_sym.documentation) > 0

    def test_catalog_contains_standard_keywords(self, catalog: PineScriptCatalog):
        """Verify Pine Script keywords like 'strategy', 'var', 'if' exist."""
        strategy_sym = catalog.get_symbol("strategy")
        assert strategy_sym is not None
        assert strategy_sym.name == "strategy"
        assert strategy_sym.kind == "keyword"
        assert len(strategy_sym.documentation) > 0
        assert len(strategy_sym.signature) > 0

    def test_catalog_prefix_search_builtins(self, catalog: PineScriptCatalog):
        """Verify prefix search returns all matching built-in symbols."""
        results = catalog.find_completions("ta.s")
        result_names = [item.name for item in results]

        assert "ta.sma" in result_names
        assert "ta.stoch" in result_names
        for item in results:
            assert item.name.startswith("ta.s")

    def test_catalog_prefix_search_namespace(self, catalog: PineScriptCatalog):
        """Verify searching for a namespace prefix returns all members."""
        results = catalog.find_completions("ta.")
        result_names = [item.name for item in results]

        assert "ta.sma" in result_names
        assert "ta.ema" in result_names
        assert "ta.rsi" in result_names
        for item in results:
            assert item.name.startswith("ta.")

    def test_catalog_prefix_search_empty_or_mismatch(self, catalog: PineScriptCatalog):
        """Verify querying non-existent prefix returns an empty list."""
        results = catalog.find_completions("non_existent_prefix_xyz_")
        assert results == []

    def test_catalog_case_sensitivity(self, catalog: PineScriptCatalog):
        """Verify Pine Script catalog preserves case-sensitivity."""
        assert catalog.get_symbol("ta.sma") is not None
        assert catalog.get_symbol("TA.SMA") is None

    def test_catalog_documentation_markdown_specification(self, catalog: PineScriptCatalog):
        """Verify documentation includes Markdown formatting and signature specification."""
        symbol = catalog.get_symbol("ta.rsi")
        assert symbol is not None
        # Must contain Markdown code block or bold headers
        assert "```pine" in symbol.documentation
        assert symbol.signature in symbol.documentation or len(symbol.signature) > 0


# ============================================================================
# Handler Unit Tests: src/lsp/handlers.py
# ============================================================================

class TestLSPHandlers:
    """Tests for LSP request handler helpers and position extraction logic."""

    def test_extract_identifier_at_position_simple(self):
        """Extract a simple standalone identifier at cursor position."""
        line = "val = close + 10"
        # Character 7 is inside 'close'
        identifier = extract_identifier_at_position(line, character=7)
        assert identifier == "close"

    def test_extract_identifier_at_position_dotted_namespace(self):
        """Extract a dotted namespace identifier like ta.sma at cursor position."""
        line = "fast = ta.sma(close, 14)"
        # Positions 7 to 12 are part of 'ta.sma'
        assert extract_identifier_at_position(line, character=7) == "ta.sma"
        assert extract_identifier_at_position(line, character=9) == "ta.sma"
        assert extract_identifier_at_position(line, character=12) == "ta.sma"

    def test_extract_identifier_at_position_whitespace_or_punctuation(self):
        """Extract identifier on whitespace or parenthesis should return empty string."""
        line = "fast = ta.sma(close, 14)"
        assert extract_identifier_at_position(line, character=4) == ""  # space
        assert extract_identifier_at_position(line, character=13) == ""  # '('
        assert extract_identifier_at_position(line, character=20) == ""  # ','

    def test_extract_prefix_at_position_typing_dotted(self):
        """Extract prefix preceding cursor when typing a function call or namespace."""
        line = "result = ta.sm"
        # Cursor right after 'sm' (character 14)
        prefix = extract_prefix_at_position(line, character=14)
        assert prefix == "ta.sm"

    def test_extract_prefix_at_position_trailing_dot(self):
        """Extract prefix right after dot operator."""
        line = "indicator = ta."
        # Cursor right after 'ta.' (character 15)
        prefix = extract_prefix_at_position(line, character=15)
        assert prefix == "ta."

    def test_handle_completion_returns_matching_items_with_labels_and_details(
        self, catalog: PineScriptCatalog
    ):
        """Verify handle_completion returns items with label, detail, and kind."""
        document_text = "x = ta.s"
        # Cursor at line 0, character 8 (end of 'ta.s')
        items = handle_completion(catalog, text=document_text, line=0, character=8)

        assert isinstance(items, list)
        assert len(items) > 0
        for item in items:
            assert "label" in item
            assert "detail" in item
            assert item["label"].startswith("ta.s")
            assert len(item["detail"]) > 0

    def test_handle_completion_no_match_returns_empty(self, catalog: PineScriptCatalog):
        """Verify handle_completion returns empty list when prefix does not match."""
        document_text = "x = unk."
        items = handle_completion(catalog, text=document_text, line=0, character=8)
        assert items == []

    def test_handle_hover_builtin_function(self, catalog: PineScriptCatalog):
        """Verify handle_hover returns Markdown payload and signature for built-in functions."""
        document_text = "//@version=5\nplot(ta.sma(close, 14))"
        # Cursor inside 'ta.sma' on line 1, character 8
        hover = handle_hover(catalog, text=document_text, line=1, character=8)

        assert hover is not None
        assert "contents" in hover
        contents = hover["contents"]
        assert isinstance(contents, dict)
        assert contents.get("kind") == "markdown"
        assert "ta.sma" in contents.get("value", "")
        assert "```pine" in contents.get("value", "")

    def test_handle_hover_builtin_keyword(self, catalog: PineScriptCatalog):
        """Verify handle_hover returns Markdown documentation for keywords."""
        document_text = "//@version=5\nstrategy('Test', overlay=true)"
        # Cursor inside 'strategy' on line 1, character 3
        hover = handle_hover(catalog, text=document_text, line=1, character=3)

        assert hover is not None
        assert "contents" in hover
        contents = hover["contents"]
        assert contents.get("kind") == "markdown"
        assert "strategy" in contents.get("value", "")

    def test_handle_hover_unknown_symbol_returns_none(self, catalog: PineScriptCatalog):
        """Verify handle_hover returns None for unknown or user-defined symbols."""
        document_text = "my_custom_var = 42\nplot(my_custom_var)"
        # Cursor inside 'my_custom_var' on line 0, character 5
        hover = handle_hover(catalog, text=document_text, line=0, character=5)
        assert hover is None

    def test_handle_hover_on_whitespace_returns_none(self, catalog: PineScriptCatalog):
        """Verify handle_hover returns None when hovering over empty space."""
        document_text = "//@version=5\n   \nplot(close)"
        hover = handle_hover(catalog, text=document_text, line=1, character=1)
        assert hover is None


# ============================================================================
# Server Integration Unit Tests: src/lsp/server.py
# ============================================================================

class TestPineScriptLSPServer:
    """Tests for PineScriptLSPServer processing LSP requests and buffer states."""

    DOC_URI = "file:///workspace/script.pine"

    def test_server_buffer_did_open_and_did_change(self, lsp_server: PineScriptLSPServer):
        """Verify document buffer synchronization through didOpen and didChange."""
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="//@version=5\nindicator('Demo')\nplot(close)",
            version=1,
        )
        assert lsp_server.get_document_text(self.DOC_URI) == "//@version=5\nindicator('Demo')\nplot(close)"

        lsp_server.did_change(
            uri=self.DOC_URI,
            text="//@version=5\nindicator('Demo')\nplot(ta.ema(close, 20))",
            version=2,
        )
        assert "ta.ema" in lsp_server.get_document_text(self.DOC_URI)

    def test_completion_request_matching_builtin_identifiers(self, lsp_server: PineScriptLSPServer):
        """
        Acceptance Criteria 1:
        Given a client connected to the Pine Script LSP server,
        When the client sends a `textDocument/completion` request with a prefix
        matching Pine Script built-in identifiers,
        Then the server returns a list of matching completion items with labels
        and detail information.
        """
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="//@version=5\nindicator('Test')\ns = ta.sm",
            version=1,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 2, "character": 9},  # cursor after 'ta.sm'
            },
        }

        response = lsp_server.handle_message(request)

        assert response is not None
        assert response.get("jsonrpc") == "2.0"
        assert response.get("id") == 1
        assert "result" in response

        results = response["result"]
        assert isinstance(results, list)
        assert len(results) > 0

        # Validate that items have both label and detail information
        for item in results:
            assert "label" in item
            assert "detail" in item
            assert isinstance(item["label"], str)
            assert isinstance(item["detail"], str)
            assert len(item["label"]) > 0
            assert len(item["detail"]) > 0
            assert item["label"].startswith("ta.sm")

    def test_hover_request_on_builtin_function(self, lsp_server: PineScriptLSPServer):
        """
        Acceptance Criteria 2:
        Given an active document buffer in the Pine Script LSP server,
        When the client sends a `textDocument/hover` request at a position
        occupied by a Pine Script built-in function or keyword,
        Then the server returns a hover payload containing Markdown-formatted
        documentation and signature specifications.
        """
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="//@version=5\nindicator('Test')\nval = ta.sma(close, 20)",
            version=1,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 2, "character": 9},  # inside 'ta.sma'
            },
        }

        response = lsp_server.handle_message(request)

        assert response is not None
        assert response.get("jsonrpc") == "2.0"
        assert response.get("id") == 2
        assert "result" in response

        hover = response["result"]
        assert hover is not None
        assert "contents" in hover
        contents = hover["contents"]
        assert contents.get("kind") == "markdown"

        markdown_value = contents.get("value", "")
        # Signature specification must be present
        assert "ta.sma(" in markdown_value
        # Markdown formatting must be present
        assert "```pine" in markdown_value

    def test_hover_request_on_keyword(self, lsp_server: PineScriptLSPServer):
        """
        Acceptance Criteria 2 (Keyword variant):
        When client sends `textDocument/hover` at position of a keyword,
        server returns Markdown documentation.
        """
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="//@version=5\nvar int count = 0",
            version=1,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 1, "character": 1},  # inside 'var'
            },
        }

        response = lsp_server.handle_message(request)

        assert response is not None
        assert response.get("id") == 3
        hover = response.get("result")
        assert hover is not None
        contents = hover["contents"]
        assert contents.get("kind") == "markdown"
        assert "var" in contents.get("value", "")

    def test_hover_request_on_non_builtin_returns_null_result(self, lsp_server: PineScriptLSPServer):
        """Hovering over an uncataloged identifier returns null result."""
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="custom_func() => 123\nx = custom_func()",
            version=1,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 1, "character": 6},  # inside 'custom_func'
            },
        }

        response = lsp_server.handle_message(request)
        assert response is not None
        assert response.get("id") == 4
        assert response.get("result") is None

    def test_completion_after_document_change(self, lsp_server: PineScriptLSPServer):
        """Completions dynamically reflect newly typed contents after didChange."""
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="val = 1",
            version=1,
        )

        # Update document to include a partial built-in prefix
        lsp_server.did_change(
            uri=self.DOC_URI,
            text="val = math.r",
            version=2,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 0, "character": 12},  # after 'math.r'
            },
        }

        response = lsp_server.handle_message(request)
        assert response is not None
        results = response.get("result", [])
        assert len(results) > 0
        labels = [item["label"] for item in results]
        assert "math.round" in labels

    def test_server_handles_unknown_document_uri(self, lsp_server: PineScriptLSPServer):
        """Completion request for an unopened document raises or returns error response."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": "file:///workspace/not_opened.pine"},
                "position": {"line": 0, "character": 0},
            },
        }

        response = lsp_server.handle_message(request)
        assert response is not None
        # Must return an error or null result for missing document
        assert "error" in response or response.get("result") is None

    def test_server_handles_out_of_bounds_position_gracefully(self, lsp_server: PineScriptLSPServer):
        """Requests with out-of-range lines or characters should not crash the server."""
        lsp_server.did_open(
            uri=self.DOC_URI,
            text="val = close",
            version=1,
        )

        request = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": self.DOC_URI},
                "position": {"line": 99, "character": 99},
            },
        }

        response = lsp_server.handle_message(request)
        assert response is not None
        assert response.get("result") is None

    def test_server_direct_api_completion(self, lsp_server: PineScriptLSPServer):
        """Test high-level server.completion() method directly."""
        lsp_server.did_open(self.DOC_URI, "x = ta.e", 1)
        items = lsp_server.completion(self.DOC_URI, line=0, character=8)

        assert isinstance(items, list)
        assert any(item["label"] == "ta.ema" for item in items)
        for item in items:
            assert "label" in item
            assert "detail" in item

    def test_server_direct_api_hover(self, lsp_server: PineScriptLSPServer):
        """Test high-level server.hover() method directly."""
        lsp_server.did_open(self.DOC_URI, "x = ta.ema(close, 9)", 1)
        hover = lsp_server.hover(self.DOC_URI, line=0, character=6)

        assert hover is not None
        assert "contents" in hover
        assert "ta.ema" in hover["contents"]["value"]
        assert hover["contents"]["kind"] == "markdown"

    def test_server_raises_value_error_for_malformed_notification(
        self, lsp_server: PineScriptLSPServer
    ):
        """Assert direct exception type when missing required fields in programmatic calls."""
        with pytest.raises(ValueError):
            lsp_server.did_open(uri="", text="", version=0)