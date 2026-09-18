"""
Language Server Protocol (LSP) Server for Pine Script.

Coordinates document buffers and dispatches completion and hover JSON-RPC requests.
"""

import json
from typing import Any, Dict, List, Optional, Union

from src.lsp.catalog import PineScriptCatalog
from src.lsp.handlers import handle_completion, handle_hover


class PineScriptLSPServer:
    """Pine Script LSP Server managing buffers and servicing protocol messages."""

    def __init__(self, catalog: Optional[PineScriptCatalog] = None) -> None:
        self.catalog = catalog if catalog is not None else PineScriptCatalog()
        self._documents: Dict[str, str] = {}
        self._versions: Dict[str, int] = {}

    def did_open(self, uri: str, text: str, version: int = 1) -> None:
        """
        Notify server of a document opened in the client.

        Args:
            uri: Target document URI string.
            text: Document content string.
            version: Document version counter (must be >= 1).

        Raises:
            ValueError: If uri is empty, version < 1, or text is None.
        """
        if not uri or uri.strip() == "":
            raise ValueError("Document URI cannot be empty.")
        if version < 1:
            raise ValueError("Document version must be >= 1.")
        if text is None:
            raise ValueError("Document text cannot be None.")

        self._documents[uri] = text
        self._versions[uri] = version

    def did_change(self, uri: str, text: str, version: int = 1) -> None:
        """
        Notify server of changes to a document buffer.

        Args:
            uri: Target document URI string.
            text: Updated document text.
            version: Updated document version counter.

        Raises:
            ValueError: If uri is empty or version < 1.
        """
        if not uri or uri.strip() == "":
            raise ValueError("Document URI cannot be empty.")
        if version < 1:
            raise ValueError("Document version must be >= 1.")
        if text is None:
            raise ValueError("Document text cannot be None.")

        self._documents[uri] = text
        self._versions[uri] = version

    def get_document_text(self, uri: str) -> Optional[str]:
        """
        Retrieve stored text buffer for a document URI.

        Args:
            uri: Document URI.

        Returns:
            Document text if tracked, otherwise None.
        """
        return self._documents.get(uri)

    def completion(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
        """
        Direct API to compute completions for a document URI at cursor position.

        Args:
            uri: Document URI.
            line: 0-indexed line number.
            character: 0-indexed character offset.

        Returns:
            List of completion items.
        """
        text = self._documents.get(uri)
        if text is None:
            return []
        return handle_completion(self.catalog, text=text, line=line, character=character)

    def hover(self, uri: str, line: int, character: int) -> Optional[Dict[str, Any]]:
        """
        Direct API to compute hover tooltips for a document URI at cursor position.

        Args:
            uri: Document URI.
            line: 0-indexed line number.
            character: 0-indexed character offset.

        Returns:
            Hover dictionary payload or None.
        """
        text = self._documents.get(uri)
        if text is None:
            return None
        return handle_hover(self.catalog, text=text, line=line, character=character)

    def handle_message(
        self, message: Union[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process incoming LSP JSON-RPC message dictionary or string.

        Args:
            message: Deserialized JSON-RPC dict or raw JSON string.

        Returns:
            JSON-RPC response dictionary, or None for notifications.
        """
        if isinstance(message, str):
            req: Dict[str, Any] = json.loads(message)
        else:
            req = message

        msg_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "textDocument/didOpen":
            text_doc = params.get("textDocument", {})
            self.did_open(
                uri=text_doc.get("uri", ""),
                text=text_doc.get("text", ""),
                version=text_doc.get("version", 1),
            )
            if msg_id is not None:
                return {"jsonrpc": "2.0", "id": msg_id, "result": None}
            return None

        elif method == "textDocument/didChange":
            text_doc = params.get("textDocument", {})
            uri = text_doc.get("uri", "")
            version = text_doc.get("version", 1)
            changes = params.get("contentChanges", [])
            new_text = changes[-1].get("text", "") if changes else params.get("text", "")
            self.did_change(uri=uri, text=new_text, version=version)
            if msg_id is not None:
                return {"jsonrpc": "2.0", "id": msg_id, "result": None}
            return None

        elif method == "textDocument/completion":
            text_doc = params.get("textDocument", {})
            uri = text_doc.get("uri", "")
            position = params.get("position", {})
            line = position.get("line", 0)
            character = position.get("character", 0)

            if uri not in self._documents:
                return {"jsonrpc": "2.0", "id": msg_id, "result": None}

            results = self.completion(uri=uri, line=line, character=character)
            return {"jsonrpc": "2.0", "id": msg_id, "result": results}

        elif method == "textDocument/hover":
            text_doc = params.get("textDocument", {})
            uri = text_doc.get("uri", "")
            position = params.get("position", {})
            line = position.get("line", 0)
            character = position.get("character", 0)

            if uri not in self._documents:
                return {"jsonrpc": "2.0", "id": msg_id, "result": None}

            hover_data = self.hover(uri=uri, line=line, character=character)
            return {"jsonrpc": "2.0", "id": msg_id, "result": hover_data}

        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method {method} not found"},
            }

        return None