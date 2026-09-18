"""
Language Server Protocol request handler helpers for Pine Script.

Provides position extraction utilities, completion item generation, and
hover documentation payloads based on the Pine Script catalog.
"""

from typing import Any, Dict, List, Optional

from src.lsp.catalog import PineScriptCatalog

LSP_COMPLETION_KIND_MAP = {
    "function": 3,
    "method": 2,
    "variable": 6,
    "keyword": 14,
}


def extract_identifier_at_position(line: str, character: int) -> str:
    """
    Extract the full dotted identifier or symbol at a given 0-indexed cursor position.

    Args:
        line: The text line containing the cursor.
        character: 0-indexed column position.

    Returns:
        The identifier string surrounding the position, or empty string if on punctuation/whitespace.
    """
    if character < 0 or character >= len(line):
        return ""

    target_char = line[character]
    if not (target_char.isalnum() or target_char == "_"):
        if target_char == "." and character > 0 and character + 1 < len(line):
            left_valid = line[character - 1].isalnum() or line[character - 1] == "_"
            right_valid = line[character + 1].isalnum() or line[character + 1] == "_"
            if not (left_valid and right_valid):
                return ""
        else:
            return ""

    start = character
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] in ("_", ".")):
        start -= 1

    end = character
    while end + 1 < len(line) and (line[end + 1].isalnum() or line[end + 1] in ("_", ".")):
        end += 1

    token = line[start : end + 1].strip(".")
    parts = token.split(".")
    for part in parts:
        if not part or not (part[0].isalpha() or part[0] == "_"):
            return ""
        if not all(c.isalnum() or c == "_" for c in part):
            return ""

    return token


def extract_prefix_at_position(line: str, character: int) -> str:
    """
    Extract the prefix typed immediately preceding the cursor position.

    Args:
        line: The text line containing the cursor.
        character: 0-indexed column position where cursor is typing.

    Returns:
        The prefix string up to the cursor position.
    """
    if character <= 0:
        return ""

    character = min(character, len(line))
    end = character
    start = end
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] in ("_", ".")):
        start -= 1

    return line[start:end]


def handle_completion(
    catalog: PineScriptCatalog, text: str, line: int, character: int
) -> List[Dict[str, Any]]:
    """
    Generate LSP completion items for the document text at the given position.

    Args:
        catalog: PineScriptCatalog containing language definitions.
        text: Entire document buffer string.
        line: 0-indexed line index.
        character: 0-indexed column position.

    Returns:
        List of LSP CompletionItem dictionaries.
    """
    lines = text.split("\n")
    if line < 0 or line >= len(lines):
        return []

    current_line = lines[line].rstrip("\r")
    prefix = extract_prefix_at_position(current_line, character)
    if not prefix:
        return []

    symbols = catalog.find_completions(prefix)
    items: List[Dict[str, Any]] = []

    for sym in symbols:
        item = {
            "label": sym.name,
            "kind": LSP_COMPLETION_KIND_MAP.get(sym.kind, 1),
            "detail": sym.detail,
            "documentation": {
                "kind": "markdown",
                "value": sym.documentation,
            },
            "insertText": sym.name,
        }
        items.append(item)

    return items


def handle_hover(
    catalog: PineScriptCatalog, text: str, line: int, character: int
) -> Optional[Dict[str, Any]]:
    """
    Generate an LSP hover payload for the symbol at the given position.

    Args:
        catalog: PineScriptCatalog containing language definitions.
        text: Entire document buffer string.
        line: 0-indexed line index.
        character: 0-indexed column position.

    Returns:
        LSP Hover payload dictionary with Markdown contents, or None if no symbol.
    """
    lines = text.split("\n")
    if line < 0 or line >= len(lines):
        return None

    current_line = lines[line].rstrip("\r")
    identifier = extract_identifier_at_position(current_line, character)
    if not identifier:
        return None

    symbol = catalog.get_symbol(identifier)
    if symbol is None:
        return None

    return {
        "contents": {
            "kind": "markdown",
            "value": symbol.documentation,
        }
    }