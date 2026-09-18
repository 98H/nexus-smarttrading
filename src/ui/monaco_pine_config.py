"""Monaco Editor Pine Script v5 syntax highlighting and configuration."""

from __future__ import annotations

import copy
from typing import Any

# Supported languages in Monaco editor integration
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"pine"})

PINE_V5_KEYWORDS: list[str] = [
    "break",
    "continue",
    "else",
    "export",
    "for",
    "if",
    "import",
    "method",
    "return",
    "switch",
    "type",
    "var",
    "varip",
    "while",
]

PINE_V5_TYPES: list[str] = [
    "bool",
    "box",
    "color",
    "const",
    "float",
    "input",
    "int",
    "label",
    "line",
    "series",
    "simple",
    "string",
    "table",
]

PINE_V5_BUILTINS: list[str] = [
    "indicator",
    "math.abs",
    "plot",
    "plotshape",
    "strategy",
    "strategy.close",
    "strategy.entry",
    "ta.ema",
    "ta.rsi",
    "ta.sma",
]


def build_pine_v5_monarch_tokens() -> dict[str, Any]:
    """Generate a Monarch tokenizer specification for Pine Script v5.

    Returns:
        A dictionary containing the Monarch language definition mapping keywords,
        types, builtins, comments, and literals.
    """
    return {
        "defaultToken": "invalid",
        "keywords": list(PINE_V5_KEYWORDS),
        "types": list(PINE_V5_TYPES),
        "builtins": list(PINE_V5_BUILTINS),
        "tokenizer": {
            "root": [
                [r"//.*$", "comment"],
                [r'"([^"\\]|\\.)*"', "string"],
                [r"'([^'\\]|\\.)*'", "string"],
                [
                    r"[a-zA-Z_][\w.]*",
                    {
                        "cases": {
                            "@keywords": "keyword",
                            "@types": "type",
                            "@builtins": "builtin",
                            "@default": "identifier",
                        }
                    },
                ],
                [r"\d+(\.\d+)?", "number"],
                [r"[{}()\[\]]", "@brackets"],
            ]
        },
    }


def get_monaco_editor_settings(
    language: str = "pine",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create editor configuration options for a Monaco editor session.

    Args:
        language: Language identifier for the editor session (default: "pine").
        overrides: Optional dictionary of Monaco editor settings to merge.

    Returns:
        Dictionary of editor options configured for the target language.

    Raises:
        ValueError: If the language is not supported.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language target '{language}'. Supported languages: {sorted(SUPPORTED_LANGUAGES)}"
        )

    settings: dict[str, Any] = {
        "language": language,
        "automaticLayout": True,
        "tabSize": 4,
        "insertSpaces": True,
        "theme": "pine-theme",
        "themeData": {
            "base": "vs-dark",
            "inherit": True,
            "rules": [
                {"token": "comment", "foreground": "6A9955"},
                {"token": "keyword", "foreground": "C586C0"},
                {"token": "type", "foreground": "4EC9B0"},
                {"token": "builtin", "foreground": "DCDCAA"},
                {"token": "string", "foreground": "CE9178"},
                {"token": "number", "foreground": "B5CEA8"},
            ],
            "colors": {},
        },
    }

    result = copy.deepcopy(settings)
    if overrides:
        result.update(copy.deepcopy(overrides))

    return result