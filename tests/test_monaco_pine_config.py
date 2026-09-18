"""
Unit tests for Monaco Editor Pine Script v5 syntax highlighting integration.
Specification: Story 10.1.1: Integrate Monaco Editor with Pine Script v5 Syntax Highlighting
Target module: src/ui/monaco_pine_config.py
"""

import copy
import re
import pytest

from src.ui.monaco_pine_config import (
    build_pine_v5_monarch_tokens,
    get_monaco_editor_settings,
)


class TestBuildPineV5MonarchTokens:
    """Test suite for build_pine_v5_monarch_tokens tokenizer generation."""

    def test_returns_dict_with_required_monarch_keys(self):
        """Verify the Monarch specification contains necessary top-level fields."""
        tokens = build_pine_v5_monarch_tokens()

        assert isinstance(tokens, dict)
        assert "tokenizer" in tokens
        assert isinstance(tokens["tokenizer"], dict)
        assert "root" in tokens["tokenizer"]
        assert isinstance(tokens["tokenizer"]["root"], list)

    def test_maps_pine_v5_core_keywords(self):
        """Verify that Pine Script v5 control flow and declaration keywords are present."""
        tokens = build_pine_v5_monarch_tokens()

        # Monarch definitions typically expose a 'keywords' list or rules mapping keywords
        assert "keywords" in tokens
        keywords = set(tokens["keywords"])

        expected_keywords = {
            "if",
            "else",
            "for",
            "while",
            "return",
            "switch",
            "break",
            "continue",
            "var",
            "varip",
            "type",
            "method",
            "import",
            "export",
        }
        for kw in expected_keywords:
            assert kw in keywords, f"Expected Pine Script v5 keyword '{kw}' in keywords list"

    def test_maps_pine_v5_types(self):
        """Verify Pine Script v5 type identifiers are explicitly mapped."""
        tokens = build_pine_v5_monarch_tokens()

        assert "types" in tokens
        types_set = set(tokens["types"])

        expected_types = {
            "int",
            "float",
            "bool",
            "string",
            "color",
            "line",
            "label",
            "box",
            "table",
            "series",
            "simple",
            "const",
            "input",
        }
        for t in expected_types:
            assert t in types_set, f"Expected Pine Script v5 type '{t}' in types list"

    def test_maps_pine_v5_builtins(self):
        """Verify standard Pine Script v5 built-in functions and namespaces are mapped."""
        tokens = build_pine_v5_monarch_tokens()

        # Built-in functions or namespaces can be registered under 'builtins' or 'functions'
        builtins_key = "builtins" if "builtins" in tokens else "functions"
        assert builtins_key in tokens
        builtins = set(tokens[builtins_key])

        expected_builtins = {
            "indicator",
            "strategy",
            "plot",
            "plotshape",
            "ta.sma",
            "ta.ema",
            "ta.rsi",
            "math.abs",
            "strategy.entry",
            "strategy.close",
        }
        for b in expected_builtins:
            assert b in builtins, f"Expected Pine Script v5 built-in '{b}' in {builtins_key}"

    def test_comment_syntax_mapping_in_tokenizer_root(self):
        """Verify the tokenizer root rules define single-line comment matching."""
        tokens = build_pine_v5_monarch_tokens()
        root_rules = tokens["tokenizer"]["root"]

        # Search for a rule that maps comments starting with //
        comment_rule_found = False
        for rule in root_rules:
            if isinstance(rule, (list, tuple)) and len(rule) >= 2:
                pattern, action = rule[0], rule[1]
                pattern_str = pattern.pattern if isinstance(pattern, re.Pattern) else str(pattern)
                action_str = str(action)
                if "//" in pattern_str and "comment" in action_str.lower():
                    comment_rule_found = True
                    break
            elif isinstance(rule, dict):
                # Monarch object rule format: {"regex": ..., "action": ...}
                regex = str(rule.get("regex", ""))
                action = str(rule.get("action", "")) or str(rule.get("token", ""))
                if "//" in regex and "comment" in action.lower():
                    comment_rule_found = True
                    break

        assert comment_rule_found, "Single-line comment rule ('//') mapping to 'comment' not found in root rules"

    def test_string_literal_token_mapping(self):
        """Verify that string literal rules exist in the tokenizer."""
        tokens = build_pine_v5_monarch_tokens()
        root_rules = tokens["tokenizer"]["root"]

        string_rule_found = False
        for rule in root_rules:
            rule_str = str(rule).lower()
            if "string" in rule_str and ("'" in rule_str or '"' in rule_str or "quote" in rule_str):
                string_rule_found = True
                break

        assert string_rule_found, "String literal rule mapping to 'string' token not found in tokenizer"

    def test_returns_independent_instances(self):
        """Ensure calling build_pine_v5_monarch_tokens returns a new, mutable dictionary copy."""
        tokens1 = build_pine_v5_monarch_tokens()
        tokens2 = build_pine_v5_monarch_tokens()

        assert tokens1 is not tokens2
        tokens1["keywords"].append("custom_token_mutation")
        assert "custom_token_mutation" not in tokens2["keywords"]


class TestGetMonacoEditorSettings:
    """Test suite for get_monaco_editor_settings configuration generator."""

    def test_default_language_is_pine(self):
        """Verify invoking without explicit language defaults to 'pine'."""
        settings = get_monaco_editor_settings()

        assert isinstance(settings, dict)
        assert settings.get("language") == "pine"

    def test_explicit_pine_target(self):
        """Verify explicitly passing language target 'pine' sets language ID correctly."""
        settings = get_monaco_editor_settings(language="pine")

        assert settings.get("language") == "pine"

    def test_matching_theme_rules_included(self):
        """Verify that theme rule definitions match Pine Script tokens (comment, keyword, type, builtin)."""
        settings = get_monaco_editor_settings(language="pine")

        # Must include theme configuration with rules
        assert "theme" in settings or "themeData" in settings or "themeRules" in settings

        theme_data = settings.get("themeData") or settings.get("theme") or settings.get("themeRules")
        if isinstance(theme_data, dict) and "rules" in theme_data:
            rules = theme_data["rules"]
        elif isinstance(theme_data, list):
            rules = theme_data
        else:
            rules = []

        assert len(rules) > 0, "Theme rules must not be empty"

        # Check that token categories have corresponding style rules
        token_targets = {"comment", "keyword", "type"}
        found_tokens = set()

        for rule in rules:
            if isinstance(rule, dict) and "token" in rule:
                found_tokens.add(rule["token"].lower())

        for target in token_targets:
            assert any(target in token for token in found_tokens), f"Theme rule missing for token type: {target}"

    def test_editor_operational_options_present(self):
        """Verify that essential Monaco editor options are configured."""
        settings = get_monaco_editor_settings(language="pine")

        assert "automaticLayout" in settings
        assert settings["automaticLayout"] is True
        assert "tabSize" in settings
        assert isinstance(settings["tabSize"], int)

    def test_invalid_language_target_raises_value_error(self):
        """Verify requesting an unsupported language raises ValueError."""
        with pytest.raises(ValueError):
            get_monaco_editor_settings(language="unsupported_script_lang")

    def test_custom_overrides_applied_correctly(self):
        """Verify caller can provide custom overrides to the editor settings."""
        settings = get_monaco_editor_settings(
            language="pine",
            overrides={"readOnly": True, "fontSize": 14}
        )

        assert settings.get("language") == "pine"
        assert settings.get("readOnly") is True
        assert settings.get("fontSize") == 14

    def test_settings_returns_independent_instances(self):
        """Ensure repeated calls return distinct dict objects."""
        settings1 = get_monaco_editor_settings(language="pine")
        settings2 = get_monaco_editor_settings(language="pine")

        assert settings1 is not settings2
        settings1["language"] = "mutated"
        assert settings2["language"] == "pine"