from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import overload

from src.editor.models import (
    Diagnostic,
    Position,
    QuickFix,
    Range,
    Severity,
    SourceDocument,
    TextEdit,
)


class LintRule(ABC):
    """Abstract base class for pluggable lint rules."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for the lint rule."""
        pass

    @property
    @abstractmethod
    def severity(self) -> Severity:
        """Default severity for diagnostics emitted by this rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this rule checks."""
        pass

    @abstractmethod
    def check(self, document: SourceDocument) -> list[Diagnostic]:
        """Analyze the document and return a list of diagnostics."""
        pass


class TrailingWhitespaceRule(LintRule):
    """Rule detecting trailing whitespace characters at the end of lines."""

    @property
    def rule_id(self) -> str:
        return "trailing-whitespace"

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def description(self) -> str:
        return "Disallow trailing whitespace at line ends"

    def check(self, document: SourceDocument) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for line_num, line in enumerate(document.text.splitlines(), start=1):
            stripped = line.rstrip(" \t")
            if len(stripped) < len(line):
                start_col = len(stripped) + 1
                end_col = len(line) + 1
                rng = Range.from_coords(line_num, start_col, line_num, end_col)
                edit = TextEdit(range=rng, new_text="")
                fix = QuickFix(title="Remove trailing whitespace", edits=[edit])
                diagnostics.append(
                    Diagnostic(
                        range=rng,
                        severity=self.severity,
                        message="Trailing whitespace detected",
                        code=self.rule_id,
                        source="linter",
                        quick_fixes=[fix],
                    )
                )
        return diagnostics


class LintEngine:
    """Engine responsible for real-time code analysis and quick-fix application."""

    def __init__(self) -> None:
        self._rules: dict[str, LintRule] = {}
        self.register_rule(TrailingWhitespaceRule())

    def register_rule(self, rule: LintRule) -> None:
        """Register a new lint rule with the engine."""
        self._rules[rule.rule_id] = rule

    def unregister_rule(self, rule_id: str) -> None:
        """Unregister a lint rule by its ID."""
        self._rules.pop(rule_id, None)

    def get_rules(self) -> list[LintRule]:
        """Return a list of all currently registered rules."""
        return list(self._rules.values())

    def analyze(self, document: SourceDocument | str) -> list[Diagnostic]:
        """Analyze a source document and return all detected diagnostics."""
        if isinstance(document, str):
            doc = SourceDocument(text=document)
        else:
            doc = document

        diagnostics: list[Diagnostic] = []

        # 1. Syntax analysis
        try:
            ast.parse(doc.text, filename=doc.uri or "<string>")
        except SyntaxError as err:
            start_line = max(1, err.lineno or 1)
            start_col = max(1, err.offset or 1)
            end_line = max(start_line, err.end_lineno or start_line)

            if err.end_offset is not None:
                end_col = max(1, err.end_offset)
            else:
                end_col = start_col + 1

            if end_line == start_line and end_col <= start_col:
                end_col = start_col + 1

            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(start_line, start_col),
                        end=Position(end_line, end_col),
                    ),
                    severity=Severity.ERROR,
                    message=err.msg or "Syntax error",
                    code="syntax-error",
                    source="syntax-parser",
                )
            )

        # 2. Rule-based linting
        for rule in self._rules.values():
            diagnostics.extend(rule.check(doc))

        # 3. Sort diagnostics in document order
        diagnostics.sort(key=lambda d: (d.range.start, d.range.end))
        return diagnostics

    @overload
    def apply_quick_fix(self, document: str, fix: QuickFix) -> str: ...

    @overload
    def apply_quick_fix(self, document: SourceDocument, fix: QuickFix) -> SourceDocument: ...

    def apply_quick_fix(
        self, document: SourceDocument | str, fix: QuickFix
    ) -> SourceDocument | str:
        """Apply a quick-fix to the document, updating all affected ranges."""
        is_doc = isinstance(document, SourceDocument)
        text = document.text if is_doc else document

        # Validate that edits within the fix do not overlap
        for i in range(len(fix.edits)):
            for j in range(i + 1, len(fix.edits)):
                e1 = fix.edits[i]
                e2 = fix.edits[j]
                if e1.range.overlaps(e2.range) or e1.range == e2.range:
                    raise ValueError(
                        f"Conflicting overlapping edits in quick fix: {e1} and {e2}"
                    )

        raw_lines = text.splitlines(keepends=True)
        if not raw_lines:
            raw_lines = [""]

        line_start_offsets = [0] * len(raw_lines)
        for idx in range(1, len(raw_lines)):
            line_start_offsets[idx] = line_start_offsets[idx - 1] + len(raw_lines[idx - 1])

        def pos_to_offset(pos: Position) -> int:
            if pos.line < 1 or pos.line > len(raw_lines):
                raise IndexError(
                    f"Line {pos.line} is out of bounds (document has {len(raw_lines)} lines)"
                )

            line_str = raw_lines[pos.line - 1]
            content = line_str
            if content.endswith("\r\n"):
                content = content[:-2]
            elif content.endswith(("\r", "\n")):
                content = content[:-1]

            if pos.column < 1 or pos.column > len(content) + 1:
                raise IndexError(
                    f"Column {pos.column} is out of bounds for line {pos.line} "
                    f"(line content length is {len(content)})"
                )

            return line_start_offsets[pos.line - 1] + (pos.column - 1)

        offset_edits: list[tuple[int, int, str]] = []
        for edit in fix.edits:
            start_off = pos_to_offset(edit.range.start)
            end_off = pos_to_offset(edit.range.end)
            offset_edits.append((start_off, end_off, edit.new_text))

        # Check for overlaps in character offset space
        offset_edits.sort(key=lambda item: (item[0], item[1]))
        for idx in range(len(offset_edits) - 1):
            cur_start, cur_end, _ = offset_edits[idx]
            nxt_start, _, _ = offset_edits[idx + 1]
            if cur_end > nxt_start:
                raise ValueError("Overlapping text edit ranges detected")

        # Apply edits in reverse offset order to prevent coordinate drift
        offset_edits.sort(key=lambda item: item[0], reverse=True)
        updated_text = text
        for start_off, end_off, new_text in offset_edits:
            updated_text = updated_text[:start_off] + new_text + updated_text[end_off:]

        if is_doc:
            return SourceDocument(text=updated_text, uri=document.uri)
        return updated_text