import pytest
from src.editor.models import (
    Diagnostic,
    Position,
    QuickFix,
    Range,
    Severity,
    SourceDocument,
    TextEdit,
)
from src.editor.lint_engine import LintEngine, LintRule


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> LintEngine:
    """Fixture providing a default LintEngine instance."""
    return LintEngine()


@pytest.fixture
def sample_python_code() -> str:
    """Fixture providing standard valid Python source code."""
    return (
        "def calculate_total(prices: list[float], tax_rate: float) -> float:\n"
        "    subtotal = sum(prices)\n"
        "    return subtotal * (1.0 + tax_rate)\n"
    )


# ============================================================================
# 1. Models Unit Tests (src/editor/models.py)
# ============================================================================


class TestPositionModel:
    """Tests for Position coordinate validation and behavior."""

    def test_position_instantiation_valid(self):
        pos = Position(line=1, column=5)
        assert pos.line == 1
        assert pos.column == 5

    @pytest.mark.parametrize(
        "line, column",
        [
            (0, 1),
            (-1, 5),
            (1, 0),
            (2, -3),
            (-5, -5),
        ],
    )
    def test_position_invalid_coordinates_raise_value_error(self, line: int, column: int):
        with pytest.raises(ValueError):
            Position(line=line, column=column)

    def test_position_ordering_and_comparison(self):
        p1 = Position(line=1, column=2)
        p2 = Position(line=1, column=5)
        p3 = Position(line=2, column=1)

        assert p1 < p2
        assert p2 < p3
        assert p1 < p3
        assert p1 <= Position(line=1, column=2)
        assert p3 > p2
        assert p1 == Position(line=1, column=2)
        assert p1 != p2

    def test_position_immutable(self):
        pos = Position(line=1, column=1)
        with pytest.raises(Exception):
            pos.line = 2  # type: ignore[misc]


class TestRangeModel:
    """Tests for Range validation, boundaries, and containment."""

    def test_range_instantiation_valid(self):
        start = Position(line=1, column=1)
        end = Position(line=1, column=10)
        rng = Range(start=start, end=end)
        assert rng.start == start
        assert rng.end == end
        assert not rng.is_empty

    def test_range_zero_width_is_empty(self):
        pos = Position(line=3, column=4)
        rng = Range(start=pos, end=pos)
        assert rng.start == rng.end
        assert rng.is_empty

    def test_range_inverted_coordinates_raise_value_error(self):
        start = Position(line=2, column=5)
        end = Position(line=1, column=10)
        with pytest.raises(ValueError):
            Range(start=start, end=end)

        col_start = Position(line=2, column=10)
        col_end = Position(line=2, column=5)
        with pytest.raises(ValueError):
            Range(start=col_start, end=col_end)

    def test_range_from_coords_helper(self):
        rng = Range.from_coords(1, 2, 3, 4)
        assert rng.start == Position(line=1, column=2)
        assert rng.end == Position(line=3, column=4)

    def test_range_contains_position(self):
        rng = Range.from_coords(2, 5, 2, 15)
        assert rng.contains(Position(line=2, column=5))
        assert rng.contains(Position(line=2, column=10))
        assert not rng.contains(Position(line=2, column=15))  # Half-open [start, end)
        assert not rng.contains(Position(line=2, column=4))
        assert not rng.contains(Position(line=1, column=10))
        assert not rng.contains(Position(line=3, column=5))

    def test_range_overlaps(self):
        r1 = Range.from_coords(1, 1, 1, 10)
        r2 = Range.from_coords(1, 5, 1, 15)
        r3 = Range.from_coords(1, 10, 1, 20)
        r4 = Range.from_coords(2, 1, 2, 5)

        assert r1.overlaps(r2)
        assert r2.overlaps(r1)
        assert not r1.overlaps(r3)  # Boundary touching at half-open endpoint is disjoint
        assert not r1.overlaps(r4)


class TestDiagnosticAndQuickFixModels:
    """Tests for Diagnostic, TextEdit, and QuickFix data contracts."""

    def test_severity_enumeration_values(self):
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"
        assert Severity.HINT.value == "hint"

    def test_text_edit_creation(self):
        rng = Range.from_coords(1, 1, 1, 4)
        edit = TextEdit(range=rng, new_text="def")
        assert edit.range == rng
        assert edit.new_text == "def"

    def test_quick_fix_creation(self):
        edit = TextEdit(range=Range.from_coords(1, 1, 1, 4), new_text="def")
        fix = QuickFix(title="Replace with 'def'", edits=[edit])
        assert fix.title == "Replace with 'def'"
        assert len(fix.edits) == 1
        assert fix.edits[0] == edit

    def test_quick_fix_empty_edits_raises_value_error(self):
        with pytest.raises(ValueError):
            QuickFix(title="Invalid empty fix", edits=[])

    def test_diagnostic_creation_defaults_and_fields(self):
        rng = Range.from_coords(1, 1, 1, 10)
        diag = Diagnostic(
            range=rng,
            severity=Severity.ERROR,
            message="Unexpected token",
            code="SYN001",
            source="syntax-parser",
        )
        assert diag.range == rng
        assert diag.severity == Severity.ERROR
        assert diag.message == "Unexpected token"
        assert diag.code == "SYN001"
        assert diag.source == "syntax-parser"
        assert diag.quick_fixes == []

    def test_source_document_properties(self):
        content = "line 1\nline 2\nline 3"
        doc = SourceDocument(text=content, uri="file:///test.py")
        assert doc.text == content
        assert doc.uri == "file:///test.py"
        assert doc.line_count == 3
        assert doc.get_line(1) == "line 1"
        assert doc.get_line(2) == "line 2"
        assert doc.get_line(3) == "line 3"

    def test_source_document_get_line_out_of_bounds_raises_index_error(self):
        doc = SourceDocument("single line")
        with pytest.raises(IndexError):
            doc.get_line(0)
        with pytest.raises(IndexError):
            doc.get_line(2)


# ============================================================================
# 2. Lint Engine Analysis Tests (src/editor/lint_engine.py)
# ============================================================================


class TestLintEngineAnalysis:
    """Tests for document syntax analysis and diagnostic emission."""

    def test_lint_clean_document_returns_empty_diagnostics(self, engine: LintEngine, sample_python_code: str):
        diagnostics = engine.analyze(sample_python_code)
        assert diagnostics == []

    def test_lint_empty_document_returns_empty_diagnostics(self, engine: LintEngine):
        diagnostics = engine.analyze("")
        assert diagnostics == []

    def test_lint_syntax_error_returns_error_diagnostic(self, engine: LintEngine):
        bad_code = "def unclosed_function(\n"
        diagnostics = engine.analyze(bad_code)

        assert len(diagnostics) >= 1
        syntax_errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        assert len(syntax_errors) >= 1

        diag = syntax_errors[0]
        assert diag.range.start.line == 1
        assert diag.range.start.column >= 1
        assert diag.range.end.line >= diag.range.start.line
        assert len(diag.message) > 0

    def test_lint_multiline_syntax_error_exact_location(self, engine: LintEngine):
        bad_code = (
            "x = 10\n"
            "y = 20\n"
            "if x > y\n"  # Missing colon on line 3
            "    print(x)\n"
        )
        diagnostics = engine.analyze(bad_code)

        syntax_errors = [d for d in diagnostics if d.severity == Severity.ERROR]
        assert len(syntax_errors) >= 1
        diag = syntax_errors[0]
        assert diag.range.start.line == 3
        assert diag.range.start.column > 1

    def test_lint_accepts_source_document_instance(self, engine: LintEngine):
        doc = SourceDocument(text="def invalid(\n", uri="file:///main.py")
        diagnostics = engine.analyze(doc)

        assert len(diagnostics) >= 1
        assert diagnostics[0].severity == Severity.ERROR

    def test_lint_trailing_whitespace_rule_attaches_quick_fix(self, engine: LintEngine):
        # Line 1 has 2 trailing spaces: "x = 1  \n" -> "x = 1" is 5 chars, spaces at cols 6-7
        code = "x = 1  \ny = 2\n"
        diagnostics = engine.analyze(code)

        ws_diags = [d for d in diagnostics if "trailing whitespace" in d.message.lower()]
        assert len(ws_diags) == 1

        diag = ws_diags[0]
        assert diag.severity in (Severity.WARNING, Severity.INFO)
        assert diag.range.start.line == 1
        assert diag.range.start.column == 6
        assert diag.range.end.line == 1
        assert diag.range.end.column == 8
        assert len(diag.quick_fixes) == 1

        fix = diag.quick_fixes[0]
        assert fix.title.lower().startswith("remove") or "whitespace" in fix.title.lower()
        assert len(fix.edits) == 1
        assert fix.edits[0].range == diag.range
        assert fix.edits[0].new_text == ""

    def test_lint_diagnostics_ordered_by_line_and_column(self, engine: LintEngine):
        # Two warnings on line 1, one warning on line 2
        code = "a = 1  \nb = 2  \n"
        diagnostics = engine.analyze(code)

        assert len(diagnostics) == 2
        assert diagnostics[0].range.start < diagnostics[1].range.start
        assert diagnostics[0].range.start.line == 1
        assert diagnostics[1].range.start.line == 2

    def test_lint_engine_custom_rule_registration(self, engine: LintEngine):
        class NoPrintRule(LintRule):
            @property
            def rule_id(self) -> str:
                return "no-print"

            @property
            def severity(self) -> Severity:
                return Severity.WARNING

            @property
            def description(self) -> str:
                return "Usage of print is discouraged"

            def check(self, document: SourceDocument) -> list[Diagnostic]:
                results = []
                for line_idx, line in enumerate(document.text.splitlines(), start=1):
                    col = line.find("print(")
                    if col != -1:
                        rng = Range.from_coords(line_idx, col + 1, line_idx, col + 7)
                        edit = TextEdit(range=rng, new_text="logger.info(")
                        fix = QuickFix(title="Use logger.info instead", edits=[edit])
                        results.append(
                            Diagnostic(
                                range=rng,
                                severity=self.severity,
                                message="Avoid print statement",
                                code=self.rule_id,
                                quick_fixes=[fix],
                            )
                        )
                return results

        rule = NoPrintRule()
        engine.register_rule(rule)
        assert rule in engine.get_rules()

        code = "x = 42\nprint('hello')\n"
        diagnostics = engine.analyze(code)
        custom_diags = [d for d in diagnostics if d.code == "no-print"]
        assert len(custom_diags) == 1
        assert custom_diags[0].range.start == Position(line=2, column=1)
        assert len(custom_diags[0].quick_fixes) == 1

        # Unregister rule and verify diagnostics clear
        engine.unregister_rule("no-print")
        assert rule not in engine.get_rules()
        assert len(engine.analyze(code)) == 0


# ============================================================================
# 3. Quick-Fix Engine Application Tests (src/editor/lint_engine.py)
# ============================================================================


class TestQuickFixApplication:
    """Tests for applying single, multiple, and boundary quick-fixes to source text."""

    def test_apply_quick_fix_single_token_replacement(self, engine: LintEngine):
        # 1-indexed: "let" starts at col 1, ends before col 4
        source = "let counter = 0\n"
        edit = TextEdit(range=Range.from_coords(1, 1, 1, 4), new_text="var")
        fix = QuickFix(title="Change 'let' to 'var'", edits=[edit])

        updated = engine.apply_quick_fix(source, fix)
        assert updated == "var counter = 0\n"

    def test_apply_quick_fix_insertion(self, engine: LintEngine):
        # Zero-width range: insert ":" at the end of "def process()" (col 14)
        source = "def process()\n    pass\n"
        edit = TextEdit(range=Range.from_coords(1, 14, 1, 14), new_text=":")
        fix = QuickFix(title="Add missing colon", edits=[edit])

        updated = engine.apply_quick_fix(source, fix)
        assert updated == "def process():\n    pass\n"

    def test_apply_quick_fix_deletion(self, engine: LintEngine):
        # Delete trailing spaces at cols 6-7
        source = "x = 1  \ny = 2\n"
        edit = TextEdit(range=Range.from_coords(1, 6, 1, 8), new_text="")
        fix = QuickFix(title="Remove trailing whitespace", edits=[edit])

        updated = engine.apply_quick_fix(source, fix)
        assert updated == "x = 1\ny = 2\n"

    def test_apply_quick_fix_multiline_replacement(self, engine: LintEngine):
        source = (
            "def calculate():\n"
            "    a = 1\n"
            "    b = 2\n"
            "    return a + b\n"
        )
        # Replace lines 2 and 3 ("    a = 1\n    b = 2") with "    a, b = 1, 2"
        # Line 2 col 1 to Line 3 col 10
        edit = TextEdit(
            range=Range.from_coords(2, 1, 3, 10),
            new_text="    a, b = 1, 2",
        )
        fix = QuickFix(title="Combine assignment lines", edits=[edit])

        updated = engine.apply_quick_fix(source, fix)
        expected = (
            "def calculate():\n"
            "    a, b = 1, 2\n"
            "    return a + b\n"
        )
        assert updated == expected

    def test_apply_quick_fix_multiple_non_overlapping_edits_in_document(self, engine: LintEngine):
        # Rename 'var1' to 'first' and 'var2' to 'second'
        source = "total = var1 + var2\n"
        # "var1" is at col 9 to 13
        # "var2" is at col 16 to 20
        edit1 = TextEdit(range=Range.from_coords(1, 9, 1, 13), new_text="first")
        edit2 = TextEdit(range=Range.from_coords(1, 16, 1, 20), new_text="second")

        # Supply edits in forward order; engine must apply them without offset drift
        fix = QuickFix(title="Rename variables", edits=[edit1, edit2])
        updated = engine.apply_quick_fix(source, fix)

        assert updated == "total = first + second\n"

    def test_apply_quick_fix_to_source_document_returns_new_instance(self, engine: LintEngine):
        doc = SourceDocument(text="old_val = 10", uri="file:///module.py")
        edit = TextEdit(range=Range.from_coords(1, 1, 1, 8), new_text="new_val")
        fix = QuickFix(title="Rename to new_val", edits=[edit])

        new_doc = engine.apply_quick_fix(doc, fix)

        assert isinstance(new_doc, SourceDocument)
        assert new_doc.text == "new_val = 10"
        assert new_doc.uri == "file:///module.py"
        # Ensure original instance remained unmodified
        assert doc.text == "old_val = 10"

    def test_apply_quick_fix_at_document_boundaries(self, engine: LintEngine):
        source = "core_logic()"

        # Prefix insertion at document start (1, 1)
        start_fix = QuickFix(
            title="Prepend comment",
            edits=[TextEdit(range=Range.from_coords(1, 1, 1, 1), new_text="# Start\n")],
        )
        res1 = engine.apply_quick_fix(source, start_fix)
        assert res1 == "# Start\ncore_logic()"

        # Suffix insertion at document end (1, 13)
        end_fix = QuickFix(
            title="Append comment",
            edits=[TextEdit(range=Range.from_coords(1, 13, 1, 13), new_text="\n# End")],
        )
        res2 = engine.apply_quick_fix(source, end_fix)
        assert res2 == "core_logic()\n# End"

    def test_apply_quick_fix_with_unicode_characters(self, engine: LintEngine):
        # 1-indexed column count:
        # "greeting = '👋 hello world'  \n"
        # 'greeting = ' (12) + '👋' (1) + ' hello world' (12) + "'" (1) = 26 chars
        # Trailing spaces are at columns 27 and 28 (Range: col 27 to 29)
        source = "greeting = '👋 hello world'  \n"
        edit = TextEdit(range=Range.from_coords(1, 27, 1, 29), new_text="")
        fix = QuickFix(title="Strip unicode trailing whitespace", edits=[edit])

        updated = engine.apply_quick_fix(source, fix)
        assert updated == "greeting = '👋 hello world'\n"

    def test_apply_quick_fix_from_diagnostic_end_to_end(self, engine: LintEngine):
        source = "flag = True  \n"
        diagnostics = engine.analyze(source)
        assert len(diagnostics) == 1
        assert len(diagnostics[0].quick_fixes) == 1

        applied = engine.apply_quick_fix(source, diagnostics[0].quick_fixes[0])
        assert applied == "flag = True\n"

        # Re-analyzing updated text should yield no diagnostics
        new_diagnostics = engine.analyze(applied)
        assert new_diagnostics == []

    def test_apply_quick_fix_overlapping_edits_raises_value_error(self, engine: LintEngine):
        source = "sample text"
        edit1 = TextEdit(range=Range.from_coords(1, 1, 1, 7), new_text="replacement1")
        edit2 = TextEdit(range=Range.from_coords(1, 5, 1, 11), new_text="replacement2")
        fix = QuickFix(title="Conflicting overlapping fix", edits=[edit1, edit2])

        with pytest.raises(ValueError):
            engine.apply_quick_fix(source, fix)

    def test_apply_quick_fix_out_of_bounds_line_raises_index_error(self, engine: LintEngine):
        source = "line 1\nline 2\n"
        edit = TextEdit(range=Range.from_coords(5, 1, 5, 3), new_text="foo")
        fix = QuickFix(title="OOB Line", edits=[edit])

        with pytest.raises(IndexError):
            engine.apply_quick_fix(source, fix)

    def test_apply_quick_fix_out_of_bounds_column_raises_index_error(self, engine: LintEngine):
        source = "line 1\n"
        edit = TextEdit(range=Range.from_coords(1, 20, 1, 25), new_text="foo")
        fix = QuickFix(title="OOB Column", edits=[edit])

        with pytest.raises(IndexError):
            engine.apply_quick_fix(source, fix)