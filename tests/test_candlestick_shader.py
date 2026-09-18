"""
Unit tests for the Instanced Batch Candlestick WebGL Shader module.

Story 1.3.1: Implement Instanced Batch Candlestick WebGL Shader
Acceptance Criteria:
- Given instanced OHLC batch data buffers,
- When the candlestick WebGL shader module is compiled,
- Then it successfully exports valid vertex and fragment GLSL programs that define
  instanced attributes (open, high, low, close, index) and uniforms for rendering
  bodies and wicks in a single draw call.
"""

import re
import pytest

from src.visualization.shaders import CandlestickShader
from src.visualization.shaders.candlestick import (
    CandlestickShader,
    CompiledCandlestickProgram,
    AttributeDescriptor,
    UniformDescriptor,
)


# ============================================================================
# Package and Module Export Tests
# ============================================================================

def test_shaders_package_exports_candlestick_shader():
    """Verify that src.visualization.shaders properly re-exports CandlestickShader."""
    import src.visualization.shaders as shaders_pkg

    assert hasattr(shaders_pkg, "CandlestickShader"), (
        "src.visualization.shaders must export CandlestickShader"
    )
    assert shaders_pkg.CandlestickShader is CandlestickShader


def test_candlestick_module_exports_expected_types():
    """Verify core classes and descriptors are accessible from candlestick module."""
    assert issubclass(CompiledCandlestickProgram, object)
    assert issubclass(AttributeDescriptor, object)
    assert issubclass(UniformDescriptor, object)


# ============================================================================
# Shader Compilation and GLSL Output Tests
# ============================================================================

def test_shader_compilation_produces_compiled_program():
    """Test that compiling CandlestickShader produces a valid CompiledCandlestickProgram."""
    shader = CandlestickShader()
    program = shader.compile()

    assert isinstance(program, CompiledCandlestickProgram)
    assert isinstance(program.vertex_source, str)
    assert isinstance(program.fragment_source, str)
    assert len(program.vertex_source.strip()) > 0
    assert len(program.fragment_source.strip()) > 0


@pytest.mark.parametrize("glsl_version", ["300 es", "100"])
def test_shader_compilation_supports_specified_glsl_versions(glsl_version):
    """Test that shader compiles with WebGL 2.0 (300 es) or WebGL 1.0 (100) directives."""
    shader = CandlestickShader(glsl_version=glsl_version)
    program = shader.compile()

    expected_version_tag = f"#version {glsl_version}"
    if glsl_version == "300 es":
        assert expected_version_tag in program.vertex_source
        assert expected_version_tag in program.fragment_source
    else:
        # GLSL 100 ES typically omits version directive or defines 100
        assert "#version 300 es" not in program.vertex_source
        assert "#version 300 es" not in program.fragment_source


def test_shader_contains_glsl_precision_qualifiers():
    """Test that generated GLSL sources define required precision qualifiers for WebGL."""
    shader = CandlestickShader()
    program = shader.compile()

    precision_pattern = re.compile(r"precision\s+(highp|mediump)\s+float;", re.MULTILINE)
    assert precision_pattern.search(program.vertex_source) is not None, (
        "Vertex shader missing precision qualifier"
    )
    assert precision_pattern.search(program.fragment_source) is not None, (
        "Fragment shader missing precision qualifier"
    )


def test_shader_contains_main_entry_points():
    """Test that both vertex and fragment GLSL shaders declare void main()."""
    shader = CandlestickShader()
    program = shader.compile()

    main_pattern = re.compile(r"void\s+main\s*\(\s*\)", re.MULTILINE)
    assert main_pattern.search(program.vertex_source) is not None, (
        "Vertex shader must contain void main()"
    )
    assert main_pattern.search(program.fragment_source) is not None, (
        "Fragment shader must contain void main()"
    )


# ============================================================================
# Instanced Attributes Tests (open, high, low, close, index)
# ============================================================================

def test_instanced_attributes_declared_in_metadata():
    """
    Verify that the compiled program exports descriptors for all required OHLC
    instanced attributes: open, high, low, close, and index with divisor == 1.
    """
    shader = CandlestickShader()
    program = shader.compile()

    required_attributes = {"open", "high", "low", "close", "index"}
    declared_attrs = program.attributes

    for attr in required_attributes:
        assert attr in declared_attrs, f"Instanced attribute '{attr}' not found in metadata"
        desc = declared_attrs[attr]
        assert isinstance(desc, AttributeDescriptor)
        assert desc.divisor == 1, (
            f"Attribute '{attr}' must have divisor=1 for instanced batch rendering"
        )
        assert desc.is_instanced is True, (
            f"Attribute '{attr}' must be explicitly marked as instanced"
        )


def test_instanced_attributes_declared_in_vertex_glsl():
    """
    Verify that vertex GLSL program contains declarations for the instanced attributes:
    open, high, low, close, index.
    """
    shader = CandlestickShader(glsl_version="300 es")
    program = shader.compile()

    vertex_glsl = program.vertex_source
    required_attributes = ["open", "high", "low", "close", "index"]

    for attr in required_attributes:
        # Regex matches either 'in float a_open;' or 'in float open;'
        pattern = rf"\bin\s+(float|int|uint)\s+(?:a_)?{attr}\b"
        assert re.search(pattern, vertex_glsl, re.IGNORECASE) is not None, (
            f"Vertex shader does not declare instanced attribute: {attr}"
        )


def test_instanced_attribute_data_types_and_strides():
    """Verify that instanced attributes have valid GL types and byte sizes configured."""
    shader = CandlestickShader()
    program = shader.compile()

    ohlc_attributes = ["open", "high", "low", "close"]
    for attr in ohlc_attributes:
        desc = program.attributes[attr]
        assert desc.gl_type in ("FLOAT", "HIGH_FLOAT", 0x1406), (
            f"OHLC attribute '{attr}' must be single-precision float"
        )
        assert desc.components == 1

    index_desc = program.attributes["index"]
    assert index_desc.gl_type in ("FLOAT", "INT", "UNSIGNED_INT", 0x1406, 0x1404, 0x1405)
    assert index_desc.components == 1


# ============================================================================
# Uniforms Definition Tests (Bodies and Wicks)
# ============================================================================

def test_uniforms_declared_in_metadata():
    """
    Verify program metadata defines uniforms for both body and wick rendering:
    candle width, wick width, bull color, bear color, and transform/projection matrix.
    """
    shader = CandlestickShader()
    program = shader.compile()

    declared_uniforms = program.uniforms

    expected_uniform_keys = [
        "candle_width",
        "wick_width",
        "bull_color",
        "bear_color",
        "projection_matrix",
    ]

    for uniform_key in expected_uniform_keys:
        assert uniform_key in declared_uniforms, (
            f"Uniform descriptor '{uniform_key}' not found in metadata"
        )
        desc = declared_uniforms[uniform_key]
        assert isinstance(desc, UniformDescriptor)
        assert desc.name is not None
        assert desc.gl_type is not None


def test_uniforms_declared_in_glsl_programs():
    """Verify that uniforms for rendering bodies and wicks are declared in GLSL source."""
    shader = CandlestickShader(glsl_version="300 es")
    program = shader.compile()

    combined_glsl = program.vertex_source + "\n" + program.fragment_source

    expected_glsl_uniform_patterns = [
        r"uniform\s+float\s+u_candle_width",
        r"uniform\s+float\s+u_wick_width",
        r"uniform\s+vec4\s+u_bull_color",
        r"uniform\s+vec4\s+u_bear_color",
        r"uniform\s+mat4\s+u_projection_matrix",
    ]

    for pattern in expected_glsl_uniform_patterns:
        assert re.search(pattern, combined_glsl) is not None, (
            f"Required uniform declaration matching '{pattern}' missing from GLSL code"
        )


# ============================================================================
# Single Draw Call Geometry and Shading Architecture Tests
# ============================================================================

def test_single_draw_call_handles_both_body_and_wick_geometry():
    """
    Verify vertex shader contains geometry branching or parametric vertex mapping
    to render both candle body (quad) and wick (lines/quad) in a single draw call.
    """
    shader = CandlestickShader()
    program = shader.compile()

    vertex_glsl = program.vertex_source

    # Vertex shader must evaluate both wick bounds (high, low) and body bounds (open, close)
    assert re.search(r"\bhigh\b", vertex_glsl, re.IGNORECASE) is not None, (
        "Vertex shader must reference 'high' for wick geometry"
    )
    assert re.search(r"\blow\b", vertex_glsl, re.IGNORECASE) is not None, (
        "Vertex shader must reference 'low' for wick geometry"
    )
    assert re.search(r"\bopen\b", vertex_glsl, re.IGNORECASE) is not None, (
        "Vertex shader must reference 'open' for body geometry"
    )
    assert re.search(r"\bclose\b", vertex_glsl, re.IGNORECASE) is not None, (
        "Vertex shader must reference 'close' for body geometry"
    )

    # In a single draw call, the vertex shader must utilize a vertex index, corner attribute,
    # or gl_VertexID to distinguish between wick vertices and body quad vertices.
    draw_call_distinguisher_pattern = re.compile(
        r"(gl_VertexID|a_position|a_corner|a_vertex_type|u_candle_width\s*\*\s*|u_wick_width)",
        re.MULTILINE
    )
    assert draw_call_distinguisher_pattern.search(vertex_glsl) is not None, (
        "Vertex shader must contain logic to build both body and wick geometry in one draw call"
    )


def test_bull_vs_bear_coloring_logic():
    """
    Verify GLSL code compares close vs open (or receives direction)
    to select bull_color vs bear_color.
    """
    shader = CandlestickShader()
    program = shader.compile()

    combined_glsl = program.vertex_source + "\n" + program.fragment_source

    # Check for comparison condition between close and open prices
    comparison_pattern = re.compile(
        r"(close\s*>=\s*open|close\s*>\s*open|open\s*<=\s*close|open\s*<\s*close)",
        re.IGNORECASE
    )
    assert comparison_pattern.search(combined_glsl) is not None, (
        "Shader must contain logic comparing close and open to determine bull/bear state"
    )

    # Check that both bull and bear colors are conditionally selected
    assert "u_bull_color" in combined_glsl
    assert "u_bear_color" in combined_glsl


def test_fragment_shader_outputs_valid_color():
    """Verify fragment shader exports fragment color via WebGL 2.0 'out vec4' or 'gl_FragColor'."""
    shader = CandlestickShader(glsl_version="300 es")
    program = shader.compile()

    frag_glsl = program.fragment_source
    frag_out_pattern = re.compile(r"(out\s+vec4\s+\w+|gl_FragColor\s*=)", re.MULTILINE)
    assert frag_out_pattern.search(frag_glsl) is not None, (
        "Fragment shader must output color to a fragment target or gl_FragColor"
    )


# ============================================================================
# Validation and Error Handling Tests
# ============================================================================

def test_unsupported_glsl_version_raises_error():
    """Verify that requesting an invalid or unsupported GLSL version raises ValueError."""
    with pytest.raises(ValueError):
        CandlestickShader(glsl_version="999 es")


def test_invalid_candle_width_uniform_ratio_raises_error():
    """
    Verify that invalid default configuration where wick width exceeds candle width
    raises a ValueError.
    """
    with pytest.raises(ValueError):
        CandlestickShader(default_candle_width=2.0, default_wick_width=5.0)


def test_negative_dimensions_raise_error():
    """Verify that negative candle or wick width dimensions raise ValueError."""
    with pytest.raises(ValueError):
        CandlestickShader(default_candle_width=-1.0)

    with pytest.raises(ValueError):
        CandlestickShader(default_wick_width=-0.5)


def test_invalid_color_dimensions_raise_error():
    """Verify that bull or bear colors not conforming to RGBA 4-tuples raise ValueError."""
    with pytest.raises(ValueError):
        CandlestickShader(default_bull_color=(1.0, 0.0, 0.0))  # Only 3 elements

    with pytest.raises(ValueError):
        CandlestickShader(default_bear_color=(1.0, 0.0, 0.0, 1.0, 0.5))  # 5 elements


def test_color_channel_out_of_bounds_raises_error():
    """Verify color channels outside [0.0, 1.0] raise ValueError."""
    with pytest.raises(ValueError):
        CandlestickShader(default_bull_color=(1.5, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError):
        CandlestickShader(default_bear_color=(-0.1, 0.0, 0.0, 1.0))