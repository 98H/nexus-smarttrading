"""
Instanced Batch Candlestick WebGL Shader module.

Defines GLSL shader generation, attribute descriptors, and uniform configurations
for rendering instanced candlestick bodies and wicks in a single WebGL draw call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class AttributeDescriptor:
    """Descriptor for a WebGL vertex attribute."""

    name: str
    gl_type: str = "FLOAT"
    components: int = 1
    divisor: int = 1
    is_instanced: bool = True
    stride: int = 4
    offset: int = 0
    byte_size: int = 4


@dataclass(frozen=True)
class UniformDescriptor:
    """Descriptor for a WebGL uniform variable."""

    name: str
    gl_type: str
    value: Any = None


@dataclass(frozen=True)
class CompiledCandlestickProgram:
    """Container for compiled WebGL candlestick shader sources and metadata."""

    vertex_source: str
    fragment_source: str
    attributes: dict[str, AttributeDescriptor]
    uniforms: dict[str, UniformDescriptor]
    glsl_version: str = "300 es"


class CandlestickShader:
    """WebGL shader generator for instanced batch candlestick rendering."""

    SUPPORTED_GLSL_VERSIONS: tuple[str, ...] = ("300 es", "100")

    def __init__(
        self,
        glsl_version: str = "300 es",
        default_candle_width: float = 0.8,
        default_wick_width: float = 0.1,
        default_bull_color: Sequence[float] = (0.15, 0.75, 0.35, 1.0),
        default_bear_color: Sequence[float] = (0.90, 0.25, 0.25, 1.0),
    ) -> None:
        if glsl_version not in self.SUPPORTED_GLSL_VERSIONS:
            raise ValueError(
                f"Unsupported GLSL version: {glsl_version!r}. "
                f"Supported versions: {', '.join(self.SUPPORTED_GLSL_VERSIONS)}"
            )

        if default_candle_width < 0.0:
            raise ValueError(
                f"Candle width cannot be negative, got {default_candle_width}"
            )
        if default_wick_width < 0.0:
            raise ValueError(
                f"Wick width cannot be negative, got {default_wick_width}"
            )
        if default_wick_width > default_candle_width:
            raise ValueError(
                f"Wick width ({default_wick_width}) cannot exceed candle width ({default_candle_width})"
            )

        if len(default_bull_color) != 4:
            raise ValueError(
                f"Bull color must be RGBA 4-tuple, got {len(default_bull_color)} elements"
            )
        if len(default_bear_color) != 4:
            raise ValueError(
                f"Bear color must be RGBA 4-tuple, got {len(default_bear_color)} elements"
            )

        for c in default_bull_color:
            if not (0.0 <= c <= 1.0):
                raise ValueError(
                    f"Bull color channel value {c} out of bounds [0.0, 1.0]"
                )
        for c in default_bear_color:
            if not (0.0 <= c <= 1.0):
                raise ValueError(
                    f"Bear color channel value {c} out of bounds [0.0, 1.0]"
                )

        self.glsl_version = glsl_version
        self.default_candle_width = float(default_candle_width)
        self.default_wick_width = float(default_wick_width)
        self.default_bull_color = tuple(float(c) for c in default_bull_color)
        self.default_bear_color = tuple(float(c) for c in default_bear_color)

    def _generate_vertex_shader_300(self) -> str:
        return """#version 300 es
precision highp float;

in float a_open;
in float a_high;
in float a_low;
in float a_close;
in float a_index;
in vec2 a_corner;
in float a_vertex_type;

uniform float u_candle_width;
uniform float u_wick_width;
uniform vec4 u_bull_color;
uniform vec4 u_bear_color;
uniform mat4 u_projection_matrix;

out vec4 v_color;

void main() {
    float open = a_open;
    float high = a_high;
    float low = a_low;
    float close = a_close;

    bool is_bull = close >= open;
    v_color = is_bull ? u_bull_color : u_bear_color;

    float body_min = min(open, close);
    float body_max = max(open, close);
    if (body_max - body_min < 0.0001) {
        body_max = body_min + 0.0001;
    }

    float x_center = a_index;
    float x = 0.0;
    float y = 0.0;

    if (a_vertex_type > 0.5) {
        x = x_center + a_corner.x * u_candle_width;
        y = mix(body_min, body_max, a_corner.y);
    } else {
        x = x_center + a_corner.x * u_wick_width;
        y = mix(low, high, a_corner.y);
    }

    gl_Position = u_projection_matrix * vec4(x, y, 0.0, 1.0);
}
"""

    def _generate_fragment_shader_300(self) -> str:
        return """#version 300 es
precision highp float;

in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""

    def _generate_vertex_shader_100(self) -> str:
        return """precision highp float;

attribute float a_open;   // in float a_open;
attribute float a_high;   // in float a_high;
attribute float a_low;    // in float a_low;
attribute float a_close;  // in float a_close;
attribute float a_index;  // in float a_index;
attribute vec2 a_corner;
attribute float a_vertex_type;

uniform float u_candle_width;
uniform float u_wick_width;
uniform vec4 u_bull_color;
uniform vec4 u_bear_color;
uniform mat4 u_projection_matrix;

varying vec4 v_color;

void main() {
    float open = a_open;
    float high = a_high;
    float low = a_low;
    float close = a_close;

    bool is_bull = close >= open;
    v_color = is_bull ? u_bull_color : u_bear_color;

    float body_min = min(open, close);
    float body_max = max(open, close);
    if (body_max - body_min < 0.0001) {
        body_max = body_min + 0.0001;
    }

    float x_center = a_index;
    float x = 0.0;
    float y = 0.0;

    if (a_vertex_type > 0.5) {
        x = x_center + a_corner.x * u_candle_width;
        y = mix(body_min, body_max, a_corner.y);
    } else {
        x = x_center + a_corner.x * u_wick_width;
        y = mix(low, high, a_corner.y);
    }

    gl_Position = u_projection_matrix * vec4(x, y, 0.0, 1.0);
}
"""

    def _generate_fragment_shader_100(self) -> str:
        return """precision highp float;

varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""

    def compile(self) -> CompiledCandlestickProgram:
        """Compile and return the WebGL candlestick shader program and metadata."""
        if self.glsl_version == "300 es":
            vertex_source = self._generate_vertex_shader_300()
            fragment_source = self._generate_fragment_shader_300()
        else:
            vertex_source = self._generate_vertex_shader_100()
            fragment_source = self._generate_fragment_shader_100()

        attributes = {
            "open": AttributeDescriptor(
                name="a_open",
                gl_type="FLOAT",
                components=1,
                divisor=1,
                is_instanced=True,
                stride=4,
                offset=0,
                byte_size=4,
            ),
            "high": AttributeDescriptor(
                name="a_high",
                gl_type="FLOAT",
                components=1,
                divisor=1,
                is_instanced=True,
                stride=4,
                offset=0,
                byte_size=4,
            ),
            "low": AttributeDescriptor(
                name="a_low",
                gl_type="FLOAT",
                components=1,
                divisor=1,
                is_instanced=True,
                stride=4,
                offset=0,
                byte_size=4,
            ),
            "close": AttributeDescriptor(
                name="a_close",
                gl_type="FLOAT",
                components=1,
                divisor=1,
                is_instanced=True,
                stride=4,
                offset=0,
                byte_size=4,
            ),
            "index": AttributeDescriptor(
                name="a_index",
                gl_type="FLOAT",
                components=1,
                divisor=1,
                is_instanced=True,
                stride=4,
                offset=0,
                byte_size=4,
            ),
        }

        uniforms = {
            "candle_width": UniformDescriptor(
                name="u_candle_width",
                gl_type="FLOAT",
                value=self.default_candle_width,
            ),
            "wick_width": UniformDescriptor(
                name="u_wick_width",
                gl_type="FLOAT",
                value=self.default_wick_width,
            ),
            "bull_color": UniformDescriptor(
                name="u_bull_color",
                gl_type="FLOAT_VEC4",
                value=self.default_bull_color,
            ),
            "bear_color": UniformDescriptor(
                name="u_bear_color",
                gl_type="FLOAT_VEC4",
                value=self.default_bear_color,
            ),
            "projection_matrix": UniformDescriptor(
                name="u_projection_matrix",
                gl_type="FLOAT_MAT4",
            ),
        }

        return CompiledCandlestickProgram(
            vertex_source=vertex_source,
            fragment_source=fragment_source,
            attributes=attributes,
            uniforms=uniforms,
            glsl_version=self.glsl_version,
        )