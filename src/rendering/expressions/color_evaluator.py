"""Color and fill expression resolver against runtime context."""

import ast
import logging
import operator
import re
from typing import Any, Callable, Dict, Optional

from src.rendering.styles.fill_style import Element

logger = logging.getLogger(__name__)

# Standard CSS and SVG named colors plus special color keywords
NAMED_COLORS = {
    "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige",
    "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown",
    "burlywood", "cadetblue", "chartreuse", "chocolate", "coral",
    "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
    "darkgoldenrod", "darkgray", "darkgreen", "darkgrey", "darkkhaki",
    "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred",
    "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
    "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
    "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite",
    "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod",
    "gray", "green", "greenyellow", "grey", "honeydew", "hotpink",
    "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush",
    "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
    "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
    "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
    "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow",
    "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
    "mediumslateblue", "mediumspringgreen", "mediumturquoise",
    "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
    "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange",
    "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
    "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum",
    "powderblue", "purple", "rebeccapurple", "red", "rosybrown",
    "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen",
    "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray",
    "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
    "thistle", "tomato", "turquoise", "violet", "wheat", "white",
    "whitesmoke", "yellow", "yellowgreen",
    # Special CSS keywords
    "transparent", "none", "currentcolor",
}

HEX_COLOR_PATTERN = re.compile(
    r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)

RGB_COLOR_PATTERN = re.compile(
    r"^rgb\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$",
    re.IGNORECASE,
)

RGBA_COLOR_PATTERN = re.compile(
    r"^rgba\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*\)$",
    re.IGNORECASE,
)


def _safe_pow(a: Any, b: Any) -> Any:
    if isinstance(b, (int, float)) and b > 1000:
        raise ValueError("Exponent too large")
    return operator.pow(a, b)


def _safe_mul(a: Any, b: Any) -> Any:
    if isinstance(a, (str, list, tuple)) and isinstance(b, int) and (len(a) * b > 10000):
        raise ValueError("Sequence multiplication result too large")
    if isinstance(b, (str, list, tuple)) and isinstance(a, int) and (len(b) * a > 10000):
        raise ValueError("Sequence multiplication result too large")
    return operator.mul(a, b)


class SafeExpressionResolver:
    """Safely resolves expression AST trees against a context dictionary without dynamic execution."""

    SAFE_OPERATORS: Dict[type, Callable[..., Any]] = {
        # Unary operators
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
        # Binary operators
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: _safe_mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: _safe_pow,
        # Comparison operators
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        self.context: Dict[str, Any] = context if context is not None else {}

    def resolve(self, expression: str) -> Any:
        """Parse expression into an AST tree and resolve its value."""
        parsed = ast.parse(expression.strip())
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
            raise ValueError("Expression must be a single expression statement")
        return self._resolve_node(parsed.body[0].value)

    def _resolve_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in self.context:
                return self.context[node.id]
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id == "None":
                return None
            raise NameError(f"Undefined variable '{node.id}'")

        if isinstance(node, ast.Attribute):
            val = self._resolve_node(node.value)
            attr = node.attr
            if attr.startswith("_"):
                raise AttributeError(f"Access to private attribute '{attr}' is forbidden")
            if isinstance(val, dict):
                if attr in val:
                    return val[attr]
                raise AttributeError(f"Context object has no property '{attr}'")
            if hasattr(val, attr):
                return getattr(val, attr)
            raise AttributeError(f"'{type(val).__name__}' object has no property '{attr}'")

        if isinstance(node, ast.IfExp):
            test = self._resolve_node(node.test)
            if test:
                return self._resolve_node(node.body)
            return self._resolve_node(node.orelse)

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                res = True
                for v in node.values:
                    res = self._resolve_node(v)
                    if not res:
                        return res
                return res
            if isinstance(node.op, ast.Or):
                res = False
                for v in node.values:
                    res = self._resolve_node(v)
                    if res:
                        return res
                return res
            raise TypeError(f"Unsupported boolean operator: {type(node.op).__name__}")

        if isinstance(node, ast.UnaryOp):
            op_func = self.SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_func(self._resolve_node(node.operand))

        if isinstance(node, ast.BinOp):
            op_func = self.SAFE_OPERATORS.get(type(node.op))
            if op_func is None:
                raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            left = self._resolve_node(node.left)
            right = self._resolve_node(node.right)
            return op_func(left, right)

        if isinstance(node, ast.Compare):
            left = self._resolve_node(node.left)
            for op, comp in zip(node.ops, node.comparators):
                op_func = self.SAFE_OPERATORS.get(type(op))
                if op_func is None:
                    raise TypeError(f"Unsupported comparison operator: {type(op).__name__}")
                right = self._resolve_node(comp)
                if not op_func(left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.List):
            return [self._resolve_node(elt) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._resolve_node(elt) for elt in node.elts)

        if isinstance(node, ast.Set):
            return {self._resolve_node(elt) for elt in node.elts}

        if isinstance(node, ast.Dict):
            return {
                self._resolve_node(k): self._resolve_node(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }

        if isinstance(node, ast.Subscript):
            val = self._resolve_node(node.value)
            slice_node = node.slice
            if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
                slice_node = slice_node.value  # type: ignore[attr-defined]
            idx = self._resolve_node(slice_node)
            return val[idx]

        if isinstance(node, ast.Call):
            raise TypeError("Function calls are not permitted in expressions")

        raise TypeError(f"Unsupported expression construct: {type(node).__name__}")


class ColorEvaluator:
    """Evaluates dynamic color and fill expressions against a runtime context."""

    def __init__(self, default_fill_color: str = "#000000") -> None:
        if not self.is_valid_color(default_fill_color):
            raise ValueError(f"Invalid default fill color: {default_fill_color!r}")
        self.default_fill_color = default_fill_color

    @staticmethod
    def is_valid_color(color: Any) -> bool:
        """Validate whether a value is a valid hex, rgb, rgba, or named color."""
        if not isinstance(color, str):
            return False

        color_str = color.strip()
        if not color_str:
            return False

        if HEX_COLOR_PATTERN.fullmatch(color_str):
            return True

        rgb_match = RGB_COLOR_PATTERN.fullmatch(color_str)
        if rgb_match:
            r, g, b = (
                int(rgb_match.group(1)),
                int(rgb_match.group(2)),
                int(rgb_match.group(3)),
            )
            return (0 <= r <= 255) and (0 <= g <= 255) and (0 <= b <= 255)

        rgba_match = RGBA_COLOR_PATTERN.fullmatch(color_str)
        if rgba_match:
            try:
                r = int(rgba_match.group(1))
                g = int(rgba_match.group(2))
                b = int(rgba_match.group(3))
                a = float(rgba_match.group(4))
                return (
                    (0 <= r <= 255)
                    and (0 <= g <= 255)
                    and (0 <= b <= 255)
                    and (0.0 <= a <= 1.0)
                )
            except ValueError:
                return False

        return color_str.lower() in NAMED_COLORS

    def evaluate(
        self, expression: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Evaluate a dynamic color expression against runtime context data.

        Falls back gracefully to the configured default fill color and logs a warning
        if the expression is invalid, non-string, or cannot be resolved.
        """
        if not isinstance(expression, str):
            logger.warning(
                "Color expression must be a string, got %s: %r",
                type(expression).__name__,
                expression,
            )
            return self.default_fill_color

        try:
            resolver = SafeExpressionResolver(context)
            result = resolver.resolve(expression)
        except Exception as exc:
            logger.warning(
                "Failed to evaluate color expression %r: %s",
                expression,
                exc,
            )
            return self.default_fill_color

        if not isinstance(result, str):
            logger.warning(
                "Color expression %r evaluated to non-string type %s: %r",
                expression,
                type(result).__name__,
                result,
            )
            return self.default_fill_color

        if not self.is_valid_color(result):
            logger.warning(
                "Color expression %r evaluated to invalid color representation: %r",
                expression,
                result,
            )
            return self.default_fill_color

        return result

    def resolve_element(
        self, element: Element, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Resolve fill and color expressions on an element and update style properties."""
        if not hasattr(element, "style") or element.style is None:
            return

        if getattr(element.style, "fill_expression", None) is not None:
            element.style.fill = self.evaluate(element.style.fill_expression, context)

        if getattr(element.style, "color_expression", None) is not None:
            element.style.color = self.evaluate(element.style.color_expression, context)