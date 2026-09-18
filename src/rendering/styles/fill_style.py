"""Defines style definitions and elements for rendering."""

from typing import Any, Optional


class FillStyle:
    """Represents the styling properties of a renderable element."""

    def __init__(
        self,
        fill: Optional[str] = None,
        color: Optional[str] = None,
        fill_expression: Optional[str] = None,
        color_expression: Optional[str] = None,
        opacity: Optional[float] = None,
        stroke: Optional[str] = None,
        stroke_width: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.fill = fill
        self.color = color
        self.fill_expression = fill_expression
        self.color_expression = color_expression
        self.opacity = opacity
        self.stroke = stroke
        self.stroke_width = stroke_width
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"FillStyle({attrs})"


class Element:
    """Represents a renderable element with an identifier and style."""

    def __init__(
        self,
        id: str,
        style: Optional[FillStyle] = None,
        **kwargs: Any,
    ) -> None:
        self.id = id
        self.style = style if style is not None else FillStyle()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Element(id={self.id!r}, style={self.style!r})"