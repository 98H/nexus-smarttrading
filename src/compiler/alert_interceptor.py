from __future__ import annotations

import ast
from typing import Any, Optional


class InvalidAlertConditionError(ValueError):
    """Raised when an alertcondition() invocation is invalid or missing mandatory arguments."""


class AlertHook:
    """Represents an intercepted alertcondition() hook."""

    def __init__(
        self,
        condition: Any = None,
        title: Any = None,
        message: Optional[Any] = None,
        *,
        condition_expression: Any = None,
        message_template: Optional[Any] = None,
    ) -> None:
        self.condition = condition if condition is not None else condition_expression
        self.title = title
        if message_template is not None:
            self.message = message_template
        else:
            self.message = message

    @property
    def condition_expression(self) -> Any:
        return self.condition

    @property
    def message_template(self) -> Optional[Any]:
        return self.message

    def __getitem__(self, item: str) -> Any:
        if item in ("condition", "condition_expression"):
            return self.condition
        if item == "title":
            return self.title
        if item in ("message", "message_template"):
            return self.message
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def __repr__(self) -> str:
        return (
            f"AlertHook(condition={self.condition!r}, "
            f"title={self.title!r}, message={self.message!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AlertHook):
            return False
        return (
            self.condition == other.condition
            and self.title == other.title
            and self.message == other.message
        )


def _is_missing_or_none(val: Any) -> bool:
    """Check whether an AST node or value is missing or explicitly None."""
    if val is None:
        return True
    if isinstance(val, ast.Constant) and val.value is None:
        return True
    if hasattr(ast, "NameConstant") and isinstance(val, ast.NameConstant) and val.value is None:
        return True
    return False


def _unwrap_value(val: Any) -> Any:
    """Unwrap an ast.Constant node to its primitive value if applicable."""
    if isinstance(val, ast.Constant):
        return val.value
    return val


class AlertConditionInterceptor(ast.NodeVisitor):
    """AST visitor and structure parser that intercepts alertcondition() invocations."""

    def __init__(self) -> None:
        self.hooks: list[AlertHook] = []

    def get_hooks(self) -> list[AlertHook]:
        """Return the list of intercepted alert hooks."""
        return self.hooks

    def clear(self) -> None:
        """Reset the collected hooks."""
        self.hooks.clear()

    def intercept(self, node: Any) -> list[AlertHook]:
        """Intercept alertcondition invocations from AST nodes or call structures."""
        self.visit(node)
        return self.hooks

    def process(self, node: Any) -> list[AlertHook]:
        """Alias for intercept()."""
        return self.intercept(node)

    def visit(self, node: Any) -> Any:
        """Visit AST nodes, dictionaries, or collections of nodes."""
        if isinstance(node, dict):
            self._process_dict(node)
            return self.hooks
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return self.hooks
        if isinstance(node, ast.AST):
            super().visit(node)
            return self.hooks
        return self.hooks

    def visit_Call(self, node: ast.Call) -> None:
        """Visit call nodes and intercept alertcondition invocations."""
        func_name = self._get_call_name(node)
        if func_name == "alertcondition":
            self._process_call_node(node)
        self.generic_visit(node)

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Extract function name from ast.Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _process_call_node(self, node: ast.Call) -> None:
        """Extract and validate alertcondition arguments from an ast.Call node."""
        condition = None
        title = None
        message = None

        if len(node.args) > 0:
            condition = node.args[0]
        if len(node.args) > 1:
            title = node.args[1]
        if len(node.args) > 2:
            message = node.args[2]

        for kw in node.keywords:
            if kw.arg in ("condition", "condition_expression"):
                condition = kw.value
            elif kw.arg == "title":
                title = kw.value
            elif kw.arg in ("message", "message_template"):
                message = kw.value

        if _is_missing_or_none(condition):
            raise InvalidAlertConditionError("alertcondition() requires a condition expression")
        if _is_missing_or_none(title):
            raise InvalidAlertConditionError("alertcondition() requires a title argument")

        hook = AlertHook(
            condition=condition,
            title=_unwrap_value(title),
            message=_unwrap_value(message) if message is not None else None,
        )
        self.hooks.append(hook)

    def _process_dict(self, struct: dict[str, Any]) -> None:
        """Extract and validate alertcondition arguments from a dictionary structure."""
        func_name = struct.get("func") or struct.get("name") or struct.get("type")
        if func_name is not None and func_name != "alertcondition":
            return

        if (
            func_name != "alertcondition"
            and "condition" not in struct
            and "condition_expression" not in struct
        ):
            return

        condition = struct.get("condition", struct.get("condition_expression"))
        title = struct.get("title")
        message = struct.get("message", struct.get("message_template"))

        if "args" in struct and isinstance(struct["args"], (list, tuple)):
            args = struct["args"]
            if len(args) > 0 and condition is None:
                condition = args[0]
            if len(args) > 1 and title is None:
                title = args[1]
            if len(args) > 2 and message is None:
                message = args[2]

        if "kwargs" in struct and isinstance(struct["kwargs"], dict):
            kwargs = struct["kwargs"]
            if condition is None:
                condition = kwargs.get("condition", kwargs.get("condition_expression"))
            if title is None:
                title = kwargs.get("title")
            if message is None:
                message = kwargs.get("message", kwargs.get("message_template"))

        if _is_missing_or_none(condition):
            raise InvalidAlertConditionError("alertcondition() requires a condition expression")
        if _is_missing_or_none(title):
            raise InvalidAlertConditionError("alertcondition() requires a title argument")

        hook = AlertHook(
            condition=condition,
            title=_unwrap_value(title),
            message=_unwrap_value(message) if message is not None else None,
        )
        self.hooks.append(hook)