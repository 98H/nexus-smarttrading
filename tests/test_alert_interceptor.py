import ast
import pytest

from src.compiler import (
    AlertConditionInterceptor as RootAlertConditionInterceptor,
    InvalidAlertConditionError as RootInvalidAlertConditionError,
)
from src.compiler.alert_interceptor import (
    AlertConditionInterceptor,
    AlertHook,
    InvalidAlertConditionError,
)


def _get_condition(hook):
    """Retrieve condition expression attribute regardless of naming convention."""
    if hasattr(hook, "condition_expression"):
        return hook.condition_expression
    if hasattr(hook, "condition"):
        return hook.condition
    if isinstance(hook, dict):
        return hook.get("condition_expression", hook.get("condition"))
    raise AttributeError("Alert hook must contain 'condition' or 'condition_expression'")


def _get_title(hook):
    """Retrieve title attribute regardless of naming convention."""
    if hasattr(hook, "title"):
        val = hook.title
    elif isinstance(hook, dict):
        val = hook.get("title")
    else:
        raise AttributeError("Alert hook must contain 'title'")

    if isinstance(val, ast.Constant):
        return val.value
    return val


def _get_message(hook):
    """Retrieve message template attribute regardless of naming convention."""
    if hasattr(hook, "message_template"):
        val = hook.message_template
    elif hasattr(hook, "message"):
        val = hook.message
    elif isinstance(hook, dict):
        val = hook.get("message_template", hook.get("message"))
    else:
        raise AttributeError("Alert hook must contain 'message' or 'message_template'")

    if isinstance(val, ast.Constant):
        return val.value
    return val


def _run_interceptor(interceptor, node):
    """Run interceptor using either intercept(), process(), or standard visit()."""
    if hasattr(interceptor, "intercept"):
        res = interceptor.intercept(node)
        if res is not None:
            return res
    if hasattr(interceptor, "process"):
        res = interceptor.process(node)
        if res is not None:
            return res
    if hasattr(interceptor, "visit"):
        interceptor.visit(node)

    if hasattr(interceptor, "hooks"):
        return interceptor.hooks
    if hasattr(interceptor, "get_hooks"):
        return interceptor.get_hooks()

    raise AttributeError("AlertConditionInterceptor does not expose intercepted hooks")


# ============================================================================
# 1. Package & Module Export Tests
# ============================================================================


def test_compiler_package_exports():
    """Verify AlertConditionInterceptor and InvalidAlertConditionError are exported at src.compiler."""
    assert RootAlertConditionInterceptor is AlertConditionInterceptor
    assert RootInvalidAlertConditionError is InvalidAlertConditionError


def test_invalid_alert_condition_error_is_exception():
    """Verify InvalidAlertConditionError inherits from standard Exception."""
    assert issubclass(InvalidAlertConditionError, Exception)


def test_alert_hook_instantiation():
    """Verify AlertHook can be instantiated with condition expression, title, and message."""
    hook = AlertHook(condition="close > open", title="Bullish", message="Price is rising")
    assert _get_title(hook) == "Bullish"
    assert _get_message(hook) == "Price is rising"
    assert _get_condition(hook) is not None


# ============================================================================
# 2. Acceptance Criteria 1: Successful alertcondition() Interception
# ============================================================================


def test_intercept_alertcondition_positional_arguments():
    """Verify alertcondition() with positional arguments is intercepted and registered."""
    code = 'alertcondition(close > 100, "Price Spike", "Close crossed 100")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    hook = hooks[0]

    assert _get_title(hook) == "Price Spike"
    assert _get_message(hook) == "Close crossed 100"

    cond = _get_condition(hook)
    assert cond is not None
    if isinstance(cond, ast.AST):
        assert isinstance(cond, (ast.Compare, ast.expr))
    elif isinstance(cond, str):
        assert "close" in cond


def test_intercept_alertcondition_keyword_arguments():
    """Verify alertcondition() with keyword arguments is intercepted and registered."""
    code = 'alertcondition(condition=rsi < 30, title="Oversold Alert", message="RSI below 30")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    hook = hooks[0]
    assert _get_title(hook) == "Oversold Alert"
    assert _get_message(hook) == "RSI below 30"
    assert _get_condition(hook) is not None


def test_intercept_alertcondition_mixed_positional_and_keyword_arguments():
    """Verify alertcondition() with mixed positional and keyword arguments."""
    code = 'alertcondition(close > open, title="Bullish Bar", message="Green candle detected")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    hook = hooks[0]
    assert _get_title(hook) == "Bullish Bar"
    assert _get_message(hook) == "Green candle detected"
    assert _get_condition(hook) is not None


def test_intercept_alertcondition_optional_message_omitted():
    """Verify alertcondition() when message template argument is omitted defaults gracefully."""
    code = 'alertcondition(volume > 1000000, "High Volume")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    hook = hooks[0]
    assert _get_title(hook) == "High Volume"
    assert _get_message(hook) in ("", None)


def test_intercept_alertcondition_keyword_message_omitted():
    """Verify alertcondition() with keywords but without message defaults message properly."""
    code = 'alertcondition(condition=volume > 500000, title="Moderate Volume")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    hook = hooks[0]
    assert _get_title(hook) == "Moderate Volume"
    assert _get_message(hook) in ("", None)


def test_intercept_multiple_alertconditions_in_script():
    """Verify multiple alertcondition() calls in an AST are intercepted in order."""
    code = """
alertcondition(rsi > 70, "RSI High", "Overbought level reached")
alertcondition(rsi < 30, "RSI Low", "Oversold level reached")
alertcondition(cross(fast, slow), "MA Cross", "Fast crossed Slow")
"""
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 3

    assert _get_title(hooks[0]) == "RSI High"
    assert _get_message(hooks[0]) == "Overbought level reached"

    assert _get_title(hooks[1]) == "RSI Low"
    assert _get_message(hooks[1]) == "Oversold level reached"

    assert _get_title(hooks[2]) == "MA Cross"
    assert _get_message(hooks[2]) == "Fast crossed Slow"


def test_ignores_non_alertcondition_function_calls():
    """Verify calls other than alertcondition() are ignored and not registered as hooks."""
    code = """
study("My Indicator", overlay=true)
plot(close, color="red")
alertcondition(close > 100, "Breakout", "Above 100")
strategy.entry("Long", true)
"""
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, tree)

    assert len(hooks) == 1
    assert _get_title(hooks[0]) == "Breakout"


def test_intercept_direct_ast_call_node():
    """Verify interceptor can visit/process an isolated ast.Call node directly."""
    call_node = ast.Call(
        func=ast.Name(id="alertcondition", ctx=ast.Load()),
        args=[
            ast.Compare(
                left=ast.Name(id="close", ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=200)],
            ),
            ast.Constant(value="Direct Call Test"),
        ],
        keywords=[ast.keyword(arg="message", value=ast.Constant(value="Direct Msg"))],
    )

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, call_node)

    assert len(hooks) == 1
    assert _get_title(hooks[0]) == "Direct Call Test"
    assert _get_message(hooks[0]) == "Direct Msg"


def test_intercept_call_structure_dictionary():
    """Verify interceptor can process a call structure dictionary representing alertcondition."""
    call_struct = {
        "func": "alertcondition",
        "condition": "ta.crossover(close, ema)",
        "title": "EMA Crossover",
        "message": "Price crossed EMA",
    }

    interceptor = AlertConditionInterceptor()
    hooks = _run_interceptor(interceptor, call_struct)

    assert len(hooks) == 1
    assert _get_title(hooks[0]) == "EMA Crossover"
    assert _get_message(hooks[0]) == "Price crossed EMA"


# ============================================================================
# 3. Acceptance Criteria 2: Mandatory Argument Validation & Exceptions
# ============================================================================


def test_error_raised_when_zero_arguments_provided():
    """Verify InvalidAlertConditionError is raised when alertcondition() has no arguments."""
    code = "alertcondition()"
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_title_missing_positional():
    """Verify InvalidAlertConditionError is raised when only condition is provided positionally."""
    code = "alertcondition(close > 100)"
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_title_missing_keyword():
    """Verify InvalidAlertConditionError is raised when condition keyword is passed but title is omitted."""
    code = "alertcondition(condition=close > 100)"
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_condition_missing_keyword():
    """Verify InvalidAlertConditionError is raised when title keyword is passed but condition is omitted."""
    code = 'alertcondition(title="Missing Condition")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_condition_missing_with_message():
    """Verify InvalidAlertConditionError is raised when title and message are passed but condition is missing."""
    code = 'alertcondition(title="No Condition Alert", message="Should fail")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_condition_is_none():
    """Verify InvalidAlertConditionError is raised when condition argument evaluates to None."""
    code = 'alertcondition(None, "Invalid None Condition")'
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_title_is_none():
    """Verify InvalidAlertConditionError is raised when title argument evaluates to None."""
    code = "alertcondition(close > open, None)"
    tree = ast.parse(code)

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, tree)


def test_error_raised_when_call_structure_dict_missing_mandatory():
    """Verify InvalidAlertConditionError is raised when a dictionary call structure lacks title."""
    call_struct = {
        "func": "alertcondition",
        "condition": "close > open",
        # title is missing
    }

    interceptor = AlertConditionInterceptor()
    with pytest.raises(InvalidAlertConditionError):
        _run_interceptor(interceptor, call_struct)


# ============================================================================
# 4. State Isolation & Idempotency
# ============================================================================


def test_interceptor_state_isolation_between_instances():
    """Verify distinct interceptor instances maintain separate hook registries."""
    tree_a = ast.parse('alertcondition(close > 10, "A", "Message A")')
    tree_b = ast.parse('alertcondition(close < 5, "B", "Message B")')

    interceptor_a = AlertConditionInterceptor()
    interceptor_b = AlertConditionInterceptor()

    hooks_a = _run_interceptor(interceptor_a, tree_a)
    hooks_b = _run_interceptor(interceptor_b, tree_b)

    assert len(hooks_a) == 1
    assert _get_title(hooks_a[0]) == "A"

    assert len(hooks_b) == 1
    assert _get_title(hooks_b[0]) == "B"