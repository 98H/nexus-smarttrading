"""
Pine Script Sandboxed WebWorker Execution Sandbox Orchestrator.
"""

from typing import Any, Dict, List, Optional, Set, Union
import math
import multiprocessing
import re
import resource
import threading
import time
import tracemalloc

from src.models.pine_execution import (
    MarketSeriesData,
    PineCompiledPayload,
    PineExecutionResult,
    PineExecutionStatus,
    PineIndicatorOutput,
    PineSignal,
    PineSignalType,
    SandboxExecutionError,
    SandboxExecutionLimits,
    SandboxMemoryLimitError,
    SandboxSecurityViolationError,
    SandboxTimeoutError,
)

# =====================================================================
# Strict Property Whitelists & Identifier Safeguards
# =====================================================================

FORBIDDEN_IDENTIFIERS: Set[str] = {
    "process",
    "require",
    "child_process",
    "fs",
    "eval",
    "Function",
    "exec",
    "spawn",
    "global",
    "window",
    "document",
    "mainModule",
    "constructor",
    "prototype",
    "__proto__",
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__init__",
    "__globals__",
    "__code__",
    "__dict__",
    "__builtins__",
    "__import__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
}

TA_WHITELIST: Set[str] = {"sma", "ema", "crossover", "crossunder"}
STRATEGY_WHITELIST: Set[str] = {"entry", "long", "short", "close"}
ARRAY_WHITELIST: Set[str] = {"push", "length", "fill"}
NUMBER_WHITELIST: Set[str] = {"toString"}
STRING_WHITELIST: Set[str] = {"length", "toString"}


def _is_na(val: Any) -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def _safe_add(a: Any, b: Any) -> Any:
    return float("nan") if (_is_na(a) or _is_na(b)) else a + b


def _safe_sub(a: Any, b: Any) -> Any:
    return float("nan") if (_is_na(a) or _is_na(b)) else a - b


def _safe_mul(a: Any, b: Any) -> Any:
    return float("nan") if (_is_na(a) or _is_na(b)) else a * b


def _safe_div(a: Any, b: Any) -> Any:
    return float("nan") if (_is_na(a) or _is_na(b) or b == 0) else a / b


def _safe_gt(a: Any, b: Any) -> bool:
    return False if (_is_na(a) or _is_na(b)) else bool(a > b)


def _safe_ge(a: Any, b: Any) -> bool:
    return False if (_is_na(a) or _is_na(b)) else bool(a >= b)


def _safe_lt(a: Any, b: Any) -> bool:
    return False if (_is_na(a) or _is_na(b)) else bool(a < b)


def _safe_le(a: Any, b: Any) -> bool:
    return False if (_is_na(a) or _is_na(b)) else bool(a <= b)


def _safe_eq(a: Any, b: Any) -> bool:
    if _is_na(a) and _is_na(b):
        return True
    if _is_na(a) or _is_na(b):
        return False
    return bool(a == b)


# =====================================================================
# Domain Math & Strategy APIs
# =====================================================================


class Series(list):
    """Vectorized numerical series following ECMAScript object truthiness rules."""

    def __init__(self, data: Any = None) -> None:
        if data is not None:
            super().__init__(data)
        else:
            super().__init__()

    def __bool__(self) -> bool:
        return True

    def __getitem__(self, item: Any) -> Any:
        res = super().__getitem__(item)
        return Series(res) if isinstance(item, slice) else res

    def _validate_other(self, other: Any) -> None:
        if isinstance(other, (list, Series)) and len(self) != len(other):
            raise ValueError(
                f"Series length mismatch: expected length {len(self)}, got {len(other)}"
            )

    def __add__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_add(a, b) for a, b in zip(self, other)])
        return Series([_safe_add(a, other) for a in self])

    def __radd__(self, other: Any) -> "Series":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_sub(a, b) for a, b in zip(self, other)])
        return Series([_safe_sub(a, other) for a in self])

    def __rsub__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_sub(b, a) for a, b in zip(self, other)])
        return Series([_safe_sub(other, a) for a in self])

    def __mul__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_mul(a, b) for a, b in zip(self, other)])
        return Series([_safe_mul(a, other) for a in self])

    def __rmul__(self, other: Any) -> "Series":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_div(a, b) for a, b in zip(self, other)])
        return Series([_safe_div(a, other) for a in self])

    def __rtruediv__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_div(b, a) for a, b in zip(self, other)])
        return Series([_safe_div(other, a) for a in self])

    def __gt__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_gt(a, b) for a, b in zip(self, other)])
        return Series([_safe_gt(a, other) for a in self])

    def __ge__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_ge(a, b) for a, b in zip(self, other)])
        return Series([_safe_ge(a, other) for a in self])

    def __lt__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_lt(a, b) for a, b in zip(self, other)])
        return Series([_safe_lt(a, other) for a in self])

    def __le__(self, other: Any) -> "Series":
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_le(a, b) for a, b in zip(self, other)])
        return Series([_safe_le(a, other) for a in self])

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([_safe_eq(a, b) for a, b in zip(self, other)])
        return Series([_safe_eq(a, other) for a in self])

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        if isinstance(other, (list, Series)):
            self._validate_other(other)
            return Series([not _safe_eq(a, b) for a, b in zip(self, other)])
        return Series([not _safe_eq(a, other) for a in self])


class SimulatedArray(list):
    """Memory-bounded array container for sandboxed worker execution."""

    def __init__(self, size: int = 0, ctx: Optional["ExecutionContext"] = None) -> None:
        self.ctx = ctx
        int_size = max(0, int(size))
        limit_bytes = (ctx.limits.memory_limit_mb * 1024 * 1024) if ctx else (64 * 1024 * 1024)
        if int_size * 8 > limit_bytes:
            script_id = ctx.payload.script_id if ctx else ""
            raise SandboxMemoryLimitError(
                f"Array allocation of {int_size} elements exceeds memory limit",
                script_id=script_id,
            )
        super().__init__([None] * int_size)

    def fill(self, val: Any) -> "SimulatedArray":
        if len(self) > 0:
            self[:] = [val] * len(self)
        return self

    def push(self, item: Any) -> int:
        self.append(item)
        if self.ctx is not None:
            self.ctx.check_limits()
        return len(self)


class TaLib:
    """Sandboxed Technical Analysis indicator library."""

    def __init__(self, ctx: "ExecutionContext") -> None:
        self.ctx = ctx

    def sma(self, source: List[float], length: Union[int, float]) -> Series:
        len_int = max(1, int(length))
        n = len(source)
        res: List[float] = [float("nan")] * n
        if len_int > n:
            return Series(res)

        running_sum = 0.0
        nan_count = 0
        for i in range(len_int):
            val = source[i]
            if _is_na(val):
                nan_count += 1
            else:
                running_sum += float(val)

        if nan_count == 0:
            res[len_int - 1] = running_sum / len_int

        for i in range(len_int, n):
            old_val = source[i - len_int]
            new_val = source[i]

            if _is_na(old_val):
                nan_count -= 1
            else:
                running_sum -= float(old_val)

            if _is_na(new_val):
                nan_count += 1
            else:
                running_sum += float(new_val)

            res[i] = running_sum / len_int if nan_count == 0 else float("nan")

        return Series(res)

    def crossover(self, s1: Any, s2: Any) -> Series:
        n = min(len(s1), len(s2))
        res = [False] * n
        for i in range(1, n):
            if not _is_na(s1[i]) and not _is_na(s2[i]) and not _is_na(s1[i - 1]) and not _is_na(s2[i - 1]):
                if s1[i - 1] <= s2[i - 1] and s1[i] > s2[i]:
                    res[i] = True
        return Series(res)

    def crossunder(self, s1: Any, s2: Any) -> Series:
        n = min(len(s1), len(s2))
        res = [False] * n
        for i in range(1, n):
            if not _is_na(s1[i]) and not _is_na(s2[i]) and not _is_na(s1[i - 1]) and not _is_na(s2[i - 1]):
                if s1[i - 1] >= s2[i - 1] and s1[i] < s2[i]:
                    res[i] = True
        return Series(res)

    def ema(self, source: List[float], length: Union[int, float]) -> Series:
        len_int = max(1, int(length))
        alpha = 2.0 / (len_int + 1.0)
        res: List[float] = []
        prev = float("nan")
        for val in source:
            if _is_na(val):
                res.append(float("nan"))
            elif math.isnan(prev):
                prev = float(val)
                res.append(prev)
            else:
                prev = alpha * float(val) + (1.0 - alpha) * prev
                res.append(prev)
        return Series(res)


class StrategyLib:
    """Sandboxed trading strategy order execution API."""

    def __init__(self, ctx: "ExecutionContext") -> None:
        self.ctx = ctx
        self.long = PineSignalType.BUY
        self.short = PineSignalType.SELL
        self.close = PineSignalType.CLOSE

    def entry(self, order_id: str, direction: PineSignalType) -> None:
        idx = self.ctx.current_bar_index
        if idx is None:
            idx = len(self.ctx.series.timestamps) - 1

        if 0 <= idx < len(self.ctx.series.timestamps):
            sig = PineSignal(
                timestamp=self.ctx.series.timestamps[idx],
                signal_type=direction,
                price=float(self.ctx.series.close[idx]),
                label=str(order_id),
                metadata={"bar_index": idx, "order_id": str(order_id)},
            )
            self.ctx.add_signal(sig)


# =====================================================================
# Execution Context & Resource Telemetry
# =====================================================================


class ExecutionContext:
    """State, resource boundary monitor, and telemetry tracker for sandboxed execution."""

    def __init__(
        self,
        payload: PineCompiledPayload,
        series: MarketSeriesData,
        limits: SandboxExecutionLimits,
    ) -> None:
        self.payload = payload
        self.series = series
        self.limits = limits
        self.start_time = time.monotonic()
        self.total_output_size_bytes: int = 0
        self.check_step: int = 0

        self.global_this: Dict[str, Any] = {}
        self.indicator_outputs: Dict[str, PineIndicatorOutput] = {}
        self.signals: List[PineSignal] = []
        self.current_bar_index: Optional[int] = None

        self.series_open = Series(series.open)
        self.series_high = Series(series.high)
        self.series_low = Series(series.low)
        self.series_close = Series(series.close)
        self.series_volume = Series(series.volume)

        self.ta = TaLib(self)
        self.strategy = StrategyLib(self)

    def plot(self, data: Any, name: str) -> None:
        name_str = str(name)
        vals = list(data) if isinstance(data, (list, Series)) else [float(data)] * len(self.series.timestamps)

        if name_str in self.indicator_outputs:
            old_output = self.indicator_outputs[name_str]
            old_size = len(name_str.encode("utf-8")) + len(old_output.values) * 8 + 64
            self.total_output_size_bytes = max(0, self.total_output_size_bytes - old_size)

        new_size = len(name_str.encode("utf-8")) + len(vals) * 8 + 64
        self.total_output_size_bytes += new_size
        self.indicator_outputs[name_str] = PineIndicatorOutput(name=name_str, values=vals)

        self.check_output_size()

    def add_signal(self, signal: PineSignal) -> None:
        sig_size = len(signal.label.encode("utf-8")) + 128
        self.total_output_size_bytes += sig_size
        self.signals.append(signal)
        self.check_output_size()

    def check_limits(self) -> None:
        self.check_step += 1
        if (self.check_step & 0x07) == 0:
            elapsed_ms = (time.monotonic() - self.start_time) * 1000.0
            if elapsed_ms > self.limits.timeout_ms:
                raise SandboxTimeoutError(
                    f"Execution timed out after {self.limits.timeout_ms} ms",
                    script_id=self.payload.script_id,
                    status=PineExecutionStatus.TIMEOUT,
                )

            if tracemalloc.is_tracing():
                curr_traced, _ = tracemalloc.get_traced_memory()
                limit_bytes = self.limits.memory_limit_mb * 1024 * 1024
                if curr_traced > limit_bytes:
                    raise SandboxMemoryLimitError(
                        f"Memory limit of {self.limits.memory_limit_mb} MB exceeded",
                        script_id=self.payload.script_id,
                        status=PineExecutionStatus.MEMORY_EXCEEDED,
                    )

    def check_output_size(self) -> None:
        if self.total_output_size_bytes > self.limits.max_output_size_bytes:
            raise SandboxExecutionError(
                f"Maximum output size limit exceeded: {self.total_output_size_bytes} bytes > {self.limits.max_output_size_bytes} bytes",
                script_id=self.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )


# =====================================================================
# Lexer & AST Parser
# =====================================================================


class Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type_: str, value: str, pos: int) -> None:
        self.type = type_
        self.value = value
        self.pos = pos


def _decode_escapes(s: str) -> str:
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n:
            i += 1
            esc = s[i]
            if esc == "n":
                out.append("\n")
            elif esc == "t":
                out.append("\t")
            elif esc == "r":
                out.append("\r")
            elif esc == "b":
                out.append("\b")
            elif esc == "f":
                out.append("\f")
            elif esc == "v":
                out.append("\v")
            elif esc == "\\":
                out.append("\\")
            elif esc in ('"', "'", "`"):
                out.append(esc)
            elif esc == "x" and i + 2 < n:
                try:
                    out.append(chr(int(s[i + 1 : i + 3], 16)))
                    i += 2
                except ValueError:
                    out.append("x")
            elif esc == "u":
                if i + 1 < n and s[i + 1] == "{":
                    close_brace = s.find("}", i + 2)
                    if close_brace != -1:
                        try:
                            out.append(chr(int(s[i + 2 : close_brace], 16)))
                            i = close_brace
                        except (ValueError, OverflowError):
                            out.append("u")
                    else:
                        out.append("u")
                elif i + 4 < n:
                    try:
                        out.append(chr(int(s[i + 1 : i + 5], 16)))
                        i += 4
                    except ValueError:
                        out.append("u")
                else:
                    out.append("u")
            elif esc == "0":
                out.append("\0")
            else:
                out.append(esc)
            i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def tokenize(code: str) -> List[Token]:
    if len(code) > 100_000:
        raise SandboxExecutionError("Script bytecode length exceeds maximum limit", status=PineExecutionStatus.ERROR)

    tokens: List[Token] = []
    i = 0
    n = len(code)
    while i < n:
        if len(tokens) > 25_000:
            raise SandboxExecutionError("Maximum token count exceeded", status=PineExecutionStatus.ERROR)

        c = code[i]
        if c.isspace():
            i += 1
            continue

        if code[i : i + 2] == "//":
            i += 2
            while i < n and code[i] != "\n":
                i += 1
            continue

        if code[i : i + 2] == "/*":
            start_pos = i
            i += 2
            closed = False
            while i + 1 < n:
                if code[i : i + 2] == "*/":
                    i += 2
                    closed = True
                    break
                i += 1
            if not closed:
                raise SandboxExecutionError(f"SyntaxError: Unterminated comment at {start_pos}", status=PineExecutionStatus.ERROR)
            continue

        if c in ('"', "'", "`"):
            quote = c
            start = i
            i += 1
            raw_chars: List[str] = []
            closed = False
            while i < n:
                if code[i] == quote:
                    i += 1
                    closed = True
                    break
                if code[i] == "\\" and i + 1 < n:
                    raw_chars.append(code[i : i + 2])
                    i += 2
                else:
                    raw_chars.append(code[i])
                    i += 1

            if not closed:
                raise SandboxExecutionError(f"SyntaxError: Unterminated string literal at {start}", status=PineExecutionStatus.ERROR)
            tokens.append(Token("STRING", _decode_escapes("".join(raw_chars)), start))
            continue

        if code[i : i + 3] in ("===", "!=="):
            tokens.append(Token("OP", code[i : i + 3], i))
            i += 3
            continue

        if code[i : i + 2] in ("==", "!=", "<=", ">=", "++", "--", "&&", "||", "+=", "-=", "*=", "/="):
            tokens.append(Token("OP", code[i : i + 2], i))
            i += 2
            continue

        if c in "+-*/=<>!%&|^~":
            tokens.append(Token("OP", c, i))
            i += 1
            continue

        if c in ";,.(){}[]?:":
            tokens.append(Token("PUNCT", c, i))
            i += 1
            continue

        if c.isdigit() or (c == "." and i + 1 < n and code[i + 1].isdigit()):
            start = i
            has_dot = (c == ".")
            i += 1
            while i < n:
                if code[i] == ".":
                    if has_dot:
                        while i < n and (code[i].isdigit() or code[i] == "."):
                            i += 1
                        raise SandboxExecutionError(f"SyntaxError: Invalid numeric literal at {start}", status=PineExecutionStatus.ERROR)
                    has_dot = True
                    i += 1
                elif code[i].isdigit():
                    i += 1
                elif code[i] in ("e", "E"):
                    i += 1
                    if i < n and code[i] in ("+", "-"):
                        i += 1
                    while i < n and code[i].isdigit():
                        i += 1
                    break
                else:
                    break
            tokens.append(Token("NUMBER", code[start:i], start))
            continue

        if c.isalpha() or c in "_$":
            start = i
            while i < n and (code[i].isalnum() or code[i] in "_$"):
                i += 1
            tokens.append(Token("IDENT", code[start:i], start))
            continue

        raise SandboxExecutionError(f"SyntaxError: Unexpected character '{c}' at position {i}", status=PineExecutionStatus.ERROR)

    tokens.append(Token("EOF", "", n))
    return tokens


class Parser:
    """Recursive descent AST parser bounded by recursion limit."""

    def __init__(self, tokens: List[Token], script_id: str = "") -> None:
        self.tokens = tokens
        self.script_id = script_id
        self.pos = 0
        self.depth = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def match(self, *expected: str) -> Optional[Token]:
        cur = self.peek()
        for exp in expected:
            if cur.type == exp or cur.value == exp:
                self.pos += 1
                return cur
        return None

    def expect(self, expected: str) -> Token:
        tok = self.match(expected)
        if tok is None:
            raise SandboxExecutionError(
                f"SyntaxError: Expected '{expected}', got '{self.peek().value}'",
                script_id=self.script_id,
                status=PineExecutionStatus.ERROR,
            )
        return tok

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > 80:
            raise SandboxExecutionError(
                "Maximum parser recursion depth exceeded",
                script_id=self.script_id,
                status=PineExecutionStatus.ERROR,
            )

    def _leave(self) -> None:
        self.depth -= 1

    def parse_program(self) -> List[Any]:
        stmts = []
        while self.peek().type != "EOF":
            if self.match(";"):
                continue
            stmts.append(self.parse_statement())
        return stmts

    def parse_statement(self) -> Any:
        self._enter()
        try:
            tok = self.peek()
            if tok.value == "{":
                return self.parse_block()

            if tok.value == "if":
                self.pos += 1
                self.expect("(")
                cond = self.parse_expression()
                self.expect(")")
                body = self.parse_statement()
                else_body = self.parse_statement() if self.match("else") else None
                return ("if", cond, body, else_body)

            if tok.value == "while":
                self.pos += 1
                self.expect("(")
                cond = self.parse_expression()
                self.expect(")")
                body = self.parse_statement()
                return ("while", cond, body)

            if tok.value == "for":
                self.pos += 1
                self.expect("(")
                init = None
                if self.peek().value in ("const", "let", "var"):
                    self.pos += 1
                    name = self.expect("IDENT").value
                    init_expr = self.parse_assignment() if self.match("=") else None
                    init = ("var_decl", name, init_expr)
                elif self.peek().value != ";":
                    init = ("expr_stmt", self.parse_assignment())
                self.expect(";")
                cond = self.parse_expression() if self.peek().value != ";" else None
                self.expect(";")
                step = self.parse_assignment() if self.peek().value != ")" else None
                self.expect(")")
                body = self.parse_statement()
                return ("for", init, cond, step, body)

            if tok.value == "throw":
                self.pos += 1
                expr = self.parse_expression()
                self.match(";")
                return ("throw", expr)

            if tok.value in ("const", "let", "var"):
                self.pos += 1
                name = self.expect("IDENT").value
                init = self.parse_assignment() if self.match("=") else None
                self.match(";")
                return ("var_decl", name, init)

            expr = self.parse_expression()
            self.match(";")
            return ("expr_stmt", expr)
        finally:
            self._leave()

    def parse_block(self) -> Any:
        self.expect("{")
        stmts = []
        while self.peek().value != "}" and self.peek().type != "EOF":
            if self.match(";"):
                continue
            stmts.append(self.parse_statement())
        self.expect("}")
        return ("block", stmts)

    def parse_expression(self) -> Any:
        return self.parse_assignment()

    def parse_assignment(self) -> Any:
        self._enter()
        try:
            expr = self.parse_ternary()
            tok = self.peek()
            if tok.value in ("=", "+=", "-=", "*=", "/="):
                op = tok.value
                self.pos += 1
                rhs = self.parse_assignment()
                return ("assign", op, expr, rhs)
            return expr
        finally:
            self._leave()

    def parse_ternary(self) -> Any:
        expr = self.parse_logical_or()
        if self.match("?"):
            true_expr = self.parse_assignment()
            self.expect(":")
            false_expr = self.parse_assignment()
            return ("ternary", expr, true_expr, false_expr)
        return expr

    def parse_logical_or(self) -> Any:
        expr = self.parse_logical_and()
        while self.peek().value == "||":
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_logical_and())
        return expr

    def parse_logical_and(self) -> Any:
        expr = self.parse_equality()
        while self.peek().value == "&&":
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_equality())
        return expr

    def parse_equality(self) -> Any:
        expr = self.parse_relational()
        while self.peek().value in ("===", "!==", "==", "!="):
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_relational())
        return expr

    def parse_relational(self) -> Any:
        expr = self.parse_additive()
        while self.peek().value in ("<", "<=", ">", ">="):
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_additive())
        return expr

    def parse_additive(self) -> Any:
        expr = self.parse_multiplicative()
        while self.peek().value in ("+", "-"):
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_multiplicative())
        return expr

    def parse_multiplicative(self) -> Any:
        expr = self.parse_unary()
        while self.peek().value in ("*", "/", "%"):
            op = self.peek().value
            self.pos += 1
            expr = ("bin_op", op, expr, self.parse_unary())
        return expr

    def parse_unary(self) -> Any:
        self._enter()
        try:
            tok = self.peek()
            if tok.value == "typeof":
                self.pos += 1
                return ("typeof", self.parse_unary())
            if tok.value in ("-", "!", "+"):
                self.pos += 1
                return ("unary_op", tok.value, self.parse_unary())
            return self.parse_postfix()
        finally:
            self._leave()

    def parse_postfix(self) -> Any:
        expr = self.parse_primary()
        while True:
            tok = self.peek()
            if tok.value == ".":
                self.pos += 1
                expr = ("member", expr, self.expect("IDENT").value)
            elif tok.value == "(":
                self.pos += 1
                args = self.parse_arg_list()
                self.expect(")")
                expr = ("call", expr, args)
            elif tok.value == "[":
                self.pos += 1
                idx = self.parse_expression()
                self.expect("]")
                expr = ("index", expr, idx)
            elif tok.value == "++":
                self.pos += 1
                expr = ("postfix_inc", expr)
            elif tok.value == "--":
                self.pos += 1
                expr = ("postfix_dec", expr)
            else:
                break
        return expr

    def parse_arg_list(self) -> List[Any]:
        args = []
        if self.peek().value != ")":
            args.append(self.parse_expression())
            while self.match(","):
                args.append(self.parse_expression())
        return args

    def parse_primary(self) -> Any:
        self._enter()
        try:
            tok = self.peek()
            if tok.type == "NUMBER":
                self.pos += 1
                val = float(tok.value) if ("." in tok.value or "e" in tok.value or "E" in tok.value) else int(tok.value)
                return ("number", val)
            if tok.type == "STRING":
                self.pos += 1
                return ("string", tok.value)
            if tok.type == "IDENT":
                self.pos += 1
                if tok.value == "true":
                    return ("bool", True)
                if tok.value == "false":
                    return ("bool", False)
                if tok.value in ("undefined", "null"):
                    return (tok.value, None)
                if tok.value == "new":
                    ctor = self.expect("IDENT").value
                    self.expect("(")
                    args = self.parse_arg_list()
                    self.expect(")")
                    return ("new", ctor, args)
                return ("ident", tok.value)
            if tok.value == "(":
                self.pos += 1
                expr = self.parse_expression()
                self.expect(")")
                return expr
            if tok.value == "[":
                self.pos += 1
                elements = []
                if self.peek().value != "]":
                    elements.append(self.parse_expression())
                    while self.match(","):
                        elements.append(self.parse_expression())
                self.expect("]")
                return ("array_lit", elements)

            raise SandboxExecutionError(
                f"SyntaxError: Unexpected token '{tok.value}' at {tok.pos}",
                script_id=self.script_id,
                status=PineExecutionStatus.ERROR,
            )
        finally:
            self._leave()


# =====================================================================
# Evaluator with Strict Whitelist Boundaries
# =====================================================================


class Evaluator:
    """Executes AST nodes with capability checks and property whitelist verification."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx
        self.env: Dict[str, Any] = {}

    def execute_program(self, stmts: List[Any]) -> None:
        for stmt in stmts:
            self.ctx.check_limits()
            self.execute_stmt(stmt)

    def execute_stmt(self, stmt: Any) -> None:
        tag = stmt[0]
        if tag == "var_decl":
            _, name, init_node = stmt
            val = self.eval_expr(init_node) if init_node is not None else None
            self.env[name] = val
        elif tag == "expr_stmt":
            self.eval_expr(stmt[1])
        elif tag == "throw":
            val = self.eval_expr(stmt[1])
            msg = str(val.args[0]) if isinstance(val, Exception) and val.args else str(val)
            raise SandboxExecutionError(
                msg,
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )
        elif tag == "if":
            _, cond_node, body_node, else_node = stmt
            cond = self.eval_expr(cond_node)
            if isinstance(cond, (list, Series)):
                for bar_idx, is_true in enumerate(cond):
                    if is_true:
                        prev = self.ctx.current_bar_index
                        self.ctx.current_bar_index = bar_idx
                        try:
                            self.execute_stmt(body_node)
                        finally:
                            self.ctx.current_bar_index = prev
                    elif else_node is not None:
                        prev = self.ctx.current_bar_index
                        self.ctx.current_bar_index = bar_idx
                        try:
                            self.execute_stmt(else_node)
                        finally:
                            self.ctx.current_bar_index = prev
            else:
                if cond:
                    self.execute_stmt(body_node)
                elif else_node is not None:
                    self.execute_stmt(else_node)
        elif tag == "while":
            _, cond_node, body_node = stmt
            while True:
                self.ctx.check_limits()
                if not self.eval_expr(cond_node):
                    break
                self.execute_stmt(body_node)
        elif tag == "for":
            _, init_node, cond_node, step_node, body_node = stmt
            if init_node is not None:
                self.execute_stmt(init_node)
            while True:
                self.ctx.check_limits()
                if cond_node is not None and not self.eval_expr(cond_node):
                    break
                self.execute_stmt(body_node)
                if step_node is not None:
                    self.eval_expr(step_node)
        elif tag == "block":
            for s in stmt[1]:
                self.ctx.check_limits()
                self.execute_stmt(s)

    def eval_expr(self, node: Any) -> Any:
        tag = node[0]
        if tag in ("number", "string", "bool"):
            return node[1]
        if tag in ("undefined", "null"):
            return None
        if tag == "array_lit":
            return [self.eval_expr(e) for e in node[1]]

        if tag == "ident":
            name = node[1]
            if name in FORBIDDEN_IDENTIFIERS or name.startswith("_"):
                raise SandboxSecurityViolationError(
                    f"Security violation: access to identifier '{name}' is prohibited",
                    script_id=self.ctx.payload.script_id,
                    status=PineExecutionStatus.ERROR,
                )
            if name in ("globalThis", "self"):
                return self.ctx.global_this
            if name == "ta":
                return self.ctx.ta
            if name == "strategy":
                return self.ctx.strategy
            if name == "plot":
                return self.ctx.plot
            if name == "close":
                return self.ctx.series_close
            if name == "open":
                return self.ctx.series_open
            if name == "high":
                return self.ctx.series_high
            if name == "low":
                return self.ctx.series_low
            if name == "volume":
                return self.ctx.series_volume
            if name in self.env:
                return self.env[name]
            if name in self.ctx.global_this:
                return self.ctx.global_this[name]
            if name in self.ctx.payload.inputs:
                return self.ctx.payload.inputs[name]
            if name == "Array":
                return "Array"
            if name == "Error":
                return Exception
            raise SandboxExecutionError(
                f"ReferenceError: {name} is not defined",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )

        if tag == "member":
            _, target_node, prop = node
            return self._resolve_prop(self.eval_expr(target_node), prop)

        if tag == "index":
            _, target_node, idx_node = node
            return self._resolve_prop(self.eval_expr(target_node), self.eval_expr(idx_node))

        if tag == "call":
            _, func_node, args_nodes = node
            if func_node == ("ident", "Error"):
                msg = self.eval_expr(args_nodes[0]) if args_nodes else ""
                return Exception(str(msg))
            func = self.eval_expr(func_node)
            args = [self.eval_expr(a) for a in args_nodes]
            if callable(func):
                return func(*args)
            raise SandboxExecutionError(
                f"TypeError: {func} is not a function",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )

        if tag == "new":
            _, ctor_name, args_nodes = node
            args = [self.eval_expr(a) for a in args_nodes]
            if ctor_name == "Array":
                size = int(args[0]) if args else 0
                return SimulatedArray(size, self.ctx)
            if ctor_name == "Error":
                return Exception(str(args[0]) if args else "")
            raise SandboxExecutionError(
                f"TypeError: {ctor_name} is not a constructor",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )

        if tag == "assign":
            _, op, target_node, val_node = node
            rhs = self.eval_expr(val_node)

            if op != "=":
                cur_val = self.eval_expr(target_node)
                rhs = self._eval_bin_op(op[:-1], cur_val, rhs)

            if target_node[0] == "ident":
                name = target_node[1]
                if name in FORBIDDEN_IDENTIFIERS or name.startswith("_"):
                    raise SandboxSecurityViolationError(
                        f"Modification of identifier '{name}' is prohibited",
                        script_id=self.ctx.payload.script_id,
                        status=PineExecutionStatus.ERROR,
                    )
                self.env[name] = rhs
            elif target_node[0] == "member":
                self._assign_prop(self.eval_expr(target_node[1]), target_node[2], rhs)
            elif target_node[0] == "index":
                self._assign_prop(self.eval_expr(target_node[1]), self.eval_expr(target_node[2]), rhs)
            return rhs

        if tag == "postfix_inc":
            target_node = node[1]
            if target_node[0] == "ident":
                name = target_node[1]
                old = self.env.get(name, 0)
                self.env[name] = old + 1
                return old
            return 0

        if tag == "postfix_dec":
            target_node = node[1]
            if target_node[0] == "ident":
                name = target_node[1]
                old = self.env.get(name, 0)
                self.env[name] = old - 1
                return old
            return 0

        if tag == "typeof":
            expr_node = node[1]
            if expr_node[0] == "member":
                target = self.eval_expr(expr_node[1])
                prop = expr_node[2]
                if isinstance(target, dict) and prop not in target:
                    return "undefined"
            if expr_node[0] == "ident":
                name = expr_node[1]
                if (
                    name not in self.env
                    and name not in self.ctx.global_this
                    and name not in ("globalThis", "self", "ta", "strategy", "plot", "close", "open", "high", "low", "volume")
                    and name not in self.ctx.payload.inputs
                ):
                    return "undefined"
            val = self.eval_expr(expr_node)
            if val is None:
                return "undefined"
            if isinstance(val, bool):
                return "boolean"
            if isinstance(val, (int, float)):
                return "number"
            if isinstance(val, str):
                return "string"
            if callable(val):
                return "function"
            return "object"

        if tag == "ternary":
            cond = self.eval_expr(node[1])
            return self.eval_expr(node[2]) if cond else self.eval_expr(node[3])

        if tag == "unary_op":
            op, sub = node[1], self.eval_expr(node[2])
            return not bool(sub) if op == "!" else -sub if op == "-" else +sub if op == "+" else sub

        if tag == "bin_op":
            return self._eval_bin_op(node[1], self.eval_expr(node[2]), self.eval_expr(node[3]))

        return None

    def _resolve_prop(self, target: Any, prop: Any) -> Any:
        prop_str = str(prop)
        if (
            prop_str.startswith("_")
            or "__" in prop_str
            or prop_str in FORBIDDEN_IDENTIFIERS
        ):
            raise SandboxSecurityViolationError(
                f"Security violation: access to property '{prop_str}' is prohibited",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )

        if target is None:
            raise SandboxExecutionError(
                f"TypeError: Cannot read properties of undefined (reading '{prop_str}')",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )

        if target is self.ctx.global_this or isinstance(target, dict):
            return target.get(prop_str, None)

        if isinstance(target, TaLib):
            if prop_str in TA_WHITELIST:
                return getattr(target, prop_str)
            raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on ta is prohibited")

        if isinstance(target, StrategyLib):
            if prop_str in STRATEGY_WHITELIST:
                return getattr(target, prop_str)
            raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on strategy is prohibited")

        if isinstance(target, (list, Series, SimulatedArray)):
            if isinstance(prop, int) or prop_str.isdigit():
                idx = int(prop)
                return target[idx] if 0 <= idx < len(target) else None
            if prop_str in ARRAY_WHITELIST:
                if prop_str == "push":
                    return target.push if hasattr(target, "push") else (lambda x: target.append(x) or len(target))
                if prop_str == "length":
                    return len(target)
                if prop_str == "fill" and hasattr(target, "fill"):
                    return target.fill
            raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on Array is prohibited")

        if isinstance(target, (int, float)):
            if prop_str in NUMBER_WHITELIST:
                return lambda: str(target)
            raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on number is prohibited")

        if isinstance(target, str):
            if prop_str in STRING_WHITELIST:
                if prop_str == "length":
                    return len(target)
                if prop_str == "toString":
                    return lambda: str(target)
            raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on string is prohibited")

        raise SandboxSecurityViolationError(f"Security violation: access to '{prop_str}' on {type(target).__name__} is prohibited")

    def _assign_prop(self, target: Any, prop: Any, val: Any) -> None:
        prop_str = str(prop)
        if (
            prop_str.startswith("_")
            or "__" in prop_str
            or prop_str in FORBIDDEN_IDENTIFIERS
        ):
            raise SandboxSecurityViolationError(
                f"Security violation: modification of property '{prop_str}' is prohibited",
                script_id=self.ctx.payload.script_id,
                status=PineExecutionStatus.ERROR,
            )
        if isinstance(target, dict):
            target[prop_str] = val
        elif isinstance(target, list) and isinstance(prop, int):
            if 0 <= prop < len(target):
                target[prop] = val
        else:
            raise SandboxSecurityViolationError("Property assignment is prohibited on this object")

    def _eval_bin_op(self, op: str, left: Any, right: Any) -> Any:
        if isinstance(left, (list, Series)) or isinstance(right, (list, Series)):
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op in ("==", "==="):
                return left == right
            if op in ("!=", "!=="):
                return left != right
            if op in ("&&", "&"):
                n = len(left) if isinstance(left, (list, Series)) else len(right)
                l_arr = left if isinstance(left, (list, Series)) else [bool(left)] * n
                r_arr = right if isinstance(right, (list, Series)) else [bool(right)] * n
                return Series([bool(a and b) for a, b in zip(l_arr, r_arr)])
            if op in ("||", "|"):
                n = len(left) if isinstance(left, (list, Series)) else len(right)
                l_arr = left if isinstance(left, (list, Series)) else [bool(left)] * n
                r_arr = right if isinstance(right, (list, Series)) else [bool(right)] * n
                return Series([bool(a or b) for a, b in zip(l_arr, r_arr)])

        if op == "+":
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right if right != 0 else float("nan")
        if op in ("===", "=="):
            if (left is None and right == "undefined") or (right is None and left == "undefined"):
                return True
            return left == right
        if op in ("!==", "!="):
            if (left is None and right == "undefined") or (right is None and left == "undefined"):
                return False
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "&&":
            return left and right
        if op == "||":
            return left or right

        return None


# =====================================================================
# Isolated Worker Runner Process
# =====================================================================


def _apply_os_limits(limits: SandboxExecutionLimits) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except Exception:
        pass
    try:
        cpu_sec = max(1, int(math.ceil(limits.timeout_ms / 1000.0) + 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
    except Exception:
        pass


def _worker_process_main(
    pipe_conn: Any,
    payload: PineCompiledPayload,
    series: MarketSeriesData,
    limits: SandboxExecutionLimits,
) -> None:
    try:
        _apply_os_limits(limits)
        tracemalloc.start()
        ctx = ExecutionContext(payload=payload, series=series, limits=limits)

        tokens = tokenize(payload.bytecode)
        parser = Parser(tokens, script_id=payload.script_id)
        stmts = parser.parse_program()

        evaluator = Evaluator(ctx)
        evaluator.execute_program(stmts)

        elapsed_ms = max(0.01, (time.monotonic() - ctx.start_time) * 1000.0)
        _, peak_mem = tracemalloc.get_traced_memory()
        mem_used = max(1024, peak_mem)

        res = PineExecutionResult(
            status=PineExecutionStatus.SUCCESS,
            signals=ctx.signals,
            indicator_outputs=ctx.indicator_outputs,
            execution_time_ms=elapsed_ms,
            memory_used_bytes=mem_used,
        )
        pipe_conn.send((True, res))
    except (
        SandboxExecutionError,
        SandboxTimeoutError,
        SandboxMemoryLimitError,
        SandboxSecurityViolationError,
    ) as exc:
        pipe_conn.send((False, exc))
    except RecursionError:
        pipe_conn.send(
            (
                False,
                SandboxExecutionError(
                    "Parser/Runtime recursion limit exceeded",
                    script_id=payload.script_id,
                    status=PineExecutionStatus.ERROR,
                ),
            )
        )
    except MemoryError:
        pipe_conn.send(
            (
                False,
                SandboxMemoryLimitError(
                    f"Memory limit of {limits.memory_limit_mb} MB exceeded",
                    script_id=payload.script_id,
                    status=PineExecutionStatus.MEMORY_EXCEEDED,
                ),
            )
        )
    except Exception as exc:
        pipe_conn.send(
            (
                False,
                SandboxExecutionError(
                    str(exc),
                    script_id=payload.script_id,
                    status=PineExecutionStatus.ERROR,
                ),
            )
        )
    finally:
        if tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except Exception:
                pass
        try:
            pipe_conn.close()
        except Exception:
            pass


# =====================================================================
# Sandbox Orchestrator
# =====================================================================


class PineWorkerSandbox:
    """Isolated WebWorker execution sandbox orchestrator for compiled Pine Script payloads."""

    FORBIDDEN_PATTERNS = [
        r"\brequire\b",
        r"\bprocess\b",
        r"\bchild_process\b",
        r"\bfs\b",
        r"\beval\b",
        r"\bFunction\b",
        r"\bconstructor\b",
        r"\bprototype\b",
        r"\b__proto__\b",
        r"\b__dirname\b",
        r"\b__filename\b",
        r"\bimport\b",
        r"\bexec\b",
        r"\bspawn\b",
        r"\b__class__\b",
        r"\b__bases__\b",
        r"\b__mro__\b",
        r"\b__subclasses__\b",
        r"\b__globals__\b",
        r"\b__builtins__\b",
        r"\b__dict__\b",
        r"\b__code__\b",
    ]

    def __init__(self) -> None:
        self._active_workers: int = 0
        self._lock = threading.Lock()

    def get_active_worker_count(self) -> int:
        """Returns the number of currently active sandboxed worker processes."""
        with self._lock:
            return self._active_workers

    def _verify_security_boundaries(self, payload: PineCompiledPayload) -> None:
        """Scans payload bytecode to block host escape primitives before execution."""
        code = payload.bytecode
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                raise SandboxSecurityViolationError(
                    f"Security violation: unauthorized host primitive access in script {payload.script_id}",
                    script_id=payload.script_id,
                    status=PineExecutionStatus.ERROR,
                )

    def execute(
        self,
        payload: PineCompiledPayload,
        series: MarketSeriesData,
        limits: Optional[SandboxExecutionLimits] = None,
    ) -> PineExecutionResult:
        """Executes the script in an isolated runner context within bounded limits."""
        if limits is None:
            limits = SandboxExecutionLimits()

        with self._lock:
            self._active_workers += 1

        proc: Optional[multiprocessing.Process] = None
        parent_conn = None

        try:
            self._verify_security_boundaries(payload)

            mp_ctx = multiprocessing.get_context(
                "fork" if "fork" in multiprocessing.get_all_start_methods() else None
            )
            parent_conn, child_conn = mp_ctx.Pipe()

            proc = mp_ctx.Process(
                target=_worker_process_main,
                args=(child_conn, payload, series, limits),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()

            timeout_sec = limits.timeout_ms / 1000.0
            proc.join(timeout=timeout_sec + 0.3)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=0.05)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=0.05)
                raise SandboxTimeoutError(
                    f"Execution timed out after {limits.timeout_ms} ms",
                    script_id=payload.script_id,
                    status=PineExecutionStatus.TIMEOUT,
                )

            if parent_conn.poll():
                try:
                    success, result_or_exc = parent_conn.recv()
                    if success:
                        return result_or_exc
                    else:
                        raise result_or_exc
                except EOFError:
                    pass

            if proc.exitcode is not None and proc.exitcode != 0:
                if proc.exitcode in (-9, 137, -11):
                    raise SandboxMemoryLimitError(
                        f"Memory limit of {limits.memory_limit_mb} MB exceeded (terminated by OS)",
                        script_id=payload.script_id,
                        status=PineExecutionStatus.MEMORY_EXCEEDED,
                    )
                raise SandboxExecutionError(
                    f"Worker runner crashed with exit code {proc.exitcode}",
                    script_id=payload.script_id,
                    status=PineExecutionStatus.ERROR,
                )

            raise SandboxExecutionError(
                "Worker runner terminated without returning output",
                script_id=payload.script_id,
                status=PineExecutionStatus.ERROR,
            )
        finally:
            if parent_conn is not None:
                try:
                    parent_conn.close()
                except Exception:
                    pass
            if proc is not None:
                if proc.is_alive():
                    try:
                        proc.terminate()
                        proc.join(timeout=0.05)
                    except Exception:
                        pass
                    if proc.is_alive():
                        try:
                            proc.kill()
                            proc.join(timeout=0.05)
                        except Exception:
                            pass
                try:
                    proc.close()
                except Exception:
                    pass
            with self._lock:
                self._active_workers -= 1