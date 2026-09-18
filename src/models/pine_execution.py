"""
Data models and exceptions for Pine Script Sandboxed WebWorker Execution.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import re
from typing import Any, Dict, List, Optional, Set


def _init_sensitive_roots() -> List[str]:
    """Precomputes and caches sensitive host filesystem roots at module initialization."""
    roots: Set[str] = set()
    for getter in (
        os.getcwd,
        lambda: os.path.expanduser("~"),
        lambda: os.environ.get("HOME", ""),
        lambda: os.environ.get("VIRTUAL_ENV", ""),
    ):
        try:
            p = getter()
            if p and p != "/" and os.path.isabs(p):
                roots.add(p)
                roots.add(os.path.realpath(p))
        except Exception:
            pass
    return sorted(roots, key=len, reverse=True)


_CACHED_SENSITIVE_ROOTS: List[str] = _init_sensitive_roots()

_WIN_PATH_RE = re.compile(
    r"[a-zA-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
)

# Known Unix system mount roots, anchored to prevent matching arithmetic expressions or labels
_SYSTEM_MOUNTS = (
    r"bin|boot|dev|etc|home|lib|lib64|media|mnt|opt|proc|root|run|sbin|srv|sys|tmp|usr|var|private|Volumes"
)
_UNIX_SYSTEM_PATH_RE = re.compile(
    r"(?<![a-zA-Z0-9_.~-])/(?:(?:" + _SYSTEM_MOUNTS + r")(?:/[a-zA-Z0-9_.~-]+)+)(?=[^\w.~-]|$)"
)
_REDUNDANT_REDACTION_RE = re.compile(r"\[REDACTED_PATH\](?:/[a-zA-Z0-9_.~-]+)+")


def sanitize_host_paths(data: Any, seen: Optional[Set[int]] = None) -> Any:
    """Sanitizes host file paths and system mounts without altering non-filesystem tokens."""
    if data is None:
        return None

    if seen is None:
        seen = set()

    obj_id = id(data)
    if isinstance(data, (dict, list, tuple, set)):
        if obj_id in seen:
            return "[CIRCULAR_REFERENCE]"
        seen.add(obj_id)

    try:
        if isinstance(data, dict):
            return {k: sanitize_host_paths(v, seen) for k, v in data.items()}
        if isinstance(data, list):
            return [sanitize_host_paths(v, seen) for v in data]
        if isinstance(data, tuple):
            return tuple(sanitize_host_paths(v, seen) for v in data)
        if isinstance(data, set):
            return {sanitize_host_paths(v, seen) for v in data}
        if isinstance(data, Exception):
            return sanitize_host_paths(str(data), seen)
        if not isinstance(data, str):
            return data

        s = data

        # 1. Exact match against cached known sensitive host roots
        for root in _CACHED_SENSITIVE_ROOTS:
            if root in s:
                s = s.replace(root, "[REDACTED_PATH]")

        # 2. Windows absolute filesystem paths
        s = _WIN_PATH_RE.sub("[REDACTED_PATH]", s)

        # 3. Known Unix system mounts and standard directories
        s = _UNIX_SYSTEM_PATH_RE.sub("[REDACTED_PATH]", s)

        # 4. Clean up redundant nested redactions
        s = _REDUNDANT_REDACTION_RE.sub("[REDACTED_PATH]", s)

        return s
    finally:
        if isinstance(data, (dict, list, tuple, set)):
            seen.discard(obj_id)


class PineExecutionStatus(str, Enum):
    """Execution status outcomes for sandboxed Pine script runs."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"


class PineSignalType(str, Enum):
    """Trading signal directions produced by Pine strategies."""

    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


@dataclass
class MarketSeriesData:
    """Deterministic OHLCV market time series data."""

    timestamps: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if n == 0:
            raise ValueError("Market series data cannot be empty")
        for field_name, arr in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"All OHLCV arrays must have matching dimensions. "
                    f"Expected length {n}, got {len(arr)} for {field_name}"
                )


@dataclass
class PineCompiledPayload:
    """Compiled Pine Script IR/bytecode payload and input configurations."""

    script_id: str
    version: str
    bytecode: str
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxExecutionLimits:
    """Boundary limits enforced during script execution."""

    timeout_ms: int = 1000
    memory_limit_mb: int = 64
    max_output_size_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be strictly positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be strictly positive")
        if self.max_output_size_bytes <= 0:
            raise ValueError("max_output_size_bytes must be strictly positive")


@dataclass
class PineSignal:
    """Normalized trading signal generated by a Pine Script strategy."""

    timestamp: int
    signal_type: PineSignalType
    price: float
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PineIndicatorOutput:
    """Normalized plotted indicator series output."""

    name: str
    values: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PineExecutionResult:
    """Comprehensive result of a completed Pine Script execution."""

    status: PineExecutionStatus
    signals: List[PineSignal]
    indicator_outputs: Dict[str, PineIndicatorOutput]
    execution_time_ms: float
    memory_used_bytes: int


# =====================================================================
# Structured Sandbox Exceptions
# =====================================================================


class SandboxExecutionError(Exception):
    """Base structured error raised when sandboxed execution fails."""

    def __init__(
        self,
        message: str,
        script_id: Optional[str] = None,
        status: PineExecutionStatus = PineExecutionStatus.ERROR,
        details: Any = None,
    ) -> None:
        clean_message = str(sanitize_host_paths(message))
        super().__init__(clean_message)
        self.message = clean_message
        self.script_id = script_id
        self.status = status
        self.details = (
            sanitize_host_paths(details) if details is not None else clean_message
        )

    def __reduce__(self) -> Any:
        return (
            self.__class__,
            (self.message, self.script_id, self.status, self.details),
        )


class SandboxTimeoutError(SandboxExecutionError):
    """Raised when script execution exceeds configured execution timeout limit."""

    def __init__(
        self,
        message: str,
        script_id: Optional[str] = None,
        status: PineExecutionStatus = PineExecutionStatus.TIMEOUT,
        details: Any = None,
    ) -> None:
        super().__init__(message, script_id=script_id, status=status, details=details)


class SandboxMemoryLimitError(SandboxExecutionError):
    """Raised when memory allocated during execution exceeds bounded limits."""

    def __init__(
        self,
        message: str,
        script_id: Optional[str] = None,
        status: PineExecutionStatus = PineExecutionStatus.MEMORY_EXCEEDED,
        details: Any = None,
    ) -> None:
        super().__init__(message, script_id=script_id, status=status, details=details)


class SandboxSecurityViolationError(SandboxExecutionError):
    """Raised when script attempts access to restricted host primitives or runtime escaping."""

    def __init__(
        self,
        message: str,
        script_id: Optional[str] = None,
        status: PineExecutionStatus = PineExecutionStatus.ERROR,
        details: Any = None,
    ) -> None:
        super().__init__(message, script_id=script_id, status=status, details=details)