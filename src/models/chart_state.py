from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ChartState:
    """Represents the persisted state and version of a chart."""

    chart_id: str
    version: int
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError(f"Version must be non-negative, got {self.version}")
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


@dataclass
class ChartDelta:
    """Represents an incremental change payload applied against an expected chart version."""

    chart_id: str
    expected_version: int
    changes: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_version < 0:
            raise ValueError(
                f"Expected version must be non-negative, got {self.expected_version}"
            )