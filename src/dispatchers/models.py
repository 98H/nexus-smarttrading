"""Notification data models for multi-platform dispatchers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Notification:
    """Outbound notification data structure."""

    platform: str
    recipient_id: str
    message: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None