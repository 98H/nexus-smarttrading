"""Multi-tier cache layer coordinating L1 in-memory and L2 Redis cluster tiers."""

from typing import Any, Optional


class MultiTierCache:
    """Multi-tier cache coordinator combining L1 (in-memory) and L2 (Redis cluster)."""

    def __init__(self, l1_cache: Any, l2_cache: Any, default_ttl: int = 300) -> None:
        if default_ttl is not None:
            if isinstance(default_ttl, bool) or not isinstance(default_ttl, (int, float)) or default_ttl <= 0:
                raise ValueError("default_ttl must be greater than 0")
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.default_ttl = default_ttl

    def _validate_key(self, key: Any) -> None:
        if key is None or not isinstance(key, str):
            raise TypeError(f"Cache key must be a string, got {type(key).__name__}")
        if not key:
            raise ValueError("Cache key cannot be empty")

    def get(self, key: str) -> Any:
        """Fetch value for key from L1, falling back to L2 and backfilling L1 on hit."""
        self._validate_key(key)

        value = self.l1_cache.get(key)
        if value is not None:
            return value

        value = self.l2_cache.get(key)
        if value is not None:
            self.l1_cache.set(key, value)
            return value

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store key-value pair in both L1 and L2 with specified or default TTL."""
        self._validate_key(key)

        if ttl is not None:
            if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
                raise ValueError("TTL must be greater than 0")
            effective_ttl = ttl
        else:
            effective_ttl = self.default_ttl

        self.l1_cache.set(key, value, ttl=effective_ttl)
        self.l2_cache.set(key, value, ttl=effective_ttl)

    def delete(self, key: str) -> None:
        """Remove key from both L1 and L2 cache tiers."""
        self._validate_key(key)
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)