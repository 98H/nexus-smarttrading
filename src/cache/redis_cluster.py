"""Redis Cluster L2 caching tier implementation."""

from typing import Any, Optional


class RedisClusterCache:
    """Redis Cluster L2 caching tier wrapper."""

    def __init__(self, client: Any = None) -> None:
        self.client = client

    def get(self, key: str) -> Any:
        """Retrieve key value from Redis cluster client."""
        return self.client.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> Any:
        """Store key-value pair in Redis cluster with optional expiration in seconds."""
        return self.client.set(key, value, ex=ttl)

    def delete(self, key: str) -> bool:
        """Delete key from Redis cluster, returning True if removed, False otherwise."""
        result = self.client.delete(key)
        return bool(result)