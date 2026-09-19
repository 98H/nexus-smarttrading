import pytest
from unittest.mock import MagicMock, create_autospec

from src.cache.multi_tier import MultiTierCache
from src.cache.redis_cluster import RedisClusterCache


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_l1():
    """Mock L1 in-memory cache layer."""
    l1 = MagicMock()
    # Default behavior: cache miss returns None
    l1.get.return_value = None
    return l1


@pytest.fixture
def mock_l2():
    """Mock L2 Redis cluster cache layer."""
    l2 = create_autospec(RedisClusterCache, instance=True)
    # Default behavior: cache miss returns None
    l2.get.return_value = None
    return l2


@pytest.fixture
def multi_tier_cache(mock_l1, mock_l2):
    """MultiTierCache instance configured with mocked L1 and L2 tiers."""
    return MultiTierCache(l1_cache=mock_l1, l2_cache=mock_l2, default_ttl=300)


@pytest.fixture
def mock_redis_cluster_client():
    """Mock underlying RedisCluster client for RedisClusterCache tests."""
    return MagicMock()


@pytest.fixture
def redis_cluster_cache(mock_redis_cluster_client):
    """RedisClusterCache instance with an injected mock cluster client."""
    return RedisClusterCache(client=mock_redis_cluster_client)


# =====================================================================
# Story 11.2.2: MultiTierCache Unit Tests
# =====================================================================

class TestMultiTierCacheL1Hit:
    """Acceptance Criterion 1:

    Given a key exists in the L1 in-memory cache,
    When get(key) is called,
    Then return the value without querying the L2 Redis cluster.
    """

    def test_get_returns_l1_value_without_querying_l2(self, multi_tier_cache, mock_l1, mock_l2):
        key = "session:user_123"
        expected_value = {"user_id": 123, "role": "admin"}
        mock_l1.get.return_value = expected_value

        result = multi_tier_cache.get(key)

        assert result == expected_value
        mock_l1.get.assert_called_once_with(key)
        mock_l2.get.assert_not_called()

    def test_get_l1_hit_does_not_trigger_l1_backfill(self, multi_tier_cache, mock_l1, mock_l2):
        key = "config:app_mode"
        mock_l1.get.return_value = "production"

        multi_tier_cache.get(key)

        mock_l1.set.assert_not_called()
        mock_l2.get.assert_not_called()


class TestMultiTierCacheL1MissL2Hit:
    """Acceptance Criterion 2:

    Given a key is missing from L1 but exists in the L2 Redis cluster,
    When get(key) is called,
    Then fetch the value from L2, backfill L1, and return the value.
    """

    def test_get_l1_miss_fetches_from_l2_backfills_l1_and_returns_value(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        key = "session:user_456"
        expected_value = {"user_id": 456, "role": "member"}
        mock_l1.get.return_value = None
        mock_l2.get.return_value = expected_value

        result = multi_tier_cache.get(key)

        assert result == expected_value
        mock_l1.get.assert_called_once_with(key)
        mock_l2.get.assert_called_once_with(key)
        mock_l1.set.assert_called_once_with(key, expected_value)

    def test_subsequent_get_hits_l1_after_l2_backfill(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        key = "session:user_789"
        expected_value = {"user_id": 789}

        # First call: L1 miss, L2 hit -> backfills L1
        mock_l1.get.return_value = None
        mock_l2.get.return_value = expected_value
        result_first = multi_tier_cache.get(key)
        assert result_first == expected_value

        # Simulate that L1 now contains the value for the second call
        mock_l1.get.return_value = expected_value
        result_second = multi_tier_cache.get(key)
        assert result_second == expected_value

        # L2 should still have only been called once (from the first get)
        assert mock_l2.get.call_count == 1


class TestMultiTierCacheSet:
    """Acceptance Criterion 3:

    Given a key and value,
    When set(key, value, ttl) is called,
    Then store the key-value pair in both L1 memory and the L2 Redis cluster
    with the specified TTL.
    """

    def test_set_stores_in_both_l1_and_l2_with_explicit_ttl(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        key = "item:100"
        value = {"name": "Widget", "price": 9.99}
        ttl = 60

        multi_tier_cache.set(key, value, ttl=ttl)

        mock_l1.set.assert_called_once_with(key, value, ttl=ttl)
        mock_l2.set.assert_called_once_with(key, value, ttl=ttl)

    def test_set_stores_in_both_l1_and_l2_with_default_ttl(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        key = "item:200"
        value = {"name": "Gadget"}

        multi_tier_cache.set(key, value)

        # multi_tier_cache initialized with default_ttl=300
        mock_l1.set.assert_called_once_with(key, value, ttl=300)
        mock_l2.set.assert_called_once_with(key, value, ttl=300)


class TestMultiTierCacheEdgeCasesAndErrors:
    """Edge cases, cache misses across all layers, and input validation."""

    def test_get_both_miss_returns_none(self, multi_tier_cache, mock_l1, mock_l2):
        key = "nonexistent:key"
        mock_l1.get.return_value = None
        mock_l2.get.return_value = None

        result = multi_tier_cache.get(key)

        assert result is None
        mock_l1.get.assert_called_once_with(key)
        mock_l2.get.assert_called_once_with(key)
        mock_l1.set.assert_not_called()

    def test_set_with_invalid_ttl_raises_value_error(self, multi_tier_cache):
        with pytest.raises(ValueError):
            multi_tier_cache.set("key", "value", ttl=-1)

        with pytest.raises(ValueError):
            multi_tier_cache.set("key", "value", ttl=0)

    def test_set_with_empty_or_invalid_key_raises_value_error(self, multi_tier_cache):
        with pytest.raises(ValueError):
            multi_tier_cache.set("", "value", ttl=60)

        with pytest.raises(TypeError):
            multi_tier_cache.set(None, "value", ttl=60)

    def test_get_with_invalid_key_raises_exception(self, multi_tier_cache):
        with pytest.raises(ValueError):
            multi_tier_cache.get("")

        with pytest.raises(TypeError):
            multi_tier_cache.get(None)

    def test_delete_removes_key_from_both_tiers(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        key = "session:to_delete"

        multi_tier_cache.delete(key)

        mock_l1.delete.assert_called_once_with(key)
        mock_l2.delete.assert_called_once_with(key)

    def test_get_l1_hit_resilient_when_l2_fails(
        self, multi_tier_cache, mock_l1, mock_l2
    ):
        """If L1 has the key, L2 cluster failure must not affect the read."""
        key = "resilient:key"
        mock_l1.get.return_value = "healthy_value"
        mock_l2.get.side_effect = ConnectionError("Redis cluster unreachable")

        result = multi_tier_cache.get(key)

        assert result == "healthy_value"
        mock_l2.get.assert_not_called()


# =====================================================================
# Story 11.2.2: RedisClusterCache Unit Tests
# =====================================================================

class TestRedisClusterCache:
    """Tests for the RedisClusterCache L2 wrapper implementation."""

    def test_get_queries_redis_cluster_client(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        key = "cache:key1"
        mock_redis_cluster_client.get.return_value = b'{"data": 123}'

        result = redis_cluster_cache.get(key)

        assert result == b'{"data": 123}'
        mock_redis_cluster_client.get.assert_called_once_with(key)

    def test_get_returns_none_when_key_does_not_exist(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        mock_redis_cluster_client.get.return_value = None

        result = redis_cluster_cache.get("missing:key")

        assert result is None

    def test_set_with_ttl_passes_ex_to_redis_cluster(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        key = "cache:key2"
        value = "sample_value"
        ttl = 120

        redis_cluster_cache.set(key, value, ttl=ttl)

        mock_redis_cluster_client.set.assert_called_once_with(key, value, ex=ttl)

    def test_set_without_ttl_stores_indefinitely(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        key = "cache:key_no_ttl"
        value = "persistent_value"

        redis_cluster_cache.set(key, value, ttl=None)

        mock_redis_cluster_client.set.assert_called_once_with(key, value, ex=None)

    def test_delete_delegates_to_redis_cluster_client(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        key = "cache:to_remove"
        mock_redis_cluster_client.delete.return_value = 1

        result = redis_cluster_cache.delete(key)

        assert result is True
        mock_redis_cluster_client.delete.assert_called_once_with(key)

    def test_delete_returns_false_when_key_not_found(
        self, redis_cluster_cache, mock_redis_cluster_client
    ):
        key = "cache:missing_remove"
        mock_redis_cluster_client.delete.return_value = 0

        result = redis_cluster_cache.delete(key)

        assert result is False
        mock_redis_cluster_client.delete.assert_called_once_with(key)