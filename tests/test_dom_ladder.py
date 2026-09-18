"""
Unit tests for Real-Time Visual DOM Ladder and Bid/Ask Imbalance Meter.

Specification / Story 2.4.2:
- generate_dom_ladder: Takes an order book state and a depth limit, returning a structured
  DOM ladder with sorted bids (descending) and asks (ascending) limited to depth.
- calculate_order_imbalance: Computes normalized imbalance score in range [-1.0, 1.0]
  via (bid_volume - ask_volume) / (bid_volume + ask_volume).
- Zero-division safety: Safely returns 0.0 for empty or zero-volume order books.
"""

from typing import Any, List, Tuple
import pytest

from src.analytics.dom_ladder import generate_dom_ladder
from src.analytics.imbalance import calculate_order_imbalance


# ============================================================================
# Helpers for Deterministic Test Evaluation
# ============================================================================

def _extract_bids_asks(ladder: Any) -> Tuple[List[Any], List[Any]]:
    """Extract bids and asks lists from dictionary or structured object."""
    if hasattr(ladder, "bids") and hasattr(ladder, "asks"):
        return ladder.bids, ladder.asks
    if isinstance(ladder, dict) and "bids" in ladder and "asks" in ladder:
        return ladder["bids"], ladder["asks"]
    raise TypeError(f"DOM ladder returned unexpected structure: {type(ladder)}")


def _normalize_levels(levels: List[Any]) -> List[Tuple[float, float]]:
    """Normalize levels to a uniform list of (price, volume) tuples."""
    normalized = []
    for level in levels:
        if isinstance(level, (tuple, list)) and len(level) >= 2:
            normalized.append((float(level[0]), float(level[1])))
        elif hasattr(level, "price") and hasattr(level, "volume"):
            normalized.append((float(level.price), float(level.volume)))
        elif isinstance(level, dict) and "price" in level and "volume" in level:
            normalized.append((float(level["price"]), float(level["volume"])))
        else:
            raise TypeError(f"Unexpected level format in ladder: {type(level)}")
    return normalized


# ============================================================================
# DOM Ladder Unit Tests (src/analytics/dom_ladder.py)
# ============================================================================

class TestGenerateDomLadder:
    """Tests for generate_dom_ladder according to Story 2.4.2."""

    def test_bids_sorted_descending_by_price(self):
        """Bids must be ordered from highest price to lowest price."""
        order_book = {
            "bids": [
                (100.0, 1.5),
                (105.5, 2.0),
                (98.0, 5.0),
                (102.0, 0.5),
            ],
            "asks": [
                (106.0, 1.0),
                (107.0, 2.0),
            ],
        }

        ladder = generate_dom_ladder(order_book, depth=4)
        bids_raw, _ = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)

        expected_bids = [
            (105.5, 2.0),
            (102.0, 0.5),
            (100.0, 1.5),
            (98.0, 5.0),
        ]
        assert bids == expected_bids

    def test_asks_sorted_ascending_by_price(self):
        """Asks must be ordered from lowest price to highest price."""
        order_book = {
            "bids": [
                (99.0, 1.0),
            ],
            "asks": [
                (105.0, 1.5),
                (101.5, 0.8),
                (103.0, 2.5),
                (108.0, 4.0),
            ],
        }

        ladder = generate_dom_ladder(order_book, depth=4)
        _, asks_raw = _extract_bids_asks(ladder)
        asks = _normalize_levels(asks_raw)

        expected_asks = [
            (101.5, 0.8),
            (103.0, 2.5),
            (105.0, 1.5),
            (108.0, 4.0),
        ]
        assert asks == expected_asks

    def test_truncates_to_specified_depth_limit(self):
        """DOM ladder must limit both bids and asks to the configured depth."""
        order_book = {
            "bids": [
                (100.0, 1.0),
                (101.0, 2.0),
                (102.0, 3.0),
                (103.0, 4.0),
                (104.0, 5.0),
            ],
            "asks": [
                (105.0, 1.0),
                (106.0, 2.0),
                (107.0, 3.0),
                (108.0, 4.0),
                (109.0, 5.0),
            ],
        }

        ladder = generate_dom_ladder(order_book, depth=3)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        assert len(bids) == 3
        assert len(asks) == 3
        assert bids == [(104.0, 5.0), (103.0, 4.0), (102.0, 3.0)]
        assert asks == [(105.0, 1.0), (106.0, 2.0), (107.0, 3.0)]

    def test_depth_greater_than_available_levels(self):
        """When depth limit exceeds order book levels, returns all available levels."""
        order_book = {
            "bids": [(100.0, 1.0), (99.0, 2.0)],
            "asks": [(101.0, 1.5)],
        }

        ladder = generate_dom_ladder(order_book, depth=10)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        assert len(bids) == 2
        assert len(asks) == 1
        assert bids == [(100.0, 1.0), (99.0, 2.0)]
        assert asks == [(101.0, 1.5)]

    def test_depth_one_returns_top_of_book(self):
        """Depth 1 returns only the best bid and best ask."""
        order_book = {
            "bids": [(100.0, 10.0), (105.0, 5.0), (95.0, 2.0)],
            "asks": [(110.0, 8.0), (106.0, 4.0), (120.0, 1.0)],
        }

        ladder = generate_dom_ladder(order_book, depth=1)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        assert len(bids) == 1
        assert len(asks) == 1
        assert bids[0] == (105.0, 5.0)  # Highest bid
        assert asks[0] == (106.0, 4.0)  # Lowest ask

    def test_depth_zero_returns_empty_ladder(self):
        """Depth 0 returns empty bid and ask ladders."""
        order_book = {
            "bids": [(100.0, 1.0), (99.0, 2.0)],
            "asks": [(101.0, 1.0), (102.0, 2.0)],
        }

        ladder = generate_dom_ladder(order_book, depth=0)
        bids_raw, asks_raw = _extract_bids_asks(ladder)

        assert len(bids_raw) == 0
        assert len(asks_raw) == 0

    def test_empty_order_book_input(self):
        """An order book with no bids and no asks returns empty ladders."""
        order_book = {"bids": [], "asks": []}

        ladder = generate_dom_ladder(order_book, depth=5)
        bids_raw, asks_raw = _extract_bids_asks(ladder)

        assert len(bids_raw) == 0
        assert len(asks_raw) == 0

    def test_one_sided_order_book(self):
        """Handles books with bids only or asks only safely."""
        book_bids_only = {"bids": [(100.0, 1.0), (99.0, 2.0)], "asks": []}
        ladder_bids = generate_dom_ladder(book_bids_only, depth=2)
        bids_b, asks_b = _extract_bids_asks(ladder_bids)

        assert len(bids_b) == 2
        assert len(asks_b) == 0

        book_asks_only = {"bids": [], "asks": [(101.0, 3.0), (102.0, 4.0)]}
        ladder_asks = generate_dom_ladder(book_asks_only, depth=2)
        bids_a, asks_a = _extract_bids_asks(ladder_asks)

        assert len(bids_a) == 0
        assert len(asks_a) == 2

    def test_preserves_volumes_accurately_with_prices(self):
        """Price-volume mapping must remain intact during sorting."""
        order_book = {
            "bids": [(10.0, 100.0), (30.0, 300.0), (20.0, 200.0)],
            "asks": [(60.0, 600.0), (40.0, 400.0), (50.0, 500.0)],
        }

        ladder = generate_dom_ladder(order_book, depth=3)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        assert bids == [(30.0, 300.0), (20.0, 200.0), (10.0, 100.0)]
        assert asks == [(40.0, 400.0), (50.0, 500.0), (60.0, 600.0)]

    def test_negative_depth_raises_value_error(self):
        """A negative depth value is invalid and must raise ValueError."""
        order_book = {"bids": [(100.0, 1.0)], "asks": [(101.0, 1.0)]}
        with pytest.raises(ValueError):
            generate_dom_ladder(order_book, depth=-1)

    def test_non_integer_depth_raises_type_error(self):
        """Non-integer depth parameters must raise TypeError."""
        order_book = {"bids": [(100.0, 1.0)], "asks": [(101.0, 1.0)]}
        with pytest.raises(TypeError):
            generate_dom_ladder(order_book, depth="5")  # type: ignore

    def test_high_precision_floating_prices(self):
        """Handles fractional micro-tick prices correctly."""
        order_book = {
            "bids": [(0.00010002, 50.0), (0.00010008, 10.0), (0.00010005, 30.0)],
            "asks": [(0.00010012, 40.0), (0.00010009, 20.0), (0.00010015, 60.0)],
        }

        ladder = generate_dom_ladder(order_book, depth=3)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        assert bids[0] == (0.00010008, 10.0)
        assert bids[1] == (0.00010005, 30.0)
        assert bids[2] == (0.00010002, 50.0)

        assert asks[0] == (0.00010009, 20.0)
        assert asks[1] == (0.00010012, 40.0)
        assert asks[2] == (0.00010015, 60.0)


# ============================================================================
# Imbalance Meter Unit Tests (src/analytics/imbalance.py)
# ============================================================================

class TestCalculateOrderImbalance:
    """Tests for calculate_order_imbalance according to Story 2.4.2."""

    def test_equal_bid_and_ask_volume_yields_zero(self):
        """Balanced book (bid_volume == ask_volume) results in an imbalance score of 0.0."""
        score = calculate_order_imbalance(bid_volume=50.0, ask_volume=50.0)
        assert score == pytest.approx(0.0)

    def test_pure_bid_volume_yields_max_positive(self):
        """100% bid volume and 0 ask volume results in score of 1.0."""
        score = calculate_order_imbalance(bid_volume=100.0, ask_volume=0.0)
        assert score == pytest.approx(1.0)

    def test_pure_ask_volume_yields_max_negative(self):
        """0 bid volume and 100% ask volume results in score of -1.0."""
        score = calculate_order_imbalance(bid_volume=0.0, ask_volume=100.0)
        assert score == pytest.approx(-1.0)

    def test_bid_heavy_imbalance_calculation(self):
        """Calculates (75 - 25) / (75 + 25) = 50 / 100 = 0.5 accurately."""
        score = calculate_order_imbalance(bid_volume=75.0, ask_volume=25.0)
        assert score == pytest.approx(0.5)

    def test_ask_heavy_imbalance_calculation(self):
        """Calculates (20 - 80) / (20 + 80) = -60 / 100 = -0.6 accurately."""
        score = calculate_order_imbalance(bid_volume=20.0, ask_volume=80.0)
        assert score == pytest.approx(-0.6)

    def test_zero_volumes_safely_returns_zero(self):
        """When both bid and ask volumes are 0.0, return 0.0 without ZeroDivisionError."""
        score = calculate_order_imbalance(bid_volume=0.0, ask_volume=0.0)
        assert score == 0.0

    def test_integer_zero_volumes_safely_returns_zero(self):
        """Integer zeroes also safely return 0.0."""
        score = calculate_order_imbalance(bid_volume=0, ask_volume=0)
        assert score == 0.0

    @pytest.mark.parametrize(
        "bid_vol, ask_vol, expected_score",
        [
            (10.0, 90.0, -0.8),
            (90.0, 10.0, 0.8),
            (1.0, 3.0, -0.5),
            (3.0, 1.0, 0.5),
            (0.001, 0.002, -1.0 / 3.0),
            (500_000.0, 500_000.0, 0.0),
            (1_000_000.0, 0.0, 1.0),
            (0.0, 2_500_000.0, -1.0),
        ],
    )
    def test_normalized_score_range_and_values(
        self, bid_vol: float, ask_vol: float, expected_score: float
    ):
        """Normalized imbalance scores stay strictly within [-1.0, 1.0] and match formula."""
        score = calculate_order_imbalance(bid_volume=bid_vol, ask_volume=ask_vol)
        assert -1.0 <= score <= 1.0
        assert score == pytest.approx(expected_score, rel=1e-7)

    def test_negative_bid_volume_raises_value_error(self):
        """Negative bid volume is non-physical in an order book and must raise ValueError."""
        with pytest.raises(ValueError):
            calculate_order_imbalance(bid_volume=-10.0, ask_volume=50.0)

    def test_negative_ask_volume_raises_value_error(self):
        """Negative ask volume is non-physical in an order book and must raise ValueError."""
        with pytest.raises(ValueError):
            calculate_order_imbalance(bid_volume=50.0, ask_volume=-10.0)

    def test_non_numeric_bid_volume_raises_type_error(self):
        """Non-numeric input types must raise TypeError."""
        with pytest.raises(TypeError):
            calculate_order_imbalance(bid_volume="50.0", ask_volume=50.0)  # type: ignore

    def test_non_numeric_ask_volume_raises_type_error(self):
        """Non-numeric input types must raise TypeError."""
        with pytest.raises(TypeError):
            calculate_order_imbalance(bid_volume=50.0, ask_volume=None)  # type: ignore

    def test_fractional_volumes_precision(self):
        """Floating point precision is preserved with tiny crypto/FX volumes."""
        bid_vol = 1e-8
        ask_vol = 3e-8
        # (1e-8 - 3e-8) / (4e-8) = -2e-8 / 4e-8 = -0.5
        score = calculate_order_imbalance(bid_volume=bid_vol, ask_volume=ask_vol)
        assert score == pytest.approx(-0.5)


# ============================================================================
# End-to-End Visual DOM Ladder & Imbalance Meter Integration Tests
# ============================================================================

class TestDomLadderAndImbalanceIntegration:
    """Integration scenarios combining DOM Ladder extraction and Imbalance Meter."""

    def test_imbalance_across_configured_ladder_depth(self):
        """Calculates imbalance correctly from a truncated visual DOM ladder."""
        order_book = {
            "bids": [
                (100.0, 20.0),
                (99.0, 30.0),
                (98.0, 50.0),   # Outside depth limit 2
                (97.0, 100.0),  # Outside depth limit 2
            ],
            "asks": [
                (101.0, 10.0),
                (102.0, 40.0),
                (103.0, 50.0),  # Outside depth limit 2
            ],
        }

        # 1. Generate DOM ladder limited to depth=2
        ladder = generate_dom_ladder(order_book, depth=2)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        # 2. Aggregate volumes across the configured depth levels
        bid_volume = sum(volume for _, volume in bids)
        ask_volume = sum(volume for _, volume in asks)

        # Bids: 20.0 + 30.0 = 50.0
        # Asks: 10.0 + 40.0 = 50.0
        assert bid_volume == pytest.approx(50.0)
        assert ask_volume == pytest.approx(50.0)

        # 3. Calculate imbalance across depth
        imbalance = calculate_order_imbalance(bid_volume, ask_volume)
        assert imbalance == pytest.approx(0.0)

    def test_imbalance_with_empty_ladder_returns_zero(self):
        """When an empty order book is passed through the ladder and meter, safely returns 0.0."""
        empty_book = {"bids": [], "asks": []}
        ladder = generate_dom_ladder(empty_book, depth=5)
        bids_raw, asks_raw = _extract_bids_asks(ladder)
        bids = _normalize_levels(bids_raw)
        asks = _normalize_levels(asks_raw)

        bid_volume = sum(volume for _, volume in bids)
        ask_volume = sum(volume for _, volume in asks)

        imbalance = calculate_order_imbalance(bid_volume, ask_volume)
        assert imbalance == 0.0

    def test_imbalance_shifts_as_depth_expands(self):
        """Demonstrates imbalance changes as DOM ladder depth expands to include deeper levels."""
        order_book = {
            "bids": [(100.0, 10.0), (99.0, 10.0), (98.0, 80.0)],
            "asks": [(101.0, 40.0), (102.0, 40.0), (103.0, 20.0)],
        }

        # At Depth 1:
        # Bids: (100.0, 10.0) -> sum=10.0
        # Asks: (101.0, 40.0) -> sum=40.0
        # Imbalance = (10 - 40) / 50 = -0.6
        ladder_d1 = generate_dom_ladder(order_book, depth=1)
        bids_d1, asks_d1 = _extract_bids_asks(ladder_d1)
        imb_d1 = calculate_order_imbalance(
            bid_volume=sum(v for _, v in _normalize_levels(bids_d1)),
            ask_volume=sum(v for _, v in _normalize_levels(asks_d1)),
        )
        assert imb_d1 == pytest.approx(-0.6)

        # At Depth 3:
        # Bids: 10.0 + 10.0 + 80.0 = 100.0
        # Asks: 40.0 + 40.0 + 20.0 = 100.0
        # Imbalance = (100 - 100) / 200 = 0.0
        ladder_d3 = generate_dom_ladder(order_book, depth=3)
        bids_d3, asks_d3 = _extract_bids_asks(ladder_d3)
        imb_d3 = calculate_order_imbalance(
            bid_volume=sum(v for _, v in _normalize_levels(bids_d3)),
            ask_volume=sum(v for _, v in _normalize_levels(asks_d3)),
        )
        assert imb_d3 == pytest.approx(0.0)