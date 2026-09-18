import math
import random
import pytest

from src.orderbook.rb_tree import Color, RBNode, RedBlackTree
from src.orderbook.l2_order_book import L2OrderBook


# ---------------------------------------------------------------------------
# Invariant Verification Helpers for Red-Black Tree
# ---------------------------------------------------------------------------


def verify_rb_tree_invariants(tree: RedBlackTree) -> None:
    """
    Validates standard Red-Black Tree structural and color invariants:
    1. Root is BLACK (or None).
    2. Red nodes cannot have Red children (no two consecutive red nodes).
    3. Every path from a node to descendant NIL leaves has the same black-height.
    4. Binary search tree order is preserved (strict left < node < right).
    5. Parent pointers are symmetric.
    6. Overall tree depth is strictly O(log N): height <= 2 * floor(log2(N + 1)) + 1.
    """
    root = tree.root
    if root is None:
        assert len(tree) == 0
        return

    # Invariant 1: Root is BLACK
    assert root.color == Color.BLACK, f"Tree root must be BLACK, got {root.color}"
    assert root.parent is None, "Root parent must be None"

    def _validate_node(node: RBNode) -> int:
        if node is None:
            return 1  # Black height contribution of NIL leaf

        # Invariant 4: BST order and Parent pointers
        if node.left is not None:
            assert node.left.key < node.key, (
                f"BST violation: left key {node.left.key} is not strictly less than parent key {node.key}"
            )
            assert node.left.parent is node, (
                f"Parent pointer corrupted for node {node.left.key}"
            )

        if node.right is not None:
            assert node.right.key > node.key, (
                f"BST violation: right key {node.right.key} is not strictly greater than parent key {node.key}"
            )
            assert node.right.parent is node, (
                f"Parent pointer corrupted for node {node.right.key}"
            )

        # Invariant 2: No consecutive RED nodes
        if node.color == Color.RED:
            if node.left is not None:
                assert node.left.color == Color.BLACK, (
                    f"Red violation: red node {node.key} has red left child {node.left.key}"
                )
            if node.right is not None:
                assert node.right.color == Color.BLACK, (
                    f"Red violation: red node {node.key} has red right child {node.right.key}"
                )

        # Invariant 3: Black-height balance
        left_bh = _validate_node(node.left)
        right_bh = _validate_node(node.right)
        assert left_bh == right_bh, (
            f"Black-height imbalance at node {node.key}: left={left_bh}, right={right_bh}"
        )

        return left_bh + (1 if node.color == Color.BLACK else 0)

    _validate_node(root)

    # Invariant 6: O(log N) depth bound
    n = len(tree)
    if n > 0:
        max_allowed_height = 2 * math.floor(math.log2(n + 1)) + 1
        assert tree.get_height() <= max_allowed_height, (
            f"Tree height {tree.get_height()} exceeds Red-Black bound {max_allowed_height} for N={n}"
        )


# ---------------------------------------------------------------------------
# Unit Tests: RedBlackTree (src/orderbook/rb_tree.py)
# ---------------------------------------------------------------------------


def test_rb_tree_initialization_empty():
    tree = RedBlackTree()
    assert len(tree) == 0
    assert tree.root is None
    assert tree.find_min() is None
    assert tree.find_max() is None
    assert tree.keys() == []
    assert tree.items() == []
    assert tree.get_height() == 0
    verify_rb_tree_invariants(tree)


def test_rb_tree_single_insert():
    tree = RedBlackTree()
    tree.insert(100.0, 10.0)

    assert len(tree) == 1
    assert tree.root is not None
    assert tree.root.key == 100.0
    assert tree.root.value == 10.0
    assert tree.root.color == Color.BLACK
    assert tree.find_min() == (100.0, 10.0)
    assert tree.find_max() == (100.0, 10.0)
    assert tree.get(100.0) == 10.0
    verify_rb_tree_invariants(tree)


def test_rb_tree_duplicate_key_updates_value_in_place():
    tree = RedBlackTree()
    tree.insert(100.0, 10.0)
    tree.insert(100.0, 25.0)

    assert len(tree) == 1
    assert tree.get(100.0) == 25.0
    verify_rb_tree_invariants(tree)


def test_rb_tree_insert_ascending_triggers_left_rotations():
    tree = RedBlackTree()
    for key in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        tree.insert(key, key * 2)
        verify_rb_tree_invariants(tree)

    assert len(tree) == 6
    assert tree.keys() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert tree.find_min() == (10.0, 20.0)
    assert tree.find_max() == (60.0, 120.0)


def test_rb_tree_insert_descending_triggers_right_rotations():
    tree = RedBlackTree()
    for key in [60.0, 50.0, 40.0, 30.0, 20.0, 10.0]:
        tree.insert(key, key * 2)
        verify_rb_tree_invariants(tree)

    assert len(tree) == 6
    assert tree.keys() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert tree.find_min() == (10.0, 20.0)
    assert tree.find_max() == (60.0, 120.0)


def test_rb_tree_search_and_membership():
    tree = RedBlackTree()
    for k, v in [(15.0, 1.0), (10.0, 2.0), (20.0, 3.0), (5.0, 4.0), (12.0, 5.0)]:
        tree.insert(k, v)

    assert 10.0 in tree
    assert 20.0 in tree
    assert 99.0 not in tree

    assert tree.get(12.0) == 5.0
    assert tree.get(99.0, default=None) is None

    with pytest.raises(KeyError):
        tree.get(99.0)


def test_rb_tree_inorder_keys_and_items():
    tree = RedBlackTree()
    keys = [50.0, 20.0, 70.0, 10.0, 30.0, 60.0, 80.0]
    for k in keys:
        tree.insert(k, k * 10)

    assert tree.keys(reverse=False) == [10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0]
    assert tree.keys(reverse=True) == [80.0, 70.0, 60.0, 50.0, 30.0, 20.0, 10.0]

    expected_items = [(k, k * 10) for k in [10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0]]
    assert tree.items(reverse=False) == expected_items


def test_rb_tree_delete_leaf_nodes():
    tree = RedBlackTree()
    for k in [20.0, 10.0, 30.0]:
        tree.insert(k, k)

    # Delete red leaf
    tree.delete(10.0)
    assert len(tree) == 2
    assert 10.0 not in tree
    verify_rb_tree_invariants(tree)

    # Delete remaining leaf
    tree.delete(30.0)
    assert len(tree) == 1
    assert 30.0 not in tree
    verify_rb_tree_invariants(tree)


def test_rb_tree_delete_node_with_one_child():
    tree = RedBlackTree()
    for k in [20.0, 10.0, 30.0, 25.0]:
        tree.insert(k, k)

    tree.delete(30.0)  # Has single left child 25.0
    assert len(tree) == 3
    assert 30.0 not in tree
    assert tree.get(25.0) == 25.0
    verify_rb_tree_invariants(tree)


def test_rb_tree_delete_node_with_two_children():
    tree = RedBlackTree()
    for k in [20.0, 10.0, 30.0, 25.0, 40.0]:
        tree.insert(k, k)

    tree.delete(30.0)  # Two children: 25.0 and 40.0
    assert len(tree) == 4
    assert 30.0 not in tree
    assert tree.keys() == [10.0, 20.0, 25.0, 40.0]
    verify_rb_tree_invariants(tree)


def test_rb_tree_delete_root_until_empty():
    tree = RedBlackTree()
    keys = [40.0, 20.0, 60.0, 10.0, 30.0, 50.0, 70.0]
    for k in keys:
        tree.insert(k, k)

    for k in keys:
        tree.delete(k)
        verify_rb_tree_invariants(tree)

    assert len(tree) == 0
    assert tree.root is None
    assert tree.find_min() is None
    assert tree.find_max() is None


def test_rb_tree_delete_non_existent_key_raises():
    tree = RedBlackTree()
    tree.insert(10.0, 10.0)

    with pytest.raises(KeyError):
        tree.delete(99.0)


def test_rb_tree_logarithmic_depth_guarantee():
    rng = random.Random(42)
    tree = RedBlackTree()
    n = 250

    elements = [float(x) for x in rng.sample(range(1, 10000), n)]
    for elem in elements:
        tree.insert(elem, elem)
        verify_rb_tree_invariants(tree)

    max_bound = 2 * math.floor(math.log2(n + 1)) + 1
    assert tree.get_height() <= max_bound

    # Randomly delete half the elements
    to_delete = rng.sample(elements, n // 2)
    for elem in to_delete:
        tree.delete(elem)
        verify_rb_tree_invariants(tree)

    remaining = n - len(to_delete)
    new_bound = 2 * math.floor(math.log2(remaining + 1)) + 1
    assert tree.get_height() <= new_bound


# ---------------------------------------------------------------------------
# Unit Tests: L2OrderBook (src/orderbook/l2_order_book.py)
# ---------------------------------------------------------------------------


def test_l2_order_book_initial_state():
    book = L2OrderBook()
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.get_bids() == []
    assert book.get_asks() == []
    assert len(book.bids_tree) == 0
    assert len(book.asks_tree) == 0


def test_l2_order_book_add_bid_and_ask_levels():
    book = L2OrderBook()
    book.update_bid(price=100.0, quantity=5.0)
    book.update_ask(price=101.5, quantity=10.0)

    assert book.best_bid == (100.0, 5.0)
    assert book.best_ask == (101.5, 10.0)
    assert book.get_bids() == [(100.0, 5.0)]
    assert book.get_asks() == [(101.5, 10.0)]
    verify_rb_tree_invariants(book.bids_tree)
    verify_rb_tree_invariants(book.asks_tree)


def test_l2_order_book_bids_sorted_descending():
    book = L2OrderBook()
    # Insert bids in non-sequential order
    book.update_bid(price=99.0, quantity=2.0)
    book.update_bid(price=102.5, quantity=1.5)
    book.update_bid(price=101.0, quantity=3.0)
    book.update_bid(price=98.5, quantity=4.0)

    # Acceptance Criteria: Bids must be sorted descending
    expected_bids = [
        (102.5, 1.5),
        (101.0, 3.0),
        (99.0, 2.0),
        (98.5, 4.0),
    ]
    assert book.get_bids() == expected_bids
    assert book.best_bid == (102.5, 1.5)
    verify_rb_tree_invariants(book.bids_tree)


def test_l2_order_book_asks_sorted_ascending():
    book = L2OrderBook()
    # Insert asks in non-sequential order
    book.update_ask(price=105.0, quantity=10.0)
    book.update_ask(price=103.0, quantity=5.0)
    book.update_ask(price=106.5, quantity=2.0)
    book.update_ask(price=104.0, quantity=7.5)

    # Acceptance Criteria: Asks must be sorted ascending
    expected_asks = [
        (103.0, 5.0),
        (104.0, 7.5),
        (105.0, 10.0),
        (106.5, 2.0),
    ]
    assert book.get_asks() == expected_asks
    assert book.best_ask == (103.0, 5.0)
    verify_rb_tree_invariants(book.asks_tree)


def test_l2_order_book_update_quantity_existing_level():
    book = L2OrderBook()
    book.update_bid(price=100.0, quantity=10.0)
    book.update_bid(price=100.0, quantity=25.0)

    assert book.get_bid_quantity(100.0) == 25.0
    assert book.best_bid == (100.0, 25.0)
    assert len(book.bids_tree) == 1

    book.update_ask(price=101.0, quantity=15.0)
    book.update_ask(price=101.0, quantity=30.0)

    assert book.get_ask_quantity(101.0) == 30.0
    assert book.best_ask == (101.0, 30.0)
    assert len(book.asks_tree) == 1
    verify_rb_tree_invariants(book.bids_tree)
    verify_rb_tree_invariants(book.asks_tree)


def test_l2_order_book_delete_on_zero_quantity():
    book = L2OrderBook()
    book.update_bid(price=100.0, quantity=10.0)
    book.update_bid(price=99.0, quantity=5.0)

    # Delete bid level with quantity 0
    book.update_bid(price=100.0, quantity=0.0)
    assert book.best_bid == (99.0, 5.0)
    assert book.get_bids() == [(99.0, 5.0)]
    assert len(book.bids_tree) == 1

    book.update_ask(price=101.0, quantity=2.0)
    book.update_ask(price=102.0, quantity=4.0)

    # Delete ask level with quantity 0
    book.update_ask(price=101.0, quantity=0.0)
    assert book.best_ask == (102.0, 4.0)
    assert book.get_asks() == [(102.0, 4.0)]
    assert len(book.asks_tree) == 1
    verify_rb_tree_invariants(book.bids_tree)
    verify_rb_tree_invariants(book.asks_tree)


def test_l2_order_book_delete_non_existent_price_level_is_noop():
    book = L2OrderBook()
    book.update_bid(price=100.0, quantity=5.0)
    # Deleting a price level that does not exist with quantity 0 should succeed silently
    book.update_bid(price=95.0, quantity=0.0)
    book.update_ask(price=105.0, quantity=0.0)

    assert book.best_bid == (100.0, 5.0)
    assert len(book.bids_tree) == 1
    assert len(book.asks_tree) == 0


def test_l2_order_book_top_of_book_transitions():
    book = L2OrderBook()
    # Add multiple levels
    book.update_bid(price=100.0, quantity=10.0)
    book.update_bid(price=105.0, quantity=5.0)
    book.update_bid(price=102.0, quantity=8.0)

    # Top bid should be 105.0
    assert book.best_bid == (105.0, 5.0)

    # Delete top bid -> 102.0 becomes top
    book.update_bid(price=105.0, quantity=0.0)
    assert book.best_bid == (102.0, 8.0)

    # Delete 102.0 -> 100.0 becomes top
    book.update_bid(price=102.0, quantity=0.0)
    assert book.best_bid == (100.0, 10.0)

    # Delete last bid -> book empty
    book.update_bid(price=100.0, quantity=0.0)
    assert book.best_bid is None
    assert book.get_bids() == []


def test_l2_order_book_get_depth_limits():
    book = L2OrderBook()
    for i in range(1, 11):
        book.update_bid(price=float(100 - i), quantity=float(i))
        book.update_ask(price=float(100 + i), quantity=float(i))

    bids_depth_3 = book.get_bids(depth=3)
    assert len(bids_depth_3) == 3
    assert bids_depth_3 == [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)]

    asks_depth_2 = book.get_asks(depth=2)
    assert len(asks_depth_2) == 2
    assert asks_depth_2 == [(101.0, 1.0), (102.0, 2.0)]


def test_l2_order_book_negative_price_raises_error():
    book = L2OrderBook()
    with pytest.raises(ValueError):
        book.update_bid(price=-100.0, quantity=10.0)

    with pytest.raises(ValueError):
        book.update_bid(price=0.0, quantity=10.0)

    with pytest.raises(ValueError):
        book.update_ask(price=-50.0, quantity=5.0)

    with pytest.raises(ValueError):
        book.update_ask(price=0.0, quantity=5.0)


def test_l2_order_book_negative_quantity_raises_error():
    book = L2OrderBook()
    with pytest.raises(ValueError):
        book.update_bid(price=100.0, quantity=-1.0)

    with pytest.raises(ValueError):
        book.update_ask(price=105.0, quantity=-0.01)


def test_l2_order_book_clear():
    book = L2OrderBook()
    book.update_bid(100.0, 10.0)
    book.update_ask(105.0, 20.0)
    book.clear()

    assert book.best_bid is None
    assert book.best_ask is None
    assert len(book.bids_tree) == 0
    assert len(book.asks_tree) == 0


def test_l2_order_book_stress_and_logarithmic_depth_compliance():
    """
    Applies 500 mixed add, update, and delete updates to both sides.
    Verifies that:
    1. Bids are strictly descending.
    2. Asks are strictly ascending.
    3. Underlying Red-Black trees maintain O(log N) depth.
    4. Best bid and best ask always reflect maximum bid and minimum ask.
    """
    rng = random.Random(1337)
    book = L2OrderBook()

    reference_bids = {}
    reference_asks = {}

    for _ in range(500):
        side = rng.choice(["bid", "ask"])
        price = round(rng.uniform(10.0, 500.0), 2)
        # 25% chance to set quantity to 0 (delete), else update/add
        quantity = 0.0 if rng.random() < 0.25 else round(rng.uniform(0.1, 100.0), 4)

        if side == "bid":
            book.update_bid(price, quantity)
            if quantity == 0.0:
                reference_bids.pop(price, None)
            else:
                reference_bids[price] = quantity
        else:
            book.update_ask(price, quantity)
            if quantity == 0.0:
                reference_asks.pop(price, None)
            else:
                reference_asks[price] = quantity

    # Verify Red-Black invariants on both trees
    verify_rb_tree_invariants(book.bids_tree)
    verify_rb_tree_invariants(book.asks_tree)

    # Verify Bids sorted descending
    sorted_ref_bids = sorted(reference_bids.items(), key=lambda x: x[0], reverse=True)
    actual_bids = book.get_bids()
    assert actual_bids == sorted_ref_bids

    # Verify Asks sorted ascending
    sorted_ref_asks = sorted(reference_asks.items(), key=lambda x: x[0])
    actual_asks = book.get_asks()
    assert actual_asks == sorted_ref_asks

    # Verify Top of Book
    expected_best_bid = sorted_ref_bids[0] if sorted_ref_bids else None
    expected_best_ask = sorted_ref_asks[0] if sorted_ref_asks else None
    assert book.best_bid == expected_best_bid
    assert book.best_ask == expected_best_ask

    # Verify O(log N) depth guarantee
    if len(book.bids_tree) > 0:
        max_bid_height = 2 * math.floor(math.log2(len(book.bids_tree) + 1)) + 1
        assert book.bids_tree.get_height() <= max_bid_height

    if len(book.asks_tree) > 0:
        max_ask_height = 2 * math.floor(math.log2(len(book.asks_tree) + 1)) + 1
        assert book.asks_tree.get_height() <= max_ask_height