"""
In-memory Red-Black Tree implementation providing O(log N) insertion,
lookup, and deletion while strictly maintaining balanced binary search
tree invariants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Tuple

_MISSING = object()


class Color(Enum):
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """A single node within the Red-Black Tree."""

    __slots__ = ("key", "value", "color", "parent", "left", "right")

    def __init__(
        self,
        key: Any,
        value: Any,
        color: Color = Color.RED,
        parent: Optional[RBNode] = None,
        left: Optional[RBNode] = None,
        right: Optional[RBNode] = None,
    ) -> None:
        self.key = key
        self.value = value
        self.color = color
        self.parent = parent
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"RBNode({self.key}: {self.value}, color={self.color.value})"


class RedBlackTree:
    """
    Self-balancing binary search tree conforming to Red-Black Tree invariants:
    1. Every node is either RED or BLACK.
    2. The root node is BLACK.
    3. No two consecutive RED nodes exist on any simple path.
    4. Every path from a node to descendant NIL leaves has identical black-height.
    5. NIL leaves are represented by None.
    """

    def __init__(self) -> None:
        self.root: Optional[RBNode] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self._find_node(key) is not None

    def _find_node(self, key: Any) -> Optional[RBNode]:
        curr = self.root
        while curr is not None:
            if key < curr.key:
                curr = curr.left
            elif key > curr.key:
                curr = curr.right
            else:
                return curr
        return None

    def get(self, key: Any, default: Any = _MISSING) -> Any:
        """Retrieve value associated with key or return default if missing."""
        node = self._find_node(key)
        if node is not None:
            return node.value
        if default is not _MISSING:
            return default
        raise KeyError(f"Key {key} not found")

    def find_min(self) -> Optional[Tuple[Any, Any]]:
        """Return the (key, value) pair of the minimum element, or None if empty."""
        if self.root is None:
            return None
        node = self._minimum(self.root)
        return (node.key, node.value)

    def find_max(self) -> Optional[Tuple[Any, Any]]:
        """Return the (key, value) pair of the maximum element, or None if empty."""
        if self.root is None:
            return None
        node = self._maximum(self.root)
        return (node.key, node.value)

    def _minimum(self, node: RBNode) -> RBNode:
        curr = node
        while curr.left is not None:
            curr = curr.left
        return curr

    def _maximum(self, node: RBNode) -> RBNode:
        curr = node
        while curr.right is not None:
            curr = curr.right
        return curr

    def get_height(self) -> int:
        """Return the 1-based maximum path height, or 0 if tree is empty."""
        def _height(node: Optional[RBNode]) -> int:
            if node is None:
                return 0
            left_h = _height(node.left)
            right_h = _height(node.right)
            return 1 + (left_h if left_h > right_h else right_h)

        return _height(self.root)

    def clear(self) -> None:
        """Reset the tree to an empty state."""
        self.root = None
        self._size = 0

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key-value pair or update value in-place if key already exists."""
        if self.root is None:
            self.root = RBNode(key, value, color=Color.BLACK, parent=None)
            self._size = 1
            return

        curr = self.root
        parent = None
        while curr is not None:
            parent = curr
            if key < curr.key:
                curr = curr.left
            elif key > curr.key:
                curr = curr.right
            else:
                curr.value = value
                return

        new_node = RBNode(key, value, color=Color.RED, parent=parent)
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        self._size += 1
        self._insert_fixup(new_node)

    def _insert_fixup(self, node: RBNode) -> None:
        while node.parent is not None and node.parent.color == Color.RED:
            parent = node.parent
            grandparent = parent.parent
            if grandparent is None:
                break

            if parent is grandparent.left:
                uncle = grandparent.right
                if uncle is not None and uncle.color == Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node is parent.right:
                        node = parent
                        self._rotate_left(node)
                        parent = node.parent
                        grandparent = parent.parent if parent else None
                        if grandparent is None:
                            break

                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_right(grandparent)
            else:
                uncle = grandparent.left
                if uncle is not None and uncle.color == Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node is parent.left:
                        node = parent
                        self._rotate_right(node)
                        parent = node.parent
                        grandparent = parent.parent if parent else None
                        if grandparent is None:
                            break

                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_left(grandparent)

        if self.root is not None:
            self.root.color = Color.BLACK

    def delete(self, key: Any) -> None:
        """Delete key-value pair or raise KeyError if missing."""
        z = self._find_node(key)
        if z is None:
            raise KeyError(f"Key {key} not found")

        if z.left is None or z.right is None:
            y = z
        else:
            y = self._minimum(z.right)

        if y.left is not None:
            x = y.left
        else:
            x = y.right

        x_parent = y.parent
        if x is not None:
            x.parent = x_parent

        if y.parent is None:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        y_color = y.color

        if y is not z:
            z.key = y.key
            z.value = y.value

        self._size -= 1

        if y_color == Color.BLACK:
            self._delete_fixup(x, x_parent)

    def _delete_fixup(self, x: Optional[RBNode], x_parent: Optional[RBNode]) -> None:
        while x is not self.root and (x is None or x.color == Color.BLACK):
            if x_parent is None:
                break

            if x is x_parent.left:
                w = x_parent.right
                if w is not None and w.color == Color.RED:
                    w.color = Color.BLACK
                    x_parent.color = Color.RED
                    self._rotate_left(x_parent)
                    w = x_parent.right

                if w is None:
                    x = x_parent
                    x_parent = x.parent
                    continue

                left_black = (w.left is None or w.left.color == Color.BLACK)
                right_black = (w.right is None or w.right.color == Color.BLACK)

                if left_black and right_black:
                    w.color = Color.RED
                    x = x_parent
                    x_parent = x.parent
                else:
                    if right_black:
                        if w.left is not None:
                            w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._rotate_right(w)
                        w = x_parent.right

                    w.color = x_parent.color
                    x_parent.color = Color.BLACK
                    if w.right is not None:
                        w.right.color = Color.BLACK
                    self._rotate_left(x_parent)
                    x = self.root
                    break
            else:
                w = x_parent.left
                if w is not None and w.color == Color.RED:
                    w.color = Color.BLACK
                    x_parent.color = Color.RED
                    self._rotate_right(x_parent)
                    w = x_parent.left

                if w is None:
                    x = x_parent
                    x_parent = x.parent
                    continue

                left_black = (w.left is None or w.left.color == Color.BLACK)
                right_black = (w.right is None or w.right.color == Color.BLACK)

                if left_black and right_black:
                    w.color = Color.RED
                    x = x_parent
                    x_parent = x.parent
                else:
                    if left_black:
                        if w.right is not None:
                            w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._rotate_left(w)
                        w = x_parent.left

                    w.color = x_parent.color
                    x_parent.color = Color.BLACK
                    if w.left is not None:
                        w.left.color = Color.BLACK
                    self._rotate_right(x_parent)
                    x = self.root
                    break

        if x is not None:
            x.color = Color.BLACK
        if self.root is not None:
            self.root.color = Color.BLACK

    def _rotate_left(self, x: RBNode) -> None:
        y = x.right
        if y is None:
            return

        x.right = y.left
        if y.left is not None:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def _rotate_right(self, y: RBNode) -> None:
        x = y.left
        if x is None:
            return

        y.left = x.right
        if x.right is not None:
            x.right.parent = y

        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        x.right = y
        y.parent = x

    def keys(self, reverse: bool = False) -> List[Any]:
        """Return in-order (or reverse in-order) list of keys."""
        result: List[Any] = []
        if reverse:
            def _traverse(node: Optional[RBNode]) -> None:
                if node is None:
                    return
                _traverse(node.right)
                result.append(node.key)
                _traverse(node.left)
        else:
            def _traverse(node: Optional[RBNode]) -> None:
                if node is None:
                    return
                _traverse(node.left)
                result.append(node.key)
                _traverse(node.right)

        _traverse(self.root)
        return result

    def items(self, reverse: bool = False) -> List[Tuple[Any, Any]]:
        """Return in-order (or reverse in-order) list of (key, value) pairs."""
        result: List[Tuple[Any, Any]] = []
        if reverse:
            def _traverse(node: Optional[RBNode]) -> None:
                if node is None:
                    return
                _traverse(node.right)
                result.append((node.key, node.value))
                _traverse(node.left)
        else:
            def _traverse(node: Optional[RBNode]) -> None:
                if node is None:
                    return
                _traverse(node.left)
                result.append((node.key, node.value))
                _traverse(node.right)

        _traverse(self.root)
        return result