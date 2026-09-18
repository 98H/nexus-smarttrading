from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _validate_coordinate(name: str, value: float) -> float:
    """Validate that a coordinate or dimension value is a finite number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class BoundingBox:
    """2D axis-aligned bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        for name, val in [
            ("min_x", self.min_x),
            ("min_y", self.min_y),
            ("max_x", self.max_x),
            ("max_y", self.max_y),
        ]:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val!r}")

        mx = float(self.min_x)
        my = float(self.min_y)
        xx = float(self.max_x)
        xy = float(self.max_y)

        if mx > xx:
            raise ValueError(f"min_x ({mx}) cannot be greater than max_x ({xx})")
        if my > xy:
            raise ValueError(f"min_y ({my}) cannot be greater than max_y ({xy})")

        object.__setattr__(self, "min_x", mx)
        object.__setattr__(self, "min_y", my)
        object.__setattr__(self, "max_x", xx)
        object.__setattr__(self, "max_y", xy)

    def intersects(self, other: BoundingBox) -> bool:
        """Check whether this bounding box intersects another (inclusive of boundaries)."""
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
        )

    def contains_point(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        """Check whether point (x, y) falls inside the bounding box expanded by tolerance."""
        return (
            self.min_x - tolerance <= x <= self.max_x + tolerance
            and self.min_y - tolerance <= y <= self.max_y + tolerance
        )

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the minimal bounding box enclosing both self and other."""
        return BoundingBox(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y),
        )

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def enlargement_area(self, other: BoundingBox) -> float:
        """Calculate the increase in area required to enclose other."""
        new_min_x = min(self.min_x, other.min_x)
        new_min_y = min(self.min_y, other.min_y)
        new_max_x = max(self.max_x, other.max_x)
        new_max_y = max(self.max_y, other.max_y)
        new_area = (new_max_x - new_min_x) * (new_max_y - new_min_y)
        return max(0.0, new_area - self.area)


@dataclass(frozen=True)
class HitResult:
    """Hit-test or query result representation."""

    item_id: str
    bounds: BoundingBox
    data: Any = None


@dataclass
class _IndexEntry:
    """Internal item entry stored within R-Tree leaf nodes."""

    item_id: str
    bounds: BoundingBox
    data: Any = None
    is_point: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0

    def to_hit_result(self) -> HitResult:
        return HitResult(item_id=self.item_id, bounds=self.bounds, data=self.data)

    def hit_test(self, x: float, y: float, tolerance: float) -> bool:
        if self.is_point:
            distance = math.hypot(self.center_x - x, self.center_y - y)
            return distance <= (self.radius + tolerance)
        return self.bounds.contains_point(x, y, tolerance)


def _compute_bounds(bboxes: Iterable[BoundingBox]) -> Optional[BoundingBox]:
    """Compute the minimal bounding box enclosing an iterable of BoundingBoxes."""
    it = iter(bboxes)
    try:
        first = next(it)
    except StopIteration:
        return None

    min_x = first.min_x
    min_y = first.min_y
    max_x = first.max_x
    max_y = first.max_y

    for b in it:
        if b.min_x < min_x:
            min_x = b.min_x
        if b.min_y < min_y:
            min_y = b.min_y
        if b.max_x > max_x:
            max_x = b.max_x
        if b.max_y > max_y:
            max_y = b.max_y

    return BoundingBox(min_x, min_y, max_x, max_y)


class _RTreeNode:
    """Node in the R-Tree spatial index."""

    def __init__(self, is_leaf: bool, parent: Optional[_RTreeNode] = None) -> None:
        self.is_leaf = is_leaf
        self.parent = parent
        self.items: List[_IndexEntry] = []
        self.children: List[_RTreeNode] = []
        self.bounds: Optional[BoundingBox] = None

    def update_bounds(self) -> None:
        """Recompute bounding box from items or children."""
        if self.is_leaf:
            self.bounds = _compute_bounds(item.bounds for item in self.items)
        else:
            self.bounds = _compute_bounds(
                child.bounds for child in self.children if child.bounds is not None
            )


def _split_elements(elements: List[Any], min_entries: int) -> Tuple[List[Any], List[Any]]:
    """Guttman's Quadratic Split algorithm."""
    n = len(elements)
    max_waste = -float("inf")
    seed1_idx, seed2_idx = 0, 1

    for i in range(n):
        b1 = elements[i].bounds
        for j in range(i + 1, n):
            b2 = elements[j].bounds
            u = b1.union(b2)
            waste = u.area - b1.area - b2.area
            if waste > max_waste:
                max_waste = waste
                seed1_idx, seed2_idx = i, j

    group1 = [elements[seed1_idx]]
    group2 = [elements[seed2_idx]]
    b1 = elements[seed1_idx].bounds
    b2 = elements[seed2_idx].bounds

    remaining = [elements[k] for k in range(n) if k not in (seed1_idx, seed2_idx)]

    while remaining:
        if len(group1) + len(remaining) == min_entries:
            group1.extend(remaining)
            break
        if len(group2) + len(remaining) == min_entries:
            group2.extend(remaining)
            break

        best_diff = -1.0
        best_idx = 0
        best_enl1 = 0.0
        best_enl2 = 0.0

        for idx, item in enumerate(remaining):
            ib = item.bounds
            enl1 = b1.enlargement_area(ib)
            enl2 = b2.enlargement_area(ib)
            diff = abs(enl1 - enl2)
            if diff > best_diff:
                best_diff = diff
                best_idx = idx
                best_enl1 = enl1
                best_enl2 = enl2

        chosen = remaining.pop(best_idx)
        cb = chosen.bounds

        if best_enl1 < best_enl2:
            group1.append(chosen)
            b1 = b1.union(cb)
        elif best_enl2 < best_enl1:
            group2.append(chosen)
            b2 = b2.union(cb)
        else:
            if b1.area < b2.area:
                group1.append(chosen)
                b1 = b1.union(cb)
            elif b2.area < b1.area:
                group2.append(chosen)
                b2 = b2.union(cb)
            else:
                if len(group1) <= len(group2):
                    group1.append(chosen)
                    b1 = b1.union(cb)
                else:
                    group2.append(chosen)
                    b2 = b2.union(cb)

    return group1, group2


class SpatialIndex:
    """R-Tree based 2D spatial index for fast point and bar hit-testing."""

    def __init__(self, max_entries: int = 8, min_entries: int = 2) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 2:
            raise ValueError(f"max_entries must be an integer >= 2, got {max_entries!r}")
        if not isinstance(min_entries, int) or isinstance(min_entries, bool) or min_entries < 1:
            raise ValueError(f"min_entries must be an integer >= 1, got {min_entries!r}")
        if min_entries > (max_entries + 1) // 2:
            raise ValueError(
                f"min_entries ({min_entries}) cannot exceed (max_entries + 1) // 2 ({(max_entries + 1) // 2})"
            )

        self._max_entries = max_entries
        self._min_entries = min_entries
        self._root = _RTreeNode(is_leaf=True)
        self._items: Dict[str, Tuple[_IndexEntry, _RTreeNode]] = {}

    def __len__(self) -> int:
        return len(self._items)

    @property
    def bounds(self) -> Optional[BoundingBox]:
        """Return the minimal bounding box enclosing all elements in the index."""
        if not self._items:
            return None
        return self._root.bounds

    def insert(self, item_id: str, bounds: BoundingBox, data: Any = None) -> None:
        """Insert a generic item using a BoundingBox instance."""
        if not isinstance(bounds, BoundingBox):
            raise TypeError(
                f"bounds must be an instance of BoundingBox, got {type(bounds).__name__}"
            )
        self._insert_entry(
            _IndexEntry(
                item_id=str(item_id),
                bounds=bounds,
                data=data,
                is_point=False,
            )
        )

    def insert_bar(
        self,
        item_id: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        data: Any = None,
    ) -> None:
        """Insert a bar bounding box into the spatial index."""
        bbox = BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
        self._insert_entry(
            _IndexEntry(
                item_id=str(item_id),
                bounds=bbox,
                data=data,
                is_point=False,
            )
        )

    def insert_point(
        self,
        item_id: str,
        x: float,
        y: float,
        radius: float = 0.0,
        data: Any = None,
    ) -> None:
        """Insert a scatter point into the spatial index."""
        qx = _validate_coordinate("x", x)
        qy = _validate_coordinate("y", y)
        r = _validate_coordinate("radius", radius)
        if r < 0.0:
            raise ValueError(f"radius must be non-negative, got {radius!r}")

        bbox = BoundingBox(min_x=qx - r, min_y=qy - r, max_x=qx + r, max_y=qy + r)
        self._insert_entry(
            _IndexEntry(
                item_id=str(item_id),
                bounds=bbox,
                data=data,
                is_point=True,
                center_x=qx,
                center_y=qy,
                radius=r,
            )
        )

    def remove(self, item_id: str) -> bool:
        """Remove an item by identifier. Prunes empty subtrees and maintains bounding boxes."""
        key = str(item_id)
        if key not in self._items:
            return False

        entry, leaf = self._items.pop(key)
        leaf.items.remove(entry)

        # Bottom-up update: update node bounds first, then prune if empty
        current: Optional[_RTreeNode] = leaf
        while current is not None:
            current.update_bounds()
            parent = current.parent
            is_empty = (len(current.items) == 0) if current.is_leaf else (len(current.children) == 0)
            if is_empty and current != self._root:
                if parent is not None and current in parent.children:
                    parent.children.remove(current)
                current.parent = None
            current = parent

        # Collapse trivial intermediate single-child roots
        while not self._root.is_leaf:
            if len(self._root.children) == 0:
                self._root = _RTreeNode(is_leaf=True)
                break
            if len(self._root.children) == 1:
                self._root = self._root.children[0]
                self._root.parent = None
            else:
                break

        self._root.update_bounds()
        return True

    def clear(self) -> None:
        """Reset the spatial index to an empty state."""
        self._root = _RTreeNode(is_leaf=True)
        self._items.clear()

    def hit_test(self, x: float, y: float, tolerance: float = 0.0) -> List[HitResult]:
        """Hit test for elements containing or within tolerance distance of (x, y)."""
        qx = _validate_coordinate("x", x)
        qy = _validate_coordinate("y", y)
        tol = _validate_coordinate("tolerance", tolerance)
        if tol < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")

        if not self._items or self._root.bounds is None:
            return []

        query_box = BoundingBox(
            min_x=qx - tol,
            min_y=qy - tol,
            max_x=qx + tol,
            max_y=qy + tol,
        )

        if not self._root.bounds.intersects(query_box):
            return []

        results: List[HitResult] = []
        stack = [self._root]

        while stack:
            node = stack.pop()
            if node.is_leaf:
                for item in node.items:
                    if item.bounds.intersects(query_box) and item.hit_test(qx, qy, tol):
                        results.append(item.to_hit_result())
            else:
                for child in node.children:
                    if child.bounds is not None and child.bounds.intersects(query_box):
                        stack.append(child)

        return results

    def query_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> List[HitResult]:
        """Query for all items whose bounding box intersects the query rectangle."""
        query_box = BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

        if not self._items or self._root.bounds is None:
            return []

        if not self._root.bounds.intersects(query_box):
            return []

        results: List[HitResult] = []
        stack = [self._root]

        while stack:
            node = stack.pop()
            if node.is_leaf:
                for item in node.items:
                    if item.bounds.intersects(query_box):
                        results.append(item.to_hit_result())
            else:
                for child in node.children:
                    if child.bounds is not None and child.bounds.intersects(query_box):
                        stack.append(child)

        return results

    def _insert_entry(self, entry: _IndexEntry) -> None:
        """Insert an index entry, updating existing position if duplicate ID."""
        if entry.item_id in self._items:
            self.remove(entry.item_id)

        leaf = self._choose_leaf(self._root, entry.bounds)
        leaf.items.append(entry)
        self._items[entry.item_id] = (entry, leaf)

        new_node = None
        if len(leaf.items) > self._max_entries:
            new_node = self._split_node(leaf)
            for item in leaf.items:
                self._items[item.item_id] = (item, leaf)
            for item in new_node.items:
                self._items[item.item_id] = (item, new_node)

        self._adjust_tree(leaf, new_node)

    def _choose_leaf(self, node: _RTreeNode, entry_bounds: BoundingBox) -> _RTreeNode:
        """Find the leaf node best suited for inserting the entry."""
        if node.is_leaf:
            return node

        if not node.children:
            node.is_leaf = True
            return node

        best_child: Optional[_RTreeNode] = None
        min_enlargement = float("inf")
        min_area = float("inf")

        for child in node.children:
            if child.bounds is None:
                return self._choose_leaf(child, entry_bounds)
            enl = child.bounds.enlargement_area(entry_bounds)
            area = child.bounds.area
            if enl < min_enlargement:
                min_enlargement = enl
                min_area = area
                best_child = child
            elif enl == min_enlargement:
                if area < min_area:
                    min_area = area
                    best_child = child

        if best_child is None:
            best_child = node.children[0]

        return self._choose_leaf(best_child, entry_bounds)

    def _split_node(self, node: _RTreeNode) -> _RTreeNode:
        """Split a node and return the newly created sibling node."""
        new_node = _RTreeNode(is_leaf=node.is_leaf, parent=node.parent)
        if node.is_leaf:
            group1, group2 = _split_elements(node.items, self._min_entries)
            node.items = group1
            new_node.items = group2
        else:
            group1, group2 = _split_elements(node.children, self._min_entries)
            node.children = group1
            new_node.children = group2
            for child in new_node.children:
                child.parent = new_node

        node.update_bounds()
        new_node.update_bounds()
        return new_node

    def _adjust_tree(self, node: _RTreeNode, new_node: Optional[_RTreeNode] = None) -> None:
        """Ascend from leaf to root, updating bounding boxes and propagating splits."""
        node.update_bounds()
        if new_node is not None:
            new_node.update_bounds()

        if node == self._root:
            if new_node is not None:
                new_root = _RTreeNode(is_leaf=False)
                new_root.children = [node, new_node]
                node.parent = new_root
                new_node.parent = new_root
                new_root.update_bounds()
                self._root = new_root
            return

        parent = node.parent
        assert parent is not None

        parent_new_node = None
        if new_node is not None:
            new_node.parent = parent
            parent.children.append(new_node)
            if len(parent.children) > self._max_entries:
                parent_new_node = self._split_node(parent)

        self._adjust_tree(parent, parent_new_node)