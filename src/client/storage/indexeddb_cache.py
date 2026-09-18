"""Low-level IndexedDB-like cache supporting composite keys and range indexing."""

import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


class IndexedDBCache:
    """In-memory IndexedDB-compatible store supporting composite keys and indexed range scans."""

    def __init__(
        self,
        db_name: str,
        store_name: str,
        key_path: Union[str, Sequence[str]],
        indexes: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
    ) -> None:
        self.db_name = db_name
        self.store_name = store_name
        self.key_path = tuple(key_path) if isinstance(key_path, (tuple, list)) else key_path

        self.indexes: Dict[str, Tuple[str, ...]] = {}
        if indexes:
            for name, path in indexes.items():
                if isinstance(path, str):
                    self.indexes[name] = (path,)
                else:
                    self.indexes[name] = tuple(path)

        self._records: Dict[Any, Dict[str, Any]] = {}

    def extract_primary_key(self, record: Dict[str, Any]) -> Any:
        """Extract primary key value or composite key tuple from record."""
        if isinstance(self.key_path, str):
            if self.key_path not in record or record[self.key_path] is None:
                raise ValueError(f"Record missing key path field: '{self.key_path}'")
            return record[self.key_path]

        keys = []
        for field in self.key_path:
            if field not in record or record[field] is None:
                raise ValueError(f"Record missing key path field: '{field}'")
            keys.append(record[field])
        return tuple(keys)

    def _extract_primary_key(self, record: Dict[str, Any]) -> Any:
        """Internal helper maintained for backwards compatibility."""
        return self.extract_primary_key(record)

    def extract_index_key(self, record: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Any]:
        """Extract indexed key value or composite tuple from record.

        Returns None if any field in the index path is missing or None to avoid
        unhandled TypeError crashes on heterogeneous or partial composite keys.
        """
        for field in path:
            if field not in record or record[field] is None:
                return None

        if len(path) == 1:
            return record[path[0]]
        return tuple(record[f] for f in path)

    def _extract_index_key(self, record: Dict[str, Any], path: Tuple[str, ...]) -> Optional[Any]:
        """Internal helper maintained for backwards compatibility."""
        return self.extract_index_key(record, path)

    def find_index_for_fields(self, fields: Sequence[str]) -> Optional[str]:
        """Find an index name whose path exactly matches the given sequence of fields."""
        target = tuple(fields)
        for name, path in self.indexes.items():
            if path == target:
                return name
        return None

    def put(self, record: Dict[str, Any]) -> Any:
        """Store or upsert a record identified by its primary/composite key."""
        pk = self.extract_primary_key(record)
        self._records[pk] = dict(record)
        return pk

    def put_batch(self, records: Iterable[Dict[str, Any]]) -> List[Any]:
        """Store or upsert a batch of records."""
        keys = []
        for record in records:
            keys.append(self.put(record))
        return keys

    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        """Retrieve a record by its primary/composite key."""
        if isinstance(key, list):
            key = tuple(key)
        if key in self._records:
            return dict(self._records[key])
        if isinstance(self.key_path, (tuple, list)) and len(self.key_path) == 1 and not isinstance(key, tuple):
            if (key,) in self._records:
                return dict(self._records[(key,)])
        return None

    def get_range(
        self,
        index_name: Optional[str] = None,
        lower: Any = None,
        upper: Any = None,
        include_lower: bool = True,
        include_upper: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scan records within a key range, ordered by index key."""
        if limit is not None and limit <= 0:
            return []

        if index_name is not None:
            if index_name not in self.indexes:
                raise ValueError(f"Index '{index_name}' does not exist.")
            path = self.indexes[index_name]
        else:
            if isinstance(self.key_path, str):
                path = (self.key_path,)
            else:
                path = self.key_path

        if len(path) == 1:
            lower_val = lower[0] if isinstance(lower, (tuple, list)) and len(lower) == 1 else lower
            upper_val = upper[0] if isinstance(upper, (tuple, list)) and len(upper) == 1 else upper
        else:
            lower_val = tuple(lower) if isinstance(lower, list) else lower
            upper_val = tuple(upper) if isinstance(upper, list) else upper

        matched: List[Tuple[Any, Any, Dict[str, Any]]] = []
        for pk, record in self._records.items():
            idx_key = self.extract_index_key(record, path)
            if idx_key is None:
                continue

            try:
                if lower_val is not None:
                    if include_lower:
                        if idx_key < lower_val:
                            continue
                    else:
                        if idx_key <= lower_val:
                            continue

                if upper_val is not None:
                    if include_upper:
                        if idx_key > upper_val:
                            continue
                    else:
                        if idx_key >= upper_val:
                            continue
            except TypeError:
                continue

            matched.append((idx_key, pk, record))

        try:
            if limit is not None and limit < len(matched):
                matched = heapq.nsmallest(limit, matched, key=lambda item: (item[0], item[1]))
            else:
                matched.sort(key=lambda item: (item[0], item[1]))
        except TypeError:
            if limit is not None and limit < len(matched):
                matched = heapq.nsmallest(limit, matched, key=lambda item: (str(item[0]), str(item[1])))
            else:
                matched.sort(key=lambda item: (str(item[0]), str(item[1])))

        return [dict(item[2]) for item in matched]

    def delete(self, key: Any) -> bool:
        """Delete a record by its primary/composite key."""
        if isinstance(key, list):
            key = tuple(key)
        if key in self._records:
            del self._records[key]
            return True
        if isinstance(self.key_path, (tuple, list)) and len(self.key_path) == 1 and not isinstance(key, tuple):
            if (key,) in self._records:
                del self._records[(key,)]
                return True
        return False

    def delete_batch(self, keys: Iterable[Any]) -> int:
        """Delete multiple records by primary keys."""
        deleted = 0
        for k in keys:
            if self.delete(k):
                deleted += 1
        return deleted

    def delete_record(self, record: Dict[str, Any]) -> bool:
        """Delete a single record by extracting its primary key."""
        try:
            pk = self.extract_primary_key(record)
        except ValueError:
            return False
        return self.delete(pk)

    def delete_records(self, records: Iterable[Dict[str, Any]]) -> int:
        """Delete multiple records by extracting their primary keys."""
        keys = []
        for r in records:
            try:
                keys.append(self.extract_primary_key(r))
            except ValueError:
                continue
        return self.delete_batch(keys)

    def clear(self) -> None:
        """Clear all stored records."""
        self._records.clear()

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all records in the store."""
        return [dict(r) for r in self._records.values()]