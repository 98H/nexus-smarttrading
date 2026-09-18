from typing import Iterable, Optional, Set


class BuildScript:
    """Represents a versioned build script with tagging metadata."""

    def __init__(
        self,
        id: str,
        name: str,
        content: str,
        version: str,
        tags: Optional[Iterable[str]] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.content = content
        self.version = version
        self.tags: Set[str] = set(tags) if tags is not None else set()
        self.parent_id = parent_id

    def add_tags(self, tags: Iterable[str]) -> None:
        """Add new tags to the script, ignoring duplicates."""
        if tags:
            self.tags.update(tags)

    def __repr__(self) -> str:
        return (
            f"BuildScript(id={self.id!r}, name={self.name!r}, "
            f"version={self.version!r}, tags={self.tags!r}, parent_id={self.parent_id!r})"
        )