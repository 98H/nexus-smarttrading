import uuid
from typing import Dict, Iterable, List, Optional

from src.build_scripts.models import BuildScript


class BuildScriptRepository:
    """Repository for storing, searching, tagging, and forking build scripts."""

    def __init__(self) -> None:
        self._scripts: Dict[str, BuildScript] = {}

    def add(self, script: BuildScript) -> None:
        """Add or update a build script in the repository."""
        self._scripts[script.id] = script

    def get(self, script_id: str) -> BuildScript:
        """Retrieve a build script by its unique identifier."""
        if script_id not in self._scripts:
            raise KeyError(f"BuildScript with id '{script_id}' not found.")
        return self._scripts[script_id]

    def add_tags(self, script_id: str, tags: Iterable[str]) -> BuildScript:
        """Apply new tags to an existing build script."""
        script = self.get(script_id)
        script.add_tags(tags)
        return script

    def fork(
        self,
        script_id: str,
        new_version: Optional[str] = None,
        new_id: Optional[str] = None,
        new_name: Optional[str] = None,
    ) -> BuildScript:
        """
        Create a fork of an existing build script.

        Creates a new build script instance referencing the parent script ID,
        with an initialized fork version identifier and cloned tags.
        """
        parent = self.get(script_id)
        fork_id = new_id or f"{parent.id}-fork-{uuid.uuid4().hex[:8]}"
        fork_version = new_version if new_version is not None else f"{parent.version}-fork.1"
        fork_name = new_name if new_name is not None else parent.name

        forked_script = BuildScript(
            id=fork_id,
            name=fork_name,
            content=parent.content,
            version=fork_version,
            tags=set(parent.tags),
            parent_id=parent.id,
        )
        self.add(forked_script)
        return forked_script

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[BuildScript]:
        """
        Search for build scripts matching query keywords and required tags.

        Both query keywords and all required tags must match if provided.
        """
        results: List[BuildScript] = []
        required_tags = set(tags) if tags is not None else None

        for script in self._scripts.values():
            if required_tags is not None:
                if not required_tags.issubset(script.tags):
                    continue

            if query is not None and query.strip():
                q_lower = query.lower()
                name_lower = script.name.lower()
                content_lower = script.content.lower()

                match_full = q_lower in name_lower or q_lower in content_lower
                words = q_lower.split()
                match_words = bool(words) and all(
                    w in name_lower or w in content_lower for w in words
                )

                if not (match_full or match_words):
                    continue

            results.append(script)

        return results