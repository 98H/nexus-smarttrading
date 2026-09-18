import pytest
from src.build_scripts.models import BuildScript
from src.build_scripts.repository import BuildScriptRepository


@pytest.fixture
def empty_repository() -> BuildScriptRepository:
    """Provides a fresh, empty BuildScriptRepository instance."""
    return BuildScriptRepository()


@pytest.fixture
def base_script() -> BuildScript:
    """Provides a baseline valid BuildScript instance."""
    return BuildScript(
        id="script-base-001",
        name="linux-deploy",
        content="echo 'deploying to staging' && ./deploy.sh",
        version="1.0.0",
        tags={"deploy", "linux", "staging"},
        parent_id=None,
    )


@pytest.fixture
def populated_repository(empty_repository: BuildScriptRepository) -> BuildScriptRepository:
    """
    Provides a repository populated with distinct build scripts to test
    keyword search and tag filtering boundaries.
    """
    scripts = [
        BuildScript(
            id="bs-1",
            name="docker-build-image",
            content="docker build -t app:latest . && docker push app:latest",
            version="1.0.0",
            tags={"docker", "ci", "linux"},
        ),
        BuildScript(
            id="bs-2",
            name="docker-test-runner",
            content="pytest tests/ && docker system prune -f",
            version="1.2.0",
            tags={"docker", "test", "python"},
        ),
        BuildScript(
            id="bs-3",
            name="node-compile-assets",
            content="npm run build && npm test",
            version="2.0.1",
            tags={"node", "frontend", "ci"},
        ),
        BuildScript(
            id="bs-4",
            name="python-security-scan",
            content="bandit -r src/ && safety check",
            version="0.9.0",
            tags={"security", "ci", "python"},
        ),
    ]
    for script in scripts:
        empty_repository.add(script)
    return empty_repository


# ==============================================================================
# Model Unit Tests: BuildScript
# ==============================================================================

class TestBuildScriptModel:
    """Direct tests for the BuildScript domain model."""

    def test_build_script_instantiation_stores_attributes_correctly(self):
        # Given / When
        script = BuildScript(
            id="script-100",
            name="compile-binaries",
            content="gcc -O3 main.c -o app",
            version="1.0.0",
            tags={"c", "release"},
            parent_id="script-000",
        )

        # Then
        assert script.id == "script-100"
        assert script.name == "compile-binaries"
        assert script.content == "gcc -O3 main.c -o app"
        assert script.version == "1.0.0"
        assert set(script.tags) == {"c", "release"}
        assert script.parent_id == "script-000"

    def test_model_add_tags_appends_and_deduplicates(self, base_script: BuildScript):
        # Given
        initial_tags = set(base_script.tags)

        # When
        base_script.add_tags(["production", "linux", "v2"])

        # Then
        assert set(base_script.tags) == initial_tags | {"production", "v2"}

    def test_model_tags_collection_is_isolated_from_input_mutation(self):
        # Given
        input_tags = ["ci", "test"]
        script = BuildScript(
            id="iso-1",
            name="isolated-test",
            content="pytest",
            version="1.0.0",
            tags=input_tags,
        )

        # When
        input_tags.append("mutated")

        # Then
        assert "mutated" not in script.tags


# ==============================================================================
# Acceptance Criterion 1: Build Script Search (Keywords + Filter Tags)
# ==============================================================================

class TestBuildScriptSearch:
    """
    Acceptance Criterion:
    Given an existing repository of build scripts with assigned tags,
    When a search query is submitted specifying text keywords and filter tags,
    Then only build scripts matching both the query keywords and the required tags are returned.
    """

    def test_search_returns_scripts_matching_both_keywords_and_required_tags(
        self, populated_repository: BuildScriptRepository
    ):
        # Given: 'bs-1' has keyword 'docker' and tags {'docker', 'ci', 'linux'}
        #        'bs-2' has keyword 'docker' and tags {'docker', 'test', 'python'}
        # When
        results = populated_repository.search(query="docker", tags=["ci"])

        # Then: Only bs-1 matches both the keyword 'docker' and required tag 'ci'
        result_ids = [s.id for s in results]
        assert result_ids == ["bs-1"]

    def test_search_excludes_script_matching_keyword_but_lacking_required_tag(
        self, populated_repository: BuildScriptRepository
    ):
        # Given: bs-2 matches keyword 'pytest', but has tags {'docker', 'test', 'python'}
        # When: Searching for keyword 'pytest' with non-matching tag 'security'
        results = populated_repository.search(query="pytest", tags=["security"])

        # Then
        assert results == []

    def test_search_excludes_script_matching_tags_but_lacking_keyword(
        self, populated_repository: BuildScriptRepository
    ):
        # Given: bs-3 has tags {'ci', 'frontend'}, but content/name does not contain 'docker'
        # When: Searching for keyword 'docker' with tag 'frontend'
        results = populated_repository.search(query="docker", tags=["frontend"])

        # Then
        assert results == []

    def test_search_requires_all_filter_tags_to_be_present(
        self, populated_repository: BuildScriptRepository
    ):
        # Given: bs-1 has {'docker', 'ci', 'linux'}.
        # Searching for 'docker' with tags ['ci', 'linux'] should match bs-1.
        results_match = populated_repository.search(query="docker", tags=["ci", "linux"])
        assert [s.id for s in results_match] == ["bs-1"]

        # Searching for 'docker' with tags ['ci', 'linux', 'test'] must NOT match bs-1
        results_no_match = populated_repository.search(
            query="docker", tags=["ci", "linux", "test"]
        )
        assert results_no_match == []

    def test_search_matches_keywords_in_script_name_or_content(
        self, populated_repository: BuildScriptRepository
    ):
        # 'docker-build-image' matches by name
        by_name = populated_repository.search(query="build-image", tags=["docker"])
        assert [s.id for s in by_name] == ["bs-1"]

        # 'bandit' matches only in content of bs-4
        by_content = populated_repository.search(query="bandit", tags=["security"])
        assert [s.id for s in by_content] == ["bs-4"]

    def test_search_is_case_insensitive_for_keywords(
        self, populated_repository: BuildScriptRepository
    ):
        # When
        results = populated_repository.search(query="DOCKER", tags=["ci"])

        # Then
        assert len(results) == 1
        assert results[0].id == "bs-1"

    def test_search_with_keywords_only_without_filter_tags(
        self, populated_repository: BuildScriptRepository
    ):
        # When
        results = populated_repository.search(query="docker")

        # Then: bs-1 and bs-2 contain 'docker'
        result_ids = {s.id for s in results}
        assert result_ids == {"bs-1", "bs-2"}

    def test_search_with_tags_only_without_keywords(
        self, populated_repository: BuildScriptRepository
    ):
        # When
        results = populated_repository.search(tags=["ci"])

        # Then: bs-1, bs-3, bs-4 all have the 'ci' tag
        result_ids = {s.id for s in results}
        assert result_ids == {"bs-1", "bs-3", "bs-4"}

    def test_search_with_no_criteria_returns_all_scripts(
        self, populated_repository: BuildScriptRepository
    ):
        # When
        results = populated_repository.search()

        # Then
        assert len(results) == 4

    def test_search_empty_repository_returns_empty_list(
        self, empty_repository: BuildScriptRepository
    ):
        # When
        results = empty_repository.search(query="any", tags=["any"])

        # Then
        assert results == []


# ==============================================================================
# Acceptance Criterion 2: Tagging Existing Build Scripts
# ==============================================================================

class TestBuildScriptTagging:
    """
    Acceptance Criterion:
    Given an existing build script,
    When new tags are applied to the script,
    Then the script's metadata is updated to include the tags
    and it becomes searchable under those tags.
    """

    def test_apply_tags_updates_script_metadata(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        updated_script = empty_repository.add_tags(
            script_id=base_script.id, tags=["production", "aws"]
        )

        # Then
        assert {"production", "aws"}.issubset(set(updated_script.tags))
        persisted = empty_repository.get(base_script.id)
        assert {"production", "aws"}.issubset(set(persisted.tags))

    def test_script_becomes_searchable_under_newly_applied_tags(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)
        # Verify initially not found with future tag
        assert empty_repository.search(query=base_script.name, tags=["k8s"]) == []

        # When
        empty_repository.add_tags(script_id=base_script.id, tags=["k8s"])

        # Then: Found when searching both with keyword and newly added tag
        results = empty_repository.search(query=base_script.name, tags=["k8s"])
        assert len(results) == 1
        assert results[0].id == base_script.id
        assert "k8s" in results[0].tags

    def test_apply_tags_preserves_preexisting_tags(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        original_tags = set(base_script.tags)
        empty_repository.add(base_script)

        # When
        updated = empty_repository.add_tags(script_id=base_script.id, tags=["nightly"])

        # Then
        assert original_tags.issubset(set(updated.tags))
        assert "nightly" in updated.tags

    def test_apply_duplicate_tags_is_idempotent(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When: Re-applying existing tags together with a new one
        updated = empty_repository.add_tags(
            script_id=base_script.id, tags=["deploy", "deploy", "canary"]
        )

        # Then
        tags_list = list(updated.tags)
        assert tags_list.count("deploy") == 1
        assert "canary" in updated.tags

    def test_apply_tags_to_nonexistent_script_raises_exception(
        self, empty_repository: BuildScriptRepository
    ):
        # Given an empty repository
        # When / Then
        with pytest.raises((KeyError, ValueError)):
            empty_repository.add_tags(script_id="non-existent-id", tags=["new-tag"])

    def test_apply_empty_tags_iterable_does_not_modify_metadata(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)
        original_tags = set(base_script.tags)

        # When
        updated = empty_repository.add_tags(script_id=base_script.id, tags=[])

        # Then
        assert set(updated.tags) == original_tags


# ==============================================================================
# Acceptance Criterion 3: Versioned Forking
# ==============================================================================

class TestBuildScriptForking:
    """
    Acceptance Criterion:
    Given an existing versioned build script,
    When a user forks the script,
    Then a new build script instance is created with a reference to the parent
    script ID, an initialized fork version identifier, and cloned tags.
    """

    def test_fork_creates_new_instance_referencing_parent_id(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(script_id=base_script.id)

        # Then
        assert forked_script is not None
        assert forked_script.id != base_script.id
        assert forked_script.parent_id == base_script.id

    def test_fork_initializes_fork_version_identifier(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(script_id=base_script.id)

        # Then: Fork version identifier must be initialized, distinct, and non-empty
        assert forked_script.version is not None
        assert isinstance(forked_script.version, str)
        assert len(forked_script.version.strip()) > 0
        assert forked_script.version != base_script.version

    def test_fork_supports_explicit_initial_fork_version(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(
            script_id=base_script.id, new_version="1.0.0-fork.1"
        )

        # Then
        assert forked_script.version == "1.0.0-fork.1"

    def test_fork_clones_tags_from_parent(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(script_id=base_script.id)

        # Then
        assert set(forked_script.tags) == set(base_script.tags)

    def test_fork_tags_are_isolated_from_parent_mutations(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)
        forked_script = empty_repository.fork(script_id=base_script.id)

        # When: Adding a new tag to the fork
        empty_repository.add_tags(forked_script.id, ["fork-only-tag"])

        # Then: Parent script does not receive the new tag
        parent_current = empty_repository.get(base_script.id)
        fork_current = empty_repository.get(forked_script.id)
        assert "fork-only-tag" in fork_current.tags
        assert "fork-only-tag" not in parent_current.tags

        # When: Adding a new tag to the parent
        empty_repository.add_tags(base_script.id, ["parent-only-tag"])

        # Then: Fork does not receive parent's new tag
        parent_current = empty_repository.get(base_script.id)
        fork_current = empty_repository.get(forked_script.id)
        assert "parent-only-tag" in parent_current.tags
        assert "parent-only-tag" not in fork_current.tags

    def test_fork_preserves_content_and_script_lineage(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(script_id=base_script.id)

        # Then
        assert forked_script.content == base_script.content
        assert base_script.name in forked_script.name

    def test_fork_persists_in_repository_and_is_retrievable(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)

        # When
        forked_script = empty_repository.fork(script_id=base_script.id)

        # Then
        retrieved = empty_repository.get(forked_script.id)
        assert retrieved.id == forked_script.id
        assert retrieved.parent_id == base_script.id

    def test_fork_of_a_fork_tracks_immediate_parent_id(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)
        fork_1 = empty_repository.fork(script_id=base_script.id)

        # When
        fork_2 = empty_repository.fork(script_id=fork_1.id)

        # Then
        assert fork_2.parent_id == fork_1.id
        assert fork_1.parent_id == base_script.id
        assert base_script.parent_id is None

    def test_fork_nonexistent_script_raises_exception(
        self, empty_repository: BuildScriptRepository
    ):
        # Given empty repository
        # When / Then
        with pytest.raises((KeyError, ValueError)):
            empty_repository.fork(script_id="missing-id")

    def test_fork_is_searchable_alongside_parent_under_cloned_tags(
        self, empty_repository: BuildScriptRepository, base_script: BuildScript
    ):
        # Given
        empty_repository.add(base_script)
        forked = empty_repository.fork(script_id=base_script.id)

        # When: Searching by shared keyword and shared tag
        results = empty_repository.search(query=base_script.name, tags=["linux"])

        # Then: Both parent and fork are matched
        result_ids = {s.id for s in results}
        assert result_ids == {base_script.id, forked.id}