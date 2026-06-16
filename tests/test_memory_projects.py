"""Tests for the memory-project detection helper.

`detect_active_memory_project` is a pure read-only helper: it takes
a workspace path and the registry dict and returns the matching
`ActiveMemoryProject`, or None when no root matches. These tests
pin the algorithm's correctness pieces: path containment (not
string prefix), longest-prefix tie-breaking, symlink resolution
before matching, and the disabled-project semantics that later
retrieval and write paths will depend on.

Detection wiring into pool/backend is deferred to #544; this issue
only ships the helper.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest

from kai.config import MemoryProjectConfig
from kai.memory_projects import ActiveMemoryProject, detect_active_memory_project


def _project(
    project_id: str,
    *roots,
    memory_enabled: bool = True,
    default_scope: str | None = None,
):
    """Build a `MemoryProjectConfig` from a list of roots. Test-only
    convenience so each test reads as data, not as scaffolding."""
    return MemoryProjectConfig(
        project_id=project_id,
        display_name=project_id.capitalize(),
        workspace_roots=tuple(r.resolve() for r in roots),
        memory_enabled=memory_enabled,
        default_scope_for_new_facts=default_scope,
    )


class TestDetectActiveMemoryProject:
    def test_detect_active_memory_project_exact_root(self, tmp_path):
        """A cwd that equals a registered root matches it. Equality
        is structurally different from `is_relative_to()` (which
        returns False for equal paths), so the detector handles both
        as a single "matched" outcome."""
        root = tmp_path / "kai"
        root.mkdir()
        projects = {"kai": _project("kai", root)}

        active = detect_active_memory_project(root, projects)

        assert active is not None
        assert active.project_id == "kai"
        assert active.matched_root == root.resolve()
        assert active.memory_enabled is True

    def test_detect_active_memory_project_child_path(self, tmp_path):
        """A nested path under a registered root resolves to the
        project that owns the root. Path containment is the matching
        rule; arbitrary descendants count."""
        root = tmp_path / "kai"
        nested = root / "src" / "kai" / "memory_projects.py"
        nested.parent.mkdir(parents=True)
        nested.touch()
        projects = {"kai": _project("kai", root)}

        active = detect_active_memory_project(nested, projects)

        assert active is not None
        assert active.project_id == "kai"
        assert active.matched_root == root.resolve()

    def test_detect_active_memory_project_uses_longest_matching_root(self, tmp_path):
        """When two registered roots both contain the cwd, the root
        with the most path parts wins. Common config shape: a broad
        umbrella root and one or more narrower child roots; the
        narrower one is the authoritative match."""
        # Two roots configured under different projects; both contain
        # the cwd `cwd_path`. The narrower root (more path parts)
        # belongs to the inner project; it should win.
        umbrella = tmp_path / "projects"
        umbrella.mkdir()
        inner = umbrella / "kai"
        inner.mkdir()
        cwd_path = inner / "src" / "kai"
        cwd_path.mkdir(parents=True)
        projects = {
            "umbrella": _project("umbrella", umbrella),
            "kai": _project("kai", inner),
        }

        active = detect_active_memory_project(cwd_path, projects)

        assert active is not None
        assert active.project_id == "kai"
        assert active.matched_root == inner.resolve()

    def test_detect_active_memory_project_rejects_string_prefix_match(self, tmp_path):
        """`/work/foo2` MUST NOT match a root of `/work/foo`. This
        is the load-bearing correctness pin for using
        `Path.is_relative_to()` over `str.startswith()`. A
        regression here would silently merge unrelated projects."""
        sibling_root = tmp_path / "foo"
        sibling_root.mkdir()
        confusing_cwd = tmp_path / "foo2"
        confusing_cwd.mkdir()
        projects = {"foo": _project("foo", sibling_root)}

        active = detect_active_memory_project(confusing_cwd, projects)

        assert active is None

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink fixtures require os.symlink, skipped on Windows")
    def test_detect_active_memory_project_resolves_symlink_before_matching(self, tmp_path):
        """A cwd that reaches a registered root through a symlink
        resolves to the canonical path before matching. This keeps
        memory scope tied to filesystem identity instead of the
        access alias, preventing one project from looking like two
        depending on which path the operator used."""
        real_root = tmp_path / "real_kai"
        real_root.mkdir()
        link_path = tmp_path / "kai_alias"
        os.symlink(real_root, link_path)
        projects = {"kai": _project("kai", real_root)}

        # Enter via the symlink alias; detection should resolve it
        # to the canonical real_root before matching.
        active = detect_active_memory_project(link_path, projects)

        assert active is not None
        assert active.project_id == "kai"
        assert active.matched_root == real_root.resolve()

    def test_detect_active_memory_project_unknown_workspace_returns_none(self, tmp_path):
        """A cwd that is not inside any registered root returns
        None. The detector does not guess; callers fall back to
        global-only behavior."""
        root = tmp_path / "known"
        root.mkdir()
        unknown_cwd = tmp_path / "elsewhere"
        unknown_cwd.mkdir()
        projects = {"known": _project("known", root)}

        active = detect_active_memory_project(unknown_cwd, projects)

        assert active is None

    def test_detect_active_memory_project_disabled_project_detected_but_not_allowed(self, tmp_path):
        """A disabled project still returns an `ActiveMemoryProject`
        so logs and the `memory.recall` payload's `scoped_debug` block
        can distinguish "no project here" from "known project with
        memory disabled". The `memory_enabled=False` signal is what
        retrieval and write paths consume; the helper does not make
        the global-only decision itself."""
        root = tmp_path / "disabled_proj"
        root.mkdir()
        projects = {"disabled_proj": _project("disabled_proj", root, memory_enabled=False)}

        active = detect_active_memory_project(root, projects)

        assert active is not None
        assert active.project_id == "disabled_proj"
        assert active.memory_enabled is False

    def test_detect_active_memory_project_empty_registry_returns_none(self, tmp_path):
        """Defensive: an empty registry short-circuits to None
        before any path work happens. Pinning this stops a future
        refactor from doing pointless filesystem work on the common
        no-projects-configured path."""
        active = detect_active_memory_project(tmp_path, {})
        assert active is None


class TestActiveMemoryProjectDataclass:
    """Light pin that the returned dataclass shape is frozen and
    carries the exact fields the helper documents. Catches
    accidental field rename or attribute injection in a refactor."""

    def test_dataclass_is_frozen_and_carries_expected_fields(self, tmp_path):
        root = tmp_path
        active = ActiveMemoryProject(
            project_id="x",
            display_name="X",
            matched_root=root,
            memory_enabled=True,
            default_scope_for_new_facts="project",
        )
        # Frozen: attribute assignment must raise FrozenInstanceError.
        with pytest.raises(dataclasses.FrozenInstanceError):
            active.project_id = "y"  # type: ignore[misc]
        # Field shape is what callers will rely on.
        assert active.project_id == "x"
        assert active.display_name == "X"
        assert active.matched_root == root
        assert active.memory_enabled is True
        assert active.default_scope_for_new_facts == "project"


# ── DB registry cache and merge ──────────────────────────────────────


from kai.memory_projects import (  # noqa: E402
    db_registry_creator,
    db_registry_remove,
    db_registry_upsert,
    load_db_registry,
    merged_registry,
)


@pytest.fixture(autouse=True)
def _reset_db_registry():
    """The DB registry cache is module-level state; tests must not
    leak entries into each other (or into the detector tests above,
    which assume a YAML-only world)."""
    load_db_registry([])
    yield
    load_db_registry([])


def _row(project_id: str, root, *, created_by: int = 123, **overrides) -> dict:
    row = {
        "project_id": project_id,
        "display_name": project_id.capitalize(),
        "workspace_root": str(root),
        "memory_enabled": True,
        "default_scope_for_new_facts": "project",
        "created_by": created_by,
    }
    row.update(overrides)
    return row


class TestDbRegistryCache:
    def test_load_and_merge_with_empty_yaml(self, tmp_path):
        root = tmp_path / "phi"
        root.mkdir()
        load_db_registry([_row("phi", root)])
        merged = merged_registry({})
        assert set(merged) == {"phi"}
        assert merged["phi"].workspace_roots == (root.resolve(),)
        assert merged["phi"].default_scope_for_new_facts == "project"
        assert db_registry_creator("phi") == 123

    def test_empty_db_returns_yaml_dict_unchanged(self, tmp_path):
        """The pre-registration call sites must behave byte-
        identically: with no DB rows the merge returns the exact
        yaml dict object, not a copy."""
        root = tmp_path / "kai"
        root.mkdir()
        yaml_projects = {"kai": _project("kai", root)}
        assert merged_registry(yaml_projects) is yaml_projects

    def test_yaml_wins_on_project_id(self, tmp_path):
        yaml_root = tmp_path / "kai-yaml"
        yaml_root.mkdir()
        db_root = tmp_path / "kai-db"
        db_root.mkdir()
        load_db_registry([_row("kai", db_root)])
        merged = merged_registry({"kai": _project("kai", yaml_root)})
        assert merged["kai"].workspace_roots == (yaml_root.resolve(),)

    def test_yaml_wins_on_root_ownership(self, tmp_path):
        """A DB project whose root is already owned by a YAML project
        is dropped entirely; a user registration can never steal an
        operator-pinned root under a different id."""
        root = tmp_path / "kai"
        root.mkdir()
        load_db_registry([_row("kai-imposter", root)])
        merged = merged_registry({"kai": _project("kai", root)})
        assert "kai-imposter" not in merged
        assert set(merged) == {"kai"}

    def test_yaml_parent_evicts_persisted_db_child_root(self, tmp_path):
        """Containment, not just equality: a persisted DB root INSIDE
        a later-pinned YAML root would win that subtree via
        longest-prefix detection, so the merge must evict it. The
        register-time nested guard cannot catch this ordering (the
        DB row predates the YAML pin); the merge is the only gate."""
        parent = tmp_path / "parent"
        child = parent / "child"
        deep = child / "src"
        deep.mkdir(parents=True)
        load_db_registry([_row("child", child)])
        merged = merged_registry({"parent": _project("parent", parent)})
        assert "child" not in merged
        active = detect_active_memory_project(deep, merged)
        assert active is not None
        assert active.project_id == "parent"

    def test_yaml_child_inside_db_parent_coexists(self, tmp_path):
        """The inverse direction is ordinary nested-project
        coexistence: a deeper YAML root owns its subtree through
        longest-prefix while the DB parent keeps the surrounding
        area, exactly like nested YAML entries."""
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        load_db_registry([_row("parent", parent)])
        merged = merged_registry({"child": _project("child", child)})
        assert set(merged) == {"parent", "child"}
        inside_child = detect_active_memory_project(child, merged)
        assert inside_child is not None and inside_child.project_id == "child"
        inside_parent = detect_active_memory_project(parent, merged)
        assert inside_parent is not None and inside_parent.project_id == "parent"

    def test_upsert_and_remove_visible_immediately(self, tmp_path):
        """Restart-free contract: cache mutations show up in the very
        next merged_registry call."""
        root = tmp_path / "anvil"
        root.mkdir()
        assert db_registry_upsert(_row("anvil", root, created_by=456)) is True
        assert "anvil" in merged_registry({})
        assert db_registry_creator("anvil") == 456
        db_registry_remove("anvil")
        assert merged_registry({}) == {}
        assert db_registry_creator("anvil") is None

    @pytest.mark.parametrize(
        "bad_overrides",
        [
            {"project_id": ""},
            {"project_id": None},
            {"display_name": ""},
            {"memory_enabled": "true"},
            {"default_scope_for_new_facts": "task"},
            {"workspace_root": ""},
        ],
    )
    def test_malformed_rows_skipped_fail_closed(self, tmp_path, bad_overrides):
        """Validation parity with the YAML loader: a row the loader
        would have skipped never reaches detection, and never
        crashes the load."""
        root = tmp_path / "phi"
        root.mkdir()
        row = _row("phi", root)
        row.update(bad_overrides)
        load_db_registry([row])
        assert merged_registry({}) == {}

    def test_detection_through_merged_registry(self, tmp_path):
        """End-to-end with the detector: a cached DB project is
        detectable exactly like a YAML one."""
        root = tmp_path / "phi"
        root.mkdir()
        sub = root / "src"
        sub.mkdir()
        load_db_registry([_row("phi", root)])
        active = detect_active_memory_project(sub, merged_registry({}))
        assert active is not None
        assert active.project_id == "phi"
        assert active.matched_root == root.resolve()
