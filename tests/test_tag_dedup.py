"""
Tests for scripts/tag-dedup.py (issue #418, Sub D of #388).

The script lives at scripts/tag-dedup.py because that is the project
convention for one-shot operator scripts. The hyphenated filename is
not a legal Python module name, so this test loads it via
importlib.util at module-import time and exposes the helpers as
`dedup.<helper>` for the test functions below.

Tests cover:
- compute_distribution: empty, single tag, multi-tag, case-preserved counts.
- identify_clusters: case-insensitive equality, plural/singular pairs,
  -ies/-y swap, -es removal for sibilant endings, singletons,
  confirmed_action skip (lowercase and mixed-case).
- canonical selection: most-frequent wins, lexicographic tiebreak,
  single-member cluster's canonical is itself.
- apply_cluster_to_row: no-change, single rewrite, multi rewrite,
  within-row dedup, confirmed_action row untouched, fixture-driven
  row-level guard against a cluster that bypasses cluster-skip.
- apply_rewrites: success path with metadata preservation, failure path,
  audit-log entry shape.
- CLI dispatch: dry-run vs apply, --all-users iteration, mutex
  enforcement, exit codes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from kai.memory import MemoryResult

# Load the dedup script as a module so its helpers are importable
# despite the hyphenated filename. spec_from_file_location is the
# stdlib idiom for loading code from an arbitrary path; the loaded
# module behaves like any other import target.
#
# CRITICAL: register in sys.modules BEFORE exec_module(). The script
# uses @dataclass, whose decorator looks up cls.__module__ in
# sys.modules at class-creation time. Without the pre-registration,
# the loaded module is not yet visible there and dataclass crashes
# with AttributeError on a NoneType __dict__ lookup.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "tag-dedup.py"
_spec = importlib.util.spec_from_file_location("tag_dedup", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
dedup = importlib.util.module_from_spec(_spec)
sys.modules["tag_dedup"] = dedup
_spec.loader.exec_module(dedup)


# ── Fixture helpers ───────────────────────────────────────────────────


def _row(
    fact_id: str,
    text: str,
    tags: list[str],
    source: str = "extracted",
    extra_metadata: dict[str, Any] | None = None,
) -> MemoryResult:
    """Construct a MemoryResult with the metadata shape the dedup pass
    actually walks. `extra_metadata` lets a test pin non-tag fields
    (confidence, prompt_version, etc.) so the apply-pass tests can
    verify they survive the read-merge-write."""
    metadata = {"source": source, "tags": list(tags), "type": "fact"}
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return MemoryResult(
        id=fact_id,
        text=text,
        score=0.0,
        memory_type="fact",
        metadata=metadata,
        created_at="2026-04-30T13:00:00",
        updated_at="2026-04-30T13:00:00",
    )


# ── compute_distribution ──────────────────────────────────────────────


class TestComputeDistribution:
    def test_empty_rows_returns_empty_distribution(self):
        assert dedup.compute_distribution([]) == {}

    def test_single_tag_row(self):
        rows = [_row("a", "x", ["preference"])]
        assert dedup.compute_distribution(rows) == {"preference": 1}

    def test_multi_tag_row(self):
        rows = [_row("a", "x", ["preference", "constraint"])]
        assert dedup.compute_distribution(rows) == {"preference": 1, "constraint": 1}

    def test_case_preserved_counts(self):
        # Case is preserved at this stage so the canonical-selection step
        # in identify_clusters can pick the most-frequent variant. A
        # case-folding distribution would lose the variant data.
        rows = [
            _row("a", "x", ["Preference"]),
            _row("b", "y", ["preference"]),
            _row("c", "z", ["preference"]),
        ]
        assert dedup.compute_distribution(rows) == {"Preference": 1, "preference": 2}

    def test_missing_tags_field_contributes_nothing(self):
        # A row with no tags key (or empty list) must not crash and
        # must not appear in the distribution.
        result = MemoryResult(
            id="a",
            text="x",
            score=0.0,
            memory_type="fact",
            metadata={"source": "extracted"},  # no tags key
            created_at="2026-04-30T13:00:00",
            updated_at="2026-04-30T13:00:00",
        )
        assert dedup.compute_distribution([result]) == {}


# ── identify_clusters ─────────────────────────────────────────────────


class TestIdentifyClusters:
    def test_case_insensitive_equality_clusters(self):
        clusters = dedup.identify_clusters({"Preference": 1, "preference": 5})
        assert len(clusters) == 1
        assert sorted(clusters[0].members) == ["Preference", "preference"]
        assert clusters[0].canonical == "preference"  # most-frequent
        assert clusters[0].total_occurrences == 6

    def test_plural_singular_s_pair(self):
        # Both forms exist; the bare -s rule fires.
        clusters = dedup.identify_clusters({"preferences": 2, "preference": 5})
        assert len(clusters) == 1
        assert sorted(clusters[0].members) == ["preference", "preferences"]
        assert clusters[0].canonical == "preference"

    def test_ies_y_swap(self):
        clusters = dedup.identify_clusters({"categories": 1, "category": 4})
        assert len(clusters) == 1
        assert sorted(clusters[0].members) == ["categories", "category"]
        assert clusters[0].canonical == "category"

    def test_sibilant_es_removal(self):
        # `boxes` -> `box` only fires because (a) `boxes` ends in `-xes`
        # and (b) `box` is also in the corpus.
        clusters = dedup.identify_clusters({"boxes": 1, "box": 3})
        assert len(clusters) == 1
        assert sorted(clusters[0].members) == ["box", "boxes"]
        assert clusters[0].canonical == "box"

    def test_three_way_case_and_plural_collapse(self):
        # A case-fold pair AND a plural/singular link must collapse into
        # one cluster, not two. The earlier two-pass implementation
        # eagerly clustered `["Preferences", "preferences"]` in pass 1,
        # then refused to link them to `"preference"` in pass 2 because
        # the plural side was already assigned. Union-find over case-fold
        # keys joins all three into one component.
        clusters = dedup.identify_clusters({"Preferences": 1, "preferences": 2, "preference": 5})
        assert len(clusters) == 1
        assert sorted(clusters[0].members) == ["Preferences", "preference", "preferences"]
        # Most-frequent wins: "preference" with 5 occurrences.
        assert clusters[0].canonical == "preference"
        assert clusters[0].total_occurrences == 8

    def test_singleton_tag_produces_no_cluster(self):
        # A tag with no near-duplicate in the corpus is not in a cluster.
        clusters = dedup.identify_clusters({"preference": 5})
        assert clusters == []

    def test_bare_s_rule_skips_double_s_endings(self):
        # The bare-s rule has an explicit `not endswith("ss")` guard
        # so genuinely-singular -ss nouns are not paired with a stem.
        # Fixture: "mass" (singular, ends in -ss) plus "ma" (a
        # hypothetical separate tag); the bare-s rule must NOT pair
        # them because "mass" doesn't end in a single trailing -s.
        clusters = dedup.identify_clusters({"mass": 5, "ma": 1})
        assert clusters == []

    def test_confirmed_action_lowercase_skipped(self):
        # The cluster-skip rule fires when any case-fold member equals
        # `confirmed_action`. Verified by giving the corpus two case
        # variants that would cluster under rule 1 but should be left
        # untouched.
        clusters = dedup.identify_clusters({"confirmed_action": 5, "Confirmed_Action": 1})
        assert clusters == []

    def test_confirmed_action_mixed_case_skipped(self):
        # Even an all-uppercase variant cannot drag the cluster into
        # play; case-fold equality is the comparison key.
        clusters = dedup.identify_clusters({"CONFIRMED_ACTION": 1, "confirmed_action": 7})
        assert clusters == []


# ── canonical selection ───────────────────────────────────────────────


class TestCanonicalSelection:
    def test_most_frequent_wins(self):
        clusters = dedup.identify_clusters({"alpha": 10, "Alpha": 1})
        assert clusters[0].canonical == "alpha"

    def test_lexicographic_tiebreak(self):
        # Equal counts: "Alpha" sorts before "alpha" under lexicographic
        # comparison (uppercase ASCII precedes lowercase). Pin so a future
        # change to the sort key trips this regression.
        clusters = dedup.identify_clusters({"alpha": 1, "Alpha": 1})
        assert clusters[0].canonical == "Alpha"

    def test_single_member_cluster_canonical_is_itself(self):
        # Direct construction: bypasses identify_clusters (which would
        # reject single-member groups). Verifies the canonical-selection
        # helper handles the trivial case.
        canonical = dedup._select_canonical(["sole"], {"sole": 3})
        assert canonical == "sole"


# ── apply_cluster_to_row ──────────────────────────────────────────────


class TestApplyClusterToRow:
    def test_no_change_row(self):
        # Tags already canonical; the rewrite is a no-op.
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        new_tags, changed = dedup.apply_cluster_to_row(["preference"], [cluster])
        assert new_tags == ["preference"]
        assert changed is False

    def test_single_tag_rewritten(self):
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        new_tags, changed = dedup.apply_cluster_to_row(["Preference"], [cluster])
        assert new_tags == ["preference"]
        assert changed is True

    def test_multi_tag_rewrite(self):
        c1 = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        c2 = dedup.Cluster(members=["constraint", "Constraint"], canonical="constraint")
        new_tags, changed = dedup.apply_cluster_to_row(["Preference", "Constraint"], [c1, c2])
        assert new_tags == ["preference", "constraint"]
        assert changed is True

    def test_within_row_duplicate_collapse(self):
        # Both `preference` and `Preference` rewrite to `preference`,
        # producing a duplicate that the within-row dedup pass must
        # collapse. The change flag is True because the list shape
        # changed from 2 elements to 1.
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        new_tags, changed = dedup.apply_cluster_to_row(["preference", "Preference"], [cluster])
        assert new_tags == ["preference"]
        assert changed is True

    def test_confirmed_action_row_untouched_via_cluster_skip(self):
        # Production path: identify_clusters refused to build a cluster
        # touching confirmed_action, so apply_cluster_to_row sees no
        # cluster matching the confirmation tag. The other tags are
        # rewritten, the confirmation tag stays put.
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        new_tags, changed = dedup.apply_cluster_to_row(["confirmed_action", "Preference"], [cluster])
        assert new_tags == ["confirmed_action", "preference"]
        assert changed is True

    def test_row_level_guard_when_cluster_bypasses_skip(self):
        # Defense-in-depth path: a future rule addition could sneak
        # confirmed_action into a cluster, bypassing the cluster-skip
        # in identify_clusters. Construct that hypothetical cluster
        # directly. The row-level guard in apply_cluster_to_row must
        # refuse to drop confirmed_action from the output and return
        # the row unchanged. The script never builds this cluster in
        # production; the test exists to pin the row-level guard's
        # contract against future rule additions.
        cluster = dedup.Cluster(
            members=["confirmed_action", "user_confirmed"],
            canonical="user_confirmed",
        )
        new_tags, changed = dedup.apply_cluster_to_row(["confirmed_action"], [cluster])
        assert new_tags == ["confirmed_action"]
        assert changed is False


# ── apply_rewrites ────────────────────────────────────────────────────


class TestApplyRewrites:
    def test_successful_rewrite_increments_success_count(self, monkeypatch, tmp_path):
        # Mock memory.update_metadata to succeed for every call. The
        # test verifies (a) success/failure counts, (b) the audit log
        # gets one JSONL entry per success, and (c) update_metadata is
        # called with a merged metadata dict that preserves non-tags
        # fields (the read-merge-write contract).
        captured_calls: list[dict] = []

        def fake_update(*, user_id, memory_id, data, metadata):
            captured_calls.append({"memory_id": memory_id, "metadata": metadata})
            return True

        from kai import memory as kai_memory

        monkeypatch.setattr(kai_memory, "update_metadata", fake_update)

        rows = [
            _row(
                "row-1",
                "fact text",
                ["Preference"],
                extra_metadata={
                    "confidence": 0.92,
                    "prompt_version": "5",
                    "session_id": "sess-abc",
                    "confirmation_quote": None,
                },
            )
        ]
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        audit_log = tmp_path / "audit.jsonl"

        success, failure = dedup.apply_rewrites(rows, [cluster], 12345, audit_log)

        assert success == 1
        assert failure == 0
        assert len(captured_calls) == 1

        # Read-merge-write verification: the metadata passed to
        # update_metadata MUST include every non-tags field from the
        # original row. Without read-merge-write the script would pass
        # only the tags and Mem0 would destroy the rest.
        passed_metadata = captured_calls[0]["metadata"]
        assert passed_metadata["tags"] == ["preference"]
        assert passed_metadata["confidence"] == 0.92
        assert passed_metadata["prompt_version"] == "5"
        assert passed_metadata["session_id"] == "sess-abc"
        assert passed_metadata["source"] == "extracted"

        # Audit log entry shape.
        audit_entries = [json.loads(line) for line in audit_log.read_text().splitlines()]
        assert len(audit_entries) == 1
        entry = audit_entries[0]
        assert entry["memory_id"] == "row-1"
        assert entry["user_id"] == "12345"
        assert entry["old_tags"] == ["Preference"]
        assert entry["new_tags"] == ["preference"]
        assert entry["old_text"] == "fact text"

    def test_failed_rewrite_increments_failure_count(self, monkeypatch, tmp_path):
        # update_metadata returns False (Mem0 raise, row vanished, etc.).
        # The audit log file is NOT created at all when no rewrite
        # succeeds; the lazy-flush path skips the open call entirely
        # so an `--apply` run that fails for every row leaves no
        # zero-byte file behind.
        from kai import memory as kai_memory

        monkeypatch.setattr(kai_memory, "update_metadata", lambda **kw: False)

        rows = [_row("row-1", "x", ["Preference"])]
        cluster = dedup.Cluster(members=["preference", "Preference"], canonical="preference")
        audit_log = tmp_path / "audit.jsonl"

        success, failure = dedup.apply_rewrites(rows, [cluster], 12345, audit_log)

        assert success == 0
        assert failure == 1
        # File should not exist: no entries means no flush.
        assert not audit_log.exists()


# ── CLI dispatch ──────────────────────────────────────────────────────


class TestCliDispatch:
    def test_chat_id_and_all_users_are_mutually_exclusive(self):
        # argparse exits with code 2 on mutex violation. SystemExit
        # carries that code; pytest.raises captures it.
        with pytest.raises(SystemExit) as excinfo:
            dedup.main(["--chat-id", "12345", "--all-users"])
        assert excinfo.value.code == 2

    def test_one_target_required(self):
        # Neither --chat-id nor --all-users is provided.
        with pytest.raises(SystemExit) as excinfo:
            dedup.main([])
        assert excinfo.value.code == 2

    def test_memory_disabled_returns_exit_code_2(self, monkeypatch):
        # When config.memory_enabled is False, the script must NOT
        # init_memory and must return EXIT_MEMORY_DISABLED.
        from kai import config as kai_config
        from kai import memory as kai_memory

        fake_config = MagicMock()
        fake_config.memory_enabled = False
        monkeypatch.setattr(kai_config, "load_config", lambda: fake_config)
        # init_memory must NOT be called when memory_enabled is False.
        monkeypatch.setattr(kai_memory, "init_memory", lambda c: pytest.fail("init_memory should not run"))
        rc = dedup.main(["--chat-id", "12345"])
        assert rc == dedup.EXIT_MEMORY_DISABLED

    def test_dry_run_does_not_call_update_metadata(self, monkeypatch):
        # Without --apply, update_metadata must never be called even
        # if the corpus has clusterable tags.
        from kai import config as kai_config
        from kai import memory as kai_memory

        fake_config = MagicMock()
        fake_config.memory_enabled = True
        fake_config.allowed_user_ids = {12345}
        monkeypatch.setattr(kai_config, "load_config", lambda: fake_config)
        monkeypatch.setattr(kai_memory, "init_memory", lambda c: None)
        # Two rows that would form a case-fold cluster.
        monkeypatch.setattr(
            dedup,
            "load_rows",
            lambda user_id: [
                _row("a", "x", ["Preference"]),
                _row("b", "y", ["preference"]),
            ],
        )
        # Trip the test if update_metadata is called.
        monkeypatch.setattr(
            kai_memory,
            "update_metadata",
            lambda **kw: pytest.fail("update_metadata called in dry-run"),
        )
        rc = dedup.main(["--chat-id", "12345"])
        assert rc == dedup.EXIT_OK

    def test_all_users_iterates_each_authorised_user(self, monkeypatch):
        # --all-users walks config.allowed_user_ids and runs the
        # per-user pipeline for each. Each user's load_rows is invoked
        # exactly once.
        from kai import config as kai_config
        from kai import memory as kai_memory

        fake_config = MagicMock()
        fake_config.memory_enabled = True
        fake_config.allowed_user_ids = {111, 222, 333}
        monkeypatch.setattr(kai_config, "load_config", lambda: fake_config)
        monkeypatch.setattr(kai_memory, "init_memory", lambda c: None)
        seen_user_ids: list[int] = []

        def fake_load(user_id):
            seen_user_ids.append(user_id)
            return []

        monkeypatch.setattr(dedup, "load_rows", fake_load)
        rc = dedup.main(["--all-users"])
        assert rc == dedup.EXIT_OK
        assert sorted(seen_user_ids) == [111, 222, 333]

    def test_apply_no_prompt_invokes_update_metadata(self, monkeypatch, tmp_path):
        # End-to-end CLI verification of the apply path: --apply with
        # --no-prompt accepts every proposed cluster and runs the
        # rewrite pipeline. Without this test, a regression that wires
        # --apply to the wrong branch of _run_for_user (e.g. an early
        # return that skips apply_rewrites) would not trip any other
        # test - the unit tests for apply_rewrites itself would still
        # pass since they call the helper directly.
        from kai import config as kai_config
        from kai import memory as kai_memory

        fake_config = MagicMock()
        fake_config.memory_enabled = True
        fake_config.allowed_user_ids = {12345}
        monkeypatch.setattr(kai_config, "load_config", lambda: fake_config)
        monkeypatch.setattr(kai_memory, "init_memory", lambda c: None)
        # Three rows. The two extra "preference" rows make "preference"
        # the most-frequent canonical (3 occurrences vs 1 for
        # "Preference"); without that imbalance, lexicographic tiebreak
        # picks "Preference" and the canonical-matching row would not
        # be rewritten, leaving only one update_metadata call.
        monkeypatch.setattr(
            dedup,
            "load_rows",
            lambda user_id: [
                _row("a", "first", ["Preference"]),
                _row("b", "second", ["preference"]),
                _row("c", "third", ["preference"]),
                _row("d", "fourth", ["preference"]),
            ],
        )
        update_calls: list[dict[str, Any]] = []

        def fake_update(*, user_id, memory_id, data, metadata):
            update_calls.append({"memory_id": memory_id, "tags": metadata["tags"]})
            return True

        monkeypatch.setattr(kai_memory, "update_metadata", fake_update)

        # Audit log to a temp path; the real default would write to
        # scripts/.tag-dedup-audit-* which the test should not touch.
        audit_log = tmp_path / "audit.jsonl"
        rc = dedup.main(
            [
                "--chat-id",
                "12345",
                "--apply",
                "--no-prompt",
                "--audit-log",
                str(audit_log),
            ]
        )
        assert rc == dedup.EXIT_OK
        # Only row "a" (tagged "Preference") has a non-canonical tag,
        # so only it gets rewritten. The three "preference" rows match
        # the canonical and skip the update_metadata call. The point
        # of this test is end-to-end CLI -> update_metadata wiring;
        # one call is sufficient to verify the pipeline.
        assert len(update_calls) == 1
        assert update_calls[0]["memory_id"] == "a"
        assert update_calls[0]["tags"] == ["preference"]
        # Audit log should exist with one entry (success path).
        assert audit_log.exists()
        assert len(audit_log.read_text().splitlines()) == 1
