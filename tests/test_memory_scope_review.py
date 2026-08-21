"""Residual legacy-scope review, apply, rollback, and status tests."""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai import memory
from kai import memory_scope_review as review
from kai.config import MemoryProjectConfig
from kai.memory import MemoryResult


def _row(memory_id: str, *, text: str | None = None, scope: str | None = None) -> MemoryResult:
    metadata = {"source": "extracted", "type": "fact", "other": "preserved"}
    if scope is not None:
        metadata.update(memory.build_scope_metadata(scope=scope, scope_source=memory.SCOPE_SOURCE_OPERATOR))
    return MemoryResult(
        id=memory_id,
        text=text or f"text {memory_id}",
        score=0.0,
        memory_type="fact",
        metadata=metadata,
        created_at="2026-08-21T00:00:00Z",
    )


def _project(project_id: str, *, enabled: bool = True) -> MemoryProjectConfig:
    return MemoryProjectConfig(
        project_id=project_id,
        display_name=project_id,
        workspace_roots=(Path("/work") / project_id,),
        memory_enabled=enabled,
        default_scope_for_new_facts="project",
    )


def _manifest_rows(*rows: MemoryResult) -> list[review.ReviewRow]:
    return [
        review.ReviewRow(
            memory_id=row.id,
            text_sha256=review._sha256(row.text),
            text=row.text,
            source=str(row.metadata["source"]),
            created_at=row.created_at,
            disposition=review.DISPOSITION_REVIEW_REQUIRED,
            project_id=None,
            operator_note="",
        )
        for row in rows
    ]


def _header(user_id: str = "prn_1", review_id: str = "lsr-1") -> dict:
    return {
        "kind": "kai.memory_scope_review",
        "version": 1,
        "review_id": review_id,
        "user_id": user_id,
    }


def test_select_legacy_rows_is_complete_and_user_visible_only():
    legacy = _row("legacy")
    scoped = _row("scoped", scope="global")
    hidden = _row("hidden")
    hidden.metadata["source"] = "user_raw"
    assert review.select_legacy_rows([scoped, hidden, legacy]) == [legacy]


def test_manifest_round_trip_and_hash_tamper_rejection():
    row = _row("m1")
    text = review.render_manifest(_header(), _manifest_rows(row))
    header, parsed = review.parse_manifest(text, user_id="prn_1")
    assert header["review_id"] == "lsr-1"
    assert parsed[0].memory_id == "m1"
    with pytest.raises(ValueError, match="hash"):
        review.parse_manifest(text.replace("text m1", "changed"), user_id="prn_1")


def test_complete_review_rejects_pending_missing_and_disabled_project():
    first, second = _row("a"), _row("b")
    rows = _manifest_rows(first, second)
    assert "review_required" in (review._validate_complete_review(rows, [first, second], {}) or "")

    only_first = [replace(rows[0], disposition=review.DISPOSITION_GLOBAL)]
    assert "missing=['b']" in (review._validate_complete_review(only_first, [first, second], {}) or "")

    project_row = replace(rows[0], disposition=review.DISPOSITION_PROJECT, project_id="kai")
    assert "memory-disabled" in (
        review._validate_complete_review([project_row], [first], {"kai": _project("kai", enabled=False)}) or ""
    )


@pytest.mark.asyncio
async def test_census_writes_complete_private_manifest(tmp_path, monkeypatch):
    rows = [_row("b"), _row("a"), _row("scoped", scope="global")]
    monkeypatch.setattr(review.memory, "get_all_for_admin", lambda *, user_id: rows)
    monkeypatch.setattr(review, "load_project_registry", AsyncMock(return_value={"kai": _project("kai")}))
    monkeypatch.setattr(review, "refresh_scope_review_status", lambda: {})

    assert await review.run_census(MagicMock(), "prn_1", out_dir=tmp_path) == 0
    manifests = list(tmp_path.glob("*-review.jsonl"))
    assert len(manifests) == 1
    header, manifest_rows = review.parse_manifest(manifests[0].read_text(), user_id="prn_1")
    assert header["corpus_count"] == 3
    assert header["legacy_default_count"] == 2
    assert [row.memory_id for row in manifest_rows] == ["a", "b"]
    assert all(row.disposition == review.DISPOSITION_REVIEW_REQUIRED for row in manifest_rows)
    assert stat.S_IMODE(manifests[0].stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_apply_requires_fresh_complete_manifest_and_preserves_metadata(tmp_path, monkeypatch, caplog):
    rows = [_row("global"), _row("project"), _row("delete"), _row("quarantine")]
    manifest_rows = _manifest_rows(*rows)
    manifest_rows = [
        replace(
            row,
            disposition={
                "global": review.DISPOSITION_GLOBAL,
                "project": review.DISPOSITION_PROJECT,
                "delete": review.DISPOSITION_DELETE,
                "quarantine": review.DISPOSITION_QUARANTINE,
            }[row.memory_id],
            project_id="kai" if row.memory_id == "project" else None,
        )
        for row in manifest_rows
    ]
    manifest = tmp_path / "review.jsonl"
    manifest.write_text(review.render_manifest(_header(), manifest_rows))
    store = {row.id: row for row in rows}
    updates: list[tuple[str, dict]] = []
    deleted: list[str] = []

    monkeypatch.setattr(review, "load_project_registry", AsyncMock(return_value={"kai": _project("kai")}))
    monkeypatch.setattr(review.memory, "get_all_for_admin", lambda *, user_id: list(store.values()))

    def update(*, user_id, memory_id, data, metadata):
        updates.append((memory_id, metadata))
        source = store[memory_id]
        store[memory_id] = MemoryResult(
            id=source.id,
            text=data,
            score=0,
            memory_type=source.memory_type,
            metadata=metadata,
            created_at=source.created_at,
        )
        return True

    monkeypatch.setattr(review.memory, "update_metadata", update)
    monkeypatch.setattr(
        review.memory,
        "delete_by_id",
        lambda *, user_id, memory_id: (deleted.append(memory_id), store.pop(memory_id), True)[2],
    )
    monkeypatch.setattr(review, "_record_review", lambda **kwargs: None)
    monkeypatch.setattr(review, "refresh_scope_review_status", lambda: {})

    with caplog.at_level("INFO", logger="kai.memory_scope_review"):
        assert await review.run_apply(MagicMock(), "prn_1", manifest_path=manifest, out_dir=tmp_path) == 0
    assert {memory_id for memory_id, _ in updates} == {"global", "project"}
    assert deleted == ["delete"]
    assert store["quarantine"].metadata.get("scope") is None
    assert all(metadata["other"] == "preserved" for _, metadata in updates)
    assert store["global"].metadata["scope"] == "global"
    assert store["project"].metadata["project_id"] == "kai"
    audit_events = [record for record in caplog.records if record.message.startswith(memory.SCOPE_CHANGE_EVENT)]
    assert len(audit_events) == 2
    assert all('"operator_reviewed":true' in record.message for record in audit_events)
    preimages = list(tmp_path.glob("*-preimages.jsonl"))
    assert len(preimages) == 1
    assert stat.S_IMODE(preimages[0].stat().st_mode) == 0o600


def test_refresh_status_distinguishes_reviewed_quarantine_and_unreviewed(tmp_path, monkeypatch):
    status_path = tmp_path / review.STATUS_FILENAME
    status_path.write_text(
        json.dumps(
            {
                "version": 1,
                "reviews": {"prn_1": {"quarantined_ids": ["reviewed"]}},
            }
        )
    )
    monkeypatch.setattr(
        review.memory,
        "list_scope_census_points",
        lambda: [
            ("reviewed", "prn_1", {"source": "extracted"}),
            ("new", "prn_1", {"source": "episode"}),
            (
                "invalid",
                "prn_2",
                {"source": "explicit", "scope": "global", "scope_source": "bogus"},
            ),
        ],
    )
    document = review.refresh_scope_review_status(tmp_path)
    observed = document["observed"]
    assert observed["raw_legacy_default"] == 2
    assert observed["quarantined"] == 1
    assert observed["unreviewed"] == 1
    assert observed["invalid_quarantined"] == 1
    assert stat.S_IMODE(status_path.stat().st_mode) == 0o600
