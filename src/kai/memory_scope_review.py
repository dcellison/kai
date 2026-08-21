"""Complete operator review and fail-closed accounting for legacy scope.

The earlier classifier pass deliberately left low-confidence rows untouched.
This module provides the deterministic second pass: export every remaining
legacy-default row, require an explicit operator disposition for every row,
apply only the reviewed manifest, dump pre-images first, and retain unresolved
rows as fail-closed quarantine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai import memory
from kai.config import DATA_DIR, Config, MemoryProjectConfig
from kai.memory import MemoryResult
from kai.memory_projects import load_project_registry

log = logging.getLogger(__name__)

_MANIFEST_KIND = "kai.memory_scope_review"
_PREIMAGE_KIND = "kai.memory_scope_review_preimages"
_FORMAT_VERSION = 1
_STATUS_VERSION = 1
STATUS_FILENAME = "memory-scope-review.json"
SCOPE_REVIEW_ID_KEY = "scope_review_id"

DISPOSITION_REVIEW_REQUIRED = "review_required"
DISPOSITION_GLOBAL = "global"
DISPOSITION_PROJECT = "project"
DISPOSITION_DELETE = "delete"
DISPOSITION_QUARANTINE = "quarantine"
_DISPOSITIONS = frozenset(
    {
        DISPOSITION_REVIEW_REQUIRED,
        DISPOSITION_GLOBAL,
        DISPOSITION_PROJECT,
        DISPOSITION_DELETE,
        DISPOSITION_QUARANTINE,
    }
)


@dataclass(frozen=True, slots=True)
class ReviewRow:
    memory_id: str
    text_sha256: str
    text: str
    source: str
    created_at: str
    disposition: str
    project_id: str | None
    operator_note: str


@dataclass(frozen=True, slots=True)
class PreImage:
    memory_id: str
    text: str
    metadata: dict[str, Any]
    disposition: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_legacy_rows(rows: list[MemoryResult]) -> list[MemoryResult]:
    """Select every user-visible row with absent scope metadata."""
    selected = [
        row
        for row in rows
        if (row.metadata or {}).get("source") in memory.USER_VISIBLE_SOURCES
        and memory.resolve_memory_scope(row.metadata).legacy_defaulted
    ]
    return sorted(selected, key=lambda row: row.id)


def _header(
    *, review_id: str, user_id: str, rows: list[MemoryResult], registry: dict[str, MemoryProjectConfig]
) -> dict:
    return {
        "kind": _MANIFEST_KIND,
        "version": _FORMAT_VERSION,
        "review_id": review_id,
        "user_id": user_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_count": len(rows),
        "legacy_default_count": len(select_legacy_rows(rows)),
        "registered_projects": [
            {
                "project_id": project_id,
                "display_name": cfg.display_name,
                "memory_enabled": cfg.memory_enabled,
            }
            for project_id, cfg in sorted(registry.items())
        ],
    }


def render_manifest(header: dict[str, Any], rows: list[ReviewRow]) -> str:
    lines = [json.dumps({"header": header}, separators=(",", ":"), ensure_ascii=False)]
    lines.extend(json.dumps({"row": asdict(row)}, separators=(",", ":"), ensure_ascii=False) for row in rows)
    return "\n".join(lines) + "\n"


def _validate_header(header: Any, *, user_id: str | None = None) -> dict[str, Any]:
    if not isinstance(header, dict):
        raise ValueError("manifest header must be an object")
    if header.get("kind") != _MANIFEST_KIND or header.get("version") != _FORMAT_VERSION:
        raise ValueError("not a supported memory scope review manifest")
    for key in ("review_id", "user_id"):
        if not isinstance(header.get(key), str) or not header[key]:
            raise ValueError(f"manifest header {key} must be a non-empty string")
    if user_id is not None and header["user_id"] != user_id:
        raise ValueError(f"manifest belongs to user {header['user_id']}, not {user_id}")
    return header


def _parse_review_row(value: Any) -> ReviewRow:
    if not isinstance(value, dict):
        raise ValueError("manifest row must be an object")
    required_strings = ("memory_id", "text_sha256", "text", "source", "created_at", "disposition", "operator_note")
    for key in required_strings:
        if not isinstance(value.get(key), str):
            raise ValueError(f"manifest row {key} must be a string")
    if not value["memory_id"] or not value["text_sha256"] or not value["text"]:
        raise ValueError("manifest row memory_id, text_sha256, and text must be non-empty")
    disposition = value["disposition"]
    if disposition not in _DISPOSITIONS:
        raise ValueError(f"unsupported disposition {disposition!r}")
    project_id = value.get("project_id")
    if project_id is not None and (not isinstance(project_id, str) or not project_id):
        raise ValueError("manifest row project_id must be null or a non-empty string")
    if disposition == DISPOSITION_PROJECT and project_id is None:
        raise ValueError("project disposition requires project_id")
    if disposition != DISPOSITION_PROJECT and project_id is not None:
        raise ValueError(f"{disposition} disposition requires project_id=null")
    if _sha256(value["text"]) != value["text_sha256"]:
        raise ValueError(f"manifest row {value['memory_id']} text hash does not match")
    return ReviewRow(**{key: value.get(key) for key in ReviewRow.__dataclass_fields__})


def parse_manifest(text: str, *, user_id: str | None = None) -> tuple[dict[str, Any], list[ReviewRow]]:
    try:
        objects = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL: {exc}") from exc
    if not objects or set(objects[0]) != {"header"}:
        raise ValueError("manifest must begin with one header record")
    header = _validate_header(objects[0]["header"], user_id=user_id)
    rows: list[ReviewRow] = []
    seen: set[str] = set()
    for record in objects[1:]:
        if not isinstance(record, dict) or set(record) != {"row"}:
            raise ValueError("manifest may contain only row records after its header")
        row = _parse_review_row(record["row"])
        if row.memory_id in seen:
            raise ValueError(f"duplicate memory_id {row.memory_id}")
        seen.add(row.memory_id)
        rows.append(row)
    return header, rows


def render_preimages(header: dict[str, Any], rows: list[PreImage]) -> str:
    pre_header = dict(header)
    pre_header["kind"] = _PREIMAGE_KIND
    lines = [json.dumps({"header": pre_header}, separators=(",", ":"), ensure_ascii=False)]
    lines.extend(json.dumps({"row": asdict(row)}, separators=(",", ":"), ensure_ascii=False) for row in rows)
    return "\n".join(lines) + "\n"


def parse_preimages(text: str, *, user_id: str | None = None) -> tuple[dict[str, Any], list[PreImage]]:
    try:
        objects = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL: {exc}") from exc
    if not objects or set(objects[0]) != {"header"}:
        raise ValueError("pre-image file must begin with one header record")
    header = objects[0]["header"]
    if not isinstance(header, dict) or header.get("kind") != _PREIMAGE_KIND or header.get("version") != _FORMAT_VERSION:
        raise ValueError("not a supported memory scope review pre-image file")
    if user_id is not None and header.get("user_id") != user_id:
        raise ValueError(f"pre-image file belongs to user {header.get('user_id')}, not {user_id}")
    rows: list[PreImage] = []
    seen: set[str] = set()
    for record in objects[1:]:
        value = record.get("row") if isinstance(record, dict) and set(record) == {"row"} else None
        if not isinstance(value, dict):
            raise ValueError("invalid pre-image row")
        if not isinstance(value.get("memory_id"), str) or not value["memory_id"]:
            raise ValueError("pre-image memory_id must be non-empty")
        if not isinstance(value.get("text"), str) or not value["text"]:
            raise ValueError("pre-image text must be non-empty")
        if not isinstance(value.get("metadata"), dict) or value.get("disposition") not in _DISPOSITIONS:
            raise ValueError("invalid pre-image metadata or disposition")
        row = PreImage(**{key: value.get(key) for key in PreImage.__dataclass_fields__})
        if row.memory_id in seen:
            raise ValueError(f"duplicate memory_id {row.memory_id}")
        seen.add(row.memory_id)
        rows.append(row)
    return header, rows


def _secure_write(path: Path, content: str, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"refusing symlinked artifact path: {path}")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _status_path(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / STATUS_FILENAME


def _read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": _STATUS_VERSION, "reviews": {}}
    if path.is_symlink():
        raise OSError(f"refusing symlinked scope-review status: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != _STATUS_VERSION:
        raise ValueError("unsupported scope-review status document")
    if not isinstance(value.get("reviews", {}), dict):
        raise ValueError("scope-review status reviews must be an object")
    return value


def _atomic_status_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise OSError(f"refusing symlinked scope-review status: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def refresh_scope_review_status(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Refresh content-free legacy/quarantine counts from the live store."""
    path = _status_path(data_dir)
    document = _read_status(path)
    reviews = document.setdefault("reviews", {})
    by_owner: dict[str, dict[str, set[str]]] = {}
    for point_id, owner, metadata in memory.list_scope_census_points():
        resolved = memory.resolve_memory_scope(metadata)
        owner_sets = by_owner.setdefault(owner, {"legacy": set(), "invalid": set()})
        if resolved.legacy_defaulted:
            owner_sets["legacy"].add(point_id)
        elif resolved.invalid_defaulted:
            owner_sets["invalid"].add(point_id)

    owners = set(by_owner) | set(reviews)
    principals: dict[str, Any] = {}
    totals = {"raw_legacy_default": 0, "quarantined": 0, "unreviewed": 0, "invalid_quarantined": 0}
    for owner in sorted(owners):
        current = by_owner.get(owner, {"legacy": set(), "invalid": set()})
        review = reviews.get(owner, {}) if isinstance(reviews.get(owner, {}), dict) else {}
        reviewed_quarantine = set(review.get("quarantined_ids", []))
        quarantined = current["legacy"] & reviewed_quarantine
        unreviewed = current["legacy"] - reviewed_quarantine
        values = {
            "raw_legacy_default": len(current["legacy"]),
            "quarantined": len(quarantined),
            "unreviewed": len(unreviewed),
            "invalid_quarantined": len(current["invalid"]),
        }
        principals[owner] = values
        for key, value in values.items():
            totals[key] += value
    document["observed"] = {
        "observed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **totals,
        "principals": principals,
    }
    _atomic_status_write(path, document)
    return document


def _record_review(*, user_id: str, review_id: str, rows: list[ReviewRow], data_dir: Path = DATA_DIR) -> None:
    path = _status_path(data_dir)
    document = _read_status(path)
    reviews = document.setdefault("reviews", {})
    reviews[user_id] = {
        "review_id": review_id,
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "quarantined_ids": sorted(row.memory_id for row in rows if row.disposition == DISPOSITION_QUARANTINE),
        "global": sum(row.disposition == DISPOSITION_GLOBAL for row in rows),
        "project": sum(row.disposition == DISPOSITION_PROJECT for row in rows),
        "deleted": sum(row.disposition == DISPOSITION_DELETE for row in rows),
        "quarantined": sum(row.disposition == DISPOSITION_QUARANTINE for row in rows),
    }
    _atomic_status_write(path, document)


def _emit_scope_change(
    *,
    memory_id: str,
    user_id: str,
    from_metadata: dict[str, Any],
    to_metadata: dict[str, Any],
    review_id: str,
    rollback: bool = False,
) -> None:
    """Emit a content-free audit event for an operator-reviewed change."""
    before = memory.resolve_memory_scope(from_metadata)
    after = memory.resolve_memory_scope(to_metadata)
    payload: dict[str, Any] = {
        "memory_id": memory_id,
        "chat_id": user_id,
        "from_scope": before.scope,
        "from_project_id": before.project_id,
        "from_scope_source": before.scope_source,
        "to_scope": after.scope,
        "to_project_id": after.project_id,
        "run_id": review_id,
        "operator_reviewed": True,
    }
    if rollback:
        payload["rollback"] = True
    log.info("%s %s", memory.SCOPE_CHANGE_EVENT, json.dumps(payload, separators=(",", ":")))


async def run_census(config: Config, user_id: str, *, out_dir: Path) -> int:
    registry = await load_project_registry(config)
    rows = memory.get_all_for_admin(user_id=user_id)
    selected = select_legacy_rows(rows)
    review_id = datetime.now(UTC).strftime("lsr-%Y%m%d-%H%M%S")
    header = _header(review_id=review_id, user_id=user_id, rows=rows, registry=registry)
    review_rows = [
        ReviewRow(
            memory_id=row.id,
            text_sha256=_sha256(row.text),
            text=row.text,
            source=str(row.metadata.get("source", "")),
            created_at=row.created_at,
            disposition=DISPOSITION_REVIEW_REQUIRED,
            project_id=None,
            operator_note="",
        )
        for row in selected
    ]
    out_path = out_dir / f"legacy-scope-{review_id}-review.jsonl"
    _secure_write(out_path, render_manifest(header, review_rows), exclusive=True)
    refresh_scope_review_status()
    print(f"memory admin: legacy-scope census {review_id}: scanned {len(rows)}, legacy-default {len(selected)}.")
    print(f"memory admin: review manifest: {out_path}")
    print("memory admin: replace every review_required disposition before --apply.")
    return 0


def _validate_complete_review(
    manifest_rows: list[ReviewRow], current_rows: list[MemoryResult], registry: dict[str, MemoryProjectConfig]
) -> str | None:
    pending = [row.memory_id for row in manifest_rows if row.disposition == DISPOSITION_REVIEW_REQUIRED]
    if pending:
        return f"{len(pending)} row(s) still have disposition review_required"
    current = {row.id: row for row in select_legacy_rows(current_rows)}
    manifest = {row.memory_id: row for row in manifest_rows}
    missing = sorted(set(current) - set(manifest))
    extra = sorted(set(manifest) - set(current))
    if missing or extra:
        return f"manifest is not a fresh complete census (missing={missing}, extra={extra})"
    for memory_id, review in manifest.items():
        row = current[memory_id]
        if _sha256(row.text) != review.text_sha256 or row.text != review.text:
            return f"row {memory_id} text drifted since census"
        if str(row.metadata.get("source", "")) != review.source:
            return f"row {memory_id} source drifted since census"
        if review.disposition == DISPOSITION_PROJECT:
            project = registry.get(review.project_id or "")
            if project is None:
                return f"row {memory_id} targets an unregistered project {review.project_id!r}"
            if not project.memory_enabled:
                return f"row {memory_id} targets memory-disabled project {review.project_id!r}"
    return None


async def run_apply(config: Config, user_id: str, *, manifest_path: Path, out_dir: Path) -> int:
    try:
        header, review_rows = parse_manifest(manifest_path.read_text(encoding="utf-8"), user_id=user_id)
    except (OSError, ValueError) as exc:
        print(f"memory admin: cannot read review manifest: {exc}")
        return 1
    registry = await load_project_registry(config)
    current_rows = memory.get_all_for_admin(user_id=user_id)
    error = _validate_complete_review(review_rows, current_rows, registry)
    if error is not None:
        print(f"memory admin: {error}")
        return 1
    current = {row.id: row for row in select_legacy_rows(current_rows)}
    review_id = str(header["review_id"])
    preimages = [
        PreImage(row.memory_id, current[row.memory_id].text, dict(current[row.memory_id].metadata), row.disposition)
        for row in review_rows
    ]
    preimage_path = out_dir / f"legacy-scope-{review_id}-preimages.jsonl"
    _secure_write(preimage_path, render_preimages(header, preimages), exclusive=True)

    applied = deleted = quarantined = failed = 0
    for review in review_rows:
        row = current[review.memory_id]
        if review.disposition == DISPOSITION_QUARANTINE:
            quarantined += 1
            continue
        if review.disposition == DISPOSITION_DELETE:
            if memory.delete_by_id(user_id=user_id, memory_id=row.id):
                deleted += 1
            else:
                failed += 1
            continue
        merged = dict(row.metadata)
        merged.update(
            memory.build_scope_metadata(
                scope=review.disposition,
                project_id=review.project_id,
                scope_confidence=1.0,
                scope_source=memory.SCOPE_SOURCE_OPERATOR,
            )
        )
        merged[SCOPE_REVIEW_ID_KEY] = review_id
        if memory.update_metadata(user_id=user_id, memory_id=row.id, data=row.text, metadata=merged):
            applied += 1
            _emit_scope_change(
                memory_id=row.id,
                user_id=user_id,
                from_metadata=row.metadata,
                to_metadata=merged,
                review_id=review_id,
            )
        else:
            failed += 1

    if failed:
        print(
            f"memory admin: review {review_id} incomplete: applied={applied}, deleted={deleted}, "
            f"quarantined={quarantined}, failed={failed}; pre-images: {preimage_path}"
        )
        return 1
    remaining = select_legacy_rows(memory.get_all_for_admin(user_id=user_id))
    expected_quarantine = {row.memory_id for row in review_rows if row.disposition == DISPOSITION_QUARANTINE}
    if {row.id for row in remaining} != expected_quarantine:
        print("memory admin: post-apply census does not match the reviewed quarantine set")
        return 1
    _record_review(user_id=user_id, review_id=review_id, rows=review_rows)
    refresh_scope_review_status()
    print(
        f"memory admin: review {review_id} complete: applied={applied}, deleted={deleted}, "
        f"quarantined={quarantined}, unreviewed=0."
    )
    print(f"memory admin: pre-images: {preimage_path}")
    return 0


async def run_rollback(config: Config, user_id: str, *, preimages_path: Path) -> int:
    _ = config
    try:
        header, preimages = parse_preimages(preimages_path.read_text(encoding="utf-8"), user_id=user_id)
    except (OSError, ValueError) as exc:
        print(f"memory admin: cannot read pre-images: {exc}")
        return 1
    review_id = str(header.get("review_id", ""))
    restored = skipped = failed = 0
    for preimage in preimages:
        current = memory.get_by_id(user_id=user_id, memory_id=preimage.memory_id)
        if current is None:
            skipped += 1  # Deleted rows remain recoverable through Mem0 history.
            continue
        if preimage.disposition == DISPOSITION_QUARANTINE:
            skipped += 1
            continue
        if current.metadata.get(SCOPE_REVIEW_ID_KEY) != review_id:
            skipped += 1  # A later operator correction is authoritative.
            continue
        if memory.update_metadata(
            user_id=user_id,
            memory_id=preimage.memory_id,
            data=preimage.text,
            metadata=preimage.metadata,
        ):
            restored += 1
            _emit_scope_change(
                memory_id=preimage.memory_id,
                user_id=user_id,
                from_metadata=current.metadata,
                to_metadata=preimage.metadata,
                review_id=review_id,
                rollback=True,
            )
        else:
            failed += 1
    refresh_scope_review_status()
    print(
        f"memory admin: rollback {review_id}: restored={restored}, skipped={skipped}, failed={failed}; "
        "deleted rows remain recorded in Mem0 history."
    )
    return 1 if failed else 0
