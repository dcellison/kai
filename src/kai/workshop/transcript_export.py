"""Explicit, non-authoritative exports of canonical Workshop transcripts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kai.named_access import replace_named_read_access
from kai.workshop.domain import ChannelId
from kai.workshop.store import WorkshopEventStore

_EXPORT_FORMAT = "kai-workshop-transcript"
_EXPORT_VERSION = 1


class CanonicalTranscriptExportError(RuntimeError):
    """A canonical channel cannot be exported safely."""


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptExport:
    """One deterministic canonical transcript snapshot."""

    channel_id: ChannelId
    records: tuple[dict[str, object], ...]
    through_event_position: int

    def jsonl(self) -> str:
        """Render newline-delimited JSON with no transport identifiers."""
        return "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in self.records
        )


async def build_canonical_transcript_export(
    store: WorkshopEventStore,
    channel_id: ChannelId,
) -> CanonicalTranscriptExport:
    """Build a stable export from the canonical message projection."""
    if not isinstance(channel_id, ChannelId):
        raise ValueError("channel_id must be a ChannelId")
    async with store.connection.execute(
        "SELECT 1 FROM channels WHERE id = ?",
        (channel_id,),
    ) as cursor:
        if await cursor.fetchone() is None:
            raise CanonicalTranscriptExportError("Canonical channel does not exist")
    async with store.connection.execute(
        "SELECT m.id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.body, m.created_event_position, m.created_at "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? ORDER BY m.created_event_position, m.id",
        (channel_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    records = tuple(
        {
            "author": {
                "display_name": str(row[3]),
                "kind": str(row[2]),
                "principal_id": str(row[1]),
            },
            "body": str(row[5]),
            "channel_id": str(channel_id),
            "created_at": str(row[7]),
            "event_position": int(row[6]),
            "format": _EXPORT_FORMAT,
            "format_version": _EXPORT_VERSION,
            "message_id": str(row[0]),
            "reply_to_message_id": str(row[4]) if row[4] is not None else None,
        }
        for row in rows
    )
    through = int(rows[-1][6]) if rows else 0
    return CanonicalTranscriptExport(channel_id, records, through)


def write_canonical_transcript_snapshot(
    export: CanonicalTranscriptExport,
    export_root: Path,
    *,
    reader_user: str | None,
) -> Path:
    """Atomically materialize a derived transcript for backend search.

    The snapshot uses a distinct ``.ndjson`` filename that compatibility
    readers do not scan.  Keeping it in the channel history directory makes
    it inherit the installer's existing per-channel ownership and ACL policy.
    """
    root = export_root
    channel_directory = root / str(export.channel_id)
    target = channel_directory / "canonical-transcript.ndjson"
    if root.is_symlink():
        raise CanonicalTranscriptExportError(f"Refusing symlinked export directory: {root}")
    if root.exists() and not root.is_dir():
        raise CanonicalTranscriptExportError(f"Export directory is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o711)
    os.chmod(root, 0o711)
    if channel_directory.is_symlink():
        raise CanonicalTranscriptExportError("Refusing symlinked channel export directory")
    if channel_directory.exists() and not channel_directory.is_dir():
        raise CanonicalTranscriptExportError("Channel export path is not a directory")
    new_channel_directory = not channel_directory.exists()
    channel_directory.mkdir(mode=0o700, exist_ok=True)
    if new_channel_directory or reader_user is None:
        os.chmod(channel_directory, 0o700)
    replace_named_read_access(channel_directory, reader_user, directory=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise CanonicalTranscriptExportError("Canonical transcript export target is unsafe")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=channel_directory,
        prefix=".transcript-",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(export.jsonl())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        replace_named_read_access(target, reader_user, directory=False)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return target
