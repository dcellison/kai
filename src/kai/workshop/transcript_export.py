"""Derived, recoverable exports of canonical Workshop transcripts."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kai.named_access import replace_named_read_access
from kai.workshop.domain import ChannelId
from kai.workshop.store import WorkshopEventStore

_EXPORT_FORMAT = "kai-workshop-transcript"
_EXPORT_VERSION = 1
_SNAPSHOT_NAME = "canonical-transcript.ndjson"


class CanonicalTranscriptExportError(RuntimeError):
    """A canonical channel transcript cannot be exported safely."""


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptExport:
    """One deterministic slice of a canonical channel transcript."""

    channel_id: ChannelId
    records: tuple[dict[str, object], ...]
    after_event_position: int
    through_event_position: int

    def ndjson(self) -> str:
        """Render newline-delimited canonical records."""
        return "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for record in self.records
        )


async def build_canonical_transcript_export(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    *,
    after_event_position: int = 0,
) -> CanonicalTranscriptExport:
    """Build a stable canonical export slice after one event position."""
    if not isinstance(channel_id, ChannelId):
        raise ValueError("channel_id must be a ChannelId")
    if not isinstance(after_event_position, int) or isinstance(after_event_position, bool) or after_event_position < 0:
        raise ValueError("after_event_position must be a non-negative integer")
    async with store.connection.execute(
        "SELECT 1 FROM channels WHERE id = ?",
        (channel_id,),
    ) as cursor:
        if await cursor.fetchone() is None:
            raise CanonicalTranscriptExportError("Canonical channel does not exist")
    async with store.connection.execute(
        "SELECT COALESCE(MAX(created_event_position), 0) FROM messages WHERE channel_id = ?",
        (channel_id,),
    ) as cursor:
        row = await cursor.fetchone()
    through = int(row[0]) if row is not None else 0
    async with store.connection.execute(
        "SELECT m.id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.body, m.created_event_position, m.created_at "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND m.created_event_position > ? "
        "ORDER BY m.created_event_position, m.id",
        (channel_id, after_event_position),
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
    return CanonicalTranscriptExport(channel_id, records, after_event_position, through)


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptSnapshotState:
    """Validated state of one derived transcript file."""

    target: Path
    exists: bool
    valid: bool
    through_event_position: int


class CanonicalTranscriptProjection:
    """Maintain recoverable per-channel transcript projections.

    SQLite remains authoritative.  Each channel is serialized in-process, the
    database is held only while selecting a consistent export slice, and the
    comparatively slow filesystem write and fsync happen after the database
    lock is released.
    """

    def __init__(self, export_root: Path) -> None:
        self._export_root = export_root
        self._locks: dict[ChannelId, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def refresh(
        self,
        store: WorkshopEventStore,
        channel_id: ChannelId,
        *,
        reader_user: str | None,
        database_lock: asyncio.Lock,
    ) -> Path:
        """Incrementally refresh one transcript, rebuilding after drift."""
        lock = await self._channel_lock(channel_id)
        async with lock:
            state = await asyncio.to_thread(
                inspect_canonical_transcript_snapshot,
                self._export_root,
                channel_id,
            )
            after = state.through_event_position if state.valid else 0
            async with database_lock:
                export = await build_canonical_transcript_export(
                    store,
                    channel_id,
                    after_event_position=after,
                )
                if after > export.through_event_position:
                    export = await build_canonical_transcript_export(store, channel_id)
                    state = CanonicalTranscriptSnapshotState(state.target, state.exists, False, 0)
            return await asyncio.to_thread(
                write_canonical_transcript_snapshot,
                export,
                self._export_root,
                reader_user=reader_user,
                replace=not state.exists or not state.valid,
            )

    async def _channel_lock(self, channel_id: ChannelId) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(channel_id, asyncio.Lock())


def inspect_canonical_transcript_snapshot(
    export_root: Path,
    channel_id: ChannelId,
) -> CanonicalTranscriptSnapshotState:
    """Inspect only the last record needed for incremental recovery."""
    target = export_root / str(channel_id) / _SNAPSHOT_NAME
    if export_root.is_symlink() or target.parent.is_symlink() or target.is_symlink():
        raise CanonicalTranscriptExportError("Refusing a symlinked canonical transcript path")
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return CanonicalTranscriptSnapshotState(target, False, True, 0)
    if not stat.S_ISREG(metadata.st_mode):
        raise CanonicalTranscriptExportError("Canonical transcript target is not a regular file")
    try:
        last_line = _last_complete_line(target)
        if last_line is None:
            return CanonicalTranscriptSnapshotState(target, True, True, 0)
        record = json.loads(last_line)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return CanonicalTranscriptSnapshotState(target, True, False, 0)
    if (
        not isinstance(record, dict)
        or record.get("format") != _EXPORT_FORMAT
        or record.get("format_version") != _EXPORT_VERSION
        or record.get("channel_id") != str(channel_id)
        or not isinstance(record.get("event_position"), int)
        or isinstance(record.get("event_position"), bool)
        or int(record["event_position"]) <= 0
    ):
        return CanonicalTranscriptSnapshotState(target, True, False, 0)
    return CanonicalTranscriptSnapshotState(target, True, True, int(record["event_position"]))


def _last_complete_line(path: Path) -> str | None:
    """Read the final non-empty line without loading the whole transcript."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        if position == 0:
            return None
        chunks: list[bytes] = []
        while position > 0:
            size = min(8192, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            joined = b"".join(reversed(chunks)).rstrip(b"\r\n")
            split = joined.rfind(b"\n")
            if split >= 0 or position == 0:
                line = joined[split + 1 :] if split >= 0 else joined
                return line.decode("utf-8") if line else None
    return None


def write_canonical_transcript_snapshot(
    export: CanonicalTranscriptExport,
    export_root: Path,
    *,
    reader_user: str | None,
    replace: bool,
) -> Path:
    """Atomically replace or incrementally extend a derived transcript."""
    channel_directory = export_root / str(export.channel_id)
    target = channel_directory / _SNAPSHOT_NAME
    if export_root.is_symlink():
        raise CanonicalTranscriptExportError("Refusing symlinked export directory")
    if export_root.exists() and not export_root.is_dir():
        raise CanonicalTranscriptExportError("Export root is not a directory")
    export_root.mkdir(parents=True, exist_ok=True, mode=0o711)
    if channel_directory.is_symlink():
        raise CanonicalTranscriptExportError("Refusing symlinked channel export directory")
    if channel_directory.exists() and not channel_directory.is_dir():
        raise CanonicalTranscriptExportError("Channel export path is not a directory")
    channel_directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(channel_directory, 0o700)
    replace_named_read_access(channel_directory, reader_user, directory=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise CanonicalTranscriptExportError("Canonical transcript export target is unsafe")

    payload = export.ndjson().encode("utf-8")
    if replace or not target.exists():
        _replace_snapshot(target, payload)
    elif payload:
        _append_snapshot(target, payload)
    os.chmod(target, 0o600)
    replace_named_read_access(target, reader_user, directory=False)
    return target


def _replace_snapshot(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=".canonical-transcript-",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _append_snapshot(target: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
