"""Additive SQLite schema migrations for the Workshop foundation."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

WORKSHOP_SCHEMA_VERSION = 7


@dataclass(frozen=True, slots=True)
class SchemaMigration:
    version: int
    name: str
    statements: tuple[str, ...]


_INITIAL_SCHEMA = SchemaMigration(
    version=1,
    name="canonical_conversation_foundation",
    statements=(
        """
        CREATE TABLE workshops (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE principals (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('human', 'agent', 'service', 'integration')),
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE external_identities (
            id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            external_subject TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (provider, external_subject)
        )
        """,
        """
        CREATE TABLE workshop_memberships (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (workshop_id, principal_id)
        )
        """,
        """
        CREATE TABLE channels (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            name TEXT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE channel_bindings (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            transport TEXT NOT NULL,
            external_channel_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (transport, external_channel_id)
        )
        """,
        """
        CREATE TABLE agents (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL UNIQUE REFERENCES principals(id) ON DELETE RESTRICT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE channel_agents (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            UNIQUE (channel_id, agent_id)
        )
        """,
        """
        CREATE TABLE event_log (
            position INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            envelope_version INTEGER NOT NULL CHECK (envelope_version > 0),
            event_type TEXT NOT NULL,
            event_version INTEGER NOT NULL CHECK (event_version > 0),
            workshop_id TEXT NOT NULL,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            actor_principal_id TEXT,
            occurred_at TEXT NOT NULL,
            idempotency_key TEXT UNIQUE,
            payload_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
        "CREATE INDEX event_log_workshop_position_idx ON event_log (workshop_id, position)",
        "CREATE INDEX event_log_aggregate_position_idx ON event_log (aggregate_type, aggregate_id, position)",
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            author_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            reply_to_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            body TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE projection_checkpoints (
            name TEXT PRIMARY KEY,
            version INTEGER NOT NULL CHECK (version > 0),
            last_position INTEGER NOT NULL CHECK (last_position >= 0),
            updated_at TEXT NOT NULL
        )
        """,
    ),
)

_DELIVERY_SCHEMA = SchemaMigration(
    version=2,
    name="canonical_delivery_observations",
    statements=(
        """
        CREATE TABLE deliveries (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            transport TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (message_id, transport, mode)
        )
        """,
    ),
)

_CHANNEL_MEMBERSHIP_SCHEMA = SchemaMigration(
    version=3,
    name="explicit_channel_memberships",
    statements=(
        """
        CREATE TABLE channel_memberships (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            role TEXT NOT NULL CHECK (role IN ('owner', 'participant')),
            created_at TEXT NOT NULL,
            UNIQUE (channel_id, principal_id)
        )
        """,
        "CREATE INDEX channel_memberships_principal_channel_idx ON channel_memberships (principal_id, channel_id)",
    ),
)

_CLIENT_SESSION_SCHEMA = SchemaMigration(
    version=4,
    name="revocable_human_client_sessions",
    statements=(
        """
        CREATE TABLE workshop_client_devices (
            id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            display_name TEXT NOT NULL CHECK (
                length(trim(display_name)) > 0 AND length(display_name) <= 200
            ),
            created_at TEXT NOT NULL,
            last_seen_at TEXT,
            revoked_at TEXT,
            UNIQUE (id, principal_id)
        )
        """,
        "CREATE INDEX workshop_client_devices_principal_idx ON workshop_client_devices (principal_id, revoked_at)",
        """
        CREATE TABLE workshop_client_sessions (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE CHECK (length(token_hash) = 64),
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_seen_at TEXT,
            revoked_at TEXT,
            FOREIGN KEY (device_id, principal_id)
                REFERENCES workshop_client_devices(id, principal_id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX workshop_client_sessions_principal_idx "
        "ON workshop_client_sessions (principal_id, revoked_at, expires_at)",
        "CREATE INDEX workshop_client_sessions_device_idx ON workshop_client_sessions (device_id, revoked_at)",
    ),
)

_CLIENT_ENROLLMENT_SCHEMA = SchemaMigration(
    version=5,
    name="single_use_human_client_enrollment",
    statements=(
        """
        CREATE TABLE workshop_client_enrollment_grants (
            id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE CHECK (length(token_hash) = 64),
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            redeemed_at TEXT,
            revoked_at TEXT,
            device_id TEXT,
            session_id TEXT REFERENCES workshop_client_sessions(id) ON DELETE RESTRICT,
            FOREIGN KEY (device_id, principal_id)
                REFERENCES workshop_client_devices(id, principal_id) ON DELETE RESTRICT,
            CHECK (
                (redeemed_at IS NULL AND device_id IS NULL AND session_id IS NULL)
                OR (redeemed_at IS NOT NULL AND device_id IS NOT NULL AND session_id IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX workshop_client_enrollment_principal_idx "
        "ON workshop_client_enrollment_grants (principal_id, redeemed_at, revoked_at, expires_at)",
    ),
)

_ARTIFACT_SCHEMA = SchemaMigration(
    version=6,
    name="canonical_artifact_metadata",
    statements=(
        """
        CREATE TABLE artifacts (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            created_by_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            kind TEXT NOT NULL CHECK (kind IN ('photo', 'document', 'voice')),
            media_type TEXT NOT NULL,
            byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
            content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
            original_filename TEXT CHECK (
                original_filename IS NULL OR (
                    length(trim(original_filename)) > 0 AND length(original_filename) <= 255
                )
            ),
            storage_path TEXT NOT NULL,
            source_transport TEXT NOT NULL,
            source_unique_id TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            UNIQUE (message_id, source_transport, source_unique_id)
        )
        """,
        "CREATE INDEX artifacts_channel_position_idx ON artifacts (channel_id, created_event_position)",
        "CREATE INDEX artifacts_message_idx ON artifacts (message_id)",
    ),
)

_DELIVERY_OUTBOX_SCHEMA = SchemaMigration(
    version=7,
    name="durable_delivery_outbox_foundation",
    statements=(
        """
        CREATE TABLE delivery_outbox (
            -- These canonical IDs are deliberately not foreign keys. Projection
            -- rebuilds temporarily delete and restore their rows, while durable
            -- delivery work and attempt history must survive that operation.
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_binding_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            transport TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'leased', 'retry_wait', 'succeeded', 'failed')
            ),
            max_attempts INTEGER NOT NULL CHECK (max_attempts BETWEEN 1 AND 20),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (
                attempt_count >= 0 AND attempt_count <= max_attempts
            ),
            available_at TEXT NOT NULL,
            lease_id TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            last_error_code TEXT,
            requested_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE (message_id, channel_binding_id, mode),
            CHECK (
                (
                    status = 'leased'
                    AND lease_id IS NOT NULL
                    AND lease_owner IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                ) OR (
                    status != 'leased'
                    AND lease_id IS NULL
                    AND lease_owner IS NULL
                    AND lease_expires_at IS NULL
                )
            ),
            CHECK (
                (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL)
                OR (status NOT IN ('succeeded', 'failed') AND completed_at IS NULL)
            )
        )
        """,
        "CREATE INDEX delivery_outbox_due_idx ON delivery_outbox (status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_lease_expiry_idx ON delivery_outbox (status, lease_expires_at)",
        """
        CREATE TABLE delivery_attempts (
            id TEXT PRIMARY KEY,
            delivery_id TEXT NOT NULL REFERENCES delivery_outbox(id) ON DELETE CASCADE,
            attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
            worker_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            completed_at TEXT,
            outcome TEXT CHECK (
                outcome IN ('succeeded', 'retry_scheduled', 'failed', 'lease_expired')
            ),
            error_code TEXT,
            UNIQUE (delivery_id, attempt_number),
            CHECK (
                (completed_at IS NULL AND outcome IS NULL)
                OR (completed_at IS NOT NULL AND outcome IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX delivery_attempts_delivery_idx ON delivery_attempts (delivery_id, attempt_number)",
    ),
)

_MIGRATIONS = (
    _INITIAL_SCHEMA,
    _DELIVERY_SCHEMA,
    _CHANNEL_MEMBERSHIP_SCHEMA,
    _CLIENT_SESSION_SCHEMA,
    _CLIENT_ENROLLMENT_SCHEMA,
    _ARTIFACT_SCHEMA,
    _DELIVERY_OUTBOX_SCHEMA,
)


async def migrate_workshop_schema(
    connection: aiosqlite.Connection,
    *,
    manage_transaction: bool = True,
) -> None:
    """Apply pending migrations, optionally inside the caller's transaction."""
    if manage_transaction:
        await connection.execute("PRAGMA foreign_keys=ON")
    try:
        if manage_transaction:
            await connection.execute("BEGIN IMMEDIATE")
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workshop_schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        async with connection.execute("SELECT version FROM workshop_schema_migrations") as cursor:
            applied = {int(row[0]) for row in await cursor.fetchall()}
        unknown = {version for version in applied if version > WORKSHOP_SCHEMA_VERSION}
        if unknown:
            raise RuntimeError(f"Workshop schema is newer than this Kai build: {sorted(unknown)}")

        for migration in _MIGRATIONS:
            if migration.version in applied:
                continue
            for statement in migration.statements:
                await connection.execute(statement)
            await connection.execute(
                "INSERT INTO workshop_schema_migrations (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
        if manage_transaction:
            await connection.commit()
    except Exception:
        if manage_transaction:
            await connection.rollback()
        raise
