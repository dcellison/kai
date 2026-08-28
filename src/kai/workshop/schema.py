"""Additive SQLite schema migrations for the Workshop foundation."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

WORKSHOP_SCHEMA_VERSION = 39


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

_DELIVERY_FRAGMENT_SCHEMA = SchemaMigration(
    version=8,
    name="durable_delivery_fragment_progress",
    statements=(
        """
        CREATE TABLE delivery_fragments (
            delivery_id TEXT NOT NULL REFERENCES delivery_outbox(id) ON DELETE CASCADE,
            fragment_index INTEGER NOT NULL CHECK (fragment_index >= 0),
            fragment_count INTEGER NOT NULL CHECK (fragment_count > 0),
            body TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 4096),
            status TEXT NOT NULL CHECK (status IN ('pending', 'sending', 'sent', 'uncertain')),
            attempt_id TEXT REFERENCES delivery_attempts(id) ON DELETE RESTRICT,
            external_message_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            sent_at TEXT,
            PRIMARY KEY (delivery_id, fragment_index),
            CHECK (fragment_index < fragment_count),
            CHECK (
                (
                    status = 'pending'
                    AND attempt_id IS NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status IN ('sending', 'uncertain')
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status = 'sent'
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NOT NULL
                    AND sent_at IS NOT NULL
                )
            )
        )
        """,
        "CREATE INDEX delivery_fragments_status_idx ON delivery_fragments (delivery_id, status, fragment_index)",
    ),
)

_BINDING_AWARE_DELIVERY_SCHEMA = SchemaMigration(
    version=9,
    name="binding_aware_delivery_outcomes",
    statements=(
        "ALTER TABLE deliveries RENAME TO legacy_deliveries_v8",
        """
        CREATE TABLE deliveries (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            channel_binding_id TEXT REFERENCES channel_bindings(id) ON DELETE CASCADE,
            transport TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT
        )
        """,
        """
        INSERT INTO deliveries (
            id, message_id, channel_id, channel_binding_id, transport, mode,
            status, created_at, updated_at, last_event_position
        )
        SELECT id, message_id, channel_id, NULL, transport, mode,
               status, created_at, updated_at, last_event_position
        FROM legacy_deliveries_v8
        """,
        "DROP TABLE legacy_deliveries_v8",
        "CREATE UNIQUE INDEX deliveries_legacy_identity_idx "
        "ON deliveries (message_id, transport, mode) WHERE channel_binding_id IS NULL",
        "CREATE UNIQUE INDEX deliveries_binding_identity_idx "
        "ON deliveries (message_id, channel_binding_id, mode) WHERE channel_binding_id IS NOT NULL",
    ),
)

_DELIVERY_BINDING_ORDER_SCHEMA = SchemaMigration(
    version=10,
    name="delivery_binding_order_index",
    statements=(
        "CREATE INDEX delivery_outbox_binding_order_idx "
        "ON delivery_outbox (channel_binding_id, requested_event_position, status)",
    ),
)

_DELIVERY_PURPOSE_SCHEMA = SchemaMigration(
    version=11,
    name="durable_delivery_purpose",
    statements=(
        "ALTER TABLE delivery_outbox ADD COLUMN purpose TEXT NOT NULL "
        "DEFAULT 'qualification' CHECK (purpose IN ('conversation_reply', 'qualification'))",
        "CREATE INDEX delivery_outbox_purpose_due_idx "
        "ON delivery_outbox (purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_purpose_binding_order_idx "
        "ON delivery_outbox (purpose, channel_binding_id, requested_event_position, status)",
    ),
)

_TELEGRAM_STREAMING_PREVIEW_SCHEMA = SchemaMigration(
    version=12,
    name="durable_telegram_streaming_preview_bindings",
    statements=(
        """
        CREATE TABLE telegram_streaming_previews (
            -- These canonical IDs deliberately are not foreign keys. A
            -- projection rebuild temporarily removes and restores their rows,
            -- while a confirmed external Telegram effect must survive replay.
            inbound_message_id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_binding_id TEXT NOT NULL,
            external_message_id INTEGER NOT NULL CHECK (external_message_id > 0),
            state TEXT NOT NULL DEFAULT 'confirmed_non_final'
                CHECK (state = 'confirmed_non_final'),
            confirmed_at TEXT NOT NULL,
            UNIQUE (channel_binding_id, external_message_id)
        )
        """,
        "CREATE INDEX telegram_streaming_previews_channel_idx "
        "ON telegram_streaming_previews (channel_id, confirmed_at)",
    ),
)

_STREAMING_FINALIZATION_SCHEMA = SchemaMigration(
    version=13,
    name="immutable_streaming_finalization_operations",
    statements=(
        "ALTER TABLE delivery_outbox ADD COLUMN execution_contract TEXT NOT NULL "
        "DEFAULT 'send_fragments' CHECK (execution_contract IN "
        "('send_fragments', 'streaming_finalization'))",
        "ALTER TABLE delivery_fragments ADD COLUMN operation TEXT NOT NULL "
        "DEFAULT 'send' CHECK (operation IN ('send', 'edit'))",
        "ALTER TABLE delivery_fragments ADD COLUMN target_external_message_id INTEGER "
        "CHECK ((operation = 'send' AND target_external_message_id IS NULL) OR "
        "(operation = 'edit' AND target_external_message_id > 0))",
        "CREATE INDEX delivery_outbox_contract_due_idx ON delivery_outbox "
        "(execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_contract_binding_order_idx ON delivery_outbox "
        "(execution_contract, purpose, channel_binding_id, requested_event_position, status)",
    ),
)

_DELIVERY_AUTHORITY_EPOCH_SCHEMA = SchemaMigration(
    version=14,
    name="conversation_delivery_authority_epochs",
    statements=(
        """
        CREATE TABLE delivery_authority_epochs (
            id TEXT PRIMARY KEY,
            lane TEXT NOT NULL CHECK (lane = 'telegram_conversation_streaming_finalization'),
            status TEXT NOT NULL CHECK (status IN ('active', 'deactivated')),
            activated_at TEXT NOT NULL,
            deactivated_at TEXT,
            terminal_failures_acknowledged_at TEXT,
            CHECK (
                (status = 'active' AND deactivated_at IS NULL)
                OR (status = 'deactivated' AND deactivated_at IS NOT NULL)
            ),
            CHECK (
                status = 'deactivated' OR terminal_failures_acknowledged_at IS NULL
            )
        )
        """,
        "CREATE UNIQUE INDEX delivery_authority_epochs_active_lane_idx "
        "ON delivery_authority_epochs (lane) WHERE status = 'active'",
        "ALTER TABLE delivery_outbox ADD COLUMN authority_epoch_id TEXT "
        "REFERENCES delivery_authority_epochs(id) ON DELETE RESTRICT",
        "CREATE INDEX delivery_outbox_authority_due_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_authority_binding_order_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, channel_binding_id, "
        "requested_event_position, status)",
    ),
)

_DURABLE_RUN_LIFECYCLE_SCHEMA = SchemaMigration(
    version=15,
    name="durable_workshop_run_lifecycle",
    statements=(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            requested_by_principal_id TEXT NOT NULL
                REFERENCES principals(id) ON DELETE RESTRICT,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
            inbound_message_id TEXT NOT NULL UNIQUE
                REFERENCES messages(id) ON DELETE RESTRICT,
            status TEXT NOT NULL CHECK (
                status IN ('accepted', 'started', 'completed', 'failed', 'cancelled')
            ),
            accepted_at TEXT NOT NULL,
            started_at TEXT,
            terminal_at TEXT,
            terminal_code TEXT,
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            CHECK (
                (status = 'accepted' AND started_at IS NULL
                    AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'started' AND started_at IS NOT NULL
                    AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'completed' AND started_at IS NOT NULL
                    AND terminal_at IS NOT NULL AND terminal_code IS NULL)
                OR (status = 'failed' AND started_at IS NOT NULL
                    AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status = 'cancelled' AND terminal_at IS NOT NULL
                    AND terminal_code IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX runs_channel_status_idx ON runs (channel_id, status, accepted_at)",
        "CREATE INDEX runs_agent_status_idx ON runs (agent_id, status, accepted_at)",
    ),
)

_RUN_EXECUTION_AUTHORITY_SCHEMA = SchemaMigration(
    version=16,
    name="durable_run_execution_authority",
    statements=(
        "ALTER TABLE runs ADD COLUMN cancellation_requested_at TEXT",
        "ALTER TABLE runs ADD COLUMN cancellation_code TEXT",
        "ALTER TABLE runs ADD COLUMN result_message_id TEXT REFERENCES messages(id) ON DELETE RESTRICT",
        """
        CREATE TABLE run_attempts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            attempt_sequence INTEGER NOT NULL CHECK (attempt_sequence > 0),
            owner_id TEXT NOT NULL,
            fence_token INTEGER NOT NULL CHECK (fence_token > 0),
            status TEXT NOT NULL CHECK (
                status IN (
                    'granted', 'started', 'expired', 'interrupted',
                    'completed', 'failed', 'cancelled'
                )
            ),
            backend TEXT NOT NULL,
            provider TEXT,
            model TEXT NOT NULL,
            execution_contract TEXT NOT NULL,
            lease_version INTEGER NOT NULL DEFAULT 1 CHECK (lease_version > 0),
            granted_at TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            started_at TEXT,
            terminal_at TEXT,
            terminal_code TEXT,
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (run_id, attempt_sequence),
            UNIQUE (run_id, fence_token),
            CHECK (
                (status = 'granted' AND started_at IS NULL
                    AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'started' AND started_at IS NOT NULL
                    AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'expired' AND started_at IS NULL
                    AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status IN ('interrupted', 'failed') AND started_at IS NOT NULL
                    AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status = 'cancelled'
                    AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status = 'completed' AND started_at IS NOT NULL
                    AND terminal_at IS NOT NULL
                    AND terminal_code IS NULL)
            )
        )
        """,
        "CREATE UNIQUE INDEX run_attempts_active_run_idx ON run_attempts (run_id) "
        "WHERE status IN ('granted', 'started')",
        "CREATE INDEX run_attempts_lease_idx ON run_attempts (status, lease_expires_at)",
        "CREATE INDEX run_attempts_owner_idx ON run_attempts (owner_id, fence_token, status)",
    ),
)

_CLIENT_SECURITY_STATE_ISOLATION_SCHEMA = SchemaMigration(
    version=17,
    name="isolate_client_security_state_from_projection_rebuilds",
    statements=(
        """
        CREATE TABLE workshop_client_devices_v17 (
            -- principal_id is deliberately not a foreign key. Principals are
            -- replayed collaboration projections, while devices are mutable
            -- security state that must survive projection reset/replay.
            id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL,
            display_name TEXT NOT NULL CHECK (
                length(trim(display_name)) > 0 AND length(display_name) <= 200
            ),
            created_at TEXT NOT NULL,
            last_seen_at TEXT,
            revoked_at TEXT,
            UNIQUE (id, principal_id)
        )
        """,
        """
        CREATE TABLE workshop_client_sessions_v17 (
            id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE CHECK (length(token_hash) = 64),
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_seen_at TEXT,
            revoked_at TEXT,
            FOREIGN KEY (device_id, principal_id)
                REFERENCES workshop_client_devices_v17(id, principal_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE workshop_client_enrollment_grants_v17 (
            id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE CHECK (length(token_hash) = 64),
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            redeemed_at TEXT,
            revoked_at TEXT,
            device_id TEXT,
            session_id TEXT REFERENCES workshop_client_sessions_v17(id) ON DELETE RESTRICT,
            FOREIGN KEY (device_id, principal_id)
                REFERENCES workshop_client_devices_v17(id, principal_id) ON DELETE RESTRICT,
            CHECK (
                (redeemed_at IS NULL AND device_id IS NULL AND session_id IS NULL)
                OR (redeemed_at IS NOT NULL AND device_id IS NOT NULL AND session_id IS NOT NULL)
            )
        )
        """,
        """
        INSERT INTO workshop_client_devices_v17 (
            id, principal_id, display_name, created_at, last_seen_at, revoked_at
        ) SELECT id, principal_id, display_name, created_at, last_seen_at, revoked_at
          FROM workshop_client_devices
        """,
        """
        INSERT INTO workshop_client_sessions_v17 (
            id, device_id, principal_id, token_hash, created_at, expires_at,
            last_seen_at, revoked_at
        ) SELECT id, device_id, principal_id, token_hash, created_at, expires_at,
                 last_seen_at, revoked_at
          FROM workshop_client_sessions
        """,
        """
        INSERT INTO workshop_client_enrollment_grants_v17 (
            id, principal_id, token_hash, created_at, expires_at, redeemed_at,
            revoked_at, device_id, session_id
        ) SELECT id, principal_id, token_hash, created_at, expires_at, redeemed_at,
                 revoked_at, device_id, session_id
          FROM workshop_client_enrollment_grants
        """,
        "DROP TABLE workshop_client_enrollment_grants",
        "DROP TABLE workshop_client_sessions",
        "DROP TABLE workshop_client_devices",
        "ALTER TABLE workshop_client_devices_v17 RENAME TO workshop_client_devices",
        "ALTER TABLE workshop_client_sessions_v17 RENAME TO workshop_client_sessions",
        "ALTER TABLE workshop_client_enrollment_grants_v17 RENAME TO workshop_client_enrollment_grants",
        "CREATE INDEX workshop_client_devices_principal_idx ON workshop_client_devices (principal_id, revoked_at)",
        "CREATE INDEX workshop_client_sessions_principal_idx "
        "ON workshop_client_sessions (principal_id, revoked_at, expires_at)",
        "CREATE INDEX workshop_client_sessions_device_idx ON workshop_client_sessions (device_id, revoked_at)",
        "CREATE INDEX workshop_client_enrollment_principal_idx "
        "ON workshop_client_enrollment_grants (principal_id, redeemed_at, revoked_at, expires_at)",
    ),
)

_RUNTIME_ASSIGNMENT_SCHEMA = SchemaMigration(
    version=18,
    name="explicit_channel_agent_runtime_assignments",
    statements=(
        """
        CREATE TABLE channel_agent_runtime_assignments (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (channel_id, agent_id),
            UNIQUE (runtime_profile_id)
        )
        """,
        "CREATE INDEX channel_agent_runtime_profile_idx ON channel_agent_runtime_assignments (runtime_profile_id)",
    ),
)

_NOTIFICATION_DELIVERY_PURPOSE_SCHEMA = SchemaMigration(
    version=19,
    name="durable_notification_delivery_purpose",
    statements=(
        # SQLite cannot alter a column CHECK constraint in place. Defer the
        # purpose expansion to a replacement table. Preserve and rebuild the
        # dependent attempt/fragment tables too: dropping their parent alone
        # would apply ON DELETE CASCADE and silently discard durable progress.
        """
        CREATE TABLE delivery_outbox_v19 (
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
            purpose TEXT NOT NULL DEFAULT 'qualification' CHECK (
                purpose IN ('conversation_reply', 'notification', 'qualification')
            ),
            execution_contract TEXT NOT NULL DEFAULT 'send_fragments' CHECK (
                execution_contract IN ('send_fragments', 'streaming_finalization')
            ),
            authority_epoch_id TEXT
                REFERENCES delivery_authority_epochs(id) ON DELETE RESTRICT,
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
        """
        INSERT INTO delivery_outbox_v19 (
            id, workshop_id, channel_id, channel_binding_id, message_id,
            transport, mode, status, max_attempts, attempt_count, available_at,
            lease_id, lease_owner, lease_expires_at, last_error_code,
            requested_event_position, created_at, updated_at, completed_at,
            purpose, execution_contract, authority_epoch_id
        ) SELECT
            id, workshop_id, channel_id, channel_binding_id, message_id,
            transport, mode, status, max_attempts, attempt_count, available_at,
            lease_id, lease_owner, lease_expires_at, last_error_code,
            requested_event_position, created_at, updated_at, completed_at,
            purpose, execution_contract, authority_epoch_id
        FROM delivery_outbox
        """,
        """
        CREATE TABLE delivery_attempts_v19 (
            id TEXT PRIMARY KEY,
            delivery_id TEXT NOT NULL REFERENCES delivery_outbox_v19(id) ON DELETE CASCADE,
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
        """
        INSERT INTO delivery_attempts_v19 (
            id, delivery_id, attempt_number, worker_id, started_at,
            lease_expires_at, completed_at, outcome, error_code
        ) SELECT
            id, delivery_id, attempt_number, worker_id, started_at,
            lease_expires_at, completed_at, outcome, error_code
        FROM delivery_attempts
        """,
        """
        CREATE TABLE delivery_fragments_v19 (
            delivery_id TEXT NOT NULL
                REFERENCES delivery_outbox_v19(id) ON DELETE CASCADE,
            fragment_index INTEGER NOT NULL CHECK (fragment_index >= 0),
            fragment_count INTEGER NOT NULL CHECK (fragment_count > 0),
            body TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 4096),
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'sending', 'sent', 'uncertain')
            ),
            attempt_id TEXT REFERENCES delivery_attempts_v19(id) ON DELETE RESTRICT,
            external_message_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            sent_at TEXT,
            operation TEXT NOT NULL DEFAULT 'send' CHECK (
                operation IN ('send', 'edit')
            ),
            target_external_message_id INTEGER CHECK (
                (operation = 'send' AND target_external_message_id IS NULL)
                OR (operation = 'edit' AND target_external_message_id > 0)
            ),
            PRIMARY KEY (delivery_id, fragment_index),
            CHECK (fragment_index < fragment_count),
            CHECK (
                (
                    status = 'pending'
                    AND attempt_id IS NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status IN ('sending', 'uncertain')
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status = 'sent'
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NOT NULL
                    AND sent_at IS NOT NULL
                )
            )
        )
        """,
        """
        INSERT INTO delivery_fragments_v19 (
            delivery_id, fragment_index, fragment_count, body, status,
            attempt_id, external_message_id, created_at, updated_at, sent_at,
            operation, target_external_message_id
        ) SELECT
            delivery_id, fragment_index, fragment_count, body, status,
            attempt_id, external_message_id, created_at, updated_at, sent_at,
            operation, target_external_message_id
        FROM delivery_fragments
        """,
        "DROP TABLE delivery_fragments",
        "DROP TABLE delivery_attempts",
        "DROP TABLE delivery_outbox",
        "ALTER TABLE delivery_outbox_v19 RENAME TO delivery_outbox",
        "ALTER TABLE delivery_attempts_v19 RENAME TO delivery_attempts",
        "ALTER TABLE delivery_fragments_v19 RENAME TO delivery_fragments",
        "CREATE INDEX delivery_outbox_due_idx ON delivery_outbox (status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_lease_expiry_idx ON delivery_outbox (status, lease_expires_at)",
        "CREATE INDEX delivery_outbox_binding_order_idx "
        "ON delivery_outbox (channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_purpose_due_idx "
        "ON delivery_outbox (purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_purpose_binding_order_idx "
        "ON delivery_outbox (purpose, channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_contract_due_idx ON delivery_outbox "
        "(execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_contract_binding_order_idx ON delivery_outbox "
        "(execution_contract, purpose, channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_authority_due_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_authority_binding_order_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, channel_binding_id, "
        "requested_event_position, status)",
        "CREATE INDEX delivery_attempts_delivery_idx ON delivery_attempts (delivery_id, attempt_number)",
        "CREATE INDEX delivery_fragments_status_idx ON delivery_fragments (delivery_id, status, fragment_index)",
    ),
)

_CANONICAL_RUNTIME_SESSION_SCHEMA = SchemaMigration(
    version=20,
    name="canonical_channel_agent_runtime_sessions",
    statements=(
        "CREATE UNIQUE INDEX channel_agent_runtime_assignment_tuple_idx "
        "ON channel_agent_runtime_assignments (channel_id, agent_id, runtime_profile_id)",
        """
        CREATE TABLE workshop_continuity_cutover (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            event_position INTEGER NOT NULL CHECK (event_position >= 0)
        )
        """,
        "INSERT INTO workshop_continuity_cutover (singleton, event_position) "
        "SELECT 1, COALESCE(MAX(position), 0) FROM event_log",
        """
        CREATE TABLE channel_agent_runtime_sessions (
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            backend TEXT NOT NULL CHECK (length(trim(backend)) > 0),
            provider TEXT,
            model TEXT NOT NULL CHECK (length(trim(model)) > 0),
            workspace TEXT NOT NULL CHECK (length(trim(workspace)) > 0),
            provider_session_id TEXT,
            -- These canonical IDs deliberately are not foreign keys. The
            -- referenced collaboration tables are replayable projections;
            -- mutable continuity state must survive their reset/rebuild.
            last_run_id TEXT NOT NULL,
            last_result_message_id TEXT NOT NULL,
            context_through_event_position INTEGER NOT NULL CHECK (
                context_through_event_position > 0
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (channel_id, agent_id)
        )
        """,
        "CREATE INDEX channel_agent_runtime_sessions_profile_idx "
        "ON channel_agent_runtime_sessions (runtime_profile_id)",
        "CREATE INDEX channel_agent_runtime_sessions_run_idx ON channel_agent_runtime_sessions (last_run_id)",
    ),
)

_CANONICAL_EXECUTION_STATE_SCHEMA = SchemaMigration(
    version=21,
    name="canonical_execution_state",
    statements=(
        # Canonical IDs deliberately are not foreign keys in these mutable
        # state tables. Collaboration tables are replayable projections;
        # execution settings and workspace policy must survive their rebuild.
        """
        CREATE TABLE channel_agent_execution_settings (
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            field TEXT NOT NULL CHECK (field IN ('model', 'timeout', 'workspace')),
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (channel_id, agent_id, field)
        )
        """,
        "CREATE INDEX channel_agent_execution_settings_profile_idx "
        "ON channel_agent_execution_settings (runtime_profile_id)",
        """
        CREATE TABLE channel_agent_workspace_settings (
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            workspace_path TEXT NOT NULL CHECK (length(trim(workspace_path)) > 0),
            field TEXT NOT NULL CHECK (field IN ('model', 'timeout', 'env', 'prompt')),
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (channel_id, agent_id, workspace_path, field)
        )
        """,
        "CREATE INDEX channel_agent_workspace_settings_profile_idx "
        "ON channel_agent_workspace_settings (runtime_profile_id)",
        """
        CREATE TABLE principal_workspace_history (
            principal_id TEXT NOT NULL,
            path TEXT NOT NULL CHECK (length(trim(path)) > 0),
            last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (principal_id, path)
        )
        """,
        "CREATE INDEX principal_workspace_history_recency_idx "
        "ON principal_workspace_history (principal_id, last_used_at DESC)",
        """
        CREATE TABLE principal_workspace_grants (
            principal_id TEXT NOT NULL,
            path TEXT NOT NULL CHECK (length(trim(path)) > 0),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (principal_id, path)
        )
        """,
        """
        CREATE TABLE workshop_execution_state_migrations (
            runtime_profile_id TEXT PRIMARY KEY CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            runtime_config_id INTEGER NOT NULL UNIQUE CHECK (runtime_config_id > 0),
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            settings_count INTEGER NOT NULL CHECK (settings_count >= 0),
            workspace_settings_count INTEGER NOT NULL CHECK (workspace_settings_count >= 0),
            history_count INTEGER NOT NULL CHECK (history_count >= 0),
            grants_count INTEGER NOT NULL CHECK (grants_count >= 0),
            migrated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
    ),
)

_CANONICAL_MEMORY_AUTHORITY_SCHEMA = SchemaMigration(
    version=22,
    name="canonical_memory_authority",
    statements=(
        """
        CREATE TABLE workshop_memory_authority_migrations (
            runtime_profile_id TEXT PRIMARY KEY CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            runtime_config_id INTEGER NOT NULL UNIQUE CHECK (runtime_config_id > 0),
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            moved_count INTEGER NOT NULL CHECK (moved_count >= 0),
            stamped_count INTEGER NOT NULL CHECK (stamped_count >= 0),
            total_count INTEGER NOT NULL CHECK (total_count >= 0),
            migrated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
    ),
)

_CANONICAL_OPERATIONAL_STATE_SCHEMA = SchemaMigration(
    version=23,
    name="canonical_operational_state_authority",
    statements=(
        # Job definitions remain in the compatibility table while scheduling
        # and ownership are core-owned and canonical. The legacy chat_id is
        # only a protected runtime compatibility alias.
        """
        CREATE TABLE workshop_job_owners (
            job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
        "CREATE INDEX workshop_job_owners_principal_idx ON workshop_job_owners (principal_id, job_id)",
        "CREATE INDEX workshop_job_owners_lane_idx ON workshop_job_owners (channel_id, agent_id, job_id)",
        # GitHub subscription policy belongs to a human principal. Operator
        # baselines are synchronized from protected configuration at startup;
        # interactive additions, removals, and toggle overrides remain here.
        """
        CREATE TABLE principal_github_subscriptions (
            principal_id TEXT PRIMARY KEY,
            baseline_repos_json TEXT NOT NULL,
            added_repos_json TEXT NOT NULL,
            removed_repos_json TEXT NOT NULL,
            pr_review_enabled INTEGER NOT NULL CHECK (pr_review_enabled IN (0, 1)),
            issue_triage_enabled INTEGER NOT NULL CHECK (issue_triage_enabled IN (0, 1)),
            pr_review_source TEXT NOT NULL CHECK (
                pr_review_source IN ('default', 'operator', 'user')
            ),
            issue_triage_source TEXT NOT NULL CHECK (
                issue_triage_source IN ('default', 'operator', 'user')
            ),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
        """
        CREATE TABLE workshop_operational_state_migrations (
            runtime_profile_id TEXT PRIMARY KEY CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            runtime_config_id INTEGER NOT NULL UNIQUE CHECK (runtime_config_id > 0),
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            jobs_count INTEGER NOT NULL CHECK (jobs_count >= 0),
            github_subscription_count INTEGER NOT NULL CHECK (
                github_subscription_count IN (0, 1)
            ),
            migrated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
    ),
)

_RUN_TRACE_SCHEMA = SchemaMigration(
    version=24,
    name="durable_run_traces",
    statements=(
        """
        CREATE TABLE run_traces (
            -- run_id deliberately is not a foreign key, matching
            -- telegram_streaming_previews: rows are pruned with their
            -- run row by run pruning, not by cascade, and there is no
            -- separate TTL. seq is dense per run, assigned at
            -- persistence time under the write transaction.
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            kind TEXT NOT NULL,
            tool_name TEXT,
            tool_use_id TEXT,
            summary TEXT NOT NULL,
            detail TEXT NOT NULL,
            is_diff INTEGER NOT NULL DEFAULT 0,
            is_error INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, seq)
        )
        """,
    ),
)

_CANONICAL_TRANSCRIPT_AUTHORITY_SCHEMA = SchemaMigration(
    version=25,
    name="canonical_transcript_authority",
    statements=(
        """
        CREATE TABLE workshop_transcript_authority (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            protected_private_text_cutover_position INTEGER NOT NULL CHECK (
                protected_private_text_cutover_position >= 0
            ),
            activated_at TEXT NOT NULL
        )
        """,
        """
        INSERT INTO workshop_transcript_authority (
            singleton, protected_private_text_cutover_position, activated_at
        )
        SELECT 1, COALESCE(MAX(position), 0),
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
          FROM event_log
        """,
    ),
)

_CORE_SCHEDULER_SCHEMA = SchemaMigration(
    version=26,
    name="core_scheduler_firing_authority",
    statements=(
        """
        CREATE TABLE workshop_schedule_firings (
            firing_id TEXT PRIMARY KEY CHECK (length(firing_id) BETWEEN 1 AND 128),
            job_id INTEGER NOT NULL CHECK (job_id > 0),
            occurrence_id TEXT NOT NULL CHECK (
                length(occurrence_id) BETWEEN 1 AND 128
            ),
            scheduled_for TEXT NOT NULL,
            job_type TEXT NOT NULL CHECK (job_type IN ('reminder', 'agent')),
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'executing', 'succeeded', 'failed')
            ),
            canonical_message_id TEXT,
            run_id TEXT,
            terminal_code TEXT,
            condition_met INTEGER NOT NULL DEFAULT 0 CHECK (
                condition_met IN (0, 1)
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            last_error_code TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE (job_id, occurrence_id)
        )
        """,
        "CREATE INDEX workshop_schedule_firings_status_idx "
        "ON workshop_schedule_firings (status, scheduled_for, firing_id)",
        "CREATE INDEX workshop_schedule_firings_job_idx "
        "ON workshop_schedule_firings (job_id, scheduled_for, firing_id)",
    ),
)

_GITHUB_AUTOMATION_SCHEMA = SchemaMigration(
    version=27,
    name="canonical_github_automation",
    statements=(
        """
        CREATE TABLE workshop_github_automation_work (
            work_id TEXT PRIMARY KEY CHECK (length(work_id) BETWEEN 1 AND 128),
            delivery_id TEXT NOT NULL CHECK (length(delivery_id) BETWEEN 1 AND 128),
            kind TEXT NOT NULL CHECK (kind IN ('pr_review', 'issue_triage')),
            event_type TEXT NOT NULL,
            action TEXT NOT NULL,
            repository TEXT NOT NULL,
            item_number INTEGER NOT NULL CHECK (item_number > 0),
            principal_id TEXT NOT NULL,
            execution_channel_id TEXT NOT NULL,
            notification_channel_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL,
            local_repo_path TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'executing', 'succeeded', 'failed', 'uncertain')
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            canonical_message_id TEXT,
            last_error_code TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE (delivery_id, kind, principal_id)
        )
        """,
        "CREATE INDEX workshop_github_automation_status_idx "
        "ON workshop_github_automation_work (status, created_at, work_id)",
    ),
)

_INTEGRATION_ROUTE_SCHEMA = SchemaMigration(
    version=28,
    name="canonical_integration_routes",
    statements=(
        """
        CREATE TABLE workshop_integration_routes (
            source TEXT NOT NULL CHECK (
                length(source) BETWEEN 1 AND 32
                AND source NOT GLOB '*[^a-z0-9_]*'
            ),
            route_name TEXT NOT NULL CHECK (
                length(route_name) BETWEEN 1 AND 64
                AND route_name NOT GLOB '*[^a-z0-9_-]*'
            ),
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (source, route_name)
        )
        """,
        "CREATE INDEX workshop_integration_routes_channel_idx "
        "ON workshop_integration_routes (channel_id, source, route_name)",
    ),
)

_CANONICAL_SCHEDULED_JOB_SCHEMA = SchemaMigration(
    version=29,
    name="canonical_scheduled_job_definitions",
    statements=(
        """
        CREATE TABLE workshop_scheduled_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            name TEXT NOT NULL,
            job_type TEXT NOT NULL CHECK (job_type IN ('reminder', 'agent')),
            prompt TEXT NOT NULL,
            schedule_type TEXT NOT NULL CHECK (
                schedule_type IN ('once', 'daily', 'interval')
            ),
            schedule_data TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            auto_remove INTEGER NOT NULL DEFAULT 0 CHECK (auto_remove IN (0, 1)),
            notify_on_check INTEGER NOT NULL DEFAULT 0 CHECK (notify_on_check IN (0, 1))
        )
        """,
        "CREATE INDEX workshop_scheduled_jobs_owner_idx ON workshop_scheduled_jobs "
        "(principal_id, channel_id, agent_id, runtime_profile_id, active, id)",
        "CREATE INDEX workshop_scheduled_jobs_active_idx ON workshop_scheduled_jobs (active, id)",
        """
        CREATE TABLE workshop_scheduled_job_migrations (
            runtime_profile_id TEXT PRIMARY KEY CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            runtime_config_id INTEGER NOT NULL,
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            legacy_jobs_count INTEGER NOT NULL CHECK (legacy_jobs_count >= 0),
            migrated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
    ),
)

_ADAPTER_PLUGGABLE_DELIVERY_SCHEMA = SchemaMigration(
    version=30,
    name="adapter_pluggable_delivery_authority",
    statements=(
        # Replace the transport-named authority lane without losing durable
        # outbox, attempt, or fragment state. SQLite cannot alter the lane
        # CHECK constraint or retarget the dependent foreign keys in place.
        """
        CREATE TABLE delivery_authority_epochs_v30 (
            id TEXT PRIMARY KEY,
            lane TEXT NOT NULL CHECK (lane = 'conversation_streaming_finalization'),
            status TEXT NOT NULL CHECK (status IN ('active', 'deactivated')),
            activated_at TEXT NOT NULL,
            deactivated_at TEXT,
            terminal_failures_acknowledged_at TEXT,
            CHECK (
                (status = 'active' AND deactivated_at IS NULL)
                OR (status = 'deactivated' AND deactivated_at IS NOT NULL)
            ),
            CHECK (
                status = 'deactivated' OR terminal_failures_acknowledged_at IS NULL
            )
        )
        """,
        """
        INSERT INTO delivery_authority_epochs_v30 (
            id, lane, status, activated_at, deactivated_at,
            terminal_failures_acknowledged_at
        ) SELECT
            id, 'conversation_streaming_finalization', status, activated_at,
            deactivated_at, terminal_failures_acknowledged_at
        FROM delivery_authority_epochs
        """,
        """
        CREATE TABLE delivery_outbox_v30 (
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
            purpose TEXT NOT NULL DEFAULT 'qualification' CHECK (
                purpose IN ('conversation_reply', 'notification', 'qualification')
            ),
            execution_contract TEXT NOT NULL DEFAULT 'send_fragments' CHECK (
                execution_contract IN ('send_fragments', 'streaming_finalization')
            ),
            authority_epoch_id TEXT
                REFERENCES delivery_authority_epochs_v30(id) ON DELETE RESTRICT,
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
        """
        INSERT INTO delivery_outbox_v30 (
            id, workshop_id, channel_id, channel_binding_id, message_id,
            transport, mode, status, max_attempts, attempt_count, available_at,
            lease_id, lease_owner, lease_expires_at, last_error_code,
            requested_event_position, created_at, updated_at, completed_at,
            purpose, execution_contract, authority_epoch_id
        ) SELECT
            id, workshop_id, channel_id, channel_binding_id, message_id,
            transport, mode, status, max_attempts, attempt_count, available_at,
            lease_id, lease_owner, lease_expires_at, last_error_code,
            requested_event_position, created_at, updated_at, completed_at,
            purpose, execution_contract, authority_epoch_id
        FROM delivery_outbox
        """,
        """
        CREATE TABLE delivery_attempts_v30 (
            id TEXT PRIMARY KEY,
            delivery_id TEXT NOT NULL REFERENCES delivery_outbox_v30(id) ON DELETE CASCADE,
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
        """
        INSERT INTO delivery_attempts_v30 (
            id, delivery_id, attempt_number, worker_id, started_at,
            lease_expires_at, completed_at, outcome, error_code
        ) SELECT
            id, delivery_id, attempt_number, worker_id, started_at,
            lease_expires_at, completed_at, outcome, error_code
        FROM delivery_attempts
        """,
        """
        CREATE TABLE delivery_fragments_v30 (
            delivery_id TEXT NOT NULL
                REFERENCES delivery_outbox_v30(id) ON DELETE CASCADE,
            fragment_index INTEGER NOT NULL CHECK (fragment_index >= 0),
            fragment_count INTEGER NOT NULL CHECK (fragment_count > 0),
            body TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 1000000),
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'sending', 'sent', 'uncertain')
            ),
            attempt_id TEXT REFERENCES delivery_attempts_v30(id) ON DELETE RESTRICT,
            external_message_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            sent_at TEXT,
            operation TEXT NOT NULL DEFAULT 'send' CHECK (
                operation IN ('send', 'edit')
            ),
            target_external_message_id TEXT CHECK (
                (operation = 'send' AND target_external_message_id IS NULL)
                OR (operation = 'edit' AND length(target_external_message_id) BETWEEN 1 AND 255)
            ),
            PRIMARY KEY (delivery_id, fragment_index),
            CHECK (fragment_index < fragment_count),
            CHECK (
                (
                    status = 'pending'
                    AND attempt_id IS NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status IN ('sending', 'uncertain')
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NULL
                    AND sent_at IS NULL
                ) OR (
                    status = 'sent'
                    AND attempt_id IS NOT NULL
                    AND external_message_id IS NOT NULL
                    AND sent_at IS NOT NULL
                )
            )
        )
        """,
        """
        INSERT INTO delivery_fragments_v30 (
            delivery_id, fragment_index, fragment_count, body, status,
            attempt_id, external_message_id, created_at, updated_at, sent_at,
            operation, target_external_message_id
        ) SELECT
            delivery_id, fragment_index, fragment_count, body, status,
            attempt_id, external_message_id, created_at, updated_at, sent_at,
            operation, target_external_message_id
        FROM delivery_fragments
        """,
        "DROP TABLE delivery_fragments",
        "DROP TABLE delivery_attempts",
        "DROP TABLE delivery_outbox",
        "DROP TABLE delivery_authority_epochs",
        "ALTER TABLE delivery_authority_epochs_v30 RENAME TO delivery_authority_epochs",
        "ALTER TABLE delivery_outbox_v30 RENAME TO delivery_outbox",
        "ALTER TABLE delivery_attempts_v30 RENAME TO delivery_attempts",
        "ALTER TABLE delivery_fragments_v30 RENAME TO delivery_fragments",
        "CREATE UNIQUE INDEX delivery_authority_epochs_active_lane_idx "
        "ON delivery_authority_epochs (lane) WHERE status = 'active'",
        "CREATE INDEX delivery_outbox_due_idx ON delivery_outbox (status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_lease_expiry_idx ON delivery_outbox (status, lease_expires_at)",
        "CREATE INDEX delivery_outbox_binding_order_idx ON delivery_outbox "
        "(channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_purpose_due_idx ON delivery_outbox "
        "(purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_purpose_binding_order_idx ON delivery_outbox "
        "(purpose, channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_contract_due_idx ON delivery_outbox "
        "(execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_contract_binding_order_idx ON delivery_outbox "
        "(execution_contract, purpose, channel_binding_id, requested_event_position, status)",
        "CREATE INDEX delivery_outbox_authority_due_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, status, available_at, requested_event_position)",
        "CREATE INDEX delivery_outbox_authority_binding_order_idx ON delivery_outbox "
        "(authority_epoch_id, execution_contract, purpose, channel_binding_id, "
        "requested_event_position, status)",
        "CREATE INDEX delivery_attempts_delivery_idx ON delivery_attempts (delivery_id, attempt_number)",
        "CREATE INDEX delivery_fragments_status_idx ON delivery_fragments (delivery_id, status, fragment_index)",
    ),
)

_CANONICAL_POST_RUN_EFFECTS_SCHEMA = SchemaMigration(
    version=31,
    name="canonical_post_run_effects",
    statements=(
        """
        CREATE TABLE workshop_post_run_effects (
            run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
            effect_type TEXT NOT NULL CHECK (
                effect_type = 'semantic_memory_ingestion'
            ),
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE RESTRICT,
            result_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE RESTRICT,
            workspace TEXT NOT NULL CHECK (length(trim(workspace)) > 0),
            provider_session_id TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'executing', 'succeeded', 'failed')
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            last_error_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            CHECK (
                (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL)
                OR (status IN ('pending', 'executing') AND completed_at IS NULL)
            )
        )
        """,
        "CREATE INDEX workshop_post_run_effects_status_idx ON workshop_post_run_effects (status, created_at, run_id)",
        "CREATE INDEX workshop_post_run_effects_profile_idx ON workshop_post_run_effects (runtime_profile_id, status)",
    ),
)

# These IDs deliberately are not foreign keys. Runs and messages are a
# rebuildable projection: startup deletes and replays them from event_log.
# The effect receipt is an independent durable ledger and must survive that
# temporary absence. The worker validates all three IDs against the rebuilt
# canonical run before executing pending work.
_DURABLE_POST_RUN_EFFECT_RECEIPTS_SCHEMA = SchemaMigration(
    version=32,
    name="durable_post_run_effect_receipts",
    statements=(
        """
        CREATE TABLE workshop_post_run_effects_durable (
            run_id TEXT PRIMARY KEY,
            effect_type TEXT NOT NULL CHECK (
                effect_type = 'semantic_memory_ingestion'
            ),
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            source_message_id TEXT NOT NULL,
            result_message_id TEXT NOT NULL,
            workspace TEXT NOT NULL CHECK (length(trim(workspace)) > 0),
            provider_session_id TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('pending', 'executing', 'succeeded', 'failed')
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            last_error_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            CHECK (
                (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL)
                OR (status IN ('pending', 'executing') AND completed_at IS NULL)
            )
        )
        """,
        """
        INSERT INTO workshop_post_run_effects_durable (
            run_id, effect_type, runtime_profile_id, source_message_id,
            result_message_id, workspace, provider_session_id, status,
            attempt_count, last_error_code, created_at, updated_at, completed_at
        )
        SELECT
            run_id, effect_type, runtime_profile_id, source_message_id,
            result_message_id, workspace, provider_session_id, status,
            attempt_count, last_error_code, created_at, updated_at, completed_at
        FROM workshop_post_run_effects
        """,
        "DROP TABLE workshop_post_run_effects",
        "ALTER TABLE workshop_post_run_effects_durable RENAME TO workshop_post_run_effects",
        "CREATE INDEX workshop_post_run_effects_status_idx ON workshop_post_run_effects (status, created_at, run_id)",
        "CREATE INDEX workshop_post_run_effects_profile_idx ON workshop_post_run_effects (runtime_profile_id, status)",
    ),
)

_CANONICAL_RUNTIME_KEYS_SCHEMA = SchemaMigration(
    version=33,
    name="canonical_runtime_keys",
    statements=(
        "ALTER TABLE principal_github_subscriptions ADD COLUMN github_token TEXT",
        "ALTER TABLE principal_github_subscriptions ADD COLUMN allowed_triage_projects_json TEXT NOT NULL DEFAULT '[]'",
        """
        CREATE TABLE workshop_runtime_key_cutovers (
            runtime_profile_id TEXT PRIMARY KEY CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            legacy_runtime_key INTEGER UNIQUE CHECK (
                legacy_runtime_key IS NULL OR legacy_runtime_key > 0
            ),
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            settings_rows INTEGER NOT NULL CHECK (settings_rows >= 0),
            workspace_rows INTEGER NOT NULL CHECK (workspace_rows >= 0),
            session_rows INTEGER NOT NULL CHECK (session_rows >= 0),
            lock_rows INTEGER NOT NULL CHECK (lock_rows >= 0),
            history_rows INTEGER NOT NULL CHECK (history_rows >= 0),
            grant_rows INTEGER NOT NULL CHECK (grant_rows >= 0),
            github_rows INTEGER NOT NULL CHECK (github_rows >= 0),
            memory_rows INTEGER NOT NULL CHECK (memory_rows >= 0),
            legacy_reads_disabled INTEGER NOT NULL DEFAULT 1 CHECK (
                legacy_reads_disabled = 1
            ),
            cutover_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """,
    ),
)


_CANONICAL_BACKEND_SELECTION_SCHEMA = SchemaMigration(
    version=34,
    name="canonical_backend_selection",
    statements=(
        "ALTER TABLE channel_agent_execution_settings RENAME TO channel_agent_execution_settings_v33",
        """
        CREATE TABLE channel_agent_execution_settings (
            channel_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            field TEXT NOT NULL CHECK (
                field IN ('backend', 'model', 'timeout', 'workspace')
            ),
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            PRIMARY KEY (channel_id, agent_id, field)
        )
        """,
        """
        INSERT INTO channel_agent_execution_settings (
            channel_id, agent_id, runtime_profile_id, field, value, updated_at
        )
        SELECT
            channel_id, agent_id, runtime_profile_id, field, value, updated_at
        FROM channel_agent_execution_settings_v33
        """,
        "DROP TABLE channel_agent_execution_settings_v33",
        "CREATE INDEX channel_agent_execution_settings_profile_idx "
        "ON channel_agent_execution_settings (runtime_profile_id)",
    ),
)

_PRINCIPAL_NOTIFICATION_PREFERENCES_SCHEMA = SchemaMigration(
    version=35,
    name="principal_notification_delivery_preferences",
    statements=(
        """
        CREATE TABLE principal_notification_delivery_preferences (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            integration_class TEXT NOT NULL CHECK (
                integration_class IN ('github', 'generic')
            ),
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            PRIMARY KEY (principal_id, integration_class)
        )
        """,
        "CREATE INDEX principal_notification_preferences_channel_idx "
        "ON principal_notification_delivery_preferences (channel_id, principal_id)",
        """
        CREATE TABLE workshop_integration_route_owners (
            source TEXT NOT NULL,
            route_name TEXT NOT NULL,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            protected_channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            PRIMARY KEY (source, route_name),
            FOREIGN KEY (source, route_name)
                REFERENCES workshop_integration_routes(source, route_name)
                ON DELETE CASCADE
        )
        """,
        "CREATE INDEX workshop_integration_route_owners_principal_idx "
        "ON workshop_integration_route_owners (principal_id, source, route_name)",
    ),
)

_CLIENT_BINDING_VOICE_PREFERENCES_SCHEMA = SchemaMigration(
    version=36,
    name="client_binding_voice_preferences",
    statements=(
        """
        CREATE TABLE client_binding_voice_preferences (
            channel_binding_id TEXT PRIMARY KEY
                REFERENCES channel_bindings(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            mode TEXT NOT NULL CHECK (
                mode IN ('off', 'text_and_voice', 'voice_only')
            ),
            voice_name TEXT NOT NULL CHECK (length(trim(voice_name)) > 0),
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """,
        "CREATE INDEX client_binding_voice_preferences_principal_idx "
        "ON client_binding_voice_preferences (principal_id, channel_binding_id)",
        """
        CREATE TABLE client_binding_voice_migrations (
            channel_binding_id TEXT PRIMARY KEY
                REFERENCES channel_bindings(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            mode_migrated INTEGER NOT NULL CHECK (mode_migrated IN (0, 1)),
            voice_migrated INTEGER NOT NULL CHECK (voice_migrated IN (0, 1)),
            legacy_reads_disabled INTEGER NOT NULL DEFAULT 1 CHECK (
                legacy_reads_disabled = 1
            ),
            rollback_dual_writes INTEGER NOT NULL DEFAULT 1 CHECK (
                rollback_dual_writes = 1
            ),
            migrated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """,
        "CREATE INDEX client_binding_voice_migrations_principal_idx "
        "ON client_binding_voice_migrations (principal_id, channel_binding_id)",
    ),
)

_PRINCIPAL_APPEARANCE_PREFERENCES_SCHEMA = SchemaMigration(
    version=37,
    name="principal_appearance_preferences",
    statements=(
        """
        CREATE TABLE principal_appearance_preferences (
            principal_id TEXT PRIMARY KEY REFERENCES principals(id) ON DELETE CASCADE,
            theme_id TEXT NOT NULL CHECK (length(trim(theme_id)) > 0),
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """,
    ),
)

_DURABLE_PRINCIPAL_APPEARANCE_PREFERENCES_SCHEMA = SchemaMigration(
    version=38,
    name="isolate_principal_appearance_preferences_from_projection_rebuilds",
    statements=(
        """
        CREATE TABLE principal_appearance_preferences_v38 (
            -- principal_id is deliberately not a foreign key. Principals are
            -- replayed collaboration projections, while appearance choices
            -- are mutable principal state that must survive projection reset.
            principal_id TEXT PRIMARY KEY,
            theme_id TEXT NOT NULL CHECK (length(trim(theme_id)) > 0),
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
        )
        """,
        """
        INSERT INTO principal_appearance_preferences_v38 (
            principal_id, theme_id, created_at, updated_at
        ) SELECT principal_id, theme_id, created_at, updated_at
          FROM principal_appearance_preferences
        """,
        "DROP TABLE principal_appearance_preferences",
        "ALTER TABLE principal_appearance_preferences_v38 RENAME TO principal_appearance_preferences",
    ),
)

_CANONICAL_MODEL_CATALOGUE_SCHEMA = SchemaMigration(
    version=39,
    name="canonical_model_catalogue",
    statements=(
        """
        CREATE TABLE workshop_model_catalogue_refreshes (
            cache_key TEXT PRIMARY KEY CHECK (length(cache_key) = 64),
            principal_id TEXT NOT NULL CHECK (length(principal_id) BETWEEN 1 AND 128),
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            backend TEXT NOT NULL CHECK (length(trim(backend)) > 0),
            provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
            auth_fingerprint TEXT NOT NULL CHECK (length(auth_fingerprint) = 64),
            executable_fingerprint TEXT NOT NULL CHECK (
                length(executable_fingerprint) = 64
            ),
            status TEXT NOT NULL CHECK (
                status IN (
                    'refreshing', 'succeeded', 'failed', 'unsupported',
                    'malformed', 'timed_out', 'invalidated'
                )
            ),
            generation INTEGER NOT NULL CHECK (generation > 0),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            discovery_source TEXT,
            last_error_code TEXT,
            last_error_detail TEXT,
            refresh_started_at TEXT NOT NULL,
            last_attempt_at TEXT NOT NULL,
            last_successful_refresh_at TEXT,
            expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE UNIQUE INDEX workshop_model_catalogue_active_lane_idx "
        "ON workshop_model_catalogue_refreshes (runtime_profile_id, backend, provider) "
        "WHERE active = 1",
        "CREATE INDEX workshop_model_catalogue_refresh_status_idx "
        "ON workshop_model_catalogue_refreshes (status, active, updated_at)",
        """
        CREATE TABLE workshop_model_catalogue_discovered_entries (
            cache_key TEXT NOT NULL REFERENCES workshop_model_catalogue_refreshes(cache_key)
                ON DELETE CASCADE,
            model_id TEXT NOT NULL CHECK (length(trim(model_id)) BETWEEN 1 AND 512),
            display_label TEXT NOT NULL CHECK (
                length(trim(display_label)) BETWEEN 1 AND 512
            ),
            discovery_source TEXT NOT NULL CHECK (
                length(trim(discovery_source)) BETWEEN 1 AND 128
            ),
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK (
                status IN ('available', 'not_advertised', 'unavailable', 'unknown')
            ),
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_successful_refresh_at TEXT NOT NULL,
            expires_at TEXT,
            PRIMARY KEY (cache_key, model_id)
        )
        """,
        "CREATE INDEX workshop_model_catalogue_discovered_status_idx "
        "ON workshop_model_catalogue_discovered_entries (cache_key, status, model_id)",
        """
        CREATE TABLE workshop_model_catalogue_operator_entries (
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            backend TEXT NOT NULL CHECK (length(trim(backend)) > 0),
            provider TEXT NOT NULL CHECK (length(trim(provider)) > 0),
            model_id TEXT NOT NULL CHECK (length(trim(model_id)) BETWEEN 1 AND 512),
            display_label TEXT NOT NULL CHECK (
                length(trim(display_label)) BETWEEN 1 AND 512
            ),
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (runtime_profile_id, backend, provider, model_id)
        )
        """,
        "CREATE INDEX workshop_model_catalogue_operator_active_idx "
        "ON workshop_model_catalogue_operator_entries "
        "(runtime_profile_id, backend, provider, active, model_id)",
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
    _DELIVERY_FRAGMENT_SCHEMA,
    _BINDING_AWARE_DELIVERY_SCHEMA,
    _DELIVERY_BINDING_ORDER_SCHEMA,
    _DELIVERY_PURPOSE_SCHEMA,
    _TELEGRAM_STREAMING_PREVIEW_SCHEMA,
    _STREAMING_FINALIZATION_SCHEMA,
    _DELIVERY_AUTHORITY_EPOCH_SCHEMA,
    _DURABLE_RUN_LIFECYCLE_SCHEMA,
    _RUN_EXECUTION_AUTHORITY_SCHEMA,
    _CLIENT_SECURITY_STATE_ISOLATION_SCHEMA,
    _RUNTIME_ASSIGNMENT_SCHEMA,
    _NOTIFICATION_DELIVERY_PURPOSE_SCHEMA,
    _CANONICAL_RUNTIME_SESSION_SCHEMA,
    _CANONICAL_EXECUTION_STATE_SCHEMA,
    _CANONICAL_MEMORY_AUTHORITY_SCHEMA,
    _CANONICAL_OPERATIONAL_STATE_SCHEMA,
    _RUN_TRACE_SCHEMA,
    _CANONICAL_TRANSCRIPT_AUTHORITY_SCHEMA,
    _CORE_SCHEDULER_SCHEMA,
    _GITHUB_AUTOMATION_SCHEMA,
    _INTEGRATION_ROUTE_SCHEMA,
    _CANONICAL_SCHEDULED_JOB_SCHEMA,
    _ADAPTER_PLUGGABLE_DELIVERY_SCHEMA,
    _CANONICAL_POST_RUN_EFFECTS_SCHEMA,
    _DURABLE_POST_RUN_EFFECT_RECEIPTS_SCHEMA,
    _CANONICAL_RUNTIME_KEYS_SCHEMA,
    _CANONICAL_BACKEND_SELECTION_SCHEMA,
    _PRINCIPAL_NOTIFICATION_PREFERENCES_SCHEMA,
    _CLIENT_BINDING_VOICE_PREFERENCES_SCHEMA,
    _PRINCIPAL_APPEARANCE_PREFERENCES_SCHEMA,
    _DURABLE_PRINCIPAL_APPEARANCE_PREFERENCES_SCHEMA,
    _CANONICAL_MODEL_CATALOGUE_SCHEMA,
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
