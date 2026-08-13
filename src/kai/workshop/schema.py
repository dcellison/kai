"""Additive SQLite schema migrations for the Workshop foundation."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

WORKSHOP_SCHEMA_VERSION = 19


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
