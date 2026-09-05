"""Additive SQLite schema migrations for the Workshop foundation."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

WORKSHOP_SCHEMA_VERSION = 67


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

_REUSABLE_GROUP_RUNTIME_ASSIGNMENT_SCHEMA = SchemaMigration(
    version=40,
    name="reusable_group_runtime_assignments",
    statements=(
        """
        CREATE TABLE channel_agent_runtime_assignments_v40 (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            runtime_profile_id TEXT NOT NULL CHECK (
                length(runtime_profile_id) BETWEEN 1 AND 128
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (channel_id, agent_id)
        )
        """,
        """
        INSERT INTO channel_agent_runtime_assignments_v40 (
            id, channel_id, agent_id, runtime_profile_id, created_at,
            created_event_position
        ) SELECT
            id, channel_id, agent_id, runtime_profile_id, created_at,
            created_event_position
          FROM channel_agent_runtime_assignments
        """,
        "DROP TABLE channel_agent_runtime_assignments",
        "ALTER TABLE channel_agent_runtime_assignments_v40 RENAME TO channel_agent_runtime_assignments",
        "CREATE INDEX channel_agent_runtime_profile_idx ON channel_agent_runtime_assignments (runtime_profile_id)",
        "CREATE UNIQUE INDEX channel_agent_runtime_assignment_tuple_idx "
        "ON channel_agent_runtime_assignments "
        "(channel_id, agent_id, runtime_profile_id)",
    ),
)

_MESSAGE_MENTIONS_SCHEMA = SchemaMigration(
    version=41,
    name="canonical_message_mentions",
    statements=(
        "ALTER TABLE messages ADD COLUMN mentions_json TEXT NOT NULL DEFAULT '[]' "
        "CHECK (json_valid(mentions_json) AND json_type(mentions_json) = 'array')",
    ),
)

_GROUP_WAKE_POLICY_SCHEMA = SchemaMigration(
    version=42,
    name="group_agent_wake_policy",
    statements=(
        """
        CREATE TABLE runs_v42 (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            requested_by_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
            inbound_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE RESTRICT,
            status TEXT NOT NULL CHECK (
                status IN ('accepted', 'started', 'completed', 'failed', 'cancelled')
            ),
            accepted_at TEXT NOT NULL,
            started_at TEXT,
            terminal_at TEXT,
            terminal_code TEXT,
            last_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            cancellation_requested_at TEXT,
            cancellation_code TEXT,
            result_message_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            UNIQUE (inbound_message_id, agent_id),
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
        """
        INSERT INTO runs_v42 (
            id, workshop_id, channel_id, requested_by_principal_id, agent_id,
            inbound_message_id, status, accepted_at, started_at, terminal_at,
            terminal_code, last_event_position, cancellation_requested_at,
            cancellation_code, result_message_id
        ) SELECT
            id, workshop_id, channel_id, requested_by_principal_id, agent_id,
            inbound_message_id, status, accepted_at, started_at, terminal_at,
            terminal_code, last_event_position, cancellation_requested_at,
            cancellation_code, result_message_id
          FROM runs
        """,
        """
        CREATE TABLE run_attempts_v42 (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs_v42(id) ON DELETE CASCADE,
            attempt_sequence INTEGER NOT NULL CHECK (attempt_sequence > 0),
            owner_id TEXT NOT NULL,
            fence_token INTEGER NOT NULL CHECK (fence_token > 0),
            status TEXT NOT NULL CHECK (
                status IN ('granted', 'started', 'expired', 'interrupted',
                    'completed', 'failed', 'cancelled')
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
            last_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (run_id, attempt_sequence),
            UNIQUE (run_id, fence_token),
            CHECK (
                (status = 'granted' AND started_at IS NULL AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'started' AND started_at IS NOT NULL AND terminal_at IS NULL AND terminal_code IS NULL)
                OR (status = 'expired' AND started_at IS NULL AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status IN ('interrupted', 'failed') AND started_at IS NOT NULL AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status = 'cancelled' AND terminal_at IS NOT NULL AND terminal_code IS NOT NULL)
                OR (status = 'completed' AND started_at IS NOT NULL AND terminal_at IS NOT NULL AND terminal_code IS NULL)
            )
        )
        """,
        """
        INSERT INTO run_attempts_v42 (
            id, run_id, attempt_sequence, owner_id, fence_token, status,
            backend, provider, model, execution_contract, lease_version,
            granted_at, lease_expires_at, started_at, terminal_at,
            terminal_code, last_event_position
        ) SELECT
            id, run_id, attempt_sequence, owner_id, fence_token, status,
            backend, provider, model, execution_contract, lease_version,
            granted_at, lease_expires_at, started_at, terminal_at,
            terminal_code, last_event_position
          FROM run_attempts
        """,
        "DROP TABLE run_attempts",
        "DROP TABLE runs",
        "ALTER TABLE runs_v42 RENAME TO runs",
        "ALTER TABLE run_attempts_v42 RENAME TO run_attempts",
        "CREATE INDEX runs_channel_status_idx ON runs (channel_id, status, accepted_at)",
        "CREATE INDEX runs_agent_status_idx ON runs (agent_id, status, accepted_at)",
        "CREATE UNIQUE INDEX run_attempts_active_run_idx ON run_attempts (run_id) WHERE status IN ('granted', 'started')",
        "CREATE INDEX run_attempts_lease_idx ON run_attempts (status, lease_expires_at)",
        "CREATE INDEX run_attempts_owner_idx ON run_attempts (owner_id, fence_token, status)",
        """
        CREATE TABLE channel_agent_dismissals (
            id TEXT PRIMARY KEY,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            dismissed_by_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            thread_root_message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
            dismissed_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT
        )
        """,
        "CREATE INDEX channel_agent_dismissals_scope_idx ON channel_agent_dismissals (channel_id, agent_id, thread_root_message_id, dismissed_at)",
    ),
)

_MESSAGE_THREADS_SCHEMA = SchemaMigration(
    version=43,
    name="canonical_message_threads",
    statements=(
        "ALTER TABLE messages ADD COLUMN thread_root_id TEXT REFERENCES messages(id) ON DELETE CASCADE",
        "CREATE INDEX messages_channel_thread_position_idx ON messages "
        "(channel_id, thread_root_id, created_event_position)",
    ),
)

_EXPLICIT_TASK_ROUTING_SCHEMA = SchemaMigration(
    version=44,
    name="explicit_task_routing",
    statements=(
        """
        CREATE TABLE workshop_routing_policies (
            runtime_profile_id TEXT NOT NULL,
            task_class TEXT NOT NULL CHECK (
                task_class IN ('conversation', 'coding', 'vision')
            ),
            backend_option_id TEXT CHECK (
                backend_option_id IS NULL
                OR length(trim(backend_option_id)) BETWEEN 1 AND 128
            ),
            fallback TEXT NOT NULL CHECK (
                fallback IN ('selected', 'fail_closed')
            ),
            revision INTEGER NOT NULL CHECK (revision > 0),
            updated_at TEXT NOT NULL,
            PRIMARY KEY (runtime_profile_id, task_class)
        )
        """,
        """
        CREATE TABLE workshop_run_routing_decisions (
            run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
            runtime_profile_id TEXT NOT NULL,
            requested_task_class TEXT CHECK (
                requested_task_class IS NULL
                OR requested_task_class IN ('conversation', 'coding', 'vision')
            ),
            requested_backend_option_id TEXT,
            selected_backend_option_id TEXT,
            disposition TEXT NOT NULL CHECK (
                disposition IN (
                    'selected_default', 'routed', 'fallback_selected', 'rejected'
                )
            ),
            reason_code TEXT NOT NULL CHECK (
                length(trim(reason_code)) BETWEEN 1 AND 64
            ),
            policy_revision INTEGER CHECK (
                policy_revision IS NULL OR policy_revision > 0
            ),
            backend TEXT NOT NULL CHECK (length(trim(backend)) > 0),
            provider TEXT,
            model TEXT NOT NULL CHECK (length(trim(model)) > 0),
            evidence_version INTEGER NOT NULL CHECK (evidence_version > 0),
            decided_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX workshop_run_routing_decisions_profile_idx "
        "ON workshop_run_routing_decisions (runtime_profile_id, decided_at)",
        "CREATE INDEX workshop_run_routing_decisions_disposition_idx "
        "ON workshop_run_routing_decisions (disposition, decided_at)",
    ),
)

_MESSAGE_REACTIONS_SCHEMA = SchemaMigration(
    version=45,
    name="canonical_message_reactions",
    statements=(
        """
        CREATE TABLE message_reactions (
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            reaction TEXT NOT NULL CHECK (
                reaction IN ('thumbs_up', 'heart', 'laugh', 'celebrate', 'eyes', 'check')
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            PRIMARY KEY (message_id, principal_id, reaction)
        )
        """,
        "CREATE INDEX message_reactions_message_idx ON message_reactions "
        "(message_id, reaction, created_event_position)",
    ),
)

_VERSIONED_AGENT_DEFINITION_SCHEMA = SchemaMigration(
    version=46,
    name="versioned_agent_definitions",
    statements=(
        """
        CREATE TABLE agent_definitions (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL UNIQUE REFERENCES agents(id) ON DELETE CASCADE,
            handle TEXT NOT NULL COLLATE NOCASE,
            display_name TEXT NOT NULL,
            description TEXT NOT NULL,
            presentation_json TEXT NOT NULL CHECK (
                json_valid(presentation_json)
                AND json_type(presentation_json) = 'object'
            ),
            lifecycle_state TEXT NOT NULL CHECK (
                lifecycle_state IN ('draft', 'active', 'archived')
            ),
            active_revision_id TEXT
                REFERENCES agent_definition_revisions(id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (workshop_id, handle)
        )
        """,
        """
        CREATE TABLE agent_definition_revisions (
            id TEXT PRIMARY KEY,
            agent_definition_id TEXT NOT NULL
                REFERENCES agent_definitions(id) ON DELETE CASCADE,
            revision_number INTEGER NOT NULL CHECK (revision_number > 0),
            purpose TEXT NOT NULL,
            instructions TEXT NOT NULL,
            capabilities_json TEXT NOT NULL CHECK (
                json_valid(capabilities_json)
                AND json_type(capabilities_json) = 'array'
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (agent_definition_id, revision_number)
        )
        """,
        "CREATE UNIQUE INDEX agent_definitions_active_revision_idx "
        "ON agent_definitions (active_revision_id) WHERE active_revision_id IS NOT NULL",
        "ALTER TABLE runs ADD COLUMN agent_definition_revision_id TEXT "
        "REFERENCES agent_definition_revisions(id) ON DELETE RESTRICT",
        "CREATE INDEX runs_agent_definition_revision_idx ON runs (agent_definition_revision_id, accepted_at)",
    ),
)

_PRINCIPAL_AGENT_ENABLEMENT_SCHEMA = SchemaMigration(
    version=47,
    name="principal_agent_enablement",
    statements=(
        """
        CREATE TABLE principal_agent_enablements (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            agent_definition_id TEXT NOT NULL
                REFERENCES agent_definitions(id) ON DELETE CASCADE,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
            direct_channel_id TEXT NOT NULL UNIQUE
                REFERENCES channels(id) ON DELETE RESTRICT,
            runtime_profile_id TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL CHECK (
                lifecycle_state IN ('enabled', 'disabled')
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (principal_id, agent_definition_id)
        )
        """,
        "CREATE INDEX principal_agent_enablements_runtime_idx "
        "ON principal_agent_enablements (runtime_profile_id, principal_id)",
    ),
)

_CHANNEL_AGENT_ATTACHMENT_LIFECYCLE_SCHEMA = SchemaMigration(
    version=48,
    name="channel_agent_attachment_lifecycle",
    statements=(
        "ALTER TABLE channel_agents ADD COLUMN sponsor_principal_id TEXT REFERENCES principals(id) ON DELETE RESTRICT",
        "ALTER TABLE channel_agents ADD COLUMN sponsored_runtime_profile_id TEXT",
        "ALTER TABLE channel_agents ADD COLUMN attached_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
        "ALTER TABLE channel_agents ADD COLUMN detached_at TEXT",
        "ALTER TABLE channel_agents ADD COLUMN detached_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
        "UPDATE channel_agents SET attached_event_position = ("
        "SELECT MAX(position) FROM event_log "
        "WHERE aggregate_type = 'channel_agent' AND aggregate_id = channel_agents.id "
        "AND event_type = 'channel.agent_attached')",
        "UPDATE channel_agents SET sponsored_runtime_profile_id = ("
        "SELECT runtime_profile_id FROM channel_agent_runtime_assignments ra "
        "WHERE ra.channel_id = channel_agents.channel_id "
        "AND ra.agent_id = channel_agents.agent_id)",
        "UPDATE channel_agents SET sponsor_principal_id = COALESCE(("
        "SELECT owner.principal_id FROM channel_agent_runtime_assignments current_ra "
        "JOIN channel_agent_runtime_assignments direct_ra "
        "ON direct_ra.runtime_profile_id = current_ra.runtime_profile_id "
        "AND direct_ra.agent_id = current_ra.agent_id "
        "JOIN channels direct_channel ON direct_channel.id = direct_ra.channel_id "
        "AND direct_channel.kind = 'direct' "
        "JOIN channel_memberships owner ON owner.channel_id = direct_channel.id "
        "AND owner.role = 'owner' "
        "WHERE current_ra.channel_id = channel_agents.channel_id "
        "AND current_ra.agent_id = channel_agents.agent_id "
        "ORDER BY direct_ra.created_event_position LIMIT 1), ("
        "SELECT migrated.principal_id FROM workshop_execution_state_migrations migrated "
        "WHERE migrated.runtime_profile_id = channel_agents.sponsored_runtime_profile_id "
        "LIMIT 1))",
        "ALTER TABLE runs ADD COLUMN runtime_profile_id TEXT",
        "ALTER TABLE runs ADD COLUMN sponsor_principal_id TEXT REFERENCES principals(id) ON DELETE RESTRICT",
        "UPDATE runs SET runtime_profile_id = ("
        "SELECT runtime_profile_id FROM channel_agent_runtime_assignments ra "
        "WHERE ra.channel_id = runs.channel_id AND ra.agent_id = runs.agent_id)",
        "UPDATE runs SET sponsor_principal_id = ("
        "SELECT sponsor_principal_id FROM channel_agents ca "
        "WHERE ca.channel_id = runs.channel_id AND ca.agent_id = runs.agent_id)",
        "CREATE INDEX channel_agents_active_idx ON channel_agents (channel_id, agent_id) WHERE detached_at IS NULL",
    ),
)

_BOUNDED_AGENT_DELEGATION_SCHEMA = SchemaMigration(
    version=49,
    name="bounded_agent_delegation",
    statements=(
        "ALTER TABLE runs ADD COLUMN parent_run_id TEXT REFERENCES runs(id) ON DELETE RESTRICT",
        "ALTER TABLE runs ADD COLUMN delegation_id TEXT",
        "CREATE UNIQUE INDEX runs_delegation_idx ON runs (delegation_id) WHERE delegation_id IS NOT NULL",
        "CREATE INDEX runs_parent_run_idx ON runs (parent_run_id, accepted_at) WHERE parent_run_id IS NOT NULL",
        """
        CREATE TABLE agent_delegations (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            thread_root_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            root_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
            parent_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
            parent_delegation_id TEXT REFERENCES agent_delegations(id) ON DELETE RESTRICT,
            child_run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE RESTRICT,
            requesting_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            caller_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
            target_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
            caller_sponsor_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            caller_runtime_profile_id TEXT NOT NULL,
            target_sponsor_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            target_runtime_profile_id TEXT NOT NULL,
            caller_definition_revision_id TEXT NOT NULL
                REFERENCES agent_definition_revisions(id) ON DELETE RESTRICT,
            target_definition_revision_id TEXT NOT NULL
                REFERENCES agent_definition_revisions(id) ON DELETE RESTRICT,
            request_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id) ON DELETE RESTRICT,
            response_message_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            task TEXT NOT NULL,
            context_json TEXT NOT NULL CHECK (
                json_valid(context_json) AND json_type(context_json) = 'object'
            ),
            request_hash TEXT NOT NULL,
            depth INTEGER NOT NULL CHECK (depth > 0),
            status TEXT NOT NULL CHECK (
                status IN ('requested', 'executing', 'completed', 'failed', 'cancelled')
            ),
            outcome_code TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            terminal_at TEXT,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT
        )
        """,
        "CREATE INDEX agent_delegations_root_idx ON agent_delegations (root_run_id, created_event_position)",
        "CREATE INDEX agent_delegations_parent_idx ON agent_delegations (parent_run_id, created_event_position)",
        "CREATE INDEX agent_delegations_active_idx "
        "ON agent_delegations (status, created_event_position) "
        "WHERE status IN ('requested', 'executing')",
    ),
)

_REVERSIBLE_CHANNEL_ARCHIVAL_SCHEMA = SchemaMigration(
    version=50,
    name="reversible_channel_archival",
    statements=(
        "ALTER TABLE channels ADD COLUMN archived_at TEXT",
        "ALTER TABLE channels ADD COLUMN lifecycle_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
        "CREATE INDEX channels_archived_idx ON channels (workshop_id, archived_at, kind, name)",
    ),
)

_CANONICAL_HUMAN_HANDLE_SCHEMA = SchemaMigration(
    version=51,
    name="canonical_human_handles",
    statements=(
        """
        CREATE TABLE human_handles (
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            handle TEXT NOT NULL COLLATE NOCASE CHECK (
                length(handle) BETWEEN 1 AND 32
                AND handle NOT GLOB '*[^a-z0-9_]*'
                AND substr(handle, 1, 1) GLOB '[a-z]'
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            PRIMARY KEY (workshop_id, principal_id),
            UNIQUE (workshop_id, handle)
        )
        """,
        "CREATE INDEX human_handles_principal_idx ON human_handles (principal_id, workshop_id)",
    ),
)

_MULTI_HUMAN_CHANNEL_MEMBERSHIP_SCHEMA = SchemaMigration(
    version=52,
    name="multi_human_channel_membership",
    statements=(
        "ALTER TABLE channels ADD COLUMN membership_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
    ),
)

_CANONICAL_HUMAN_NOTIFICATION_SCHEMA = SchemaMigration(
    version=53,
    name="canonical_human_notifications",
    statements=(
        """
        CREATE TABLE human_notifications (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            recipient_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind = 'mention'),
            source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            source_channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            source_thread_root_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            read_at TEXT,
            read_event_position INTEGER REFERENCES event_log(position) ON DELETE RESTRICT,
            state_version INTEGER NOT NULL DEFAULT 0 CHECK (state_version >= 0),
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (recipient_principal_id, source_message_id, kind),
            CHECK (
                (read_at IS NULL AND read_event_position IS NULL)
                OR (read_at IS NOT NULL AND read_event_position IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX human_notifications_recipient_inbox_idx "
        "ON human_notifications (recipient_principal_id, created_event_position DESC, id DESC)",
        "CREATE INDEX human_notifications_recipient_unread_idx "
        "ON human_notifications (recipient_principal_id, created_event_position DESC) "
        "WHERE read_at IS NULL",
        "CREATE INDEX human_notifications_channel_idx "
        "ON human_notifications (source_channel_id, recipient_principal_id)",
    ),
)

_CHANNEL_NOTIFICATION_POLICY_SCHEMA = SchemaMigration(
    version=54,
    name="principal_channel_notification_policy",
    statements=(
        """
        CREATE TABLE principal_human_notification_policies (
            principal_id TEXT PRIMARY KEY REFERENCES principals(id) ON DELETE CASCADE,
            muted_mentions_notify INTEGER NOT NULL DEFAULT 1
                CHECK (muted_mentions_notify IN (0, 1)),
            dnd_enabled INTEGER NOT NULL DEFAULT 0 CHECK (dnd_enabled IN (0, 1)),
            dnd_timezone TEXT NOT NULL DEFAULT 'UTC' CHECK (length(dnd_timezone) BETWEEN 1 AND 100),
            dnd_start_minute INTEGER NOT NULL DEFAULT 1320
                CHECK (dnd_start_minute BETWEEN 0 AND 1439),
            dnd_end_minute INTEGER NOT NULL DEFAULT 420
                CHECK (dnd_end_minute BETWEEN 0 AND 1439),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            CHECK (dnd_start_minute != dnd_end_minute)
        )
        """,
        """
        CREATE TABLE principal_channel_notification_policies (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            level TEXT NOT NULL CHECK (level IN ('all', 'mentions_replies', 'muted')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (principal_id, channel_id)
        )
        """,
        "CREATE INDEX principal_channel_notification_policies_channel_idx "
        "ON principal_channel_notification_policies (channel_id, principal_id)",
        "ALTER TABLE human_notifications RENAME TO human_notifications_v53",
        """
        CREATE TABLE human_notifications (
            id TEXT PRIMARY KEY,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            recipient_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('mention', 'reply', 'message')),
            source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            source_channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            source_thread_root_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            read_at TEXT,
            read_event_position INTEGER REFERENCES event_log(position) ON DELETE RESTRICT,
            state_version INTEGER NOT NULL DEFAULT 0 CHECK (state_version >= 0),
            last_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            UNIQUE (recipient_principal_id, source_message_id),
            CHECK (
                (read_at IS NULL AND read_event_position IS NULL)
                OR (read_at IS NOT NULL AND read_event_position IS NOT NULL)
            )
        )
        """,
        "INSERT INTO human_notifications SELECT * FROM human_notifications_v53",
        "DROP TABLE human_notifications_v53",
        "CREATE INDEX human_notifications_recipient_inbox_idx "
        "ON human_notifications (recipient_principal_id, created_event_position DESC, id DESC)",
        "CREATE INDEX human_notifications_recipient_unread_idx "
        "ON human_notifications (recipient_principal_id, created_event_position DESC) "
        "WHERE read_at IS NULL",
        "CREATE INDEX human_notifications_channel_idx "
        "ON human_notifications (source_channel_id, recipient_principal_id)",
    ),
)

_HUMAN_NOTIFICATION_PUBLICATION_SCHEMA = SchemaMigration(
    version=55,
    name="canonical_human_notification_publication",
    statements=(
        """
        CREATE TABLE human_notification_publications (
            notification_id TEXT PRIMARY KEY
                REFERENCES human_notifications(id) ON DELETE CASCADE,
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            recipient_principal_id TEXT NOT NULL
                REFERENCES principals(id) ON DELETE CASCADE,
            source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            source_channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            source_thread_root_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            policy_result TEXT NOT NULL CHECK (
                policy_result IN ('eligible', 'suppressed_dnd')
            ),
            alert_body TEXT NOT NULL CHECK (length(alert_body) BETWEEN 1 AND 2000),
            deep_link TEXT CHECK (deep_link IS NULL OR length(deep_link) BETWEEN 1 AND 1000),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT
        )
        """,
        "CREATE INDEX human_notification_publications_recipient_idx "
        "ON human_notification_publications (recipient_principal_id, created_event_position)",
        "CREATE INDEX human_notification_publications_policy_idx "
        "ON human_notification_publications (policy_result, created_event_position)",
        # The durable outbox deliberately survives projection rebuilds, so it
        # cannot hold a foreign key into a replayed projection table.
        "ALTER TABLE delivery_outbox ADD COLUMN human_notification_id TEXT",
        "CREATE INDEX delivery_outbox_human_notification_idx "
        "ON delivery_outbox (human_notification_id, status, requested_event_position)",
    ),
)

_HUMAN_NOTIFICATION_ADAPTER_PREFERENCE_SCHEMA = SchemaMigration(
    version=56,
    name="principal_human_notification_adapter_preference",
    statements=(
        """
        CREATE TABLE principal_human_notification_adapter_preferences (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            transport TEXT NOT NULL CHECK (
                length(transport) BETWEEN 1 AND 32
                AND transport NOT GLOB '*[^a-z0-9_]*'
                AND transport GLOB '[a-z]*'
            ),
            enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (principal_id, transport)
        )
        """,
        "CREATE INDEX principal_human_notification_adapter_preferences_transport_idx "
        "ON principal_human_notification_adapter_preferences (transport, enabled, principal_id)",
        """
        CREATE TABLE human_notification_adapter_delivery_decisions (
            notification_id TEXT NOT NULL
                REFERENCES human_notification_publications(notification_id) ON DELETE CASCADE,
            transport TEXT NOT NULL CHECK (
                length(transport) BETWEEN 1 AND 32
                AND transport NOT GLOB '*[^a-z0-9_]*'
                AND transport GLOB '[a-z]*'
            ),
            policy_result TEXT NOT NULL CHECK (
                policy_result IN ('eligible', 'suppressed_preference')
            ),
            created_event_position INTEGER NOT NULL
                REFERENCES event_log(position) ON DELETE RESTRICT,
            PRIMARY KEY (notification_id, transport)
        )
        """,
        "CREATE INDEX human_notification_adapter_delivery_decisions_policy_idx "
        "ON human_notification_adapter_delivery_decisions "
        "(transport, policy_result, created_event_position)",
    ),
)

_SINGLE_OWNER_AGENT_AUTHORITY_SCHEMA = SchemaMigration(
    version=57,
    name="single_owner_agent_authority",
    statements=(
        "ALTER TABLE agent_definitions ADD COLUMN owner_principal_id TEXT REFERENCES principals(id) ON DELETE RESTRICT",
        "ALTER TABLE agent_definitions ADD COLUMN owner_runtime_profile_id TEXT",
        "ALTER TABLE agent_definitions ADD COLUMN owner_direct_channel_id TEXT "
        "REFERENCES channels(id) ON DELETE RESTRICT",
        "ALTER TABLE agent_definitions ADD COLUMN authority_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
        "CREATE INDEX agent_definitions_owner_idx ON agent_definitions (owner_principal_id, lifecycle_state, handle)",
        "CREATE UNIQUE INDEX agent_definitions_owner_lane_idx "
        "ON agent_definitions (owner_direct_channel_id) "
        "WHERE owner_direct_channel_id IS NOT NULL",
        """
        CREATE TABLE runtime_profile_owners (
            runtime_profile_id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL,
            captured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
        "CREATE INDEX runtime_profile_owners_principal_idx "
        "ON runtime_profile_owners (principal_id, runtime_profile_id)",
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) "
        "SELECT ra.runtime_profile_id, MIN(cm.principal_id) "
        "FROM channel_agent_runtime_assignments ra "
        "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
        "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
        "GROUP BY ra.runtime_profile_id HAVING COUNT(DISTINCT cm.principal_id) = 1",
    ),
)

_EXPLICIT_AGENT_CONVERSATION_SCHEMA = SchemaMigration(
    version=58,
    name="explicit_agent_conversation",
    statements=(
        "ALTER TABLE principal_agent_enablements ADD COLUMN conversation_started_at TEXT",
        "ALTER TABLE principal_agent_enablements ADD COLUMN conversation_started_event_position INTEGER "
        "REFERENCES event_log(position) ON DELETE RESTRICT",
        "UPDATE principal_agent_enablements SET "
        "conversation_started_at = (SELECT MIN(m.created_at) FROM messages m "
        "WHERE m.channel_id = principal_agent_enablements.direct_channel_id), "
        "conversation_started_event_position = (SELECT MIN(m.created_event_position) FROM messages m "
        "WHERE m.channel_id = principal_agent_enablements.direct_channel_id) "
        "WHERE EXISTS (SELECT 1 FROM messages m "
        "WHERE m.channel_id = principal_agent_enablements.direct_channel_id)",
        "UPDATE principal_agent_enablements SET lifecycle_state = 'disabled', "
        "updated_at = COALESCE((SELECT occurred_at FROM event_log e "
        "WHERE e.aggregate_id = principal_agent_enablements.agent_definition_id "
        "AND e.event_type = 'agent_definition.archived' ORDER BY e.position DESC LIMIT 1), updated_at) "
        "WHERE lifecycle_state = 'enabled' AND EXISTS (SELECT 1 FROM agent_definitions d "
        "WHERE d.id = principal_agent_enablements.agent_definition_id "
        "AND d.lifecycle_state = 'archived')",
        "CREATE INDEX principal_agent_enablements_conversation_idx "
        "ON principal_agent_enablements (principal_id, conversation_started_at, direct_channel_id)",
    ),
)

_OWNER_RUNTIME_SESSION_RECONCILIATION_SCHEMA = SchemaMigration(
    version=59,
    name="owner_runtime_session_reconciliation",
    statements=(
        # Provider sessions are resumable only under the runtime authority that
        # created them.  The single-owner cutover changed that authority for
        # non-owner direct lanes, so retire any historical session that no
        # longer matches the effective owner/sponsored/assignment profile.
        "DELETE FROM channel_agent_runtime_sessions AS s WHERE NOT EXISTS ("
        "SELECT 1 FROM channel_agents ca "
        "JOIN channels c ON c.id = ca.channel_id "
        "JOIN agent_definitions d ON d.agent_id = ca.agent_id "
        "LEFT JOIN channel_agent_runtime_assignments a ON a.channel_id = ca.channel_id "
        "AND a.agent_id = ca.agent_id "
        "LEFT JOIN principal_agent_enablements e ON e.direct_channel_id = ca.channel_id "
        "AND e.agent_id = ca.agent_id "
        "WHERE ca.channel_id = s.channel_id AND ca.agent_id = s.agent_id "
        "AND ca.detached_at IS NULL AND d.lifecycle_state = 'active' "
        "AND (c.kind != 'direct' OR e.id IS NULL OR e.lifecycle_state = 'enabled') "
        "AND COALESCE(d.owner_runtime_profile_id, ca.sponsored_runtime_profile_id, "
        "a.runtime_profile_id) = s.runtime_profile_id)",
    ),
)

_CANONICAL_CHANNEL_READ_POSITION_SCHEMA = SchemaMigration(
    version=60,
    name="canonical_channel_read_positions",
    statements=(
        # This cutover receipt deliberately survives projection rebuilds.  It
        # prevents pre-authority history from becoming unread when the event
        # projection is replayed from position zero.
        """
        CREATE TABLE channel_unread_migration_baselines (
            principal_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            baseline_event_position INTEGER NOT NULL CHECK (baseline_event_position >= 0),
            captured_at TEXT NOT NULL,
            PRIMARY KEY (principal_id, channel_id)
        )
        """,
        "INSERT INTO channel_unread_migration_baselines "
        "(principal_id, channel_id, baseline_event_position, captured_at) "
        "SELECT cm.principal_id, cm.channel_id, "
        "(SELECT COALESCE(MAX(position), 0) FROM event_log), "
        "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        "FROM channel_memberships cm JOIN principals p ON p.id = cm.principal_id "
        "AND p.kind = 'human'",
        """
        CREATE TABLE channel_read_positions (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            membership_baseline_event_position INTEGER NOT NULL
                CHECK (membership_baseline_event_position >= 0),
            read_through_event_position INTEGER NOT NULL CHECK (read_through_event_position >= 0),
            read_through_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            state_version INTEGER NOT NULL DEFAULT 0 CHECK (state_version >= 0),
            last_event_position INTEGER NOT NULL CHECK (last_event_position >= 0),
            updated_at TEXT NOT NULL,
            PRIMARY KEY (principal_id, channel_id),
            CHECK (read_through_event_position >= membership_baseline_event_position)
        )
        """,
        "CREATE INDEX channel_read_positions_channel_idx ON channel_read_positions (channel_id, principal_id)",
        "INSERT INTO channel_read_positions "
        "(principal_id, channel_id, membership_baseline_event_position, "
        "read_through_event_position, read_through_message_id, state_version, "
        "last_event_position, updated_at) "
        "SELECT principal_id, channel_id, baseline_event_position, "
        "baseline_event_position, NULL, 0, baseline_event_position, captured_at "
        "FROM channel_unread_migration_baselines",
    ),
)

_CANONICAL_CHANNEL_READ_POSITION_BOUNDARY_REPAIR_SCHEMA = SchemaMigration(
    version=61,
    name="canonical_channel_read_position_boundary_repair",
    statements=(
        "UPDATE channel_read_positions SET "
        "last_event_position = membership_baseline_event_position, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        "WHERE last_event_position < membership_baseline_event_position",
    ),
)

_CANONICAL_FOLLOWED_THREAD_UNREAD_SCHEMA = SchemaMigration(
    version=62,
    name="canonical_followed_thread_unread",
    statements=(
        # This receipt is deliberately outside the replayed projection. It
        # prevents historical messages from auto-following old threads when
        # canonical projections are rebuilt after this authority is deployed.
        """
        CREATE TABLE thread_unread_authority_cutover (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            baseline_event_position INTEGER NOT NULL CHECK (baseline_event_position >= 0),
            captured_at TEXT NOT NULL
        )
        """,
        "INSERT INTO thread_unread_authority_cutover "
        "(singleton, baseline_event_position, captured_at) VALUES "
        "(1, (SELECT COALESCE(MAX(position), 0) FROM event_log), "
        "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
        """
        CREATE TABLE thread_read_positions (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            thread_root_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            followed INTEGER NOT NULL CHECK (followed IN (0, 1)),
            follow_baseline_event_position INTEGER NOT NULL
                CHECK (follow_baseline_event_position >= 0),
            read_through_event_position INTEGER NOT NULL CHECK (read_through_event_position >= 0),
            read_through_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            state_version INTEGER NOT NULL CHECK (state_version >= 0),
            last_event_position INTEGER NOT NULL CHECK (last_event_position >= 0),
            updated_at TEXT NOT NULL,
            PRIMARY KEY (principal_id, thread_root_id),
            CHECK (read_through_event_position >= follow_baseline_event_position)
        )
        """,
        "CREATE INDEX thread_read_positions_channel_idx ON thread_read_positions (principal_id, channel_id, followed)",
        "CREATE INDEX thread_read_positions_root_idx ON thread_read_positions (thread_root_id, principal_id, followed)",
    ),
)

_PRINCIPAL_DIRECT_MESSAGE_ARCHIVE_SCHEMA = SchemaMigration(
    version=63,
    name="principal_direct_message_archives",
    statements=(
        """
        CREATE TABLE principal_direct_message_archives (
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            archived_at TEXT NOT NULL,
            archived_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            PRIMARY KEY (principal_id, channel_id)
        )
        """,
        "CREATE INDEX principal_direct_message_archives_channel_idx "
        "ON principal_direct_message_archives (channel_id, principal_id)",
    ),
)

_CANONICAL_HUMAN_PROFILE_SCHEMA = SchemaMigration(
    version=64,
    name="canonical_human_profile_display_names",
    statements=(
        "ALTER TABLE principals ADD COLUMN display_name_state_version "
        "INTEGER NOT NULL DEFAULT 0 CHECK (display_name_state_version >= 0)",
        "ALTER TABLE principals ADD COLUMN display_name_event_position "
        "INTEGER REFERENCES event_log(position) ON DELETE RESTRICT",
    ),
)

_CANONICAL_HUMAN_AVATAR_SCHEMA = SchemaMigration(
    version=65,
    name="canonical_human_profile_avatars",
    statements=(
        """
        CREATE TABLE principal_avatars (
            principal_id TEXT PRIMARY KEY REFERENCES principals(id) ON DELETE CASCADE,
            state_version INTEGER NOT NULL CHECK (state_version > 0),
            active INTEGER NOT NULL CHECK (active IN (0, 1)),
            media_type TEXT,
            byte_size INTEGER CHECK (byte_size IS NULL OR byte_size > 0),
            width INTEGER CHECK (width IS NULL OR width > 0),
            height INTEGER CHECK (height IS NULL OR height > 0),
            sha256 TEXT,
            event_position INTEGER NOT NULL UNIQUE REFERENCES event_log(position) ON DELETE RESTRICT,
            CHECK (
                (active = 1 AND media_type = 'image/png' AND byte_size IS NOT NULL
                    AND width IS NOT NULL AND height IS NOT NULL AND length(sha256) = 64)
                OR
                (active = 0 AND media_type IS NULL AND byte_size IS NULL
                    AND width IS NULL AND height IS NULL AND sha256 IS NULL)
            )
        )
        """,
    ),
)

_EXPANDED_MESSAGE_REACTIONS_SCHEMA = SchemaMigration(
    version=66,
    name="expanded_message_reactions",
    statements=(
        "ALTER TABLE message_reactions RENAME TO message_reactions_v66",
        """
        CREATE TABLE message_reactions (
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE CASCADE,
            reaction TEXT NOT NULL CHECK (
                reaction IN (
                    'thumbs_up', 'thumbs_down', 'heart', 'laugh', 'celebrate',
                    'eyes', 'check', 'thinking', 'surprised', 'sad', 'fire', 'question'
                )
            ),
            created_at TEXT NOT NULL,
            created_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            PRIMARY KEY (message_id, principal_id, reaction)
        )
        """,
        "INSERT INTO message_reactions "
        "(message_id, principal_id, reaction, created_at, created_event_position) "
        "SELECT message_id, principal_id, reaction, created_at, created_event_position "
        "FROM message_reactions_v66",
        "DROP TABLE message_reactions_v66",
        "CREATE INDEX message_reactions_message_idx ON message_reactions "
        "(message_id, reaction, created_event_position)",
    ),
)

_ATTEMPT_SCOPED_COLLABORATION_AUTHORITY_SCHEMA = SchemaMigration(
    version=67,
    name="attempt_scoped_collaboration_authority",
    statements=(
        "ALTER TABLE agent_definition_revisions ADD COLUMN collaboration_operations_json "
        "TEXT NOT NULL DEFAULT '[]' CHECK ("
        "json_valid(collaboration_operations_json) "
        "AND json_type(collaboration_operations_json) = 'array')",
        "UPDATE agent_definition_revisions SET collaboration_operations_json = "
        "'[\"agent_delegation\"]' WHERE EXISTS (SELECT 1 FROM json_each(capabilities_json) "
        "WHERE value = 'agent_delegation')",
        """
        CREATE TABLE collaboration_grants (
            id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE REFERENCES run_attempts(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            execution_owner_id TEXT NOT NULL,
            fence_token INTEGER NOT NULL CHECK (fence_token > 0),
            workshop_id TEXT NOT NULL REFERENCES workshops(id) ON DELETE CASCADE,
            requested_by_principal_id TEXT NOT NULL
                REFERENCES principals(id) ON DELETE RESTRICT,
            agent_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
            agent_definition_revision_id TEXT NOT NULL
                REFERENCES agent_definition_revisions(id) ON DELETE RESTRICT,
            sponsor_principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
            runtime_profile_id TEXT NOT NULL,
            channel_id TEXT NOT NULL REFERENCES channels(id) ON DELETE RESTRICT,
            thread_root_id TEXT REFERENCES messages(id) ON DELETE RESTRICT,
            requested_operations_json TEXT NOT NULL CHECK (
                json_valid(requested_operations_json)
                AND json_type(requested_operations_json) = 'array'
            ),
            owner_allowed_operations_json TEXT NOT NULL CHECK (
                json_valid(owner_allowed_operations_json)
                AND json_type(owner_allowed_operations_json) = 'array'
            ),
            host_allowed_operations_json TEXT NOT NULL CHECK (
                json_valid(host_allowed_operations_json)
                AND json_type(host_allowed_operations_json) = 'array'
            ),
            effective_operations_json TEXT NOT NULL CHECK (
                json_valid(effective_operations_json)
                AND json_type(effective_operations_json) = 'array'
            ),
            owner_policy_version INTEGER NOT NULL CHECK (owner_policy_version >= 0),
            host_policy_version INTEGER NOT NULL CHECK (host_policy_version > 0),
            quotas_json TEXT NOT NULL CHECK (
                json_valid(quotas_json) AND json_type(quotas_json) = 'object'
            ),
            proof_fingerprint TEXT NOT NULL UNIQUE CHECK (length(proof_fingerprint) = 64),
            issued_at TEXT NOT NULL,
            initial_lease_expires_at TEXT NOT NULL,
            issued_event_position INTEGER NOT NULL UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            revoked_at TEXT,
            revocation_code TEXT,
            revoked_event_position INTEGER UNIQUE
                REFERENCES event_log(position) ON DELETE RESTRICT,
            state_version INTEGER NOT NULL DEFAULT 0 CHECK (state_version >= 0),
            CHECK (
                (revoked_at IS NULL AND revocation_code IS NULL AND revoked_event_position IS NULL)
                OR
                (revoked_at IS NOT NULL AND revocation_code IS NOT NULL
                    AND revoked_event_position IS NOT NULL)
            )
        )
        """,
        "CREATE INDEX collaboration_grants_run_idx ON collaboration_grants (run_id, issued_event_position)",
        "CREATE INDEX collaboration_grants_active_idx ON collaboration_grants "
        "(attempt_id, revoked_at) WHERE revoked_at IS NULL",
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
    _REUSABLE_GROUP_RUNTIME_ASSIGNMENT_SCHEMA,
    _MESSAGE_MENTIONS_SCHEMA,
    _GROUP_WAKE_POLICY_SCHEMA,
    _MESSAGE_THREADS_SCHEMA,
    _EXPLICIT_TASK_ROUTING_SCHEMA,
    _MESSAGE_REACTIONS_SCHEMA,
    _VERSIONED_AGENT_DEFINITION_SCHEMA,
    _PRINCIPAL_AGENT_ENABLEMENT_SCHEMA,
    _CHANNEL_AGENT_ATTACHMENT_LIFECYCLE_SCHEMA,
    _BOUNDED_AGENT_DELEGATION_SCHEMA,
    _REVERSIBLE_CHANNEL_ARCHIVAL_SCHEMA,
    _CANONICAL_HUMAN_HANDLE_SCHEMA,
    _MULTI_HUMAN_CHANNEL_MEMBERSHIP_SCHEMA,
    _CANONICAL_HUMAN_NOTIFICATION_SCHEMA,
    _CHANNEL_NOTIFICATION_POLICY_SCHEMA,
    _HUMAN_NOTIFICATION_PUBLICATION_SCHEMA,
    _HUMAN_NOTIFICATION_ADAPTER_PREFERENCE_SCHEMA,
    _SINGLE_OWNER_AGENT_AUTHORITY_SCHEMA,
    _EXPLICIT_AGENT_CONVERSATION_SCHEMA,
    _OWNER_RUNTIME_SESSION_RECONCILIATION_SCHEMA,
    _CANONICAL_CHANNEL_READ_POSITION_SCHEMA,
    _CANONICAL_CHANNEL_READ_POSITION_BOUNDARY_REPAIR_SCHEMA,
    _CANONICAL_FOLLOWED_THREAD_UNREAD_SCHEMA,
    _PRINCIPAL_DIRECT_MESSAGE_ARCHIVE_SCHEMA,
    _CANONICAL_HUMAN_PROFILE_SCHEMA,
    _CANONICAL_HUMAN_AVATAR_SCHEMA,
    _EXPANDED_MESSAGE_REACTIONS_SCHEMA,
    _ATTEMPT_SCOPED_COLLABORATION_AUTHORITY_SCHEMA,
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
