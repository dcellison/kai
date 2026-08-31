"""Non-secret operator diagnostics for Workshop transition state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai.workshop.appearance_preferences import WORKSHOP_APPEARANCE_THEMES
from kai.workshop.domain import (
    ChannelId,
    EventEnvelope,
    EventId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    parse_opaque_id,
)
from kai.workshop.human_handles import (
    HUMAN_HANDLE_PATTERN,
    WorkshopHumanHandleError,
    derive_human_handle,
)

_REQUIRED_TABLES = {
    "workshops",
    "principals",
    "workshop_memberships",
    "agents",
    "channel_bindings",
    "channel_agent_runtime_assignments",
    "channel_memberships",
    "projection_checkpoints",
}

_PARITY_TABLES = {
    "channel_memberships",
    "event_log",
    "messages",
    "principals",
    "channel_bindings",
    "workshop_memberships",
}
_TRANSCRIPT_AUTHORITY_TABLES = _PARITY_TABLES | {
    "runs",
    "workshop_transcript_authority",
}
_DELIVERY_AUTHORITY_TABLES = {
    "delivery_authority_epochs",
    "delivery_outbox",
    "delivery_fragments",
}
_RUNTIME_SESSION_TABLES = {
    "channel_agent_runtime_assignments",
    "channel_agent_runtime_sessions",
    "messages",
    "runs",
    "workshop_continuity_cutover",
}
_EXECUTION_STATE_TABLES = {
    "channel_agent_execution_settings",
    "channel_agent_runtime_assignments",
    "channel_agent_workspace_settings",
    "principal_workspace_grants",
    "principal_workspace_history",
    "workshop_execution_state_migrations",
}
_MEMORY_AUTHORITY_TABLES = {
    "channel_agent_runtime_assignments",
    "channel_memberships",
    "channels",
    "principals",
    "workshop_execution_state_migrations",
    "workshop_memory_authority_migrations",
}
_OPERATIONAL_STATE_TABLES = {
    "channel_agent_runtime_assignments",
    "jobs",
    "principal_github_subscriptions",
    "workshop_execution_state_migrations",
    "workshop_job_owners",
    "workshop_operational_state_migrations",
    "workshop_scheduled_job_migrations",
    "workshop_scheduled_jobs",
}
_AGENT_AUTHORITY_TABLES = {
    "agent_definitions",
    "agent_definition_revisions",
    "agent_delegations",
    "agents",
    "channel_agent_runtime_assignments",
    "channel_agents",
    "channel_memberships",
    "channels",
    "principal_agent_enablements",
    "principals",
    "runs",
    "workshop_memberships",
}
_AGENT_HANDLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TELEGRAM_SUBJECT_PATTERN = re.compile(r"^-?[0-9]+$")
_SYNTHETIC_ASSISTANT_PATTERN = re.compile(
    r"\[(stopped by user|no response|error: .+)\]",
    re.DOTALL,
)
_SCHEDULED_ASSISTANT_PREFIXES = ("[Reminder:", "[Job:")


@dataclass(frozen=True, slots=True)
class _ReplayMessage:
    message_id: str
    channel_id: str
    author_principal_id: str
    reply_to_message_id: str | None
    body: str
    event_position: int
    created_at: str
    direction: str
    source: str

    @property
    def legacy_key(self) -> tuple[str, str]:
        return self.direction, hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ReplayState:
    principal_kinds: dict[str, str]
    channel_kinds: dict[str, str]
    workshop_memberships: dict[str, tuple[str, str, str]]
    channel_bindings: dict[str, tuple[str, str, str]]
    channel_memberships: dict[str, tuple[str, str, str]]
    runtime_assignments: dict[str, tuple[str, str, str]]
    messages: tuple[_ReplayMessage, ...]


def _pending_status(expected_humans: int | None) -> str:
    if expected_humans is None:
        return "Workshop bootstrap: pending; configured-human count unavailable"
    return (
        "Workshop bootstrap: pending; service startup will seed "
        f"1 workshop, {expected_humans} human principal(s), "
        f"{expected_humans} direct channel(s), runtime assignment(s), and 1 Kai agent"
    )


def _scalar(connection: sqlite3.Connection, query: str, parameters: tuple[object, ...] = ()) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        return 0
    return int(row[0])


def workshop_bootstrap_status(db_path: Path, *, expected_humans: int | None) -> str:
    """Describe canonical bootstrap state without exposing identity data."""
    if not db_path.is_file():
        return _pending_status(expected_humans)

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            tables = {str(row[0]) for row in rows}
            if not tables >= _REQUIRED_TABLES:
                return _pending_status(expected_humans)

            workshop_count = _scalar(connection, "SELECT COUNT(*) FROM workshops")
            human_count = _scalar(connection, "SELECT COUNT(*) FROM principals WHERE kind = 'human'")
            agent_count = _scalar(connection, "SELECT COUNT(*) FROM agents")
            telegram_binding_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = ?",
                ("telegram",),
            )
            channel_membership_count = _scalar(connection, "SELECT COUNT(*) FROM channel_memberships")
            runtime_assignment_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments",
            )
            projection_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM projection_checkpoints WHERE name = ?",
                ("canonical_conversations",),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return f"Workshop bootstrap: NOT VERIFIED ({type(exc).__name__})"

    expected_memberships = 2 * (human_count if expected_humans is None else expected_humans)
    expected_state_present = (
        expected_humans is None or (human_count >= expected_humans and runtime_assignment_count >= expected_humans)
    ) and channel_membership_count >= expected_memberships
    initialized = workshop_count >= 1 and agent_count >= 1 and projection_count == 1 and expected_state_present
    state = "initialized" if initialized else "pending"
    expectation = (
        "configured-human count unavailable" if expected_humans is None else f"expected humans={expected_humans}"
    )
    return (
        f"Workshop bootstrap: {state}; workshops={workshop_count}, humans={human_count}, "
        f"Telegram bindings={telegram_binding_count}, channel memberships={channel_membership_count}, "
        f"agents={agent_count}, runtime assignments={runtime_assignment_count}; {expectation}"
    )


def workshop_agent_authority_status(db_path: Path) -> str:
    """Describe canonical multi-agent authority and integrity without exposing content."""
    prefix = "Workshop agent authority:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _AGENT_AUTHORITY_TABLES:
                return f"{prefix} pending; canonical schema unavailable"
            definitions = _scalar(connection, "SELECT COUNT(*) FROM agent_definitions")
            revisions = _scalar(connection, "SELECT COUNT(*) FROM agent_definition_revisions")
            active = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions WHERE lifecycle_state = 'active'",
            )
            drafts = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions WHERE lifecycle_state = 'draft'",
            )
            archived = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions WHERE lifecycle_state = 'archived'",
            )
            missing_definitions = _scalar(
                connection,
                "SELECT COUNT(*) FROM agents a LEFT JOIN agent_definitions d ON d.agent_id = a.id WHERE d.id IS NULL",
            )
            missing_revisions = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions d WHERE NOT EXISTS ("
                "SELECT 1 FROM agent_definition_revisions r WHERE r.agent_definition_id = d.id)",
            )
            stale_active_revisions = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions d LEFT JOIN agent_definition_revisions r "
                "ON r.id = d.active_revision_id AND r.agent_definition_id = d.id WHERE "
                "(d.lifecycle_state = 'active' AND r.id IS NULL) OR "
                "(d.lifecycle_state = 'draft' AND d.active_revision_id IS NOT NULL)",
            )
            ambiguous_revisions = _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT d.id, COUNT(r.id) AS revision_count, "
                "COALESCE(MIN(r.revision_number), 0) AS minimum_revision, "
                "COALESCE(MAX(r.revision_number), 0) AS maximum_revision "
                "FROM agent_definitions d LEFT JOIN agent_definition_revisions r "
                "ON r.agent_definition_id = d.id GROUP BY d.id "
                "HAVING revision_count > 0 AND "
                "(minimum_revision != 1 OR maximum_revision != revision_count))",
            )
            orphaned_principals = _scalar(
                connection,
                "SELECT COUNT(*) FROM agents a LEFT JOIN principals p ON p.id = a.principal_id "
                "LEFT JOIN workshop_memberships wm ON wm.principal_id = a.principal_id "
                "AND wm.workshop_id = a.workshop_id WHERE p.kind IS NULL OR p.kind != 'agent' "
                "OR wm.id IS NULL",
            )
            definition_mismatches = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_definitions d LEFT JOIN agents a ON a.id = d.agent_id "
                "WHERE a.id IS NULL OR a.workshop_id != d.workshop_id",
            )
            handles = [str(row[0]) for row in connection.execute("SELECT handle FROM agent_definitions").fetchall()]
            invalid_handles = sum(not _AGENT_HANDLE_PATTERN.fullmatch(handle) for handle in handles)

            enablements = _scalar(connection, "SELECT COUNT(*) FROM principal_agent_enablements")
            enabled = _scalar(
                connection,
                "SELECT COUNT(*) FROM principal_agent_enablements WHERE lifecycle_state = 'enabled'",
            )
            direct_channels = _scalar(
                connection,
                "SELECT COUNT(DISTINCT direct_channel_id) FROM principal_agent_enablements "
                "WHERE lifecycle_state = 'enabled'",
            )
            invalid_enablements = _scalar(
                connection,
                "SELECT COUNT(*) FROM principal_agent_enablements e "
                "LEFT JOIN principals hp ON hp.id = e.principal_id "
                "LEFT JOIN workshop_memberships hwm ON hwm.principal_id = e.principal_id "
                "AND hwm.workshop_id = e.workshop_id "
                "LEFT JOIN agent_definitions d ON d.id = e.agent_definition_id "
                "LEFT JOIN agents a ON a.id = e.agent_id "
                "LEFT JOIN principals ap ON ap.id = a.principal_id "
                "LEFT JOIN channels c ON c.id = e.direct_channel_id WHERE "
                "e.lifecycle_state = 'enabled' AND (hp.kind IS NULL OR hp.kind != 'human' "
                "OR hwm.id IS NULL OR d.id IS NULL OR d.lifecycle_state != 'active' "
                "OR d.active_revision_id IS NULL OR d.agent_id != e.agent_id "
                "OR d.workshop_id != e.workshop_id OR a.workshop_id != e.workshop_id "
                "OR ap.kind IS NULL OR ap.kind != 'agent' OR c.kind IS NULL OR c.kind != 'direct' "
                "OR c.workshop_id != e.workshop_id OR (SELECT COUNT(*) FROM channel_memberships cm "
                "WHERE cm.channel_id = e.direct_channel_id) != 2 "
                "OR NOT EXISTS (SELECT 1 FROM channel_memberships cm WHERE "
                "cm.channel_id = e.direct_channel_id AND cm.principal_id = e.principal_id "
                "AND cm.role = 'owner') OR NOT EXISTS (SELECT 1 FROM channel_memberships cm "
                "WHERE cm.channel_id = e.direct_channel_id AND cm.principal_id = a.principal_id) "
                "OR NOT EXISTS (SELECT 1 FROM channel_agents ca WHERE "
                "ca.channel_id = e.direct_channel_id AND ca.agent_id = e.agent_id "
                "AND ca.detached_at IS NULL) OR NOT EXISTS (SELECT 1 FROM "
                "channel_agent_runtime_assignments ra WHERE ra.channel_id = e.direct_channel_id "
                "AND ra.agent_id = e.agent_id AND ra.runtime_profile_id = e.runtime_profile_id))",
            )
            unauthorized_runtime_bindings = _scalar(
                connection,
                "WITH runtime_owners AS (SELECT ra.runtime_profile_id, "
                "COUNT(DISTINCT owner.principal_id) AS owner_count, "
                "MIN(owner.principal_id) AS owner_id FROM channel_agent_runtime_assignments ra "
                "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
                "JOIN channel_memberships owner ON owner.channel_id = c.id AND owner.role = 'owner' "
                "JOIN principals p ON p.id = owner.principal_id AND p.kind = 'human' "
                "GROUP BY ra.runtime_profile_id) SELECT COUNT(*) "
                "FROM principal_agent_enablements e LEFT JOIN runtime_owners ro "
                "ON ro.runtime_profile_id = e.runtime_profile_id WHERE e.lifecycle_state = 'enabled' "
                "AND (ro.owner_count IS NULL OR ro.owner_count != 1 OR ro.owner_id != e.principal_id)",
            )
            namespace_conflicts = _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT ra.runtime_profile_id FROM "
                "channel_agent_runtime_assignments ra JOIN channels c ON c.id = ra.channel_id "
                "AND c.kind = 'direct' JOIN channel_memberships owner ON owner.channel_id = c.id "
                "AND owner.role = 'owner' JOIN principals p ON p.id = owner.principal_id "
                "AND p.kind = 'human' GROUP BY ra.runtime_profile_id "
                "HAVING COUNT(DISTINCT owner.principal_id) > 1)",
            )

            attachments = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agents ca JOIN channels c ON c.id = ca.channel_id "
                "WHERE c.kind = 'group' AND ca.detached_at IS NULL",
            )
            detached_attachments = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agents ca JOIN channels c ON c.id = ca.channel_id "
                "WHERE c.kind = 'group' AND ca.detached_at IS NOT NULL",
            )
            runtime_sponsorships = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agents WHERE detached_at IS NULL "
                "AND sponsor_principal_id IS NOT NULL AND sponsored_runtime_profile_id IS NOT NULL",
            )
            dangling_attachments = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agents ca JOIN channels c ON c.id = ca.channel_id "
                "LEFT JOIN agent_definitions d ON d.agent_id = ca.agent_id "
                "WHERE c.kind = 'group' AND ca.detached_at IS NULL AND ("
                "d.id IS NULL OR d.workshop_id != c.workshop_id OR d.lifecycle_state != 'active' "
                "OR ca.sponsor_principal_id IS NULL OR ca.sponsored_runtime_profile_id IS NULL "
                "OR NOT EXISTS (SELECT 1 FROM principals p WHERE p.id = ca.sponsor_principal_id "
                "AND p.kind = 'human') OR NOT EXISTS (SELECT 1 FROM channel_memberships owner "
                "WHERE owner.channel_id = ca.channel_id AND owner.principal_id = ca.sponsor_principal_id "
                "AND owner.role = 'owner') OR NOT EXISTS (SELECT 1 FROM "
                "principal_agent_enablements e WHERE e.principal_id = ca.sponsor_principal_id "
                "AND e.agent_id = ca.agent_id AND e.runtime_profile_id = "
                "ca.sponsored_runtime_profile_id AND e.lifecycle_state = 'enabled') "
                "OR NOT EXISTS (SELECT 1 FROM channel_agent_runtime_assignments ra "
                "WHERE ra.channel_id = ca.channel_id AND ra.agent_id = ca.agent_id "
                "AND ra.runtime_profile_id = ca.sponsored_runtime_profile_id))",
            )

            delegations = _scalar(connection, "SELECT COUNT(*) FROM agent_delegations")
            delegation_trees = _scalar(
                connection,
                "SELECT COUNT(DISTINCT root_run_id) FROM agent_delegations",
            )
            nonterminal_delegations = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_delegations WHERE status IN ('requested', 'executing')",
            )
            delegation_gaps = _scalar(
                connection,
                "SELECT COUNT(*) FROM agent_delegations d "
                "LEFT JOIN channels c ON c.id = d.channel_id "
                "LEFT JOIN runs root ON root.id = d.root_run_id "
                "LEFT JOIN runs parent ON parent.id = d.parent_run_id "
                "LEFT JOIN runs child ON child.id = d.child_run_id "
                "LEFT JOIN agent_definition_revisions caller_revision "
                "ON caller_revision.id = d.caller_definition_revision_id "
                "LEFT JOIN agent_definitions caller_definition "
                "ON caller_definition.id = caller_revision.agent_definition_id "
                "LEFT JOIN agent_definition_revisions target_revision "
                "ON target_revision.id = d.target_definition_revision_id "
                "LEFT JOIN agent_definitions target_definition "
                "ON target_definition.id = target_revision.agent_definition_id "
                "LEFT JOIN agent_delegations prior ON prior.id = d.parent_delegation_id WHERE "
                "c.workshop_id IS NULL OR c.workshop_id != d.workshop_id "
                "OR root.workshop_id IS NULL OR root.workshop_id != d.workshop_id "
                "OR parent.workshop_id IS NULL OR parent.workshop_id != d.workshop_id "
                "OR child.workshop_id IS NULL OR child.workshop_id != d.workshop_id "
                "OR parent.channel_id IS NOT d.channel_id OR child.channel_id IS NOT d.channel_id "
                "OR parent.agent_id IS NOT d.caller_agent_id OR child.agent_id IS NOT d.target_agent_id "
                "OR child.parent_run_id IS NOT d.parent_run_id OR child.delegation_id IS NOT d.id "
                "OR parent.sponsor_principal_id IS NOT d.caller_sponsor_principal_id "
                "OR parent.runtime_profile_id IS NOT d.caller_runtime_profile_id "
                "OR child.sponsor_principal_id IS NOT d.target_sponsor_principal_id "
                "OR child.runtime_profile_id IS NOT d.target_runtime_profile_id "
                "OR caller_definition.agent_id IS NULL "
                "OR caller_definition.agent_id != d.caller_agent_id "
                "OR target_definition.agent_id IS NULL "
                "OR target_definition.agent_id != d.target_agent_id "
                "OR (d.depth = 1 AND d.parent_delegation_id IS NOT NULL) "
                "OR (d.depth > 1 AND (prior.id IS NULL OR prior.child_run_id != d.parent_run_id "
                "OR prior.depth + 1 != d.depth))",
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    integrity_gaps = sum(
        (
            missing_definitions,
            missing_revisions,
            stale_active_revisions,
            ambiguous_revisions,
            orphaned_principals,
            definition_mismatches,
            invalid_handles,
            invalid_enablements,
            unauthorized_runtime_bindings,
            namespace_conflicts,
            dangling_attachments,
            delegation_gaps,
        )
    )
    state = "active" if definitions > 0 and integrity_gaps == 0 else "INCOMPLETE"
    return (
        f"{prefix} {state}; definitions={definitions} "
        f"(active={active}, draft={drafts}, archived={archived}), revisions={revisions}, "
        f"enablements={enablements} (enabled={enabled}), direct channels={direct_channels}, "
        f"attachments={attachments} (detached={detached_attachments}), "
        f"runtime sponsorships={runtime_sponsorships}, delegation trees={delegation_trees}, "
        f"delegations={delegations} (nonterminal={nonterminal_delegations}); "
        f"integrity gaps={integrity_gaps} (definitions={missing_definitions + definition_mismatches}, "
        f"missing revisions={missing_revisions}, stale revisions={stale_active_revisions}, "
        f"ambiguous revisions={ambiguous_revisions}, principals={orphaned_principals}, "
        f"handles={invalid_handles}, enablements={invalid_enablements}, "
        f"runtime bindings={unauthorized_runtime_bindings}, namespaces={namespace_conflicts}, "
        f"attachments={dangling_attachments}, delegations={delegation_gaps}); authority=canonical"
    )


def workshop_delivery_authority_status(db_path: Path) -> str:
    """Describe delivery-authority readiness using aggregate, non-secret state."""
    prefix = "Workshop delivery authority:"
    if not db_path.is_file():
        return f"{prefix} pending; authority schema unavailable"

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _DELIVERY_AUTHORITY_TABLES:
                return f"{prefix} pending; authority schema unavailable"

            epoch_count = _scalar(connection, "SELECT COUNT(*) FROM delivery_authority_epochs")
            active_rows = connection.execute(
                "SELECT id FROM delivery_authority_epochs WHERE status = 'active'"
            ).fetchall()
            active_count = len(active_rows)
            unclassified = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_outbox "
                "WHERE purpose = 'conversation_reply' "
                "AND execution_contract = 'streaming_finalization' "
                "AND authority_epoch_id IS NULL",
            )
            active_epoch_id = str(active_rows[0][0]) if active_count == 1 else None
            prior_parameters: tuple[object, ...] = ()
            prior_filter = "authority_epoch_id IS NOT NULL"
            if active_epoch_id is not None:
                prior_filter += " AND authority_epoch_id != ?"
                prior_parameters = (active_epoch_id,)
            prior_nonterminal = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_outbox WHERE "
                f"{prior_filter} AND status IN ('pending', 'leased', 'retry_wait')",
                prior_parameters,
            )
            prior_failed = _scalar(
                connection,
                f"SELECT COUNT(*) FROM delivery_outbox WHERE {prior_filter} AND status = 'failed'",
                prior_parameters,
            )
            prior_uncertain = _scalar(
                connection,
                "SELECT COUNT(DISTINCT o.id) FROM delivery_outbox o "
                "JOIN delivery_fragments f ON f.delivery_id = o.id WHERE "
                f"{prior_filter.replace('authority_epoch_id', 'o.authority_epoch_id')} "
                "AND o.status = 'failed' AND f.status = 'uncertain'",
                prior_parameters,
            )
            unacknowledged_epochs = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_authority_epochs ae "
                "WHERE ae.status = 'deactivated' "
                "AND ae.terminal_failures_acknowledged_at IS NULL "
                "AND EXISTS (SELECT 1 FROM delivery_outbox o "
                "WHERE o.authority_epoch_id = ae.id AND o.status = 'failed')",
            )

            counts = {
                "pending": 0,
                "leased": 0,
                "retrying": 0,
                "succeeded": 0,
                "failed": 0,
                "uncertain": 0,
            }
            if active_epoch_id is not None:
                rows = connection.execute(
                    "SELECT o.status, EXISTS ("
                    "SELECT 1 FROM delivery_fragments f "
                    "WHERE f.delivery_id = o.id AND f.status = 'uncertain'"
                    ") AS uncertain FROM delivery_outbox o WHERE o.authority_epoch_id = ?",
                    (active_epoch_id,),
                ).fetchall()
                for status_value, uncertain_value in rows:
                    status = str(status_value)
                    if status == "failed" and int(uncertain_value):
                        counts["uncertain"] += 1
                    elif status == "retry_wait":
                        counts["retrying"] += 1
                    elif status in counts:
                        counts[status] += 1
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"

    if active_count > 1 or unclassified or prior_nonterminal or unacknowledged_epochs:
        state = "NOT READY"
    elif active_count == 1:
        state = "active"
    else:
        state = "inactive"
    return (
        f"{prefix} {state}; epochs={epoch_count}, unclassified={unclassified}, "
        f"prior nonterminal={prior_nonterminal}, prior failed={prior_failed}, "
        f"prior uncertain={prior_uncertain}, unacknowledged epochs={unacknowledged_epochs}, "
        f"active pending={counts['pending']}, "
        f"leased={counts['leased']}, retrying={counts['retrying']}, "
        f"succeeded={counts['succeeded']}, failed={counts['failed']}, "
        f"uncertain={counts['uncertain']}"
    )


def workshop_runtime_session_status(db_path: Path) -> str:
    """Describe canonical restart-context and runtime-session coverage."""
    prefix = "Workshop conversation continuity:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical runtime-session schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _RUNTIME_SESSION_TABLES:
                return f"{prefix} pending; canonical runtime-session schema unavailable"
            cutover_row = connection.execute(
                "SELECT event_position FROM workshop_continuity_cutover WHERE singleton = 1"
            ).fetchone()
            if cutover_row is None:
                return f"{prefix} pending; canonical runtime-session migration unavailable"
            cutover = int(cutover_row[0])
            successful_lanes = _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT DISTINCT r.channel_id, r.agent_id FROM runs r "
                "JOIN messages m ON m.id = r.result_message_id "
                "WHERE r.status = 'completed' AND m.created_event_position > ?)",
                (cutover,),
            )
            sessions = _scalar(connection, "SELECT COUNT(*) FROM channel_agent_runtime_sessions")
            provider_sessions = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_sessions WHERE provider_session_id IS NOT NULL",
            )
            missing = _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT DISTINCT r.channel_id, r.agent_id FROM runs r "
                "JOIN messages m ON m.id = r.result_message_id "
                "WHERE r.status = 'completed' AND m.created_event_position > ?) lanes "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_sessions s "
                "WHERE s.channel_id = lanes.channel_id AND s.agent_id = lanes.agent_id)",
                (cutover,),
            )
            stale = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_sessions s WHERE "
                "NOT EXISTS (SELECT 1 FROM channel_agent_runtime_assignments a "
                "WHERE a.channel_id = s.channel_id AND a.agent_id = s.agent_id "
                "AND a.runtime_profile_id = s.runtime_profile_id) OR "
                "s.context_through_event_position < COALESCE(("
                "SELECT MAX(m.created_event_position) FROM runs r "
                "JOIN messages m ON m.id = r.result_message_id "
                "WHERE r.channel_id = s.channel_id AND r.agent_id = s.agent_id "
                "AND r.status = 'completed' AND m.created_event_position > ?), 0)",
                (cutover,),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    state = "active" if missing == 0 and stale == 0 else "INCOMPLETE"
    return (
        f"{prefix} {state}; successful lanes={successful_lanes}, sessions={sessions}, "
        f"provider sessions={provider_sessions}, missing={missing}, stale={stale}; "
        "cold-start context=canonical timeline"
    )


def workshop_execution_state_status(db_path: Path) -> str:
    """Describe canonical mutable execution-state authority and backfill."""
    prefix = "Workshop execution state:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical execution-state schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _EXECUTION_STATE_TABLES:
                return f"{prefix} pending; canonical execution-state schema unavailable"
            profiles = _scalar(connection, "SELECT COUNT(*) FROM channel_agent_runtime_assignments")
            migrated = _scalar(connection, "SELECT COUNT(*) FROM workshop_execution_state_migrations")
            missing = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments a WHERE NOT EXISTS ("
                "SELECT 1 FROM workshop_execution_state_migrations m "
                "WHERE m.runtime_profile_id = a.runtime_profile_id AND m.channel_id = a.channel_id "
                "AND m.agent_id = a.agent_id)",
            )
            stale = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_execution_state_migrations m WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a "
                "JOIN channels c ON c.id = a.channel_id AND c.kind = 'direct' "
                "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "WHERE a.runtime_profile_id = m.runtime_profile_id "
                "AND a.channel_id = m.channel_id AND a.agent_id = m.agent_id "
                "AND cm.principal_id = m.principal_id)",
            )
            settings = _scalar(connection, "SELECT COUNT(*) FROM channel_agent_execution_settings")
            workspace_settings = _scalar(connection, "SELECT COUNT(*) FROM channel_agent_workspace_settings")
            history = _scalar(connection, "SELECT COUNT(*) FROM principal_workspace_history")
            grants = _scalar(connection, "SELECT COUNT(*) FROM principal_workspace_grants")
            orphaned = _scalar(
                connection,
                "SELECT (SELECT COUNT(*) FROM channel_agent_execution_settings s WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a WHERE a.channel_id = s.channel_id "
                "AND a.agent_id = s.agent_id AND a.runtime_profile_id = s.runtime_profile_id)) + "
                "(SELECT COUNT(*) FROM channel_agent_workspace_settings s WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a WHERE a.channel_id = s.channel_id "
                "AND a.agent_id = s.agent_id AND a.runtime_profile_id = s.runtime_profile_id))",
            )
            mapped_config_ids = {
                int(row[0])
                for row in connection.execute(
                    "SELECT runtime_config_id FROM workshop_execution_state_migrations"
                ).fetchall()
            }
            unclassified = 0
            for (raw_key,) in connection.execute("SELECT key FROM settings"):
                key = str(raw_key)
                runtime_text: str | None = None
                field, separator, suffix = key.partition(":")
                if separator and field in {"model", "timeout", "workspace"}:
                    runtime_text = suffix
                elif key.startswith("ws_config:"):
                    runtime_text = key[len("ws_config:") :].partition(":")[0]
                if runtime_text and runtime_text.isdigit() and int(runtime_text) > 0:
                    unclassified += int(int(runtime_text) not in mapped_config_ids)
            unclassified += sum(
                int(int(row[0]) not in mapped_config_ids)
                for row in connection.execute("SELECT DISTINCT chat_id FROM workspace_history WHERE chat_id > 0")
            )
            unclassified += sum(
                int(int(row[0]) not in mapped_config_ids)
                for row in connection.execute("SELECT DISTINCT chat_id FROM allowed_workspaces WHERE chat_id > 0")
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    state = (
        "active"
        if profiles > 0 and missing == 0 and stale == 0 and orphaned == 0 and unclassified == 0
        else "INCOMPLETE"
    )
    return (
        f"{prefix} {state}; profiles={profiles}, migrated={migrated}, "
        f"missing={missing}, stale={stale}, orphaned={orphaned}, unclassified={unclassified}, settings={settings}, "
        f"workspace settings={workspace_settings}, history={history}, grants={grants}; "
        "protected legacy reads=disabled, rollback dual writes=disabled"
    )


def workshop_memory_authority_status(db_path: Path, *, memory_enabled: bool | None) -> str:
    """Describe canonical semantic-memory ownership without reading memory content."""
    prefix = "Workshop memory authority:"
    if memory_enabled is False:
        return f"{prefix} disabled by policy"
    if not db_path.is_file():
        return f"{prefix} pending; canonical memory schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _MEMORY_AUTHORITY_TABLES:
                return f"{prefix} pending; canonical memory schema unavailable"
            profiles = _scalar(connection, "SELECT COUNT(*) FROM channel_agent_runtime_assignments")
            migrated = _scalar(connection, "SELECT COUNT(*) FROM workshop_memory_authority_migrations")
            missing = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments a "
                "JOIN channels c ON c.id = a.channel_id AND c.kind = 'direct' "
                "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "JOIN workshop_execution_state_migrations e "
                "ON e.runtime_profile_id = a.runtime_profile_id AND e.channel_id = a.channel_id "
                "AND e.agent_id = a.agent_id AND e.principal_id = cm.principal_id "
                "WHERE NOT EXISTS (SELECT 1 FROM workshop_memory_authority_migrations m "
                "WHERE m.runtime_profile_id = a.runtime_profile_id "
                "AND m.runtime_config_id = e.runtime_config_id AND m.channel_id = a.channel_id "
                "AND m.agent_id = a.agent_id AND m.principal_id = cm.principal_id)",
            )
            stale = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_memory_authority_migrations m WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a "
                "JOIN channels c ON c.id = a.channel_id AND c.kind = 'direct' "
                "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "JOIN workshop_execution_state_migrations e "
                "ON e.runtime_profile_id = a.runtime_profile_id "
                "AND e.runtime_config_id = m.runtime_config_id "
                "WHERE a.runtime_profile_id = m.runtime_profile_id "
                "AND a.channel_id = m.channel_id AND a.agent_id = m.agent_id "
                "AND cm.principal_id = m.principal_id)",
            )
            counts = connection.execute(
                "SELECT COALESCE(SUM(moved_count), 0), COALESCE(SUM(stamped_count), 0), "
                "COALESCE(SUM(total_count), 0) FROM workshop_memory_authority_migrations"
            ).fetchone()
            moved, stamped, total = (int(value) for value in counts) if counts is not None else (0, 0, 0)
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    if memory_enabled is None:
        state = "NOT VERIFIED"
    else:
        state = "active" if profiles > 0 and missing == 0 and stale == 0 and migrated == profiles else "INCOMPLETE"
    return (
        f"{prefix} {state}; profiles={profiles}, migrated={migrated}, missing={missing}, "
        f"stale={stale}, moved={moved}, stamped={stamped}, migration rows={total}; "
        "protected legacy reads=disabled"
    )


def workshop_operational_state_status(db_path: Path) -> str:
    """Describe canonical job ownership and GitHub subscription authority."""
    prefix = "Workshop operational state:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical operational-state schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _OPERATIONAL_STATE_TABLES:
                return f"{prefix} pending; canonical operational-state schema unavailable"
            profiles = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments",
            )
            migrated = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_operational_state_migrations",
            )
            missing = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments a WHERE NOT EXISTS ("
                "SELECT 1 FROM workshop_operational_state_migrations m "
                "WHERE m.runtime_profile_id = a.runtime_profile_id "
                "AND m.channel_id = a.channel_id AND m.agent_id = a.agent_id)",
            )
            stale = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_operational_state_migrations m WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a "
                "WHERE a.runtime_profile_id = m.runtime_profile_id "
                "AND a.channel_id = m.channel_id AND a.agent_id = m.agent_id)",
            )
            jobs = _scalar(connection, "SELECT COUNT(*) FROM workshop_scheduled_jobs")
            legacy_jobs = _scalar(connection, "SELECT COUNT(*) FROM jobs")
            job_migrations = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_scheduled_job_migrations",
            )
            missing_job_migrations = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments a WHERE NOT EXISTS ("
                "SELECT 1 FROM workshop_scheduled_job_migrations m "
                "WHERE m.runtime_profile_id = a.runtime_profile_id "
                "AND m.channel_id = a.channel_id AND m.agent_id = a.agent_id)",
            )
            stale_job_migrations = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_scheduled_job_migrations m WHERE NOT EXISTS ("
                "SELECT 1 FROM channel_agent_runtime_assignments a "
                "WHERE a.runtime_profile_id = m.runtime_profile_id "
                "AND a.channel_id = m.channel_id AND a.agent_id = m.agent_id)",
            )
            unmigrated_jobs = _scalar(
                connection,
                "SELECT COUNT(*) FROM jobs j JOIN workshop_execution_state_migrations e "
                "ON e.runtime_config_id = j.chat_id LEFT JOIN workshop_job_owners o "
                "ON o.job_id = j.id WHERE o.job_id IS NULL OR o.principal_id != e.principal_id "
                "OR o.channel_id != e.channel_id OR o.agent_id != e.agent_id "
                "OR o.runtime_profile_id != e.runtime_profile_id",
            )
            conflicting_jobs = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_scheduled_jobs c "
                "JOIN workshop_job_owners o ON o.job_id = c.id "
                "WHERE c.principal_id != o.principal_id OR c.channel_id != o.channel_id "
                "OR c.agent_id != o.agent_id OR c.runtime_profile_id != o.runtime_profile_id",
            )
            github_subscriptions = _scalar(
                connection,
                "SELECT COUNT(*) FROM principal_github_subscriptions",
            )
            missing_subscriptions = _scalar(
                connection,
                "SELECT COUNT(DISTINCT m.principal_id) "
                "FROM workshop_operational_state_migrations m WHERE NOT EXISTS ("
                "SELECT 1 FROM principal_github_subscriptions g "
                "WHERE g.principal_id = m.principal_id)",
            )
            notification_overrides = _scalar(
                connection,
                "SELECT COUNT(*) FROM principal_notification_delivery_preferences",
            )
            integration_route_owners = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_integration_route_owners",
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    state = (
        "active"
        if profiles > 0
        and migrated == profiles
        and missing == 0
        and stale == 0
        and job_migrations == profiles
        and missing_job_migrations == 0
        and stale_job_migrations == 0
        and unmigrated_jobs == 0
        and conflicting_jobs == 0
        and missing_subscriptions == 0
        else "INCOMPLETE"
    )
    return (
        f"{prefix} {state}; profiles={profiles}, migrated={migrated}, "
        f"missing={missing}, stale={stale}, jobs={jobs}, legacy archived={legacy_jobs}, "
        f"job migrations={job_migrations}, missing job migrations={missing_job_migrations}, "
        f"stale job migrations={stale_job_migrations}, "
        f"unmigrated jobs={unmigrated_jobs}, conflicting jobs={conflicting_jobs}, "
        f"GitHub principals={github_subscriptions}, "
        f"missing subscriptions={missing_subscriptions}; personal GitHub settings=canonical, "
        f"notification overrides={notification_overrides}, "
        f"integration route owners={integration_route_owners}, "
        "personal notification delivery=canonical, "
        "protected legacy ownership "
        "reads=disabled, compatibility job writes=disabled"
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _event_from_row(row: sqlite3.Row) -> EventEnvelope:
    actor = row["actor_principal_id"]
    envelope = EventEnvelope(
        event_id=EventId(str(row["event_id"])),
        envelope_version=int(row["envelope_version"]),
        event_type=str(row["event_type"]),
        event_version=int(row["event_version"]),
        workshop_id=WorkshopId(str(row["workshop_id"])),
        aggregate_type=str(row["aggregate_type"]),
        aggregate_id=parse_opaque_id(str(row["aggregate_id"])),
        actor_principal_id=PrincipalId(str(actor)) if actor is not None else None,
        occurred_at=_parse_timestamp(str(row["occurred_at"])),
        idempotency_key=str(row["idempotency_key"]) if row["idempotency_key"] is not None else None,
        payload=json.loads(str(row["payload_json"])),
        metadata=json.loads(str(row["metadata_json"])),
    )
    if envelope.content_hash != row["content_hash"]:
        raise ValueError("Workshop event content hash mismatch")
    return envelope


def _required_payload_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workshop event lacks {key}")
    return value


def _insert_replayed_fact(facts: dict[str, Any], fact_id: str, value: Any) -> None:
    if fact_id in facts:
        raise ValueError("Workshop event replay contains a duplicate aggregate")
    facts[fact_id] = value


def _replay_state(connection: sqlite3.Connection) -> _ReplayState:
    principal_kinds: dict[str, str] = {}
    channel_kinds: dict[str, str] = {}
    workshop_memberships: dict[str, tuple[str, str, str]] = {}
    channel_bindings: dict[str, tuple[str, str, str]] = {}
    channel_memberships: dict[str, tuple[str, str, str]] = {}
    runtime_assignments: dict[str, tuple[str, str, str]] = {}
    replayed: list[_ReplayMessage] = []
    for row in connection.execute("SELECT * FROM event_log ORDER BY position"):
        envelope = _event_from_row(row)
        payload = envelope.payload
        aggregate_id = str(envelope.aggregate_id)
        if (
            envelope.event_type
            in {
                WorkshopEventType.CHANNEL_CREATED,
                WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                WorkshopEventType.CHANNEL_MEMBER_ADDED,
                WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
                WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
            }
            and envelope.event_version != 1
        ):
            raise ValueError("Workshop event replay encountered an unsupported event version")
        if envelope.event_type == WorkshopEventType.PRINCIPAL_CREATED:
            if envelope.event_version not in {1, 2}:
                raise ValueError("Workshop event replay encountered an unsupported principal version")
            _insert_replayed_fact(
                principal_kinds,
                aggregate_id,
                _required_payload_text(payload, "kind"),
            )
            continue
        if envelope.event_type == WorkshopEventType.MESSAGE_CREATED and envelope.event_version not in {1, 2}:
            raise ValueError("Workshop event replay encountered an unsupported message version")
        if envelope.event_type == WorkshopEventType.CHANNEL_CREATED:
            _insert_replayed_fact(
                channel_kinds,
                aggregate_id,
                _required_payload_text(payload, "kind"),
            )
            continue
        if envelope.event_type == WorkshopEventType.WORKSHOP_MEMBER_ADDED:
            _insert_replayed_fact(
                workshop_memberships,
                aggregate_id,
                (
                    str(envelope.workshop_id),
                    _required_payload_text(payload, "principal_id"),
                    _required_payload_text(payload, "role"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.TRANSPORT_CHANNEL_BOUND:
            _insert_replayed_fact(
                channel_bindings,
                aggregate_id,
                (
                    _required_payload_text(payload, "channel_id"),
                    _required_payload_text(payload, "transport"),
                    _required_payload_text(payload, "external_channel_id"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.CHANNEL_MEMBER_ADDED:
            _insert_replayed_fact(
                channel_memberships,
                aggregate_id,
                (
                    _required_payload_text(payload, "channel_id"),
                    _required_payload_text(payload, "principal_id"),
                    _required_payload_text(payload, "role"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.RUNTIME_PROFILE_ASSIGNED:
            _insert_replayed_fact(
                runtime_assignments,
                aggregate_id,
                (
                    _required_payload_text(payload, "channel_id"),
                    _required_payload_text(payload, "agent_id"),
                    _required_payload_text(payload, "runtime_profile_id"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.RUNTIME_PROFILE_REASSIGNED:
            prior = runtime_assignments.get(aggregate_id)
            channel_id = _required_payload_text(payload, "channel_id")
            agent_id = _required_payload_text(payload, "agent_id")
            if prior is None or prior[:2] != (channel_id, agent_id):
                raise ValueError("Workshop runtime reassignment has no matching replayed assignment")
            runtime_assignments[aggregate_id] = (
                channel_id,
                agent_id,
                _required_payload_text(payload, "runtime_profile_id"),
            )
            continue
        if envelope.event_type != WorkshopEventType.MESSAGE_CREATED:
            continue

        author_id = _required_payload_text(payload, "author_principal_id")
        kind = principal_kinds.get(author_id)
        if kind == "human":
            direction = "user"
        elif kind == "agent":
            direction = "assistant"
        else:
            raise ValueError("Workshop message author kind is not parity-comparable")
        reply_to = payload.get("reply_to_message_id")
        if reply_to is not None and not isinstance(reply_to, str):
            raise ValueError("Workshop message reply reference is invalid")
        replayed.append(
            _ReplayMessage(
                message_id=str(envelope.aggregate_id),
                channel_id=_required_payload_text(payload, "channel_id"),
                author_principal_id=author_id,
                reply_to_message_id=reply_to,
                body=_required_payload_text(payload, "body"),
                event_position=int(row["position"]),
                created_at=envelope.occurred_at.isoformat(),
                direction=direction,
                source=str(envelope.metadata.get("source", "unknown")),
            )
        )
    return _ReplayState(
        principal_kinds=principal_kinds,
        channel_kinds=channel_kinds,
        workshop_memberships=workshop_memberships,
        channel_bindings=channel_bindings,
        channel_memberships=channel_memberships,
        runtime_assignments=runtime_assignments,
        messages=tuple(replayed),
    )


def _mapping_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> int:
    return sum(expected.get(fact_id) != actual.get(fact_id) for fact_id in expected.keys() | actual.keys())


def _projection_mismatches(connection: sqlite3.Connection, replayed: _ReplayState) -> tuple[int, int]:
    expected = {
        message.message_id: (
            message.channel_id,
            message.author_principal_id,
            message.reply_to_message_id,
            message.body,
            message.event_position,
            message.created_at,
            message.direction,
        )
        for message in replayed.messages
    }
    actual = {
        str(row[0]): (
            str(row[1]),
            str(row[2]),
            str(row[3]) if row[3] is not None else None,
            str(row[4]),
            int(row[5]),
            str(row[6]),
            "user" if row[7] == "human" else "assistant" if row[7] == "agent" else "unknown",
        )
        for row in connection.execute(
            "SELECT m.id, m.channel_id, m.author_principal_id, m.reply_to_message_id, m.body, "
            "m.created_event_position, m.created_at, p.kind FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id"
        ).fetchall()
    }
    actual_principal_kinds = {
        str(row[0]): str(row[1]) for row in connection.execute("SELECT id, kind FROM principals").fetchall()
    }
    actual_channel_kinds = {
        str(row[0]): str(row[1]) for row in connection.execute("SELECT id, kind FROM channels").fetchall()
    }
    actual_bindings = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT id, channel_id, transport, external_channel_id FROM channel_bindings"
        ).fetchall()
    }
    actual_workshop_memberships = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute("SELECT id, workshop_id, principal_id, role FROM workshop_memberships").fetchall()
    }
    actual_channel_memberships = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute("SELECT id, channel_id, principal_id, role FROM channel_memberships").fetchall()
    }
    actual_runtime_assignments = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT id, channel_id, agent_id, runtime_profile_id FROM channel_agent_runtime_assignments"
        ).fetchall()
    }
    mismatches = (
        _mapping_mismatches(expected, actual)
        + _mapping_mismatches(replayed.principal_kinds, actual_principal_kinds)
        + _mapping_mismatches(replayed.channel_kinds, actual_channel_kinds)
        + _mapping_mismatches(replayed.workshop_memberships, actual_workshop_memberships)
        + _mapping_mismatches(replayed.channel_bindings, actual_bindings)
        + _mapping_mismatches(replayed.channel_memberships, actual_channel_memberships)
        + _mapping_mismatches(replayed.runtime_assignments, actual_runtime_assignments)
    )
    return len(actual), mismatches


def _read_legacy_keys(
    history_root: Path,
    external_channel_id: str,
    channel_id: str,
) -> tuple[list[tuple[str, str]], int, int]:
    if not _TELEGRAM_SUBJECT_PATTERN.fullmatch(external_channel_id):
        return [], 0, 1
    try:
        canonical_channel_id = ChannelId(channel_id)
    except (TypeError, ValueError):
        return [], 0, 1
    if history_root.is_symlink():
        return [], 0, 1
    history_dirs = (
        history_root / str(canonical_channel_id),
        history_root / external_channel_id,
    )

    keys: list[tuple[str, str]] = []
    pending_user_turn: bool | None = None
    malformed = 0
    unreadable = 0
    records: list[dict] = []
    for history_dir in history_dirs:
        if history_dir.is_symlink():
            unreadable += 1
            continue
        if not history_dir.exists():
            continue
        if not history_dir.is_dir():
            unreadable += 1
            continue
        for path in sorted(history_dir.glob("*.jsonl")):
            if path.is_symlink():
                unreadable += 1
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                unreadable += 1
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    malformed += 1
                    continue
                records.append(record)
    records.sort(key=lambda record: str(record.get("ts", "")))
    for record in records:
        direction = record.get("dir")
        text = record.get("text")
        chat_id = record.get("chat_id")
        if direction not in {"user", "assistant"} or not isinstance(text, str):
            continue
        if str(chat_id) != external_channel_id:
            continue
        if direction == "user":
            media = record.get("media")
            is_shadowed_message = media is None or (
                isinstance(media, dict)
                and media.get("type") in {"photo", "document", "voice"}
                and media.get("workshop_message_shadowed") is True
            )
            # Only the most recent user record can own the next ordinary
            # assistant record. Abandoned historical turns must not queue
            # indefinitely and suppress otherwise valid later history.
            pending_user_turn = is_shadowed_message
            if is_shadowed_message:
                keys.append((direction, hashlib.sha256(text.encode("utf-8")).hexdigest()))
            continue
        if text.startswith(_SCHEDULED_ASSISTANT_PREFIXES):
            continue
        if pending_user_turn is None:
            continue
        shadowed_turn = pending_user_turn
        pending_user_turn = None
        if not shadowed_turn or _SYNTHETIC_ASSISTANT_PATTERN.fullmatch(text):
            continue
        keys.append((direction, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return keys, malformed, unreadable


def _classify_legacy_suffix(
    canonical: list[_ReplayMessage],
    legacy_keys: list[tuple[str, str]],
) -> tuple[set[int], int]:
    """Return exact canonical match indices and unmatched archive count."""
    if not canonical:
        return set(), 0
    canonical_keys = [message.legacy_key for message in canonical]
    suffix_matches = _longest_common_subsequence_suffix_lengths(canonical_keys[1:], legacy_keys)
    candidates = [
        (1 + suffix_matches[start + 1], start) for start, key in enumerate(legacy_keys) if key == canonical_keys[0]
    ]
    if not candidates:
        return set(), 0
    matched_count, start = min(
        candidates,
        key=lambda item: (-item[0], len(canonical_keys) - item[0] + len(legacy_keys) - item[1] - item[0], -item[1]),
    )
    left = canonical_keys[1:]
    right = legacy_keys[start + 1 :]
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index in range(len(left) - 1, -1, -1):
        for right_index in range(len(right) - 1, -1, -1):
            table[left_index][right_index] = (
                table[left_index + 1][right_index + 1] + 1
                if left[left_index] == right[right_index]
                else max(table[left_index + 1][right_index], table[left_index][right_index + 1])
            )
    indices = {0}
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            indices.add(left_index + 1)
            left_index += 1
            right_index += 1
        elif table[left_index + 1][right_index] >= table[left_index][right_index + 1]:
            left_index += 1
        else:
            right_index += 1
    return indices, len(legacy_keys) - start - matched_count


def _longest_common_subsequence_suffix_lengths(
    canonical_keys: list[tuple[str, str]],
    legacy_keys: list[tuple[str, str]],
) -> list[int]:
    """Return ordered-match counts for every legacy suffix in linear memory."""
    following = [0] * (len(legacy_keys) + 1)
    for canonical_key in reversed(canonical_keys):
        current = [0] * (len(legacy_keys) + 1)
        for legacy_index in range(len(legacy_keys) - 1, -1, -1):
            legacy_key = legacy_keys[legacy_index]
            if canonical_key == legacy_key:
                current[legacy_index] = following[legacy_index + 1] + 1
            else:
                current[legacy_index] = max(
                    following[legacy_index],
                    current[legacy_index + 1],
                )
        following = current
    return following


def workshop_canonical_message_integrity_status(db_path: Path) -> str:
    """Verify the authoritative message projection against its event log."""
    prefix = "Workshop canonical message integrity:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical message schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _PARITY_TABLES:
                return f"{prefix} pending; canonical message schema unavailable"
            replayed = _replay_state(connection)
            projected_count, mismatches = _projection_mismatches(connection, replayed)
        finally:
            connection.close()
    except (sqlite3.Error, OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    state = "clean" if mismatches == 0 else "DIVERGED"
    return (
        f"{prefix} {state}; canonical={len(replayed.messages)}, "
        f"projected={projected_count}, replay mismatches={mismatches}"
    )


def workshop_transcript_authority_status(db_path: Path) -> str:
    """Describe the durable protected direct-interaction transcript cutover."""
    prefix = "Workshop transcript authority:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical transcript schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _TRANSCRIPT_AUTHORITY_TABLES:
                return f"{prefix} pending; canonical transcript schema unavailable"
            marker = connection.execute(
                "SELECT protected_private_text_cutover_position FROM workshop_transcript_authority WHERE singleton = 1"
            ).fetchone()
            if marker is None:
                return f"{prefix} INCOMPLETE; durable cutover marker missing"
            cutover = int(marker[0])
            completed_runs = _scalar(
                connection,
                "SELECT COUNT(*) FROM runs WHERE status = 'completed'",
            )
            post_cutover_messages = _scalar(
                connection,
                "SELECT COUNT(*) FROM messages WHERE created_event_position > ?",
                (cutover,),
            )
        finally:
            connection.close()
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    return (
        f"{prefix} active; completed runs={completed_runs}, "
        f"post-cutover messages={post_cutover_messages}; "
        "protected inputs=text/media, JSONL reads=disabled, writes=disabled; "
        "canonical export=v1"
    )


def workshop_legacy_jsonl_archive_status(db_path: Path, history_root: Path) -> str:
    """Classify retained JSONL without treating it as canonical authority."""
    prefix = "Workshop legacy JSONL archive:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical message schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _TRANSCRIPT_AUTHORITY_TABLES:
                return f"{prefix} pending; canonical message schema unavailable"
            replayed = _replay_state(connection)
            (
                matched,
                required_missing,
                archive_only,
                canonical_only,
                channels,
                malformed,
                unreadable,
                missing_classes,
                canonical_only_classes,
            ) = _classified_legacy_archive(
                connection,
                replayed,
                history_root,
            )
        finally:
            connection.close()
    except (sqlite3.Error, OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    if malformed or unreadable:
        return f"{prefix} NOT VERIFIED; malformed records={malformed}, unreadable sources={unreadable}"
    return (
        f"{prefix} classified; historical required matched={matched}, "
        f"historical required missing={required_missing}, "
        f"archive-only={archive_only}, intentional canonical-only={canonical_only}, "
        f"missing classes={_format_message_classes(missing_classes)}, "
        f"canonical-only classes={_format_message_classes(canonical_only_classes)}, "
        f"Telegram channels={channels}; "
        "archive is non-authoritative"
    )


def workshop_transition_tooling_status() -> str:
    """Describe source-level transition surfaces retired after authority cutover.

    Historical database rows, migration receipts, and JSONL files remain
    deliberate archives. This build contract covers only executable tooling
    that could shadow, compare, or manually switch live production authority.
    Regression guards pin the corresponding source deletions.
    """
    return (
        "Workshop transition tooling: retired; shadow recorders=disabled, "
        "crash flags=disabled, parity comparator=disabled, "
        "delivery qualification=disabled; archives=retained"
    )


def workshop_client_preference_status(
    db_path: Path,
    *,
    telegram_enabled: bool | None = True,
    tts_enabled: bool | None = True,
) -> str:
    """Report canonical client-binding voice authority without transport IDs."""
    prefix = "Workshop client preferences:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical client preference schema unavailable"
    required = {
        "channels",
        "channel_bindings",
        "channel_memberships",
        "external_identities",
        "client_binding_voice_preferences",
        "client_binding_voice_migrations",
    }
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= required:
                return f"{prefix} pending; canonical client preference schema unavailable"
            eligible = (
                _scalar(
                    connection,
                    "SELECT COUNT(DISTINCT cb.id) FROM external_identities ei "
                    "JOIN channel_bindings cb ON cb.transport = ei.provider "
                    "AND cb.external_channel_id = ei.external_subject "
                    "JOIN channels c ON c.id = cb.channel_id AND c.kind = 'direct' "
                    "JOIN channel_memberships cm ON cm.channel_id = cb.channel_id "
                    "AND cm.principal_id = ei.principal_id WHERE cb.transport = 'telegram'",
                )
                if telegram_enabled is not False
                else 0
            )
            preferences = _scalar(connection, "SELECT COUNT(*) FROM client_binding_voice_preferences")
            migrations = _scalar(connection, "SELECT COUNT(*) FROM client_binding_voice_migrations")
            missing = (
                _scalar(
                    connection,
                    "SELECT COUNT(DISTINCT cb.id) FROM external_identities ei "
                    "JOIN channel_bindings cb ON cb.transport = ei.provider "
                    "AND cb.external_channel_id = ei.external_subject "
                    "JOIN channels c ON c.id = cb.channel_id AND c.kind = 'direct' "
                    "JOIN channel_memberships cm ON cm.channel_id = cb.channel_id "
                    "AND cm.principal_id = ei.principal_id "
                    "LEFT JOIN client_binding_voice_preferences p ON p.channel_binding_id = cb.id "
                    "LEFT JOIN client_binding_voice_migrations m ON m.channel_binding_id = cb.id "
                    "WHERE cb.transport = 'telegram' "
                    "AND (p.channel_binding_id IS NULL OR m.channel_binding_id IS NULL)",
                )
                if telegram_enabled is not False
                else 0
            )
            stale = _scalar(
                connection,
                "SELECT COUNT(*) FROM client_binding_voice_preferences p "
                "JOIN client_binding_voice_migrations m ON m.channel_binding_id = p.channel_binding_id "
                "WHERE p.principal_id != m.principal_id OR m.legacy_reads_disabled != 1 "
                "OR m.rollback_dual_writes != 1",
            )
        finally:
            connection.close()
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    if telegram_enabled is False:
        state = "inactive" if stale == 0 else "INCOMPLETE"
        capability = "adapter disabled"
    else:
        state = "active" if eligible == preferences == migrations and missing == 0 and stale == 0 else "INCOMPLETE"
        if telegram_enabled is None or tts_enabled is None:
            capability = "not verified"
        else:
            capability = "enabled" if tts_enabled else "TTS disabled"
    return (
        f"{prefix} {state}; eligible bindings={eligible}, preferences={preferences}, "
        f"migrations={migrations}, missing={missing}, stale={stale}; "
        f"voice capability={capability}, authority=canonical, legacy reads=disabled, rollback dual writes=active"
    )


def workshop_appearance_preference_status(db_path: Path) -> str:
    """Report principal-scoped Workshop appearance authority."""
    prefix = "Workshop appearance preferences:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical appearance preference schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= {"principals", "principal_appearance_preferences"}:
                return f"{prefix} pending; canonical appearance preference schema unavailable"
            principals = _scalar(connection, "SELECT COUNT(*) FROM principals WHERE kind = 'human'")
            stored_themes = [
                str(row[0])
                for row in connection.execute("SELECT theme_id FROM principal_appearance_preferences").fetchall()
            ]
        finally:
            connection.close()
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    known = {item.theme_id for item in WORKSHOP_APPEARANCE_THEMES}
    invalid = sum(theme_id not in known for theme_id in stored_themes)
    explicit = len(stored_themes)
    defaulted = max(0, principals - explicit)
    state = "active" if explicit <= principals and invalid == 0 else "INCOMPLETE"
    return (
        f"{prefix} {state}; principals={principals}, explicit={explicit}, "
        f"defaulted={defaulted}, invalid={invalid}; authority=canonical"
    )


def workshop_human_handle_status(db_path: Path) -> str:
    """Report canonical human-handle coverage and shared-namespace integrity."""
    prefix = "Workshop human handles:"
    if not db_path.is_file():
        return f"{prefix} pending; canonical human-handle schema unavailable"
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            required = {
                "agent_definitions",
                "event_log",
                "human_handles",
                "principals",
                "workshop_memberships",
            }
            if not tables >= required:
                return f"{prefix} pending; canonical human-handle schema unavailable"
            eligible = _scalar(
                connection,
                "SELECT COUNT(*) FROM workshop_memberships wm "
                "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human'",
            )
            assigned = _scalar(
                connection,
                "SELECT COUNT(*) FROM human_handles hh "
                "JOIN workshop_memberships wm ON wm.workshop_id = hh.workshop_id "
                "AND wm.principal_id = hh.principal_id "
                "JOIN principals p ON p.id = hh.principal_id AND p.kind = 'human'",
            )
            rows = connection.execute("SELECT handle FROM human_handles").fetchall()
            invalid = sum(not HUMAN_HANDLE_PATTERN.fullmatch(str(row[0])) for row in rows)
            conflicting = _scalar(
                connection,
                "SELECT COUNT(*) FROM human_handles hh JOIN agent_definitions ad "
                "ON ad.workshop_id = hh.workshop_id AND ad.handle = hh.handle COLLATE NOCASE",
            )
            reserved = {
                (str(row[0]), str(row[1]).casefold())
                for row in connection.execute(
                    "SELECT workshop_id, handle FROM human_handles UNION ALL "
                    "SELECT workshop_id, handle FROM agent_definitions"
                ).fetchall()
            }
            unresolved: dict[tuple[str, str], int] = {}
            for row in connection.execute(
                "SELECT wm.workshop_id, p.display_name FROM workshop_memberships wm "
                "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
                "LEFT JOIN human_handles hh ON hh.workshop_id = wm.workshop_id "
                "AND hh.principal_id = p.id WHERE hh.principal_id IS NULL"
            ).fetchall():
                try:
                    candidate = derive_human_handle(str(row[1]))
                except WorkshopHumanHandleError:
                    invalid += 1
                    continue
                key = (str(row[0]), candidate.casefold())
                unresolved[key] = unresolved.get(key, 0) + 1
            conflicting += sum(count for key, count in unresolved.items() if count > 1 or key in reserved)
            orphaned = _scalar(connection, "SELECT COUNT(*) FROM human_handles") - assigned
            migrations = _scalar(
                connection,
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal.handle_assigned'",
            )
        finally:
            connection.close()
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"
    missing = max(0, eligible - assigned)
    state = "active" if missing == 0 and invalid == 0 and conflicting == 0 and orphaned == 0 else "INCOMPLETE"
    return (
        f"{prefix} {state}; eligible={eligible}, assigned={assigned}, missing={missing}, "
        f"invalid={invalid}, conflicting={conflicting}, orphaned={orphaned}, "
        f"migration rows={migrations}; authority=canonical"
    )


def _classified_legacy_archive(
    connection: sqlite3.Connection,
    replayed: _ReplayState,
    history_root: Path,
) -> tuple[int, int, int, int, int, int, int, dict[str, int], dict[str, int]]:
    """Classify pre-cutover rollback records separately from canonical-only data."""
    marker = connection.execute(
        "SELECT protected_private_text_cutover_position FROM workshop_transcript_authority WHERE singleton = 1"
    ).fetchone()
    if marker is None:
        raise ValueError("Canonical transcript cutover marker is absent")
    cutover = int(marker[0])
    required_ids = {
        str(value)
        for row in connection.execute(
            "SELECT inbound_message_id, result_message_id FROM runs WHERE result_message_id IS NOT NULL"
        ).fetchall()
        for value in row
        if value is not None
    }
    bindings = {
        channel_id: external_channel_id
        for channel_id, transport, external_channel_id in replayed.channel_bindings.values()
        if transport == "telegram" and replayed.channel_kinds.get(channel_id) == "direct"
    }
    matched = required_missing = archive_only = canonical_only = malformed = unreadable = 0
    missing_classes: dict[str, int] = {}
    canonical_only_classes: dict[str, int] = {}
    for channel_id, external_channel_id in bindings.items():
        messages = [message for message in replayed.messages if message.channel_id == channel_id]
        required = [
            message for message in messages if message.message_id in required_ids and message.event_position <= cutover
        ]
        intentional = [message for message in messages if message not in required]
        canonical_only += len(intentional)
        for message in intentional:
            key = f"{message.source}/{message.direction}"
            canonical_only_classes[key] = canonical_only_classes.get(key, 0) + 1
        legacy_keys, channel_malformed, channel_unreadable = _read_legacy_keys(
            history_root,
            external_channel_id,
            channel_id,
        )
        matched_indices, channel_unmatched = _classify_legacy_suffix(required, legacy_keys)
        channel_matched = len(matched_indices)
        channel_missing = len(required) - channel_matched
        matched += channel_matched
        required_missing += channel_missing
        archive_only += channel_unmatched
        malformed += channel_malformed
        unreadable += channel_unreadable
        if channel_missing:
            # Count the unmatched required population by non-identifying
            # origin/direction. Exact sequence alignment remains content-free.
            for index, message in enumerate(required):
                if index in matched_indices:
                    continue
                key = f"{message.source}/{message.direction}"
                missing_classes[key] = missing_classes.get(key, 0) + 1
    return (
        matched,
        required_missing,
        archive_only,
        canonical_only,
        len(bindings),
        malformed,
        unreadable,
        missing_classes,
        canonical_only_classes,
    )


def _format_message_classes(classes: dict[str, int]) -> str:
    if not classes:
        return "none"
    return ",".join(f"{key}={classes[key]}" for key in sorted(classes))
