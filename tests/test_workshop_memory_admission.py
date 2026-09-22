"""Separate provenance quality from explicit operator retrieval admission."""

from __future__ import annotations

import sqlite3

import pytest

from kai.workshop import schema
from kai.workshop.temporal_memory import MemoryAdmissionAuthority, resolve_memory_admission


def test_schema_migration_quarantines_incomplete_rows_and_verifies_complete_rows() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE memory_fact_revisions (migration_classification TEXT NOT NULL);
        CREATE TABLE memory_episodes (migration_classification TEXT NOT NULL);
        INSERT INTO memory_fact_revisions VALUES ('canonical'), ('legacy_complete'), ('legacy_incomplete');
        INSERT INTO memory_episodes VALUES ('canonical'), ('legacy_incomplete'), ('legacy_quarantined');
        """
    )

    for statement in schema._MEMORY_ADMISSION_AUTHORITY_SCHEMA.statements:
        connection.execute(statement)

    facts = connection.execute(
        "SELECT migration_classification, admission_authority FROM memory_fact_revisions ORDER BY rowid"
    ).fetchall()
    episodes = connection.execute(
        "SELECT migration_classification, admission_authority FROM memory_episodes ORDER BY rowid"
    ).fetchall()
    connection.close()

    assert facts == [
        ("canonical", "provenance_verified"),
        ("legacy_complete", "provenance_verified"),
        ("legacy_incomplete", "quarantined"),
    ]
    assert episodes == [
        ("canonical", "provenance_verified"),
        ("legacy_incomplete", "quarantined"),
        ("legacy_quarantined", "quarantined"),
    ]


def test_admission_defaults_fail_closed_and_operator_review_preserves_incomplete_provenance() -> None:
    assert resolve_memory_admission("canonical") == MemoryAdmissionAuthority.PROVENANCE_VERIFIED
    assert resolve_memory_admission("legacy_incomplete") == MemoryAdmissionAuthority.QUARANTINED
    assert resolve_memory_admission("legacy_incomplete", "operator_review") == MemoryAdmissionAuthority.OPERATOR_REVIEW

    with pytest.raises(ValueError, match="cannot claim provenance-verified"):
        resolve_memory_admission("legacy_incomplete", "provenance_verified")
    with pytest.raises(ValueError, match="cannot be admitted without correction"):
        resolve_memory_admission("legacy_quarantined", "operator_review")
