"""Contracts for separating principal policy from canonical agent identity."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from kai.principal_policy import (
    PRINCIPAL_POLICY_MIGRATION,
    PrincipalPolicyMigrationConflict,
    plan_principal_policy_migration,
    record_principal_policy_migration,
)
from kai.workshop.agent_definitions import (
    AgentDefinitionRevision,
    active_agent_definition_revision,
    render_agent_definition_context,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import AgentDefinitionId, AgentDefinitionRevisionId, AgentId, WorkshopId
from kai.workshop.store import WorkshopEventStore

_LEGACY_PREFIX = (
    "# Kai\n\n"
    "## About This File\n\n"
    "This file is the bootstrap template for Kai's backend-neutral identity. The installer "
    "copies it to `<DATA_DIR>/home/<principal_id>/AGENTS.md` for each canonical Workshop "
    "human with an assigned runtime; `backend.ensure_user_home` lazily seeds it for profiles "
    "added later in development mode. Claude receives a thin `.claude/CLAUDE.md` import "
    "adapter; all managed identity content remains here. Edit the per-principal `AGENTS.md` "
    "to add operator-personal content; the tracked template ships universal content only. "
    'Once customized, you can delete this "About This File" section from the per-principal copy.\n\n'
    "## Who You Are\n\n"
    "You're Kai, a personal AI assistant available through configured clients such as "
    "Workshop and Telegram. You run locally on the operator's machine and have access to a "
    "shell, the filesystem, the web, a scheduler, and a per-principal memory store.\n\n"
)


def test_tracked_policy_is_identity_neutral() -> None:
    policy = (Path(__file__).parents[1] / "templates" / "AGENTS.md").read_text()

    assert policy.startswith("# Principal Policy\n")
    assert "## Who You Are" not in policy
    assert "You're Kai" not in policy
    assert "You are Kai" not in policy
    assert "Agent identity belongs exclusively to the active canonical agent definition." in policy


def test_known_legacy_prefix_migrates_without_changing_custom_rules() -> None:
    custom_rules = "## Operator Rules\n\n- Preserve this exact text.  \n- Keep spacing.\n"

    plan = plan_principal_policy_migration(_LEGACY_PREFIX + custom_rules)

    assert plan.changed is True
    assert plan.content.startswith("# Principal Policy\n")
    assert "## Who You Are" not in plan.content
    assert "backend-neutral principal policy" in plan.content
    assert plan.content.endswith(custom_rules)


@pytest.mark.parametrize(
    "content",
    (
        "# Kai\n\n## Who You Are\n\nYou are Kai with customized behavior.\n",
        "# Personal Policy\n\nCustom preface. You're Kai in this context.\n",
        "# Personal Policy\n\nCustom preface. YOU ARE KAI in this context.\n",
        "# Personal Policy\n\nCustom preface. You\u2019re Kai in this context.\n",
    ),
)
def test_ambiguous_identity_text_fails_closed(content: str) -> None:
    with pytest.raises(PrincipalPolicyMigrationConflict, match="original was not changed"):
        plan_principal_policy_migration(content)


def test_migration_backup_and_receipt_are_private_and_replay_safe(tmp_path: Path) -> None:
    before = _LEGACY_PREFIX + "## Operator Rules\n\nKeep me.\n"
    after = plan_principal_policy_migration(before).content

    first = record_principal_policy_migration(tmp_path, before, after)
    second = record_principal_policy_migration(tmp_path, before, after)

    assert first == second == tmp_path / ".kai-migrations" / PRINCIPAL_POLICY_MIGRATION
    assert (first / "AGENTS.md.before").read_text() == before
    receipt = json.loads((first / "receipt.json").read_text())
    assert receipt["migration"] == PRINCIPAL_POLICY_MIGRATION
    assert receipt["backup"] == "AGENTS.md.before"
    assert len(receipt["before_sha256"]) == 64
    assert len(receipt["after_sha256"]) == 64
    assert stat.S_IMODE((tmp_path / ".kai-migrations").stat().st_mode) == 0o700
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE((first / "AGENTS.md.before").stat().st_mode) == 0o600
    assert stat.S_IMODE((first / "receipt.json").stat().st_mode) == 0o600


def test_conflicting_existing_backup_is_not_overwritten(tmp_path: Path) -> None:
    before = _LEGACY_PREFIX
    after = plan_principal_policy_migration(before).content
    migration_dir = tmp_path / ".kai-migrations" / PRINCIPAL_POLICY_MIGRATION
    migration_dir.mkdir(parents=True)
    backup = migration_dir / "AGENTS.md.before"
    backup.write_text("different original\n")

    with pytest.raises(PrincipalPolicyMigrationConflict, match="backup does not match"):
        record_principal_policy_migration(tmp_path, before, after)

    assert backup.read_text() == "different original\n"


async def test_predefined_kai_identity_remains_in_canonical_versioned_definition(tmp_path: Path) -> None:
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        bootstrap = await bootstrap_default_workshop(
            store,
            (BootstrapHuman("Operator", "admin", "desktop", "operator", "operator"),),
        )
        revision = await active_agent_definition_revision(store, bootstrap.agent_id)
        assert revision is not None
        assert revision.handle == "kai"
        assert revision.revision_number == 2
        assert "Act as Kai." in revision.instructions
    finally:
        await store.close()


@pytest.mark.parametrize("lane", ["private-owner", "private-nonowner", "shared-owner", "shared-nonowner"])
def test_custom_agent_context_has_only_its_own_identity_in_every_lane(lane: str) -> None:
    policy = (Path(__file__).parents[1] / "templates" / "AGENTS.md").read_text()
    revision = AgentDefinitionRevision(
        definition_id=AgentDefinitionId.new(),
        revision_id=AgentDefinitionRevisionId.new(),
        workshop_id=WorkshopId.new(),
        agent_id=AgentId.new(),
        handle="qualification_agent",
        display_name="Qualification agent",
        description=f"Qualification fixture for {lane}.",
        lifecycle_state="active",
        revision_number=1,
        purpose="Qualify agent identity separation.",
        instructions="Act only as Qualification agent.",
        capabilities=("text_generation",),
        collaboration_operations=(),
    )

    context = policy + "\n" + render_agent_definition_context(revision)

    assert "Act only as Qualification agent." in context
    assert "You're Kai" not in context
    assert "You are Kai" not in context
    assert "Act as Kai" not in context
