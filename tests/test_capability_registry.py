"""Contract tests for the transport-neutral capability inventory."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kai.bot import _TELEGRAM_COMMAND_HANDLERS
from kai.capability_registry import (
    CAPABILITY_REGISTRY,
    AdapterDisposition,
    AdapterId,
    AuthorityScope,
    CapabilityContext,
    ConfirmationPolicy,
    ContextRequirement,
    WorkshopSurface,
    capability_availability,
    capability_by_id,
    render_telegram_help,
    telegram_command_dispositions,
    telegram_command_inventory,
    validate_capability_registry,
)
from kai.telegram_adapter import _TELEGRAM_COMMANDS


def test_registry_accounts_for_every_registered_telegram_command() -> None:
    registered_commands = {command for command, _handler in _TELEGRAM_COMMAND_HANDLERS}

    assert registered_commands == set(telegram_command_inventory())
    assert validate_capability_registry(telegram_commands=registered_commands) == CAPABILITY_REGISTRY


def test_telegram_menu_is_an_inventoried_subset_of_registered_commands() -> None:
    registered_commands = {command for command, _handler in _TELEGRAM_COMMAND_HANDLERS}
    menu_commands = {command.command for command in _TELEGRAM_COMMANDS}

    assert menu_commands <= registered_commands
    assert menu_commands <= set(telegram_command_inventory())


@pytest.mark.parametrize(
    ("primary", "aliases"),
    [
        ("model", ("models",)),
        ("backend", ("backends",)),
        ("workspace", ("ws", "workspaces")),
        ("voice", ("voices",)),
        ("job", ("jobs",)),
    ],
)
def test_telegram_aliases_share_one_operation_without_becoming_operations(
    primary: str,
    aliases: tuple[str, ...],
) -> None:
    inventory = telegram_command_inventory()
    dispositions = telegram_command_dispositions()

    assert dispositions[primary] != AdapterDisposition.PRESENTATION_ALIAS
    assert all(inventory[alias] == inventory[primary] for alias in aliases)
    assert all(dispositions[alias] == AdapterDisposition.PRESENTATION_ALIAS for alias in aliases)


def test_registry_accounts_for_every_workshop_surface() -> None:
    inventoried_surfaces = {
        presentation.surface
        for capability in CAPABILITY_REGISTRY
        if (presentation := capability.presentations[AdapterId.WORKSHOP]).surface is not None
    }

    assert inventoried_surfaces == {surface.value for surface in WorkshopSurface}


def test_registry_definitions_have_complete_adapter_and_authority_metadata() -> None:
    for capability in CAPABILITY_REGISTRY:
        assert set(capability.presentations) == set(AdapterId)
        assert capability.canonical_service
        assert capability.authority_scope in AuthorityScope
        if capability.mutates_state:
            assert capability.idempotency.value != "none"


def test_contextual_availability_is_redacted_and_excludes_admin_operations() -> None:
    results = capability_availability(
        AdapterId.WORKSHOP,
        CapabilityContext(authenticated=True),
        include_unavailable=True,
    )
    operation_ids = {result.operation_id for result in results}
    reset = next(result for result in results if result.operation_id == "conversation.session.reset")
    serialized = json.dumps([result.as_dict() for result in results])

    assert "administration.webhook_status.read" not in operation_ids
    assert reset.available is False
    assert reset.unavailable_reason == "Open a conversation to use this action."
    assert "prn_" not in serialized
    assert "rtp_" not in serialized
    assert "/Users/" not in serialized
    assert "secret" not in serialized.lower()


def test_unimplemented_workshop_operations_are_not_advertised() -> None:
    results = capability_availability(
        AdapterId.WORKSHOP,
        CapabilityContext(authenticated=True, administrator=True),
    )
    assert "administration.webhook_status.read" not in {result.operation_id for result in results}


def test_telegram_help_is_registry_backed_and_filters_administrator_operations() -> None:
    member_help = render_telegram_help(administrator=False)
    administrator_help = render_telegram_help(administrator=True)

    assert "/stop - Interrupt current response" in member_help
    assert "/start" not in member_help
    assert "/webhooks" not in member_help
    assert "/webhooks - Show webhook server status" in administrator_help
    assert {
        line.split()[0].removeprefix("/") for line in administrator_help.splitlines() if line.startswith("/")
    } <= set(telegram_command_inventory())


def test_unsupported_operations_do_not_appear_in_adapter_availability() -> None:
    results = capability_availability(
        AdapterId.TELEGRAM,
        CapabilityContext(authenticated=True, conversation=True),
        include_unavailable=True,
    )

    assert "threads.manage" not in {result.operation_id for result in results}


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown capability operation"):
        capability_by_id("missing.operation")


def test_registry_rejects_duplicate_operation_ids() -> None:
    capability = CAPABILITY_REGISTRY[0]

    with pytest.raises(ValueError, match="operation IDs must be unique"):
        validate_capability_registry((capability, capability))


def test_registry_rejects_duplicate_telegram_command_owners() -> None:
    first = CAPABILITY_REGISTRY[0]
    second = CAPABILITY_REGISTRY[1]
    duplicate_presentation = replace(
        second.presentations[AdapterId.TELEGRAM],
        primary_command=first.presentations[AdapterId.TELEGRAM].primary_command,
        help_entries=tuple(
            replace(entry, commands=(first.presentations[AdapterId.TELEGRAM].primary_command,))
            for entry in second.presentations[AdapterId.TELEGRAM].help_entries
        ),
    )
    duplicate = replace(
        second,
        presentations={
            **second.presentations,
            AdapterId.TELEGRAM: duplicate_presentation,
        },
    )

    with pytest.raises(ValueError, match="belongs to multiple capabilities"):
        validate_capability_registry((first, duplicate))


def test_registry_rejects_read_only_mutation_semantics() -> None:
    capability = replace(
        CAPABILITY_REGISTRY[0],
        confirmation=ConfirmationPolicy.REQUIRED,
    )

    with pytest.raises(ValueError, match="declares mutation semantics"):
        validate_capability_registry((capability,))


def test_registry_rejects_missing_adapter_disposition() -> None:
    capability = replace(
        CAPABILITY_REGISTRY[0],
        presentations={AdapterId.TELEGRAM: CAPABILITY_REGISTRY[0].presentations[AdapterId.TELEGRAM]},
    )

    with pytest.raises(ValueError, match="define every adapter disposition"):
        validate_capability_registry((capability,))


def test_registry_rejects_admin_operation_without_admin_context() -> None:
    capability = replace(
        capability_by_id("administration.webhook_status.read"),
        requirements=frozenset({ContextRequirement.AUTHENTICATED}),
    )

    with pytest.raises(ValueError, match="lacks an administrator requirement"):
        validate_capability_registry((capability,))


def test_registry_rejects_unsupported_adapter_with_surface() -> None:
    capability = capability_by_id("threads.manage")
    telegram = replace(
        capability.presentations[AdapterId.TELEGRAM],
        primary_command="threads",
    )
    invalid = replace(
        capability,
        presentations={**capability.presentations, AdapterId.TELEGRAM: telegram},
    )

    with pytest.raises(ValueError, match="declares an adapter surface"):
        validate_capability_registry((invalid,))


def test_registry_rejects_help_for_unregistered_command() -> None:
    capability = capability_by_id("conversation.run.cancel")
    telegram = capability.presentations[AdapterId.TELEGRAM]
    invalid = replace(
        capability,
        presentations={
            **capability.presentations,
            AdapterId.TELEGRAM: replace(
                telegram,
                help_entries=(replace(telegram.help_entries[0], commands=("retired",)),),
            ),
        },
    )

    with pytest.raises(ValueError, match="advertises unregistered Telegram commands"):
        validate_capability_registry((invalid,))
