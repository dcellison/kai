"""Regression guards for transition mechanisms retired after #917 cutover."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kai import workshop_cli
from kai.workshop.diagnostics import workshop_transition_tooling_status
from kai.workshop_cli import WorkshopOperatorCommandError

_ROOT = Path(__file__).parent.parent


def _source(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_retired_transition_surfaces_are_absent_from_production_source() -> None:
    main_source = _source("src/kai/main.py")
    cli_source = _source("src/kai/workshop_cli.py")
    diagnostics_source = _source("src/kai/workshop/diagnostics.py")
    bot_source = _source("src/kai/bot.py")

    assert ".responding" not in main_source
    assert "delivery-qualification" not in cli_source
    assert "delivery-authority" not in cli_source
    assert "workshop_message_parity_status" not in diagnostics_source
    assert "workshop_message_shadowed" not in bot_source
    assert "workshop_inbound_recorder" not in bot_source
    assert "workshop_artifact_recorder" not in bot_source
    assert "workshop_outbound_recorder" not in bot_source
    assert "workshop_delivery_recorder" not in bot_source
    assert not (_ROOT / "src/kai/workshop/delivery_qualification.py").exists()


def test_operator_cli_no_longer_advertises_retired_qualification_or_rollback() -> None:
    help_text = workshop_cli._parser().format_help()

    assert "delivery-qualification" not in help_text
    assert "delivery-authority" not in help_text
    assert "client-access" in help_text
    assert "integration-route" in help_text
    assert "transcript" in help_text


def test_deployed_database_must_exist_without_creating_it(tmp_path: Path) -> None:
    with pytest.raises(WorkshopOperatorCommandError, match="was not found"):
        workshop_cli._deployed_database(tmp_path)

    assert not (tmp_path / "kai.db").exists()


def test_deployed_database_must_be_regular_and_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "kai.db"
    directory.mkdir()
    with pytest.raises(WorkshopOperatorCommandError, match="regular file"):
        workshop_cli._deployed_database(tmp_path)

    directory.rmdir()
    database = tmp_path / "kai.db"
    database.touch()
    monkeypatch.setattr(os, "geteuid", lambda: database.stat().st_uid + 1)
    with pytest.raises(WorkshopOperatorCommandError, match="account that owns"):
        workshop_cli._deployed_database(tmp_path)


def test_regular_deployed_database_owned_by_invoker_is_accepted(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    database.touch()

    assert workshop_cli._deployed_database(tmp_path) == database


def test_transition_tooling_status_distinguishes_retired_code_from_archives() -> None:
    assert workshop_transition_tooling_status() == (
        "Workshop transition tooling: retired; shadow recorders=disabled, "
        "crash flags=disabled, parity comparator=disabled, "
        "delivery qualification=disabled; archives=retained"
    )
