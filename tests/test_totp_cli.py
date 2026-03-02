"""
Tests for the TOTP CLI commands in src/kai/totp.py.

The CLI commands (setup/status/reset) are interactive and require root, so tests
focus on the parts that are meaningful to cover without spawning real subprocesses
or needing elevated privileges:
  - Root guard (sys.exit(1) when geteuid() != 0)
  - _cmd_status output for configured/not-configured states
  - _cmd_reset file removal logic and graceful handling of missing files
  - cli() dispatch and error handling for unknown/missing subcommands

_cmd_setup is only tested for the root guard - the rest of its body (makedirs,
writing files, QR code rendering, interactive input) is not worth mocking out in
isolation since it is a run-once operator-facing flow.
"""

from unittest.mock import patch

import pytest

import kai.totp
from kai.totp import (
    TOTP_ATTEMPTS_PATH,
    TOTP_SECRET_PATH,
    _cmd_reset,
    _cmd_setup,
    _cmd_status,
    cli,
)


@pytest.fixture(autouse=True)
def _reset_totp_cache():
    """Reset the is_totp_configured module-level cache around each test."""
    kai.totp._totp_is_configured = False
    yield
    kai.totp._totp_is_configured = False


# ---------------------------------------------------------------------------
# _cmd_status
# ---------------------------------------------------------------------------


def test_cmd_status_prints_configured(capsys):
    """_cmd_status prints a 'configured' message when is_totp_configured() is True."""
    with patch("kai.totp.is_totp_configured", return_value=True):
        _cmd_status()

    out = capsys.readouterr().out
    assert "configured" in out.lower()
    assert "not" not in out.lower()


def test_cmd_status_prints_not_configured(capsys):
    """_cmd_status prints a 'not configured' message when is_totp_configured() is False."""
    with patch("kai.totp.is_totp_configured", return_value=False):
        _cmd_status()

    out = capsys.readouterr().out
    assert "not configured" in out.lower()


# ---------------------------------------------------------------------------
# _cmd_reset
# ---------------------------------------------------------------------------


def test_cmd_reset_exits_if_not_root():
    """_cmd_reset calls sys.exit(1) when not running as root."""
    with (
        patch("kai.totp.os.geteuid", return_value=1000),
        pytest.raises(SystemExit) as exc,
    ):
        _cmd_reset()

    assert exc.value.code == 1


def test_cmd_reset_removes_both_files(capsys):
    """_cmd_reset removes both secret and attempts files when present."""
    with (
        patch("kai.totp.os.geteuid", return_value=0),
        patch("kai.totp.os.remove") as mock_remove,
    ):
        _cmd_reset()

    # Both paths should have been removed.
    removed_paths = [call.args[0] for call in mock_remove.call_args_list]
    assert TOTP_SECRET_PATH in removed_paths
    assert TOTP_ATTEMPTS_PATH in removed_paths

    out = capsys.readouterr().out
    assert "Removed" in out


def test_cmd_reset_handles_missing_files_gracefully(capsys):
    """_cmd_reset prints 'Nothing to remove' when both files are already absent."""
    with (
        patch("kai.totp.os.geteuid", return_value=0),
        patch("kai.totp.os.remove", side_effect=FileNotFoundError),
    ):
        _cmd_reset()

    out = capsys.readouterr().out
    assert "Nothing to remove" in out


def test_cmd_reset_handles_partially_missing_files(capsys):
    """_cmd_reset removes whichever files exist and doesn't error on missing ones."""

    def _selective_remove(path: str) -> None:
        # Simulate only the secret file being present.
        if path == TOTP_ATTEMPTS_PATH:
            raise FileNotFoundError

    with (
        patch("kai.totp.os.geteuid", return_value=0),
        patch("kai.totp.os.remove", side_effect=_selective_remove),
    ):
        _cmd_reset()

    out = capsys.readouterr().out
    assert TOTP_SECRET_PATH in out


# ---------------------------------------------------------------------------
# _cmd_setup root guard
# ---------------------------------------------------------------------------


def test_cmd_setup_exits_if_not_root():
    """_cmd_setup calls sys.exit(1) immediately when not running as root."""
    with (
        patch("kai.totp.os.geteuid", return_value=1000),
        pytest.raises(SystemExit) as exc,
    ):
        _cmd_setup()

    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# cli() dispatch
# ---------------------------------------------------------------------------


def test_cli_exits_on_unknown_subcommand():
    """cli() calls sys.exit(1) for an unrecognized subcommand."""
    with pytest.raises(SystemExit) as exc:
        cli(["badcmd"])

    assert exc.value.code == 1


def test_cli_exits_on_empty_args():
    """cli() calls sys.exit(1) when invoked with no arguments."""
    with pytest.raises(SystemExit) as exc:
        cli([])

    assert exc.value.code == 1


def test_cli_dispatches_status():
    """cli() dispatches 'status' to _cmd_status."""
    with patch("kai.totp._cmd_status") as mock_status:
        cli(["status"])

    mock_status.assert_called_once()


def test_cli_dispatches_reset():
    """cli() dispatches 'reset' to _cmd_reset."""
    with patch("kai.totp._cmd_reset") as mock_reset:
        cli(["reset"])

    mock_reset.assert_called_once()


def test_cli_dispatches_setup():
    """cli() dispatches 'setup' to _cmd_setup."""
    with patch("kai.totp._cmd_setup") as mock_setup:
        cli(["setup"])

    mock_setup.assert_called_once()
