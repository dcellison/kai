"""Tests for the `python -m kai memory` administrative CLI.

Covers the Phase 4 purge subcommand from spec 320. The CLI itself is a
thin dispatcher around `memory.delete_by_source`; these tests verify
the dispatch, arg parsing, authorization gate (--yes), known-source
allow-list, and init-failure handling - not the delete primitive,
which has its own coverage in tests/test_memory.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai import memory_admin

# ── _build_parser: arg surface ───────────────────────────────────────


class TestBuildParser:
    """The parser defines the public CLI surface. Pin it down so a
    future edit to the arg list can't silently change semantics."""

    def test_purge_requires_user_id(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["purge", "--source", "extracted"])

    def test_purge_requires_source(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["purge", "12345"])

    def test_purge_yes_defaults_false(self):
        """--yes is opt-in. Operators must type it explicitly."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", "extracted"])
        assert args.yes is False

    def test_purge_yes_flag_accepted(self):
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", "extracted", "--yes"])
        assert args.yes is True

    def test_unknown_subcommand_exits(self):
        """argparse rejects typos before any delete attempt."""
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["scramble", "12345"])

    def test_no_subcommand_exits(self):
        """subparsers=required keeps `python -m kai memory` by itself
        from silently doing nothing."""
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ── _cmd_purge: authorization gate ───────────────────────────────────


class TestPurgeAuthorizationGate:
    """Without --yes, the command exits 2 AFTER running memory init.
    Running init in the dry-run path (PR #333 review finding #7) means
    a misconfigured store surfaces as exit 1 during the dry-run - so
    the operator does not get a confident 'would run...' message
    followed by an init failure on the real invocation.
    """

    def test_no_yes_with_init_ok_returns_2(self, capsys):
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", "extracted"])
        with patch.object(memory_admin, "_initialize_memory", return_value=True) as init_mock:
            code = memory_admin._cmd_purge(args)
        assert code == 2
        # Init DOES run now so the dry-run validates reachability.
        init_mock.assert_called_once()
        out = capsys.readouterr().out
        assert "would run" in out
        assert "12345" in out
        assert "--yes" in out

    def test_no_yes_with_init_failure_returns_1(self):
        """Init failure in dry-run exits 1, not 2. Exit 1 takes
        precedence over the 'authorization missing' exit so the
        operator sees the real blocker first."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", "extracted"])
        with patch.object(memory_admin, "_initialize_memory", return_value=False):
            code = memory_admin._cmd_purge(args)
        assert code == 1

    def test_dry_run_labels_legacy_clearly(self, capsys):
        """Empty source must read as `<legacy ...>` in the plan, not as
        an ambiguous `''` that an operator might mistake for nothing."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", ""])
        with patch.object(memory_admin, "_initialize_memory", return_value=True):
            code = memory_admin._cmd_purge(args)
        assert code == 2
        out = capsys.readouterr().out
        assert "legacy" in out


# ── _cmd_purge: known-source allow-list ──────────────────────────────


class TestPurgeSourceAllowList:
    """Unknown --source values are rejected before memory init so a
    typo doesn't no-op silently against every row."""

    @pytest.mark.parametrize("bad_source", ["user-raw", "extract", "facts", "all"])
    def test_unknown_source_returns_2(self, bad_source, capsys):
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "12345", "--source", bad_source, "--yes"])
        with patch.object(memory_admin, "_initialize_memory") as init_mock:
            code = memory_admin._cmd_purge(args)
        assert code == 2
        init_mock.assert_not_called()
        err = capsys.readouterr().err
        assert "unknown source" in err
        # The error message lists accepted values so the operator can
        # fix the invocation without re-reading --help.
        assert "extracted" in err

    @pytest.mark.parametrize("good_source", ["extracted", "user_raw", ""])
    def test_known_sources_proceed(self, good_source):
        """Valid source values pass the allow-list and reach the
        memory init path when --yes is supplied."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(
            ["purge", "12345", "--source", good_source, "--yes"],
        )
        with patch.object(memory_admin, "_initialize_memory", return_value=False):
            # init_memory returns False -> _cmd_purge should exit 1, NOT
            # 2. That proves we made it past the allow-list check.
            code = memory_admin._cmd_purge(args)
        assert code == 1


# ── _cmd_purge: happy path ──────────────────────────────────────────


class TestPurgeHappyPath:
    """--yes + known source + working memory layer -> delete_by_source
    is invoked with the right args and the row count is printed."""

    def test_calls_delete_by_source_with_parsed_args(self, capsys):
        parser = memory_admin._build_parser()
        args = parser.parse_args(
            ["purge", "99999", "--source", "extracted", "--yes"],
        )
        # delete_by_source is async; wrap the mock so asyncio.run can
        # await it. AsyncMock(return_value=7) resolves to 7.
        delete_mock = AsyncMock(return_value=7)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.delete_by_source", delete_mock),
        ):
            code = memory_admin._cmd_purge(args)
        assert code == 0
        delete_mock.assert_awaited_once_with(user_id="99999", source="extracted")
        out = capsys.readouterr().out
        assert "deleted 7 row(s)" in out

    def test_legacy_source_passes_empty_string(self):
        """source='' must reach delete_by_source as the literal empty
        string (not None, not missing) - memory.py's implementation
        branches on `source == ''` for the legacy path."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["purge", "42", "--source", "", "--yes"])
        delete_mock = AsyncMock(return_value=0)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.delete_by_source", delete_mock),
        ):
            memory_admin._cmd_purge(args)
        delete_mock.assert_awaited_once_with(user_id="42", source="")

    def test_delete_exception_returns_1(self, capsys):
        """Runtime failure during delete (e.g. Qdrant disk error) exits
        with status 1 so the operator sees a clear non-success code."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(
            ["purge", "12345", "--source", "extracted", "--yes"],
        )
        delete_mock = AsyncMock(side_effect=RuntimeError("qdrant down"))
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.delete_by_source", delete_mock),
        ):
            code = memory_admin._cmd_purge(args)
        assert code == 1
        err = capsys.readouterr().err
        assert "purge failed" in err


# ── _initialize_memory: init path ────────────────────────────────────


class TestInitializeMemory:
    """Isolated tests for the init helper. The CLI tests above patch
    this helper out; these ensure its own branches are exercised."""

    def test_returns_false_when_is_enabled_false(self, capsys):
        """init_memory ran but memory stays disabled (e.g. MEMORY_ENABLED=
        false or dimension mismatch). The CLI must not proceed."""
        with (
            patch("kai.config.load_config", return_value=MagicMock()),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=False),
        ):
            ok = memory_admin._initialize_memory()
        assert ok is False
        err = capsys.readouterr().err
        assert "not enabled" in err

    def test_returns_true_on_success(self):
        with (
            patch("kai.config.load_config", return_value=MagicMock()),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=True),
        ):
            ok = memory_admin._initialize_memory()
        assert ok is True

    def test_returns_false_on_exception(self, capsys):
        """load_config raising (bad env, missing file) surfaces as a
        stderr message and False, not a crash."""
        with patch("kai.config.load_config", side_effect=RuntimeError("bad env")):
            ok = memory_admin._initialize_memory()
        assert ok is False
        err = capsys.readouterr().err
        assert "init failed" in err


# ── cli() dispatch ───────────────────────────────────────────────────


class TestCliDispatch:
    """The top-level cli() function calls sys.exit with the
    subcommand's return code. SystemExit is the expected control flow."""

    def test_dispatch_calls_purge(self):
        with (
            patch.object(memory_admin, "_cmd_purge", return_value=0) as cmd_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            memory_admin.cli(["purge", "12345", "--source", "extracted"])
        assert exc_info.value.code == 0
        cmd_mock.assert_called_once()
        # The Namespace passed to _cmd_purge carries the parsed args.
        args = cmd_mock.call_args[0][0]
        assert args.user_id == "12345"
        assert args.source == "extracted"

    def test_dispatch_propagates_nonzero_exit(self):
        with patch.object(memory_admin, "_cmd_purge", return_value=2), pytest.raises(SystemExit) as exc_info:
            memory_admin.cli(["purge", "12345", "--source", "extracted"])
        assert exc_info.value.code == 2
