"""Tests for the `python -m kai memory` administrative CLI.

Covers the Phase 4 purge subcommand from spec 320. The CLI itself is a
thin dispatcher around `memory.delete_by_source`; these tests verify
the dispatch, arg parsing, authorization gate (--yes), known-source
allow-list, and init-failure handling - not the delete primitive,
which has its own coverage in tests/test_memory.py.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai import memory_admin


def test_default_report_directory_uses_canonical_workshop_principal(tmp_path, monkeypatch):
    db_path = tmp_path / "kai.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE external_identities ("
        "principal_id TEXT NOT NULL, provider TEXT NOT NULL, external_subject TEXT NOT NULL)"
    )
    principal_id = "prn_" + "c" * 32
    connection.execute(
        "INSERT INTO external_identities (principal_id, provider, external_subject) VALUES (?, ?, ?)",
        (principal_id, "telegram", "42"),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr("kai.config.DATA_DIR", tmp_path / "data")

    result = memory_admin._default_human_report_directory(
        SimpleNamespace(session_db_path=db_path),
        "42",
        "reclassify",
    )

    assert result == tmp_path / "data" / "home" / principal_id / "docs" / "reclassify"


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

    def test_returns_none_when_is_enabled_false(self, capsys):
        """init_memory ran but memory stays disabled (e.g. MEMORY_ENABLED=
        false or dimension mismatch). The CLI must not proceed."""
        with (
            patch("kai.config.load_config", return_value=MagicMock()),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=False),
        ):
            ok = memory_admin._initialize_memory()
        assert ok is None
        err = capsys.readouterr().err
        assert "not enabled" in err

    def test_returns_config_on_success(self):
        """Success hands back the loaded Config (truthy) so subcommands
        that need config values do not load the environment twice."""
        cfg = MagicMock()
        with (
            patch("kai.config.load_config", return_value=cfg),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=True),
        ):
            ok = memory_admin._initialize_memory()
        assert ok is cfg

    def test_returns_none_on_exception(self, capsys):
        """load_config raising (bad env, missing file) surfaces as a
        stderr message and None, not a crash."""
        with patch("kai.config.load_config", side_effect=RuntimeError("bad env")):
            ok = memory_admin._initialize_memory()
        assert ok is None
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

    def test_dispatch_calls_reclassify(self):
        with (
            patch.object(memory_admin, "_cmd_reclassify", return_value=0) as cmd_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            memory_admin.cli(["reclassify-scope", "100"])
        assert exc_info.value.code == 0
        assert cmd_mock.call_args[0][0].user_id == "100"


# ── reclassify-scope: parser ─────────────────────────────────────────


class TestReclassifyParser:
    def test_modes_are_mutually_exclusive(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["reclassify-scope", "100", "--apply", "a.jsonl", "--rollback", "b.jsonl"])

    def test_classification_flags_default_to_none(self):
        """None defaults are load-bearing: the mutating-mode rejection
        distinguishes explicitly-passed flags from defaults by None."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["reclassify-scope", "100"])
        assert args.backend is None
        assert args.os_user is None
        assert args.provider is None
        assert args.threshold is None
        assert args.sample is None
        assert args.out_dir is None
        assert args.apply is None
        assert args.rollback is None
        assert args.yes is False

    def test_backend_choices_reject_unknown(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["reclassify-scope", "100", "--backend", "acp"])


# ── reclassify-scope: gates ──────────────────────────────────────────


def _write_proposals(tmp_path, *, user_id="100", run_id="rs-1", n=2):
    """A minimal valid proposals file for gate tests."""
    from kai import memory_reclassify

    proposals = [
        memory_reclassify.Proposal(
            memory_id=f"m{i}",
            verdict="global",
            project_id=None,
            confidence=0.9,
            reason="r",
            prior_scope_source="legacy_default",
            text_sha256="ab",
        )
        for i in range(n)
    ]
    path = tmp_path / "proposals.jsonl"
    path.write_text(memory_reclassify.render_proposals({"run_id": run_id, "user_id": user_id}, proposals))
    return path


class TestReclassifyGates:
    def _args(self, argv):
        return memory_admin._build_parser().parse_args(["reclassify-scope", *argv])

    def test_classification_flags_rejected_in_mutating_mode(self, tmp_path, capsys):
        """A typo'd dry-run flag must not silently change apply
        semantics; it is rejected before memory init runs."""
        args = self._args(["100", "--apply", str(tmp_path / "p.jsonl"), "--threshold", "0.9"])
        with patch.object(memory_admin, "_initialize_memory") as init_mock:
            code = memory_admin._cmd_reclassify(args)
        assert code == 2
        init_mock.assert_not_called()
        assert "--threshold" in capsys.readouterr().err

    def test_threshold_out_of_range_rejected(self, capsys):
        args = self._args(["100", "--threshold", "1.5"])
        with patch.object(memory_admin, "_initialize_memory") as init_mock:
            code = memory_admin._cmd_reclassify(args)
        assert code == 2
        init_mock.assert_not_called()
        assert "[0.0, 1.0]" in capsys.readouterr().err

    def test_negative_sample_rejected_before_init(self, capsys):
        """A negative sample would only blow up at report rendering,
        after every reasoner call has been paid for; the typo must
        die before memory init."""
        args = self._args(["100", "--sample", "-1"])
        with patch.object(memory_admin, "_initialize_memory") as init_mock:
            code = memory_admin._cmd_reclassify(args)
        assert code == 2
        init_mock.assert_not_called()
        assert ">= 0" in capsys.readouterr().err

    def test_zero_sample_accepted(self):
        args = self._args(["100", "--sample", "0"])
        run_dry_run = AsyncMock(return_value=0)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()),
            patch("kai.memory_reclassify.run_dry_run", run_dry_run),
        ):
            code = memory_admin._cmd_reclassify(args)
        assert code == 0
        assert run_dry_run.call_args.kwargs["sample"] == 0

    def test_apply_plan_path_rejects_malformed_rows(self, tmp_path, capsys):
        """The no-yes plan path uses the same strict row validation as
        the driver, so a hand-edited file fails before authorization."""
        path = _write_proposals(tmp_path)
        path.write_text(path.read_text().replace('"verdict":"global"', '"verdict":"oops"'))
        args = self._args(["100", "--apply", str(path)])
        with patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()):
            code = memory_admin._cmd_reclassify(args)
        assert code == 1
        assert "verdict" in capsys.readouterr().err

    def test_init_failure_returns_1(self):
        args = self._args(["100"])
        with patch.object(memory_admin, "_initialize_memory", return_value=None):
            code = memory_admin._cmd_reclassify(args)
        assert code == 1

    def test_apply_without_yes_prints_count_and_exits_2(self, tmp_path, capsys):
        path = _write_proposals(tmp_path, n=3)
        args = self._args(["100", "--apply", str(path)])
        with patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()):
            code = memory_admin._cmd_reclassify(args)
        assert code == 2
        out = capsys.readouterr().out
        assert "3 row(s)" in out
        assert "rs-1" in out
        assert "--yes" in out

    def test_apply_wrong_user_header_aborts(self, tmp_path, capsys):
        path = _write_proposals(tmp_path, user_id="200")
        args = self._args(["100", "--apply", str(path), "--yes"])
        with patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()):
            code = memory_admin._cmd_reclassify(args)
        assert code == 1
        assert "200" in capsys.readouterr().err

    def test_apply_with_yes_dispatches_run_apply(self, tmp_path):
        path = _write_proposals(tmp_path)
        args = self._args(["100", "--apply", str(path), "--yes"])
        run_apply = AsyncMock(return_value=0)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()),
            patch("kai.memory_reclassify.run_apply", run_apply),
        ):
            code = memory_admin._cmd_reclassify(args)
        assert code == 0
        run_apply.assert_awaited_once()
        assert run_apply.call_args.kwargs["proposals_path"] == path

    def test_rollback_with_yes_dispatches_run_rollback(self, tmp_path):
        from kai import memory_reclassify

        path = tmp_path / "pre.jsonl"
        path.write_text(
            memory_reclassify.render_preimages(
                {"run_id": "rs-1", "user_id": "100"},
                [memory_reclassify.PreImage(memory_id="a", text="t", metadata={})],
            )
        )
        args = self._args(["100", "--rollback", str(path), "--yes"])
        run_rollback = AsyncMock(return_value=0)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()),
            patch("kai.memory_reclassify.run_rollback", run_rollback),
        ):
            code = memory_admin._cmd_reclassify(args)
        assert code == 0
        run_rollback.assert_awaited_once()

    def test_dry_run_dispatches_with_documented_defaults(self):
        args = self._args(["100"])
        run_dry_run = AsyncMock(return_value=0)
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=MagicMock()),
            patch("kai.memory_reclassify.run_dry_run", run_dry_run),
        ):
            code = memory_admin._cmd_reclassify(args)
        assert code == 0
        kwargs = run_dry_run.call_args.kwargs
        assert kwargs["threshold"] == 0.8
        assert kwargs["sample"] == 10
        assert kwargs["backend"] is None


# ── purge-sandbox ────────────────────────────────────────────────────


class TestPurgeSandbox:
    """The eval-residue purge: census -> delete per sandbox identity ->
    re-census verification. delete_all swallows provider errors by
    design, so the command's success signal is the after-census, and
    these tests pin that contract."""

    @staticmethod
    def _args(yes: bool):
        parser = memory_admin._build_parser()
        return parser.parse_args(["purge-sandbox", "--yes"] if yes else ["purge-sandbox"])

    def test_dry_run_lists_plan_and_exits_2(self, capsys):
        census = {"prn_a": 500, "sandbox-464": 300, "sandbox-465": 74}
        delete_mock = MagicMock()
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", return_value=census),
            patch("kai.memory.delete_all", delete_mock),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=False))
        assert code == 2
        delete_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "sandbox-464: 300" in out
        assert "prn_a: 500" in out
        assert "--yes" in out

    def test_yes_deletes_each_identity_and_verifies(self, capsys):
        before = {"prn_a": 500, "sandbox-464": 300, "sandbox-465": 74}
        after = {"prn_a": 500}
        delete_mock = MagicMock()
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", side_effect=[before, after]),
            patch("kai.memory.delete_all", delete_mock),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=True))
        assert code == 0
        assert {c.kwargs["user_id"] for c in delete_mock.call_args_list} == {
            "sandbox-464",
            "sandbox-465",
        }
        out = capsys.readouterr().out
        assert "deleted 374 row(s)" in out
        assert "unchanged" in out

    def test_surviving_sandbox_rows_fail_the_run(self, capsys):
        before = {"prn_a": 500, "sandbox-464": 300}
        after = {"prn_a": 500, "sandbox-464": 12}
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", side_effect=[before, after]),
            patch("kai.memory.delete_all", MagicMock()),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=True))
        assert code == 1
        assert "sandbox rows remain" in capsys.readouterr().err

    def test_changed_real_principal_counts_fail_the_run(self, capsys):
        before = {"prn_a": 500, "sandbox-464": 300}
        after = {"prn_a": 499}
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", side_effect=[before, after]),
            patch("kai.memory.delete_all", MagicMock()),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=True))
        assert code == 1
        assert "real principal counts changed" in capsys.readouterr().err

    def test_no_sandbox_identities_is_a_clean_noop(self, capsys):
        delete_mock = MagicMock()
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", return_value={"prn_a": 5}),
            patch("kai.memory.delete_all", delete_mock),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=True))
        assert code == 0
        delete_mock.assert_not_called()
        assert "nothing to purge" in capsys.readouterr().out

    def test_init_failure_exits_1(self):
        with patch.object(memory_admin, "_initialize_memory", return_value=None):
            assert memory_admin._cmd_purge_sandbox(self._args(yes=True)) == 1

    def test_census_failure_exits_1(self, capsys):
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.count_points_by_owner", side_effect=RuntimeError("no scroll")),
        ):
            code = memory_admin._cmd_purge_sandbox(self._args(yes=True))
        assert code == 1
        assert "census failed" in capsys.readouterr().err

    def test_cli_dispatches_purge_sandbox(self):
        with (
            patch.object(memory_admin, "_cmd_purge_sandbox", return_value=0) as cmd,
            pytest.raises(SystemExit) as exc,
        ):
            memory_admin.cli(["purge-sandbox"])
        assert exc.value.code == 0
        cmd.assert_called_once()


# ── backfill-explicit ────────────────────────────────────────────────


class TestBackfillExplicit:
    """Provenance backfill for API-written rows: per-row patches
    (source normalization, missing speaker/confidence), payload-only
    application, and verify-by-rescan."""

    @staticmethod
    def _args(yes: bool):
        parser = memory_admin._build_parser()
        return parser.parse_args(["backfill-explicit", "--yes"] if yes else ["backfill-explicit"])

    # Three shapes from the live corpus: a variant source with no
    # stamps, a canonical source missing both stamps, and a fully
    # stamped row needing nothing.
    _POINTS = [
        ("p1", {"user_id": "prn_a", "source": "explicit-decision"}),
        ("p2", {"user_id": "prn_a", "source": "explicit"}),
        ("p3", {"user_id": "prn_b", "source": "explicit", "speaker": "user", "confidence": 0.8}),
    ]

    def test_dry_run_plans_per_row_patches_and_exits_2(self, capsys):
        patch_mock = MagicMock()
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.list_api_written_points", return_value=self._POINTS),
            patch("kai.memory.patch_point_payload", patch_mock),
        ):
            code = memory_admin._cmd_backfill_explicit(self._args(yes=False))
        assert code == 2
        patch_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "2 of 3 API-written row(s) need patches" in out
        assert "1 source normalization(s)" in out
        assert "2 missing speaker(s)" in out
        assert "2 missing confidence value(s)" in out

    def test_yes_patches_only_what_each_row_lacks(self, capsys):
        patch_mock = MagicMock()
        clean = [
            ("p1", {"user_id": "prn_a", "source": "explicit", "speaker": "assistant", "confidence": 0.9}),
            ("p2", {"user_id": "prn_a", "source": "explicit", "speaker": "assistant", "confidence": 0.9}),
            ("p3", {"user_id": "prn_b", "source": "explicit", "speaker": "user", "confidence": 0.8}),
        ]
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.list_api_written_points", side_effect=[self._POINTS, clean]),
            patch("kai.memory.patch_point_payload", patch_mock),
        ):
            code = memory_admin._cmd_backfill_explicit(self._args(yes=True))
        assert code == 0
        patches = {c.kwargs["point_id"]: c.kwargs["payload"] for c in patch_mock.call_args_list}
        assert patches == {
            "p1": {"source": "explicit", "speaker": "assistant", "confidence": 0.9},
            "p2": {"speaker": "assistant", "confidence": 0.9},
        }
        assert "patched 2 row(s); re-scan clean" in capsys.readouterr().out

    def test_fully_stamped_corpus_is_a_clean_noop(self, capsys):
        patch_mock = MagicMock()
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.list_api_written_points", return_value=[self._POINTS[2]]),
            patch("kai.memory.patch_point_payload", patch_mock),
        ):
            code = memory_admin._cmd_backfill_explicit(self._args(yes=True))
        assert code == 0
        patch_mock.assert_not_called()
        assert "nothing to backfill" in capsys.readouterr().out

    def test_dirty_rescan_fails_the_run(self, capsys):
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.list_api_written_points", side_effect=[self._POINTS, self._POINTS]),
            patch("kai.memory.patch_point_payload", MagicMock()),
        ):
            code = memory_admin._cmd_backfill_explicit(self._args(yes=True))
        assert code == 1
        assert "backfill incomplete" in capsys.readouterr().err

    def test_scan_failure_exits_1(self, capsys):
        with (
            patch.object(memory_admin, "_initialize_memory", return_value=True),
            patch("kai.memory.list_api_written_points", side_effect=RuntimeError("no scroll")),
        ):
            code = memory_admin._cmd_backfill_explicit(self._args(yes=True))
        assert code == 1
        assert "source scan failed" in capsys.readouterr().err

    def test_init_failure_exits_1(self):
        with patch.object(memory_admin, "_initialize_memory", return_value=None):
            assert memory_admin._cmd_backfill_explicit(self._args(yes=True)) == 1

    def test_cli_dispatches_backfill_explicit(self):
        with (
            patch.object(memory_admin, "_cmd_backfill_explicit", return_value=0) as cmd,
            pytest.raises(SystemExit) as exc,
        ):
            memory_admin.cli(["backfill-explicit"])
        assert exc.value.code == 0
        cmd.assert_called_once()


class TestBackfillProvenanceParser:
    """Parser pins for backfill-provenance, previously the only
    subcommand with no parser tests (audit test-gap item). Shapes
    mirror the reclassify pins: required user_id, None defaults that
    the mutating-mode rejection depends on, exclusive modes."""

    def test_requires_user_id(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["backfill-provenance"])

    def test_flags_default_to_none(self):
        """None defaults are load-bearing: the mutating-mode rejection
        distinguishes explicitly-passed flags from defaults by None,
        including the store_const shape of --no-assistant-pass."""
        parser = memory_admin._build_parser()
        args = parser.parse_args(["backfill-provenance", "100"])
        assert args.window_seconds is None
        assert args.min_overlap is None
        assert args.strong_overlap_ratio is None
        assert args.overlap_shingle_n is None
        assert args.no_assistant_pass is None
        assert args.assistant_max_user_gap_seconds is None
        assert args.sample is None
        assert args.out_dir is None
        assert args.apply is None
        assert args.rollback is None
        assert args.yes is False

    def test_modes_are_mutually_exclusive(self):
        parser = memory_admin._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["backfill-provenance", "100", "--apply", "a.jsonl", "--rollback", "b.jsonl"])

    def test_dispatch_calls_backfill_provenance(self):
        with (
            patch.object(memory_admin, "_cmd_backfill_provenance", return_value=0) as cmd_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            memory_admin.cli(["backfill-provenance", "100"])
        assert exc_info.value.code == 0
        assert cmd_mock.call_args[0][0].user_id == "100"


class TestKnownSourcesDrift:
    def test_known_sources_track_user_visible_sources(self):
        """_KNOWN_SOURCES is a literal copy (this module keeps
        kai.memory imports lazy so --help stays fast); this pin is
        the drift guard the literal relies on. The two purge-only
        extras are the Track-1 legacy source and the empty string
        for rows with no source key."""
        from kai.memory import USER_VISIBLE_SOURCES

        expected = frozenset(USER_VISIBLE_SOURCES) | {"user_raw", ""}
        assert expected == memory_admin._KNOWN_SOURCES


class TestReportDirectoryFallback:
    def test_unmapped_user_falls_back_to_literal_path(self, tmp_path, monkeypatch):
        """An unmapped user (no external-identity row) lands on the
        literal-user-id path like every other unresolvable branch,
        instead of raising and killing the whole admin command over
        a default that --out-dir can override anyway."""
        db_path = tmp_path / "kai.db"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE external_identities ("
            "principal_id TEXT NOT NULL, provider TEXT NOT NULL, external_subject TEXT NOT NULL)"
        )
        connection.commit()
        connection.close()
        monkeypatch.setattr("kai.config.DATA_DIR", tmp_path / "data")

        result = memory_admin._default_human_report_directory(
            SimpleNamespace(session_db_path=db_path),
            "42",
            "reclassify",
        )

        assert result == tmp_path / "data" / "home" / "42" / "docs" / "reclassify"
