"""Tests for the Reading Recalled Memory migration helper.

Covers `kai.install._migrate_recalled_memory_section` and its wiring into
the per-user AGENTS.md provisioning loop in `_apply_migrate`.

The helper appends the `## Reading Recalled Memory` section from the
tracked AGENTS.md template to pre-existing per-user copies that predate
the section. Without it, operators already on prior installs would
never pick up the new section (the seed step in `_apply_migrate`'s
per-user AGENTS.md block guards with `if not identity_dst.exists()`, so
a stale per-user copy survives reinstall untouched), and the rule
would only apply to users added after the section landed in the template.

The unit tests drive the helper directly with `tmp_path` fixtures; the
integration tests drive it through `_apply_migrate` with the same
stubbed `_set_ownership` / `os.chown` patches the sibling install tests
use, so the ownership-reconciliation contract is exercised end-to-end
without needing real OS accounts on the test host.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from kai.install import (
    _RECALLED_MEMORY_SECTION_HEADER,
    _apply_migrate,
    _migrate_recalled_memory_section,
)

# The header-bounded scan in the helper needs at least two top-level
# `##` sections to exercise the terminator path; we keep the fixture
# minimal and stable so a future template-shape change does not bleed
# into these tests' expected outputs.
_TEMPLATE_WITH_SECTION = (
    "# Kai\n"
    "\n"
    "## Memory Write Routing\n"
    "\n"
    "Write rules here.\n"
    "\n"
    "## Reading Recalled Memory\n"
    "\n"
    "Read rules here.\n"
    "\n"
    "Worked example: `(2026-04-15, fact) operator prefers Earl Grey`.\n"
    "\n"
    "## Behavioral Rules\n"
    "\n"
    "Behavioral rules here.\n"
)

# The same template content but without the recalled-memory section,
# used to exercise the "template missing the section header" branch.
_TEMPLATE_WITHOUT_SECTION = (
    "# Kai\n\n## Memory Write Routing\n\nWrite rules here.\n\n## Behavioral Rules\n\nBehavioral rules here.\n"
)

# Operator-customized per-user copy that lacks the recalled-memory
# section. Represents the pre-migration state of operators who installed
# before this section landed in the tracked template.
_STALE_PER_USER_COPY = (
    "# Kai\n"
    "\n"
    "## Memory Write Routing\n"
    "\n"
    "OPERATOR LOCAL EDIT: prefer terse memory writes.\n"
    "\n"
    "## Behavioral Rules\n"
    "\n"
    "Behavioral rules here.\n"
)


# ── Unit tests: _migrate_recalled_memory_section ─────────────────────


class TestMigrateRecalledMemorySection:
    """Spec §6.2: six unit cases covering the helper's full input matrix.

    These tests run the helper in isolation against `tmp_path` fixtures
    so the failure modes (sentinel absent, sentinel present, template
    missing section, broken symlink, partial-write recovery) are pinned
    independent of the install-flow context.
    """

    # ── Case 1: per-user copy missing the section ──

    def test_appends_section_when_missing(self, tmp_path):
        """Append happens; file size grows; sentinel header is now present."""
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        per_user.write_text(_STALE_PER_USER_COPY)
        original_size = per_user.stat().st_size

        result = _migrate_recalled_memory_section(per_user, template, dry_run=False)

        assert result is True
        new_content = per_user.read_text()
        assert per_user.stat().st_size > original_size
        assert _RECALLED_MEMORY_SECTION_HEADER in new_content
        # The append should land at end-of-file with a single blank-line
        # separator. The stale copy's last content line was "Behavioral
        # rules here."; after the append, the section follows it.
        assert "Behavioral rules here." in new_content
        # Verify the operator's prior customization survives byte-for-byte
        # in the prefix; the migration appends, it does not rewrite.
        assert new_content.startswith(_STALE_PER_USER_COPY.rstrip())

    def test_dry_run_appends_nothing_but_returns_true(self, tmp_path):
        """Dry-run returns True (would-modify) without touching the file."""
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        per_user.write_text(_STALE_PER_USER_COPY)

        result = _migrate_recalled_memory_section(per_user, template, dry_run=True)

        assert result is True
        # File contents unchanged: byte-for-byte equality.
        assert per_user.read_text() == _STALE_PER_USER_COPY

    # ── Case 2: per-user copy already has the section ──

    def test_noop_when_section_already_present(self, tmp_path):
        """Sentinel present in per-user copy: helper returns False, file unchanged."""
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        # A per-user copy that already carries the section, e.g. seeded
        # from a current template or manually merged by the operator.
        per_user.write_text(_TEMPLATE_WITH_SECTION)
        original = per_user.read_text()

        result = _migrate_recalled_memory_section(per_user, template, dry_run=False)

        assert result is False
        # Byte-for-byte unchanged. No idempotent re-append, no double
        # section, no whitespace drift from a stray rstrip-then-write.
        assert per_user.read_text() == original

    # ── Case 3: operator-customized variant ──

    def test_appends_to_operator_customized_variant_at_eof(self, tmp_path):
        """An extra section between Memory Write Routing and Behavioral Rules survives.

        The migration appends at end-of-file rather than mid-file, so
        intervening operator-authored content is not displaced. This is
        the §3.4 v3 guarantee that operators can leave the appended
        section at EOF or move it to match their existing section
        ordering after the fact (the sentinel check keeps the next
        install idempotent regardless of position).
        """
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        customized = (
            "# Kai\n"
            "\n"
            "## Memory Write Routing\n"
            "\n"
            "Write rules here.\n"
            "\n"
            "## Operator's Custom Section\n"
            "\n"
            "Some private operator notes that must not be moved or rewritten.\n"
            "\n"
            "## Behavioral Rules\n"
            "\n"
            "Behavioral rules here.\n"
        )
        per_user.write_text(customized)

        result = _migrate_recalled_memory_section(per_user, template, dry_run=False)

        assert result is True
        new_content = per_user.read_text()
        # Operator's custom section is intact at its original position.
        assert "## Operator's Custom Section" in new_content
        assert "Some private operator notes" in new_content
        # The appended section lands at EOF, after Behavioral Rules.
        operator_pos = new_content.find("## Operator's Custom Section")
        behavioral_pos = new_content.find("## Behavioral Rules")
        recalled_pos = new_content.find(_RECALLED_MEMORY_SECTION_HEADER)
        assert operator_pos < behavioral_pos < recalled_pos

    # ── Case 4: template missing the section header ──

    def test_skip_with_warning_when_template_missing_section(self, tmp_path, capsys):
        """Template lacks the section: skip, warn, do not modify per-user copy.

        Returns None (failure path), not False (no-op). The dry-run preview
        relies on this distinction to avoid printing "already present" when
        the helper has actually warned about a different problem.
        """
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITHOUT_SECTION)
        per_user = tmp_path / "per_user.md"
        per_user.write_text(_STALE_PER_USER_COPY)

        result = _migrate_recalled_memory_section(per_user, template, dry_run=False)

        assert result is None
        # Per-user copy is byte-for-byte unchanged.
        assert per_user.read_text() == _STALE_PER_USER_COPY
        # Warning surfaces to operator stdout per the placeholder-warning
        # shape the per-user AGENTS.md provisioning loop uses when the template
        # is missing.
        warning = capsys.readouterr().out
        assert "WARNING" in warning
        assert str(template) in warning

    # ── Case 5: per-user copy is a broken symlink ──

    def test_skip_with_warning_on_broken_symlink(self, tmp_path, capsys):
        """Broken symlink at identity_dst: skip with warning, no exception escapes.

        Returns None (failure path) so the dry-run preview can suppress the
        "already present" line that would otherwise mislead the operator.
        """
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        # Point the symlink at a path that does not exist. is_file()
        # returns False for a broken symlink, so the helper's existence
        # guard hits first and returns None cleanly.
        per_user.symlink_to(tmp_path / "nonexistent.md")

        result = _migrate_recalled_memory_section(per_user, template, dry_run=False)

        assert result is None
        # The symlink itself is untouched (still a broken symlink).
        assert per_user.is_symlink()
        # No write occurred at the symlink target.
        assert not (tmp_path / "nonexistent.md").exists()

    # ── Case 6: atomic write recovery on mid-write failure ──

    def test_atomic_write_preserves_original_on_failure(self, tmp_path, monkeypatch):
        """Simulated write failure leaves the original per-user copy intact.

        The helper writes a `.tmp` file in the same directory and then
        Path.replace's it over the destination. A failure during the
        temp write (the only point where partial state could exist)
        must leave the destination byte-for-byte unchanged and clean up
        the temp file. This pins the atomic-write contract referenced
        in the helper's docstring against a real partial-write scenario,
        which a coverage check alone would not catch.
        """
        template = tmp_path / "template.md"
        template.write_text(_TEMPLATE_WITH_SECTION)
        per_user = tmp_path / "per_user.md"
        per_user.write_text(_STALE_PER_USER_COPY)

        # Patch Path.write_text on the tmp_path namespace so the temp
        # file write fails partway. The helper's try/except cleans up
        # the temp file and re-raises OSError; the original per-user
        # copy at `per_user` must remain untouched.
        original_write_text = Path.write_text

        def _failing_write_text(self, content, *args, **kwargs):
            # Touch the tmp file partway through, then fail. Verifies
            # the cleanup branch handles a partially-created temp file.
            self.touch()
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(Path, "write_text", _failing_write_text)

        with pytest.raises(OSError, match="simulated mid-write failure"):
            _migrate_recalled_memory_section(per_user, template, dry_run=False)

        # Restore the real write_text so the assertion below can read.
        monkeypatch.setattr(Path, "write_text", original_write_text)

        # Original per-user content is intact.
        assert per_user.read_text() == _STALE_PER_USER_COPY
        # Temp file was cleaned up by the helper's exception path.
        temp_path = per_user.parent / (per_user.name + ".tmp")
        assert not temp_path.exists()


# ── Integration tests: per-user seed loop in _apply_migrate ─────────


class TestApplyMigrateRecalledMemoryIntegration:
    """Spec §6.3: three integration cases covering the dry-run wiring.

    Drives the migration through `_apply_migrate` so the dry-run preview
    branch (the `if dry_run:` branch in the per-user AGENTS.md provisioning
    loop) and the live branch (after `copy2`, before `_set_ownership`)
    are exercised together. The patches mirror the sibling
    TestApplyMigrateClaudeMdSeed in test_install.py so we do not need
    real OS accounts on the test host.
    """

    def _write_users_yaml(self, path: Path, entries: list[dict]) -> None:
        path.write_text(yaml.safe_dump({"users": entries}))

    def _stub_pwd_getpwnam(self):
        # See sibling note in test_install.py: _apply_migrate calls
        # pwd.getpwnam to resolve os_user uid/gid; the stub keeps the
        # test independent of any actual user on the host machine.
        class _Pw:
            pw_uid = 1234
            pw_gid = 1234

        return _Pw()

    def _build_install_layout(self, tmp_path, template_content, agents_dst_content=None):
        """Set up an install layout with a populated template and optional pre-existing per-user copy."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        (src / "templates" / "AGENTS.md").write_text(template_content)
        # MEMORY.md and PREFERENCES.md templates are referenced by the
        # sibling seed blocks in _apply_migrate; provide them so the
        # function does not divert into a placeholder branch that
        # would emit unrelated warnings into the test's capsys.
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")

        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 12345, "os_user": "alice"}])

        data_path = tmp_path / "data"
        if agents_dst_content is not None:
            user_home = data_path / "home" / "12345"
            user_home.mkdir(parents=True)
            (user_home / "AGENTS.md").write_text(agents_dst_content)

        install_path = tmp_path / "install"
        install_path.mkdir()

        return src, data_path, install_path, users_yaml

    # ── Case A: stale + dry-run ──

    def test_dry_run_previews_append_for_stale_copy(self, tmp_path, capsys):
        """Stale per-user copy + dry-run: file unchanged, preview line printed."""
        src, data_path, install_path, users_yaml = self._build_install_layout(
            tmp_path,
            template_content=_TEMPLATE_WITH_SECTION,
            agents_dst_content=_STALE_PER_USER_COPY,
        )

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=True,
                users_yaml_path=Path(users_yaml),
            )

        agents_dst = data_path / "home" / "12345" / "AGENTS.md"
        # File contents unchanged (dry-run).
        assert agents_dst.read_text() == _STALE_PER_USER_COPY
        # The dry-run preview surfaces in stdout so an operator running
        # `make install --dry-run` sees the planned migration.
        output = capsys.readouterr().out
        assert "[DRY RUN] Would append Reading Recalled Memory section" in output
        assert str(agents_dst) in output

    # ── Case B: current + dry-run ──

    def test_dry_run_reports_noop_when_section_already_present(self, tmp_path, capsys):
        """Per-user copy already has section + dry-run: file unchanged, no-op line printed."""
        src, data_path, install_path, users_yaml = self._build_install_layout(
            tmp_path,
            template_content=_TEMPLATE_WITH_SECTION,
            # Per-user copy carries the section already (e.g. seeded
            # from a current template or migrated in a prior install).
            agents_dst_content=_TEMPLATE_WITH_SECTION,
        )

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=True,
                users_yaml_path=Path(users_yaml),
            )

        agents_dst = data_path / "home" / "12345" / "AGENTS.md"
        assert agents_dst.read_text() == _TEMPLATE_WITH_SECTION
        output = capsys.readouterr().out
        assert "Reading Recalled Memory section already present" in output

    # ── Case D: dry-run when the tracked template is missing entirely ──

    def test_dry_run_surfaces_template_missing_when_agents_dst_exists(self, tmp_path, capsys):
        """agents_dst exists + template absent + dry-run: surface the gap.

        Without an explicit preview line for this case the dry-run output
        would silently omit the migration step the operator might assume
        would run. The seed branch above only fires when agents_dst is
        missing, so under the (agents_dst exists, template missing)
        combination the operator would otherwise see no relevant output.
        """
        src, data_path, install_path, users_yaml = self._build_install_layout(
            tmp_path,
            # Note: _build_install_layout normally writes a CLAUDE.md
            # template; passing None here is not how the helper is set
            # up. We rebuild the layout manually to omit the template.
            template_content=_TEMPLATE_WITH_SECTION,
            agents_dst_content=_STALE_PER_USER_COPY,
        )
        # Remove the AGENTS.md template so home_template_exists is False
        # at the seed-loop's check. MEMORY.md and PREFERENCES.md templates
        # remain so the sibling seed blocks do not emit unrelated noise.
        (src / "templates" / "AGENTS.md").unlink()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=True,
                users_yaml_path=Path(users_yaml),
            )

        agents_dst = data_path / "home" / "12345" / "AGENTS.md"
        # File unchanged in dry-run mode.
        assert agents_dst.read_text() == _STALE_PER_USER_COPY
        # The dry-run preview names the failure explicitly so the operator
        # knows the migration was considered and could not proceed.
        output = capsys.readouterr().out
        assert "cannot evaluate migration" in output
        assert "template missing" in output

    # ── Case C: stale + live install ──

    def test_live_install_appends_and_chowns(self, tmp_path):
        """Live install on a stale per-user copy: section appended; _set_ownership called.

        Ownership reconciliation is the load-bearing reason the migration
        sits between the seed copy2 and the `_set_ownership` chown step;
        the §4.3 ordering rationale documents the #347 regression shape
        this prevents. Verifying that `_set_ownership` is invoked after
        the migration runs pins the contract end-to-end.
        """
        src, data_path, install_path, users_yaml = self._build_install_layout(
            tmp_path,
            template_content=_TEMPLATE_WITH_SECTION,
            agents_dst_content=_STALE_PER_USER_COPY,
        )

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership") as set_ownership_mock,
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )

        agents_dst = data_path / "home" / "12345" / "AGENTS.md"
        content = agents_dst.read_text()
        # The section landed.
        assert _RECALLED_MEMORY_SECTION_HEADER in content
        # The operator's prior content survived.
        assert "OPERATOR LOCAL EDIT" in content
        # _set_ownership was called for the per-user AGENTS.md
        # AFTER the migration ran (the migration is sync, so any
        # _set_ownership call we see was issued after the migration's
        # Path.replace returned). At least one recursive call against
        # the canonical identity is the contract; the provisioning loop calls
        # _set_ownership unconditionally per-iteration.
        called_paths = [call.args[0] for call in set_ownership_mock.call_args_list]
        assert agents_dst in called_paths, (
            f"_set_ownership was not called against the per-user AGENTS.md: {called_paths}"
        )
