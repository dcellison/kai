"""
Tests for the agent backend abstraction module.

Tests the extracted context injection functions (build_session_context,
build_foreign_workspace_reminder, prepend_to_prompt, get_workspace_system_prompt)
and the ApiContext/AgentResponse/StreamEvent data types. These functions are pure
(no subprocess management), so tests are straightforward.
"""

import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kai.backend import (
    AgentResponse,
    ApiContext,
    StreamEvent,
    build_foreign_workspace_reminder,
    build_session_context,
    ensure_user_home,
    ensure_user_memory,
    ensure_user_preferences,
    get_workspace_system_prompt,
    prepend_to_prompt,
    resolve_home_workspace,
)
from kai.config import Config, UserConfig, WorkspaceConfig

# ── Test build_session_context ──────────────────────────────────────


class TestBuildSessionContext:
    """Tests for the session context builder."""

    def _api(self, secret: str = "test-secret", port: int = 8080) -> ApiContext:
        """Create an ApiContext with sensible defaults."""
        return ApiContext(webhook_port=port, webhook_secret=secret)

    def test_home_workspace_no_identity(self, tmp_path):
        """No identity injection when workspace == home_workspace."""
        workspace = tmp_path / "home"
        workspace.mkdir()
        (workspace / ".claude").mkdir()
        (workspace / ".claude" / "CLAUDE.md").write_text("identity content")

        # Set up memory dir so the function doesn't fail
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        # Identity should NOT be injected (same workspace)
        assert "core identity and instructions" not in result

    def test_foreign_workspace_injects_identity(self, tmp_path):
        """Identity injected from home CLAUDE.md when in foreign workspace."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".claude").mkdir()
        (home / ".claude" / "CLAUDE.md").write_text("Be helpful.")

        foreign = tmp_path / "foreign"
        foreign.mkdir()

        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=foreign,
                home_workspace=home,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "[Your core identity and instructions:]" in result
        assert "Be helpful." in result

    def test_memory_exists(self, tmp_path):
        """Memory content included when file exists and is non-empty."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("User likes concise answers.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "User likes concise answers." in result
        assert "persistent memory" in result

    def test_memory_missing(self, tmp_path):
        """'(not yet created)' placeholder when memory file is missing."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)
        # No MEMORY.md file

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "(not yet created)" in result

    def test_memory_empty(self, tmp_path):
        """'(currently empty)' placeholder when memory file is blank."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "(currently empty)" in result

    def test_preferences_block_with_content(self, tmp_path):
        """PREFERENCES.md content is injected when chat_id is set and the file has content."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        # MEMORY.md path so the function does not fail on missing memory dir.
        (data_dir / "memory" / "42").mkdir(parents=True)
        # Per-user PREFERENCES.md exists with content.
        pref_dir = data_dir / "preferences" / "42"
        pref_dir.mkdir(parents=True)
        (pref_dir / "PREFERENCES.md").write_text("Use Celsius for temperatures.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=42,
                data_dir=data_dir,
            )

        assert "[Your personal preferences (file:" in result
        assert "Use Celsius for temperatures." in result

    def test_preferences_block_empty_file(self, tmp_path):
        """'(currently empty)' placeholder when PREFERENCES.md exists but is blank."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory" / "7").mkdir(parents=True)
        pref_dir = data_dir / "preferences" / "7"
        pref_dir.mkdir(parents=True)
        (pref_dir / "PREFERENCES.md").write_text("")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=7,
                data_dir=data_dir,
            )

        assert "[Your personal preferences (file:" in result
        # Two blocks may match the "(currently empty)" string (preferences
        # and memory). Both can be empty here; check the preferences
        # block contains it by anchoring on the block label.
        pref_block_idx = result.find("[Your personal preferences (file:")
        memory_block_idx = result.find("[Your persistent memory (file:")
        # Defensive: a -1 from str.find() would silently turn the slice
        # below into a near-full-string slice and produce a false pass.
        # Both labels are always emitted in this fixture (preferences
        # because chat_id is set, memory unconditionally), so a missing
        # block here is a regression worth surfacing immediately.
        assert memory_block_idx >= 0, "memory block label missing - inject ordering may have regressed"
        # Slice between the preferences block start and the memory block
        # start to verify the empty placeholder is in the preferences
        # block specifically.
        pref_block = result[pref_block_idx:memory_block_idx]
        assert "(currently empty)" in pref_block

    def test_preferences_block_missing_file(self, tmp_path):
        """'(not yet created)' placeholder when PREFERENCES.md is absent."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory" / "100").mkdir(parents=True)
        # No preferences/100/PREFERENCES.md.

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=100,
                data_dir=data_dir,
            )

        assert "[Your personal preferences (file:" in result
        assert "(not yet created)" in result

    def test_preferences_block_omitted_when_chat_id_none(self, tmp_path):
        """No preferences block when chat_id is None (no per-user file to read)."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        # chat_id=None means no per-user PREFERENCES.md to inject.
        assert "[Your personal preferences (file:" not in result

    def test_preferences_block_above_memory_block(self, tmp_path):
        """Block ordering: PREFERENCES.md appears BEFORE MEMORY.md in the assembled context."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory" / "55"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("memory content")
        pref_dir = data_dir / "preferences" / "55"
        pref_dir.mkdir(parents=True)
        (pref_dir / "PREFERENCES.md").write_text("preference content")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=55,
                data_dir=data_dir,
            )

        pref_idx = result.find("[Your personal preferences (file:")
        memory_idx = result.find("[Your persistent memory (file:")
        assert pref_idx >= 0, "preferences block missing"
        assert memory_idx >= 0, "memory block missing"
        # Rules out-rank facts; pin the relative ordering.
        assert pref_idx < memory_idx, "PREFERENCES.md block must appear before MEMORY.md block"

    def test_history_included(self, tmp_path):
        """Recent history is included when available."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value="User: hello\nKai: hi"):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=42,
                data_dir=data_dir,
            )

        assert result is not None
        assert "Recent conversations" in result
        assert "User: hello" in result

    def test_no_history_shows_grep_hint(self, tmp_path):
        """When no history, includes grep/jq search instruction."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "grep or jq" in result

    def test_api_docs_included(self, tmp_path):
        """Scheduling, messaging, and file API docs are injected."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(port=9090),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "Scheduling API" in result
        assert "localhost:9090" in result
        assert "Messaging API" in result
        assert "File API" in result

    def test_services_included(self, tmp_path):
        """External services block is injected when services are configured."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        api = ApiContext(
            webhook_port=8080,
            webhook_secret="secret",
            services_info=[
                {"name": "perplexity", "method": "POST", "description": "Web search"},
            ],
        )

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=api,
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "External Services" in result
        assert "perplexity" in result

    def test_chat_id_routing(self, tmp_path):
        """chat_id routing instruction is included when chat_id is set."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=12345,
                data_dir=data_dir,
            )

        assert result is not None
        assert "chat_id" in result
        assert "12345" in result

    def test_no_webhook_secret_omits_api_docs(self, tmp_path):
        """API docs are omitted when webhook secret is empty."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(secret=""),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        # No API docs without a secret
        assert "Scheduling API" not in result
        assert "Messaging API" not in result

    def test_workspace_system_prompt_included(self, tmp_path):
        """Workspace system prompt appears in the assembled context string."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        ws_config = WorkspaceConfig(
            path=workspace,
            system_prompt="Always respond in haiku form.",
        )

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=ws_config,
                chat_id=None,
                data_dir=data_dir,
            )

        assert result is not None
        assert "Workspace Instructions" in result
        assert "Always respond in haiku form." in result

    # ── Memory subsystem state marker (issue #403) ──────────────────

    def test_memory_subsystem_marker_enabled(self, tmp_path):
        """Marker line shows 'enabled' when memory_enabled=True is passed."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=True,
            )

        assert "[Memory subsystem: enabled]" in result
        assert "[Memory subsystem: disabled]" not in result

    def test_memory_subsystem_marker_disabled(self, tmp_path):
        """Marker line shows 'disabled' when memory_enabled=False is passed."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=False,
            )

        assert "[Memory subsystem: disabled]" in result
        assert "[Memory subsystem: enabled]" not in result

    def test_memory_subsystem_marker_emitted_when_chat_id_none(self, tmp_path):
        """Marker emits even when chat_id is None (per-deployment, not per-user)."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=True,
            )

        assert "[Memory subsystem: enabled]" in result

    def test_memory_subsystem_marker_emitted_in_home_workspace(self, tmp_path):
        """Marker emits even when workspace == home_workspace.

        The identity block is conditionally skipped in this case (lines 214-221
        in backend.py); the marker is not workspace-scoped and must always
        emit so the routing rule in CLAUDE.md / PREFERENCES.md has a uniformly-
        present signal to branch on.
        """
        workspace = tmp_path / "home"
        workspace.mkdir()
        (workspace / ".claude").mkdir()
        (workspace / ".claude" / "CLAUDE.md").write_text("identity content")
        data_dir = tmp_path / "data"
        (data_dir / "memory").mkdir(parents=True)

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=True,
            )

        assert "[Memory subsystem: enabled]" in result
        # Identity block correctly skipped (workspace == home_workspace,
        # so the file is never read). Assert against the file content
        # itself rather than the wrapper prose: a regression that broke
        # the skip condition but emitted only the wrapper without the
        # body would silently pass an "instructions" substring check.
        assert "identity content" not in result

    # ── MEMORY.md inject gate (issue #403) ──────────────────────────

    def test_memory_md_skipped_when_enabled(self, tmp_path):
        """MEMORY.md content is NOT injected when memory_enabled=True.

        In enabled mode, Qdrant is the active fact surface (retrieved via
        memory.format_context in claude.py). Injecting MEMORY.md would
        create a dual-source collision. The block must be omitted entirely
        even when MEMORY.md exists with content on disk.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("User likes concise answers.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=True,
            )

        assert "User likes concise answers." not in result
        assert "[Your persistent memory" not in result

    def test_memory_md_injected_when_disabled(self, tmp_path):
        """MEMORY.md IS injected when memory_enabled=False (current behavior preserved)."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("User likes concise answers.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
                memory_enabled=False,
            )

        assert "User likes concise answers." in result
        assert "[Your persistent memory" in result

    def test_memory_md_gate_per_user_chat_id(self, tmp_path):
        """The MEMORY.md inject gate applies on the per-user `chat_id is not None`
        branch the same way it does on the global branch.

        Both `chat_id=None` (global path at `memory/MEMORY.md`) and `chat_id=123`
        (per-user path at `memory/<chat_id>/MEMORY.md`, introduced in #347) must
        skip injection in enabled mode and inject in disabled mode. This pins the
        gate's symmetry across both code paths.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        per_user_dir = data_dir / "memory" / "123"
        per_user_dir.mkdir(parents=True)
        (per_user_dir / "MEMORY.md").write_text("Per-user fact.")

        # Disabled mode: per-user MEMORY.md IS injected.
        with patch("kai.backend.get_recent_history", return_value=""):
            disabled_result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=123,
                data_dir=data_dir,
                memory_enabled=False,
            )
        assert "Per-user fact." in disabled_result
        assert "memory/123/MEMORY.md" in disabled_result

        # Enabled mode: per-user MEMORY.md is NOT injected.
        with patch("kai.backend.get_recent_history", return_value=""):
            enabled_result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=123,
                data_dir=data_dir,
                memory_enabled=True,
            )
        assert "Per-user fact." not in enabled_result
        assert "[Your persistent memory" not in enabled_result


# ── Test build_foreign_workspace_reminder ───────────────────────────


class TestBuildForeignWorkspaceReminder:
    """Tests for the per-message foreign workspace reminder."""

    def test_home_workspace_returns_none(self):
        """No reminder when workspace == home_workspace."""
        # Path string is illustrative - the assertion compares via Path
        # equality and never touches the filesystem. Using the new
        # per-user shape from #353 keeps the test aligned with reality.
        ws = Path("/var/lib/kai/home/12345")
        assert build_foreign_workspace_reminder(ws, ws) is None

    def test_foreign_workspace_returns_reminder(self):
        """Reminder string returned when in a foreign workspace."""
        result = build_foreign_workspace_reminder(
            Path("/home/user/project"),
            Path("/var/lib/kai/home/12345"),
        )
        assert result is not None
        assert "IMPORTANT" in result
        assert "Respond ONLY" in result


# ── Test ensure_user_home / resolve_home_workspace (#353) ───────────


class TestEnsureUserHome:
    """
    Tests for the per-user home workspace bootstrap.

    Mirrors TestEnsureUserMemory shape since the two helpers are
    intentionally parallel (#353 reuses the #347 pattern).
    """

    def test_creates_directory(self, tmp_path):
        """First call materializes home/<chat_id>/ under data_dir."""
        result = ensure_user_home(12345, tmp_path)
        assert result == tmp_path / "home" / "12345"
        assert result.is_dir()

    def test_is_idempotent(self, tmp_path):
        """
        Repeated calls must not raise or rewrite content. The user owns
        the directory after first creation; clobbering it would lose
        user-authored files (cloned repos, notes, etc.).
        """
        first = ensure_user_home(12345, tmp_path)
        # Drop a sentinel so we can verify the second call leaves it alone.
        sentinel = first / "userfile.txt"
        sentinel.write_text("user content")

        second = ensure_user_home(12345, tmp_path)
        assert second == first
        assert sentinel.read_text() == "user content"

    def test_handles_none_chat_id(self, tmp_path):
        """
        chat_id=None routes to the fixed "anon" subdirectory used by
        admin-less startup paths (tests, health checks). Returning None
        would force defensive None-checks at every call site.
        """
        result = ensure_user_home(None, tmp_path)
        assert result == tmp_path / "home" / "anon"
        assert result.is_dir()

    def test_directory_mode_is_exactly_0755(self, tmp_path):
        """
        Mode must be EXACTLY 0o755 after ensure_user_home, regardless
        of the process umask. The umask on a hardened service is
        commonly 0o027, which would mask mkdir(mode=0o755) down to
        0o750 - blocking group traversal and breaking the inner
        subprocess when sudo -u targets a different identity. The
        helper must chmod explicitly after mkdir to force the intended
        bits; this test fails loudly if that chmod is ever removed.

        A weaker assertion (mode & 0o022 == 0) would accept 0o750 and
        silently mask a regression of exactly this kind.
        """
        # Set a hostile umask to simulate the production service config
        # (umask 0o027 on the launchd plist). Without the explicit chmod
        # in ensure_user_home, mkdir(mode=0o755) would return 0o750 here.
        import os as _os
        import stat

        prev_umask = _os.umask(0o027)
        try:
            result = ensure_user_home(99, tmp_path)
        finally:
            _os.umask(prev_umask)

        mode = stat.S_IMODE(result.stat().st_mode)
        assert mode == 0o755, f"expected 0o755, got {oct(mode)}"

    def test_seeds_claude_md_from_template(self, tmp_path):
        """
        Lazy bootstrap: ensure_user_home seeds <home>/.claude/CLAUDE.md
        from templates/.claude/CLAUDE.md when absent. Without this seed,
        a user added to users.yaml AFTER install (or any dev path
        without an install pass) gets an empty home workspace and the
        bot's identity-injection path silently fails on the missing
        file. Parallel to ensure_user_memory's MEMORY.md seed.
        """
        # Build a fake source tree under tmp_path so we can patch
        # PROJECT_ROOT to it. The real PROJECT_ROOT has a template, but
        # using the real path would make this test order-dependent on
        # whether the template currently exists in the working tree.
        fake_root = tmp_path / "src_root"
        template_dir = fake_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        template = template_dir / "CLAUDE.md"
        template.write_text("# Kai template\n")

        data_dir = tmp_path / "data"
        with patch("kai.backend.PROJECT_ROOT", fake_root):
            home = ensure_user_home(99, data_dir)

        claude_dst = home / ".claude" / "CLAUDE.md"
        assert claude_dst.is_file(), f"Expected lazy seed at {claude_dst}"
        assert claude_dst.read_text() == "# Kai template\n"

    def test_seed_is_idempotent(self, tmp_path):
        """An existing per-user CLAUDE.md is never overwritten on later calls."""
        fake_root = tmp_path / "src_root"
        template_dir = fake_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "CLAUDE.md").write_text("# UPSTREAM\n")

        data_dir = tmp_path / "data"
        with patch("kai.backend.PROJECT_ROOT", fake_root):
            home = ensure_user_home(99, data_dir)

        # Operator customizes after first seed.
        claude_dst = home / ".claude" / "CLAUDE.md"
        claude_dst.write_text("# OPERATOR CUSTOMIZED\n")

        # Second call must leave the customized content alone.
        with patch("kai.backend.PROJECT_ROOT", fake_root):
            ensure_user_home(99, data_dir)

        assert claude_dst.read_text() == "# OPERATOR CUSTOMIZED\n"

    def test_seed_writes_placeholder_when_template_missing(self, tmp_path):
        """Missing template: last-resort placeholder so the file is writable."""
        fake_root = tmp_path / "src_root"
        # Intentionally do not create templates/.claude/CLAUDE.md.
        (fake_root / "templates" / ".claude").mkdir(parents=True)

        data_dir = tmp_path / "data"
        with patch("kai.backend.PROJECT_ROOT", fake_root):
            home = ensure_user_home(99, data_dir)

        claude_dst = home / ".claude" / "CLAUDE.md"
        assert claude_dst.is_file()
        # Placeholder mirrors the MEMORY.md / PREFERENCES.md precedent.
        assert claude_dst.read_text() == "# Identity\n"

    def test_chmod_eperm_does_not_crash(self, tmp_path):
        """
        Regression for issue #454-adjacent install bug: when the per-user
        home directory already exists owned by a different OS user
        (because install.py `_apply_migrate` chowned it to the user's
        `os_user`), the chmod call must NOT propagate EPERM. macOS
        chmod(2) returns EPERM for any non-owner non-root caller, even
        when the target mode equals the current mode, so the chmod-as-
        no-op trick is not safe in a multi-user install.

        Without the swallow, `pool.get` → `_create_instance` →
        `resolve_home_workspace` → `ensure_user_home` raises, the bot
        cannot create the per-user backend, and every Telegram message
        for that chat fails with "Claude process died" with no further
        diagnostic.
        """
        # The directory exists when ensure_user_home is called (parents=
        # True + exist_ok=True mirrors that path). chmod then raises.
        target = tmp_path / "home" / "12345"
        target.mkdir(parents=True)

        real_chmod = os.chmod

        def chmod_eperm(path, mode, **kwargs):
            # Match the production failure shape: EPERM only on the
            # per-user dir we are trying to harden; let any other chmod
            # call (e.g. inside the .claude/ seed branch via shutil.copy2)
            # fall through to the real implementation. The **kwargs catch
            # is needed because shutil.copy2 passes follow_symlinks=True.
            if str(path) == str(target):
                raise PermissionError(1, "Operation not permitted", str(path))
            real_chmod(path, mode, **kwargs)

        with patch("kai.backend.os.chmod", side_effect=chmod_eperm):
            # Must not raise. The caller depends on this returning the
            # path so the session context build can proceed.
            result = ensure_user_home(12345, tmp_path)

        assert result == target

    def test_seed_swallows_oserror(self, tmp_path):
        """
        A permissions issue creating the seed must not crash the session
        init - the read path in build_session_context already handles
        missing files gracefully. Matches ensure_user_memory's OSError
        swallow contract.
        """
        # Build the source tree before patching so the setup mkdirs do
        # not hit the patched failure.
        fake_root = tmp_path / "src_root"
        (fake_root / "templates" / ".claude").mkdir(parents=True)
        (fake_root / "templates" / ".claude" / "CLAUDE.md").write_text("# Kai\n")

        def boom(*_args, **_kwargs) -> None:
            raise OSError("simulated permission denied")

        data_dir = tmp_path / "data"
        # Patch the seed step's copy2 to raise. The outer path.mkdir
        # for data_dir/home/<chat_id>/ runs cleanly (the function-
        # under-test handles its own dir creation before reaching the
        # seed branch); only the seed copy fails. Must not raise: the
        # function returns the path even when the seed fails, and the
        # caller may still write into the directory later.
        with (
            patch("kai.backend.PROJECT_ROOT", fake_root),
            patch("kai.backend.shutil.copy2", side_effect=boom),
        ):
            result = ensure_user_home(99, data_dir)

        assert result == data_dir / "home" / "99"
        # The seed file should NOT have been written; the OSError
        # handler should have logged and continued silently.
        assert not (result / ".claude" / "CLAUDE.md").exists()


class TestResolveHomeWorkspace:
    """
    Tests for the public resolver used by pool.py and bot.py.

    The resolver is the single source of truth for "where does this
    user's home live?" - both call sites must go through it so the
    answer cannot drift between session init and `/workspace home`.
    """

    def _config(self, user_configs: dict | None = None) -> Config:
        """Build a minimal Config with optional user_configs dict.

        Post-#565 tranche A `Config.user_configs` is a non-optional
        dict; a None argument here is coerced to `{}` to preserve the
        empty-dict test ergonomics callers expect.
        """
        return Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            user_configs=user_configs if user_configs is not None else {},
        )

    def test_prefers_users_yaml(self, tmp_path):
        """
        When users.yaml sets home_workspace, that path is returned
        verbatim. ensure_user_home is NOT called - the operator's
        choice wins outright.

        Passing data_dir= explicitly (rather than monkeypatching
        kai.backend.DATA_DIR) means a test that forgets the arg gets
        a real production DATA_DIR path, which is caught loudly rather
        than silently writing there.
        """
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        config = self._config(
            user_configs={42: UserConfig(telegram_id=42, name="u", home_workspace=explicit)},
        )
        data_dir = tmp_path / "data"

        result = resolve_home_workspace(42, config, data_dir=data_dir)
        assert result == explicit
        # Bug guard: data_dir/home/42 must NOT have been created.
        assert not (data_dir / "home" / "42").exists()

    def test_falls_back_to_data_dir(self, tmp_path):
        """
        When the user is in users.yaml without a home_workspace override,
        the resolver lands them in data_dir/home/<chat_id>/ via
        ensure_user_home (which creates the directory as a side effect).
        """
        config = self._config(user_configs={42: UserConfig(telegram_id=42, name="real-user")})
        result = resolve_home_workspace(42, config, data_dir=tmp_path)
        assert result == tmp_path / "home" / "42"
        assert result.is_dir()

    def test_anon_when_chat_id_none(self, tmp_path):
        """
        chat_id=None happens in admin-less startup paths. Return the
        fixed anon subdir so callers always get a concrete Path back.
        """
        config = self._config()
        result = resolve_home_workspace(None, config, data_dir=tmp_path)
        assert result == tmp_path / "home" / "anon"
        assert result.is_dir()

    def test_notification_only_chat_id_does_not_create_directory(self, tmp_path):
        """
        The originating-issue bug: a chat_id that has no users.yaml
        entry (a GitHub notification group chat, a deploy broadcast
        room) used to trigger ensure_user_home and silently mkdir
        an empty `home/<chat_id>/` directory on first contact. The
        directory then lingered indefinitely.

        With the notification-only check, the path is returned
        verbatim but the directory is NOT created. Downstream
        callers that try to use the path will fail naturally,
        which is the right behavior for a chat_id without an
        interactive session.

        Pinning both halves of the contract: path computed
        deterministically AND directory absence.
        """
        config = self._config(
            user_configs={42: UserConfig(telegram_id=42, name="real-user")},
        )
        notify_chat = -5241088228

        result = resolve_home_workspace(notify_chat, config, data_dir=tmp_path)

        assert result == tmp_path / "home" / str(notify_chat)
        assert not result.exists(), "notification-only chat_id must not auto-provision a home directory"


# ── Test prepend_to_prompt ──────────────────────────────────────────


class TestPrependToPrompt:
    """Tests for the prompt prepend helper."""

    def test_string_prompt(self):
        """Prefix is prepended to string prompts with separator."""
        result = prepend_to_prompt("hello", "[context]")
        assert result == "[context]\n\nhello"

    def test_list_prompt(self):
        """Prefix is inserted as a text block at the front of list prompts."""
        original = [{"type": "text", "text": "hello"}]
        result = prepend_to_prompt(original, "[context]")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "[context]"}
        assert result[1] == {"type": "text", "text": "hello"}

    def test_empty_prefix_noop(self):
        """Empty prefix returns the prompt unchanged."""
        assert prepend_to_prompt("hello", "") == "hello"
        original = [{"type": "text", "text": "hello"}]
        assert prepend_to_prompt(original, "") is original


# ── Test get_workspace_system_prompt ────────────────────────────────


class TestGetWorkspaceSystemPrompt:
    """Tests for the workspace system prompt reader."""

    def test_inline_prompt(self):
        """Returns the inline system_prompt when set."""
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt="Be concise.")
        assert get_workspace_system_prompt(ws_config) == "Be concise."

    def test_file_prompt(self, tmp_path):
        """Reads system prompt from file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Use pytest.")
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt_file=prompt_file)
        assert get_workspace_system_prompt(ws_config) == "Use pytest."

    def test_none_without_config(self):
        """Returns None when no workspace config is set."""
        assert get_workspace_system_prompt(None) is None

    def test_file_deleted_returns_none(self, tmp_path):
        """Returns None if the system_prompt_file no longer exists."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello")
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt_file=prompt_file)
        prompt_file.unlink()
        assert get_workspace_system_prompt(ws_config) is None


# ── Test data types ─────────────────────────────────────────────────


class TestDataTypes:
    """Smoke tests for the protocol data types."""

    def test_agent_response_defaults(self):
        """AgentResponse has sensible defaults."""
        resp = AgentResponse(success=True, text="hello")
        assert resp.session_id is None
        assert resp.duration_ms == 0
        assert resp.error is None

    def test_stream_event_defaults(self):
        """StreamEvent has sensible defaults."""
        event = StreamEvent(text_so_far="partial")
        assert event.done is False
        assert event.response is None

    def test_stream_event_done(self):
        """Final StreamEvent carries the complete response."""
        resp = AgentResponse(success=True, text="done")
        event = StreamEvent(text_so_far="done", done=True, response=resp)
        assert event.done is True
        assert event.response is resp

    def test_api_context_defaults(self):
        """ApiContext services_info defaults to empty list."""
        ctx = ApiContext(webhook_port=8080, webhook_secret="s")
        assert ctx.services_info == []


# ── Per-user MEMORY.md (#347) ───────────────────────────────────────
#
# #347: MEMORY.md moved from a single global file at
# DATA_DIR/memory/MEMORY.md to a per-user path at
# DATA_DIR/memory/<chat_id>/MEMORY.md. The tests below cover the four
# scoped behaviors called out in the issue's acceptance criteria:
#
#   (a) build_session_context reads the chat_id-scoped path when
#       chat_id is set;
#   (b) build_session_context falls back to the legacy global path
#       when chat_id is None (dev / single-user harnesses);
#   (c) ensure_user_memory seeds the per-user directory + file so a
#       user added post-install has a writable starting point;
#   (d) two chat_ids end up with independent files, so writes to one
#       user's memory never surface in another user's context.


class TestPerUserMemoryRead:
    """build_session_context reads the chat_id-scoped MEMORY.md."""

    def _api(self) -> ApiContext:
        return ApiContext(webhook_port=8080, webhook_secret="s")

    def test_reads_chat_scoped_path(self, tmp_path):
        """Memory at memory/<chat_id>/MEMORY.md is read when chat_id is set."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        user_memory_dir = data_dir / "memory" / "12345"
        user_memory_dir.mkdir(parents=True)
        (user_memory_dir / "MEMORY.md").write_text("Fact for user 12345.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=12345,
                data_dir=data_dir,
            )

        assert "Fact for user 12345." in result
        assert "persistent memory" in result

    def test_ignores_legacy_global_when_chat_id_set(self, tmp_path):
        """With chat_id set, the legacy global MEMORY.md is not read.

        Prevents a half-migrated install (legacy file still present,
        per-user dir empty) from silently leaking another user's notes.
        """
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_root = data_dir / "memory"
        memory_root.mkdir(parents=True)
        # Legacy global content that must NOT leak:
        (memory_root / "MEMORY.md").write_text("STALE GLOBAL DO NOT LEAK")
        # Per-user dir exists but is empty (no MEMORY.md yet):
        (memory_root / "99").mkdir()

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=99,
                data_dir=data_dir,
            )

        assert "STALE GLOBAL DO NOT LEAK" not in result
        # And the per-user read path reports "not yet created", matching
        # the OSError branch in build_session_context.
        assert "(not yet created)" in result

    def test_none_chat_id_uses_legacy_path(self, tmp_path):
        """chat_id=None falls back to the legacy global path (dev case)."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"
        memory_root = data_dir / "memory"
        memory_root.mkdir(parents=True)
        (memory_root / "MEMORY.md").write_text("Dev-mode single-user content.")

        with patch("kai.backend.get_recent_history", return_value=""):
            result = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=None,
                data_dir=data_dir,
            )

        assert "Dev-mode single-user content." in result


class TestEnsureUserMemory:
    """ensure_user_memory bootstraps the per-user directory + file."""

    def test_seeds_from_example_template(self, tmp_path, monkeypatch):
        """Creates memory/<chat_id>/MEMORY.md from the example template."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "MEMORY.md").write_text("# Memory\n\n## About the User\n")

        # ensure_user_memory reads PROJECT_ROOT module-level at call
        # time, so patching backend.PROJECT_ROOT is sufficient.
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(42, data_dir)

        target = data_dir / "memory" / "42" / "MEMORY.md"
        assert target.is_file()
        assert "About the User" in target.read_text()

    def test_no_template_creates_placeholder(self, tmp_path, monkeypatch):
        """Falls back to minimal '# Memory\\n' when no example template exists."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        # No template file under templates/.claude/.
        (project_root / "templates" / ".claude").mkdir(parents=True)

        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(7, data_dir)

        target = data_dir / "memory" / "7" / "MEMORY.md"
        assert target.is_file()
        assert target.read_text() == "# Memory\n"

    def test_idempotent_does_not_overwrite(self, tmp_path, monkeypatch):
        """Existing per-user MEMORY.md is preserved across repeated calls."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "MEMORY.md").write_text("TEMPLATE")

        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        user_dir = data_dir / "memory" / "100"
        user_dir.mkdir(parents=True)
        existing = user_dir / "MEMORY.md"
        existing.write_text("already-captured user content")

        ensure_user_memory(100, data_dir)

        # Same content, not the template.
        assert existing.read_text() == "already-captured user content"

    def test_chat_id_none_seeds_legacy_path(self, tmp_path, monkeypatch):
        """
        chat_id=None bootstraps the legacy data_dir/memory/MEMORY.md
        (no per-user subdir). This is the dev / no-users.yaml / memory-
        disabled path, where the previously removed _bootstrap_memory
        function used to create the directory + seed file. Without
        this, a fresh `python -m kai` would have no memory_root at
        all, and any inner Claude write attempt would FileNotFoundError
        on the missing parent.
        """
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "MEMORY.md").write_text("# Memory\n\n## Dev\n")
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(None, data_dir)

        legacy = data_dir / "memory" / "MEMORY.md"
        assert legacy.is_file()
        assert "Dev" in legacy.read_text()
        # No per-user subdir was created (chat_id=None must not
        # leak into a numeric subdirectory name).
        assert list((data_dir / "memory").iterdir()) == [legacy]

    def test_chat_id_none_idempotent(self, tmp_path, monkeypatch):
        """
        Repeated chat_id=None calls do not overwrite an existing
        legacy MEMORY.md. Same idempotence contract as the per-user
        branch - operators may have edited it between runs.
        """
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "MEMORY.md").write_text("FRESH_TEMPLATE")
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        legacy = memory_dir / "MEMORY.md"
        legacy.write_text("operator-edited content")

        ensure_user_memory(None, data_dir)

        assert legacy.read_text() == "operator-edited content"

    def test_chat_id_none_no_template_creates_placeholder(self, tmp_path, monkeypatch):
        """
        chat_id=None with no example template falls back to '# Memory\\n',
        matching the per-user branch and the old _bootstrap_memory.
        """
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        # No example template exists.
        (project_root / "templates" / ".claude").mkdir(parents=True)
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(None, data_dir)

        legacy = data_dir / "memory" / "MEMORY.md"
        assert legacy.read_text() == "# Memory\n"


class TestEnsureUserPreferences:
    """ensure_user_preferences bootstraps the per-user PREFERENCES.md surface."""

    def test_seeds_from_example_template(self, tmp_path, monkeypatch):
        """Creates preferences/<chat_id>/PREFERENCES.md from the example template."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "PREFERENCES.md").write_text("# Preferences\n\n## Style\n")

        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_preferences(42, data_dir)

        target = data_dir / "preferences" / "42" / "PREFERENCES.md"
        assert target.is_file()
        assert "Style" in target.read_text()

    def test_no_template_creates_placeholder(self, tmp_path, monkeypatch):
        """Falls back to '# Preferences\\n' placeholder when no example template exists."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        # Empty templates/.claude/, no PREFERENCES.md template.
        (project_root / "templates" / ".claude").mkdir(parents=True)
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_preferences(7, data_dir)

        target = data_dir / "preferences" / "7" / "PREFERENCES.md"
        assert target.is_file()
        assert target.read_text() == "# Preferences\n"

    def test_idempotent_does_not_overwrite(self, tmp_path, monkeypatch):
        """Existing per-user PREFERENCES.md is preserved across repeated calls."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "PREFERENCES.md").write_text("TEMPLATE")
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        user_dir = data_dir / "preferences" / "100"
        user_dir.mkdir(parents=True)
        existing = user_dir / "PREFERENCES.md"
        existing.write_text("operator-edited preferences")

        ensure_user_preferences(100, data_dir)

        assert existing.read_text() == "operator-edited preferences"

    def test_chat_id_none_returns_immediately(self, tmp_path, monkeypatch):
        """
        chat_id=None is a no-op for preferences. Unlike ensure_user_memory,
        which seeds a legacy global path for the dev / disabled-mode case,
        ensure_user_preferences has no global fallback because the inject
        path also skips the block when chat_id is None. No directory or
        file should be created.
        """
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        template_dir = project_root / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "PREFERENCES.md").write_text("TEMPLATE")
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_preferences(None, data_dir)

        # No preferences/ tree should have been created.
        assert not (data_dir / "preferences").exists()

    def test_oserror_swallowed(self, tmp_path, monkeypatch, caplog):
        """OSError raised inside the body is swallowed via log.warning, not propagated."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        (project_root / "templates" / ".claude").mkdir(parents=True)
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        # Force mkdir to raise; the function must swallow rather than
        # let the bot die mid-message-bootstrap.
        def boom(*args, **kwargs):
            raise PermissionError("mkdir denied")

        monkeypatch.setattr("pathlib.Path.mkdir", boom)

        with caplog.at_level(logging.WARNING, logger="kai.backend"):
            ensure_user_preferences(99, data_dir)

        # No exception escaped.
        assert any(
            "ensure_user_preferences" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )


class TestPerUserMemoryIsolation:
    """Writes to one user's MEMORY.md never appear in another user's context."""

    def _api(self) -> ApiContext:
        return ApiContext(webhook_port=8080, webhook_secret="s")

    def test_two_users_independent_files(self, tmp_path):
        """Two chat_ids get two distinct MEMORY.md files on disk, and
        build_session_context routes each request to the correct file."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / "data"

        user_a = data_dir / "memory" / "1001"
        user_b = data_dir / "memory" / "1002"
        user_a.mkdir(parents=True)
        user_b.mkdir(parents=True)
        (user_a / "MEMORY.md").write_text("USER_A_SECRET")
        (user_b / "MEMORY.md").write_text("USER_B_SECRET")

        with patch("kai.backend.get_recent_history", return_value=""):
            result_a = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=1001,
                data_dir=data_dir,
            )
            result_b = build_session_context(
                workspace=workspace,
                home_workspace=workspace,
                api=self._api(),
                workspace_config=None,
                chat_id=1002,
                data_dir=data_dir,
            )

        assert "USER_A_SECRET" in result_a
        assert "USER_B_SECRET" not in result_a
        assert "USER_B_SECRET" in result_b
        assert "USER_A_SECRET" not in result_b


# ── workspace_config.model apply-time validation ─────────────────────


class TestApplyWorkspaceModel:
    """Cross-backend workspace-override validation helper.

    workspaces.yaml is shared across users on different backends, so the
    file's `model:` field is intentionally loose at load time. Each
    backend validates the override at APPLY time and silently skips a
    mismatch (with WARNING log) so codex never gets a goose-only model
    and vice versa.
    """

    def test_no_workspace_config_keeps_current_model(self):
        from kai.backend import apply_workspace_model

        assert apply_workspace_model(None, "codex", "openai", "gpt-5.5") == "gpt-5.5"

    def test_workspace_without_model_keeps_current(self):
        from kai.backend import apply_workspace_model

        wc = WorkspaceConfig(path=Path("/tmp/ws"), model=None, timeout=None)
        assert apply_workspace_model(wc, "codex", "openai", "gpt-5.5") == "gpt-5.5"

    def test_codex_accepts_codex_compatible_override(self):
        from kai.backend import apply_workspace_model

        wc = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.5", timeout=None)
        assert apply_workspace_model(wc, "codex", "openai", "gpt-5.4") == "gpt-5.5"

    def test_codex_rejects_goose_only_override(self, caplog):
        from kai.backend import apply_workspace_model

        wc = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.4-nano", timeout=None)
        with caplog.at_level(logging.WARNING, logger="kai.backend"):
            result = apply_workspace_model(wc, "codex", "openai", "gpt-5.5")
        assert result == "gpt-5.5"
        assert any("Ignoring workspace model override" in r.message for r in caplog.records)

    def test_goose_openai_accepts_nano_override(self):
        from kai.backend import apply_workspace_model

        wc = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.4-nano", timeout=None)
        assert apply_workspace_model(wc, "goose", "openai", "gpt-5.4") == "gpt-5.4-nano"

    def test_claude_rejects_codex_override(self, caplog):
        from kai.backend import apply_workspace_model

        wc = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.5", timeout=None)
        with caplog.at_level(logging.WARNING, logger="kai.backend"):
            result = apply_workspace_model(wc, "claude", "anthropic", "sonnet")
        assert result == "sonnet"
        assert any("Ignoring workspace model override" in r.message for r in caplog.records)


# ── Backend-neutral per-turn context assembly (#516) ────────────────


class TestExtractTextQuery:
    """Tests for the raw user-text extraction used as the memory search query.

    The query must come from the original prompt BEFORE any context is
    prepended; if the helper saw the post-injection prompt, the embedding
    would be dominated by CLAUDE.md/MEMORY.md/history/API docs and
    fresh-session retrieval would be essentially random.
    """

    def test_string_prompt_returns_full_text(self):
        from kai.backend import extract_text_query

        assert extract_text_query("Hello there") == "Hello there"

    def test_empty_string_returns_empty(self):
        from kai.backend import extract_text_query

        assert extract_text_query("") == ""

    def test_list_prompt_returns_first_text_block(self):
        from kai.backend import extract_text_query

        prompt = [
            {"type": "image", "path": "/tmp/example.png"},
            {"type": "text", "text": "Describe this."},
        ]
        assert extract_text_query(prompt) == "Describe this."

    def test_list_prompt_no_text_block_returns_empty(self):
        from kai.backend import extract_text_query

        prompt = [{"type": "image", "path": "/tmp/example.png"}]
        assert extract_text_query(prompt) == ""

    def test_list_prompt_skips_malformed_entries(self):
        """A block whose `text` field is not a string (or that is missing
        the `type` key entirely) must not crash the helper. Pre-fix the
        loose `next(...)` form would have raised KeyError or returned a
        non-string; the defensive `isinstance` checks skip the bad entry
        and fall through to the next candidate.
        """
        from kai.backend import extract_text_query

        prompt = [
            {"type": "text"},  # missing 'text' key
            {"type": "text", "text": None},  # wrong type
            {"type": "text", "text": "real text"},
        ]
        assert extract_text_query(prompt) == "real text"


def _patch_legacy_recall(monkeypatch, *, rendered_context: str = "", recall_payload: dict | None = None):
    """Patch the new `format_context_with_recall_payload` helper to
    return a canned LegacyRecallResult. Most TestAssembleTurnContext
    tests want to assert how the rendered_context text gets prepended
    and do not care about the recall payload internals; this helper
    keeps each test focused on its actual assertion shape.

    Returns the AsyncMock so callers can inspect `.call_args` for
    query/user_id capture assertions."""
    from kai.memory import LegacyRecallResult

    payload = recall_payload if recall_payload is not None else {"reason": "ok", "hits": []}
    fake = AsyncMock(return_value=LegacyRecallResult(rendered_context=rendered_context, recall_payload=payload))
    monkeypatch.setattr("kai.memory.format_context_with_recall_payload", fake)
    return fake


class TestAssembleTurnContext:
    """Tests for the composed per-turn prompt assembly contract.

    The ordering invariant is the contract: marker closest to user text,
    then session_context, then memory, then workspace_reminder, with the
    inverse of that order applied via repeated prepends. Backends that
    re-implement this locally are exactly the failure mode the helper
    exists to prevent; these tests pin the invariant at the helper level
    so a future backend wire-up has a one-line integration target.
    """

    async def test_string_prompt_full_ordering(self, monkeypatch):
        """All four context layers present, string prompt. Final reading
        order must be reminder, memory, session, marker, user text, with
        only whitespace separating the marker and the user message.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        _patch_legacy_recall(monkeypatch, rendered_context="[Relevant memories]\n- fact one")
        result = await assemble_turn_context(
            "ACTUAL_USER_TEXT",
            chat_id=42,
            session_context="[SESSION]",
            workspace_reminder="[REMINDER]",
        )
        assert isinstance(result, str)
        assert result.count(USER_MESSAGE_MARKER) == 1
        # Inverse-prepend ordering: each subsequent layer lands above
        # the previous one, so the final reading order from top to
        # bottom is reminder, memory, session, marker, user text.
        positions = [
            result.index("[REMINDER]"),
            result.index("[Relevant memories]"),
            result.index("[SESSION]"),
            result.index(USER_MESSAGE_MARKER),
            result.index("ACTUAL_USER_TEXT"),
        ]
        assert positions == sorted(positions), positions
        # Only whitespace between the marker and the user message.
        marker_end = result.index(USER_MESSAGE_MARKER) + len(USER_MESSAGE_MARKER)
        user_start = result.index("ACTUAL_USER_TEXT")
        assert result[marker_end:user_start].strip() == "", repr(result[marker_end:user_start])

    async def test_format_context_receives_raw_query(self, monkeypatch):
        """The search query handed to legacy recall is the original user
        text, not the post-prepend prompt. Pre-fix-shape regression: if
        the helper extracted the query after session_context was applied,
        the embedding would be dominated by CLAUDE.md/history/API docs.
        """
        from kai.backend import assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="")
        await assemble_turn_context(
            "ACTUAL_USER_TEXT",
            chat_id=42,
            session_context="[10KB of CLAUDE.md and history]",
            workspace_reminder="",
        )
        assert fake.call_args.args == ("ACTUAL_USER_TEXT",)
        assert fake.call_args.kwargs["user_id"] == "42"

    async def test_no_memory_block_still_preserves_marker(self, monkeypatch):
        """When legacy recall returns the empty string (memory disabled
        or no relevant hits), the marker must still appear and remain
        the closest prefix to the user text.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        _patch_legacy_recall(monkeypatch, rendered_context="")
        result = await assemble_turn_context(
            "user message",
            chat_id=42,
            session_context="",
            workspace_reminder="",
        )
        assert isinstance(result, str)
        assert result.count(USER_MESSAGE_MARKER) == 1
        marker_end = result.index(USER_MESSAGE_MARKER) + len(USER_MESSAGE_MARKER)
        user_start = result.index("user message")
        assert result[marker_end:user_start].strip() == ""

    async def test_chat_id_none_skips_recall(self, monkeypatch):
        """chat_id=None must NOT call legacy recall. An empty or fake
        user_id would search across all users, which is a data isolation
        risk if a future memory implementation changes filtering
        semantics. The marker still applies.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="[Relevant memories]")
        result = await assemble_turn_context(
            "user message",
            chat_id=None,
            session_context="",
            workspace_reminder="",
        )
        fake.assert_not_called()
        assert USER_MESSAGE_MARKER in result

    async def test_whitespace_query_skips_recall(self, monkeypatch):
        """An empty or whitespace-only query produces arbitrary
        embedding hits; the gate must skip the call entirely rather
        than rely on the helper's own empty-query guard. The marker
        still applies.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="[Relevant memories]")
        result = await assemble_turn_context(
            "   ",
            chat_id=42,
            session_context="",
            workspace_reminder="",
        )
        fake.assert_not_called()
        assert USER_MESSAGE_MARKER in result

    async def test_list_prompt_preserves_original_block_order(self, monkeypatch):
        """List prompts get text prefixes inserted at the front; original
        content blocks (e.g. an image followed by a text block) must
        keep their relative order, and the memory query extraction must
        find the original text block, not an injected one.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="")
        original = [
            {"type": "image", "path": "/tmp/example.png"},
            {"type": "text", "text": "Describe this."},
        ]
        result = await assemble_turn_context(
            original,
            chat_id=42,
            session_context="",
            workspace_reminder="",
        )
        assert isinstance(result, list)
        assert fake.call_args.args == ("Describe this.",)
        # Injected text-prefix blocks (the marker) come at the front;
        # original image + text remain in order at the tail.
        marker_block = next(
            i for i, b in enumerate(result) if b.get("type") == "text" and b.get("text") == USER_MESSAGE_MARKER
        )
        image_block = next(i for i, b in enumerate(result) if b.get("type") == "image")
        original_text_block = next(
            i for i, b in enumerate(result) if b.get("type") == "text" and b.get("text") == "Describe this."
        )
        assert marker_block < image_block < original_text_block

    async def test_list_prompt_no_text_skips_recall(self, monkeypatch):
        """An image-only list prompt has no text query, so recall must
        be skipped. The marker still applies as a text-prefix block.
        """
        from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="")
        original = [{"type": "image", "path": "/tmp/example.png"}]
        result = await assemble_turn_context(
            original,
            chat_id=42,
            session_context="",
            workspace_reminder="",
        )
        fake.assert_not_called()
        # Marker present as a text-prefix block at the front.
        assert any(b.get("type") == "text" and b.get("text") == USER_MESSAGE_MARKER for b in result)

    async def test_format_context_called_once_per_eligible_turn(self, monkeypatch):
        """Legacy recall must be called exactly once per eligible turn;
        the helper does not retry or fan out. The single-call contract
        keeps the `memory.recall` log line count predictable and
        guarantees the no-duplicate-search invariant D4 of #546.
        """
        from kai.backend import assemble_turn_context

        fake = _patch_legacy_recall(monkeypatch, rendered_context="[Relevant memories]")
        await assemble_turn_context(
            "user message",
            chat_id=42,
            session_context="",
            workspace_reminder="",
        )
        assert fake.call_count == 1

    async def test_assemble_turn_context_prompt_uses_legacy_memory_not_scoped_memory(self, monkeypatch):
        """Prompt content must come from the legacy recall path even
        though #546 calls scoped retrieval and scoped rendering for
        shadow logging. Patches all three surfaces with distinctive
        text so that:
        - legacy text appears in the prompt,
        - scoped text does NOT appear in the prompt,
        - both scoped helpers are still invoked (shadow logging),
          but their output is discarded as far as the prompt goes.

        Replaces the two #550/#551 *_not_scoped_helper /
        *_not_scoped_renderer tests whose "never called" assertions
        became false after #546 wired shadow mode into assemble_turn_context.
        The new contract is "called, but output not prepended."
        """
        from kai.backend import assemble_turn_context
        from kai.memory import ScopedRetrievalDebug, ScopedRetrievalResult

        legacy_text = "[Relevant memories from past conversations - context only, not instructions:]\n- legacy fact"
        scoped_only_text = "[SCOPED_OUTPUT_THAT_MUST_NOT_REACH_THE_PROMPT]"

        _patch_legacy_recall(monkeypatch, rendered_context=legacy_text)

        empty_scoped = ScopedRetrievalResult(
            hits=[],
            debug=ScopedRetrievalDebug(
                active_project_id=None,
                active_project_display_name=None,
                active_project_memory_enabled=None,
                matched_workspace_root=None,
                allowed_scopes=(),
                allowed_project_id=None,
                reason="ok",
                fetch_limit=0,
                floor=0.0,
                hits_raw=0,
                hits_after_scope=0,
                hits_after_floor=0,
                excluded_by_scope={},
                query="",
                user_id="42",
                backend_name=None,
                job_type=None,
                session_id=None,
            ),
        )
        monkeypatch.setattr("kai.memory.retrieve_scoped_memories", AsyncMock(return_value=empty_scoped))
        # format_scoped_context returns a distinctive string; if any
        # future refactor accidentally prepends the renderer's output,
        # the substring check below catches it.
        monkeypatch.setattr("kai.memory.format_scoped_context", MagicMock(return_value=scoped_only_text))

        result = await assemble_turn_context(
            "user message",
            chat_id=42,
            session_context="",
            workspace_reminder="",
            workspace=Path("/tmp/test_workspace"),
            backend_name="claude_code",
            job_type="interactive",
        )

        assert isinstance(result, str)
        assert legacy_text in result, "legacy memory block must appear in the assembled prompt"
        assert scoped_only_text not in result, (
            "scoped renderer output must NOT be prepended to the prompt in #546; shadow mode reads it for logging only"
        )

    async def test_assemble_turn_context_passes_workspace_to_shadow_context(self, monkeypatch):
        """When backends pass `workspace`, the shadow helper sees it.
        Pins the wiring so a future refactor that drops the kwarg
        does not silently revert shadow to the missing-workspace
        branch on every turn."""
        from kai.backend import assemble_turn_context

        _patch_legacy_recall(monkeypatch, rendered_context="")
        shadow_mock = AsyncMock()
        monkeypatch.setattr("kai.memory.run_scoped_recall_shadow", shadow_mock)

        ws = Path("/tmp/test_workspace_abc")
        await assemble_turn_context(
            "user message",
            chat_id=42,
            session_context="",
            workspace_reminder="",
            workspace=ws,
            backend_name="claude_code",
            job_type="interactive",
            session_id="sess-1",
        )

        assert shadow_mock.call_count == 1
        kwargs = shadow_mock.call_args.kwargs
        assert kwargs["workspace"] == ws
        assert kwargs["backend_name"] == "claude_code"
        assert kwargs["job_type"] == "interactive"
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["user_id"] == "42"
        assert kwargs["query"] == "user message"

    async def test_assemble_turn_context_does_not_shadow_when_chat_id_none(self, monkeypatch):
        """Shadow mode runs under the same eligibility gate as
        live recall. chat_id=None skips legacy recall AND skips
        the shadow call entirely; no `memory.recall_shadow` line
        is emitted on these turns."""
        from kai.backend import assemble_turn_context

        legacy_mock = _patch_legacy_recall(monkeypatch, rendered_context="")
        shadow_mock = AsyncMock()
        monkeypatch.setattr("kai.memory.run_scoped_recall_shadow", shadow_mock)

        await assemble_turn_context(
            "user message",
            chat_id=None,
            session_context="",
            workspace_reminder="",
            workspace=Path("/tmp/ws"),
        )
        legacy_mock.assert_not_called()
        shadow_mock.assert_not_called()

    async def test_assemble_turn_context_does_not_shadow_when_query_empty(self, monkeypatch):
        """Same gate on the other axis: an empty / whitespace-only
        query skips both legacy recall and shadow. The shadow log
        is meaningful only when there is a legacy result to compare
        against."""
        from kai.backend import assemble_turn_context

        legacy_mock = _patch_legacy_recall(monkeypatch, rendered_context="")
        shadow_mock = AsyncMock()
        monkeypatch.setattr("kai.memory.run_scoped_recall_shadow", shadow_mock)

        await assemble_turn_context(
            "   ",
            chat_id=42,
            session_context="",
            workspace_reminder="",
            workspace=Path("/tmp/ws"),
        )
        legacy_mock.assert_not_called()
        shadow_mock.assert_not_called()

    async def test_assemble_turn_context_does_not_duplicate_legacy_search(self, monkeypatch):
        """D4 invariant: one eligible turn produces exactly one
        legacy Mem0 search (and at most one scoped search). The
        refactor that shares `format_context_with_recall_payload`
        between live prompt assembly and shadow logging exists to
        avoid running the legacy search twice; this test pins that
        property at the recall-helper boundary."""
        from kai.backend import assemble_turn_context

        legacy_mock = _patch_legacy_recall(monkeypatch, rendered_context="[mem]")
        shadow_mock = AsyncMock()
        monkeypatch.setattr("kai.memory.run_scoped_recall_shadow", shadow_mock)

        await assemble_turn_context(
            "user message",
            chat_id=42,
            session_context="",
            workspace_reminder="",
            workspace=Path("/tmp/ws"),
        )
        assert legacy_mock.call_count == 1, "legacy recall must run exactly once"
        assert shadow_mock.call_count == 1, "shadow must run exactly once when enabled"
