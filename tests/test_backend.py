"""
Tests for the agent backend abstraction module.

Tests the extracted context injection functions (build_session_context,
build_foreign_workspace_reminder, prepend_to_prompt, get_workspace_system_prompt)
and the ApiContext/AgentResponse/StreamEvent data types. These functions are pure
(no subprocess management), so tests are straightforward.
"""

from pathlib import Path
from unittest.mock import patch

from kai.backend import (
    AgentResponse,
    ApiContext,
    StreamEvent,
    build_foreign_workspace_reminder,
    build_session_context,
    ensure_user_memory,
    get_workspace_system_prompt,
    prepend_to_prompt,
)
from kai.config import WorkspaceConfig

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


# ── Test build_foreign_workspace_reminder ───────────────────────────


class TestBuildForeignWorkspaceReminder:
    """Tests for the per-message foreign workspace reminder."""

    def test_home_workspace_returns_none(self):
        """No reminder when workspace == home_workspace."""
        ws = Path("/opt/kai/home")
        assert build_foreign_workspace_reminder(ws, ws) is None

    def test_foreign_workspace_returns_reminder(self):
        """Reminder string returned when in a foreign workspace."""
        result = build_foreign_workspace_reminder(
            Path("/home/user/project"),
            Path("/opt/kai/home"),
        )
        assert result is not None
        assert "IMPORTANT" in result
        assert "Respond ONLY" in result


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
        assert resp.cost_usd == 0.0
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
        example_dir = project_root / "home" / ".claude"
        example_dir.mkdir(parents=True)
        (example_dir / "MEMORY.md.example").write_text("# Memory\n\n## About the User\n")

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
        # No example file under home/.claude/.
        (project_root / "home" / ".claude").mkdir(parents=True)

        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(7, data_dir)

        target = data_dir / "memory" / "7" / "MEMORY.md"
        assert target.is_file()
        assert target.read_text() == "# Memory\n"

    def test_idempotent_does_not_overwrite(self, tmp_path, monkeypatch):
        """Existing per-user MEMORY.md is preserved across repeated calls."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        example_dir = project_root / "home" / ".claude"
        example_dir.mkdir(parents=True)
        (example_dir / "MEMORY.md.example").write_text("TEMPLATE")

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
        example_dir = project_root / "home" / ".claude"
        example_dir.mkdir(parents=True)
        (example_dir / "MEMORY.md.example").write_text("# Memory\n\n## Dev\n")
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
        example_dir = project_root / "home" / ".claude"
        example_dir.mkdir(parents=True)
        (example_dir / "MEMORY.md.example").write_text("FRESH_TEMPLATE")
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
        (project_root / "home" / ".claude").mkdir(parents=True)
        monkeypatch.setattr("kai.backend.PROJECT_ROOT", project_root)

        ensure_user_memory(None, data_dir)

        legacy = data_dir / "memory" / "MEMORY.md"
        assert legacy.read_text() == "# Memory\n"


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
