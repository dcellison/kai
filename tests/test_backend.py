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
