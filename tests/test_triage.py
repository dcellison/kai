"""Tests for triage.py issue triage pipeline."""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.triage import (
    IssueMetadata,
    _parse_triage_json,
    _sanitize_search_query,
    _send_error_notification,
    apply_triage,
    build_triage_prompt,
    extract_issue_metadata,
    list_projects,
    run_triage,
    search_related_issues,
    triage_issue,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _issue_payload(
    number: int = 1,
    title: str = "Test issue",
    body: str = "This is a test issue body.",
    author: str = "testuser",
    labels: list[dict] | None = None,
) -> dict:
    """Build a realistic GitHub issues webhook payload."""
    return {
        "action": "opened",
        "issue": {
            "number": number,
            "title": title,
            "body": body,
            "user": {"login": author},
            "html_url": f"https://github.com/owner/repo/issues/{number}",
            "labels": labels or [],
        },
        "repository": {"full_name": "owner/repo"},
    }


def _make_metadata(**kwargs) -> IssueMetadata:
    """Build an IssueMetadata with defaults, overriding with kwargs."""
    defaults = {
        "repo": "owner/repo",
        "number": 1,
        "title": "Test issue",
        "body": "This is a test issue body.",
        "author": "testuser",
        "url": "https://github.com/owner/repo/issues/1",
        "labels": [],
    }
    defaults.update(kwargs)
    return IssueMetadata(**defaults)


def _triage_result(**kwargs) -> dict:
    """Build a triage result dict with defaults, overriding with kwargs."""
    defaults = {
        "labels": ["bug"],
        "duplicate_of": None,
        "related": [],
        "project": None,
        "summary": "Test issue needs investigation.",
        "priority": "medium",
    }
    defaults.update(kwargs)
    return defaults


def _mock_subprocess(stdout: str = "", returncode: int = 0, stderr: str = ""):
    """Create a mock async subprocess with the given outputs."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    mock_proc.returncode = returncode
    mock_proc.kill = AsyncMock()
    mock_proc.wait = AsyncMock()
    return mock_proc


def _mock_aiohttp_session_post(status: int = 200) -> tuple[MagicMock, MagicMock]:
    """Build a mocked aiohttp ClientSession for the
    `async with session, session.post(...)` pattern.

    Returns (session, response). The session is a MagicMock so its
    `.post(...)` call returns an object that supports the async
    context manager protocol directly; the response is a MagicMock
    with `.status` defaulted to the supplied value. Pair with a
    `patch("kai.triage.aiohttp.ClientSession")` context manager and
    wire the patched class via `_attach_session_to_class()`.

    The bare `AsyncMock()` predecessor of this helper made
    `session.post` itself an AsyncMock, so the production
    `async with session.post(...) as resp:` evaluated `__aenter__`
    on an unawaited coroutine and silently failed inside the
    surrounding `try / except Exception:`. The helper uses a
    MagicMock session so `.post(...)` returns a normal object whose
    `__aenter__` resolves to the configured response.
    """
    session = MagicMock()
    response = MagicMock()
    response.status = status
    session.post.return_value.__aenter__ = AsyncMock(return_value=response)
    session.post.return_value.__aexit__ = AsyncMock(return_value=None)
    return session, response


def _attach_session_to_class(session: MagicMock, mock_cls: MagicMock) -> None:
    """Wire a patched `aiohttp.ClientSession` mock class to yield
    `session` from `async with aiohttp.ClientSession() as session:`."""
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)


# ── extract_issue_metadata ──────────────────────────────────────────


class TestExtractIssueMetadata:
    def test_extracts_all_fields(self):
        """All IssueMetadata fields are correctly extracted from a full payload."""
        payload = _issue_payload(
            number=42,
            title="Bug: login fails",
            body="Login button does nothing.",
            author="alice",
            labels=[{"name": "bug"}, {"name": "urgent"}],
        )
        meta = extract_issue_metadata(payload)
        assert meta.repo == "owner/repo"
        assert meta.number == 42
        assert meta.title == "Bug: login fails"
        assert meta.body == "Login button does nothing."
        assert meta.author == "alice"
        assert meta.url == "https://github.com/owner/repo/issues/42"
        assert meta.labels == ["bug", "urgent"]

    def test_missing_fields_default_gracefully(self):
        """Missing/empty fields in the payload produce safe defaults."""
        payload = {"issue": {}, "repository": {}}
        meta = extract_issue_metadata(payload)
        assert meta.repo == ""
        assert meta.number == 0
        assert meta.title == ""
        assert meta.body == ""
        assert meta.author == ""
        assert meta.url == ""
        assert meta.labels == []

    def test_none_body_becomes_empty_string(self):
        """A None body (common for issues with no description) becomes ""."""
        payload = _issue_payload()
        payload["issue"]["body"] = None
        meta = extract_issue_metadata(payload)
        assert meta.body == ""


# ── build_triage_prompt ─────────────────────────────────────────────


class TestBuildTriagePrompt:
    def test_contains_boundary_delimiters_and_schema(self):
        """Prompt includes randomized boundary delimiters and JSON schema instructions."""
        meta = _make_metadata(
            title="Widget breaks on save",
            labels=["bug"],
        )
        prompt = build_triage_prompt(meta, "[]", "[]")
        # Randomized boundary delimiters (partial match since tokens vary)
        assert "--- BEGIN ISSUE_METADATA" in prompt
        assert "--- END ISSUE_METADATA" in prompt
        assert "--- BEGIN ISSUE_BODY" in prompt
        assert "--- END ISSUE_BODY" in prompt
        assert "--- BEGIN RELATED_ISSUES" in prompt
        assert "--- END RELATED_ISSUES" in prompt
        assert "--- BEGIN AVAILABLE_PROJECTS" in prompt
        assert "--- END AVAILABLE_PROJECTS" in prompt
        # No static XML tags remain
        assert "<issue-metadata>" not in prompt
        assert "<issue-body>" not in prompt
        assert "<related-issues>" not in prompt
        assert "<available-projects>" not in prompt
        # JSON schema instructions
        assert '"labels"' in prompt
        assert '"duplicate_of"' in prompt
        assert '"priority"' in prompt
        # Preamble references boundaries, not XML
        assert "boundary" in prompt.lower()
        # Issue content
        assert "Widget breaks on save" in prompt
        assert "bug" in prompt

    def test_no_labels_shows_none(self):
        """When no labels exist, the prompt shows (none)."""
        meta = _make_metadata(labels=[])
        prompt = build_triage_prompt(meta, "[]", "[]")
        assert "(none)" in prompt

    def test_includes_related_and_projects(self):
        """Related issues and project data are included in the prompt."""
        related = json.dumps([{"number": 5, "title": "Similar bug"}])
        projects = json.dumps([{"title": "Sprint 1"}])
        meta = _make_metadata()
        prompt = build_triage_prompt(meta, related, projects)
        assert "Similar bug" in prompt
        assert "Sprint 1" in prompt


class TestBuildTriagePromptBoundaries:
    def test_each_block_unique_in_prompt(self):
        """Each block in a single prompt gets a different token."""
        meta = _make_metadata()
        prompt = build_triage_prompt(meta, "[]", "[]")
        tokens = re.findall(r"--- BEGIN \w+ ([0-9a-f]{8}) ---", prompt)
        # Should have 4 blocks: metadata, body, related, projects
        assert len(tokens) == 4
        # All tokens should be unique
        assert len(set(tokens)) == 4


# ── search_related_issues ──────────────────────────────────────────


class TestSearchRelatedIssues:
    @pytest.mark.asyncio
    async def test_success(self):
        """Successful search returns gh output as-is."""
        expected = json.dumps([{"number": 5, "title": "Related"}])
        mock_proc = _mock_subprocess(stdout=expected)

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await search_related_issues("owner/repo", "Test issue", "body")
        assert json.loads(result) == [{"number": 5, "title": "Related"}]

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        """A failed gh command returns empty JSON array, not an exception."""
        mock_proc = _mock_subprocess(returncode=1, stderr="auth failed")

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await search_related_issues("owner/repo", "Test issue", "body")
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        """An unexpected exception returns empty JSON array."""
        with patch(
            "kai.triage.asyncio.create_subprocess_exec",
            side_effect=OSError("no gh"),
        ):
            result = await search_related_issues("owner/repo", "Test issue", "body")
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_excludes_current_issue(self):
        """The current issue is excluded from its own search results."""
        results = [
            {"number": 10, "title": "Related issue"},
            {"number": 42, "title": "Current issue"},
            {"number": 20, "title": "Another related"},
        ]
        mock_proc = _mock_subprocess(stdout=json.dumps(results))

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await search_related_issues("owner/repo", "Test", "body", issue_number=42)

        parsed = json.loads(result)
        numbers = [i["number"] for i in parsed]
        assert 42 not in numbers
        assert 10 in numbers
        assert 20 in numbers

    @pytest.mark.asyncio
    async def test_empty_after_exclusion(self):
        """If the current issue is the only result, returns empty list."""
        results = [{"number": 42, "title": "Only me"}]
        mock_proc = _mock_subprocess(stdout=json.dumps(results))

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await search_related_issues("owner/repo", "Test", "body", issue_number=42)

        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self):
        """Invalid JSON from gh is handled gracefully."""
        mock_proc = _mock_subprocess(stdout="not valid json at all")

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await search_related_issues("owner/repo", "Test", "body", issue_number=1)

        assert result == "[]"


# ── list_projects ───────────────────────────────────────────────────


class TestListProjects:
    @pytest.mark.asyncio
    async def test_success(self):
        """Successful project listing returns gh output."""
        expected = json.dumps({"projects": [{"title": "Sprint 1", "number": 1}]})
        mock_proc = _mock_subprocess(stdout=expected)

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await list_projects("owner")
        assert "Sprint 1" in result

    @pytest.mark.asyncio
    async def test_no_projects(self):
        """Empty output returns empty JSON array."""
        mock_proc = _mock_subprocess(stdout="")

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await list_projects("owner")
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_failure_returns_empty(self):
        """A failed gh command returns empty JSON array."""
        mock_proc = _mock_subprocess(returncode=1, stderr="not found")

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await list_projects("owner")
        assert result == "[]"


# ── run_triage ──────────────────────────────────────────────────────


class TestRunTriageClaudeDispatch:
    """
    `run_triage` with the default claude backend dispatches to
    `ClaudeOneShotReasoner` (NOT an inline `claude --print` spawn).
    The reasoner owns binary resolution, the free-form plain-text
    argv, per-user os_user routing, and the allow-listed subprocess
    env; this class pins the dispatch contract: ctor kwargs, registry
    model resolution, override pass-through, raw-text return, and
    the collapse of typed reasoner errors to RuntimeError.
    """

    @staticmethod
    def _fake_reasoner(text='{"labels": []}'):
        from kai.oneshot import OneShotResult

        fake = MagicMock()
        fake.run = AsyncMock(return_value=OneShotResult(text=text, backend="claude", model="sonnet"))
        return fake

    @pytest.mark.asyncio
    async def test_run_triage_dispatches_to_claude_reasoner(self):
        """The claude branch builds a ClaudeOneShotReasoner with the
        os_user threaded through, awaits its run with the registry
        model in free-form mode, and returns the reasoner's text."""
        fake = self._fake_reasoner(text='{"labels": ["bug"], "summary": "A bug."}')

        with (
            patch("kai.triage.ClaudeOneShotReasoner", return_value=fake) as ctor,
            patch("kai.triage.asyncio.create_subprocess_exec") as mock_exec,
        ):
            result = await run_triage("triage prompt", claude_user="someone")

        assert result == '{"labels": ["bug"], "summary": "A bug."}'
        ctor.assert_called_once()
        assert ctor.call_args.kwargs == {"os_user": "someone"}
        fake.run.assert_awaited_once()
        run_kwargs = fake.run.call_args.kwargs
        assert run_kwargs["prompt"] == "triage prompt"
        assert run_kwargs["model"] == "sonnet"
        assert run_kwargs["purpose"] == "issue_triage"
        # Free-form mode: no json_schema kwarg reaches the reasoner;
        # triage's downstream parser owns the JSON contract.
        assert "json_schema" not in run_kwargs
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_override_param_wins_over_registry_default(self):
        """
        The `model_override` parameter (resolved by the caller in
        webhook.py from `user_config.models["issue_triage"]` and the
        load-time legacy env-var seeding) wins over the registry
        default. Pins the wiring that lets per-user
        `models.issue_triage` reach dispatch without the historic
        ISSUE_TRIAGE_MODEL_CLAUDE env-var read at the call site.
        """
        fake = self._fake_reasoner()

        with patch("kai.triage.ClaudeOneShotReasoner", return_value=fake):
            await run_triage("prompt", model_override="opus")

        assert fake.run.call_args.kwargs["model"] == "opus"

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_timeout_to_runtime_error(self):
        from kai.oneshot import OneShotTimeout

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotTimeout())

        with (
            patch("kai.triage.ClaudeOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Triage subprocess timed out"),
        ):
            await run_triage("prompt")

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_error_to_runtime_error(self):
        from kai.oneshot import OneShotSubprocessError

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotSubprocessError(returncode=1, stderr=b"model error"))

        with (
            patch("kai.triage.ClaudeOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Claude triage failed"),
        ):
            await run_triage("prompt")


# ── run_triage (Goose backend) ─────────────────────────────────────


class TestRunTriageGooseDispatch:
    """
    `run_triage` with `agent_backend="goose"` dispatches to
    `GooseOneShotReasoner` (NOT a direct `goose run` subprocess
    spawn). The reasoner owns the argv shape, GOOSE_BIN resolution,
    provider wire-name translation, per-user os_user routing, and
    the allow-listed subprocess env; this class pins the dispatch
    contract: ctor kwargs, per-provider model resolution from the
    registry, raw-text return, and the collapse of typed reasoner
    errors to RuntimeError.
    """

    @staticmethod
    def _fake_reasoner(text='{"labels": []}'):
        from kai.oneshot import OneShotResult

        fake = MagicMock()
        fake.run = AsyncMock(return_value=OneShotResult(text=text, backend="goose", model="claude-sonnet-4-6"))
        return fake

    @pytest.mark.asyncio
    async def test_run_triage_dispatches_to_goose_reasoner(self):
        """The goose branch builds a GooseOneShotReasoner with the
        os_user and Kai provider key threaded through, awaits its run
        with the registry model, and returns the raw text."""
        fake = self._fake_reasoner(text='{"labels": ["bug"]}')

        with (
            patch("kai.triage.GooseOneShotReasoner", return_value=fake) as ctor,
            patch("kai.triage.asyncio.create_subprocess_exec") as mock_exec,
        ):
            result = await run_triage(
                "triage prompt", agent_backend="goose", provider="anthropic", claude_user="someone"
            )

        assert result == '{"labels": ["bug"]}'
        ctor.assert_called_once()
        assert ctor.call_args.kwargs == {"os_user": "someone", "provider": "anthropic"}
        fake.run.assert_awaited_once()
        run_kwargs = fake.run.call_args.kwargs
        assert run_kwargs["prompt"] == "triage prompt"
        assert run_kwargs["model"] == "claude-sonnet-4-6"
        assert run_kwargs["purpose"] == "issue_triage"
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider", "expected_model"),
        [
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-5.5"),
            ("deepseek", "deepseek-v4-pro"),
            ("ollama", "llama4:70b"),
        ],
    )
    async def test_model_resolved_from_registry_per_provider(self, provider, expected_model):
        """(goose, <provider>, ISSUE_TRIAGE) registry rows reach the
        reasoner's model kwarg. The provider key passes through in
        Kai form (deepseek stays deepseek); the reasoner owns the
        custom_deepseek wire-name translation at argv build."""
        fake = self._fake_reasoner()

        with patch("kai.triage.GooseOneShotReasoner", return_value=fake) as ctor:
            await run_triage("prompt", agent_backend="goose", provider=provider)

        assert ctor.call_args.kwargs["provider"] == provider
        assert fake.run.call_args.kwargs["model"] == expected_model

    @pytest.mark.asyncio
    async def test_model_override_wins_over_registry(self):
        """Per-user `models.issue_triage` override flows through
        `model_override` and wins over the registry default."""
        fake = self._fake_reasoner()

        with patch("kai.triage.GooseOneShotReasoner", return_value=fake):
            await run_triage("prompt", agent_backend="goose", provider="ollama", model_override="qwen3:32b")

        assert fake.run.call_args.kwargs["model"] == "qwen3:32b"

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_timeout_to_runtime_error(self):
        from kai.oneshot import OneShotTimeout

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotTimeout())

        with (
            patch("kai.triage.GooseOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Triage subprocess timed out"),
        ):
            await run_triage("prompt", agent_backend="goose", provider="anthropic")

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_error_to_runtime_error(self):
        from kai.oneshot import OneShotSubprocessError

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotSubprocessError(returncode=1, stderr=b"provider error"))

        with (
            patch("kai.triage.GooseOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Goose triage failed"),
        ):
            await run_triage("prompt", agent_backend="goose", provider="anthropic")

    @pytest.mark.asyncio
    async def test_default_backend_is_claude(self):
        """Calling run_triage with no backend args still uses Claude
        (dispatching to ClaudeOneShotReasoner, not goose)."""
        from kai.oneshot import OneShotResult

        fake = MagicMock()
        fake.run = AsyncMock(return_value=OneShotResult(text='{"labels": []}', backend="claude", model="sonnet"))

        with patch("kai.triage.ClaudeOneShotReasoner", return_value=fake) as ctor:
            await run_triage("prompt")

        ctor.assert_called_once()
        fake.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_provider_raises(self):
        """Goose backend with empty provider raises ValueError early."""
        with pytest.raises(ValueError, match="provider is empty"):
            await run_triage("prompt", agent_backend="goose", provider="")


class TestRunTriageOpenCodeDispatch:
    """
    `run_triage` with `agent_backend="opencode"` dispatches to
    `OpenCodeOneShotReasoner`. The reasoner's `OneShotResult.text`
    becomes the return value; typed reasoner errors collapse to
    RuntimeError to keep the webhook handler's failure surface
    unchanged.
    """

    @pytest.mark.asyncio
    async def test_run_triage_dispatches_to_opencode_reasoner(self):
        from kai.oneshot import OneShotResult

        fake = MagicMock()
        fake.run = AsyncMock(
            return_value=OneShotResult(
                text='{"labels": []}',
                backend="opencode",
                model="anthropic/claude-sonnet-4-5",
            )
        )

        with (
            patch("kai.triage.OpenCodeOneShotReasoner", return_value=fake) as ctor,
            patch("kai.triage.asyncio.create_subprocess_exec") as mock_exec,
        ):
            result = await run_triage(
                "prompt body", agent_backend="opencode", provider="anthropic", claude_user="someone"
            )

        assert result == '{"labels": []}'
        ctor.assert_called_once()
        assert ctor.call_args.kwargs == {"os_user": "someone"}
        fake.run.assert_awaited_once()
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_timeout_to_runtime_error(self):
        from kai.oneshot import OneShotTimeout

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotTimeout())

        with (
            patch("kai.triage.OpenCodeOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Triage subprocess timed out"),
        ):
            await run_triage("prompt", agent_backend="opencode", provider="anthropic")

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_error_to_runtime_error(self):
        from kai.oneshot import OneShotOutputError

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotOutputError("schema bad"))

        with (
            patch("kai.triage.OpenCodeOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"OpenCode triage failed.*schema bad"),
        ):
            await run_triage("prompt", agent_backend="opencode", provider="anthropic")


class TestRunTriageCodexDispatch:
    """
    `run_triage` with `agent_backend="codex"` dispatches to
    `CodexOneShotReasoner` (NOT an inline `codex exec` spawn). The
    reasoner owns the `codex exec --json` argv, CODEX_BIN resolution,
    per-user os_user routing, the allow-listed subprocess env, and
    the NDJSON event walk; this class pins the dispatch contract:
    ctor kwargs (no join_items override, so last-wins extraction
    protects the one-JSON-object contract), registry model
    resolution, override pass-through, raw-text return, and the
    collapse of typed reasoner errors to RuntimeError.
    """

    @staticmethod
    def _fake_reasoner(text='{"labels": []}'):
        from kai.oneshot import OneShotResult

        fake = MagicMock()
        fake.run = AsyncMock(return_value=OneShotResult(text=text, backend="codex", model="gpt-5.5"))
        return fake

    @pytest.mark.asyncio
    async def test_run_triage_dispatches_to_codex_reasoner(self):
        """The codex branch builds a CodexOneShotReasoner with the
        os_user threaded through, awaits its run with the registry
        model, and returns the reasoner's text."""
        fake = self._fake_reasoner(text='{"labels": ["bug"], "summary": "A bug."}')

        with (
            patch("kai.triage.CodexOneShotReasoner", return_value=fake) as ctor,
            patch("kai.triage.asyncio.create_subprocess_exec") as mock_exec,
        ):
            result = await run_triage("triage prompt", agent_backend="codex", claude_user="someone")

        assert result == '{"labels": ["bug"], "summary": "A bug."}'
        ctor.assert_called_once()
        assert ctor.call_args.kwargs == {"os_user": "someone"}
        fake.run.assert_awaited_once()
        run_kwargs = fake.run.call_args.kwargs
        assert run_kwargs["prompt"] == "triage prompt"
        assert run_kwargs["model"] == "gpt-5.5"
        assert run_kwargs["purpose"] == "issue_triage"
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_join_items_not_passed_keeps_last_wins(self):
        """
        Triage constructs the reasoner WITHOUT join_items, leaving
        the last-wins default in place: the downstream parser expects
        exactly one JSON object, and joining a preamble agent_message
        ahead of the JSON body would corrupt it (review passes
        join_items=True for its free-form markdown instead).
        """
        fake = self._fake_reasoner()

        with patch("kai.triage.CodexOneShotReasoner", return_value=fake) as ctor:
            await run_triage("prompt", agent_backend="codex")

        assert "join_items" not in ctor.call_args.kwargs

    @pytest.mark.asyncio
    async def test_codex_model_override_param_wins(self):
        """
        On the codex branch, the `model_override` parameter (resolved
        by the caller from per-user `models.issue_triage` and the
        load-time legacy env-var seeding) wins over the registry
        default the same way it does on the claude branch.
        """
        fake = self._fake_reasoner()

        with patch("kai.triage.CodexOneShotReasoner", return_value=fake):
            await run_triage("prompt", agent_backend="codex", model_override="gpt-5.4")

        assert fake.run.call_args.kwargs["model"] == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_timeout_to_runtime_error(self):
        from kai.oneshot import OneShotTimeout

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotTimeout())

        with (
            patch("kai.triage.CodexOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Triage subprocess timed out"),
        ):
            await run_triage("prompt", agent_backend="codex")

    @pytest.mark.asyncio
    async def test_run_triage_collapses_oneshot_error_to_runtime_error(self):
        from kai.oneshot import OneShotSubprocessError

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=OneShotSubprocessError(returncode=1, stderr=b"auth failed"))

        with (
            patch("kai.triage.CodexOneShotReasoner", return_value=fake),
            pytest.raises(RuntimeError, match=r"Codex triage failed"),
        ):
            await run_triage("prompt", agent_backend="codex")


class TestGooseTriageModelResolutionViaRegistry:
    """Goose triage model selection now flows through the unified
    (backend, provider, role) MODEL_REGISTRY. The legacy
    _resolve_goose_model helper retired with this spec; per-user
    overrides live in users.yaml `models.issue_triage`."""

    def test_curated_provider_registry_lookup(self):
        """Goose+openai resolves the ISSUE_TRIAGE row from the registry."""
        from kai.config import MODEL_REGISTRY, ModelRole

        assert MODEL_REGISTRY[("goose", "openai", ModelRole.ISSUE_TRIAGE)] == "gpt-5.5"

    def test_open_ended_provider_registry_lookup(self):
        """Goose+ollama has a registry-shipped default that operators
        override per-user via users.yaml `models.issue_triage`."""
        from kai.config import MODEL_REGISTRY, ModelRole

        assert MODEL_REGISTRY[("goose", "ollama", ModelRole.ISSUE_TRIAGE)] == "llama4:70b"


# ── _parse_triage_json ──────────────────────────────────────────────


class TestParseTriageJson:
    def test_clean_json(self):
        """Clean JSON string parses correctly."""
        raw = '{"labels": ["bug"], "priority": "high"}'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["bug"], "priority": "high"}

    def test_with_markdown_fencing(self):
        """JSON wrapped in ```json ... ``` is parsed correctly."""
        raw = '```json\n{"labels": ["bug"]}\n```'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["bug"]}

    def test_with_bare_fencing(self):
        """JSON wrapped in ``` ... ``` (no language tag) is parsed correctly."""
        raw = '```\n{"labels": ["enhancement"]}\n```'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["enhancement"]}

    def test_invalid_json(self):
        """Non-JSON string raises ValueError with clear message."""
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_triage_json("This is not JSON at all")

    def test_json_array_raises(self):
        """A JSON array (not object) raises ValueError."""
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_triage_json("[1, 2, 3]")

    def test_whitespace_padding(self):
        """Whitespace around JSON is handled."""
        raw = '  \n  {"labels": []}  \n  '
        result = _parse_triage_json(raw)
        assert result == {"labels": []}

    def test_fencing_without_newline(self):
        """Fencing like ```{"labels": []}``` (no newline) is handled."""
        raw = '```{"labels": ["bug"]}```'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["bug"]}

    def test_preamble_before_json(self):
        """JSON preceded by preamble text is extracted."""
        raw = 'Here is the analysis:\n{"labels": ["bug"], "priority": "high"}'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["bug"], "priority": "high"}

    def test_preamble_with_braces(self):
        """Preamble containing braces doesn't confuse the extractor."""
        raw = 'Here\'s the {"quick": "note"} before the real response:\n{"labels": ["bug"], "priority": "high"}'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["bug"], "priority": "high"}

    def test_preamble_with_multiple_brace_groups(self):
        """Multiple brace groups in preamble are skipped to find valid JSON."""
        raw = 'See {x} and {y: z} for context.\n{"labels": ["enhancement"]}'
        result = _parse_triage_json(raw)
        assert result == {"labels": ["enhancement"]}

    def test_no_valid_json_raises(self):
        """Text with braces but no valid JSON still raises ValueError."""
        raw = "Here is {some broken and {nested stuff}"
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_triage_json(raw)


# ── _sanitize_search_query ──────────────────────────────────────────


class TestSanitizeSearchQuery:
    def test_strips_special_chars(self):
        """Quotes and special characters are stripped."""
        assert _sanitize_search_query('"Bug: [urgent]"') == "Bug urgent"

    def test_caps_at_128(self):
        """Query is capped at 128 characters."""
        long_title = "A" * 200
        result = _sanitize_search_query(long_title)
        assert len(result) == 128

    def test_collapses_spaces(self):
        """Multiple spaces are collapsed to one."""
        assert _sanitize_search_query("too   many    spaces") == "too many spaces"

    def test_empty_title(self):
        """Empty title returns empty string."""
        assert _sanitize_search_query("") == ""

    def test_preserves_hyphens(self):
        """Hyphens are preserved in search queries."""
        assert _sanitize_search_query("fix-login-bug") == "fix-login-bug"


# ── apply_triage ────────────────────────────────────────────────────


class TestApplyTriage:
    @pytest.mark.asyncio
    async def test_applies_labels(self):
        """Labels from triage result are applied via gh issue edit."""
        meta = _make_metadata(labels=[])
        result = _triage_result(labels=["bug", "enhancement"])

        # Track which commands were run
        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            # Return accurate label search results: "bug" exists, others don't
            if "label" in args and "list" in args and "--search" in args:
                search_term = args[list(args).index("--search") + 1]
                if search_term == "bug":
                    return _mock_subprocess(stdout='[{"name": "bug"}]')
                return _mock_subprocess(stdout="[]")
            return _mock_subprocess(stdout="")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # Verify only existing repo labels are applied. Missing labels are skipped.
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        applied_labels = {cmd[list(cmd).index("--add-label") + 1] for cmd in add_label_calls}
        assert applied_labels == {"bug"}

        body = mock_session.post.call_args[1]["json"]
        assert "Skipped labels: enhancement" in body["text"]

        # Verify a comment was posted
        comment_calls = [cmd for cmd in commands_run if "issue" in cmd and "comment" in cmd]
        assert len(comment_calls) == 1

    @pytest.mark.asyncio
    async def test_skips_existing_labels(self):
        """Labels already on the issue are not re-applied."""
        meta = _make_metadata(labels=["bug"])
        result = _triage_result(labels=["bug", "enhancement"])

        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # "bug" should not appear in any --add-label call
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        for call in add_label_calls:
            label_idx = list(call).index("--add-label") + 1
            assert call[label_idx] != "bug"

    @pytest.mark.asyncio
    async def test_project_assignment(self):
        """When project is set, gh project item-add is called."""
        meta = _make_metadata()
        result = _triage_result(project="Sprint 1")

        # Pass the project list JSON directly (no longer fetched inside apply_triage)
        projects_json = json.dumps({"projects": [{"title": "Sprint 1", "number": 1}]})
        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(
                meta,
                result,
                8080,
                "secret",
                projects_json=projects_json,
                allowed_triage_projects=["Sprint 1"],
            )

        assert mock_session.post.called
        # Should have called gh project item-add
        item_add_calls = [cmd for cmd in commands_run if "project" in cmd and "item-add" in cmd]
        assert len(item_add_calls) > 0

    @pytest.mark.asyncio
    async def test_project_assignment_requires_allowlist(self):
        """Model-selected projects are skipped unless explicitly allowlisted."""
        meta = _make_metadata()
        result = _triage_result(project="Sprint 1")
        projects_json = json.dumps({"projects": [{"title": "Sprint 1", "number": 1}]})

        commands_run = []
        captured: dict[str, str | None] = {"comment": None}

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            if "issue" in args and "comment" in args and "--body-file" in args:
                body_path = args[list(args).index("--body-file") + 1]
                captured["comment"] = Path(body_path).read_text()
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret", projects_json=projects_json)

        assert mock_session.post.called
        item_add_calls = [cmd for cmd in commands_run if "project" in cmd and "item-add" in cmd]
        assert len(item_add_calls) == 0
        assert "**Added to project:** Sprint 1" not in captured["comment"]
        assert "**Project skipped:** Sprint 1" in captured["comment"]
        body = mock_session.post.call_args[1]["json"]
        assert "Skipped project: Sprint 1" in body["text"]

    @pytest.mark.asyncio
    async def test_no_project_skips_assignment(self):
        """When project is null, no project commands are run."""
        meta = _make_metadata()
        result = _triage_result(project=None)

        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # No project item-add calls
        item_add_calls = [cmd for cmd in commands_run if "project" in cmd and "item-add" in cmd]
        assert len(item_add_calls) == 0

    @pytest.mark.asyncio
    async def test_posts_comment(self):
        """Triage comment is posted via gh issue comment."""
        meta = _make_metadata()
        result = _triage_result(summary="Needs a fix for the widget.")

        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # Should have called gh issue comment
        comment_calls = [cmd for cmd in commands_run if "issue" in cmd and "comment" in cmd]
        assert len(comment_calls) > 0

    @pytest.mark.asyncio
    async def test_sends_telegram(self):
        """Telegram notification is sent via the send-message API."""
        meta = _make_metadata(title="Widget bug")
        result = _triage_result(priority="high")

        async def mock_exec(*args, **kwargs):
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        # Verify the send-message call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "send-message" in call_args[0][0]
        body = call_args[1]["json"]
        assert "Widget bug" in body["text"]
        assert "high" in body["text"]

    @pytest.mark.asyncio
    async def test_skips_missing_labels(self):
        """Labels that don't exist in the repo are skipped, not created."""
        meta = _make_metadata(labels=[])
        result = _triage_result(labels=["custom-label"])

        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            # label list returns empty (label doesn't exist)
            if "label" in args and "list" in args:
                return _mock_subprocess(stdout="[]")
            return _mock_subprocess(stdout="")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # Missing labels must not be created or applied.
        create_calls = [cmd for cmd in commands_run if "label" in cmd and "create" in cmd]
        assert len(create_calls) == 0
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        assert len(add_label_calls) == 0

        body = mock_session.post.call_args[1]["json"]
        assert "Labels: (none added)" in body["text"]
        assert "Skipped labels: custom-label" in body["text"]

    @pytest.mark.asyncio
    async def test_labels_string_ignored(self):
        """If Claude returns labels as a string instead of list, no labels are applied."""
        meta = _make_metadata(labels=[])
        result = _triage_result(labels="bug")  # type: ignore[arg-type]

        commands_run = []

        async def mock_exec(*args, **kwargs):
            commands_run.append(args)
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # No --add-label calls should have been made
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        assert len(add_label_calls) == 0


# ── apply_triage actionability fields (#690) ────────────────────────


async def _apply_triage_capture(meta: IssueMetadata, result: dict, **kwargs) -> tuple[str, str]:
    """Run apply_triage and capture the rendered comment / Telegram bodies.

    Reads the comment body from the tempfile path passed to
    `gh issue comment --body-file`; the tempfile is still live inside
    the mocked subprocess callback so we can read it before
    apply_triage's `with tempfile.TemporaryDirectory(...)` block
    cleans up. Reads the Telegram body from the mocked aiohttp post
    call's json payload.
    """
    captured: dict[str, str | None] = {"comment": None, "telegram": None}

    async def mock_exec(*args, **_kwargs):
        if "label" in args and "list" in args and "--search" in args:
            search_term = args[list(args).index("--search") + 1]
            return _mock_subprocess(stdout=json.dumps([{"name": search_term}]))
        if "issue" in args and "comment" in args and "--body-file" in args:
            body_path = args[list(args).index("--body-file") + 1]
            captured["comment"] = Path(body_path).read_text()
        return _mock_subprocess(stdout="[]")

    with (
        patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
        patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_session, _ = _mock_aiohttp_session_post(status=200)
        _attach_session_to_class(mock_session, mock_session_cls)
        await apply_triage(meta, result, 8080, "secret", **kwargs)
        if mock_session.post.called:
            captured["telegram"] = mock_session.post.call_args[1]["json"]["text"]

    assert captured["comment"] is not None, "apply_triage did not post a comment"
    assert captured["telegram"] is not None, "apply_triage did not send a Telegram summary"
    return captured["comment"], captured["telegram"]


class TestTriageActionability:
    """#690: status / next_action / missing_info / blocked_by output fields."""

    # ── Happy path: one fixture per status value ────────────────────

    @pytest.mark.asyncio
    async def test_ready_status_renders_status_and_next_action(self):
        meta = _make_metadata()
        result = _triage_result(
            status="ready",
            next_action="Start work on the foo module.",
            missing_info=[],
            blocked_by=None,
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment
        assert "**Next action:** Start work on the foo module." in comment
        assert "**Missing information:**" not in comment
        assert "**Blocked by:**" not in comment
        assert "Status: ready" in telegram
        assert "Next: Start work on the foo module." in telegram
        assert "Needs info:" not in telegram
        assert "Blocked by:" not in telegram

    @pytest.mark.asyncio
    async def test_needs_info_renders_missing_info_block(self):
        meta = _make_metadata()
        questions = [
            "What is the exact command you ran?",
            "What was the expected vs actual output?",
        ]
        result = _triage_result(
            status="needs_info",
            next_action="Ask the reporter for reproduction details.",
            missing_info=questions,
            blocked_by=None,
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** needs_info" in comment
        assert "**Missing information:**" in comment
        for question in questions:
            assert f"- {question}" in comment
        assert "Status: needs_info" in telegram
        assert "Needs info: 2 questions" in telegram

    @pytest.mark.asyncio
    async def test_wontfix_candidate_renders_status(self):
        meta = _make_metadata()
        result = _triage_result(
            status="wontfix_candidate",
            next_action="Confirm scope, then close or apply the wontfix label.",
            missing_info=[],
            blocked_by=None,
        )
        commands_run = []

        async def mock_exec(*args, **_kwargs):
            commands_run.append(args)
            if "issue" in args and "comment" in args and "--body-file" in args:
                pass
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret")

        assert mock_session.post.called
        # Comment renders the status.
        comment_calls = [cmd for cmd in commands_run if "issue" in cmd and "comment" in cmd]
        assert len(comment_calls) == 1
        # No automatic close: gh issue close must NEVER be invoked.
        close_calls = [cmd for cmd in commands_run if "issue" in cmd and "close" in cmd]
        assert close_calls == []

    @pytest.mark.asyncio
    async def test_blocked_status_renders_blocked_by_int(self):
        meta = _make_metadata()
        result = _triage_result(
            status="blocked",
            next_action="Track #42, then revisit.",
            missing_info=[],
            blocked_by=42,
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** blocked" in comment
        assert "**Blocked by:** #42" in comment
        assert "Status: blocked" in telegram
        assert "Blocked by: #42" in telegram

    @pytest.mark.asyncio
    async def test_blocked_status_renders_blocked_by_string(self):
        meta = _make_metadata()
        result = _triage_result(
            status="blocked",
            next_action="Track the upstream pytest fix, then revisit.",
            missing_info=[],
            blocked_by="upstream pytest fix",
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** blocked" in comment
        assert "**Blocked by:** upstream pytest fix" in comment
        assert "Blocked by: upstream pytest fix" in telegram

    # ── Consistency fallback tests ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_blocked_status_with_null_blocked_by_downgrades_to_ready(self, caplog):
        meta = _make_metadata()
        result = _triage_result(
            status="blocked",
            next_action="Track the dependency.",
            missing_info=[],
            blocked_by=None,
        )
        with caplog.at_level("WARNING", logger="kai.triage"):
            comment, _telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment
        assert "**Blocked by:**" not in comment
        assert any("blocked with null blocked_by" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_blocked_status_with_bool_blocked_by_rejects_and_downgrades(self, caplog):
        """Python bool is a subclass of int; the parser must reject it explicitly.

        Without the explicit bool exclusion, `"blocked_by": true` from
        a malformed model response would be accepted as a valid
        integer blocker (since `isinstance(True, int)` is True) and
        render as "Blocked by: #True" in the public-facing comment.
        The parser rejects the bool, leaving `blocked_by` None, which
        then triggers the blocked-with-null consistency check and
        downgrades status to ready.
        """
        meta = _make_metadata()
        result = _triage_result(
            status="blocked",
            next_action="Track the dependency.",
            missing_info=[],
            blocked_by=True,
        )
        with caplog.at_level("WARNING", logger="kai.triage"):
            comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment
        assert "**Blocked by:**" not in comment
        assert "Blocked by:" not in telegram
        assert "#True" not in comment
        assert "#True" not in telegram
        assert any("blocked with null blocked_by" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_blocked_status_with_blocked_by_ignores_blocked_by(self, caplog):
        meta = _make_metadata()
        result = _triage_result(
            status="ready",
            next_action="Start work.",
            missing_info=[],
            blocked_by=42,
        )
        with caplog.at_level("WARNING", logger="kai.triage"):
            comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment
        assert "**Blocked by:**" not in comment
        assert "Blocked by:" not in telegram
        assert any("blocked_by populated but status is ready" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_invalid_status_falls_back_to_ready_with_raw_summary(self):
        meta = _make_metadata()
        result = _triage_result(
            status="frobnicate",
            summary="The login button is misaligned on Safari 17.",
            missing_info=[],
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment

    @pytest.mark.asyncio
    async def test_invalid_status_falls_back_to_needs_info_with_questions(self):
        meta = _make_metadata()
        result = _triage_result(
            status="frobnicate",
            summary="The login button is misaligned on Safari 17.",
            missing_info=["What version of Safari?"],
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** needs_info" in comment
        assert "- What version of Safari?" in comment

    @pytest.mark.asyncio
    async def test_invalid_status_with_missing_summary_and_empty_missing_info_falls_back_to_needs_info(self):
        """W-1 regression guard: status fallback uses RAW summary, not the post-normalization default.

        With status="frobnicate", summary="", and missing_info=[], the
        raw summary has no content, so the fallback selects
        needs_info. A naive implementation that checked the
        post-normalization summary would always see "No summary
        provided." (non-empty) and wrongly pick "ready".
        """
        meta = _make_metadata()
        result = _triage_result(
            status="frobnicate",
            summary="",
            missing_info=[],
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** needs_info" in comment

    # ── Malformed field tests ───────────────────────────────────────

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,expected_default",
        [
            ("ready", "Review the issue and assign or start work."),
            ("needs_info", "Ask the reporter the questions in Missing information."),
            (
                "wontfix_candidate",
                "Confirm the scope assessment and either close or apply a wontfix label.",
            ),
            ("blocked", "Track the blocking dependency and revisit this issue when it resolves."),
        ],
    )
    async def test_missing_next_action_falls_back_per_status(self, status, expected_default):
        meta = _make_metadata()
        # blocked needs blocked_by populated, otherwise the consistency
        # check downgrades status to "ready" and the fallback default
        # changes underneath the test. Provide one so the status sticks.
        result = _triage_result(
            status=status,
            next_action=None,
            missing_info=["question?"] if status == "needs_info" else [],
            blocked_by=42 if status == "blocked" else None,
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert f"**Next action:** {expected_default}" in comment

    @pytest.mark.asyncio
    async def test_missing_info_with_non_string_entries_filters(self):
        meta = _make_metadata()
        result = _triage_result(
            status="needs_info",
            next_action="Ask the reporter.",
            missing_info=["valid?", 42, None, "another?"],
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert "- valid?" in comment
        assert "- another?" in comment
        assert "- 42" not in comment
        assert "- None" not in comment

    @pytest.mark.asyncio
    async def test_missing_info_scalar_normalizes_to_empty_list(self, caplog):
        """W-2 regression guard: non-list missing_info collapses to []."""
        meta = _make_metadata()
        result = _triage_result(
            status="needs_info",
            next_action="Ask the reporter.",
            missing_info="steps?",
        )
        with caplog.at_level("WARNING", logger="kai.triage"):
            comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Missing information:**" not in comment
        assert "Needs info:" not in telegram
        assert any("missing_info is str" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_scalar_missing_info_with_invalid_status_falls_back_to_ready(self):
        """W-1 v3 regression guard: scalar missing_info normalizes before status fallback.

        With status="frobnicate", a non-empty raw summary, and scalar
        missing_info, the parser normalizes missing_info to [] before
        the status fallback inspects it. Combined with the non-empty
        raw summary, this lands as status="ready" (not "needs_info",
        which would treat the malformed scalar as a real question
        signal).
        """
        meta = _make_metadata()
        result = _triage_result(
            status="frobnicate",
            summary="The login button is misaligned on Safari 17.",
            missing_info="steps?",
        )
        comment, _telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment

    @pytest.mark.asyncio
    async def test_missing_info_when_status_is_not_needs_info_still_renders(self, caplog):
        meta = _make_metadata()
        result = _triage_result(
            status="ready",
            next_action="Start work.",
            missing_info=["edge case to confirm?"],
        )
        with caplog.at_level("WARNING", logger="kai.triage"):
            comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Status:** ready" in comment
        assert "- edge case to confirm?" in comment
        assert "Needs info: 1 question" in telegram
        assert any("status=ready but missing_info has 1 question" in r.message for r in caplog.records)

    # ── Prompt contract test (B-1 regression guard) ─────────────────

    def test_prompt_allows_close_recommendation_in_next_action(self):
        """B-1 regression guard: prompt permits maintainer-driven close recommendations.

        The v1 spec had a prompt prohibition ("Do NOT recommend
        closing, assigning, or setting milestones") that contradicted
        the duplicate and wontfix_candidate fallbacks (which DO
        recommend closing, since the maintainer is the one closing).
        The v2 fix distinguishes maintainer-driven recommendations
        (allowed) from claims about Kai's automatic behavior (still
        forbidden). The prompt must contain phrasing that supports the
        distinction; assert on the load-bearing fragments here.
        """
        meta = _make_metadata()
        prompt = build_triage_prompt(meta, related_issues="[]", projects="[]")
        assert "next_action describes what the MAINTAINER should do" in prompt
        assert "Maintainer-driven close, assign, label, and milestone recommendations are fine" in prompt
        assert "Do NOT claim Kai will perform any of these automatically" in prompt
        # The wontfix_candidate framing as suggestion-not-verdict is
        # the load-bearing phrase for false-positive risk; assert it
        # exists too so a future prompt edit cannot silently drop it.
        assert "SUGGESTION for human review, NOT a verdict" in prompt

    # ── Bool-as-int guards on duplicate_of / related ────────────────

    @pytest.mark.asyncio
    async def test_bool_duplicate_of_normalizes_to_none(self):
        """Python bool is a subclass of int; the parser must reject it on duplicate_of.

        Without the explicit bool exclusion, `"duplicate_of": true`
        from a malformed model response would be accepted as a valid
        integer issue number (since `isinstance(True, int)` is True)
        and render as "Possible duplicate of: #True" in the
        public-facing comment. Same defect class the `blocked_by`
        guard already protects against.
        """
        meta = _make_metadata()
        result = _triage_result(
            duplicate_of=True,
            summary="Login broken on Safari.",
            status="ready",
            next_action="Start work.",
            missing_info=[],
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Possible duplicate of:**" not in comment
        assert "Possible duplicate of" not in telegram
        assert "#True" not in comment
        assert "#True" not in telegram

    @pytest.mark.asyncio
    async def test_bool_entries_in_related_filter_out(self):
        """Bool entries inside `related` must be filtered before rendering.

        Same defect class as `duplicate_of`: True / False inside the
        list would otherwise pass the per-entry `isinstance(n, int)`
        filter and render as "#True" / "#False" in the public-facing
        related-issues line.
        """
        meta = _make_metadata()
        result = _triage_result(
            related=[42, True, False, 100],
            summary="Login broken on Safari.",
            status="ready",
            next_action="Start work.",
            missing_info=[],
        )
        comment, telegram = await _apply_triage_capture(meta, result)
        assert "**Related issues:** #42, #100" in comment
        assert "Related: #42, #100" in telegram
        assert "#True" not in comment
        assert "#False" not in comment
        assert "#True" not in telegram
        assert "#False" not in telegram

    # ── Existing-behavior regression ────────────────────────────────

    @pytest.mark.asyncio
    async def test_existing_fields_still_render(self):
        """Adding the new fields does not displace the existing comment lines."""
        meta = _make_metadata(labels=[])
        result = _triage_result(
            labels=["bug", "enhancement"],
            duplicate_of=99,
            related=[100, 101],
            project="Sprint 1",
            summary="Login broken on Safari.",
            priority="high",
            status="ready",
            next_action="Start work on the login fix.",
            missing_info=[],
            blocked_by=None,
        )
        projects_json = json.dumps({"projects": [{"title": "Sprint 1", "number": 1}]})
        comment, telegram = await _apply_triage_capture(
            meta,
            result,
            projects_json=projects_json,
            allowed_triage_projects=["Sprint 1"],
        )
        # All existing fields render.
        assert "**Triage summary:** Login broken on Safari." in comment
        assert "**Priority:** high" in comment
        assert "**Labels applied:** bug, enhancement" in comment
        assert "**Possible duplicate of:** #99" in comment
        assert "**Related issues:** #100, #101" in comment
        assert "**Added to project:** Sprint 1" in comment
        # New fields render too.
        assert "**Status:** ready" in comment
        assert "**Next action:** Start work on the login fix." in comment
        # Telegram surfaces the new and existing summary bits.
        assert "Status: ready" in telegram
        assert "Priority: high" in telegram
        assert "Next: Start work on the login fix." in telegram


# ── triage_issue (full pipeline) ────────────────────────────────────


class TestTriageIssue:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """End-to-end triage executes all steps."""
        payload = _issue_payload(title="Login broken")

        triage_json = json.dumps(
            {
                "labels": ["bug"],
                "duplicate_of": None,
                "related": [5],
                "project": None,
                "summary": "Login is broken.",
                "priority": "high",
            }
        )

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Claude subprocess returns triage JSON
            if "claude" in args:
                return _mock_subprocess(stdout=triage_json)
            # gh issue list --search returns related issues
            if "issue" in args and "list" in args and "--search" in args:
                return _mock_subprocess(stdout=json.dumps([{"number": 5, "title": "Similar"}]))
            # gh project list returns empty
            if "project" in args and "list" in args:
                return _mock_subprocess(stdout="[]")
            # All other gh calls succeed
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await triage_issue(payload, 8080, "secret")

        assert mock_session.post.called
        # Pipeline ran (multiple subprocess calls)
        assert call_count > 0
        # Telegram notification was sent
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_project_prompt_and_apply_inputs_are_allowlist_filtered(self):
        """Only allowlisted projects are shown to the model and passed to apply."""
        payload = _issue_payload(title="Login broken")
        projects_json = json.dumps(
            {
                "projects": [
                    {"title": "Sprint 1", "number": 1},
                    {"title": "Secret Board", "number": 2},
                ]
            }
        )
        triage_json = json.dumps(
            {
                "labels": [],
                "duplicate_of": None,
                "related": [],
                "project": "Sprint 1",
                "summary": "Login is broken.",
                "priority": "high",
            }
        )

        captured_prompt: dict[str, str] = {}

        async def mock_run_triage(prompt: str, **kwargs):
            captured_prompt["prompt"] = prompt
            return triage_json

        with (
            patch("kai.triage.search_related_issues", new=AsyncMock(return_value="[]")),
            patch("kai.triage.list_projects", new=AsyncMock(return_value=projects_json)),
            patch("kai.triage.run_triage", side_effect=mock_run_triage),
            patch("kai.triage.apply_triage", new_callable=AsyncMock) as mock_apply,
        ):
            await triage_issue(payload, 8080, "secret", allowed_triage_projects=["Sprint 1"])

        assert "Sprint 1" in captured_prompt["prompt"]
        assert "Secret Board" not in captured_prompt["prompt"]

        mock_apply.assert_awaited_once()
        apply_kwargs = mock_apply.call_args.kwargs
        assert apply_kwargs["allowed_triage_projects"] == ["Sprint 1"]
        assert "Sprint 1" in apply_kwargs["projects_json"]
        assert "Secret Board" not in apply_kwargs["projects_json"]

    @pytest.mark.asyncio
    async def test_handles_error(self):
        """Claude subprocess failure logs error and sends Telegram notification."""
        payload = _issue_payload()

        async def mock_exec(*args, **kwargs):
            if "claude" in args:
                return _mock_subprocess(returncode=1, stderr="model error")
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            # Should not raise
            await triage_issue(payload, 8080, "secret")

        # Error notification was sent
        mock_session.post.assert_called_once()
        body = mock_session.post.call_args[1]["json"]
        assert "failed" in body["text"].lower()

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self):
        """Non-JSON response from Claude triggers error notification."""
        payload = _issue_payload()

        async def mock_exec(*args, **kwargs):
            if "claude" in args:
                return _mock_subprocess(stdout="Not JSON at all")
            return _mock_subprocess(stdout="[]")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await triage_issue(payload, 8080, "secret")

        # Error notification was sent
        mock_session.post.assert_called_once()
        body = mock_session.post.call_args[1]["json"]
        assert "failed" in body["text"].lower()


# ── _send_error_notification ──────────────────────────────────────


class TestSendErrorNotification:
    """Verify _send_error_notification never raises."""

    @pytest.mark.asyncio
    async def test_does_not_raise_on_connection_error(self):
        """Connection failure is caught and logged, not raised."""
        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        # session.post() is a sync call in production; the side effect
        # must fire when the call itself runs, not when an awaited
        # coroutine resumes. AsyncMock would queue the exception on
        # the unawaited coroutine and never raise, letting the test
        # pass for the wrong reason (and leaking a never-awaited
        # coroutine warning). MagicMock raises synchronously.
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession", return_value=mock_session):
            # Should not raise
            await _send_error_notification(metadata, "test error", 8080, "secret")

        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_timeout(self):
        """Timeout is caught and logged, not raised."""

        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession", return_value=mock_session):
            await _send_error_notification(metadata, "test error", 8080, "secret")

        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_warning_on_failure(self, caplog):
        """A warning is logged when the notification fails."""
        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=RuntimeError("boom"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kai.triage.aiohttp.ClientSession", return_value=mock_session),
            caplog.at_level("WARNING", logger="kai.triage"),
        ):
            await _send_error_notification(metadata, "test error", 8080, "secret")

        mock_session.post.assert_called_once()
        assert "Failed to send triage error notification" in caplog.text

    @pytest.mark.asyncio
    async def test_notify_chat_id_included_in_body(self):
        """When notify_chat_id is set, chat_id is included in the POST body."""
        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        with patch("kai.triage.aiohttp.ClientSession") as mock_cs:
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_cs)
            await _send_error_notification(metadata, "test error", 8080, "secret", notify_chat_id=-100999)

        assert mock_session.post.called
        body = mock_session.post.call_args[1]["json"]
        assert body["chat_id"] == -100999

    @pytest.mark.asyncio
    async def test_no_chat_id_when_notify_none(self):
        """When notify_chat_id is None, chat_id is NOT in the POST body."""
        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        with patch("kai.triage.aiohttp.ClientSession") as mock_cs:
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_cs)
            await _send_error_notification(metadata, "test error", 8080, "secret", notify_chat_id=None)

        assert mock_session.post.called
        body = mock_session.post.call_args[1]["json"]
        assert "chat_id" not in body


# ── apply_triage notify_chat_id ──────────────────────────────────────


class TestApplyTriageNotifyChatId:
    """Verify apply_triage threads notify_chat_id into the POST body."""

    @pytest.mark.asyncio
    async def test_chat_id_in_triage_summary(self):
        """apply_triage includes chat_id in the Telegram summary POST when set."""
        meta = _make_metadata(labels=[])
        result = _triage_result(labels=["bug"])

        async def mock_exec(*args, **kwargs):
            if "label" in args and "list" in args and "--search" in args:
                return _mock_subprocess(stdout="[]")
            return _mock_subprocess(stdout="")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", side_effect=mock_exec),
            patch("kai.triage.aiohttp.ClientSession") as mock_session_cls,
        ):
            mock_session, _ = _mock_aiohttp_session_post(status=200)
            _attach_session_to_class(mock_session, mock_session_cls)
            await apply_triage(meta, result, 8080, "secret", notify_chat_id=-100999)

        # The Telegram summary POST should include chat_id
        post_calls = mock_session.post.call_args_list
        # Find the send-message call (URL contains /api/send-message)
        summary_call = [c for c in post_calls if "/api/send-message" in str(c)]
        assert len(summary_call) >= 1
        body = summary_call[0][1]["json"]
        assert body["chat_id"] == -100999


# ── Triage error notification content ────────────────────────────────


class TestTriageErrorContent:
    @pytest.mark.asyncio
    async def test_error_sends_exception_type_not_message(self):
        """Triage failure notification contains only the exception type name,
        not the full message (which may leak internal paths)."""
        metadata = IssueMetadata(
            repo="owner/repo",
            number=42,
            title="Test issue",
            body="body",
            author="user",
            url="https://github.com/owner/repo/issues/42",
            labels=[],
        )

        # Substituted post-#353 because <install>/home/.claude/CLAUDE.md
        # no longer exists on production installs (the shared home was
        # removed; CLAUDE.md is per-user under
        # <DATA_DIR>/home/<chat_id>/.claude/, seeded by _apply_migrate
        # eagerly and ensure_user_home lazily, and after #447 the
        # install tree no longer holds any CLAUDE.md at all).
        # /etc/kai/env is the production secrets file - if its path ever
        # leaked into a user-facing error notification it would expose
        # the location of TELEGRAM_BOT_TOKEN and other secrets, which
        # matches the original test's "represent a sensitive path" intent.
        sensitive_path = "/etc/kai/env"
        error = FileNotFoundError(f"[Errno 2] No such file or directory: '{sensitive_path}'")

        with (
            patch("kai.triage.extract_issue_metadata", return_value=metadata),
            patch("kai.triage.search_related_issues", side_effect=error),
            patch("kai.triage._send_error_notification", new_callable=AsyncMock) as mock_notify,
        ):
            await triage_issue(
                {"issue": {}, "repository": {}},
                webhook_port=8080,
                webhook_secret="secret",
            )

        # The error_detail argument should be just the type name
        mock_notify.assert_called_once()
        error_detail = mock_notify.call_args[0][1]
        assert error_detail == "FileNotFoundError"
        assert sensitive_path not in error_detail
