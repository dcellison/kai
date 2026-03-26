"""Tests for triage.py issue triage pipeline."""

import json
import re
import signal
from unittest.mock import AsyncMock, patch

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


class TestRunTriage:
    @pytest.mark.asyncio
    async def test_success(self):
        """Successful Claude run returns stripped output."""
        expected = '{"labels": ["bug"], "summary": "A bug."}'
        mock_proc = _mock_subprocess(stdout=f"  {expected}  \n")

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_triage("test prompt")
        assert result == expected

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Timed-out subprocess raises RuntimeError and kills the process."""
        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await run_triage("test prompt")
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_with_claude_user_kills_group(self):
        """Timeout with claude_user kills the process group."""
        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        mock_proc.wait = AsyncMock()

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("kai.triage.os.killpg") as mock_killpg,
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await run_triage("test prompt", claude_user="kai")

        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
        mock_proc.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_claude_user_starts_new_session(self):
        """claude_user spawns with start_new_session=True."""
        mock_proc = _mock_subprocess(returncode=0, stdout='{"labels": []}')

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await run_triage("prompt", claude_user="kai")

        kwargs = mock_exec.call_args[1]
        assert kwargs.get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_no_claude_user_no_new_session(self):
        """Without claude_user, start_new_session is False."""
        mock_proc = _mock_subprocess(returncode=0, stdout='{"labels": []}')

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await run_triage("prompt")

        kwargs = mock_exec.call_args[1]
        assert kwargs.get("start_new_session") is False

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        """Non-zero exit code raises RuntimeError."""
        mock_proc = _mock_subprocess(returncode=1, stderr="model error")

        with (
            patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(RuntimeError, match="exit 1"),
        ):
            await run_triage("test prompt")

    @pytest.mark.asyncio
    async def test_claude_user_sudo(self):
        """When claude_user is set, command starts with sudo -u."""
        mock_proc = _mock_subprocess(stdout='{"labels": []}')

        with patch("kai.triage.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await run_triage("prompt", claude_user="testuser")

        # First args should be sudo -u testuser --
        args = mock_exec.call_args[0]
        assert args[0] == "sudo"
        assert args[1] == "-u"
        assert args[2] == "testuser"
        assert args[3] == "--"


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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

        # Verify gh issue edit --add-label was called for both labels
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        applied_labels = {cmd[list(cmd).index("--add-label") + 1] for cmd in add_label_calls}
        assert "bug" in applied_labels
        assert "enhancement" in applied_labels

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret", projects_json=projects_json)

        # Should have called gh project item-add
        item_add_calls = [cmd for cmd in commands_run if "project" in cmd and "item-add" in cmd]
        assert len(item_add_calls) > 0

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

        # Verify the send-message call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "send-message" in call_args[0][0]
        body = call_args[1]["json"]
        assert "Widget bug" in body["text"]
        assert "high" in body["text"]

    @pytest.mark.asyncio
    async def test_creates_missing_labels(self):
        """Labels that don't exist in the repo are created before applying."""
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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

        # Should have called gh label create
        create_calls = [cmd for cmd in commands_run if "label" in cmd and "create" in cmd]
        assert len(create_calls) > 0

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await apply_triage(meta, result, 8080, "secret")

        # No --add-label calls should have been made
        add_label_calls = [cmd for cmd in commands_run if "issue" in cmd and "edit" in cmd and "--add-label" in cmd]
        assert len(add_label_calls) == 0


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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            await triage_issue(payload, 8080, "secret")

        # Pipeline ran (multiple subprocess calls)
        assert call_count > 0
        # Telegram notification was sent
        mock_session.post.assert_called_once()

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
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

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(side_effect=ConnectionError("refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession", return_value=mock_session):
            # Should not raise
            await _send_error_notification(metadata, "test error", 8080, "secret")

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

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(side_effect=TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession", return_value=mock_session):
            await _send_error_notification(metadata, "test error", 8080, "secret")

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

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(side_effect=RuntimeError("boom"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("kai.triage.aiohttp.ClientSession", return_value=mock_session),
            caplog.at_level("WARNING", logger="kai.triage"),
        ):
            await _send_error_notification(metadata, "test error", 8080, "secret")

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

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_error_notification(metadata, "test error", 8080, "secret", notify_chat_id=-100999)

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

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.triage.aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)
            await _send_error_notification(metadata, "test error", 8080, "secret", notify_chat_id=None)

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
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
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

        sensitive_path = "/opt/kai/home/.claude/CLAUDE.md"
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
