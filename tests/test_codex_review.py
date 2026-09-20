"""Bounded Codex app-server review transport tests."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.codex_review import CodexAppServerReviewReasoner
from kai.oneshot import OneShotSubprocessError


def _message(value: dict) -> bytes:
    return json.dumps(value).encode("utf-8") + b"\n"


def _review_exchange(*, failed: bool = False) -> list[bytes]:
    messages = [
        _message({"id": 1, "result": {"userAgent": "codex"}}),
        _message({"id": 2, "result": {"thread": {"id": "thr_review"}}}),
        _message({"id": 3, "result": {"turn": {"id": "turn_review", "status": "inProgress"}}}),
    ]
    if failed:
        messages.append(
            _message(
                {
                    "method": "turn/completed",
                    "params": {
                        "turn": {
                            "id": "turn_review",
                            "status": "failed",
                            "error": {"message": "failed to connect to websocket: IO error"},
                        }
                    },
                }
            )
        )
    else:
        messages.extend(
            [
                _message(
                    {
                        "method": "item/completed",
                        "params": {
                            "item": {
                                "id": "msg_1",
                                "type": "agentMessage",
                                "phase": "final_answer",
                                "text": "First review section",
                            }
                        },
                    }
                ),
                _message(
                    {
                        "method": "item/completed",
                        "params": {
                            "item": {
                                "id": "msg_2",
                                "type": "agentMessage",
                                "phase": "final_answer",
                                "text": "Second review section",
                            }
                        },
                    }
                ),
                _message(
                    {
                        "method": "turn/completed",
                        "params": {"turn": {"id": "turn_review", "status": "completed", "error": None}},
                    }
                ),
            ]
        )
    messages.append(_message({"id": 4, "result": {}}))
    return messages


def _make_app_server_proc(messages: list[bytes]) -> MagicMock:
    stdout = asyncio.StreamReader()
    for message in messages:
        stdout.feed_data(message)
    stdout.feed_eof()
    stderr = asyncio.StreamReader()
    stderr.feed_eof()
    stdin = MagicMock()
    stdin.write = MagicMock()
    stdin.drain = AsyncMock()
    stdin.close = MagicMock()
    proc = MagicMock()
    proc.stdin = stdin
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = None
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    proc.pid = 4242
    return proc


def _written_messages(proc: MagicMock) -> list[dict]:
    return [json.loads(call.args[0]) for call in proc.stdin.write.call_args_list]


class TestCodexAppServerReviewReasoner:
    @pytest.mark.asyncio
    async def test_runs_one_read_only_turn_and_deletes_thread(self, tmp_path: Path):
        proc = _make_app_server_proc(_review_exchange())
        reasoner = CodexAppServerReviewReasoner(cwd=tmp_path, join_items=True)

        with (
            patch("kai.codex_review.resolve_oneshot_binary", return_value="codex"),
            patch("kai.codex_review.resolve_claude_user", return_value=None),
            patch("kai.codex_review.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
        ):
            result = await reasoner.run(
                prompt="Review this pull request",
                model="gpt-5.5",
                timeout=30,
                purpose="pr_review",
            )

        assert result.text == "First review section\n\nSecond review section"
        assert result.raw_metadata["transport"] == "app_server"
        argv = spawn.call_args.args
        assert argv[0] == "codex"
        assert argv[-1] == "app-server"
        assert "exec" not in argv
        assert "responses_websockets" not in argv
        assert "responses_websockets_v2" not in argv
        app_server_index = argv.index("app-server")
        assert all(index < app_server_index for index, arg in enumerate(argv) if arg in {"--config", "--disable"})

        messages = _written_messages(proc)
        assert [message["method"] for message in messages] == [
            "initialize",
            "initialized",
            "thread/start",
            "turn/start",
            "thread/delete",
        ]
        thread_params = messages[2]["params"]
        assert thread_params["approvalPolicy"] == "never"
        assert thread_params["sandbox"] == "read-only"
        assert thread_params["config"]["project_doc_max_bytes"] == 0
        turn_params = messages[3]["params"]
        assert turn_params["approvalPolicy"] == "never"
        assert turn_params["sandboxPolicy"] == {
            "type": "readOnly",
            "access": {
                "type": "restricted",
                "includePlatformDefaults": True,
                "readableRoots": [str(tmp_path)],
            },
        }
        assert messages[4]["params"] == {"threadId": "thr_review"}
        proc.stdin.close.assert_called_once()
        proc.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_user_spawn_preserves_auth_and_uses_an_isolated_process_group(self, tmp_path: Path):
        proc = _make_app_server_proc(_review_exchange())
        reasoner = CodexAppServerReviewReasoner(cwd=tmp_path, os_user="daniel")

        def wrap(cmd, target_user, backend, *, working_directory, preserve_vars):
            assert target_user == "daniel"
            assert backend == "codex"
            assert working_directory == tmp_path
            assert preserve_vars == ("CODEX_HOME", "OPENAI_API_KEY", "OPENAI_BASE_URL", "TMPDIR")
            return ["sudo", *cmd]

        with (
            patch("kai.codex_review.resolve_oneshot_binary", return_value="codex"),
            patch("kai.codex_review.resolve_claude_user", return_value="daniel"),
            patch("kai.codex_review._wrap_cmd_for_user", side_effect=wrap),
            patch("kai.codex_review.subprocess_spawn_cwd", return_value=tmp_path),
            patch("kai.codex_review.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
        ):
            await reasoner.run(
                prompt="Review this pull request",
                model="gpt-5.5",
                timeout=30,
                purpose="pr_review",
            )

        assert spawn.call_args.args[0] == "sudo"
        assert spawn.call_args.kwargs["start_new_session"] is True
        assert spawn.call_args.kwargs["env"]["TMPDIR"].endswith("/tmp/daniel")

    @pytest.mark.asyncio
    async def test_turn_failure_preserves_transport_detail_and_still_deletes_thread(self, tmp_path: Path):
        proc = _make_app_server_proc(_review_exchange(failed=True))
        reasoner = CodexAppServerReviewReasoner(cwd=tmp_path)

        with (
            patch("kai.codex_review.resolve_oneshot_binary", return_value="codex"),
            patch("kai.codex_review.resolve_claude_user", return_value=None),
            patch("kai.codex_review.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotSubprocessError, match="failed to connect to websocket"),
        ):
            await reasoner.run(
                prompt="Review this pull request",
                model="gpt-5.5",
                timeout=30,
                purpose="pr_review",
            )

        assert _written_messages(proc)[-1] == {
            "method": "thread/delete",
            "id": 4,
            "params": {"threadId": "thr_review"},
        }

    @pytest.mark.asyncio
    async def test_process_is_killed_when_stdin_close_does_not_stop_it(self):
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.close = MagicMock()
        proc.wait = AsyncMock(side_effect=TimeoutError())
        proc.kill = MagicMock()

        await CodexAppServerReviewReasoner._stop_process(
            proc,
            effective_user=None,
            purpose="pr_review",
        )

        proc.stdin.close.assert_called_once()
        proc.kill.assert_called_once()
