"""Tests for `kai.smoke.memory` - the operator-facing verification
command for the memory reasoner pipeline.

Two contracts under test:
- Success path: reasoner returns at least one fact whose content
  references the anchor substring; the command prints
  resolved_binary from raw_metadata, fact rows, and exits 0.
- Routing-precondition path: when the configured reasoner is codex
  and `--os-user` is not supplied, exit 1 with a clear message
  before any subprocess fires.

The tests mock at the OneShotReasoner boundary so neither agent
binary is actually invoked.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from kai.config import Config
from kai.oneshot import OneShotResult

_BASE_CONFIG = Config(
    telegram_bot_token="test",
    allowed_user_ids={1},
)


def _config(**overrides) -> Config:
    """Minimal config for smoke testing. Memory must be enabled +
    extraction-enabled for the smoke to even reach the reasoner;
    other knobs match the dataclass defaults."""
    defaults = {
        "memory_enabled": True,
        "memory_extraction_enabled": True,
        "default_backend": "claude",
        "memory_extraction_timeout_s": 30,
    }
    defaults.update(overrides)
    return replace(_BASE_CONFIG, **defaults)


def _success_envelope() -> str:
    """Envelope shape the smoke parses. Matches what
    `--output-format=json` produces on the claude branch and what
    CodexOneShotReasoner rewraps on the codex branch."""
    import json as json_mod

    return json_mod.dumps(
        {
            "structured_output": {
                "facts": [
                    {
                        "content": "User takes coffee with oat milk, no sugar.",
                        "tags": ["preference", "food"],
                        "speaker": "user",
                        "intent": "new",
                        "confidence": 0.95,
                    }
                ],
                "has_episode": False,
            }
        }
    )


class TestSmokeMemorySuccess:
    @pytest.mark.asyncio
    async def test_anchor_hit_exits_zero(self, monkeypatch, capsys):
        """A reasoner result with at least one fact referencing the
        anchor substring ('oat milk') yields exit 0. The smoke prints
        the resolved binary and argv from raw_metadata, plus the
        extracted facts list."""
        from kai.smoke import memory as smoke_module

        monkeypatch.setattr(smoke_module, "load_config", lambda: _config())
        fake_run = AsyncMock(
            return_value=OneShotResult(
                text=_success_envelope(),
                backend="claude",
                model="claude-haiku-4-5",
                raw_metadata={
                    "cmd": ["claude", "--print"],
                    "resolved_binary": "/fake/path/claude",
                    "returncode": 0,
                    "stderr": b"",
                    "cwd": "/tmp",
                },
                duration_ms=42,
            )
        )
        with patch.object(smoke_module, "_build_memory_reasoner", return_value=type("R", (), {"run": fake_run})()):
            rc = await smoke_module._run(user_id=None, os_user=None)
        out = capsys.readouterr().out
        assert rc == 0
        assert "resolved_binary: /fake/path/claude" in out
        assert "verdict: pass" in out
        assert "oat milk" in out

    @pytest.mark.asyncio
    async def test_no_anchor_hit_exits_nonzero(self, monkeypatch, capsys):
        """Reasoner succeeded structurally but extracted facts do not
        reference the anchor substring: smoke treats that as a
        content-failure and exits non-zero. Distinguishes 'extractor
        broken' from 'extractor wrong'."""
        import json as json_mod

        from kai.smoke import memory as smoke_module

        envelope = json_mod.dumps(
            {
                "structured_output": {
                    "facts": [{"content": "User likes tea", "tags": [], "speaker": "user", "intent": "new"}],
                    "has_episode": False,
                }
            }
        )
        monkeypatch.setattr(smoke_module, "load_config", lambda: _config())
        fake_run = AsyncMock(
            return_value=OneShotResult(
                text=envelope,
                backend="claude",
                model="claude-haiku-4-5",
                raw_metadata={
                    "cmd": ["claude"],
                    "resolved_binary": "/fake/claude",
                    "returncode": 0,
                    "stderr": b"",
                    "cwd": "/tmp",
                },
                duration_ms=10,
            )
        )
        with patch.object(smoke_module, "_build_memory_reasoner", return_value=type("R", (), {"run": fake_run})()):
            rc = await smoke_module._run(user_id=None, os_user=None)
        out = capsys.readouterr().out
        assert rc == 1
        assert "verdict: pass" not in out


class TestSmokeMemoryProductionMode:
    """The smoke claims to verify the memory extraction pipeline
    end-to-end. Running the reasoner against a fixed payload when
    production has memory disabled or extraction disabled produces a
    false-positive 'pass' for a path that never fires on real
    traffic. The smoke refuses to run in those configurations."""

    @pytest.mark.asyncio
    async def test_memory_disabled_exits_nonzero(self, monkeypatch, capsys):
        """MEMORY_ENABLED=false: smoke must refuse, name the var, and
        not invoke the reasoner."""
        from kai.smoke import memory as smoke_module

        monkeypatch.setattr(smoke_module, "load_config", lambda: _config(memory_enabled=False))

        def fail_factory(*args, **kwargs):
            pytest.fail("reasoner must not run when memory is disabled")

        monkeypatch.setattr(smoke_module, "_build_memory_reasoner", fail_factory)
        rc = await smoke_module._run(user_id=None, os_user=None)
        captured = capsys.readouterr()
        assert rc == 1
        assert "MEMORY_ENABLED" in captured.err

    @pytest.mark.asyncio
    async def test_extraction_disabled_exits_nonzero(self, monkeypatch, capsys):
        """Retrieval-only mode (MEMORY_ENABLED=true with extraction
        disabled): smoke must refuse rather than print verdict:pass
        against a path that will never fire on real traffic."""
        from kai.smoke import memory as smoke_module

        monkeypatch.setattr(
            smoke_module,
            "load_config",
            lambda: _config(memory_extraction_enabled=False),
        )

        def fail_factory(*args, **kwargs):
            pytest.fail("reasoner must not run when extraction is disabled")

        monkeypatch.setattr(smoke_module, "_build_memory_reasoner", fail_factory)
        rc = await smoke_module._run(user_id=None, os_user=None)
        captured = capsys.readouterr()
        assert rc == 1
        assert "MEMORY_EXTRACTION_ENABLED" in captured.err


class TestSmokeMemoryRoutingPrecondition:
    @pytest.mark.asyncio
    async def test_codex_without_os_user_proceeds(self, monkeypatch, capsys):
        """Issue #522: codex same-user spawn is supported. Smoke
        without --os-user against DEFAULT_BACKEND=codex no longer
        exits 1 on the precondition; it proceeds to the reasoner
        call. Pins the symmetry with claude same-user."""
        from kai.smoke import memory as smoke_module

        monkeypatch.setattr(smoke_module, "load_config", lambda: _config(default_backend="codex"))
        fake_run = AsyncMock(
            return_value=OneShotResult(
                text=_success_envelope(),
                backend="codex",
                model="gpt-5.4-mini",
                raw_metadata={
                    "cmd": ["codex"],
                    "resolved_binary": "/fake/codex",
                    "returncode": 0,
                    "stderr": b"",
                    "cwd": "/tmp",
                },
                duration_ms=10,
            )
        )
        with patch.object(smoke_module, "_build_memory_reasoner", return_value=type("R", (), {"run": fake_run})()):
            rc = await smoke_module._run(user_id=None, os_user=None)
        assert rc == 0

    @pytest.mark.asyncio
    async def test_claude_without_os_user_proceeds(self, monkeypatch, capsys):
        """On the claude branch, --os-user is optional. The reasoner
        is invoked with os_user=None and the historical self-sudo-skip
        path handles it gracefully."""
        from kai.smoke import memory as smoke_module

        monkeypatch.setattr(smoke_module, "load_config", lambda: _config())
        fake_run = AsyncMock(
            return_value=OneShotResult(
                text=_success_envelope(),
                backend="claude",
                model="claude-haiku-4-5",
                raw_metadata={
                    "cmd": ["claude"],
                    "resolved_binary": "/fake/claude",
                    "returncode": 0,
                    "stderr": b"",
                    "cwd": "/tmp",
                },
                duration_ms=10,
            )
        )
        with patch.object(smoke_module, "_build_memory_reasoner", return_value=type("R", (), {"run": fake_run})()):
            rc = await smoke_module._run(user_id=None, os_user=None)
        assert rc == 0


class TestSmokeMemoryUserIdDispatch:
    """The `--user-id` flag drives the smoke's effective backend
    resolution (issue #515). With per-user dispatch, the smoke must
    reach the same reasoner production would for that user; without
    `--user-id`, it falls back to the global `default_backend`."""

    @pytest.mark.asyncio
    async def test_user_id_resolves_codex_user(self, monkeypatch, capsys):
        """A `--user-id` matching a codex-effective users.yaml entry
        resolves to codex, prints `backend: codex`, and resolves the
        codex registry model. Pins per-user dispatch end-to-end at
        the smoke surface."""
        from kai.config import UserConfig
        from kai.smoke import memory as smoke_module

        config = _config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="codex_user",
                    os_user="codex_os",
                    backend="codex",
                )
            }
        )
        monkeypatch.setattr(smoke_module, "load_config", lambda: config)
        captured_backend: list[str] = []

        def _build(effective_backend, os_user=None):
            captured_backend.append(effective_backend)
            run_fn = AsyncMock(
                return_value=OneShotResult(
                    text=_success_envelope(),
                    backend=effective_backend,
                    model="gpt-5.4-mini",
                    raw_metadata={
                        "cmd": ["codex"],
                        "resolved_binary": "/fake/codex",
                        "returncode": 0,
                        "stderr": b"",
                        "cwd": "/tmp",
                    },
                    duration_ms=10,
                )
            )
            return type("R", (), {"run": run_fn})()

        monkeypatch.setattr(smoke_module, "_build_memory_reasoner", _build)
        rc = await smoke_module._run(user_id="1", os_user="codex_os")
        out = capsys.readouterr().out
        assert rc == 0
        assert captured_backend == ["codex"]
        assert "backend: codex" in out

    @pytest.mark.asyncio
    async def test_user_id_resolves_claude_user(self, monkeypatch, capsys):
        """A `--user-id` matching a claude-effective users.yaml entry
        resolves to claude even when global DEFAULT_BACKEND=codex."""
        from kai.config import UserConfig
        from kai.smoke import memory as smoke_module

        config = _config(
            default_backend="codex",
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="claude_user",
                    os_user="claude_os",
                    backend="claude",
                )
            },
        )
        monkeypatch.setattr(smoke_module, "load_config", lambda: config)
        captured_backend: list[str] = []

        def _build(effective_backend, os_user=None):
            captured_backend.append(effective_backend)
            run_fn = AsyncMock(
                return_value=OneShotResult(
                    text=_success_envelope(),
                    backend=effective_backend,
                    model="claude-haiku-4-5",
                    raw_metadata={
                        "cmd": ["claude"],
                        "resolved_binary": "/fake/claude",
                        "returncode": 0,
                        "stderr": b"",
                        "cwd": "/tmp",
                    },
                    duration_ms=10,
                )
            )
            return type("R", (), {"run": run_fn})()

        monkeypatch.setattr(smoke_module, "_build_memory_reasoner", _build)
        rc = await smoke_module._run(user_id="1", os_user=None)
        out = capsys.readouterr().out
        assert rc == 0
        assert captured_backend == ["claude"]
        assert "backend: claude" in out

    @pytest.mark.asyncio
    async def test_no_user_id_uses_global_agent_backend(self, monkeypatch, capsys):
        """Smoke without `--user-id` falls back to the global
        `default_backend`. Pins the legacy single-backend smoke path
        so the per-user dispatch change does not silently break
        operator habits ('run smoke without flags' still works)."""
        from kai.smoke import memory as smoke_module

        config = _config(default_backend="claude")
        monkeypatch.setattr(smoke_module, "load_config", lambda: config)
        captured_backend: list[str] = []

        def _build(effective_backend, os_user=None):
            captured_backend.append(effective_backend)
            run_fn = AsyncMock(
                return_value=OneShotResult(
                    text=_success_envelope(),
                    backend=effective_backend,
                    model="claude-haiku-4-5",
                    raw_metadata={
                        "cmd": ["claude"],
                        "resolved_binary": "/fake/claude",
                        "returncode": 0,
                        "stderr": b"",
                        "cwd": "/tmp",
                    },
                    duration_ms=10,
                )
            )
            return type("R", (), {"run": run_fn})()

        monkeypatch.setattr(smoke_module, "_build_memory_reasoner", _build)
        rc = await smoke_module._run(user_id=None, os_user=None)
        assert rc == 0
        assert captured_backend == ["claude"]
