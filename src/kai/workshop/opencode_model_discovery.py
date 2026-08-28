"""Metadata-only OpenCode model discovery through its documented CLI.

OpenCode documents ``opencode models [provider] --refresh`` as the
non-interactive command for refreshing its models.dev cache and listing the
models available from one configured provider.  The command emits one
``provider/model`` identifier per line and does not create a session or invoke
a model.
"""

from __future__ import annotations

import asyncio
import os
import pwd
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kai.acp import _kill_target_user_tree
from kai.backend import sanitize_agent_environment
from kai.config import DATA_DIR, is_opencode_model_shape, resolve_claude_user
from kai.subprocess_identity import subprocess_spawn_cwd, wrap_command_for_target_user
from kai.workshop.model_catalogue import (
    ModelCatalogueValidationError,
    ModelDiscoveryAuthenticationError,
    ModelDiscoveryBatch,
    ModelDiscoveryCandidate,
    ModelDiscoveryUnsupported,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryBackendInventory,
    ModelDiscoveryReadiness,
)

_SOURCE = "opencode-cli-models"
_TTL_SECONDS = 21_600
_CLOSE_TIMEOUT_SECONDS = 3.0
_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
_MAX_MODELS = 20_000
_MAX_MODEL_ID_BYTES = 512
_READ_CHUNK_BYTES = 64 * 1024
_AUTH_MARKERS = (
    "api key",
    "auth",
    "credential",
    "log in",
    "login",
    "not configured",
    "provider not found",
    "sign in",
    "token",
    "unauthorized",
)
_UNSUPPORTED_MARKERS = (
    "unknown command",
    "unknown option",
    "unsupported command",
)
_PRESERVED_AUTH_VARS = (
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENCODE_CONFIG",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_DISABLE_MODELS_FETCH",
    "OPENCODE_MODELS_PATH",
    "OPENCODE_MODELS_URL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
)


class _OpenCodeDiscoveryError(RuntimeError):
    """A sanitized OpenCode metadata-command failure."""


@dataclass(frozen=True, slots=True)
class _OpenCodeLaunch:
    argv: tuple[str, ...]
    cwd: str | None
    environment: Mapping[str, str]
    target_user: str | None


@dataclass(frozen=True, slots=True)
class _CapturedOutput:
    data: bytes
    overflowed: bool


class OpenCodeModelDiscoveryAdapter:
    """Enumerate models visible to one canonical OpenCode provider context."""

    def __init__(
        self,
        *,
        service_os_user: str,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not service_os_user.strip():
            raise ValueError("service_os_user must be non-empty")
        self._service_os_user = service_os_user.strip()
        self._environment = os.environ if environment is None else environment

    async def discover(self, lane: ModelDiscoveryBackendInventory) -> ModelDiscoveryBatch:
        self._validate_lane(lane)
        launch = _build_launch(
            lane,
            service_os_user=self._service_os_user,
            environment=self._environment,
        )
        process = await asyncio.create_subprocess_exec(
            *launch.argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=launch.cwd,
            env=dict(launch.environment),
            start_new_session=launch.target_user is not None,
        )
        stdout_task = asyncio.create_task(_read_bounded(process.stdout))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr))
        try:
            returncode = await process.wait()
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
            if stdout.overflowed or stderr.overflowed:
                raise ModelCatalogueValidationError("OpenCode model command output exceeds the safety limit")
            if returncode != 0:
                _raise_command_failure(stdout.data, stderr.data)
            models = _parse_models_output(stdout.data, lane.provider)
            return ModelDiscoveryBatch(_SOURCE, models, ttl_seconds=_TTL_SECONDS)
        finally:
            await _finish_process_cleanup(
                process,
                stdout_task,
                stderr_task,
                target_user=launch.target_user,
            )

    @staticmethod
    def _validate_lane(lane: ModelDiscoveryBackendInventory) -> None:
        if lane.backend != "opencode":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if not lane.executable.configured_path or not lane.executable.resolved_path:
            raise ModelDiscoveryUnsupported


def _build_launch(
    lane: ModelDiscoveryBackendInventory,
    *,
    service_os_user: str,
    environment: Mapping[str, str],
) -> _OpenCodeLaunch:
    command = lane.executable.configured_path
    if command is None:
        raise ModelDiscoveryUnsupported
    try:
        home = Path(pwd.getpwnam(lane.effective_os_user).pw_dir)
    except KeyError as exc:
        raise ModelDiscoveryUnsupported from exc
    env = sanitize_agent_environment(dict(environment))
    # Discovery must load the effective user's real OpenCode configuration,
    # not a one-turn model override inherited from another process.
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    env["NO_COLOR"] = "1"
    argv = [command, "models", lane.provider, "--refresh"]
    target_user = None if lane.effective_os_user == service_os_user else resolve_claude_user(lane.effective_os_user)
    if target_user is not None:
        env["TMPDIR"] = str(DATA_DIR / "tmp" / target_user)
        argv = wrap_command_for_target_user(
            argv,
            target_user=target_user,
            working_directory=home,
            preserve_env=(*_PRESERVED_AUTH_VARS, "NO_COLOR", "TMPDIR"),
        )
    return _OpenCodeLaunch(
        tuple(argv),
        subprocess_spawn_cwd(home, target_user=target_user),
        env,
        target_user,
    )


async def _read_bounded(stream: asyncio.StreamReader | None) -> _CapturedOutput:
    if stream is None:
        return _CapturedOutput(b"", False)
    output = bytearray()
    overflowed = False
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        if overflowed:
            continue
        remaining = _MAX_OUTPUT_BYTES + 1 - len(output)
        output.extend(chunk[:remaining])
        overflowed = len(output) > _MAX_OUTPUT_BYTES
    return _CapturedOutput(bytes(output[:_MAX_OUTPUT_BYTES]), overflowed)


def _raise_command_failure(stdout: bytes, stderr: bytes) -> None:
    detail = (stderr + b"\n" + stdout).decode("utf-8", errors="ignore").lower()
    if any(marker in detail for marker in _AUTH_MARKERS):
        raise ModelDiscoveryAuthenticationError
    if any(marker in detail for marker in _UNSUPPORTED_MARKERS):
        raise ModelDiscoveryUnsupported
    raise _OpenCodeDiscoveryError("OpenCode model metadata command failed")


def _parse_models_output(body: bytes, provider_id: str) -> tuple[ModelDiscoveryCandidate, ...]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelCatalogueValidationError("OpenCode model command output is not UTF-8") from exc
    prefix = f"{provider_id}/"
    candidates: list[ModelDiscoveryCandidate] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        model_id = raw_line.strip()
        if not model_id:
            continue
        if len(model_id.encode("utf-8")) > _MAX_MODEL_ID_BYTES:
            raise ModelCatalogueValidationError("OpenCode returned an oversized model identifier")
        if not model_id.startswith(prefix) or not is_opencode_model_shape(model_id):
            raise ModelCatalogueValidationError("OpenCode returned an invalid provider/model identifier")
        if model_id in seen:
            raise ModelCatalogueValidationError("OpenCode returned a duplicate provider/model identifier")
        seen.add(model_id)
        candidates.append(
            ModelDiscoveryCandidate(
                model_id,
                model_id,
                {"provider_id": provider_id},
            )
        )
        if len(candidates) > _MAX_MODELS:
            raise ModelCatalogueValidationError("OpenCode model catalogue exceeds the safety limit")
    return tuple(candidates)


async def _close_process(
    process: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[_CapturedOutput],
    stderr_task: asyncio.Task[_CapturedOutput],
    *,
    target_user: str | None,
) -> None:
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
                await process.wait()
        except TimeoutError:
            if target_user is not None:
                await _kill_target_user_tree(
                    target_user=target_user,
                    pgid=process.pid,
                    purpose="model discovery timeout",
                    backend="opencode",
                )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    for task in (stdout_task, stderr_task):
        try:
            async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
                await task
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _finish_process_cleanup(
    process: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[_CapturedOutput],
    stderr_task: asyncio.Task[_CapturedOutput],
    *,
    target_user: str | None,
) -> None:
    cleanup = asyncio.create_task(
        _close_process(
            process,
            stdout_task,
            stderr_task,
            target_user=target_user,
        )
    )
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise
