"""Metadata-only Pi model discovery through Pi's documented JSONL RPC mode.

The adapter starts an ephemeral Pi process in the canonical runtime profile's
effective OS-user and authentication context, sends only
``get_available_models``, and closes stdin.  Ambient project resources and
tools are disabled, and no prompt or model-generation request is submitted.
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
from kai.config import DATA_DIR, is_pi_model_shape, resolve_claude_user
from kai.pi import _PI_ALL_PROVIDER_ENV_VARS, _pi_provider_env_vars, _split_pi_model
from kai.pi_rpc import (
    PI_RPC_STREAM_LIMIT,
    PiRpcCommandError,
    PiRpcEOFError,
    PiRpcProtocolError,
    PiRpcTransport,
    require_pi_rpc_response,
)
from kai.subprocess_identity import subprocess_spawn_cwd, wrap_command_for_target_user
from kai.workshop.model_catalogue import (
    ModelCatalogueValidationError,
    ModelDiscoveryAuthenticationError,
    ModelDiscoveryBatch,
    ModelDiscoveryCandidate,
    ModelDiscoveryUnsupported,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryAuthMode,
    ModelDiscoveryBackendInventory,
    ModelDiscoveryReadiness,
)

_SOURCE = "pi-rpc-get-available-models"
_TTL_SECONDS = 21_600
_CLOSE_TIMEOUT_SECONDS = 3.0
_MAX_MODELS = 20_000
_MAX_CANDIDATES = 160_000
_MAX_MODEL_ID_CHARACTERS = 512
_MAX_PROVIDER_CHARACTERS = 128
_REQUEST_ID = "kai-model-discovery"
_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
_AUTH_MARKERS = (
    "auth",
    "credential",
    "expired",
    "log in",
    "login",
    "not configured",
    "sign in",
    "token",
    "unauthorized",
)
_UNSUPPORTED_MARKERS = (
    "not supported",
    "unknown command",
    "unknown request",
    "unsupported command",
)
_PRESERVED_RUNTIME_VARS = (
    "PI_SKIP_VERSION_CHECK",
    "PI_TELEMETRY",
    "TMPDIR",
)


class _PiDiscoveryError(RuntimeError):
    """A sanitized Pi discovery failure."""


@dataclass(frozen=True, slots=True)
class _PiLaunch:
    argv: tuple[str, ...]
    cwd: str | None
    environment: Mapping[str, str]
    target_user: str | None


class PiModelDiscoveryAdapter:
    """Enumerate models visible to one canonical Pi provider context."""

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
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=launch.cwd,
            env=dict(launch.environment),
            start_new_session=launch.target_user is not None,
            limit=PI_RPC_STREAM_LIMIT,
        )
        stderr_task = asyncio.create_task(_discard_stream(process.stderr))
        try:
            if process.stdin is None or process.stdout is None:
                raise _PiDiscoveryError("Pi RPC pipes are unavailable")
            transport = PiRpcTransport(process.stdin, process.stdout)
            await transport.send({"id": _REQUEST_ID, "type": "get_available_models"})
            result = await _receive_result(transport)
            candidates = _parse_available_models(result, lane.provider)
            if not candidates:
                raise ModelDiscoveryAuthenticationError
            return ModelDiscoveryBatch(_SOURCE, candidates, ttl_seconds=_TTL_SECONDS)
        except PiRpcCommandError as exc:
            detail = exc.detail.lower()
            if any(marker in detail for marker in _AUTH_MARKERS):
                raise ModelDiscoveryAuthenticationError from exc
            if any(marker in detail for marker in _UNSUPPORTED_MARKERS):
                raise ModelDiscoveryUnsupported from exc
            raise _PiDiscoveryError("Pi rejected its model metadata request") from exc
        except (PiRpcEOFError, PiRpcProtocolError) as exc:
            raise _PiDiscoveryError("Pi model metadata protocol failed") from exc
        finally:
            await _finish_process_cleanup(
                process,
                stderr_task,
                target_user=launch.target_user,
            )

    @staticmethod
    def _validate_lane(lane: ModelDiscoveryBackendInventory) -> None:
        if lane.backend != "pi":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if not lane.executable.configured_path or not lane.executable.resolved_path:
            raise ModelDiscoveryUnsupported
        if lane.auth.mode == ModelDiscoveryAuthMode.API_KEY and lane.auth.configured is not True:
            raise ModelDiscoveryAuthenticationError


async def _receive_result(transport: PiRpcTransport) -> object:
    # Pi currently responds directly, but tolerate documented asynchronous
    # startup events without accepting a response for another request.
    for _ in range(128):
        message = await transport.receive()
        if message.get("type") != "response":
            continue
        return require_pi_rpc_response(
            message,
            request_id=_REQUEST_ID,
            command="get_available_models",
        )
    raise ModelCatalogueValidationError("Pi emitted too many records before model metadata")


def _parse_available_models(
    value: object,
    provider_id: str,
) -> tuple[ModelDiscoveryCandidate, ...]:
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("Pi model metadata must be an object")
    models = value.get("models")
    if not isinstance(models, list) or len(models) > _MAX_MODELS:
        raise ModelCatalogueValidationError("Pi model metadata must contain a bounded model list")

    candidates: list[ModelDiscoveryCandidate] = []
    seen_models: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ModelCatalogueValidationError("Pi model metadata entry must be an object")
        observed_provider = _required_text(
            model,
            "provider",
            maximum=_MAX_PROVIDER_CHARACTERS,
        )
        if observed_provider != provider_id:
            continue
        model_id = _required_text(
            model,
            "id",
            maximum=_MAX_MODEL_ID_CHARACTERS,
        )
        full_id = f"{observed_provider}/{model_id}"
        if len(full_id) > _MAX_MODEL_ID_CHARACTERS or not is_pi_model_shape(full_id, provider_id):
            raise ModelCatalogueValidationError("Pi returned an invalid provider/model identifier")
        if full_id in seen_models:
            raise ModelCatalogueValidationError("Pi returned a duplicate provider/model identifier")
        seen_models.add(full_id)

        reasoning = model.get("reasoning")
        if not isinstance(reasoning, bool):
            raise ModelCatalogueValidationError("Pi model reasoning capability must be boolean")
        label = _optional_text(model, "name", maximum=_MAX_MODEL_ID_CHARACTERS) or full_id
        thinking_levels = _supported_thinking_levels(model, reasoning=reasoning)
        base_capabilities: dict[str, object] = {
            "provider_id": observed_provider,
            "api": _optional_text(model, "api", maximum=128),
            "reasoning": reasoning,
            "thinking_levels": list(thinking_levels),
            "input": _input_modalities(model.get("input")),
        }
        candidates.append(ModelDiscoveryCandidate(full_id, label, base_capabilities))
        if reasoning:
            for level in thinking_levels:
                suffixed_id = f"{full_id}:{level}"
                if len(suffixed_id) > _MAX_MODEL_ID_CHARACTERS:
                    raise ModelCatalogueValidationError("Pi thinking selection exceeds the model identifier limit")
                capabilities = {**base_capabilities, "thinking_level": level}
                candidates.append(
                    ModelDiscoveryCandidate(
                        suffixed_id,
                        f"{label} - {level}",
                        capabilities,
                    )
                )
        if len(candidates) > _MAX_CANDIDATES:
            raise ModelCatalogueValidationError("Pi expanded model catalogue exceeds the safety limit")
    return tuple(candidates)


def _supported_thinking_levels(
    model: Mapping[str, object],
    *,
    reasoning: bool,
) -> tuple[str, ...]:
    if not reasoning:
        return ("off",)
    raw_map = model.get("thinkingLevelMap")
    if raw_map is not None and not isinstance(raw_map, dict):
        raise ModelCatalogueValidationError("Pi thinking-level map must be an object")
    level_map = raw_map or {}
    for key, mapped in level_map.items():
        if key not in _THINKING_LEVELS or (mapped is not None and not isinstance(mapped, str)):
            raise ModelCatalogueValidationError("Pi thinking-level map contains invalid metadata")
    supported: list[str] = []
    for level in _THINKING_LEVELS:
        if level in level_map and level_map[level] is None:
            continue
        if level in {"xhigh", "max"} and level not in level_map:
            continue
        supported.append(level)
    return tuple(supported)


def _required_text(
    value: Mapping[str, object],
    field: str,
    *,
    maximum: int,
) -> str:
    text = _optional_text(value, field, maximum=maximum)
    if text is None:
        raise ModelCatalogueValidationError(f"Pi model field {field} must be text")
    return text


def _optional_text(
    value: Mapping[str, object],
    field: str,
    *,
    maximum: int,
) -> str | None:
    item = value.get(field)
    if item is None:
        return None
    if (
        not isinstance(item, str)
        or not item.strip()
        or item != item.strip()
        or len(item) > maximum
        or any(ord(character) < 32 for character in item)
    ):
        raise ModelCatalogueValidationError(f"Pi model field {field} must be bounded text")
    return item


def _input_modalities(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ModelCatalogueValidationError("Pi model input capability must be a non-empty list")
    if not all(isinstance(item, str) and item in {"text", "image"} for item in value) or len(set(value)) != len(value):
        raise ModelCatalogueValidationError("Pi model input capability contains invalid metadata")
    return list(value)


def _build_launch(
    lane: ModelDiscoveryBackendInventory,
    *,
    service_os_user: str,
    environment: Mapping[str, str],
) -> _PiLaunch:
    command = lane.executable.configured_path
    if command is None:
        raise ModelDiscoveryUnsupported
    try:
        home = Path(pwd.getpwnam(lane.effective_os_user).pw_dir)
    except KeyError as exc:
        raise ModelDiscoveryUnsupported from exc

    try:
        provider, model_id, thinking = _split_pi_model(lane.default_model, lane.provider)
    except ValueError as exc:
        raise ModelDiscoveryUnsupported from exc
    env = sanitize_agent_environment(dict(environment))
    selected_provider_vars = frozenset(_pi_provider_env_vars(provider))
    for name in _PI_ALL_PROVIDER_ENV_VARS - selected_provider_vars:
        env.pop(name, None)
    # Disable Pi's version check and telemetry while leaving provider-owned
    # dynamic catalogue refreshes available to this metadata operation.
    env["PI_SKIP_VERSION_CHECK"] = "1"
    env["PI_TELEMETRY"] = "0"

    argv = [
        command,
        "--mode",
        "rpc",
        "--provider",
        provider,
        "--model",
        f"{provider}/{model_id}",
    ]
    if thinking is not None:
        argv.extend(("--thinking", thinking))
    argv.extend(
        (
            "--no-session",
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-tools",
        )
    )

    target_user = None if lane.effective_os_user == service_os_user else resolve_claude_user(lane.effective_os_user)
    if target_user is not None:
        env["TMPDIR"] = str(DATA_DIR / "tmp" / target_user)
        argv = wrap_command_for_target_user(
            argv,
            target_user=target_user,
            working_directory=home,
            preserve_env=(
                *_pi_provider_env_vars(provider),
                *_PRESERVED_RUNTIME_VARS,
            ),
        )
    return _PiLaunch(
        tuple(argv),
        subprocess_spawn_cwd(home, target_user=target_user),
        env,
        target_user,
    )


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(64 * 1024):
        pass


async def _close_process(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
    *,
    target_user: str | None,
) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await process.wait()
    except TimeoutError:
        if target_user is not None:
            await _kill_target_user_tree(
                target_user=target_user,
                pgid=process.pid,
                purpose="model discovery timeout",
                backend="pi",
            )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await stderr_task
    except TimeoutError:
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass


async def _finish_process_cleanup(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
    *,
    target_user: str | None,
) -> None:
    cleanup = asyncio.create_task(
        _close_process(
            process,
            stderr_task,
            target_user=target_user,
        )
    )
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise
