"""Metadata-only OpenCode model discovery through the headless server API.

Protocol contract verified 2026-08-28 against OpenCode's server reference and
generated SDK types.  The adapter starts a short-lived, loopback-only
``opencode serve`` process and reads ``GET /provider``.  That documented route
returns both the provider catalogue and the providers connected in the
effective user authentication context.  No session or generation endpoint is
used.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pwd
import secrets
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

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

_SOURCE = "opencode-server-provider"
_TTL_SECONDS = 21_600
_STARTUP_TIMEOUT_SECONDS = 10.0
_REQUEST_TIMEOUT_SECONDS = 20.0
_POLL_INTERVAL_SECONDS = 0.05
_CLOSE_TIMEOUT_SECONDS = 3.0
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_PROVIDERS = 1_000
_MAX_MODELS = 20_000
_SERVER_USERNAME = "kai-model-discovery"
_PRESERVED_AUTH_VARS = (
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENCODE_CONFIG",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_MODELS_URL",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
)


class _OpenCodeDiscoveryError(RuntimeError):
    """A sanitized headless-server discovery failure."""


@dataclass(frozen=True, slots=True)
class _OpenCodeLaunch:
    argv: tuple[str, ...]
    cwd: str | None
    environment: Mapping[str, str]
    directory: Path
    password: str


class OpenCodeModelDiscoveryAdapter:
    """Enumerate models visible to one canonical OpenCode auth context."""

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
        port = _reserve_loopback_port()
        launch = _build_launch(
            lane,
            port=port,
            service_os_user=self._service_os_user,
            environment=self._environment,
        )
        process = await asyncio.create_subprocess_exec(
            *launch.argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=launch.cwd,
            env=dict(launch.environment),
            start_new_session=True,
        )
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
            credential = base64.b64encode(f"{_SERVER_USERNAME}:{launch.password}".encode()).decode("ascii")
            headers = {"Authorization": f"Basic {credential}"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                await _wait_until_ready(session, process, port)
                body = await _get_provider_catalogue(session, port, launch.directory)
            candidates = _parse_provider_catalogue(body, lane.provider)
            return ModelDiscoveryBatch(_SOURCE, candidates, ttl_seconds=_TTL_SECONDS)
        finally:
            await _finish_process_cleanup(process)

    @staticmethod
    def _validate_lane(lane: ModelDiscoveryBackendInventory) -> None:
        if lane.backend != "opencode":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if not lane.executable.configured_path or not lane.executable.resolved_path:
            raise ModelDiscoveryUnsupported


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    if not isinstance(port, int) or port <= 0:
        raise _OpenCodeDiscoveryError("OpenCode discovery could not reserve a loopback port")
    return port


def _build_launch(
    lane: ModelDiscoveryBackendInventory,
    *,
    port: int,
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
    password = secrets.token_urlsafe(32)
    env = sanitize_agent_environment(dict(environment))
    # Discovery must load the effective user's real OpenCode configuration,
    # not a one-turn model override inherited from another process.
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    env["OPENCODE_SERVER_USERNAME"] = _SERVER_USERNAME
    env["OPENCODE_SERVER_PASSWORD"] = password
    target_user = None if lane.effective_os_user == service_os_user else resolve_claude_user(lane.effective_os_user)
    argv = [
        command,
        "serve",
        "--hostname",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if target_user is not None:
        env["TMPDIR"] = str(DATA_DIR / "tmp" / target_user)
        argv = wrap_command_for_target_user(
            argv,
            target_user=target_user,
            working_directory=home,
            preserve_env=(
                *_PRESERVED_AUTH_VARS,
                "OPENCODE_SERVER_USERNAME",
                "OPENCODE_SERVER_PASSWORD",
                "TMPDIR",
            ),
        )
    return _OpenCodeLaunch(
        tuple(argv),
        subprocess_spawn_cwd(home, target_user=target_user),
        env,
        home,
        password,
    )


async def _wait_until_ready(
    session: aiohttp.ClientSession,
    process: asyncio.subprocess.Process,
    port: int,
) -> None:
    url = f"http://127.0.0.1:{port}/global/health"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STARTUP_TIMEOUT_SECONDS
    while True:
        if process.returncode is not None:
            raise _OpenCodeDiscoveryError("OpenCode metadata server exited during startup")
        try:
            async with session.get(url, allow_redirects=False) as response:
                if response.status == 200:
                    body = await response.content.read(1025)
                    if len(body) > 1024:
                        raise ModelCatalogueValidationError("OpenCode health response exceeds the safety limit")
                    value = json.loads(body)
                    if isinstance(value, dict) and value.get("healthy") is True:
                        return
        except (aiohttp.ClientError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        if loop.time() >= deadline:
            raise _OpenCodeDiscoveryError("OpenCode metadata server did not become ready")
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


async def _get_provider_catalogue(
    session: aiohttp.ClientSession,
    port: int,
    directory: Path,
) -> bytes:
    query = urlencode({"directory": str(directory)})
    url = f"http://127.0.0.1:{port}/provider?{query}"
    try:
        async with session.get(url, allow_redirects=False) as response:
            if response.status in {401, 403}:
                raise ModelDiscoveryAuthenticationError
            if response.status < 200 or response.status >= 300:
                raise _OpenCodeDiscoveryError("OpenCode provider metadata request failed")
            body = await response.content.read(_MAX_RESPONSE_BYTES + 1)
    except (ModelDiscoveryAuthenticationError, _OpenCodeDiscoveryError):
        raise
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise _OpenCodeDiscoveryError("OpenCode provider metadata request failed") from exc
    if len(body) > _MAX_RESPONSE_BYTES:
        raise ModelCatalogueValidationError("OpenCode provider catalogue exceeds the safety limit")
    return body


def _parse_provider_catalogue(body: bytes, provider_id: str) -> tuple[ModelDiscoveryCandidate, ...]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogueValidationError("OpenCode provider catalogue is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("OpenCode provider catalogue must be an object")
    providers = value.get("all")
    connected = value.get("connected")
    if not isinstance(providers, list) or len(providers) > _MAX_PROVIDERS:
        raise ModelCatalogueValidationError("OpenCode provider catalogue has an invalid provider list")
    if not isinstance(connected, list) or not all(isinstance(item, str) and item for item in connected):
        raise ModelCatalogueValidationError("OpenCode provider catalogue has an invalid connected list")

    matching = [provider for provider in providers if isinstance(provider, dict) and provider.get("id") == provider_id]
    if len(matching) > 1:
        raise ModelCatalogueValidationError("OpenCode provider catalogue contains a duplicate provider")
    if not matching:
        raise ModelDiscoveryUnsupported
    if provider_id not in connected:
        raise ModelDiscoveryAuthenticationError
    provider = matching[0]
    name = _required_text(provider, "name", context="provider")
    models = provider.get("models")
    if not isinstance(models, dict) or len(models) > _MAX_MODELS:
        raise ModelCatalogueValidationError("OpenCode provider model catalogue must be an object")
    return tuple(_parse_model(provider_id, name, key, model) for key, model in sorted(models.items()))


def _parse_model(
    provider_id: str,
    provider_name: str,
    catalogue_key: object,
    value: object,
) -> ModelDiscoveryCandidate:
    if not isinstance(catalogue_key, str) or not catalogue_key:
        raise ModelCatalogueValidationError("OpenCode model catalogue key must be text")
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("OpenCode model entry must be an object")
    model_id = _required_text(value, "id", context="model")
    if catalogue_key != model_id:
        raise ModelCatalogueValidationError("OpenCode model key and identifier do not match")
    full_id = f"{provider_id}/{model_id}"
    if not is_opencode_model_shape(full_id):
        raise ModelCatalogueValidationError("OpenCode returned an invalid provider/model identifier")

    capabilities: dict[str, object] = {
        "provider_name": provider_name,
        "release_date": _required_text(value, "release_date", context="model"),
        "attachment": _required_bool(value, "attachment"),
        "reasoning": _required_bool(value, "reasoning"),
        "temperature": _required_bool(value, "temperature"),
        "tool_call": _required_bool(value, "tool_call"),
        "experimental": _optional_bool(value, "experimental"),
        "status": _optional_status(value),
        "limits": _limits(value.get("limit")),
        "modalities": _modalities(value.get("modalities")),
    }
    return ModelDiscoveryCandidate(
        full_id,
        _required_text(value, "name", context="model"),
        capabilities,
    )


def _required_text(value: Mapping[str, object], field: str, *, context: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ModelCatalogueValidationError(f"OpenCode {context} field {field} must be text")
    return item


def _required_bool(value: Mapping[str, object], field: str) -> bool:
    item = value.get(field)
    if not isinstance(item, bool):
        raise ModelCatalogueValidationError(f"OpenCode model field {field} must be boolean")
    return item


def _optional_bool(value: Mapping[str, object], field: str) -> bool | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, bool):
        raise ModelCatalogueValidationError(f"OpenCode model field {field} must be boolean or null")
    return item


def _optional_status(value: Mapping[str, object]) -> str | None:
    status = value.get("status")
    if status is None:
        return None
    if status not in {"active", "alpha", "beta", "deprecated"}:
        raise ModelCatalogueValidationError("OpenCode model status is invalid")
    assert isinstance(status, str)
    return status


def _limits(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("OpenCode model limits must be an object")
    return {
        "context": _nonnegative_integer(value, "context"),
        "output": _nonnegative_integer(value, "output"),
    }


def _nonnegative_integer(value: Mapping[str, object], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ModelCatalogueValidationError(f"OpenCode model field {field} must be a non-negative integer")
    return item


def _modalities(value: object) -> dict[str, list[str]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("OpenCode model modalities must be an object or null")
    return {
        "input": _modality_list(value.get("input"), "input"),
        "output": _modality_list(value.get("output"), "output"),
    }


def _modality_list(value: object, field: str) -> list[str]:
    allowed = {"audio", "image", "pdf", "text", "video"}
    if not isinstance(value, list) or not all(isinstance(item, str) and item in allowed for item in value):
        raise ModelCatalogueValidationError(f"OpenCode model {field} modalities are invalid")
    return list(value)


async def _close_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await process.wait()
            return
    except TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


async def _finish_process_cleanup(process: asyncio.subprocess.Process) -> None:
    cleanup = asyncio.create_task(_close_process(process))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise
