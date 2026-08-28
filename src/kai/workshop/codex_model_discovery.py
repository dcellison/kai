"""Metadata-only Codex model discovery through the app-server protocol.

Protocol contract verified 2026-08-28 against OpenAI's
``codex-rs/app-server-protocol/src/protocol/v2/model.rs`` and the app-server
README. The supported request is paginated ``model/list``; the provider model
identifier is the response's ``model`` field, separate from ``displayName``.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kai
from kai.backend import sanitize_agent_environment
from kai.config import DATA_DIR, resolve_claude_user
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

_SOURCE = "codex-app-server:model/list"
_TTL_SECONDS = 21_600
_PAGE_LIMIT = 100
_MAX_PAGES = 20
_MAX_MODELS = _PAGE_LIMIT * _MAX_PAGES
_STREAM_LIMIT = 16 * 1024 * 1024
_CLOSE_TIMEOUT_SECONDS = 3.0
_AUTH_MARKERS = (
    "401",
    "auth",
    "credential",
    "expired",
    "log in",
    "logged in",
    "login",
    "sign in",
    "signed in",
    "token",
    "unauthorized",
)
_PRESERVED_AUTH_VARS = (
    "CODEX_HOME",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


class _CodexProtocolError(RuntimeError):
    """A sanitized app-server protocol failure."""


class _CodexRpcError(_CodexProtocolError):
    def __init__(self, code: int | None, message: str) -> None:
        super().__init__("Codex app-server rejected a metadata request")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class _CodexLaunch:
    argv: tuple[str, ...]
    cwd: str | None
    environment: Mapping[str, str]
    cross_user: bool


class CodexModelDiscoveryAdapter:
    """Enumerate the models visible to one canonical Codex auth context.

    The adapter starts a short-lived ``codex app-server``, performs only the
    required initialize handshake and paginated ``model/list`` requests, then
    closes stdin. It never starts a thread, turn, or model-generation request.
    """

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
            start_new_session=launch.cross_user,
            limit=_STREAM_LIMIT,
        )
        stderr_task = asyncio.create_task(_discard_stream(process.stderr))
        try:
            if process.stdin is None or process.stdout is None:
                raise _CodexProtocolError("Codex app-server pipes are unavailable")
            rpc = _CodexRpcClient(process.stdin, process.stdout)
            await rpc.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "kai-model-discovery",
                        "version": kai.__version__,
                    },
                    "capabilities": {
                        "optOutNotificationMethods": [
                            "account/updated",
                            "mcpServer/startupStatus/updated",
                            "remoteControl/status/changed",
                        ]
                    },
                },
            )
            await rpc.notify("initialized")
            models = await _list_models(rpc)
            return ModelDiscoveryBatch(_SOURCE, models, ttl_seconds=_TTL_SECONDS)
        except _CodexRpcError as exc:
            if _is_authentication_error(exc):
                raise ModelDiscoveryAuthenticationError from exc
            raise _CodexProtocolError from exc
        finally:
            await _finish_process_cleanup(process, stderr_task)

    @staticmethod
    def _validate_lane(lane: ModelDiscoveryBackendInventory) -> None:
        if lane.backend != "codex" or lane.provider != "openai":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if not lane.executable.configured_path or not lane.executable.resolved_path:
            raise ModelDiscoveryUnsupported
        if lane.auth.mode == ModelDiscoveryAuthMode.API_KEY and lane.auth.configured is not True:
            raise ModelDiscoveryAuthenticationError


class _CodexRpcClient:
    def __init__(self, stdin: asyncio.StreamWriter, stdout: asyncio.StreamReader) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._next_id = 1

    async def notify(self, method: str) -> None:
        await self._write({"method": method})

    async def request(self, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        request_id = self._next_id
        self._next_id += 1
        await self._write({"id": request_id, "method": method, "params": params})
        while True:
            line = await self._stdout.readline()
            if not line:
                raise _CodexProtocolError("Codex app-server closed before responding")
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModelCatalogueValidationError("Codex app-server emitted malformed JSON") from exc
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            error = message.get("error")
            if error is not None:
                raise _rpc_error(error)
            result = message.get("result")
            if not isinstance(result, dict):
                raise ModelCatalogueValidationError("Codex app-server response has no result object")
            return result

    async def _write(self, message: Mapping[str, object]) -> None:
        self._stdin.write(json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n")
        await self._stdin.drain()


async def _list_models(rpc: _CodexRpcClient) -> tuple[ModelDiscoveryCandidate, ...]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    candidates: list[ModelDiscoveryCandidate] = []
    for _page in range(_MAX_PAGES):
        result = await rpc.request(
            "model/list",
            {
                "cursor": cursor,
                "limit": _PAGE_LIMIT,
                "includeHidden": False,
            },
        )
        page, next_cursor = _parse_model_list_page(result)
        candidates.extend(page)
        if len(candidates) > _MAX_MODELS:
            raise ModelCatalogueValidationError("Codex model catalogue exceeds the safety limit")
        if next_cursor is None:
            return tuple(candidates)
        if next_cursor in seen_cursors:
            raise ModelCatalogueValidationError("Codex model catalogue repeated a pagination cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise ModelCatalogueValidationError("Codex model catalogue pagination did not terminate")


def _parse_model_list_page(result: Mapping[str, object]) -> tuple[tuple[ModelDiscoveryCandidate, ...], str | None]:
    raw_models = result.get("data")
    if not isinstance(raw_models, list):
        raise ModelCatalogueValidationError("Codex model catalogue data must be a list")
    next_cursor = result.get("nextCursor")
    if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
        raise ModelCatalogueValidationError("Codex model catalogue cursor is invalid")
    models = tuple(_parse_model(value) for value in raw_models)
    return models, next_cursor


def _parse_model(value: object) -> ModelDiscoveryCandidate:
    if not isinstance(value, dict):
        raise ModelCatalogueValidationError("Codex model entry must be an object")
    model_id = _required_text(value, "model")
    display_label = _required_text(value, "displayName")
    picker_id = _required_text(value, "id")
    hidden = value.get("hidden")
    if not isinstance(hidden, bool):
        raise ModelCatalogueValidationError("Codex model hidden flag must be boolean")
    if hidden:
        raise ModelCatalogueValidationError("Codex returned a hidden model without being asked")

    capabilities: dict[str, object] = {
        "codex_picker_id": picker_id,
        "description": _required_text(value, "description", allow_empty=True),
        "hidden": hidden,
        "supported_reasoning_efforts": _reasoning_efforts(value.get("supportedReasoningEfforts")),
        "default_reasoning_effort": _required_text(value, "defaultReasoningEffort"),
        "input_modalities": _text_list(value.get("inputModalities", []), "inputModalities"),
        "supports_personality": _optional_bool(value, "supportsPersonality", False),
        "multi_agent_version": _optional_text(value, "multiAgentVersion"),
        "additional_speed_tiers": _text_list(value.get("additionalSpeedTiers", []), "additionalSpeedTiers"),
        "service_tiers": _service_tiers(value.get("serviceTiers", [])),
        "default_service_tier": _optional_text(value, "defaultServiceTier"),
        "is_default": _required_bool(value, "isDefault"),
        "model_specialty": _optional_text(value, "modelSpecialty"),
        "upgrade": _optional_text(value, "upgrade"),
    }
    return ModelDiscoveryCandidate(model_id, display_label, capabilities)


def _reasoning_efforts(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ModelCatalogueValidationError("Codex reasoning efforts must be a list")
    results: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ModelCatalogueValidationError("Codex reasoning effort must be an object")
        results.append(
            {
                "effort": _required_text(item, "reasoningEffort"),
                "description": _required_text(item, "description", allow_empty=True),
            }
        )
    return results


def _service_tiers(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ModelCatalogueValidationError("Codex service tiers must be a list")
    results: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ModelCatalogueValidationError("Codex service tier must be an object")
        results.append(
            {
                "id": _required_text(item, "id"),
                "name": _required_text(item, "name"),
                "description": _required_text(item, "description", allow_empty=True),
            }
        )
    return results


def _text_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ModelCatalogueValidationError(f"Codex {field} must be a list of text values")
    return list(value)


def _required_text(value: Mapping[str, Any], field: str, *, allow_empty: bool = False) -> str:
    item = value.get(field)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise ModelCatalogueValidationError(f"Codex model field {field} must be text")
    return item


def _optional_text(value: Mapping[str, Any], field: str) -> str | None:
    item = value.get(field)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ModelCatalogueValidationError(f"Codex model field {field} must be text or null")
    return item


def _required_bool(value: Mapping[str, Any], field: str) -> bool:
    item = value.get(field)
    if not isinstance(item, bool):
        raise ModelCatalogueValidationError(f"Codex model field {field} must be boolean")
    return item


def _optional_bool(value: Mapping[str, Any], field: str, default: bool) -> bool:
    item = value.get(field, default)
    if not isinstance(item, bool):
        raise ModelCatalogueValidationError(f"Codex model field {field} must be boolean")
    return item


def _rpc_error(value: object) -> _CodexRpcError:
    if not isinstance(value, dict):
        return _CodexRpcError(None, "")
    code = value.get("code")
    message = value.get("message")
    return _CodexRpcError(code if isinstance(code, int) else None, message if isinstance(message, str) else "")


def _is_authentication_error(error: _CodexRpcError) -> bool:
    if error.code in {401, 403}:
        return True
    lowered = error.message.lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


def _build_launch(
    lane: ModelDiscoveryBackendInventory,
    *,
    service_os_user: str,
    environment: Mapping[str, str],
) -> _CodexLaunch:
    command = lane.executable.configured_path
    if command is None:
        raise ModelDiscoveryUnsupported
    try:
        home = Path(pwd.getpwnam(lane.effective_os_user).pw_dir)
    except KeyError as exc:
        raise ModelDiscoveryUnsupported from exc
    env = sanitize_agent_environment(dict(environment))
    target_user = None if lane.effective_os_user == service_os_user else resolve_claude_user(lane.effective_os_user)
    argv = [command, "app-server"]
    if target_user is not None:
        env["TMPDIR"] = str(DATA_DIR / "tmp" / target_user)
        argv = wrap_command_for_target_user(
            argv,
            target_user=target_user,
            working_directory=home,
            preserve_env=(*_PRESERVED_AUTH_VARS, "TMPDIR"),
        )
    return _CodexLaunch(
        tuple(argv),
        subprocess_spawn_cwd(home, target_user=target_user),
        env,
        target_user is not None,
    )


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(64 * 1024):
        pass


async def _close_process(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await process.wait()
    except TimeoutError:
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
) -> None:
    cleanup = asyncio.create_task(_close_process(process, stderr_task))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        # Discovery is normally cancelled by the catalogue's timeout. Finish
        # the bounded EOF shutdown before propagating cancellation so the
        # short-lived app-server cannot become an orphaned background process.
        await cleanup
        raise
