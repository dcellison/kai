"""Metadata-only Goose model discovery through Goose's ACP schema.

Protocol contract verified 2026-08-28 against Goose's generated ACP schema
and ``ProviderSupportedModelsListRequest`` implementation.  The adapter starts
a short-lived ``goose acp`` process in the canonical runtime profile's effective
OS-user context, initializes ACP, and sends only
``_goose/unstable/providers/supported-models/list``.  Goose delegates that
request to the selected provider's native ``fetch_supported_models`` method.
No session is created and no prompt or model-generation request is sent.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import kai
from kai.acp import _kill_target_user_tree
from kai.backend import sanitize_agent_environment
from kai.config import DATA_DIR, resolve_claude_user
from kai.goose import goose_provider_id
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

_SOURCE = "goose-acp-provider-supported-models"
_METHOD = "_goose/unstable/providers/supported-models/list"
_TTL_SECONDS = 21_600
_RPC_TIMEOUT_SECONDS = 20.0
_CLOSE_TIMEOUT_SECONDS = 3.0
_STREAM_LIMIT = 16 * 1024 * 1024
_MAX_MODELS = 20_000
_MAX_MODEL_ID_CHARACTERS = 512
_AUTH_MARKERS = (
    "auth",
    "credential",
    "expired",
    "not configured",
    "sign in",
    "token",
    "unauthorized",
)
_UNSUPPORTED_MARKERS = (
    "does not support",
    "not supported",
    "unknown provider",
)
_PRESERVED_AUTH_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_HOST",
    "OPENAI_HOST",
    "OPENAI_BASE_PATH",
    "OLLAMA_HOST",
    "GOOSE_PROVIDER",
    "GOOSE_MODEL",
)


class _GooseProtocolError(RuntimeError):
    """A sanitized Goose ACP metadata failure."""


class _GooseRpcError(_GooseProtocolError):
    def __init__(self, *, authentication: bool, unsupported: bool) -> None:
        super().__init__("Goose ACP rejected a metadata request")
        self.authentication = authentication
        self.unsupported = unsupported


@dataclass(frozen=True, slots=True)
class _GooseLaunch:
    argv: tuple[str, ...]
    cwd: str | None
    environment: Mapping[str, str]
    target_user: str | None


class GooseModelDiscoveryAdapter:
    """Enumerate models visible to one canonical Goose provider context."""

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
            limit=_STREAM_LIMIT,
        )
        stderr_task = asyncio.create_task(_discard_stream(process.stderr))
        try:
            if process.stdin is None or process.stdout is None:
                raise _GooseProtocolError("Goose ACP pipes are unavailable")
            rpc = _GooseRpcClient(process.stdin, process.stdout)
            await rpc.request(
                "initialize",
                {
                    "protocolVersion": "v1",
                    "clientInfo": {
                        "name": "kai-model-discovery",
                        "version": kai.__version__,
                    },
                },
            )
            result = await rpc.request(
                _METHOD,
                {"providerId": goose_provider_id(lane.provider)},
            )
            candidates = _parse_supported_models(result, goose_provider_id(lane.provider))
            return ModelDiscoveryBatch(_SOURCE, candidates, ttl_seconds=_TTL_SECONDS)
        except _GooseRpcError as exc:
            if exc.authentication:
                raise ModelDiscoveryAuthenticationError from exc
            if exc.unsupported:
                raise ModelDiscoveryUnsupported from exc
            raise _GooseProtocolError from exc
        finally:
            await _finish_process_cleanup(
                process,
                stderr_task,
                target_user=launch.target_user,
            )

    @staticmethod
    def _validate_lane(lane: ModelDiscoveryBackendInventory) -> None:
        if lane.backend != "goose":
            raise ModelDiscoveryUnsupported
        if lane.executable.readiness != ModelDiscoveryReadiness.READY:
            raise ModelDiscoveryUnsupported
        if not lane.executable.configured_path or not lane.executable.resolved_path:
            raise ModelDiscoveryUnsupported


class _GooseRpcClient:
    def __init__(self, stdin: asyncio.StreamWriter, stdout: asyncio.StreamReader) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._next_id = 1

    async def request(self, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        request_id = self._next_id
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self._stdin.write(json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
        await self._stdin.drain()
        while True:
            try:
                line = await asyncio.wait_for(
                    self._stdout.readline(),
                    timeout=_RPC_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                raise _GooseProtocolError("Goose ACP metadata request timed out") from exc
            if not line:
                raise _GooseProtocolError("Goose ACP exited before responding")
            try:
                response = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModelCatalogueValidationError("Goose ACP emitted malformed JSON") from exc
            if not isinstance(response, dict) or response.get("id") != request_id:
                continue
            if "error" in response:
                raise _classify_rpc_error(response.get("error"))
            result = response.get("result")
            if not isinstance(result, dict):
                raise ModelCatalogueValidationError("Goose ACP metadata response has no result object")
            return result


def _classify_rpc_error(value: object) -> _GooseRpcError:
    if not isinstance(value, dict):
        return _GooseRpcError(authentication=False, unsupported=False)
    code = value.get("code")
    parts = (value.get("message"), value.get("data"))
    text = " ".join(item for item in parts if isinstance(item, str)).lower()
    authentication = code in {401, 403} or any(marker in text for marker in _AUTH_MARKERS)
    unsupported = not authentication and (code == -32601 or any(marker in text for marker in _UNSUPPORTED_MARKERS))
    return _GooseRpcError(authentication=authentication, unsupported=unsupported)


def _parse_supported_models(
    result: Mapping[str, object],
    expected_provider_id: str,
) -> tuple[ModelDiscoveryCandidate, ...]:
    provider_id = result.get("providerId")
    if provider_id != expected_provider_id:
        raise ModelCatalogueValidationError("Goose ACP returned a mismatched provider identifier")
    models = result.get("models")
    if not isinstance(models, list) or len(models) > _MAX_MODELS:
        raise ModelCatalogueValidationError("Goose ACP model catalogue must be a bounded list")
    candidates: list[ModelDiscoveryCandidate] = []
    seen: set[str] = set()
    for value in models:
        if (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or len(value) > _MAX_MODEL_ID_CHARACTERS
        ):
            raise ModelCatalogueValidationError("Goose ACP returned an invalid model identifier")
        if value in seen:
            raise ModelCatalogueValidationError("Goose ACP returned a duplicate model identifier")
        seen.add(value)
        candidates.append(
            ModelDiscoveryCandidate(
                value,
                value,
                {"provider_id": expected_provider_id},
            )
        )
    return tuple(candidates)


def _build_launch(
    lane: ModelDiscoveryBackendInventory,
    *,
    service_os_user: str,
    environment: Mapping[str, str],
) -> _GooseLaunch:
    command = lane.executable.configured_path
    if command is None:
        raise ModelDiscoveryUnsupported
    try:
        home = Path(pwd.getpwnam(lane.effective_os_user).pw_dir)
    except KeyError as exc:
        raise ModelDiscoveryUnsupported from exc
    env = sanitize_agent_environment(dict(environment))
    env["GOOSE_PROVIDER"] = goose_provider_id(lane.provider)
    env["GOOSE_MODEL"] = lane.default_model
    target_user = None if lane.effective_os_user == service_os_user else resolve_claude_user(lane.effective_os_user)
    argv = [command, "acp"]
    if target_user is not None:
        # ``sudo -H`` selects the target user's Goose configuration.  Do not
        # carry a service-user XDG override across that identity boundary.
        env.pop("XDG_CONFIG_HOME", None)
        env.pop("XDG_DATA_HOME", None)
        env["TMPDIR"] = str(DATA_DIR / "tmp" / target_user)
        argv = wrap_command_for_target_user(
            argv,
            target_user=target_user,
            working_directory=home,
            preserve_env=(*_PRESERVED_AUTH_VARS, "TMPDIR"),
        )
    return _GooseLaunch(
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
                backend="goose",
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
