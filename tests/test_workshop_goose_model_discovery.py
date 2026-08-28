from __future__ import annotations

import asyncio
import json
import os
import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.workshop import goose_model_discovery as discovery_module
from kai.workshop import model_catalogue as catalogue_module
from kai.workshop.goose_model_discovery import GooseModelDiscoveryAdapter
from kai.workshop.model_catalogue import (
    ModelCatalogueValidationError,
    ModelDiscoveryAuthenticationError,
    ModelDiscoveryUnsupported,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryAuthContext,
    ModelDiscoveryAuthMode,
    ModelDiscoveryBackendInventory,
    ModelDiscoveryCacheInputs,
    ModelDiscoveryExecutableIdentity,
    ModelDiscoveryReadiness,
    ModelDiscoverySelectionStatus,
)


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _lane(
    executable: Path,
    *,
    value: int = 1,
    os_user: str | None = None,
    provider: str = "openrouter",
    backend: str = "goose",
) -> ModelDiscoveryBackendInventory:
    from kai.workshop.domain import PrincipalId, RuntimeProfileId

    principal_id = _id(PrincipalId, value)
    profile_id = _id(RuntimeProfileId, value)
    effective_user = os_user or pwd.getpwuid(os.getuid()).pw_name
    executable_identity = ModelDiscoveryExecutableIdentity(
        configured_path=str(executable),
        resolved_path=str(executable.resolve()),
        fingerprint=f"executable-{value}",
        readiness=ModelDiscoveryReadiness.READY,
        diagnostic=None,
    )
    auth = ModelDiscoveryAuthContext(
        ModelDiscoveryAuthMode.BACKEND_MANAGED,
        "backend configuration",
        None,
        f"auth-{value}",
    )
    cache_inputs = ModelDiscoveryCacheInputs(
        version=1,
        principal_id=principal_id,
        runtime_profile_id=profile_id,
        backend=backend,
        provider=provider,
        os_user=effective_user,
        executable_fingerprint=executable_identity.fingerprint,
        auth_fingerprint=auth.fingerprint,
        default_model="configured-model",
        allowed_models=None,
        role_models=(),
    )
    return ModelDiscoveryBackendInventory(
        option_id=f"{backend}:{provider}",
        backend=backend,
        provider=provider,
        selected=True,
        status=ModelDiscoverySelectionStatus.SELECTED,
        readiness=ModelDiscoveryReadiness.UNVERIFIED,
        effective_os_user=effective_user,
        executable=executable_identity,
        auth=auth,
        default_model="configured-model",
        allowed_models=None,
        role_models=(),
        cache_inputs=cache_inputs,
        cache_key=f"cache-{value}",
        diagnostic=None,
    )


def _fake_goose(
    tmp_path: Path,
    *,
    result: object | None = None,
    error: object | None = None,
) -> tuple[Path, Path]:
    executable = tmp_path / "goose-fixture"
    request_log = tmp_path / "requests.jsonl"
    script = f"""#!{sys.executable}
import json
import os
import sys

request_log = {str(request_log)!r}
result = {result!r}
error = {error!r}

for index in range(2):
    line = sys.stdin.readline()
    if not line:
        sys.exit(2)
    request = json.loads(line)
    with open(request_log, "a", encoding="utf-8") as output:
        output.write(json.dumps({{
            "method": request.get("method"),
            "params": request.get("params"),
            "provider": os.environ.get("GOOSE_PROVIDER"),
            "model": os.environ.get("GOOSE_MODEL"),
            "has_provider_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            "has_control_secret": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        }}) + "\\n")
    if index == 0:
        response = {{"jsonrpc": "2.0", "id": request["id"], "result": {{"protocolVersion": "v1"}}}}
    elif error is not None:
        response = {{"jsonrpc": "2.0", "id": request["id"], "error": error}}
    else:
        response = {{"jsonrpc": "2.0", "id": request["id"], "result": result}}
    sys.stdout.write(json.dumps(response) + "\\n")
    sys.stdout.flush()

sys.stdin.read()
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable, request_log


def test_source_is_a_valid_canonical_catalogue_identifier() -> None:
    normalized = catalogue_module._normalize_batch(
        catalogue_module.ModelDiscoveryBatch(discovery_module._SOURCE, ()),
    )
    assert normalized.source == discovery_module._SOURCE


async def test_adapter_uses_goose_metadata_request_without_session_or_generation(
    tmp_path: Path,
) -> None:
    executable, request_log = _fake_goose(
        tmp_path,
        result={
            "providerId": "openrouter",
            "models": [
                "anthropic/claude-sonnet-4-6",
                "openai/gpt-5.5",
            ],
        },
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = GooseModelDiscoveryAdapter(
        service_os_user=current_user,
        environment={
            "OPENROUTER_API_KEY": "provider-secret",
            "TELEGRAM_BOT_TOKEN": "control-plane-secret",
        },
    )

    batch = await adapter.discover(_lane(executable, os_user=current_user))

    assert batch.source == "goose-acp-provider-supported-models"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
    ]
    assert batch.models[0].display_label == "anthropic/claude-sonnet-4-6"
    assert batch.models[0].capabilities == {"provider_id": "openrouter"}
    assert "provider-secret" not in repr(batch)
    requests = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "_goose/unstable/providers/supported-models/list",
    ]
    assert requests[1]["params"] == {"providerId": "openrouter"}
    assert all(request["provider"] == "openrouter" for request in requests)
    assert all(request["model"] == "configured-model" for request in requests)
    assert all(request["has_provider_key"] is True for request in requests)
    assert all(request["has_control_secret"] is False for request in requests)
    assert not any(request["method"] in {"session/new", "session/prompt"} for request in requests)


async def test_adapter_uses_goose_wire_name_for_deepseek(tmp_path: Path) -> None:
    executable, request_log = _fake_goose(
        tmp_path,
        result={"providerId": "custom_deepseek", "models": ["deepseek-v4-pro"]},
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = GooseModelDiscoveryAdapter(service_os_user=current_user, environment={})

    batch = await adapter.discover(
        _lane(executable, os_user=current_user, provider="deepseek"),
    )

    assert [candidate.model_id for candidate in batch.models] == ["deepseek-v4-pro"]
    requests = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    assert requests[1]["params"] == {"providerId": "custom_deepseek"}
    assert requests[1]["provider"] == "custom_deepseek"


@pytest.mark.parametrize(
    ("error", "exception"),
    [
        (
            {"code": -32000, "message": "Authentication required", "data": "credentials rejected"},
            ModelDiscoveryAuthenticationError,
        ),
        (
            {"code": -32602, "message": "Invalid params", "data": "Provider is not configured: openai"},
            ModelDiscoveryAuthenticationError,
        ),
        (
            {"code": -32602, "message": "Invalid params", "data": "Unknown provider: missing"},
            ModelDiscoveryUnsupported,
        ),
        (
            {"code": -32601, "message": "Method not found"},
            ModelDiscoveryUnsupported,
        ),
    ],
)
async def test_adapter_classifies_authentication_and_unsupported_failures(
    tmp_path: Path,
    error: object,
    exception: type[Exception],
) -> None:
    executable, _ = _fake_goose(tmp_path, error=error)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = GooseModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(exception):
        await adapter.discover(_lane(executable, os_user=current_user))


def test_parser_accepts_empty_models_and_rejects_unsafe_content() -> None:
    assert (
        discovery_module._parse_supported_models(
            {"providerId": "openai", "models": []},
            "openai",
        )
        == ()
    )

    with pytest.raises(ModelCatalogueValidationError, match="mismatched provider"):
        discovery_module._parse_supported_models(
            {"providerId": "anthropic", "models": []},
            "openai",
        )
    with pytest.raises(ModelCatalogueValidationError, match="duplicate"):
        discovery_module._parse_supported_models(
            {"providerId": "openai", "models": ["gpt-5.5", "gpt-5.5"]},
            "openai",
        )
    with pytest.raises(ModelCatalogueValidationError, match="invalid model"):
        discovery_module._parse_supported_models(
            {"providerId": "openai", "models": [" secret\n"]},
            "openai",
        )


def test_launch_uses_canonical_user_auth_context_and_scrubs_control_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "goose"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    alice_home = tmp_path / "alice"
    alice_home.mkdir()
    monkeypatch.setattr(
        discovery_module.pwd,
        "getpwnam",
        lambda user: SimpleNamespace(pw_dir=str(alice_home)),
    )
    environment = {
        "OPENROUTER_API_KEY": "provider-key",
        "XDG_CONFIG_HOME": "/configured/goose",
        "TELEGRAM_BOT_TOKEN": "control-plane-secret",
    }

    launch = discovery_module._build_launch(
        _lane(executable, os_user="alice"),
        service_os_user="kai",
        environment=environment,
    )

    assert launch.argv[:6] == ("sudo", "-H", "-D", str(alice_home), "-u", "alice")
    assert launch.argv[-3:] == ("--", str(executable), "acp")
    assert launch.cwd is None
    assert launch.target_user == "alice"
    assert launch.environment["OPENROUTER_API_KEY"] == "provider-key"
    assert "XDG_CONFIG_HOME" not in launch.environment
    assert launch.environment["GOOSE_PROVIDER"] == "openrouter"
    assert launch.environment["GOOSE_MODEL"] == "configured-model"
    assert "TELEGRAM_BOT_TOKEN" not in launch.environment
    assert launch.environment["TMPDIR"].endswith("/tmp/alice")
    preserve_arg = next(arg for arg in launch.argv if arg.startswith("--preserve-env="))
    assert "OPENROUTER_API_KEY" in preserve_arg
    assert "XDG_CONFIG_HOME" not in preserve_arg
    assert "TELEGRAM_BOT_TOKEN" not in preserve_arg


def test_adapter_rejects_non_goose_or_unavailable_lanes(tmp_path: Path) -> None:
    executable = tmp_path / "goose"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    adapter = GooseModelDiscoveryAdapter(service_os_user="kai", environment={})

    with pytest.raises(ModelDiscoveryUnsupported):
        adapter._validate_lane(_lane(executable, backend="opencode"))

    lane = _lane(executable)
    unavailable = ModelDiscoveryExecutableIdentity(
        configured_path=lane.executable.configured_path,
        resolved_path=None,
        fingerprint=lane.executable.fingerprint,
        readiness=ModelDiscoveryReadiness.UNAVAILABLE,
        diagnostic="missing",
    )
    lane = ModelDiscoveryBackendInventory(
        option_id=lane.option_id,
        backend=lane.backend,
        provider=lane.provider,
        selected=lane.selected,
        status=lane.status,
        readiness=lane.readiness,
        effective_os_user=lane.effective_os_user,
        executable=unavailable,
        auth=lane.auth,
        default_model=lane.default_model,
        allowed_models=lane.allowed_models,
        role_models=lane.role_models,
        cache_inputs=lane.cache_inputs,
        cache_key=lane.cache_key,
        diagnostic=lane.diagnostic,
    )
    with pytest.raises(ModelDiscoveryUnsupported):
        adapter._validate_lane(lane)


async def test_timeout_reaps_goose_metadata_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "goose-hung-fixture"
    pid_file = tmp_path / "pid"
    script = f"""#!{sys.executable}
import json
import os
import sys
import time
from pathlib import Path
Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
sys.stdin.readline()
time.sleep(60)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(discovery_module, "_RPC_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(discovery_module, "_CLOSE_TIMEOUT_SECONDS", 0.2)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = GooseModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(discovery_module._GooseProtocolError, match="timed out"):
        await adapter.discover(_lane(executable, os_user=current_user))

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_cancellation_reaps_goose_metadata_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "goose-cancel-fixture"
    pid_file = tmp_path / "pid"
    script = f"""#!{sys.executable}
import os
import sys
import time
from pathlib import Path
Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
sys.stdin.readline()
time.sleep(60)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(discovery_module, "_RPC_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(discovery_module, "_CLOSE_TIMEOUT_SECONDS", 0.2)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = GooseModelDiscoveryAdapter(service_os_user=current_user, environment={})
    task = asyncio.create_task(adapter.discover(_lane(executable, os_user=current_user)))
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
