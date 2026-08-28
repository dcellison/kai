from __future__ import annotations

import asyncio
import json
import os
import pwd
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.workshop import model_catalogue as catalogue_module
from kai.workshop import pi_model_discovery as discovery_module
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
from kai.workshop.pi_model_discovery import PiModelDiscoveryAdapter


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _lane(
    executable: Path,
    *,
    value: int = 1,
    os_user: str | None = None,
    provider: str = "openai",
    backend: str = "pi",
    model: str = "openai/gpt-5.6-sol:xhigh",
    auth_mode: ModelDiscoveryAuthMode = ModelDiscoveryAuthMode.BACKEND_MANAGED,
    auth_configured: bool | None = None,
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
        auth_mode,
        "backend configuration",
        auth_configured,
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
        default_model=model,
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
        default_model=model,
        allowed_models=None,
        role_models=(),
        cache_inputs=cache_inputs,
        cache_key=f"cache-{value}",
        diagnostic=None,
    )


def _model(
    provider: str,
    model_id: str,
    *,
    name: str | None = None,
    reasoning: bool = False,
    thinking_level_map: dict[str, str | None] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "provider": provider,
        "id": model_id,
        "name": name or model_id,
        "api": "openai-responses",
        "reasoning": reasoning,
        "input": ["text", "image"],
    }
    if thinking_level_map is not None:
        result["thinkingLevelMap"] = thinking_level_map
    return result


def _fake_pi(
    tmp_path: Path,
    *,
    data: object | None = None,
    error: str | None = None,
) -> tuple[Path, Path]:
    executable = tmp_path / "pi-fixture"
    request_log = tmp_path / "pi-request.json"
    script = f"""#!{sys.executable}
import json
import os
import sys

request_log = {str(request_log)!r}
data = {data!r}
error = {error!r}
request = json.loads(sys.stdin.readline())
with open(request_log, "w", encoding="utf-8") as output:
    json.dump({{
        "request": request,
        "argv": sys.argv[1:],
        "has_provider_key": bool(os.environ.get("OPENAI_API_KEY")),
        "has_other_provider_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "has_control_secret": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
        "skip_version_check": os.environ.get("PI_SKIP_VERSION_CHECK"),
        "telemetry": os.environ.get("PI_TELEMETRY"),
    }}, output)
if error is None:
    response = {{
        "id": request["id"],
        "type": "response",
        "command": request["type"],
        "success": True,
        "data": data,
    }}
else:
    response = {{
        "id": request["id"],
        "type": "response",
        "command": request["type"],
        "success": False,
        "error": error,
    }}
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


async def test_adapter_uses_metadata_only_rpc_in_effective_auth_context(tmp_path: Path) -> None:
    executable, request_log = _fake_pi(
        tmp_path,
        data={
            "models": [
                _model("anthropic", "claude-sonnet-4-6"),
                _model("openai", "gpt-5.6-sol", name="GPT-5.6 Sol", reasoning=True),
            ]
        },
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = PiModelDiscoveryAdapter(
        service_os_user=current_user,
        environment={
            "OPENAI_API_KEY": "provider-secret",
            "ANTHROPIC_API_KEY": "other-provider-secret",
            "TELEGRAM_BOT_TOKEN": "control-plane-secret",
        },
    )

    batch = await adapter.discover(_lane(executable, os_user=current_user))

    assert batch.source == "pi-rpc-get-available-models"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == [
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol:off",
        "openai/gpt-5.6-sol:minimal",
        "openai/gpt-5.6-sol:low",
        "openai/gpt-5.6-sol:medium",
        "openai/gpt-5.6-sol:high",
    ]
    assert batch.models[0].display_label == "GPT-5.6 Sol"
    assert batch.models[-1].display_label == "GPT-5.6 Sol - high"
    assert batch.models[0].capabilities == {
        "provider_id": "openai",
        "api": "openai-responses",
        "reasoning": True,
        "thinking_levels": ["off", "minimal", "low", "medium", "high"],
        "input": ["text", "image"],
    }
    assert "provider-secret" not in repr(batch)

    record = json.loads(request_log.read_text(encoding="utf-8"))
    assert record["request"] == {
        "id": "kai-model-discovery",
        "type": "get_available_models",
    }
    assert record["has_provider_key"] is True
    assert record["has_other_provider_key"] is False
    assert record["has_control_secret"] is False
    assert record["skip_version_check"] == "1"
    assert record["telemetry"] == "0"
    assert record["argv"][:6] == [
        "--mode",
        "rpc",
        "--provider",
        "openai",
        "--model",
        "openai/gpt-5.6-sol",
    ]
    assert record["argv"][6:8] == ["--thinking", "xhigh"]
    for flag in (
        "--no-session",
        "--no-approve",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-tools",
    ):
        assert flag in record["argv"]
    assert not any(item in record["request"] for item in ("message", "prompt", "session"))


def test_parser_preserves_provider_prefix_colons_and_supported_thinking_suffixes() -> None:
    candidates = discovery_module._parse_available_models(
        {
            "models": [
                _model("ollama", "llama4:70b"),
                _model(
                    "openai",
                    "gpt-5.6-sol",
                    reasoning=True,
                    thinking_level_map={"minimal": None, "xhigh": "xhigh", "max": "max"},
                ),
            ]
        },
        "ollama",
    )
    assert [candidate.model_id for candidate in candidates] == ["ollama/llama4:70b"]

    reasoning_candidates = discovery_module._parse_available_models(
        {
            "models": [
                _model(
                    "openai",
                    "gpt-5.6-sol",
                    reasoning=True,
                    thinking_level_map={"minimal": None, "xhigh": "xhigh", "max": "max"},
                )
            ]
        },
        "openai",
    )
    assert [candidate.model_id for candidate in reasoning_candidates] == [
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol:off",
        "openai/gpt-5.6-sol:low",
        "openai/gpt-5.6-sol:medium",
        "openai/gpt-5.6-sol:high",
        "openai/gpt-5.6-sol:xhigh",
        "openai/gpt-5.6-sol:max",
    ]


@pytest.mark.parametrize(
    ("error", "exception"),
    [
        ("Provider is not configured: openai", ModelDiscoveryAuthenticationError),
        ("Authentication token expired", ModelDiscoveryAuthenticationError),
        ("Unknown command: get_available_models", ModelDiscoveryUnsupported),
    ],
)
async def test_adapter_classifies_authentication_and_unsupported_failures(
    tmp_path: Path,
    error: str,
    exception: type[Exception],
) -> None:
    executable, _ = _fake_pi(tmp_path, error=error)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = PiModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(exception):
        await adapter.discover(_lane(executable, os_user=current_user))


async def test_no_models_for_requested_provider_is_authentication_failure(tmp_path: Path) -> None:
    executable, _ = _fake_pi(
        tmp_path,
        data={"models": [_model("anthropic", "claude-sonnet-4-6")]},
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = PiModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(ModelDiscoveryAuthenticationError):
        await adapter.discover(_lane(executable, os_user=current_user))


@pytest.mark.parametrize(
    "models",
    [
        [_model("openai", "gpt-5.6-sol"), _model("openai", "gpt-5.6-sol")],
        [{"provider": "openai", "id": "gpt\nsecret", "reasoning": False, "input": ["text"]}],
        [_model("openai", "gpt", reasoning=True, thinking_level_map={"turbo": "turbo"})],
    ],
)
def test_parser_rejects_malformed_or_duplicate_metadata(models: list[object]) -> None:
    with pytest.raises(ModelCatalogueValidationError):
        discovery_module._parse_available_models({"models": models}, "openai")


def test_launch_uses_canonical_user_home_and_subscription_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "pi"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    alice_home = tmp_path / "alice"
    alice_home.mkdir()
    monkeypatch.setattr(
        discovery_module.pwd,
        "getpwnam",
        lambda user: SimpleNamespace(pw_dir=str(alice_home)),
    )
    lane = _lane(
        executable,
        os_user="alice",
        provider="openai-codex",
        model="openai-codex/gpt-5.6-sol:xhigh",
        auth_mode=ModelDiscoveryAuthMode.SUBSCRIPTION,
        auth_configured=None,
    )

    launch = discovery_module._build_launch(
        lane,
        service_os_user="kai",
        environment={
            "OPENAI_API_KEY": "unrelated-key",
            "TELEGRAM_BOT_TOKEN": "control-plane-secret",
        },
    )

    assert launch.argv[:6] == ("sudo", "-H", "-D", str(alice_home), "-u", "alice")
    assert launch.cwd is None
    assert launch.target_user == "alice"
    assert "OPENAI_API_KEY" not in launch.environment
    assert "TELEGRAM_BOT_TOKEN" not in launch.environment
    assert launch.environment["TMPDIR"].endswith("/tmp/alice")
    assert "--no-session" in launch.argv
    preserve_arg = next(arg for arg in launch.argv if arg.startswith("--preserve-env="))
    assert "PI_SKIP_VERSION_CHECK" in preserve_arg
    assert "PI_TELEMETRY" in preserve_arg
    assert "OPENAI_API_KEY" not in preserve_arg
    assert "TELEGRAM_BOT_TOKEN" not in preserve_arg


def test_adapter_rejects_non_pi_unavailable_and_unconfigured_api_key_lanes(tmp_path: Path) -> None:
    executable = tmp_path / "pi"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    adapter = PiModelDiscoveryAdapter(service_os_user="kai", environment={})

    with pytest.raises(ModelDiscoveryUnsupported):
        adapter._validate_lane(_lane(executable, backend="opencode"))

    lane = _lane(executable)
    with pytest.raises(ModelDiscoveryUnsupported):
        adapter._validate_lane(
            replace(
                lane,
                executable=replace(
                    lane.executable,
                    resolved_path=None,
                    readiness=ModelDiscoveryReadiness.UNAVAILABLE,
                ),
            )
        )

    with pytest.raises(ModelDiscoveryAuthenticationError):
        adapter._validate_lane(
            _lane(
                executable,
                auth_mode=ModelDiscoveryAuthMode.API_KEY,
                auth_configured=False,
            )
        )


async def test_timeout_reaps_pi_metadata_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "pi-hung-fixture"
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
    monkeypatch.setattr(discovery_module, "_CLOSE_TIMEOUT_SECONDS", 0.2)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = PiModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await adapter.discover(_lane(executable, os_user=current_user))

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_cancellation_reaps_pi_metadata_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "pi-cancel-fixture"
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
    monkeypatch.setattr(discovery_module, "_CLOSE_TIMEOUT_SECONDS", 0.2)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = PiModelDiscoveryAdapter(service_os_user=current_user, environment={})
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
