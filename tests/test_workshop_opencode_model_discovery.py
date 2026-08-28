from __future__ import annotations

import asyncio
import base64
import json
import os
import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.workshop import claude_model_discovery, codex_model_discovery
from kai.workshop import model_catalogue as catalogue_module
from kai.workshop import opencode_model_discovery as discovery_module
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
from kai.workshop.opencode_model_discovery import OpenCodeModelDiscoveryAdapter


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _lane(
    executable: Path,
    *,
    value: int = 1,
    os_user: str | None = None,
    provider: str = "openrouter",
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
        backend="opencode",
        provider=provider,
        os_user=effective_user,
        executable_fingerprint=executable_identity.fingerprint,
        auth_fingerprint=auth.fingerprint,
        default_model=f"{provider}/default",
        allowed_models=None,
        role_models=(),
    )
    return ModelDiscoveryBackendInventory(
        option_id=f"opencode:{provider}",
        backend="opencode",
        provider=provider,
        selected=True,
        status=ModelDiscoverySelectionStatus.SELECTED,
        readiness=ModelDiscoveryReadiness.UNVERIFIED,
        effective_os_user=effective_user,
        executable=executable_identity,
        auth=auth,
        default_model=f"{provider}/default",
        allowed_models=None,
        role_models=(),
        cache_inputs=cache_inputs,
        cache_key=f"cache-{value}",
        diagnostic=None,
    )


def _fake_command(
    tmp_path: Path,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> tuple[Path, Path]:
    executable = tmp_path / "opencode-fixture"
    invocation_log = tmp_path / "invocation.json"
    stdout_value = base64.b64encode(stdout).decode("ascii")
    stderr_value = base64.b64encode(stderr).decode("ascii")
    script = f"""#!{sys.executable}
import base64
import json
import os
import sys
from pathlib import Path

Path({str(invocation_log)!r}).write_text(json.dumps({{
    "argv": sys.argv[1:],
    "environment": {{
        key: os.environ.get(key)
        for key in (
            "NO_COLOR",
            "OPENROUTER_API_KEY",
            "OPENCODE_CONFIG_CONTENT",
            "TELEGRAM_BOT_TOKEN",
        )
    }},
}}), encoding="utf-8")
sys.stdout.buffer.write(base64.b64decode({stdout_value!r}))
sys.stdout.buffer.flush()
sys.stderr.buffer.write(base64.b64decode({stderr_value!r}))
sys.stderr.buffer.flush()
raise SystemExit({returncode})
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable, invocation_log


def test_registered_adapter_sources_are_valid_canonical_catalogue_ids() -> None:
    for source in (
        claude_model_discovery._SOURCE,
        codex_model_discovery._SOURCE,
        discovery_module._SOURCE,
    ):
        normalized = catalogue_module._normalize_batch(
            catalogue_module.ModelDiscoveryBatch(source, ()),
        )
        assert normalized.source == source


async def test_adapter_uses_documented_metadata_command_and_preserves_nested_ids(
    tmp_path: Path,
) -> None:
    executable, invocation_log = _fake_command(
        tmp_path,
        stdout=(b"openrouter/anthropic/claude-sonnet-4-6\nopenrouter/meta/llama/3\n"),
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})

    batch = await adapter.discover(_lane(executable, os_user=current_user))

    assert batch.source == "opencode-cli-models"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == [
        "openrouter/anthropic/claude-sonnet-4-6",
        "openrouter/meta/llama/3",
    ]
    assert [candidate.display_label for candidate in batch.models] == [
        "openrouter/anthropic/claude-sonnet-4-6",
        "openrouter/meta/llama/3",
    ]
    assert all(candidate.capabilities == {"provider_id": "openrouter"} for candidate in batch.models)
    invocation = json.loads(invocation_log.read_text(encoding="utf-8"))
    assert invocation["argv"] == ["models", "openrouter", "--refresh"]
    assert not {"run", "session", "serve"} & set(invocation["argv"])
    assert invocation["environment"]["NO_COLOR"] == "1"


def test_parser_accepts_empty_inventory_and_nested_model_ids() -> None:
    assert discovery_module._parse_models_output(b"\n", "deepseek") == ()

    models = discovery_module._parse_models_output(
        b"deepseek/vendor/family/model\n",
        "deepseek",
    )

    assert [model.model_id for model in models] == [
        "deepseek/vendor/family/model",
    ]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"\xff", "not UTF-8"),
        (b"openrouter-model\n", "invalid provider/model"),
        (b"deepseek/model\n", "invalid provider/model"),
        (b"openrouter/model\nopenrouter/model\n", "duplicate"),
    ],
)
def test_parser_rejects_malformed_mismatched_and_duplicate_output(
    body: bytes,
    match: str,
) -> None:
    with pytest.raises(ModelCatalogueValidationError, match=match):
        discovery_module._parse_models_output(body, "openrouter")


def test_parser_enforces_model_identifier_and_catalogue_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = b"openrouter/" + (b"x" * 512)
    with pytest.raises(ModelCatalogueValidationError, match="oversized"):
        discovery_module._parse_models_output(oversized, "openrouter")

    monkeypatch.setattr(discovery_module, "_MAX_MODELS", 1)
    with pytest.raises(ModelCatalogueValidationError, match="safety limit"):
        discovery_module._parse_models_output(
            b"openrouter/one\nopenrouter/two\n",
            "openrouter",
        )


@pytest.mark.parametrize(
    ("detail", "error_type"),
    [
        (b"Provider not found: openrouter", ModelDiscoveryAuthenticationError),
        (b"Unknown command: models", ModelDiscoveryUnsupported),
        (b"unexpected backend failure", discovery_module._OpenCodeDiscoveryError),
    ],
)
async def test_command_failures_are_classified_without_leaking_output(
    tmp_path: Path,
    detail: bytes,
    error_type: type[Exception],
) -> None:
    executable, _invocation_log = _fake_command(
        tmp_path,
        stderr=detail,
        returncode=1,
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(error_type) as caught:
        await adapter.discover(_lane(executable, os_user=current_user))

    assert detail.decode() not in str(caught.value)
    assert detail.decode() not in repr(caught.value)


def test_launch_uses_canonical_user_auth_context_and_scrubs_control_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "opencode"
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
        "OPENCODE_CONFIG_CONTENT": '{"model":"wrong/model"}',
        "TELEGRAM_BOT_TOKEN": "control-plane-secret",
    }

    launch = discovery_module._build_launch(
        _lane(executable, os_user="alice"),
        service_os_user="kai",
        environment=environment,
    )

    assert launch.argv[:6] == (
        "sudo",
        "-H",
        "-D",
        str(alice_home),
        "-u",
        "alice",
    )
    assert launch.argv[-4:] == (
        str(executable),
        "models",
        "openrouter",
        "--refresh",
    )
    assert launch.cwd is None
    assert launch.target_user == "alice"
    assert launch.environment["OPENROUTER_API_KEY"] == "provider-key"
    assert launch.environment["NO_COLOR"] == "1"
    assert "OPENCODE_CONFIG_CONTENT" not in launch.environment
    assert "TELEGRAM_BOT_TOKEN" not in launch.environment
    assert launch.environment["TMPDIR"].endswith("/tmp/alice")
    preserve_arg = next(arg for arg in launch.argv if arg.startswith("--preserve-env="))
    assert "OPENROUTER_API_KEY" in preserve_arg
    assert "NO_COLOR" in preserve_arg
    assert "TELEGRAM_BOT_TOKEN" not in preserve_arg


async def test_output_limit_is_enforced_while_streams_are_drained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, _invocation_log = _fake_command(
        tmp_path,
        stdout=b"openrouter/model-that-is-too-long\n",
    )
    monkeypatch.setattr(discovery_module, "_MAX_OUTPUT_BYTES", 8)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(ModelCatalogueValidationError, match="output exceeds"):
        await adapter.discover(_lane(executable, os_user=current_user))


async def test_cancellation_reaps_metadata_command(tmp_path: Path) -> None:
    executable = tmp_path / "opencode-cancel-fixture"
    pid_file = tmp_path / "pid"
    script = f"""#!{sys.executable}
import os
import time
from pathlib import Path

Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(60)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})
    task = asyncio.create_task(
        adapter.discover(_lane(executable, os_user=current_user)),
    )
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
