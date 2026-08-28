from __future__ import annotations

import json
import os
import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.workshop import codex_model_discovery as discovery_module
from kai.workshop.codex_model_discovery import CodexModelDiscoveryAdapter
from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.model_catalogue import (
    ModelCatalogueValidationError,
    ModelDiscoveryAuthenticationError,
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
    auth_mode: ModelDiscoveryAuthMode = ModelDiscoveryAuthMode.SUBSCRIPTION,
    auth_configured: bool | None = None,
) -> ModelDiscoveryBackendInventory:
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
        "backend login" if auth_mode == ModelDiscoveryAuthMode.SUBSCRIPTION else "OPENAI_API_KEY",
        auth_configured,
        f"auth-{value}",
    )
    cache_inputs = ModelDiscoveryCacheInputs(
        version=1,
        principal_id=principal_id,
        runtime_profile_id=profile_id,
        backend="codex",
        provider="openai",
        os_user=effective_user,
        executable_fingerprint=executable_identity.fingerprint,
        auth_fingerprint=auth.fingerprint,
        default_model="gpt-default",
        allowed_models=None,
        role_models=(),
    )
    return ModelDiscoveryBackendInventory(
        option_id="codex:openai",
        backend="codex",
        provider="openai",
        selected=True,
        status=ModelDiscoverySelectionStatus.SELECTED,
        readiness=ModelDiscoveryReadiness.UNVERIFIED,
        effective_os_user=effective_user,
        executable=executable_identity,
        auth=auth,
        default_model="gpt-default",
        allowed_models=None,
        role_models=(),
        cache_inputs=cache_inputs,
        cache_key=f"cache-{value}",
        diagnostic=None,
    )


def _model(model: str, *, label: str | None = None) -> dict[str, object]:
    return {
        "id": f"picker-{model}",
        "model": model,
        "upgrade": None,
        "displayName": label or model,
        "description": f"Description for {model}",
        "modelSpecialty": None,
        "hidden": False,
        "supportedReasoningEfforts": [
            {"reasoningEffort": "high", "description": "High"},
            {"reasoningEffort": "low", "description": "Low"},
        ],
        "defaultReasoningEffort": "high",
        "inputModalities": ["text", "image"],
        "supportsPersonality": True,
        "multiAgentVersion": "v2",
        "additionalSpeedTiers": ["fast"],
        "serviceTiers": [{"id": "priority", "name": "Priority", "description": "Fast queue"}],
        "defaultServiceTier": "priority",
        "isDefault": model == "gpt-alpha",
    }


def _fake_app_server(
    tmp_path: Path,
    pages: list[dict[str, object]],
    *,
    error_message: str | None = None,
) -> tuple[Path, Path]:
    executable = tmp_path / "codex-fixture"
    request_log = tmp_path / "requests.jsonl"
    script = f"""#!{sys.executable}
import json
import sys

pages = {pages!r}
error_message = {error_message!r}
request_log = {str(request_log)!r}
page = 0
for line in sys.stdin:
    request = json.loads(line)
    with open(request_log, "a", encoding="utf-8") as output:
        output.write(json.dumps(request, separators=(",", ":")) + "\\n")
    if "id" not in request:
        continue
    if request["method"] == "initialize":
        response = {{"id": request["id"], "result": {{"userAgent": "fixture"}}}}
    elif error_message is not None:
        response = {{
            "id": request["id"],
            "error": {{"code": 401, "message": error_message}},
        }}
    else:
        response = {{"id": request["id"], "result": pages[page]}}
        page += 1
    print(json.dumps(response, separators=(",", ":")), flush=True)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable, request_log


async def test_adapter_uses_paginated_metadata_only_model_list(tmp_path: Path) -> None:
    executable, request_log = _fake_app_server(
        tmp_path,
        [
            {"data": [_model("gpt-alpha", label="GPT Alpha")], "nextCursor": "page-2"},
            {"data": [_model("gpt-beta", label="GPT Beta")], "nextCursor": None},
        ],
    )
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = CodexModelDiscoveryAdapter(service_os_user=current_user, environment={})

    batch = await adapter.discover(_lane(executable, os_user=current_user))

    assert batch.source == "codex-app-server:model/list"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == ["gpt-alpha", "gpt-beta"]
    assert batch.models[0].display_label == "GPT Alpha"
    assert batch.models[0].capabilities == {
        "additional_speed_tiers": ["fast"],
        "codex_picker_id": "picker-gpt-alpha",
        "default_reasoning_effort": "high",
        "default_service_tier": "priority",
        "description": "Description for gpt-alpha",
        "hidden": False,
        "input_modalities": ["text", "image"],
        "is_default": True,
        "model_specialty": None,
        "multi_agent_version": "v2",
        "service_tiers": [{"id": "priority", "name": "Priority", "description": "Fast queue"}],
        "supported_reasoning_efforts": [
            {"effort": "high", "description": "High"},
            {"effort": "low", "description": "Low"},
        ],
        "supports_personality": True,
        "upgrade": None,
    }
    requests = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "model/list",
        "model/list",
    ]
    assert requests[2]["params"] == {
        "cursor": None,
        "limit": 100,
        "includeHidden": False,
    }
    assert requests[3]["params"]["cursor"] == "page-2"
    assert not {"thread/start", "turn/start"} & {request["method"] for request in requests}


async def test_adapter_classifies_authentication_without_leaking_rpc_message(tmp_path: Path) -> None:
    secret = "expired token secret-account-value"
    executable, _request_log = _fake_app_server(tmp_path, [], error_message=secret)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = CodexModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(ModelDiscoveryAuthenticationError) as caught:
        await adapter.discover(_lane(executable, os_user=current_user))

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


def test_parser_rejects_malformed_and_unrequested_hidden_entries() -> None:
    malformed = _model("gpt-alpha")
    malformed["supportedReasoningEfforts"] = "high"
    with pytest.raises(ModelCatalogueValidationError, match="reasoning efforts"):
        discovery_module._parse_model_list_page({"data": [malformed], "nextCursor": None})

    hidden = _model("gpt-hidden")
    hidden["hidden"] = True
    with pytest.raises(ModelCatalogueValidationError, match="hidden model"):
        discovery_module._parse_model_list_page({"data": [hidden], "nextCursor": None})


def test_launch_uses_canonical_os_user_auth_context_and_scrubs_control_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "codex"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    alice_home = tmp_path / "alice"
    bob_home = tmp_path / "bob"
    alice_home.mkdir()
    bob_home.mkdir()
    homes = {"alice": alice_home, "bob": bob_home}
    monkeypatch.setattr(
        discovery_module.pwd,
        "getpwnam",
        lambda user: SimpleNamespace(pw_dir=str(homes[user])),
    )
    environment = {
        "OPENAI_API_KEY": "provider-key",
        "TELEGRAM_BOT_TOKEN": "control-plane-secret",
    }

    alice = discovery_module._build_launch(
        _lane(executable, value=1, os_user="alice"),
        service_os_user="kai",
        environment=environment,
    )
    bob = discovery_module._build_launch(
        _lane(executable, value=2, os_user="bob"),
        service_os_user="kai",
        environment=environment,
    )

    assert alice.argv[:6] == ("sudo", "-H", "-D", str(alice_home), "-u", "alice")
    assert bob.argv[:6] == ("sudo", "-H", "-D", str(bob_home), "-u", "bob")
    assert alice.argv[-2:] == (str(executable), "app-server")
    assert alice.cwd is None
    assert alice.cross_user is True
    assert alice.environment["OPENAI_API_KEY"] == "provider-key"
    assert "TELEGRAM_BOT_TOKEN" not in alice.environment
    assert alice.environment["TMPDIR"].endswith("/tmp/alice")
    assert bob.environment["TMPDIR"].endswith("/tmp/bob")
    assert alice.argv != bob.argv


def test_api_key_lane_without_key_fails_before_starting_process(tmp_path: Path) -> None:
    executable = tmp_path / "codex"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o700)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = CodexModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(ModelDiscoveryAuthenticationError):
        adapter._validate_lane(
            _lane(
                executable,
                os_user=current_user,
                auth_mode=ModelDiscoveryAuthMode.API_KEY,
                auth_configured=False,
            )
        )
