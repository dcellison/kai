from __future__ import annotations

import json
import pwd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.config import CLAUDE_MODELS, PROVIDER_MODELS, models_for_backend, validate_model_for_backend
from kai.workshop import claude_model_discovery as discovery_module
from kai.workshop.claude_model_discovery import ClaudeModelDiscoveryAdapter
from kai.workshop.domain import PrincipalId, RuntimeProfileId
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
    auth_mode: ModelDiscoveryAuthMode = ModelDiscoveryAuthMode.API_KEY,
    auth_configured: bool | None = True,
) -> ModelDiscoveryBackendInventory:
    principal_id = _id(PrincipalId, value)
    profile_id = _id(RuntimeProfileId, value)
    effective_user = pwd.getpwuid(0).pw_name
    executable_identity = ModelDiscoveryExecutableIdentity(
        configured_path=str(executable),
        resolved_path=str(executable.resolve()),
        fingerprint=f"executable-{value}",
        readiness=ModelDiscoveryReadiness.READY,
        diagnostic=None,
    )
    auth = ModelDiscoveryAuthContext(
        auth_mode,
        "ANTHROPIC_API_KEY" if auth_mode == ModelDiscoveryAuthMode.API_KEY else "backend login",
        auth_configured,
        f"auth-{value}",
    )
    cache_inputs = ModelDiscoveryCacheInputs(
        version=1,
        principal_id=principal_id,
        runtime_profile_id=profile_id,
        backend="claude",
        provider="anthropic",
        os_user=effective_user,
        executable_fingerprint=executable_identity.fingerprint,
        auth_fingerprint=auth.fingerprint,
        default_model="sonnet",
        allowed_models=None,
        role_models=(),
    )
    return ModelDiscoveryBackendInventory(
        option_id="claude:anthropic",
        backend="claude",
        provider="anthropic",
        selected=True,
        status=ModelDiscoverySelectionStatus.SELECTED,
        readiness=ModelDiscoveryReadiness.READY,
        effective_os_user=effective_user,
        executable=executable_identity,
        auth=auth,
        default_model="sonnet",
        allowed_models=None,
        role_models=(),
        cache_inputs=cache_inputs,
        cache_key=f"cache-{value}",
        diagnostic=None,
    )


def _model(model_id: str, display_name: str) -> dict[str, object]:
    return {
        "id": model_id,
        "display_name": display_name,
        "type": "model",
        "created_at": "2026-08-28T00:00:00Z",
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        "capabilities": {
            "image_input": {"supported": True},
            "effort": {"high": {"supported": True}},
        },
    }


def _http_fixture(pages: list[dict[str, object]], *, status: int = 200):
    responses = []
    for page in pages:
        response = MagicMock()
        response.status = status
        response.content.read = AsyncMock(return_value=json.dumps(page).encode())
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=None)
        responses.append(context)
    session = MagicMock()
    session.get = MagicMock(side_effect=responses)
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=None)
    return session, session_context


async def test_api_key_discovery_uses_paginated_metadata_only_models_api(tmp_path: Path) -> None:
    executable = tmp_path / "claude"
    executable.write_text("fixture", encoding="utf-8")
    session, session_context = _http_fixture(
        [
            {
                "data": [_model("claude-opus-5", "Claude Opus 5")],
                "has_more": True,
                "last_id": "claude-opus-5",
            },
            {
                "data": [_model("claude-sonnet-5", "Claude Sonnet 5")],
                "has_more": False,
                "last_id": "claude-sonnet-5",
            },
        ]
    )
    adapter = ClaudeModelDiscoveryAdapter(
        environment={
            "ANTHROPIC_API_KEY": "provider-secret",
            "ANTHROPIC_BASE_URL": "https://gateway.example/api/",
            "TELEGRAM_BOT_TOKEN": "control-plane-secret",
        }
    )

    with patch.object(discovery_module.aiohttp, "ClientSession", return_value=session_context):
        batch = await adapter.discover(_lane(executable))

    assert batch.source == "anthropic-models-api-v1-models"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == ["claude-opus-5", "claude-sonnet-5"]
    assert batch.models[0].capabilities == {
        "created_at": "2026-08-28T00:00:00Z",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 128_000,
        "provider_capabilities": {
            "image_input": {"supported": True},
            "effort": {"high": {"supported": True}},
        },
    }
    first = session.get.call_args_list[0]
    second = session.get.call_args_list[1]
    assert first.args == ("https://gateway.example/api/v1/models?limit=1000",)
    assert second.args == ("https://gateway.example/api/v1/models?limit=1000&after_id=claude-opus-5",)
    assert first.kwargs["headers"] == {
        "anthropic-version": "2023-06-01",
        "x-api-key": "provider-secret",
    }
    assert first.kwargs["allow_redirects"] is False
    assert "control-plane-secret" not in repr(session.get.call_args_list)


async def test_subscription_context_uses_curated_fallback_without_http(tmp_path: Path) -> None:
    executable = tmp_path / "claude"
    executable.write_text("fixture", encoding="utf-8")
    adapter = ClaudeModelDiscoveryAdapter(environment={})

    with (
        patch.object(discovery_module.aiohttp, "ClientSession") as client_session,
        pytest.raises(ModelDiscoveryUnsupported),
    ):
        await adapter.discover(
            _lane(
                executable,
                auth_mode=ModelDiscoveryAuthMode.SUBSCRIPTION,
                auth_configured=None,
            )
        )

    client_session.assert_not_called()
    assert set(CLAUDE_MODELS) == {
        "best",
        "fable",
        "haiku",
        "opus",
        "opus[1m]",
        "opusplan",
        "sonnet",
        "sonnet[1m]",
    }


async def test_authentication_failure_is_secret_free(tmp_path: Path) -> None:
    executable = tmp_path / "claude"
    executable.write_text("fixture", encoding="utf-8")
    _session, session_context = _http_fixture([{}], status=401)
    secret = "anthropic-secret-value"
    adapter = ClaudeModelDiscoveryAdapter(environment={"ANTHROPIC_API_KEY": secret})

    with (
        patch.object(discovery_module.aiohttp, "ClientSession", return_value=session_context),
        pytest.raises(ModelDiscoveryAuthenticationError) as caught,
    ):
        await adapter.discover(_lane(executable))

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


def test_default_is_reset_semantics_not_a_stored_claude_model() -> None:
    assert "default" not in CLAUDE_MODELS
    assert validate_model_for_backend("default", "claude", "anthropic") is False
    assert validate_model_for_backend("best", "claude", "anthropic") is True
    assert validate_model_for_backend("opusplan", "claude", "anthropic") is True
    assert validate_model_for_backend("opus[1m]", "claude", "anthropic") is True
    assert models_for_backend("claude", "anthropic") is CLAUDE_MODELS
    assert "opusplan" not in PROVIDER_MODELS["anthropic"]
    assert validate_model_for_backend("opusplan", "goose", "anthropic") is False


def test_parser_rejects_malformed_metadata_and_base_urls() -> None:
    malformed = _model("claude-opus-5", "Claude Opus 5")
    malformed["capabilities"] = ["vision"]
    with pytest.raises(ModelCatalogueValidationError, match="capabilities"):
        discovery_module._parse_page(json.dumps({"data": [malformed], "has_more": False, "last_id": None}).encode())

    with pytest.raises(ModelCatalogueValidationError, match="base URL"):
        discovery_module._models_endpoint("https://secret@example.com?token=secret")


def test_api_key_lane_requires_present_key_without_exposing_environment(tmp_path: Path) -> None:
    executable = tmp_path / "claude"
    executable.write_text("fixture", encoding="utf-8")
    adapter = ClaudeModelDiscoveryAdapter(environment={"TELEGRAM_BOT_TOKEN": "secret"})

    with pytest.raises(ModelDiscoveryAuthenticationError):
        adapter._discovery_context(_lane(executable))

    assert "secret" not in repr(adapter)
