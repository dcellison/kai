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


def _model(model_id: str, *, name: str = "Claude Sonnet") -> dict[str, object]:
    return {
        "id": model_id,
        "name": name,
        "release_date": "2026-08-01",
        "attachment": True,
        "reasoning": True,
        "temperature": True,
        "tool_call": True,
        "cost": {"input": 1, "output": 2},
        "limit": {"context": 200_000, "output": 64_000},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "status": "active",
        "experimental": False,
        "options": {"apiKey": "provider-secret"},
        "headers": {"authorization": "Bearer provider-secret"},
    }


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


def _fake_server(
    tmp_path: Path,
    payload: object,
) -> tuple[Path, Path]:
    executable = tmp_path / "opencode-fixture"
    request_log = tmp_path / "requests.jsonl"
    encoded_payload = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    script = f"""#!{sys.executable}
import argparse
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--hostname", required=True)
parser.add_argument("--port", required=True, type=int)
args = parser.parse_args()
payload = base64.b64decode({encoded_payload!r})
request_log = {str(request_log)!r}
expected = "Basic " + base64.b64encode(
    (os.environ["OPENCODE_SERVER_USERNAME"] + ":" + os.environ["OPENCODE_SERVER_PASSWORD"]).encode()
).decode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(request_log, "a", encoding="utf-8") as output:
            output.write(json.dumps({{
                "path": self.path,
                "authenticated": self.headers.get("Authorization") == expected,
            }}) + "\\n")
        if self.headers.get("Authorization") != expected:
            self.send_response(401)
            self.end_headers()
            return
        if self.path == "/global/health":
            body = b'{{"healthy":true,"version":"fixture"}}'
        elif self.path.startswith("/provider?"):
            body = payload
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

HTTPServer((args.hostname, args.port), Handler).serve_forever()
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable, request_log


async def test_adapter_reads_connected_provider_without_generation_and_preserves_nested_ids(
    tmp_path: Path,
) -> None:
    model_id = "anthropic/claude-sonnet-4-6"
    payload = {
        "all": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "models": {model_id: _model(model_id)},
                "options": {"apiKey": "provider-secret"},
            }
        ],
        "default": {"openrouter": model_id},
        "connected": ["openrouter"],
    }
    executable, request_log = _fake_server(tmp_path, payload)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})

    batch = await adapter.discover(_lane(executable, os_user=current_user))

    assert batch.source == "opencode-server-provider"
    assert batch.ttl_seconds == 21_600
    assert [candidate.model_id for candidate in batch.models] == ["openrouter/anthropic/claude-sonnet-4-6"]
    assert batch.models[0].display_label == "Claude Sonnet"
    assert batch.models[0].capabilities == {
        "attachment": True,
        "experimental": False,
        "limits": {"context": 200_000, "output": 64_000},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "provider_name": "OpenRouter",
        "reasoning": True,
        "release_date": "2026-08-01",
        "status": "active",
        "temperature": True,
        "tool_call": True,
    }
    assert "provider-secret" not in repr(batch)
    requests = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines()]
    assert [request["path"].split("?", 1)[0] for request in requests] == [
        "/global/health",
        "/provider",
    ]
    assert all(request["authenticated"] is True for request in requests)
    assert "directory=" in requests[1]["path"]
    assert not any("session" in request["path"] for request in requests)


def test_parser_distinguishes_unavailable_auth_from_unsupported_provider() -> None:
    model = _model("model")
    payload = json.dumps(
        {
            "all": [{"id": "openrouter", "name": "OpenRouter", "models": {"model": model}}],
            "connected": [],
        }
    ).encode()
    with pytest.raises(ModelDiscoveryAuthenticationError):
        discovery_module._parse_provider_catalogue(payload, "openrouter")
    with pytest.raises(ModelDiscoveryUnsupported):
        discovery_module._parse_provider_catalogue(payload, "deepseek")


def test_parser_accepts_empty_inventory_and_rejects_malformed_or_mismatched_models() -> None:
    empty = json.dumps(
        {
            "all": [{"id": "openrouter", "name": "OpenRouter", "models": {}}],
            "connected": ["openrouter"],
        }
    ).encode()
    assert discovery_module._parse_provider_catalogue(empty, "openrouter") == ()

    with pytest.raises(ModelCatalogueValidationError, match="malformed JSON"):
        discovery_module._parse_provider_catalogue(b"not json", "openrouter")

    mismatched = json.dumps(
        {
            "all": [
                {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "models": {"catalogue-key": _model("different-id")},
                }
            ],
            "connected": ["openrouter"],
        }
    ).encode()
    with pytest.raises(ModelCatalogueValidationError, match="do not match"):
        discovery_module._parse_provider_catalogue(mismatched, "openrouter")


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
        port=43123,
        service_os_user="kai",
        environment=environment,
    )

    assert launch.argv[:6] == ("sudo", "-H", "-D", str(alice_home), "-u", "alice")
    assert launch.argv[-6:] == (
        str(executable),
        "serve",
        "--hostname",
        "127.0.0.1",
        "--port",
        "43123",
    )
    assert launch.cwd is None
    assert launch.directory == alice_home
    assert launch.environment["OPENROUTER_API_KEY"] == "provider-key"
    assert "OPENCODE_CONFIG_CONTENT" not in launch.environment
    assert "TELEGRAM_BOT_TOKEN" not in launch.environment
    assert launch.environment["OPENCODE_SERVER_USERNAME"] == "kai-model-discovery"
    assert launch.environment["OPENCODE_SERVER_PASSWORD"] == launch.password
    assert launch.environment["TMPDIR"].endswith("/tmp/alice")
    preserve_arg = next(arg for arg in launch.argv if arg.startswith("--preserve-env="))
    assert "OPENCODE_SERVER_PASSWORD" in preserve_arg
    assert "TELEGRAM_BOT_TOKEN" not in preserve_arg


async def test_startup_timeout_reaps_metadata_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "opencode-hung-fixture"
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
    monkeypatch.setattr(discovery_module, "_STARTUP_TIMEOUT_SECONDS", 0.5)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})

    with pytest.raises(discovery_module._OpenCodeDiscoveryError, match="did not become ready"):
        await adapter.discover(_lane(executable, os_user=current_user))

    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_cancellation_reaps_metadata_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(discovery_module, "_STARTUP_TIMEOUT_SECONDS", 60.0)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    adapter = OpenCodeModelDiscoveryAdapter(service_os_user=current_user, environment={})
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
