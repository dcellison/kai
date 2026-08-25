"""Packaging and import-boundary coverage for the optional Telegram adapter."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_telegram_bot_is_only_in_telegram_extra() -> None:
    """The base distribution must not install the Telegram SDK."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    base_dependencies = project["project"]["dependencies"]
    telegram_dependencies = project["project"]["optional-dependencies"]["telegram"]

    assert not any(dependency.startswith("python-telegram-bot") for dependency in base_dependencies)
    assert telegram_dependencies == ["python-telegram-bot>=20.0,<23"]


def test_core_and_workshop_execute_with_every_telegram_import_rejected() -> None:
    """Exercise core surfaces under a process-wide Telegram import tripwire."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class RejectTelegram(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "telegram" or fullname.startswith("telegram."):
                    raise AssertionError(f"core imported optional Telegram module: {fullname}")
                return None

        sys.meta_path.insert(0, RejectTelegram())

        from aiohttp import web
        from kai.application_host import KaiApplicationHost
        from kai.config import Config
        from kai.http_adapter import HttpAdapter
        from kai.internal_api_auth import InternalAPIScope
        from kai import memory, webhook
        import kai.main as main
        from kai.workshop.integration_notifications import DEFAULT_INTEGRATION_ROUTE
        from kai.workshop.scheduler import WorkshopSchedulerState

        config = Config(
            telegram_bot_token=None,
            allowed_user_ids=set(),
            enabled_adapters=frozenset({"workshop"}),
        )
        app = web.Application()
        webhook._register_routes(app, config)
        routes = {route.resource.canonical for route in app.router.routes()}

        assert config.workshop_enabled
        assert not config.telegram_enabled
        assert main._delivery_policy(config, None).enabled_transports == frozenset()
        assert "/health" in routes
        assert "/api/jobs" in routes
        assert "/webhook/telegram" not in routes
        assert memory.is_enabled() is False
        assert DEFAULT_INTEGRATION_ROUTE == "default"
        assert WorkshopSchedulerState.NEW == "new"
        assert InternalAPIScope.JOBS_READ == "jobs:read"
        assert KaiApplicationHost is not None
        assert HttpAdapter is not None

        started = []
        main.load_config = lambda: config
        main._read_protected_file = lambda _path: ""
        main.services.load_services_from_string = lambda _text: {}

        def record_start(coroutine):
            started.append(True)
            coroutine.close()

        main.asyncio.run = record_start
        main._start()
        assert started == [True]
        assert not any(name == "telegram" or name.startswith("telegram.") for name in sys.modules)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_enabled_telegram_without_extra_has_operator_diagnostic(monkeypatch) -> None:
    """An enabled adapter fails before startup with a precise remedy."""
    import kai.main

    missing = ModuleNotFoundError("No module named 'telegram'", name="telegram")

    def missing_telegram(name: str):
        assert name == "kai.telegram_adapter"
        raise missing

    monkeypatch.setattr(kai.main, "import_module", missing_telegram)

    with pytest.raises(SystemExit) as exc_info:
        kai.main._load_telegram_adapter_module()

    assert str(exc_info.value) == (
        "The Telegram adapter is enabled, but its optional dependency is not installed. "
        "Install Kai with the 'telegram' extra or disable Telegram in KAI_ENABLED_ADAPTERS."
    )
