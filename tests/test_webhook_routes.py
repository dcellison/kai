"""Route-registration tests for separated external and internal credentials."""

import json

from aiohttp import web

from kai.config import Config
from kai.webhook import _handle_health, _register_routes


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "telegram_bot_token": "test-token",
        "allowed_user_ids": {123},
    }
    values.update(overrides)
    return Config(**values)


def _routes(app: web.Application) -> set[tuple[str, str]]:
    return {(route.method, route.resource.canonical) for route in app.router.routes()}


def test_internal_api_routes_do_not_depend_on_external_secrets() -> None:
    app = web.Application()

    _register_routes(app, _config())

    routes = _routes(app)
    assert ("POST", "/api/schedule") in routes
    assert ("GET", "/api/jobs") in routes
    assert ("GET", "/api/memory/stats") in routes
    assert ("POST", "/webhook/github") not in routes
    assert ("POST", "/webhook") not in routes


def test_external_webhook_routes_are_enabled_independently() -> None:
    github_app = web.Application()
    generic_app = web.Application()

    _register_routes(github_app, _config(github_webhook_secret="github-secret"))
    _register_routes(generic_app, _config(generic_webhook_secret="generic-secret"))

    github_routes = _routes(github_app)
    generic_routes = _routes(generic_app)
    assert ("POST", "/webhook/github") in github_routes
    assert ("POST", "/webhook") not in github_routes
    assert ("POST", "/webhook") in generic_routes
    assert ("POST", "/webhook/github") not in generic_routes


def test_legacy_credential_temporarily_enables_both_external_routes() -> None:
    app = web.Application()

    _register_routes(app, _config(webhook_secret="legacy-secret"))

    routes = _routes(app)
    assert ("POST", "/webhook/github") in routes
    assert ("POST", "/webhook") in routes
    assert ("POST", "/api/schedule") in routes


def test_telegram_route_is_independent_of_other_webhooks() -> None:
    app = web.Application()

    _register_routes(
        app,
        _config(
            telegram_webhook_url="https://example.com/webhook/telegram",
            telegram_webhook_secret="telegram-secret",
        ),
    )

    routes = _routes(app)
    assert ("POST", "/webhook/telegram") in routes
    assert ("POST", "/webhook/github") not in routes
    assert ("POST", "/webhook") not in routes


async def test_health_reports_non_sensitive_memory_mode(monkeypatch) -> None:
    monkeypatch.setattr("kai.webhook.memory.is_enabled", lambda: True)

    response = await _handle_health(None)  # type: ignore[arg-type]

    assert json.loads(response.body) == {"status": "ok", "memory_enabled": True}
