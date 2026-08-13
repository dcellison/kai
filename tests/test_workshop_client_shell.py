"""Serving contracts for the packaged, read-only Workshop React client."""

import re

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.client_shell import register_workshop_shell_routes


async def _client() -> TestClient:
    app = web.Application()
    register_workshop_shell_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


class TestWorkshopClientShell:
    async def test_serves_packaged_shell_with_strict_security_headers(self):
        client = await _client()
        try:
            response = await client.get("/workshop/")
            body = await response.text()

            assert response.status == 200
            assert response.content_type == "text/html"
            assert response.headers["Cache-Control"] == "private, no-store"
            assert "default-src 'none'" in response.headers["Content-Security-Policy"]
            assert "script-src 'self'" in response.headers["Content-Security-Policy"]
            assert "style-src 'self'" in response.headers["Content-Security-Policy"]
            assert "connect-src 'self'" in response.headers["Content-Security-Policy"]
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
            assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert "Kai Workshop" in body
            assert '<div id="root"></div>' in body
            assert '<script type="module" crossorigin src="/workshop/app.js"></script>' in body
            assert 'href="/workshop/app.css"' in body
            assert "<script>" not in body
            assert "style=" not in body
        finally:
            await client.close()

    async def test_root_alias_serves_same_shell(self):
        client = await _client()
        try:
            root = await client.get("/workshop")
            slash = await client.get("/workshop/")

            assert root.status == slash.status == 200
            assert await root.read() == await slash.read()
        finally:
            await client.close()

    async def test_assets_are_same_origin_no_store_resources(self):
        client = await _client()
        try:
            stylesheet = await client.get("/workshop/app.css")
            script = await client.get("/workshop/app.js")
            stylesheet_body = await stylesheet.text()
            script_body = await script.text()

            assert stylesheet.status == script.status == 200
            assert stylesheet.content_type == "text/css"
            assert script.content_type == "application/javascript"
            for response in (stylesheet, script):
                assert response.headers["Cache-Control"] == "private, no-store"
                assert response.headers["Content-Security-Policy"] == "default-src 'none'"
                assert response.headers["X-Content-Type-Options"] == "nosniff"

            assert "sessionStorage" in script_body
            assert "localStorage" not in script_body
            assert "Authorization" in script_body
            assert "Last-Event-ID" in script_body
            assert "EventSource(" not in script_body
            assert "/v1/client/enrollment/redeem" in script_body
            assert "/timeline" in script_body
            assert "/events" in script_body

            # Every ancestor in the nested grid must be shrinkable; otherwise
            # a long timeline expands the implicit row and is clipped by the
            # viewport shell instead of scrolling within the conversation.
            assert "grid-template-rows:minmax(0,1fr)" in stylesheet_body
            conversation_rule = re.search(r"\.conversation-pane\{([^}]*)\}", stylesheet_body)
            timeline_rule = re.search(r"\.timeline-wrap\{([^}]*)\}", stylesheet_body)
            assert conversation_rule is not None
            assert "min-height:0" in conversation_rule.group(1)
            assert "overflow:hidden" in conversation_rule.group(1)
            assert timeline_rule is not None
            assert "min-height:0" in timeline_rule.group(1)
            assert "overflow-y:auto" in timeline_rule.group(1)
        finally:
            await client.close()

    async def test_shell_registers_no_write_method(self):
        client = await _client()
        try:
            for path in ("/workshop", "/workshop/", "/workshop/app.css", "/workshop/app.js"):
                response = await client.post(path, data=b"must not be accepted")
                assert response.status == 405
        finally:
            await client.close()
