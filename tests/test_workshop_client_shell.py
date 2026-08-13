"""Contracts for the static, read-only Workshop browser shell."""

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
            assert "Read-only preview" in body
            assert '<form id="enrollment-form" method="post" action="/workshop/"' in body
            assert '<script src="/workshop/app.js" defer></script>' in body
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
            assert 'headers.set("Authorization"' in script_body
            assert 'headers.set("Last-Event-ID"' in script_body
            assert ".textContent =" in script_body
            assert ".innerHTML" not in script_body
            assert "EventSource(" not in script_body
            assert 'method: "POST"' in script_body
            assert "/v1/client/enrollment/redeem" in script_body
            assert "/timeline" in script_body
            assert "/events" in script_body
            assert "correctChannel" in script_body
            assert "forget-enrollment-session" in script_body
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
