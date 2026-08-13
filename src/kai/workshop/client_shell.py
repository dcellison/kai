"""Static, read-only browser shell for the authenticated Workshop client API."""

from pathlib import Path

from aiohttp import web

_SHELL_PATH = "/workshop"
_SHELL_INDEX_PATH = "/workshop/"
_SHELL_CSS_PATH = "/workshop/app.css"
_SHELL_JS_PATH = "/workshop/app.js"
_STATIC_ROOT = Path(__file__).with_name("static")

_DOCUMENT_CSP = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "img-src 'none'",
        "font-src 'none'",
    )
)


def _apply_shell_security_headers(response: web.StreamResponse, *, document: bool = False) -> None:
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Content-Security-Policy"] = _DOCUMENT_CSP if document else "default-src 'none'"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"


def _asset_response(body: bytes, content_type: str, *, document: bool = False) -> web.Response:
    response = web.Response(body=body, content_type=content_type, charset="utf-8")
    _apply_shell_security_headers(response, document=document)
    return response


def register_workshop_shell_routes(app: web.Application) -> None:
    """Register a static shell; all collaboration data still comes from authenticated APIs."""
    index = (_STATIC_ROOT / "index.html").read_bytes()
    stylesheet = (_STATIC_ROOT / "app.css").read_bytes()
    script = (_STATIC_ROOT / "app.js").read_bytes()

    async def handle_index(request: web.Request) -> web.Response:
        return _asset_response(index, "text/html", document=True)

    async def handle_stylesheet(request: web.Request) -> web.Response:
        return _asset_response(stylesheet, "text/css")

    async def handle_script(request: web.Request) -> web.Response:
        return _asset_response(script, "application/javascript")

    app.router.add_get(_SHELL_PATH, handle_index)
    app.router.add_get(_SHELL_INDEX_PATH, handle_index)
    app.router.add_get(_SHELL_CSS_PATH, handle_stylesheet)
    app.router.add_get(_SHELL_JS_PATH, handle_script)
