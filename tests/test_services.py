"""Tests for the declarative HTTP service layer (services.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kai.services as services
from kai.services import (
    call_service,
    get_available_services,
    load_services,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_services():
    """Reset the module-level service registry between tests."""
    services._services = None
    yield
    services._services = None


def _write_yaml(tmp_path, content: str):
    """Helper to write a YAML config file and return its path."""
    p = tmp_path / "services.yaml"
    p.write_text(content)
    return p


# ── TestLoadServices ────────────────────────────────────────────────


class TestLoadServices:
    def test_missing_file_returns_empty(self, tmp_path):
        """Missing config file is not an error — graceful degradation."""
        result = load_services(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_valid_bearer_auth(self, tmp_path, monkeypatch):
        """Bearer auth service loads when env var is set."""
        monkeypatch.setenv("TEST_KEY", "secret123")
        path = _write_yaml(
            tmp_path,
            """
services:
  testapi:
    url: https://api.example.com/v1
    method: POST
    auth:
      type: bearer
      env: TEST_KEY
    description: Test API
""",
        )
        result = load_services(path)
        assert "testapi" in result
        assert result["testapi"].url == "https://api.example.com/v1"
        assert result["testapi"].method == "POST"
        assert result["testapi"].auth.type == "bearer"
        assert result["testapi"].auth.env == "TEST_KEY"
        assert result["testapi"].description == "Test API"

    def test_valid_header_auth(self, tmp_path, monkeypatch):
        """Header auth service loads with custom header name."""
        monkeypatch.setenv("BRAVE_KEY", "bravesecret")
        path = _write_yaml(
            tmp_path,
            """
services:
  brave:
    url: https://api.search.brave.com/res/v1/web/search
    method: GET
    auth:
      type: header
      name: X-Subscription-Token
      env: BRAVE_KEY
    description: Brave Search
""",
        )
        result = load_services(path)
        assert "brave" in result
        assert result["brave"].auth.type == "header"
        assert result["brave"].auth.name == "X-Subscription-Token"

    def test_valid_query_auth(self, tmp_path, monkeypatch):
        """Query param auth service loads correctly."""
        monkeypatch.setenv("GOOGLE_KEY", "googlekey")
        path = _write_yaml(
            tmp_path,
            """
services:
  google:
    url: https://customsearch.googleapis.com/customsearch/v1
    method: GET
    auth:
      type: query
      name: key
      env: GOOGLE_KEY
    description: Google Custom Search
""",
        )
        result = load_services(path)
        assert "google" in result
        assert result["google"].auth.type == "query"
        assert result["google"].auth.name == "key"

    def test_valid_none_auth(self, tmp_path):
        """No-auth services don't require env vars."""
        path = _write_yaml(
            tmp_path,
            """
services:
  local:
    url: http://localhost:8888/search
    method: GET
    auth:
      type: none
    description: Local SearXNG
""",
        )
        result = load_services(path)
        assert "local" in result
        assert result["local"].auth.type == "none"

    def test_missing_env_var_skips_service(self, tmp_path, monkeypatch):
        """Service is skipped when its required env var is not set."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        path = _write_yaml(
            tmp_path,
            """
services:
  broken:
    url: https://api.example.com
    method: GET
    auth:
      type: bearer
      env: MISSING_KEY
    description: Will be skipped
""",
        )
        result = load_services(path)
        assert "broken" not in result

    def test_optional_auth_keeps_service(self, tmp_path, monkeypatch):
        """Services with optional auth remain available when env var is unset."""
        monkeypatch.delenv("OPTIONAL_KEY", raising=False)
        path = _write_yaml(
            tmp_path,
            """
services:
  jina:
    url: https://r.jina.ai/
    method: GET
    auth:
      type: bearer
      env: OPTIONAL_KEY
      optional: true
    description: Jina Reader
""",
        )
        result = load_services(path)
        assert "jina" in result
        assert result["jina"].auth.optional is True

    def test_invalid_auth_type_skips(self, tmp_path):
        """Services with unrecognized auth types are skipped."""
        path = _write_yaml(
            tmp_path,
            """
services:
  bad:
    url: https://api.example.com
    method: POST
    auth:
      type: oauth2
    description: Invalid auth type
""",
        )
        result = load_services(path)
        assert "bad" not in result

    def test_missing_url_skips(self, tmp_path):
        """Services without a URL are skipped."""
        path = _write_yaml(
            tmp_path,
            """
services:
  nourl:
    method: GET
    auth:
      type: none
    description: No URL
""",
        )
        result = load_services(path)
        assert "nourl" not in result

    def test_invalid_yaml_raises_system_exit(self, tmp_path):
        """Malformed YAML causes SystemExit (fail fast on broken config)."""
        path = _write_yaml(tmp_path, "services:\n  bad: [unbalanced: {")
        with pytest.raises(SystemExit):
            load_services(path)

    def test_static_headers_and_params(self, tmp_path):
        """Static headers and params from config are preserved."""
        path = _write_yaml(
            tmp_path,
            """
services:
  local:
    url: http://localhost:8888/search
    method: GET
    auth:
      type: none
    headers:
      Accept: application/json
    params:
      format: json
      language: en
    description: Local search
""",
        )
        result = load_services(path)
        assert result["local"].headers == {"Accept": "application/json"}
        assert result["local"].params == {"format": "json", "language": "en"}

    def test_notes_field(self, tmp_path, monkeypatch):
        """The notes field is loaded for context injection."""
        monkeypatch.setenv("TEST_KEY", "key")
        path = _write_yaml(
            tmp_path,
            """
services:
  testapi:
    url: https://api.example.com
    method: POST
    auth:
      type: bearer
      env: TEST_KEY
    description: Test
    notes: Send JSON body with "query" field
""",
        )
        result = load_services(path)
        assert result["testapi"].notes == 'Send JSON body with "query" field'

    def test_mixed_valid_and_invalid(self, tmp_path, monkeypatch):
        """Valid services load even when other entries are invalid."""
        monkeypatch.setenv("GOOD_KEY", "value")
        path = _write_yaml(
            tmp_path,
            """
services:
  good:
    url: https://api.example.com
    method: GET
    auth:
      type: bearer
      env: GOOD_KEY
    description: Good service
  bad_no_url:
    method: GET
    auth:
      type: none
  bad_auth:
    url: https://api.example.com
    auth:
      type: magic
""",
        )
        result = load_services(path)
        assert "good" in result
        assert "bad_no_url" not in result
        assert "bad_auth" not in result

    def test_empty_file_returns_empty(self, tmp_path):
        """An empty YAML file returns an empty registry."""
        path = _write_yaml(tmp_path, "")
        result = load_services(path)
        assert result == {}

    def test_no_services_key_returns_empty(self, tmp_path):
        """A YAML file without a 'services' key returns an empty registry."""
        path = _write_yaml(tmp_path, "other_key: value\n")
        result = load_services(path)
        assert result == {}

    def test_method_defaults_to_get(self, tmp_path):
        """Method defaults to GET when not specified."""
        path = _write_yaml(
            tmp_path,
            """
services:
  simple:
    url: http://localhost/api
    auth:
      type: none
""",
        )
        result = load_services(path)
        assert result["simple"].method == "GET"

    def test_method_normalized_to_uppercase(self, tmp_path, monkeypatch):
        """Method is normalized to uppercase."""
        monkeypatch.setenv("K", "v")
        path = _write_yaml(
            tmp_path,
            """
services:
  api:
    url: https://api.example.com
    method: post
    auth:
      type: bearer
      env: K
""",
        )
        result = load_services(path)
        assert result["api"].method == "POST"


# ── TestGetAvailableServices ────────────────────────────────────────


class TestGetAvailableServices:
    def test_returns_metadata(self, tmp_path, monkeypatch):
        """Returns name, description, method, and notes for each service."""
        monkeypatch.setenv("KEY", "val")
        path = _write_yaml(
            tmp_path,
            """
services:
  myapi:
    url: https://api.example.com
    method: POST
    auth:
      type: bearer
      env: KEY
    description: My API
    notes: Use JSON body
""",
        )
        load_services(path)
        available = get_available_services()
        assert len(available) == 1
        assert available[0]["name"] == "myapi"
        assert available[0]["description"] == "My API"
        assert available[0]["method"] == "POST"
        assert available[0]["notes"] == "Use JSON body"

    def test_empty_when_no_services(self, tmp_path):
        """Returns empty list when no services are configured."""
        load_services(tmp_path / "nonexistent.yaml")
        assert get_available_services() == []


# ── TestCallService ─────────────────────────────────────────────────


class TestCallService:
    async def test_unknown_service_error(self, tmp_path):
        """Calling a non-existent service returns an error response."""
        load_services(tmp_path / "nonexistent.yaml")
        result = await call_service("nonexistent")
        assert result.success is False
        assert "Unknown service" in result.error

    async def test_bearer_auth_injected(self, tmp_path, monkeypatch):
        """Bearer token is injected into the Authorization header."""
        monkeypatch.setenv("API_KEY", "mytoken123")
        path = _write_yaml(
            tmp_path,
            """
services:
  testapi:
    url: https://api.example.com/v1
    method: POST
    auth:
      type: bearer
      env: API_KEY
    headers:
      Content-Type: application/json
""",
        )
        load_services(path)

        # Mock aiohttp.ClientSession to capture the request
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"result": "ok"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("testapi", body={"query": "test"})

        assert result.success is True
        assert result.status == 200
        assert result.body == '{"result": "ok"}'

        # Verify the Authorization header was injected
        call_kwargs = mock_session.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers["Authorization"] == "Bearer mytoken123"
        assert headers["Content-Type"] == "application/json"

    async def test_header_auth_injected(self, tmp_path, monkeypatch):
        """Custom header auth is injected with the correct header name."""
        monkeypatch.setenv("BRAVE_KEY", "bravetoken")
        path = _write_yaml(
            tmp_path,
            """
services:
  brave:
    url: https://api.search.brave.com/res/v1/web/search
    method: GET
    auth:
      type: header
      name: X-Subscription-Token
      env: BRAVE_KEY
""",
        )
        load_services(path)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="[]")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("brave", params={"q": "test"})

        assert result.success is True
        call_kwargs = mock_session.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers["X-Subscription-Token"] == "bravetoken"

    async def test_query_auth_injected(self, tmp_path, monkeypatch):
        """Query param auth is injected into the params dict."""
        monkeypatch.setenv("GOOGLE_KEY", "googlekey")
        path = _write_yaml(
            tmp_path,
            """
services:
  google:
    url: https://customsearch.googleapis.com/customsearch/v1
    method: GET
    auth:
      type: query
      name: key
      env: GOOGLE_KEY
""",
        )
        load_services(path)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="{}")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("google", params={"q": "test"})

        assert result.success is True
        call_kwargs = mock_session.request.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params["key"] == "googlekey"
        assert params["q"] == "test"

    async def test_path_suffix_appended(self, tmp_path, monkeypatch):
        """Path suffix is appended to the base URL."""
        monkeypatch.setenv("JINA_KEY", "jinakey")
        path = _write_yaml(
            tmp_path,
            """
services:
  jina:
    url: https://r.jina.ai/
    method: GET
    auth:
      type: bearer
      env: JINA_KEY
""",
        )
        load_services(path)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="# Page content")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("jina", path_suffix="https://example.com")

        assert result.success is True
        call_kwargs = mock_session.request.call_args
        url = call_kwargs.kwargs.get("url") or call_kwargs[1].get("url")
        assert url == "https://r.jina.ai/https://example.com"

    async def test_env_var_disappears_after_load(self, tmp_path, monkeypatch):
        """If an env var is unset after load, call_service returns an error."""
        monkeypatch.setenv("TEMP_KEY", "tempvalue")
        path = _write_yaml(
            tmp_path,
            """
services:
  tempapi:
    url: https://api.example.com
    method: GET
    auth:
      type: bearer
      env: TEMP_KEY
""",
        )
        load_services(path)
        assert "tempapi" in services.get_services()

        # Remove the env var after loading
        monkeypatch.delenv("TEMP_KEY")
        result = await call_service("tempapi")
        assert result.success is False
        assert "not set" in result.error

    async def test_static_params_merged(self, tmp_path):
        """Static params from config are merged with caller-provided params."""
        path = _write_yaml(
            tmp_path,
            """
services:
  local:
    url: http://localhost:8888/search
    method: GET
    auth:
      type: none
    params:
      format: json
""",
        )
        load_services(path)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="[]")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("local", params={"q": "test"})

        assert result.success is True
        call_kwargs = mock_session.request.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        # Both static and caller params should be present
        assert params["format"] == "json"
        assert params["q"] == "test"

    async def test_connection_error_handled(self, tmp_path):
        """Connection errors return a ServiceResponse with error details."""
        path = _write_yaml(
            tmp_path,
            """
services:
  failing:
    url: http://localhost:99999/api
    method: GET
    auth:
      type: none
""",
        )
        load_services(path)

        import aiohttp as _aiohttp

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=_aiohttp.ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("failing")

        assert result.success is False
        assert "Connection error" in result.error

    async def test_non_2xx_still_succeeds(self, tmp_path, monkeypatch):
        """Non-2xx status codes are still returned as success (HTTP completed)."""
        monkeypatch.setenv("KEY", "val")
        path = _write_yaml(
            tmp_path,
            """
services:
  api:
    url: https://api.example.com
    method: POST
    auth:
      type: bearer
      env: KEY
""",
        )
        load_services(path)

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value='{"error": "rate limited"}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("kai.services.aiohttp.ClientSession", return_value=mock_session):
            result = await call_service("api", body={"query": "test"})

        # HTTP completed, even though status is 429 — that's still success=True
        assert result.success is True
        assert result.status == 429
        assert "rate limited" in result.body
