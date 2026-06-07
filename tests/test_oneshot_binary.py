"""Tests for `kai.oneshot_binary` - the leaf binary resolver shared by
config validation, the OneShotReasoner argv builders, and the smoke
command.

Resolution rules under test:
- claude: shutil.which("claude") only.
- codex: branches on CODEX_BIN. Explicit override validates as
  is-file plus executable, no PATH fallback. No override falls back
  to shutil.which.
- Unknown backend: ValueError (distinct from BinaryResolutionError;
  unknown backend is a caller bug, not an operator PATH issue).
"""

from __future__ import annotations

import pytest

from kai.oneshot_binary import BinaryResolutionError, resolve_oneshot_binary


class TestClaudeResolution:
    def test_resolves_via_path(self, monkeypatch):
        """shutil.which returning a path means the resolver returns it
        verbatim. Pinned via monkeypatch so the test does not depend
        on the host having claude installed."""
        monkeypatch.setattr(
            "kai.oneshot_binary.shutil.which", lambda name: "/fake/path/claude" if name == "claude" else None
        )
        assert resolve_oneshot_binary("claude") == "/fake/path/claude"

    def test_raises_when_unreachable(self, monkeypatch):
        """No claude on PATH must raise BinaryResolutionError, NOT
        return a fallback or empty string. The message names the
        candidate so the operator can fix it."""
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: None)
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("claude")
        assert "claude" in str(exc.value)
        assert "PATH" in str(exc.value)


class TestCodexResolution:
    def test_no_override_resolves_via_path(self, monkeypatch):
        """CODEX_BIN unset, codex on PATH. Same shape as the claude
        resolution path; the test is here for parity so a future
        backend-asymmetric regression surfaces."""
        monkeypatch.delenv("CODEX_BIN", raising=False)
        monkeypatch.setattr(
            "kai.oneshot_binary.shutil.which", lambda name: "/fake/path/codex" if name == "codex" else None
        )
        assert resolve_oneshot_binary("codex") == "/fake/path/codex"

    def test_no_override_unreachable_raises_naming_both_candidates(self, monkeypatch):
        """Without CODEX_BIN and without codex on PATH, the message
        must name both candidates so the operator does not have to
        guess which branch fired."""
        monkeypatch.delenv("CODEX_BIN", raising=False)
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: None)
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("codex")
        message = str(exc.value)
        assert "CODEX_BIN" in message
        assert "PATH" in message

    def test_explicit_override_resolves_to_exact_path(self, tmp_path, monkeypatch):
        """A real executable file at CODEX_BIN resolves to that exact
        path, NOT a shutil.which lookup of its name. Tests the
        is-file plus executable validation."""
        fake = tmp_path / "codex-binary"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("CODEX_BIN", str(fake))
        # Even if PATH would also resolve codex, the override wins.
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/other/codex")
        assert resolve_oneshot_binary("codex") == str(fake)

    def test_explicit_override_not_a_file_raises_not_a_file(self, tmp_path, monkeypatch):
        """CODEX_BIN pointing at a nonexistent path raises with
        'not-a-file' in the message. NO PATH fallback fires; a bad
        explicit override is a configuration error, not a recovery
        opportunity."""
        monkeypatch.setenv("CODEX_BIN", str(tmp_path / "does-not-exist"))
        # Ensure PATH would have resolved had the fallback been used;
        # the test fails open if the fallback unexpectedly fires.
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/should/not/be/returned")
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("codex")
        message = str(exc.value)
        assert "not-a-file" in message
        assert "does-not-exist" in message

    def test_explicit_override_not_executable_raises_not_executable(self, tmp_path, monkeypatch):
        """CODEX_BIN pointing at a non-executable file raises with
        'not-executable' in the message. The two error modes (not-a-
        file vs not-executable) get distinct messages so the operator
        can resolve the right one."""
        fake = tmp_path / "codex-not-x"
        fake.write_text("not-a-script")
        fake.chmod(0o644)  # no execute bit
        monkeypatch.setenv("CODEX_BIN", str(fake))
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/should/not/be/returned")
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("codex")
        message = str(exc.value)
        assert "not-executable" in message
        assert str(fake) in message

    def test_explicit_override_no_fallback_to_path(self, tmp_path, monkeypatch):
        """Regression: when CODEX_BIN is set and resolves to a bad
        path, the resolver MUST NOT silently fall back to
        shutil.which('codex'). The fallback would hide stale-config
        bugs from the operator."""
        monkeypatch.setenv("CODEX_BIN", str(tmp_path / "missing"))

        # Set up shutil.which to fail loudly if called - confirms
        # the no-fallback contract directly.
        called = []

        def fake_which(name):
            called.append(name)
            return "/some/codex"

        monkeypatch.setattr("kai.oneshot_binary.shutil.which", fake_which)
        with pytest.raises(BinaryResolutionError):
            resolve_oneshot_binary("codex")
        assert called == [], f"shutil.which should not be called when CODEX_BIN is set; got {called}"


class TestOpenCodeResolution:
    """OpenCode arm mirrors the codex pattern: OPENCODE_BIN override
    validates as is-file plus executable with no PATH fallback; no
    override falls back to shutil.which("opencode")."""

    def test_no_override_resolves_via_path(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_BIN", raising=False)
        monkeypatch.setattr(
            "kai.oneshot_binary.shutil.which",
            lambda name: "/fake/path/opencode" if name == "opencode" else None,
        )
        assert resolve_oneshot_binary("opencode") == "/fake/path/opencode"

    def test_no_override_unreachable_raises_naming_both_candidates(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_BIN", raising=False)
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: None)
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("opencode")
        message = str(exc.value)
        assert "OPENCODE_BIN" in message
        assert "PATH" in message

    def test_explicit_override_resolves_to_exact_path(self, tmp_path, monkeypatch):
        fake = tmp_path / "opencode-binary"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("OPENCODE_BIN", str(fake))
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/other/opencode")
        assert resolve_oneshot_binary("opencode") == str(fake)

    def test_explicit_override_not_a_file_raises_not_a_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENCODE_BIN", str(tmp_path / "does-not-exist"))
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/should/not/be/returned")
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("opencode")
        message = str(exc.value)
        assert "not-a-file" in message
        assert "does-not-exist" in message

    def test_explicit_override_not_executable_raises_not_executable(self, tmp_path, monkeypatch):
        fake = tmp_path / "opencode-not-x"
        fake.write_text("not-a-script")
        fake.chmod(0o644)
        monkeypatch.setenv("OPENCODE_BIN", str(fake))
        monkeypatch.setattr("kai.oneshot_binary.shutil.which", lambda name: "/should/not/be/returned")
        with pytest.raises(BinaryResolutionError) as exc:
            resolve_oneshot_binary("opencode")
        message = str(exc.value)
        assert "not-executable" in message
        assert str(fake) in message

    def test_explicit_override_no_fallback_to_path(self, tmp_path, monkeypatch):
        """OPENCODE_BIN set to a bad path must NOT silently fall back to
        shutil.which('opencode'). Same regression guard as the codex
        arm; a fallback would hide stale-config bugs from the operator."""
        monkeypatch.setenv("OPENCODE_BIN", str(tmp_path / "missing"))

        called: list[str] = []

        def fake_which(name):
            called.append(name)
            return "/some/opencode"

        monkeypatch.setattr("kai.oneshot_binary.shutil.which", fake_which)
        with pytest.raises(BinaryResolutionError):
            resolve_oneshot_binary("opencode")
        assert called == [], f"shutil.which should not be called when OPENCODE_BIN is set; got {called}"


class TestUnknownBackend:
    def test_unknown_backend_raises_value_error(self):
        """An unknown backend string is a CALLER bug (config validation
        upstream should have rejected it), so the exception type is
        ValueError, NOT BinaryResolutionError. The exception type
        split lets call sites distinguish 'fix the operator's PATH'
        from 'fix the code that asked for the wrong backend'."""
        with pytest.raises(ValueError, match="unknown backend"):
            resolve_oneshot_binary("not-a-real-backend")
