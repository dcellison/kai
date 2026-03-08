"""
Tests for transcribe.py - voice message transcription via ffmpeg + whisper-cli.

Tests the _run() subprocess helper (success, missing binary, non-zero exit,
timeout, stderr truncation) and the transcribe_voice() pipeline (model
validation, full pipeline with mocked subprocesses).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.transcribe import TranscriptionError, _run, transcribe_voice

# ── Test helpers ─────────────────────────────────────────────────────


def _make_proc(stdout=b"", stderr=b"", returncode=0):
    """Create a mock subprocess with controllable communicate() output."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# ── _run() ───────────────────────────────────────────────────────────


class TestRun:
    @pytest.mark.asyncio
    async def test_success(self):
        """Returns decoded stdout on success."""
        proc = _make_proc(stdout=b"hello world\n", returncode=0)
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
            result = await _run("echo", "hi", label="echo")
        assert result == "hello world\n"

    @pytest.mark.asyncio
    async def test_binary_not_found(self):
        """FileNotFoundError raises TranscriptionError with install hint."""
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError,
            ),
            pytest.raises(TranscriptionError, match="not found"),
        ):
            await _run("whisper-cli", label="whisper-cli")

    @pytest.mark.asyncio
    async def test_non_zero_exit(self):
        """Non-zero exit code raises TranscriptionError with stderr snippet."""
        proc = _make_proc(stderr=b"bad input format", returncode=1)
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            pytest.raises(TranscriptionError, match=r"failed.*exit 1.*bad input"),
        ):
            await _run("ffmpeg", label="ffmpeg")

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Timeout kills process and raises TranscriptionError."""
        proc = _make_proc()
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            patch("asyncio.wait_for", side_effect=TimeoutError),
            pytest.raises(TranscriptionError, match="timed out"),
        ):
            await _run("ffmpeg", label="ffmpeg")
        proc.kill.assert_called_once()
        proc.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_stderr_truncated(self):
        """Long stderr is truncated to 200 chars in error message."""
        long_err = b"x" * 500
        proc = _make_proc(stderr=long_err, returncode=1)
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            pytest.raises(TranscriptionError) as exc_info,
        ):
            await _run("ffmpeg", label="ffmpeg")
        # The stderr portion of the error message should be at most 200 chars
        msg = str(exc_info.value)
        # Extract the part after the colon (the stderr snippet)
        snippet = msg.split(": ", 1)[1] if ": " in msg else msg
        assert len(snippet) <= 200


# ── transcribe_voice() ───────────────────────────────────────────────


class TestTranscribeVoice:
    @pytest.mark.asyncio
    async def test_model_missing(self, tmp_path):
        """Raises TranscriptionError when model file doesn't exist."""
        missing = tmp_path / "no_model.bin"
        with pytest.raises(TranscriptionError, match="not found"):
            await transcribe_voice(b"audio", missing)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """Full pipeline: writes ogg, calls ffmpeg then whisper-cli, returns text."""
        model = tmp_path / "model.bin"
        model.touch()

        call_count = 0

        async def _mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # ffmpeg call - succeeds
                return _make_proc(returncode=0)
            else:
                # whisper-cli call - returns transcript
                return _make_proc(stdout=b"  Hello world  \n", returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            result = await transcribe_voice(b"fake-audio", model)

        assert result == "Hello world"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ffmpeg_args(self, tmp_path):
        """Verifies ffmpeg is called with 16kHz mono WAV args."""
        model = tmp_path / "model.bin"
        model.touch()

        captured_args = []

        async def _mock_exec(*args, **kwargs):
            captured_args.append(args)
            return _make_proc(stdout=b"text", returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            await transcribe_voice(b"audio", model)

        # First call is ffmpeg
        ffmpeg_args = captured_args[0]
        assert ffmpeg_args[0] == "ffmpeg"
        assert "-ar" in ffmpeg_args
        assert "16000" in ffmpeg_args
        assert "-ac" in ffmpeg_args
        assert "1" in ffmpeg_args

    @pytest.mark.asyncio
    async def test_whisper_args(self, tmp_path):
        """Verifies whisper-cli is called with model path and --no-timestamps."""
        model = tmp_path / "model.bin"
        model.touch()

        captured_args = []

        async def _mock_exec(*args, **kwargs):
            captured_args.append(args)
            return _make_proc(stdout=b"text", returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            await transcribe_voice(b"audio", model)

        # Second call is whisper-cli
        whisper_args = captured_args[1]
        assert whisper_args[0] == "whisper-cli"
        assert "--model" in whisper_args
        assert str(model) in whisper_args
        assert "--no-timestamps" in whisper_args
