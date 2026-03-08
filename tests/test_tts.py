"""
Tests for tts.py - text-to-speech synthesis via Piper TTS + ffmpeg.

Tests validation paths (empty text, unknown voice, missing model), piper
subprocess errors, ffmpeg subprocess errors, and the full pipeline success
path. All subprocess calls are mocked via asyncio.create_subprocess_exec.
"""

import asyncio
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.tts import _VOICE_MODELS, DEFAULT_VOICE, TTSError, synthesize_speech

# ── Test helpers ─────────────────────────────────────────────────────


def _make_proc(stdout=b"", stderr=b"", returncode=0):
    """Create a mock subprocess with controllable communicate() output."""
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


def _model_path(tmp_path, voice=DEFAULT_VOICE):
    """Create the expected .onnx model file and return the model dir."""
    model_name = _VOICE_MODELS[voice]
    model_file = tmp_path / f"{model_name}.onnx"
    model_file.touch()
    return tmp_path


@pytest.fixture
def tts_tmpdir(tmp_path, monkeypatch):
    """
    Redirect TTS temp directory to a known location.

    synthesize_speech() uses tempfile.TemporaryDirectory() internally;
    this fixture patches it to return tmp_path so tests can predict
    and create the intermediate WAV/OGG file paths.
    """

    @contextmanager
    def _fake_tmpdir():
        yield str(tmp_path)

    monkeypatch.setattr("kai.tts.tempfile.TemporaryDirectory", _fake_tmpdir)
    return tmp_path


# ── Validation (no subprocess needed) ────────────────────────────────


class TestSynthesizeValidation:
    @pytest.mark.asyncio
    async def test_empty_text(self, tmp_path):
        with pytest.raises(TTSError, match="No text"):
            await synthesize_speech("   ", tmp_path)

    @pytest.mark.asyncio
    async def test_unknown_voice(self, tmp_path):
        with pytest.raises(TTSError, match="Unknown voice"):
            await synthesize_speech("hello", tmp_path, voice="nonexistent")

    @pytest.mark.asyncio
    async def test_model_missing(self, tmp_path):
        """Missing .onnx file raises TTSError with download hint."""
        with pytest.raises(TTSError, match="not found"):
            await synthesize_speech("hello", tmp_path, voice=DEFAULT_VOICE)


# ── Piper subprocess errors ──────────────────────────────────────────


class TestSynthesizePiper:
    @pytest.mark.asyncio
    async def test_piper_not_found(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        with (
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError,
            ),
            pytest.raises(TTSError, match="piper-tts not found"),
        ):
            await synthesize_speech("hello", model_dir)

    @pytest.mark.asyncio
    async def test_piper_timeout(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        proc = _make_proc()
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            patch("asyncio.wait_for", side_effect=TimeoutError),
            pytest.raises(TTSError, match="timed out"),
        ):
            await synthesize_speech("hello", model_dir)
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_piper_non_zero_exit(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        proc = _make_proc(stderr=b"segfault", returncode=139)
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            pytest.raises(TTSError, match=r"Piper failed.*exit 139"),
        ):
            await synthesize_speech("hello", model_dir)


# ── ffmpeg subprocess errors ─────────────────────────────────────────


class TestSynthesizeFfmpeg:
    def _piper_then_ffmpeg_error(self, tts_tmpdir, ffmpeg_side_effect):
        """
        Helper: piper succeeds (first call), ffmpeg fails (second call).

        Creates the intermediate WAV file (as piper would) so the test
        reaches the ffmpeg step. Returns a side_effect function for
        create_subprocess_exec.
        """
        call_count = 0

        async def _mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Piper succeeds - create the WAV file it would produce
                wav = tts_tmpdir / "speech.wav"
                wav.write_bytes(b"RIFF-fake-wav")
                return _make_proc(returncode=0)
            else:
                raise ffmpeg_side_effect

        return _mock_exec

    @pytest.mark.asyncio
    async def test_ffmpeg_not_found(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        mock_exec = self._piper_then_ffmpeg_error(tts_tmpdir, FileNotFoundError())
        with (
            patch("asyncio.create_subprocess_exec", side_effect=mock_exec),
            pytest.raises(TTSError, match="ffmpeg not found"),
        ):
            await synthesize_speech("hello", model_dir)

    @pytest.mark.asyncio
    async def test_ffmpeg_timeout(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        wait_for_count = 0

        async def _mock_exec(*args, **kwargs):
            if args[0] != "ffmpeg":
                # Piper call - write WAV file
                wav = tts_tmpdir / "speech.wav"
                wav.write_bytes(b"RIFF-fake-wav")
                return _make_proc(returncode=0)
            # ffmpeg call - return proc that will be timed out
            return _make_proc()

        original_wait_for = asyncio.wait_for

        async def _wait_for_side_effect(coro, timeout):
            nonlocal wait_for_count
            wait_for_count += 1
            if wait_for_count == 1:
                # Piper's communicate - let it through
                return await original_wait_for(coro, timeout=timeout)
            # ffmpeg's communicate - timeout
            raise TimeoutError

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_mock_exec),
            patch("kai.tts.asyncio.wait_for", side_effect=_wait_for_side_effect),
            pytest.raises(TTSError, match="ffmpeg timed out"),
        ):
            await synthesize_speech("hello", model_dir)

    @pytest.mark.asyncio
    async def test_ffmpeg_non_zero_exit(self, tmp_path, tts_tmpdir):
        model_dir = _model_path(tmp_path)
        call_count = 0

        async def _mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                wav = tts_tmpdir / "speech.wav"
                wav.write_bytes(b"RIFF-fake-wav")
                return _make_proc(returncode=0)
            return _make_proc(stderr=b"codec error", returncode=1)

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_mock_exec),
            pytest.raises(TTSError, match=r"ffmpeg failed.*exit 1"),
        ):
            await synthesize_speech("hello", model_dir)


# ── Full success path ────────────────────────────────────────────────


class TestSynthesizeSuccess:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path, tts_tmpdir):
        """Full pipeline: piper writes WAV, ffmpeg converts to OGG, returns bytes."""
        model_dir = _model_path(tmp_path)
        call_count = 0
        captured_args = []

        async def _mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_args.append(args)
            if call_count == 1:
                # Piper step - create the WAV file
                wav = tts_tmpdir / "speech.wav"
                wav.write_bytes(b"RIFF-fake-wav-data")
                return _make_proc(returncode=0)
            else:
                # ffmpeg step - create the OGG file
                ogg = tts_tmpdir / "speech.ogg"
                ogg.write_bytes(b"OggS-fake-audio")
                return _make_proc(returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            result = await synthesize_speech("Hello world", model_dir)

        assert result == b"OggS-fake-audio"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_piper_called_with_correct_args(self, tmp_path, tts_tmpdir):
        """Piper is called via sys.executable -m piper with model path."""
        model_dir = _model_path(tmp_path)
        captured_args = []
        captured_kwargs = []

        async def _mock_exec(*args, **kwargs):
            captured_args.append(args)
            captured_kwargs.append(kwargs)
            if len(captured_args) == 1:
                (tts_tmpdir / "speech.wav").write_bytes(b"wav")
            else:
                (tts_tmpdir / "speech.ogg").write_bytes(b"ogg")
            return _make_proc(returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            await synthesize_speech("Test text", model_dir)

        # First call is piper
        piper_args = captured_args[0]
        assert piper_args[0] == sys.executable
        assert "-m" in piper_args
        assert "piper" in piper_args
        assert "--model" in piper_args
        assert "--output_file" in piper_args

    @pytest.mark.asyncio
    async def test_piper_receives_text_via_stdin(self, tmp_path, tts_tmpdir):
        """Piper's communicate() is called with text.encode() as input."""
        model_dir = _model_path(tmp_path)
        piper_proc = _make_proc(returncode=0)

        async def _mock_exec(*args, **kwargs):
            if len(_mock_exec.calls) == 0:
                _mock_exec.calls.append(1)
                (tts_tmpdir / "speech.wav").write_bytes(b"wav")
                return piper_proc
            _mock_exec.calls.append(1)
            (tts_tmpdir / "speech.ogg").write_bytes(b"ogg")
            return _make_proc(returncode=0)

        _mock_exec.calls = []

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            await synthesize_speech("Say this", model_dir)

        # Piper's communicate should have been called with the text as bytes
        piper_proc.communicate.assert_called_once()
        call_kwargs = piper_proc.communicate.call_args
        # communicate() may receive text as positional or keyword arg
        input_bytes = call_kwargs[1].get("input") or (call_kwargs[0][0] if call_kwargs[0] else None)
        assert input_bytes == b"Say this"

    @pytest.mark.asyncio
    async def test_ffmpeg_called_with_opus(self, tmp_path, tts_tmpdir):
        """ffmpeg is called with libopus codec and ogg format."""
        model_dir = _model_path(tmp_path)
        captured_args = []

        async def _mock_exec(*args, **kwargs):
            captured_args.append(args)
            if len(captured_args) == 1:
                (tts_tmpdir / "speech.wav").write_bytes(b"wav")
            else:
                (tts_tmpdir / "speech.ogg").write_bytes(b"ogg")
            return _make_proc(returncode=0)

        with patch("asyncio.create_subprocess_exec", side_effect=_mock_exec):
            await synthesize_speech("Test", model_dir)

        # Second call is ffmpeg
        ffmpeg_args = captured_args[1]
        assert ffmpeg_args[0] == "ffmpeg"
        assert "libopus" in ffmpeg_args
        assert "ogg" in ffmpeg_args
