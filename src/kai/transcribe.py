"""
Voice message transcription using local whisper-cpp.

Provides functionality to:
1. Accept raw Ogg Opus audio bytes (as received from Telegram voice messages)
2. Convert to 16kHz mono WAV via ffmpeg (the format whisper expects)
3. Run whisper-cli locally for speech-to-text transcription
4. Return the transcribed text

This module is opt-in — controlled by VOICE_ENABLED in .env. Both ffmpeg and
whisper-cpp must be installed locally (brew install ffmpeg whisper-cpp) along
with a GGML model file (default: models/ggml-base.en.bin).

The main interface is transcribe_voice(), which handles the full pipeline
from raw audio bytes to transcript string.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """
    Raised when any step of the transcription pipeline fails.

    Includes missing dependencies, timeouts, and non-zero exit codes
    from ffmpeg or whisper-cli. Error messages include install hints
    so the user can fix the issue.
    """


async def transcribe_voice(audio_data: bytes, model_path: Path) -> str:
    """
    Transcribe voice audio bytes to text using ffmpeg + whisper-cli.

    The caller (handle_voice in bot.py) downloads the audio from Telegram
    and passes the raw bytes here. This function handles the conversion and
    transcription pipeline in a temporary directory that is cleaned up afterward.

    Args:
        audio_data: Raw Ogg Opus audio bytes from Telegram's voice message.
        model_path: Path to the whisper-cpp GGML model file.

    Returns:
        The transcribed text, stripped of leading/trailing whitespace.

    Raises:
        TranscriptionError: If the model is missing, ffmpeg fails, whisper
            fails, or either process times out (30-second limit).
    """
    if not model_path.exists():
        raise TranscriptionError(
            f"Whisper model not found at {model_path}. Download with: make models/ggml-base.en.bin"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ogg_path = Path(tmpdir) / "voice.oga"
        wav_path = Path(tmpdir) / "voice.wav"

        ogg_path.write_bytes(audio_data)

        # Step 1: Convert Ogg Opus → 16kHz mono WAV (what whisper expects)
        await _run(
            "ffmpeg",
            "-i",
            str(ogg_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(wav_path),
            label="ffmpeg",
        )

        # Step 2: Transcribe WAV → text via whisper-cli
        stdout = await _run(
            "whisper-cli",
            "--model",
            str(model_path),
            "--file",
            str(wav_path),
            "--no-prints",
            "--no-timestamps",
            "--language",
            "en",
            label="whisper-cli",
        )

    transcript = stdout.strip()
    log.info("Transcribed %d bytes of audio → %d chars", len(audio_data), len(transcript))
    return transcript


async def _run(*cmd: str, label: str) -> str:
    """
    Run a subprocess asynchronously with a 30-second timeout.

    Provides consistent error handling for the external tools (ffmpeg,
    whisper-cli) used in the transcription pipeline: missing binary,
    timeout, and non-zero exit code are all raised as TranscriptionError
    with helpful install hints.

    Args:
        *cmd: Command and arguments to execute.
        label: Human-readable name for error messages (e.g., "ffmpeg").

    Returns:
        The decoded stdout output from the subprocess.

    Raises:
        TranscriptionError: If the binary is not found, times out, or exits
            with a non-zero code.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise TranscriptionError(
            f"{label} not found. Install with: brew install {'whisper-cpp' if 'whisper' in label else label}"
        ) from None

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        raise TranscriptionError(f"{label} timed out after 30 seconds") from None

    if proc.returncode != 0:
        err = stderr.decode().strip()[:200]
        raise TranscriptionError(f"{label} failed (exit {proc.returncode}): {err}")

    return stdout.decode()
