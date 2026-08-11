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
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_MACOS_TASKPOLICY = "/usr/sbin/taskpolicy"
_TASKPOLICY_TIMEOUT_SECONDS = 5


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

        # Step 2: Transcribe WAV → text via whisper-cli. The Metal path in
        # current whisper.cpp builds is not reliable from Kai's macOS
        # LaunchDaemon context: it can hang until the outer timeout, while the
        # same input completes promptly on CPU. Keep GPU selection unchanged
        # on other platforms.
        whisper_command = ["whisper-cli"]
        if sys.platform == "darwin":
            whisper_command.append("--no-gpu")
        whisper_command.extend(
            [
                "--model",
                str(model_path),
                "--file",
                str(wav_path),
                "--no-prints",
                "--no-timestamps",
                "--language",
                "en",
            ]
        )
        stdout = await _run(
            *whisper_command,
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

    if label == "whisper-cli" and sys.platform == "darwin":
        await _remove_macos_background_policy(proc.pid)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        await proc.wait()  # Reap the killed process to avoid zombies
        raise TranscriptionError(f"{label} timed out after 30 seconds") from None

    if proc.returncode != 0:
        err = stderr.decode().strip()[:200]
        raise TranscriptionError(f"{label} failed (exit {proc.returncode}): {err}")

    return stdout.decode()


async def _remove_macos_background_policy(pid: int) -> None:
    """Remove launchd's inherited background scheduling from Whisper only."""
    try:
        policy_proc = await asyncio.create_subprocess_exec(
            _MACOS_TASKPOLICY,
            "-B",
            "-p",
            str(pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("Could not adjust Whisper scheduling: %s is missing", _MACOS_TASKPOLICY)
        return

    try:
        _, stderr = await asyncio.wait_for(policy_proc.communicate(), timeout=_TASKPOLICY_TIMEOUT_SECONDS)
    except TimeoutError:
        policy_proc.kill()
        await policy_proc.wait()
        log.warning("Could not adjust Whisper scheduling: taskpolicy timed out")
        return

    if policy_proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[:200]
        log.warning(
            "Could not adjust Whisper scheduling: taskpolicy exited %d%s",
            policy_proc.returncode,
            f": {detail}" if detail else "",
        )
