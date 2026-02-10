"""Voice message transcription using local whisper-cpp."""

import asyncio
import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


async def transcribe_voice(audio_data: bytes, model_path: Path) -> str:
    """Transcribe voice audio using ffmpeg + whisper-cli.

    Downloads are already handled by the caller; this receives raw audio bytes
    (Ogg Opus from Telegram), converts to WAV, and runs whisper-cli locally.
    """
    if not model_path.exists():
        raise TranscriptionError(
            f"Whisper model not found at {model_path}. "
            "Download with: make models/ggml-base.en.bin"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ogg_path = Path(tmpdir) / "voice.oga"
        wav_path = Path(tmpdir) / "voice.wav"

        ogg_path.write_bytes(audio_data)

        # Convert Ogg Opus → 16kHz mono WAV (what whisper expects)
        await _run(
            "ffmpeg", "-i", str(ogg_path),
            "-ar", "16000", "-ac", "1", "-f", "wav",
            str(wav_path),
            label="ffmpeg",
        )

        # Transcribe
        stdout = await _run(
            "whisper-cli",
            "--model", str(model_path),
            "--file", str(wav_path),
            "--no-prints",
            "--no-timestamps",
            "--language", "en",
            label="whisper-cli",
        )

    transcript = stdout.strip()
    log.info("Transcribed %d bytes of audio → %d chars", len(audio_data), len(transcript))
    return transcript


async def _run(*cmd: str, label: str) -> str:
    """Run a subprocess with timeout, returning stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise TranscriptionError(
            f"{label} not found. Install with: brew install "
            f"{'whisper-cpp' if 'whisper' in label else label}"
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
