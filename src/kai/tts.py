"""Text-to-speech synthesis using local Piper TTS."""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Curated English voices — each maps to a Piper model file
VOICES = {
    "cori": "Cori (F, British)",
    "alba": "Alba (F, British)",
    "jenny": "Jenny (F, British)",
    "alan": "Alan (M, British)",
    "amy": "Amy (F, American)",
    "lessac": "Lessac (F, American)",
    "ryan": "Ryan (M, American)",
    "joe": "Joe (M, American)",
}

# Short name → Piper model filename (without .onnx extension)
_VOICE_MODELS = {
    "cori": "en_GB-cori-medium",
    "alba": "en_GB-alba-medium",
    "jenny": "en_GB-jenny_dioco-medium",
    "alan": "en_GB-alan-medium",
    "amy": "en_US-amy-medium",
    "lessac": "en_US-lessac-medium",
    "ryan": "en_US-ryan-medium",
    "joe": "en_US-joe-medium",
}

DEFAULT_VOICE = "cori"


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""


async def synthesize_speech(text: str, model_dir: Path, voice: str = DEFAULT_VOICE) -> bytes:
    """Convert text to OGG Opus audio bytes via Piper TTS + ffmpeg.

    Returns bytes suitable for Telegram's send_voice().
    """
    if not text.strip():
        raise TTSError("No text to synthesize")

    model_name = _VOICE_MODELS.get(voice)
    if not model_name:
        raise TTSError(f"Unknown voice: {voice}. Choose from: {', '.join(VOICES)}")

    model_path = model_dir / f"{model_name}.onnx"
    if not model_path.exists():
        raise TTSError(
            f"Piper model not found at {model_path}. "
            "Download with: make tts-model"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "speech.wav"
        ogg_path = Path(tmpdir) / "speech.ogg"

        # Synthesize text → WAV via Piper (reads from stdin)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "piper",
                "--model", str(model_path),
                "--output_file", str(wav_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise TTSError(
                "piper-tts not found. Install with: pip install -e '.[tts]'"
            ) from None

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(input=text.encode()),
                timeout=120,
            )
        except TimeoutError:
            proc.kill()
            raise TTSError("Piper TTS timed out after 120 seconds") from None

        if proc.returncode != 0:
            err = stderr.decode().strip()[:200]
            raise TTSError(f"Piper failed (exit {proc.returncode}): {err}")

        # Convert WAV → OGG Opus via ffmpeg
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(wav_path),
                "-c:a", "libopus", "-f", "ogg",
                "-y", str(ogg_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise TTSError("ffmpeg not found. Install with: brew install ffmpeg") from None

        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            proc.kill()
            raise TTSError("ffmpeg timed out after 30 seconds") from None

        if proc.returncode != 0:
            err = stderr.decode().strip()[:200]
            raise TTSError(f"ffmpeg failed (exit {proc.returncode}): {err}")

        audio_bytes = ogg_path.read_bytes()

    log.info("Synthesized %d chars → %d bytes OGG (%s)", len(text), len(audio_bytes), voice)
    return audio_bytes
