"""Resilient Edge TTS client with sentence chunking and timeouts.

Edge TTS is free to use but is an online service. Tests should mock the network
boundary or exercise :func:`chunk_text`, which is fully offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import io
import re

try:
    import edge_tts
except ImportError:  # pragma: no cover - exercised only in incomplete installs
    edge_tts = None


VOICES = {
    "English (US) - Christopher (Male)": "en-US-ChristopherNeural",
    "English (US) - Jenny (Female)": "en-US-JennyNeural",
    "English (US) - Eric (Male)": "en-US-EricNeural",
    "Français (FR) - Henri (Homme)": "fr-FR-HenriNeural",
    "Français (FR) - Denise (Femme)": "fr-FR-DeniseNeural",
}


class SpeechGenerationError(RuntimeError):
    """Raised when the online speech service cannot generate usable audio."""


DEFAULT_CHUNK_CHARS = 600


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split long text on sentence/word boundaries for reliable synthesis."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    max_chars = max(200, int(max_chars))
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", compact)
    chunks: list[str] = []
    current = ""

    def push_words(value: str) -> None:
        nonlocal current
        words = value.split()
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(sentence) <= max_chars:
                current = sentence
            else:
                push_words(sentence)
    if current:
        chunks.append(current)
    return chunks


def _strip_leading_id3(data: bytes) -> bytes:
    """Remove a leading ID3 tag before concatenating subsequent MP3 chunks."""
    if len(data) < 10 or not data.startswith(b"ID3"):
        return data
    size_bytes = data[6:10]
    tag_size = (
        (size_bytes[0] & 0x7F) << 21
        | (size_bytes[1] & 0x7F) << 14
        | (size_bytes[2] & 0x7F) << 7
        | (size_bytes[3] & 0x7F)
    )
    return data[10 + tag_size :]


def estimate_mp3_duration(audio: bytes) -> float | None:
    """Estimate CBR MP3 duration from the first valid MPEG frame.

    Edge TTS currently returns constant-bitrate MP3 data, so this is accurate
    enough to detect a truncated response without adding another dependency.
    """
    if not audio:
        return None
    offset = 0
    if audio.startswith(b"ID3") and len(audio) >= 10:
        size_bytes = audio[6:10]
        offset = 10 + (
            (size_bytes[0] & 0x7F) << 21
            | (size_bytes[1] & 0x7F) << 14
            | (size_bytes[2] & 0x7F) << 7
            | (size_bytes[3] & 0x7F)
        )

    bitrate_tables = {
        "mpeg1_layer3": [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
        "mpeg2_layer3": [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    }
    scan_end = min(len(audio) - 4, offset + 65536)
    for index in range(offset, max(offset, scan_end)):
        header = int.from_bytes(audio[index : index + 4], "big")
        if header & 0xFFE00000 != 0xFFE00000:
            continue
        version_id = (header >> 19) & 0b11
        layer_id = (header >> 17) & 0b11
        bitrate_index = (header >> 12) & 0b1111
        if layer_id != 0b01 or bitrate_index in {0, 15}:
            continue
        table = bitrate_tables["mpeg1_layer3" if version_id == 0b11 else "mpeg2_layer3"]
        bitrate_kbps = table[bitrate_index]
        if bitrate_kbps:
            return (len(audio) - offset) * 8 / (bitrate_kbps * 1000)
    return None


async def _generate_chunk(text: str, voice: str, rate: str, timeout_seconds: float) -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    output = io.BytesIO()

    async def consume() -> None:
        async for message in communicate.stream():
            if message.get("type") == "audio":
                output.write(message["data"])

    try:
        await asyncio.wait_for(consume(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise SpeechGenerationError(
            f"La synthèse vocale a dépassé {timeout_seconds:g} secondes. Vérifie la connexion Internet."
        ) from exc
    except Exception as exc:
        raise SpeechGenerationError(f"Edge TTS est indisponible: {exc}") from exc

    audio = output.getvalue()
    if len(audio) < 256:
        raise SpeechGenerationError("Edge TTS n'a renvoyé aucun audio exploitable.")
    return audio


async def generate_speech_async(
    text: str,
    voice: str = "en-US-ChristopherNeural",
    rate: str = "+0%",
    timeout_seconds: float = 35.0,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    progress: Callable[[int, int], None] | None = None,
) -> bytes:
    """Generate MP3 bytes sequentially in bounded chunks.

    ``progress`` is called with ``(completed_chunks, total_chunks)`` after each
    segment, so a caller can report advancement on long selections instead of
    showing an indeterminate spinner for several minutes.
    """
    if edge_tts is None:
        raise ImportError("Installe edge-tts avec `pip install edge-tts`.")
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Aucun texte à lire.")

    audio_parts = []
    for index, chunk in enumerate(chunks):
        audio = await _generate_chunk(chunk, voice, rate, timeout_seconds)
        audio_parts.append(audio if index == 0 else _strip_leading_id3(audio))
        if progress is not None:
            progress(index + 1, len(chunks))
    return b"".join(audio_parts)


def generate_speech(
    text: str,
    voice: str = "en-US-ChristopherNeural",
    rate: float = 1.0,
    timeout_seconds: float = 35.0,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    progress: Callable[[int, int], None] | None = None,
) -> bytes:
    """Synchronous wrapper suitable for Streamlit's script thread."""
    percentage = round((float(rate) - 1.0) * 100)
    rate_string = f"{percentage:+d}%"
    coroutine = generate_speech_async(
        text,
        voice=voice,
        rate=rate_string,
        timeout_seconds=timeout_seconds,
        max_chars=max_chars,
        progress=progress,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # This branch is uncommon in Streamlit but useful in notebooks.
    import nest_asyncio

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coroutine)
