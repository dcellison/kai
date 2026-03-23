"""
Shared Telegram messaging utilities.

Helpers for working with Telegram's message constraints (character limits,
formatting) that are used across multiple modules.
"""


def chunk_text(text: str, max_len: int = 4096) -> list[str]:
    """
    Split text into Telegram-safe chunks at natural break points.

    Prefers splitting at double newlines (paragraph breaks), then single
    newlines, and only falls back to hard-cutting at max_len if no break
    point is found. This keeps code blocks and paragraphs intact.

    Args:
        text: The text to split.
        max_len: Maximum length per chunk (Telegram's limit is 4096).

    Returns:
        A list of text chunks, each within max_len.
    """
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try paragraph break, then line break, then hard cut
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunk = text[:split_at]
        if chunk:  # skip empty chunks (e.g., text starts with "\n\n")
            chunks.append(chunk)
        text = text[split_at:].lstrip("\n")
    return chunks
