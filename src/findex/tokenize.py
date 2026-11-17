import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:['’][^\W_]+|[-‐‑‒–—][^\W_]+)*",
    re.UNICODE,
)


def tokenize(text: str) -> Iterator[str]:
    """
    Generates normalized tokens from text.

    Rules:
    - Unicode NFKC normalization is applied.
    - All tokens are converted to lowercase using casefold().
    - Internal apostrophes are preserved: don't, l’amour.
    - Internal hyphens are preserved: well-known.
    - Numbers are treated as tokens: 2026, python3.
    - Punctuation and underscores separate tokens.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()

    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group()
