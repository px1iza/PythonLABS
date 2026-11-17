from collections.abc import Iterator
from pathlib import Path


def iter_documents(directory: str | Path) -> Iterator[tuple[str, str]]:
    """
    Reads text documents one at a time.

    Yields:
        A tuple containing the document name and its text.
    """
    directory = Path(directory)

    for path in sorted(directory.rglob("*.txt")):
        if path.is_file():
            yield path.name, path.read_text(encoding="utf-8")
