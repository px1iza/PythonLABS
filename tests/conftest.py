
from pathlib import Path

import pytest


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a small corpus of text documents for testing."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    documents = {
        "doc1.txt": "Python is a programming language.",
        "doc2.txt": "Python supports generators and iterators.",
        "doc3.txt": "Programming with Python is interesting.",
    }

    for filename, content in documents.items():
        file_path = corpus / filename
        file_path.write_text(content, encoding="utf-8")

    return corpus
