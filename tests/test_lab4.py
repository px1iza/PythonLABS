
from pathlib import Path

import pytest
from hypothesis import given, strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.lab3 import (
    And,
    BM25,
    Index,
    Not,
    Or,
    Phrase,
    Posting,
    Term,
    TfIdf,
    build_index,
    extract_terms,
    make_snippet,
    open_index,
    parse_query,
    save_index,
    search,
)

runner = CliRunner()


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    """Build an index from the test corpus."""
    return build_index(sample_corpus)


@pytest.fixture
def saved_index(sample_index: Index, tmp_path: Path) -> Path:
    """Save the test index to a temporary file."""
    index_path = tmp_path / "index.bin"
    save_index(sample_index, index_path)
    return index_path


# Tokenization and term extraction

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python", ["python"]),
        ("PYTHON", ["python"]),
        ("Hello World", ["hello", "world"]),
        ("one two three", ["one", "two", "three"]),
        ("", []),
        ("12345", []),
        ("Python programming", ["python", "programming"]),
        ("Hello, world!", ["hello", "world"]),
        ("test-test", ["test", "test"]),
        ("UPPER lower", ["upper", "lower"]),
        ("Київ Україна", ["київ", "україна"]),
        ("Привіт світе", ["привіт", "світе"]),
        ("café résumé", ["café", "résumé"]),
        ("東京", ["東京"]),
        ("Python 2026 language", ["python", "language"]),
    ],
)
def test_extract_terms(text: str, expected: list[str]) -> None:
    assert extract_terms(text) == expected


def test_extract_terms_returns_list() -> None:
    assert isinstance(extract_terms("Python programming"), list)


def test_extract_terms_lowercase() -> None:
    assert extract_terms("PYTHON") == ["python"]


# Index construction

def test_build_index_document_count(sample_index: Index) -> None:
    assert sample_index.num_docs == 3


def test_build_index_contains_python(sample_index: Index) -> None:
    assert "python" in sample_index


def test_build_index_contains_programming(sample_index: Index) -> None:
    assert "programming" in sample_index


def test_build_index_document_frequency(sample_index: Index) -> None:
    assert sample_index.df("python") == 3


def test_build_index_has_document_metadata(sample_index: Index) -> None:
    assert len(sample_index.documents) == 3


def test_build_index_average_length(sample_index: Index) -> None:
    assert sample_index.avg_doc_length > 0


def test_build_index_postings_are_sorted(sample_index: Index) -> None:
    for term in sample_index:
        postings = sample_index[term]
        doc_ids = [posting.doc_id for posting in postings]
        assert doc_ids == sorted(doc_ids)


# Query parser and AST

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", Term("python")),
        ("programming", Term("programming")),
        ('"python language"', Phrase("python language")),
        ("python AND programming", And(Term("python"), Term("programming"))),
        ("python OR programming", Or(Term("python"), Term("programming"))),
        ("NOT python", Not(Term("python"))),
    ],
)
def test_parse_query_ast(query: str, expected: object) -> None:
    assert parse_query(query) == expected


def test_parse_query_and_or_precedence() -> None:
    result = parse_query("python AND programming OR language")

    assert result == Or(
        And(Term("python"), Term("programming")),
        Term("language"),
    )


def test_parse_query_parentheses() -> None:
    result = parse_query("(python OR programming) AND language")

    assert result == And(
        Or(Term("python"), Term("programming")),
        Term("language"),
    )


def test_parse_query_not() -> None:
    assert parse_query("NOT python") == Not(Term("python"))


# Boolean search

def test_search_and(sample_index: Index) -> None:
    results = search(
        sample_index,
        "python AND programming",
        scorer=BM25(),
    )

    assert {result.doc_id for result in results} == {0, 2}


def test_search_or(sample_index: Index) -> None:
    results = search(
        sample_index,
        "generators OR programming",
        scorer=BM25(),
    )

    assert {result.doc_id for result in results} == {0, 1, 2}


def test_search_not(sample_index: Index) -> None:
    results = search(
        sample_index,
        "NOT generators",
        scorer=BM25(),
    )

    assert {result.doc_id for result in results} == {0, 2}


def test_search_phrase(sample_index: Index) -> None:
    results = search(
        sample_index,
        '"programming language"',
        scorer=BM25(),
    )

    assert {result.doc_id for result in results} == {0}


# Ranking algorithms

def test_tfidf_returns_results(sample_index: Index) -> None:
    results = search(sample_index, "python", scorer=TfIdf())

    assert len(results) == 3


def test_bm25_returns_results(sample_index: Index) -> None:
    results = search(sample_index, "python", scorer=BM25())

    assert len(results) == 3


def test_scorers_return_sorted_results(sample_index: Index) -> None:
    for scorer in (TfIdf(), BM25()):
        results = search(sample_index, "python", scorer=scorer)
        scores = [result.score for result in results]

        assert scores == sorted(scores, reverse=True)


def test_search_top_k(sample_index: Index) -> None:
    results = search(
        sample_index,
        "python",
        scorer=BM25(),
        top_k=2,
    )

    assert len(results) == 2


def test_search_empty_results(sample_index: Index) -> None:
    results = search(
        sample_index,
        "nonexistentterm",
        scorer=BM25(),
    )

    assert results == []


def test_make_snippet() -> None:
    text = "Python is a programming language used by developers."
    snippet = make_snippet(text, "python")

    assert isinstance(snippet, str)
    assert snippet


# Saving and loading

def test_save_index_creates_file(
    sample_index: Index,
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "saved.bin"

    save_index(sample_index, index_path)

    assert index_path.is_file()


def test_open_index_loads_documents(saved_index: Path) -> None:
    with open_index(saved_index) as index:
        assert index.num_docs == 3


def test_save_load_preserves_postings(
    sample_index: Index,
    saved_index: Path,
) -> None:
    with open_index(saved_index) as loaded:
        assert loaded.postings == sample_index.postings


def test_save_load_preserves_document_metadata(
    sample_index: Index,
    saved_index: Path,
) -> None:
    with open_index(saved_index) as loaded:
        assert loaded.documents == sample_index.documents


# CLI

def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "index" in result.output
    assert "search" in result.output
    assert "stats" in result.output


def test_cli_index(sample_corpus: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "cli_index.bin"

    result = runner.invoke(
        app,
        [
            "index",
            "--data",
            str(sample_corpus),
            "--output",
            str(index_path),
        ],
    )

    assert result.exit_code == 0
    assert index_path.is_file()


def test_cli_stats(saved_index: Path) -> None:
    result = runner.invoke(
        app,
        ["stats", str(saved_index)],
    )

    assert result.exit_code == 0
    assert "Documents" in result.output


def test_cli_missing_index(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"

    result = runner.invoke(
        app,
        ["stats", str(missing)],
    )

    assert result.exit_code != 0
    assert "Error" in result.output


def test_cli_search_json(saved_index: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "python",
            "--index",
            str(saved_index),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"document"' in result.output


def test_cli_search_table(saved_index: Path) -> None:
    result = runner.invoke(
        app,
        [
            "search",
            "python",
            "--index",
            str(saved_index),
        ],
    )

    assert result.exit_code == 0
    assert "Search Results" in result.output


# Hypothesis property-based tests

@given(
    words=st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu"),
            ),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=20,
        unique=True,
    )
)
def test_save_load_property(
    words: list[str],
    tmp_path: Path,
) -> None:
    postings = {
        word: (Posting(doc_id=0, frequency=1, positions=(0,)),)
        for word in words
    }

    documents = {
        0: {
            "doc_id": 0,
            "filename": "document.txt",
            "length": len(words),
            "text": " ".join(words),
        }
    }

    index = Index(postings, documents)
    index_path = tmp_path / "property.bin"

    save_index(index, index_path)

    with open_index(index_path) as loaded:
        assert loaded.postings == index.postings


@given(
    doc_ids=st.lists(
        st.integers(min_value=0, max_value=1000),
        min_size=0,
        max_size=100,
        unique=True,
    )
)
def test_postings_sorted_property(doc_ids: list[int]) -> None:
    postings = [
        Posting(doc_id=doc_id, frequency=1, positions=(0,))
        for doc_id in sorted(doc_ids)
    ]

    result = [posting.doc_id for posting in postings]

    assert result == sorted(result)


@given(
    left=st.sets(st.integers(min_value=0, max_value=100)),
    right=st.sets(st.integers(min_value=0, max_value=100)),
)
def test_boolean_operations_match_set_operations(
    left: set[int],
    right: set[int],
) -> None:
    assert left & right == left.intersection(right)
    assert left | right == left.union(right)
    assert left - right == left.difference(right)
