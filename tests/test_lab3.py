
import pytest

from findex.lab3 import (
    And,
    BM25,
    DocMeta,
    Index,
    Not,
    Or,
    Phrase,
    Posting,
    Term,
    TfIdf,
    parse_query,
)


@pytest.fixture
def sample_index():
    documents = {
        0: DocMeta(
            0,
            "doc1.txt",
            4,
            "python async event loop",
        ),
        1: DocMeta(
            1,
            "doc2.txt",
            4,
            "python java event loop",
        ),
        2: DocMeta(
            2,
            "doc3.txt",
            3,
            "async await python",
        ),
    }

    postings = {
        "python": [
            Posting(0, 1, (0,)),
            Posting(1, 1, (0,)),
            Posting(2, 1, (2,)),
        ],
        "async": [
            Posting(0, 1, (1,)),
            Posting(2, 1, (0,)),
        ],
        "event": [
            Posting(0, 1, (2,)),
            Posting(1, 1, (2,)),
        ],
        "loop": [
            Posting(0, 1, (3,)),
            Posting(1, 1, (3,)),
        ],
        "java": [
            Posting(1, 1, (1,)),
        ],
        "await": [
            Posting(2, 1, (1,)),
        ],
    }

    return Index(postings, documents)


# Index data model

def test_index_length(sample_index):
    assert len(sample_index) == 6


def test_index_contains(sample_index):
    assert "python" in sample_index
    assert "unknown" not in sample_index


def test_index_getitem(sample_index):
    assert len(sample_index["python"]) == 3


def test_index_iteration(sample_index):
    assert set(iter(sample_index)) == set(sample_index.postings)


def test_index_statistics(sample_index):
    assert sample_index.num_docs == 3
    assert sample_index.avg_doc_length == pytest.approx(11 / 3)
    assert sample_index.df("python") == 3
    assert sample_index.df("java") == 1


def test_index_repr(sample_index):
    assert "Index" in repr(sample_index)


# Query parser

def test_parse_term():
    assert parse_query("python") == Term("python")


def test_parse_and():
    assert parse_query("python AND java") == And(
        Term("python"),
        Term("java"),
    )


def test_parse_or():
    assert parse_query("python OR java") == Or(
        Term("python"),
        Term("java"),
    )


def test_parse_not():
    assert parse_query("NOT java") == Not(Term("java"))


def test_parse_parentheses():
    assert parse_query("python AND (async OR await)") == And(
        Term("python"),
        Or(
            Term("async"),
            Term("await"),
        ),
    )


def test_parse_phrase():
    assert parse_query('"event loop"') == Phrase("event loop")


def test_parse_complex_query():
    expected = And(
        Term("python"),
        And(
            Or(
                Term("async"),
                Term("await"),
            ),
            Not(
                And(
                    Term("java"),
                    Phrase("event loop"),
                )
            ),
        ),
    )

    assert parse_query(
        'python AND (async OR await) NOT java "event loop"'
    ) == expected


def test_empty_query():
    with pytest.raises(ValueError):
        parse_query("")


# Query evaluation

def test_term_evaluation(sample_index):
    assert Term("python").evaluate(sample_index) == {0, 1, 2}


def test_and_evaluation(sample_index):
    tree = And(Term("python"), Term("java"))
    assert tree.evaluate(sample_index) == {1}


def test_or_evaluation(sample_index):
    tree = Or(Term("java"), Term("async"))
    assert tree.evaluate(sample_index) == {0, 1, 2}


def test_not_evaluation(sample_index):
    tree = Not(Term("java"))
    assert tree.evaluate(sample_index) == {0, 2}


def test_phrase_evaluation(sample_index):
    assert Phrase("event loop").evaluate(sample_index) == {0, 1}


def test_missing_phrase(sample_index):
    assert Phrase("python loop").evaluate(sample_index) == set()


# Scorers

def test_tfidf_returns_number(sample_index):
    score = TfIdf().score(
        sample_index,
        0,
        ["python", "async"],
    )

    assert isinstance(score, float)
    assert score > 0


def test_bm25_returns_number(sample_index):
    score = BM25().score(
        sample_index,
        0,
        ["python", "async"],
    )

    assert isinstance(score, float)
    assert score > 0


def test_bm25_parameters():
    scorer = BM25(k1=1.5, b=0.75)

    assert scorer.k1 == 1.5
    assert scorer.b == 0.75


def test_scorers_are_interchangeable(sample_index):
    for scorer in (TfIdf(), BM25()):
        score = scorer.score(
            sample_index,
            0,
            ["python"],
        )

        assert score > 0


# Operators

def test_query_node_operators():
    python = Term("python")
    java = Term("java")

    assert python & java == And(python, java)
    assert python | java == Or(python, java)
    assert ~python == Not(python)
