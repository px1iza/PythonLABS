
from pathlib import Path

import pytest

from findex.lab3 import build_index
from findex.parallel import (
    build_index_parallel,
    build_partial,
    merge,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:

    (tmp_path / "a.txt").write_text(
        "Python programming language",
        encoding="utf-8"
    )

    (tmp_path / "b.txt").write_text(
        "Python has generators",
        encoding="utf-8"
    )

    (tmp_path / "nested").mkdir()

    (tmp_path / "nested" / "c.txt").write_text(
        "Programming with Python",
        encoding="utf-8"
    )

    return tmp_path


def test_serial_merge_matches_lab4(corpus):

    old_index = build_index(corpus)

    paths = [
        str(path)
        for path in sorted(corpus.rglob("*.txt"))
    ]

    new_index = merge([
        build_partial(paths, 0)
    ])

    assert new_index.postings == old_index.postings

    assert new_index.documents == old_index.documents


@pytest.mark.parametrize(
    "executor",
    ["serial", "threads", "processes"]
)
def test_executors_build_same_index(
    corpus,
    executor
):

    expected = build_index(corpus)

    actual = build_index_parallel(
        corpus,
        workers=2,
        executor=executor
    )

    assert actual.postings == expected.postings

    assert actual.documents == expected.documents


def test_doc_ids_are_unique(corpus):

    index = build_index_parallel(
        corpus,
        workers=8,
        executor="threads"
    )

    ids = list(index.documents)

    assert len(ids) == len(set(ids))

    assert ids == [0, 1, 2]


def test_merge_rejects_duplicate_ids(corpus):

    paths = [
        str(path)
        for path in sorted(corpus.rglob("*.txt"))
    ]

    first = build_partial(
        paths[:1],
        0
    )

    second = build_partial(
        paths[1:2],
        0
    )

    with pytest.raises(
        ValueError,
        match="Duplicate document IDs"
    ):

        merge([first, second])


def test_invalid_workers_are_rejected(corpus):

    with pytest.raises(ValueError):

        build_index_parallel(
            corpus,
            workers=0,
            executor="threads"
        )
