
from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import perf_counter
from typing import Literal

from findex.lab3 import Index, Posting, extract_terms

ExecutorName = Literal["serial", "threads", "processes"]


def build_partial(
    file_paths: list[str],
    doc_id_start: int,
) -> Index:
    """Build a partial index from a list of file paths."""

    postings_by_term: dict[str, list[Posting]] = {}
    documents = {}

    for offset, file_name in enumerate(file_paths):
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")

        doc_id = doc_id_start + offset
        terms = extract_terms(text)

        documents[doc_id] = {
            "doc_id": doc_id,
            "filename": path.name,
            "length": len(terms),
            "text": text,
        }

        positions_by_term: dict[str, list[int]] = {}

        for position, term in enumerate(terms):
            positions_by_term.setdefault(term, []).append(position)

        for term, positions in positions_by_term.items():
            posting = Posting(
                doc_id=doc_id,
                frequency=len(positions),
                positions=tuple(positions),
            )

            postings_by_term.setdefault(term, []).append(posting)

    postings = {
        term: tuple(sorted(items, key=lambda item: item.doc_id))
        for term, items in postings_by_term.items()
    }

    return Index(postings, documents)


def merge(partial_indexes: list[Index]) -> Index:
    """Merge partial indexes into one complete index."""

    postings_by_term: dict[str, list[Posting]] = {}
    documents = {}

    for partial in partial_indexes:

        duplicate_ids = documents.keys() & partial.documents.keys()

        if duplicate_ids:
            raise ValueError(
                f"Duplicate document IDs: {sorted(duplicate_ids)}"
            )

        documents.update(partial.documents)

        for term, postings in partial.postings.items():
            postings_by_term.setdefault(term, []).extend(postings)

    merged_postings = {
        term: tuple(sorted(items, key=lambda item: item.doc_id))
        for term, items in postings_by_term.items()
    }

    return Index(
        merged_postings,
        dict(sorted(documents.items()))
    )


def _build_worker(task: tuple[list[str], int]) -> Index:
    """Module-level worker for multiprocessing spawn."""

    file_paths, doc_id_start = task

    return build_partial(file_paths, doc_id_start)


def split_files(
    file_paths: list[str],
    workers: int
) -> list[tuple[list[str], int]]:
    """Split files into chunks with unique document ID ranges."""

    if workers < 1:
        raise ValueError("Workers must be at least 1")

    if not file_paths:
        return []

    chunk_count = min(workers, len(file_paths))

    base_size, remainder = divmod(
        len(file_paths),
        chunk_count
    )

    chunks = []
    offset = 0

    for i in range(chunk_count):
        size = base_size + (1 if i < remainder else 0)

        chunk = file_paths[offset:offset + size]

        chunks.append((chunk, offset))

        offset += size

    return chunks


def _build_index_measured(
    directory: str | Path,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> tuple[Index, float]:

    if workers < 1:
        raise ValueError("Workers must be at least 1")

    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unknown executor: {executor}")

    file_paths = [
        str(path)
        for path in sorted(Path(directory).rglob("*.txt"))
        if path.is_file()
    ]

    if not file_paths:
        return Index({}, {}), 0.0

    if executor == "serial":
        partial_indexes = [
            build_partial(file_paths, 0)
        ]

    else:
        tasks = split_files(file_paths, workers)

        pool_class = (
            ThreadPoolExecutor
            if executor == "threads"
            else ProcessPoolExecutor
        )

        with pool_class(max_workers=workers) as pool:
            partial_indexes = list(
                pool.map(_build_worker, tasks)
            )

    merge_start = perf_counter()

    result = merge(partial_indexes)

    merge_time = perf_counter() - merge_start

    return result, merge_time


def build_index_parallel(
    directory: str | Path,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> Index:

    index, _ = _build_index_measured(
        directory,
        workers,
        executor
    )

    return index


def build_index_measured(
    directory: str | Path,
    workers: int = 1,
    executor: ExecutorName = "serial",
) -> tuple[Index, float]:

    return _build_index_measured(
        directory,
        workers,
        executor
    )


def available_worker_counts() -> list[int]:

    cpu_count = os.cpu_count() or 1

    return list(dict.fromkeys([
        1,
        2,
        4,
        8,
        cpu_count
    ]))
