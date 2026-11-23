"""NumPy optimization helpers for Findex."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PostingArrays:
    """Compact representation of inverted-index postings."""

    doc_ids: np.ndarray
    frequencies: np.ndarray

    @classmethod
    def from_postings(cls, postings):
        return cls(
            doc_ids=np.asarray(
                [posting.doc_id for posting in postings],
                dtype=np.int32,
            ),
            frequencies=np.asarray(
                [posting.frequency for posting in postings],
                dtype=np.int32,
            ),
        )


def document_lengths_to_array(documents):
    """Convert document lengths to an int32 NumPy array."""
    if not documents:
        return np.array([], dtype=np.int32)

    max_id = max(documents)
    lengths = np.zeros(max_id + 1, dtype=np.int32)

    for doc_id, document in documents.items():
        lengths[doc_id] = document.length

    return lengths


def top_k(scores, doc_ids, k=5):
    """Return the highest-scoring document IDs."""
    scores = np.asarray(scores, dtype=np.float64)
    doc_ids = np.asarray(doc_ids, dtype=np.int32)

    if len(scores) != len(doc_ids):
        raise ValueError("scores and doc_ids must have equal lengths")

    if k <= 0 or len(scores) == 0:
        return []

    k = min(k, len(scores))

    if k < len(scores):
        selected = np.argpartition(scores, -k)[-k:]
    else:
        selected = np.arange(len(scores))

    selected = selected[
        np.argsort(scores[selected], kind="stable")[::-1]
    ]

    return [
        (int(doc_ids[i]), float(scores[i]))
        for i in selected
    ]


def bm25_scores(
    frequencies,
    document_lengths,
    average_length,
    idf,
    k1=1.5,
    b=0.75,
):
    """Calculate BM25 scores with vectorized NumPy operations."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    document_lengths = np.asarray(
        document_lengths,
        dtype=np.float64,
    )

    if average_length <= 0:
        return np.zeros_like(frequencies)

    denominator = frequencies + k1 * (
        1 - b + b * document_lengths / average_length
    )

    return idf * (
        frequencies * (k1 + 1)
    ) / np.maximum(denominator, 1e-12)


def tfidf_scores(term_frequencies, idf):
    """Calculate TF-IDF scores with NumPy."""
    term_frequencies = np.asarray(
        term_frequencies,
        dtype=np.float64,
    )

    return term_frequencies * idf
