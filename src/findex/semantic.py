"""Semantic search helpers for Findex."""

from pathlib import Path

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


def normalize_vectors(vectors):
    """Normalize vectors to unit length."""
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return vectors / norms


def create_embeddings(texts, model_name=MODEL_NAME):
    """Create normalized text embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return normalize_vectors(vectors)


def save_embeddings(vectors, path):
    """Save embedding matrix to a NumPy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, normalize_vectors(vectors))


def load_embeddings(path):
    """Load embedding matrix from a NumPy file."""
    return np.load(path)


def semantic_search(query_vector, chunk_vectors, chunk_doc_ids, top_k=5):
    """Rank documents by their best matching chunk."""
    query_vector = normalize_vectors(
        np.asarray(query_vector).reshape(1, -1)
    )[0]

    chunk_vectors = normalize_vectors(chunk_vectors)
    chunk_doc_ids = np.asarray(chunk_doc_ids, dtype=np.int32)

    if len(chunk_vectors) != len(chunk_doc_ids):
        raise ValueError("Each chunk vector must have a document ID")

    if len(chunk_vectors) == 0 or top_k <= 0:
        return []

    similarities = chunk_vectors @ query_vector

    best_scores = {}

    for doc_id, score in zip(chunk_doc_ids, similarities):
        doc_id = int(doc_id)
        score = float(score)

        if doc_id not in best_scores or score > best_scores[doc_id]:
            best_scores[doc_id] = score

    results = sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return results[:top_k]


def reciprocal_rank_fusion(
    rankings,
    k=60,
    top_k=5,
):
    """Combine ranked document lists using Reciprocal Rank Fusion."""
    scores = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            doc_id = item[0] if isinstance(item, tuple) else item
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)

    results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return results[:top_k]
