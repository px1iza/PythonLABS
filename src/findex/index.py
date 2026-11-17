
import argparse
import json
import pickle
import tracemalloc
from pathlib import Path
from time import perf_counter
from collections import defaultdict
from array import array

from findex.corpus import iter_documents
from findex.tokenize import tokenize
from findex.models import Posting, DocMeta, InvertedIndex


def build_index(directory: str | Path) -> InvertedIndex:
    """Build an inverted index in one pass through the corpus."""
    collected = defaultdict(lambda: defaultdict(int))
    documents = {}

    for doc_id, (filename, text) in enumerate(iter_documents(directory)):
        document_length = 0

        for token in tokenize(text):
            collected[token][doc_id] += 1
            document_length += 1

        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            filename=filename,
            length=document_length,
        )

    postings = {}

    for term, doc_frequencies in collected.items():
        postings[term] = [
            Posting(doc_id, frequency)
            for doc_id, frequency in sorted(doc_frequencies.items())
        ]

    return InvertedIndex(postings, documents)


def save_pickle(index: InvertedIndex, filename: str | Path) -> float:
    start = perf_counter()

    with open(filename, "wb") as file:
        pickle.dump(index, file)

    return perf_counter() - start


def load_pickle(filename: str | Path) -> tuple[InvertedIndex, float]:
    start = perf_counter()

    with open(filename, "rb") as file:
        index = pickle.load(file)

    return index, perf_counter() - start


def save_json(index: InvertedIndex, filename: str | Path) -> float:
    data = {
        "postings": {
            term: [
                [posting.doc_id, posting.frequency]
                for posting in postings
            ]
            for term, postings in index.postings.items()
        },
        "documents": {
            str(doc_id): {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "length": doc.length,
            }
            for doc_id, doc in index.documents.items()
        },
    }

    start = perf_counter()

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

    return perf_counter() - start


def load_json(filename: str | Path) -> tuple[InvertedIndex, float]:
    start = perf_counter()

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    postings = {
        term: [
            Posting(doc_id, frequency)
            for doc_id, frequency in items
        ]
        for term, items in data["postings"].items()
    }

    documents = {
        int(doc_id): DocMeta(**doc)
        for doc_id, doc in data["documents"].items()
    }

    return InvertedIndex(postings, documents), perf_counter() - start


def measure_memory(function, directory):
    tracemalloc.start()
    result = function(directory)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, peak / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Build and save an inverted index."
    )
    parser.add_argument("directory")
    parser.add_argument("--out", default="index.bin")
    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.is_dir():
        parser.error(f"Directory does not exist: {directory}")

    start = perf_counter()
    index, memory = measure_memory(build_index, directory)
    build_time = perf_counter() - start

    pickle_time = save_pickle(index, args.out)
    json_file = Path(args.out).with_suffix(".json")
    json_time = save_json(index, json_file)

    pickle_size = Path(args.out).stat().st_size
    json_size = json_file.stat().st_size

    print(f"Documents: {len(index.documents)}")
    print(f"Terms: {len(index.postings)}")
    print(f"Index saved to: {args.out}")
    print(f"JSON saved to: {json_file}")
    print(f"Build time: {build_time:.6f} seconds")
    print(f"Peak memory: {memory:.4f} MB")
    print(f"Pickle save time: {pickle_time:.6f} seconds")
    print(f"JSON save time: {json_time:.6f} seconds")
    print(f"Pickle size: {pickle_size} bytes")
    print(f"JSON size: {json_size} bytes")


if __name__ == "__main__":
    main()
