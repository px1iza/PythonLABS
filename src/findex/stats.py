import argparse
import tracemalloc
from collections import Counter
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def collect_statistics(directory: str | Path) -> dict:
    """Collect corpus statistics in one pass."""
    document_count = 0
    token_count = 0
    frequencies = Counter()

    for name, text in iter_documents(directory):
        document_count += 1

        for token in tokenize(text):
            frequencies[token] += 1
            token_count += 1

    return {
        "documents": document_count,
        "tokens": token_count,
        "vocabulary": len(frequencies),
        "top_terms": frequencies.most_common(50),
    }


def collect_statistics_eager(directory: str | Path) -> dict:
    """Greedy version: loads all documents and tokens into lists."""
    documents = list(iter_documents(directory))
    all_tokens = []

    for name, text in documents:
        all_tokens.extend(tokenize(text))

    frequencies = Counter(all_tokens)

    return {
        "documents": len(documents),
        "tokens": len(all_tokens),
        "vocabulary": len(frequencies),
        "top_terms": frequencies.most_common(50),
    }


def measure(function, directory: str | Path) -> tuple[dict, float, float]:
    """Measure execution time and peak memory."""
    tracemalloc.start()
    start = perf_counter()

    result = function(directory)

    elapsed = perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed, peak / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Analyze a text corpus.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="data",
        help="Directory containing .txt documents",
    )
    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.is_dir():
        parser.error(f"Directory does not exist: {directory}")

    lazy, lazy_time, lazy_memory = measure(
        collect_statistics, directory
    )
    eager, eager_time, eager_memory = measure(
        collect_statistics_eager, directory
    )

    print("\nCORPUS STATISTICS")
    print(f"Documents: {lazy['documents']}")
    print(f"Tokens: {lazy['tokens']}")
    print(f"Vocabulary: {lazy['vocabulary']}")

    print("\nTOP 50 TERMS")
    for word, count in lazy["top_terms"]:
        print(f"{word}: {count}")

    print("\nPERFORMANCE COMPARISON")
    print(f"{'Method':<12} {'Time (s)':<15} {'Peak memory (MB)'}")
    print("-" * 43)
    print(f"{'Lazy':<12} {lazy_time:<15.6f} {lazy_memory:.4f}")
    print(f"{'Eager':<12} {eager_time:<15.6f} {eager_memory:.4f}")

    if lazy["documents"] != eager["documents"]:
        raise RuntimeError("Document counts do not match.")
    if lazy["tokens"] != eager["tokens"]:
        raise RuntimeError("Token counts do not match.")
    if lazy["top_terms"] != eager["top_terms"]:
        raise RuntimeError("Term frequencies do not match.")


if __name__ == "__main__":
    main()
