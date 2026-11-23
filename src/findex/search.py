import argparse
import pickle
import re
import time
import tracemalloc
from pathlib import Path

from findex.models import InvertedIndex


def intersect_merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1

    return result


def union_merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def difference_merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = 0
    j = 0

    while i < len(left):
        if j >= len(right):
            result.extend(left[i:])
            break

        if left[i] == right[j]:
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            j += 1

    return result


def get_document_ids(index: InvertedIndex, term: str) -> list[int]:
    return [
        posting.doc_id
        for posting in index.postings.get(term, [])
    ]


def search_merge(index: InvertedIndex, query: str) -> list[int]:
    tokens = re.findall(r"\w+|[()]", query.casefold())

    if not tokens:
        return []

    position = 0

    def parse_primary():
        nonlocal position

        if position >= len(tokens):
            return []

        if tokens[position] == "(":
            position += 1
            result = parse_or()

            if position < len(tokens) and tokens[position] == ")":
                position += 1

            return result

        if tokens[position] == "not":
            position += 1
            return difference_merge(
                sorted(index.documents),
                parse_primary(),
            )

        term = tokens[position]
        position += 1
        return get_document_ids(index, term)

    def parse_and():
        nonlocal position
        result = parse_primary()

        while position < len(tokens):
            if tokens[position] == "and":
                position += 1
                result = intersect_merge(result, parse_primary())
            elif tokens[position] not in ("or", ")"):
                result = intersect_merge(result, parse_primary())
            else:
                break

        return result

    def parse_or():
        nonlocal position
        result = parse_and()

        while position < len(tokens) and tokens[position] == "or":
            position += 1
            result = union_merge(result, parse_and())

        return result

    return parse_or()


def search_set(index: InvertedIndex, query: str) -> list[int]:
    tokens = re.findall(r"\w+|[()]", query.casefold())

    if not tokens:
        return []

    position = 0
    all_documents = set(index.documents)

    def parse_primary():
        nonlocal position

        if position >= len(tokens):
            return set()

        if tokens[position] == "(":
            position += 1
            result = parse_or()

            if position < len(tokens) and tokens[position] == ")":
                position += 1

            return result

        if tokens[position] == "not":
            position += 1
            return all_documents - parse_primary()

        term = tokens[position]
        position += 1
        return set(get_document_ids(index, term))

    def parse_and():
        nonlocal position
        result = parse_primary()

        while position < len(tokens):
            if tokens[position] == "and":
                position += 1
                result &= parse_primary()
            elif tokens[position] not in ("or", ")"):
                result &= parse_primary()
            else:
                break

        return result

    def parse_or():
        nonlocal position
        result = parse_and()

        while position < len(tokens) and tokens[position] == "or":
            position += 1
            result |= parse_and()

        return result

    return sorted(parse_or())


def parse_query(
    index: InvertedIndex,
    query: str,
    engine: str = "merge",
) -> list[int]:
    if engine == "merge":
        return search_merge(index, query)

    if engine == "set":
        return search_set(index, query)

    raise ValueError("Engine must be 'merge' or 'set'.")


def load_index(filename: str | Path) -> InvertedIndex:
    """Load a pickle index. Never load pickle files from untrusted sources."""
    with open(filename, "rb") as file:
        return pickle.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Search an inverted index."
    )
    parser.add_argument("index", help="Path to the saved index")
    parser.add_argument("query", help="Boolean search query")
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Search algorithm",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    index = load_index(args.index)
    results = parse_query(index, args.query, args.engine)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query: {args.query}")
    print(f"Engine: {args.engine}")
    print(f"Found documents: {len(results)}")

    for doc_id in results:
        document = index.documents[doc_id]
        print(
            f"{doc_id}: {document.filename} "
            f"({document.length} tokens)"
        )

    print(f"Time: {elapsed:.6f} seconds")
    print(f"Peak memory: {peak / (1024 * 1024):.4f} MB")


if __name__ == "__main__":
    main()