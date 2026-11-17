
import argparse
import pickle
import json
import re
import time
import tracemalloc
from pathlib import Path

from findex.models import InvertedIndex, Posting, DocMeta


def intersect_merge(left, right):
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


def union_merge(left, right):
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


def difference_merge(left, right):
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


def get_document_ids(index, term):
    return [
        posting.doc_id
        for posting in index.postings.get(term, [])
    ]


def parse_query(index, query, engine):
    tokens = re.findall(r"\w+|[()]", query.upper())

    if engine == "merge":
        all_documents = sorted(index.documents)
        intersect = intersect_merge
        union = union_merge
        difference = difference_merge
    else:
        all_documents = set(index.documents)
        intersect = lambda a, b: sorted(set(a) & set(b))
        union = lambda a, b: sorted(set(a) | set(b))
        difference = lambda a, b: sorted(set(a) - set(b))

    position = 0

    def primary():
        nonlocal position

        if position >= len(tokens):
            return []

        if tokens[position] == "(":
            position += 1
            result = parse_or()
            if position < len(tokens) and tokens[position] == ")":
                position += 1
            return result

        if tokens[position] == "NOT":
            position += 1
            return difference(all_documents, primary())

        term = tokens[position]
        position += 1

        return get_document_ids(index, term.casefold())

    def parse_and():
        nonlocal position
        result = primary()

        while position < len(tokens):
            if tokens[position] == "AND":
                position += 1
                result = intersect(result, primary())
            elif tokens[position] not in ("OR", ")"):
                result = intersect(result, primary())
            else:
                break

        return result

    def parse_or():
        nonlocal position
        result = parse_and()

        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result = union(result, parse_and())

        return result

    return sorted(parse_or())


def load_index(filename):
    path = Path(filename)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        postings = {
            term: [Posting(doc_id, frequency) for doc_id, frequency in values]
            for term, values in data["postings"].items()
        }

        documents = {
            int(doc_id): DocMeta(**doc)
            for doc_id, doc in data["documents"].items()
        }

        return InvertedIndex(postings, documents)

    # Never load pickle files from untrusted sources:
    # deserialization can execute arbitrary code.
    with open(path, "rb") as file:
        return pickle.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Search an inverted index."
    )
    parser.add_argument("index")
    parser.add_argument("query")
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
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
