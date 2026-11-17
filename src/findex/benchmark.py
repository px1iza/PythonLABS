
import argparse
import gc
import tracemalloc
from array import array
from time import perf_counter

from findex.models import PostingPlain, Posting
from findex.search import parse_query, load_index


def measure_memory(function, index):
    gc.collect()
    tracemalloc.start()

    result = function(index)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, current / 1024, peak / 1024


def ordinary_postings(index):
    return {
        term: [
            PostingPlain(item.doc_id, item.frequency)
            for item in postings
        ]
        for term, postings in index.postings.items()
    }


def slots_postings(index):
    return {
        term: [
            Posting(item.doc_id, item.frequency)
            for item in postings
        ]
        for term, postings in index.postings.items()
    }


def array_postings(index):
    result = {}

    # Перетворюємо ID документів на числові позиції для array('I').
    doc_id_positions = {
        str(doc_id): position
        for position, doc_id in enumerate(sorted(index.documents))
    }

    for term, postings in index.postings.items():
        doc_ids = array("I")
        frequencies = array("I")

        for item in postings:
            doc_ids.append(doc_id_positions[str(item.doc_id)])
            frequencies.append(item.frequency)

        result[term] = (doc_ids, frequencies)

    return result


def compare_memory(index):
    methods = [
        ("dataclass", ordinary_postings),
        ("slots=True", slots_postings),
        ("array('I')", array_postings),
    ]

    print("\nMEMORY COMPARISON")
    print(f"{'Method':<15} {'Current (KB)':<18} {'Peak (KB)':<15}")

    for name, function in methods:
        _result, current, peak = measure_memory(function, index)
        print(f"{name:<15} {current:<18.2f} {peak:<15.2f}")


def compare_search(index):
    terms = sorted(
        index.postings,
        key=lambda term: len(index.postings[term]),
        reverse=True,
    )

    if len(terms) < 4:
        print("Not enough terms for search benchmark.")
        return

    frequent = terms[:2]
    rare = terms[-2:]

    queries = [
        ("Frequent", f"{frequent[0]} AND {frequent[1]}"),
        ("Rare", f"{rare[0]} AND {rare[1]}"),
    ]

    print("\nMERGE VS SET SEARCH BENCHMARK")
    print(f"{'Query type':<15} {'Engine':<10} {'Time (s)':<15} {'Results'}")

    for query_type, query in queries:
        for engine in ("merge", "set"):
            start = perf_counter()
            results = parse_query(index, query, engine)
            elapsed = perf_counter() - start

            print(
                f"{query_type:<15} "
                f"{engine:<10} "
                f"{elapsed:<15.6f} "
                f"{len(results)}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark index memory and search."
    )
    parser.add_argument("index", help="Path to saved index")
    args = parser.parse_args()

    index = load_index(args.index)

    compare_memory(index)
    compare_search(index)


if __name__ == "__main__":
    main()
