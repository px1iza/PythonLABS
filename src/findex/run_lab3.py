
import argparse

from findex.lab3 import (
    BM25,
    TfIdf,
    build_index,
    open_index,
    save_index,
    search,
)


def main():
    parser = argparse.ArgumentParser(
        description="Lab 3: Ranked search engine"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build a new index",
    )

    parser.add_argument(
        "--data",
        default="data/",
        help="Path to document corpus",
    )

    parser.add_argument(
        "--index",
        default="index_lab3.bin",
        help="Path to saved index",
    )

    parser.add_argument(
        "--query",
        default='python AND (async OR await) NOT java "event loop"',
        help="Search query",
    )

    parser.add_argument(
        "--scorer",
        choices=["tfidf", "bm25"],
        default="bm25",
        help="Ranking algorithm",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of results",
    )

    args = parser.parse_args()

    if args.build:
        index = build_index(args.data)
        save_index(index, args.index)

        print(f"Index saved to: {args.index}")
        print(f"Documents: {index.num_docs}")
        print(f"Terms: {len(index)}")
        print()

    scorer = TfIdf() if args.scorer == "tfidf" else BM25()

    with open_index(args.index) as index:
        print(f"Query: {args.query}")
        print(f"Scorer: {args.scorer}")
        print(f"Documents in index: {index.num_docs}")
        print()

        results = search(
            index=index,
            query=args.query,
            scorer=scorer,
            top_k=args.top,
        )

        if not results:
            print("No documents found.")
            return

        print("SEARCH RESULTS")
        print("-" * 60)

        for number, result in enumerate(results, start=1):
            print(f"{number}. {result.filename}")
            print(f"   Score: {result.score:.6f}")
            print(f"   Document ID: {result.doc_id}")
            print(f"   Snippet: {result.snippet}")
            print()


if __name__ == "__main__":
    main()
