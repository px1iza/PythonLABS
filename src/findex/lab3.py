
from __future__ import annotations

import logging
import math
import pickle
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
from heapq import nlargest
from pathlib import Path
from time import perf_counter
from typing import Protocol, runtime_checkable

from findex.corpus import iter_documents
from findex.tokenize import tokenize


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Decorators
# --------------------------------------------------

def timed(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        elapsed = perf_counter() - start
        logger.info("%s: %.6f seconds", function.__name__, elapsed)
        return result

    return wrapper


# --------------------------------------------------
# Data models
# --------------------------------------------------

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    filename: str
    length: int
    text: str


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    filename: str
    snippet: str


# --------------------------------------------------
# Index
# --------------------------------------------------

class Index(Mapping):
    __hash__ = object.__hash__

    def __init__(self, postings, documents):
        self.postings = postings
        self.documents = documents

    def __getitem__(self, term):
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self):
        return len(self.postings)

    def __contains__(self, term):
        return term in self.postings

    def __repr__(self):
        return (
            f"Index(terms={len(self)}, "
            f"documents={self.num_docs})"
        )

    @cached_property
    def num_docs(self):
        return len(self.documents)

    @cached_property
    def avg_doc_length(self):
        if not self.documents:
            return 0.0

        total = sum(
            document.length
            for document in self.documents.values()
        )
        return total / self.num_docs

    def df(self, term):
        return len(self.postings.get(term, ()))


# --------------------------------------------------
# Building and loading
# --------------------------------------------------

@timed
def build_index(directory):
    collected = {}
    documents = {}

    for doc_id, (filename, text) in enumerate(
        iter_documents(directory)
    ):
        tokens = tokenize(text)
        positions_by_term = {}

        for position, token in enumerate(tokens):
            positions_by_term.setdefault(token, []).append(position)

        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            filename=filename,
            length=len(tokens),
            text=text,
        )

        for term, positions in positions_by_term.items():
            collected.setdefault(term, []).append(
                Posting(
                    doc_id=doc_id,
                    frequency=len(positions),
                    positions=tuple(positions),
                )
            )

    return Index(collected, documents)


@timed
def save_index(index, path):
    with Path(path).open("wb") as file:
        pickle.dump(index, file)


@contextmanager
@timed
def open_index(path):
    file = Path(path).open("rb")

    try:
        index = pickle.load(file)
        yield index
    finally:
        file.close()


# --------------------------------------------------
# Query tree
# --------------------------------------------------

class QueryNode:
    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def evaluate(self, index):
        raise NotImplementedError


@dataclass(frozen=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index):
        return {
            posting.doc_id
            for posting in index.postings.get(self.value, ())
        }


@dataclass(frozen=True)
class Phrase(QueryNode):
    value: str

    def evaluate(self, index):
        words = tokenize(self.value)

        if not words:
            return set()

        candidate_ids = Term(words[0]).evaluate(index)

        for word in words[1:]:
            candidate_ids &= Term(word).evaluate(index)

        result = set()

        for doc_id in candidate_ids:
            positions = []

            for word in words:
                posting = next(
                    (
                        item
                        for item in index.postings[word]
                        if item.doc_id == doc_id
                    ),
                    None,
                )

                if posting is None:
                    positions = []
                    break

                positions.append(set(posting.positions))

            if not positions:
                continue

            for start in positions[0]:
                if all(
                    start + offset in positions[offset]
                    for offset in range(1, len(positions))
                ):
                    result.add(doc_id)
                    break

        return result


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index):
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index):
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(QueryNode):
    child: QueryNode

    def evaluate(self, index):
        all_documents = set(index.documents)
        return all_documents - self.child.evaluate(index)


# --------------------------------------------------
# Recursive descent parser
# --------------------------------------------------

class QueryParser:
    TOKEN_PATTERN = re.compile(
        r'"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()"]+',
        re.IGNORECASE,
    )

    def __init__(self, query):
        self.tokens = self.TOKEN_PATTERN.findall(query)
        self.position = 0

    def current(self):
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self):
        token = self.current()
        self.position += 1
        return token

    def parse(self):
        if not self.tokens:
            raise ValueError("Query cannot be empty")

        result = self.parse_or()

        if self.current() is not None:
            raise ValueError(
                f"Unexpected token: {self.current()}"
            )

        return result

    def parse_or(self):
        node = self.parse_and()

        while self.current() and self.current().upper() == "OR":
            self.consume()
            node = Or(node, self.parse_and())

        return node

    def parse_and(self):
        node = self.parse_unary()

        while self.current() is not None:
            token = self.current()

            if token.upper() == "OR" or token == ")":
                break

            if token.upper() == "AND":
                self.consume()
                node = And(node, self.parse_unary())
            else:
                # Adjacent expressions mean AND.
                node = And(node, self.parse_unary())

        return node

    def parse_unary(self):
        if self.current() and self.current().upper() == "NOT":
            self.consume()
            return Not(self.parse_unary())

        return self.parse_atom()

    def parse_atom(self):
        token = self.current()

        if token is None:
            raise ValueError("Unexpected end of query")

        if token == "(":
            self.consume()
            node = self.parse_or()

            if self.current() != ")":
                raise ValueError("Missing closing parenthesis")

            self.consume()
            return node

        if token == ")":
            raise ValueError("Unexpected closing parenthesis")

        self.consume()

        if token.startswith('"') and token.endswith('"'):
            return Phrase(token[1:-1])

        words = tokenize(token)

        if not words:
            raise ValueError(f"Invalid query term: {token}")

        node = Term(words[0])

        for word in words[1:]:
            node = And(node, Term(word))

        return node


def parse_query(query):
    return QueryParser(query).parse()


# --------------------------------------------------
# Scoring protocol and implementations
# --------------------------------------------------

@runtime_checkable
class Scorer(Protocol):
    def score(self, index, doc_id, terms) -> float:
        ...


class TfIdf:
    def score(self, index, doc_id, terms):
        document = index.documents[doc_id]
        total = 0.0

        for term in set(terms):
            postings = index.postings.get(term, ())

            posting = next(
                (
                    item
                    for item in postings
                    if item.doc_id == doc_id
                ),
                None,
            )

            if posting is None:
                continue

            tf = posting.frequency / max(document.length, 1)
            idf = math.log(
                (index.num_docs + 1) / (index.df(term) + 1)
            ) + 1

            total += tf * idf

        return total


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def score(self, index, doc_id, terms):
        document = index.documents[doc_id]
        total = 0.0

        for term in set(terms):
            postings = index.postings.get(term, ())

            posting = next(
                (
                    item
                    for item in postings
                    if item.doc_id == doc_id
                ),
                None,
            )

            if posting is None:
                continue

            frequency = posting.frequency
            document_frequency = index.df(term)

            idf = math.log(
                1 + (
                    index.num_docs - document_frequency + 0.5
                ) / (
                    document_frequency + 0.5
                )
            )

            average_length = index.avg_doc_length or 1.0

            denominator = frequency + self.k1 * (
                1 - self.b
                + self.b * document.length / average_length
            )

            total += (
                idf
                * frequency
                * (self.k1 + 1)
                / denominator
            )

        return total


# --------------------------------------------------
# Query helpers and cache
# --------------------------------------------------

def extract_terms(node):
    if isinstance(node, Term):
        return [node.value]

    if isinstance(node, Phrase):
        return tokenize(node.value)

    if isinstance(node, (And, Or)):
        return extract_terms(node.left) + extract_terms(node.right)

    if isinstance(node, Not):
        return extract_terms(node.child)

    return []


@lru_cache(maxsize=256)
def cached_document_ids(index, query):
    tree = parse_query(query)
    return tuple(sorted(tree.evaluate(index)))


# --------------------------------------------------
# Snippets
# --------------------------------------------------

def make_snippet(text, terms, window=80):
    if not text:
        return ""

    matches = []

    for term in set(terms):
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        for match in pattern.finditer(text):
            matches.append(match.start())

    if not matches:
        return text[:window * 2]

    best_position = min(matches)

    start = max(0, best_position - window)
    end = min(len(text), best_position + window)

    snippet = text[start:end]

    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue

        snippet = re.sub(
            re.escape(term),
            lambda match: f"**{match.group(0)}**",
            snippet,
            flags=re.IGNORECASE,
        )

    if start > 0:
        snippet = "..." + snippet

    if end < len(text):
        snippet += "..."

    return snippet


# --------------------------------------------------
# Ranked search
# --------------------------------------------------

@timed
def search(index, query, scorer=None, top_k=5):
    if scorer is None:
        scorer = BM25()

    tree = parse_query(query)
    terms = extract_terms(tree)

    previous_hits = cached_document_ids.cache_info().hits
    document_ids = cached_document_ids(index, query)
    current_hits = cached_document_ids.cache_info().hits

    if current_hits > previous_hits:
        logger.info("Cache hit: %s", query)
    else:
        logger.info("Cache miss: %s", query)

    results = []

    for doc_id in document_ids:
        document = index.documents[doc_id]
        score = scorer.score(index, doc_id, terms)

        snippet = make_snippet(
            document.text,
            terms,
            window=80,
        )

        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                filename=document.filename,
                snippet=snippet,
            )
        )

    return nlargest(
        top_k,
        results,
        key=lambda result: result.score,
    )
