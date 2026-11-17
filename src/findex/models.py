
from dataclasses import dataclass


@dataclass(frozen=True)
class PostingPlain:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    filename: str
    length: int


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]]
    documents: dict[int, DocMeta]
