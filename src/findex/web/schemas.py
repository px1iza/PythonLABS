from pydantic import BaseModel


class SearchHit(BaseModel):
    doc_id: int
    filename: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: str
    k: int
    page: int
    total: int
    pages: int
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    doc_id: int
    filename: str
    length: int
    text: str


class StatsResponse(BaseModel):
    documents: int
    unique_terms: int
    average_document_length: float


class HealthResponse(BaseModel):
    status: str
