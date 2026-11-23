import html
import logging
import math
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Path as ApiPath, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from findex.lab3 import BM25, TfIdf, open_index, search
from findex.web.config import Settings
from findex.web.dependencies import get_index
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger("findex.web")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def perform_search(
    index: Any,
    query: str,
    k: int,
    scorer_name: str,
    page: int,
) -> SearchResponse:
    scorer = TfIdf() if scorer_name == "tfidf" else BM25()

    all_results = search(
        index,
        query,
        scorer=scorer,
        top_k=max(len(index.documents), 1),
    )

    total = len(all_results)
    pages = math.ceil(total / k) if total else 0
    start = (page - 1) * k
    page_results = all_results[start : start + k]

    return SearchResponse(
        query=query,
        scorer=scorer_name,
        k=k,
        page=page,
        total=total,
        pages=pages,
        results=[
            SearchHit(
                doc_id=result.doc_id,
                filename=result.filename,
                score=result.score,
                snippet=result.snippet,
            )
            for result in page_results
        ],
    )


def highlighted_snippet(snippet: str) -> Markup:
    escaped = html.escape(snippet)
    highlighted = re.sub(
        r"\*\*(.*?)\*\*",
        r"<mark>\1</mark>",
        escaped,
    )
    return Markup(highlighted)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.index = None

        try:
            with open_index(app_settings.index_path) as loaded_index:
                application.state.index = loaded_index

            logger.info(
                "Search index loaded from %s",
                app_settings.index_path,
            )
        except (OSError, ValueError, TypeError, EOFError) as error:
            logger.error("Could not load search index: %s", error)

        yield

        application.state.index = None

    application = FastAPI(
        title="Findex Search",
        description="Web API for the Findex full-text search engine.",
        version="0.7.0",
        lifespan=lifespan,
    )

    application.state.index = None
    application.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ):
        request_id = getattr(request.state, "request_id", "unknown")

        logger.exception(
            "Unexpected error. request_id=%s",
            request_id,
            exc_info=error,
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    @application.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        q: str = "",
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: str = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
        index: Any = Depends(get_index),
    ):
        if scorer not in ("bm25", "tfidf"):
            scorer = "bm25"

        response = None
        error = None

        if q.strip():
            try:
                response = perform_search(
                    index,
                    q,
                    k,
                    scorer,
                    page,
                )
            except ValueError as exception:
                error = str(exception)

        results = []

        if response is not None:
            for result in response.results:
                results.append(
                    {
                        "doc_id": result.doc_id,
                        "filename": result.filename,
                        "score": result.score,
                        "snippet_html": highlighted_snippet(result.snippet),
                    }
                )

        return templates.TemplateResponse(
            request=request,
            name="search.html",
            context={
                "query": q,
                "k": k,
                "scorer": scorer,
                "page": page,
                "response": response,
                "results": results,
                "error": error,
            },
        )

    @application.get("/search", response_model=SearchResponse)
    def search_endpoint(
        q: Annotated[str, Query(min_length=1, max_length=500)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: str = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
        index: Any = Depends(get_index),
    ):
        if scorer not in ("bm25", "tfidf"):
            raise HTTPException(
                status_code=422,
                detail="scorer must be 'bm25' or 'tfidf'",
            )

        try:
            return perform_search(index, q, k, scorer, page)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get(
        "/docs/{doc_id}",
        response_model=DocumentResponse,
    )
    def document_endpoint(
        doc_id: Annotated[int, ApiPath(ge=0)],
        index: Any = Depends(get_index),
    ):
        document = index.documents.get(doc_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        return DocumentResponse(
            doc_id=document.doc_id,
            filename=document.filename,
            length=document.length,
            text=document.text,
        )

    @application.get("/document/{doc_id}", response_class=HTMLResponse)
    def document_page(
        request: Request,
        doc_id: Annotated[int, ApiPath(ge=0)],
        index: Any = Depends(get_index),
    ):
        document = index.documents.get(doc_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={"document": document},
        )

    @application.get("/stats", response_model=StatsResponse)
    def stats_endpoint(index: Any = Depends(get_index)):
        return StatsResponse(
            documents=index.num_docs,
            unique_terms=len(index),
            average_document_length=round(index.avg_doc_length, 2),
        )

    @application.get("/health", response_model=HealthResponse)
    def health_endpoint(request: Request):
        if getattr(request.app.state, "index", None) is None:
            raise HTTPException(
                status_code=503,
                detail="Search index is not loaded",
            )

        return HealthResponse(status="ok")

    return application


app = create_app()
