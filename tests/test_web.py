from fastapi.testclient import TestClient
import pytest

from findex.lab3 import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.config import Settings
from findex.web.dependencies import get_index


@pytest.fixture
def test_index():
    documents = {
        0: DocMeta(
            doc_id=0,
            filename="python.txt",
            length=4,
            text="Python is a programming language.",
        ),
        1: DocMeta(
            doc_id=1,
            filename="other.txt",
            length=3,
            text="Python is useful.",
        ),
    }

    postings = {
        "python": [
            Posting(doc_id=0, frequency=1, positions=(0,)),
            Posting(doc_id=1, frequency=1, positions=(0,)),
        ]
    }

    return Index(postings, documents)


@pytest.fixture
def client(tmp_path, test_index):
    settings = Settings(index_path=tmp_path / "missing-index.bin")
    app = create_app(settings)

    app.dependency_overrides[get_index] = lambda: test_index

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_search_success(client):
    response = client.get("/search", params={"q": "python"})

    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] in {
        "python.txt",
        "other.txt",
    }


def test_search_validation_returns_422(client):
    response = client.get("/search", params={"q": "python", "k": 0})

    assert response.status_code == 422


def test_unknown_document_returns_404(client):
    response = client.get("/docs/999")

    assert response.status_code == 404


def test_document_success(client):
    response = client.get("/docs/0")

    assert response.status_code == 200
    assert response.json()["filename"] == "python.txt"


def test_health_returns_503_when_index_is_not_loaded(tmp_path):
    settings = Settings(index_path=tmp_path / "missing-index.bin")
    app = create_app(settings)

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
