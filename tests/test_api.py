"""Automated API integration tests for CodeSage FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from codesage.api import app


@pytest.fixture(scope="module")
def client():
    """Create test client within application lifecycle context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Test /api/health endpoint returns 200 and valid index metadata."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["indexed_units"] > 0
    assert "models_loaded" in data
    assert "tfidf" in data["models_loaded"]


def test_models_endpoint(client: TestClient):
    """Test /api/models endpoint lists available retrieval engines."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 3
    model_ids = [m["id"] for m in models]
    assert "tfidf" in model_ids
    assert "semantic" in model_ids
    assert "code" in model_ids


def test_search_endpoint_tfidf(client: TestClient):
    """Test search endpoint with TF-IDF model."""
    payload = {
        "query": "merge multiple PDF files",
        "model": "tfidf",
        "top_k": 5,
        "rerank": False,
    }
    resp = client.post("/api/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "merge multiple PDF files"
    assert data["model"] == "tfidf"
    assert len(data["results"]) <= 5
    assert data["latency_ms"] >= 0.0

    if data["results"]:
        first = data["results"][0]
        assert "unit_id" in first
        assert "score" in first
        assert "rel_path" in first
        assert "preview" in first


def test_search_endpoint_with_graph_reranking(client: TestClient):
    """Test search endpoint with graph-context reranking enabled."""
    payload = {
        "query": "merge multiple PDF files",
        "model": "tfidf",
        "top_k": 5,
        "rerank": True,
    }
    resp = client.post("/api/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["rerank_enabled"] is True
    assert len(data["results"]) <= 5


def test_compare_endpoint(client: TestClient):
    """Test /api/compare endpoint executes multi-engine retrieval."""
    payload = {
        "query": "cleanup temporary files",
        "top_k": 3,
        "with_graph": True,
    }
    resp = client.post("/api/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "systems" in data
    assert "tfidf" in data["systems"]
    assert "semantic" in data["systems"]
    assert "code" in data["systems"]
    assert data["total_latency_ms"] > 0.0


def test_evaluations_endpoint(client: TestClient):
    """Test /api/evaluations endpoint serves recorded empirical data."""
    resp = client.get("/api/evaluations")
    assert resp.status_code == 200
    data = resp.json()
    # At least benchmark comparison or study results should be present
    assert "benchmark_comparison" in data
    assert "study_results" in data
    assert "graph_reranking_results" in data


def test_search_input_validation(client: TestClient):
    """Test input validation for search queries and parameters."""
    # Query too long (> 500 chars)
    long_query = "x" * 501
    resp = client.post("/api/search", json={"query": long_query, "model": "tfidf"})
    assert resp.status_code == 422

    # Empty query string
    resp = client.post("/api/search", json={"query": "", "model": "tfidf"})
    assert resp.status_code == 422

    # Invalid top_k (< 1)
    resp = client.post("/api/search", json={"query": "test", "top_k": 0})
    assert resp.status_code == 422

    # Invalid top_k (> 50)
    resp = client.post("/api/search", json={"query": "test", "top_k": 51})
    assert resp.status_code == 422

    # Invalid model
    resp = client.post("/api/search", json={"query": "test", "model": "non_existent_engine"})
    assert resp.status_code == 400


def test_unit_details_endpoint(client: TestClient):
    """Test /api/units/{unit_id} endpoint returns unit details and graph context."""
    # First get an existing unit ID from search
    search_resp = client.post("/api/search", json={"query": "PDF", "top_k": 1})
    results = search_resp.json().get("results", [])
    if results:
        unit_id = results[0]["unit_id"]
        resp = client.get(f"/api/units/{unit_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "unit" in data
        assert data["unit"]["unit_id"] == unit_id
        assert "graph_context" in data

    # Non-existent unit ID should return 404
    resp = client.get("/api/units/non_existent_unit_id_99999")
    assert resp.status_code == 404
