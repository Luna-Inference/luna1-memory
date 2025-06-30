"""Unit tests for AI Memory Layer API defined in main.py.

The tests monkey-patch `main.collection` with an in-memory fake so they run
without a running ChromaDB instance.
"""

from fastapi.testclient import TestClient
import pytest
from pathlib import Path
import uuid

import main  # The FastAPI app lives here


class FakeCollection:  # minimal substitute for ChromaDB collection
    def __init__(self):
        self._docs = {}  # id -> (doc, metadata)

    # ChromaDB `add` signature subset
    def add(self, ids, documents, metadatas=None):
        metadatas = metadatas or [{}] * len(ids)
        for i, _id in enumerate(ids):
            self._docs[_id] = (documents[i], metadatas[i])

    # delete by IDs, raises ValueError if not found (to mimic Chroma)
    def delete(self, ids):
        for _id in ids:
            if _id not in self._docs:
                raise ValueError("not found")
            del self._docs[_id]

    # highly simplified query: ignore semantics & filters
    def query(self, query_texts, n_results=5, where=None):
        where = where or {}
        # Naive filtering by equality on metadata keys
        filtered = [
            (_id, *self._docs[_id])
            for _id in self._docs
            if all(self._docs[_id][1].get(k) == v for k, v in where.items())
        ]
        # Return at most n_results
        docs_slice = filtered[:n_results]
        ids, docs, metas = zip(*docs_slice) if docs_slice else ([], [], [])
        distances = [[0.0 for _ in ids]]
        return {
            "ids": [list(ids)],
            "documents": [list(docs)],
            "metadatas": [list(metas)],
            "distances": distances,
        }


LOG_FILE = Path("tests/reports.log")

@pytest.fixture(autouse=True)
def use_fake_collection(monkeypatch):
    """Replace the real Chroma collection with an in-memory fake and clean reports log."""
    # Clear previous logs
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    fake = FakeCollection()
    monkeypatch.setattr(main, "collection", fake)
    yield


client = TestClient(main.app)


def test_add_memory_success():
    payload = {"content": "Alice likes apples."}
    resp = client.post("/memory", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["content"] == payload["content"]
    assert uuid.UUID(body["id"])  # valid UUID
    assert "createdAt" in body["metadata"]


def test_add_memory_validation():
    resp = client.post("/memory", json={})  # missing content
    assert resp.status_code == 400


def test_retrieve_memory():
    # Add two memories first
    m1 = client.post("/memory", json={"content": "Cats are cute.", "metadata": {"userId": "u1"}}).json()
    m2 = client.post("/memory", json={"content": "Dogs are loyal.", "metadata": {"userId": "u1"}}).json()

    # Retrieve with filter userId
    resp = client.post(
        "/memory/retrieve",
        json={"query": "pets", "top_k": 5, "filter": {"userId": "u1"}},
    )
    assert resp.status_code == 200
    results = resp.json()
    returned_ids = {item["id"] for item in results}
    assert {m1["id"], m2["id"]} <= returned_ids


def test_report_generation():
    # Trigger a memory add which should be logged
    client.post("/memory", json={"content": "Log me"})
    # Fetch twice to ensure log written
    reports = client.get("/reports", params={"limit": 10}).json()
    if not reports:  # retry once
        reports = client.get("/reports", params={"limit": 10}).json()
    assert reports
    entry = reports[-1]
    assert entry["path"] == "/memory"
    assert entry["status"] == "success"
    assert isinstance(entry["duration_ms"], int)


def test_remove_memory():
    mem = client.post("/memory", json={"content": "To be deleted."}).json()
    mem_id = mem["id"]

    # Delete first time
    resp = client.delete(f"/memory/{mem_id}")
    assert resp.status_code == 204

    # Deleting again should 404
    resp2 = client.delete(f"/memory/{mem_id}")
    assert resp2.status_code == 404
