
from fastapi.testclient import TestClient

from app.main import app
import app.api.chat as chat_module
import app.api.memories as memory_module
import time


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint(monkeypatch):

    class MockDocument:

        page_content = "Hybrid retrieval combines dense and sparse search."

        metadata = {
            "source": "sample.txt",
            "parent_id": "parent_001",
            "relevance_score": 0.95,
        }

    class MockRetriever:

        def retrieve(self, query):

            return {
                "dense_count": 5,
                "sparse_count": 5,
                "fused_count": 5,
                "parent_count": 2,
                "reranked_count": 1,
                "retrieval_latency_ms": 10.0,
                "rerank_latency_ms": 20.0,
                "documents": [MockDocument()],
            }

    class MockPersona:

        def build_persona(self, user_id, query):
            return "Prefer concise Python responses."

    class MockGenerator:

        def generate(self, query, context, persona):
            time.sleep(0.02)
            return "Hybrid retrieval combines both search methods."

    class MockMemory:

        def save_conversation(self, **kwargs):
            return {"status": "ok"}

    monkeypatch.setattr(
        chat_module, "retriever", MockRetriever()
    )

    monkeypatch.setattr(
        chat_module, "persona_manager", MockPersona()
    )

    monkeypatch.setattr(
        chat_module, "generator", MockGenerator()
    )

    monkeypatch.setattr(
        chat_module, "memory", MockMemory()
    )

    response = client.post(
        "/api/chat",
        json={
            "query": "Explain hybrid retrieval",
            "user_id": "usr_99",
            "session_id": "session_test",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]

    metadata = data["metadata"]

    required_fields = [
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
    ]

    for field in required_fields:
        assert isinstance(metadata[field], (int, float))
        assert metadata[field] > 0

    assert len(data["sources"]) == 1


def test_memory_endpoint(monkeypatch):

    class MockMemory:

        def get_user_memories(self, user_id):
            return [
                {"memory": "User prefers Python."}
            ]

    monkeypatch.setattr(
        memory_module, "memory", MockMemory()
    )

    response = client.get(
        "/api/users/usr_99/memories"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "usr_99"
    assert data["count"] == 1
    assert "Python" in data["memories"][0]["memory"]
