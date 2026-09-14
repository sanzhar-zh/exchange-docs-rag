"""The HTTP layer, with generation stubbed out.

The case worth testing here is the failing one. A provider failure is the normal
way this endpoint breaks, and an unhandled exception is answered by Starlette's
own 500 handler, which sits outside the CORS middleware: the browser then rejects
the response as a CORS error and the page reports the server as unreachable. The
real cause - an exhausted quota - never reaches the user.
"""

import pytest
from fastapi.testclient import TestClient

import api

ORIGIN = {"Origin": "http://localhost:3000"}
QUESTION = {"question": "how do I place an order"}


@pytest.fixture
def client(monkeypatch):
    """Startup warms the retrievers, which would load the index and the embedding
    model. The warm-up itself is asserted separately."""
    monkeypatch.setattr(api, "retrieve", lambda query: [])
    with TestClient(api.app) as test_client:
        yield test_client


def fail_with(message: str):
    def raising(question: str) -> dict:
        raise RuntimeError(message)

    return raising


def test_health_reports_ok(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_startup_warms_the_retrievers(monkeypatch) -> None:
    """Without this the first user waits several seconds for the embedding model
    and the BM25 build, and reasonably concludes the service is broken."""
    warmed = []
    monkeypatch.setattr(api, "retrieve", lambda query: warmed.append(query) or [])

    with TestClient(api.app):
        pass

    assert warmed


def test_an_answer_is_returned_with_its_sources(client, monkeypatch) -> None:
    reply = {"answer": "Use /v5/order/create.", "sources": [], "provider": "test"}
    monkeypatch.setattr(api, "answer", lambda question: reply)
    response = client.post("/ask", json=QUESTION)

    assert response.status_code == 200
    assert response.json()["answer"] == "Use /v5/order/create."


def test_an_exhausted_quota_is_reported_as_such(client, monkeypatch) -> None:
    monkeypatch.setattr(api, "answer", fail_with("429 RESOURCE_EXHAUSTED"))
    response = client.post("/ask", json=QUESTION, headers=ORIGIN)

    assert response.status_code == 429
    assert "quota" in response.json()["detail"]


def test_any_other_provider_failure_is_a_bad_gateway(client, monkeypatch) -> None:
    monkeypatch.setattr(api, "answer", fail_with("connection reset by peer"))
    response = client.post("/ask", json=QUESTION, headers=ORIGIN)

    assert response.status_code == 502
    assert "connection reset" in response.json()["detail"]


@pytest.mark.parametrize("message", ["429 RESOURCE_EXHAUSTED", "connection reset"])
def test_a_failure_still_carries_the_cors_header(
    client, monkeypatch, message: str
) -> None:
    """The regression itself: without this header the browser discards the
    response and the page blames the network instead of the provider."""
    monkeypatch.setattr(api, "answer", fail_with(message))
    response = client.post("/ask", json=QUESTION, headers=ORIGIN)

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.parametrize("question", ["", "no", "x" * 501])
def test_a_question_outside_the_accepted_length_is_rejected(
    client, question: str
) -> None:
    assert client.post("/ask", json={"question": question}).status_code == 422
