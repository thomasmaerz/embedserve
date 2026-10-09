from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from embedserve.scheduler import CoordinatorState, ModelBusy
from embedserve.server import (
    NATIVE_DIMENSION,
    EmbeddingServerSettings,
    create_embedding_app,
    model_busy_response,
)


class FakeEncoder:
    model_id = "nomic-ai/nomic-embed-text-v1.5"
    device = "cuda"
    gpu_name = "Test GPU"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        assert batch_size == 2
        self.seen.extend(texts)
        return [
            [float(index + 1)] + [0.0] * (NATIVE_DIMENSION - 1)
            for index, _ in enumerate(texts)
        ]


@pytest.fixture
def settings() -> EmbeddingServerSettings:
    return EmbeddingServerSettings(api_key="a-secure-test-key", batch_size=2, max_items=2)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def client(settings: EmbeddingServerSettings, encoder: FakeEncoder) -> TestClient:
    return TestClient(create_embedding_app(settings, encoder))


def auth(settings: EmbeddingServerSettings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_key}"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/health", None),
        ("get", "/api/tags", None),
        ("post", "/api/embed", {"input": "hello"}),
        ("get", "/v1/models", None),
        ("post", "/v1/embeddings", {"input": "hello"}),
    ],
)
def test_every_endpoint_requires_authentication(
    client: TestClient, method: str, path: str, body: dict[str, str] | None
) -> None:
    response = client.request(method, path, json=body)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("value", ["Bearer wrong", "Basic value", "Bearer", "bearer wrong"])
def test_invalid_authorization_is_rejected(client: TestClient, value: str) -> None:
    assert client.get("/health", headers={"Authorization": value}).status_code == 401


def test_model_busy_response_is_retryable_and_machine_readable() -> None:
    response = model_busy_response(
        ModelBusy(
            loaded_model="nomic",
            requested_model="e5",
            state=CoordinatorState.SWITCH_RESERVED,
            retry_after_ms=1500,
        )
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert json.loads(response.body)["error"] == {
        "code": "MODEL_BUSY",
        "message": "Requested model is waiting for the active model to finish.",
        "retryable": True,
        "loaded_model": "nomic",
        "requested_model": "e5",
        "state": "SWITCH_RESERVED",
        "retry_after_ms": 1500,
    }


def test_key_is_not_logged(
    client: TestClient, settings: EmbeddingServerSettings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        assert client.get("/health", headers=auth(settings)).status_code == 200
    assert settings.api_key not in caplog.text
    assert "Authorization" not in caplog.text


def test_health_and_tags_preserve_identity(
    client: TestClient, settings: EmbeddingServerSettings
) -> None:
    health = client.get("/health", headers=auth(settings))
    tags = client.get("/api/tags", headers=auth(settings))
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "ollama_name": "nomic-embed-text:v1.5",
        "model_revision": settings.model_revision,
        "code_revision": settings.code_revision,
        "model_digest": settings.model_digest,
        "device": "cuda",
        "gpu": "Test GPU",
        "native_dimension": 768,
    }
    model = tags.json()["models"][0]
    assert model["name"] == "nomic-embed-text:v1.5"
    assert model["digest"] == settings.model_digest
    assert model["details"]["embedding_length"] == 768


def test_ollama_and_openai_shapes_preserve_order(
    client: TestClient, settings: EmbeddingServerSettings
) -> None:
    payload = {"model": settings.model_alias, "input": ["one", "two"], "truncate": True}
    ollama = client.post("/api/embed", headers=auth(settings), json=payload)
    openai = client.post("/v1/embeddings", headers=auth(settings), json=payload)
    assert ollama.status_code == 200
    assert ollama.json()["model"] == settings.model_alias
    assert [row[0] for row in ollama.json()["embeddings"]] == [1.0, 2.0]
    assert [item["index"] for item in openai.json()["data"]] == [0, 1]
    assert all(len(row) == 768 for row in ollama.json()["embeddings"])


def test_server_does_not_add_prefixes(
    client: TestClient, settings: EmbeddingServerSettings, encoder: FakeEncoder
) -> None:
    values = ["search_document: exact document", "search_query: exact query"]
    response = client.post("/api/embed", headers=auth(settings), json={"input": values})
    assert response.status_code == 200
    assert encoder.seen == values


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "other", "input": "one"},
        {"input": ["one", "two", "three"]},
        {"input": "one", "dimensions": 512},
        {"input": "one", "encoding_format": "base64"},
        {"input": "one", "truncate": False},
        {"input": []},
    ],
)
def test_server_rejects_invalid_requests(
    client: TestClient, settings: EmbeddingServerSettings, payload: dict[str, object]
) -> None:
    response = client.post("/api/embed", headers=auth(settings), json=payload)
    assert response.status_code in {400, 404}


def test_server_rejects_malformed_and_oversized_bodies(
    client: TestClient, settings: EmbeddingServerSettings
) -> None:
    malformed = client.post(
        "/api/embed",
        headers={**auth(settings), "Content-Type": "application/json"},
        content=b"{",
    )
    assert malformed.status_code == 400
    limited = EmbeddingServerSettings(
        api_key=settings.api_key, max_chars=512, max_body_bytes=600
    )
    limited_client = TestClient(create_embedding_app(limited, FakeEncoder()))
    oversized = limited_client.post(
        "/api/embed",
        headers=auth(settings),
        content=(chunk for chunk in [b"{" + b"x" * 400, b"y" * 400 + b"}"]),
    )
    assert oversized.status_code == 413


def test_settings_load_key_from_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "api-key"
    key_file.write_text("a-private-test-key\n", encoding="ascii")
    key_file.chmod(0o600)
    monkeypatch.setenv("EMBEDSERVE_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("EMBEDSERVE_API_KEY", raising=False)
    assert EmbeddingServerSettings.from_env().api_key == "a-private-test-key"


def test_settings_reject_insecure_key_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "api-key"
    key_file.write_text("an-insecure-test-key\n", encoding="ascii")
    key_file.chmod(0o644)
    monkeypatch.setenv("EMBEDSERVE_API_KEY_FILE", str(key_file))
    with pytest.raises(ValueError, match="group or other"):
        EmbeddingServerSettings.from_env()


def test_settings_reject_arbitrary_model() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        EmbeddingServerSettings(api_key="a-secure-test-key", model_id="other/model")
