from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from embedserve.runtime import E5, NOMIC
from embedserve.scheduler import CoordinatorState, ModelBusy
from embedserve.server import (
    NATIVE_DIMENSION,
    EmbeddingServerSettings,
    create_embedding_app,
    model_busy_response,
)


class FakeRuntime:
    device = "cuda"
    gpu_name = "Test GPU"

    def __init__(self) -> None:
        self.loaded_model: str | None = NOMIC
        self.max_resident_models_observed = 1
        self.last_unload_allocated_bytes: int | None = None
        self.last_inference_failure: str | None = None
        self.inference_failure_count = 0
        self.seen: list[tuple[str, str]] = []
        self.events: list[tuple[str, str]] = []
        self.fail_load: set[str] = set()
        self.fail_encode = False
        self.encode_started: threading.Event | None = None
        self.allow_encode: threading.Event | None = None

    def load(self, model: str) -> None:
        self.events.append(("load", model))
        if model in self.fail_load:
            raise RuntimeError("synthetic load failure")
        assert self.loaded_model is None
        self.loaded_model = model

    def unload(self, model: str) -> None:
        self.events.append(("unload", model))
        assert self.loaded_model == model
        self.loaded_model = None
        self.last_unload_allocated_bytes = 0

    def encode(self, model: str, texts: list[str], batch_size: int) -> list[list[float]]:
        assert self.loaded_model == model
        assert batch_size == 2
        if self.encode_started is not None:
            self.encode_started.set()
        if self.allow_encode is not None:
            self.allow_encode.wait(timeout=5)
        if self.fail_encode:
            self.last_inference_failure = "RuntimeError"
            self.inference_failure_count += 1
            raise RuntimeError("synthetic inference failure")
        self.seen.extend((model, text) for text in texts)
        return [
            [float(index + 1)] + [0.0] * (NATIVE_DIMENSION - 1)
            for index, _ in enumerate(texts)
        ]


@pytest.fixture
def settings() -> EmbeddingServerSettings:
    return EmbeddingServerSettings(
        api_key="a-secure-test-key",
        batch_size=2,
        e5_batch_size=2,
        max_items=2,
        e5_max_items=2,
        minimum_residency_seconds=0,
    )


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def client(settings: EmbeddingServerSettings, runtime: FakeRuntime) -> TestClient:
    return TestClient(create_embedding_app(settings, runtime))


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
        ("post", "/embed", {"inputs": ["passage: hello"]}),
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
        "model": settings.model_id,
        "ollama_name": "nomic-embed-text:v1.5",
        "model_revision": settings.model_revision,
        "code_revision": settings.code_revision,
        "model_digest": settings.model_digest,
        "device": "cuda",
        "gpu": "Test GPU",
        "native_dimension": 768,
        "models": {
            "nomic": {
                "id": settings.model_id,
                "revision": settings.model_revision,
                "code_revision": settings.code_revision,
                "normalized": False,
                "max_seq_length": 2048,
            },
            "e5": {
                "id": settings.e5_model_id,
                "revision": settings.e5_model_revision,
                "normalized": True,
                "max_seq_length": 512,
            },
        },
        "scheduler": {
            "state": "READY",
            "loaded_model": "nomic",
            "active_requests": 0,
            "switch_in_progress": False,
            "single_residency_verified": True,
            "last_unload_allocated_bytes": None,
            "last_inference_failure": None,
            "inference_failure_count": 0,
        },
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
    client: TestClient, settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    values = ["search_document: exact document", "search_query: exact query"]
    response = client.post("/api/embed", headers=auth(settings), json={"input": values})
    assert response.status_code == 200
    assert runtime.seen == [(NOMIC, value) for value in values]


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
    limited_client = TestClient(create_embedding_app(limited, FakeRuntime()))
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
    with pytest.raises(ValueError, match="unsupported Nomic model"):
        EmbeddingServerSettings(api_key="a-secure-test-key", model_id="other/model")


def test_tei_contract_switches_to_e5_without_changing_inputs(
    client: TestClient, settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    inputs = ["passage: synthetic English role", "passage: synthetische Stelle"]
    first = client.post("/embed", headers=auth(settings), json={"inputs": inputs})
    assert first.status_code == 503
    assert first.headers["Retry-After"] == "2"
    assert first.json()["error"]["code"] == "MODEL_BUSY"
    assert first.json()["error"]["loaded_model"] == NOMIC
    assert first.json()["error"]["requested_model"] == E5

    second = client.post("/embed", headers=auth(settings), json={"inputs": inputs})
    assert second.status_code == 200
    assert isinstance(second.json(), list)
    assert len(second.json()) == 2
    assert all(len(vector) == 768 for vector in second.json())
    assert runtime.seen[-2:] == [(E5, text) for text in inputs]
    assert runtime.events == [("unload", NOMIC), ("load", E5)]
    assert runtime.loaded_model == E5
    assert runtime.last_unload_allocated_bytes == 0


def test_reverse_e5_to_nomic_handoff(
    client: TestClient, settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    inputs = ["passage: synthetic role"]
    assert client.post("/embed", headers=auth(settings), json={"inputs": inputs}).status_code == 503
    assert client.post("/embed", headers=auth(settings), json={"inputs": inputs}).status_code == 200

    payload = {"model": settings.model_alias, "input": ["search_query: synthetic role"]}
    first = client.post("/api/embed", headers=auth(settings), json=payload)
    assert first.status_code == 503
    second = client.post("/api/embed", headers=auth(settings), json=payload)
    assert second.status_code == 200
    assert runtime.events == [
        ("unload", NOMIC),
        ("load", E5),
        ("unload", E5),
        ("load", NOMIC),
    ]
    assert runtime.loaded_model == NOMIC


@pytest.mark.parametrize(
    "payload",
    [
        {"input": ["passage: wrong field"]},
        {"inputs": "passage: must be a list"},
        {"inputs": []},
        {"inputs": ["one", "two", "three"]},
        {"inputs": ["one"], "model": "arbitrary"},
    ],
)
def test_tei_rejects_invalid_body(
    client: TestClient, settings: EmbeddingServerSettings, payload: dict[str, object]
) -> None:
    assert client.post("/embed", headers=auth(settings), json=payload).status_code == 400


def test_e5_load_failure_restores_nomic(
    client: TestClient, settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    runtime.fail_load.add(E5)
    payload = {"inputs": ["passage: synthetic role"]}
    assert client.post("/embed", headers=auth(settings), json=payload).status_code == 503
    failed = client.post("/embed", headers=auth(settings), json=payload)
    assert failed.status_code == 503
    assert failed.json()["error"] == {
        "code": "MODEL_LOAD_FAILED",
        "message": "Requested model could not be loaded.",
        "retryable": True,
        "requested_model": E5,
        "restored_model": NOMIC,
    }
    assert runtime.loaded_model == NOMIC


@pytest.mark.asyncio
async def test_cancelled_http_request_waits_for_inference_cleanup(
    settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    runtime.encode_started = threading.Event()
    runtime.allow_encode = threading.Event()
    app = create_embedding_app(settings, runtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://embedserve"
    ) as http:
        request = asyncio.create_task(
            http.post(
                "/api/embed",
                headers=auth(settings),
                json={"input": ["search_query: synthetic cancellation"]},
            )
        )
        await asyncio.to_thread(runtime.encode_started.wait, 5)
        request.cancel()
        runtime.allow_encode.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        health = await http.get("/health", headers=auth(settings))
    assert health.json()["scheduler"]["active_requests"] == 0
    assert health.json()["scheduler"]["state"] == "READY"


def test_inference_failure_is_sanitized_and_releases_admission(
    client: TestClient, settings: EmbeddingServerSettings, runtime: FakeRuntime
) -> None:
    runtime.fail_encode = True
    failed = client.post(
        "/api/embed", headers=auth(settings), json={"input": ["search_query: synthetic"]}
    )
    assert failed.status_code == 500
    assert failed.json() == {
        "error": {
            "code": "INFERENCE_FAILED",
            "message": "Embedding inference failed.",
            "retryable": False,
        }
    }
    runtime.fail_encode = False
    health = client.get("/health", headers=auth(settings))
    assert health.json()["scheduler"]["active_requests"] == 0
