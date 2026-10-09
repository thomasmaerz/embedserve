from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from embedserve.runtime import E5, NOMIC, SentenceTransformerRuntime
from embedserve.server import EmbeddingServerSettings


class FakeValues:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeModel:
    def __init__(self, model_id: str, kwargs: dict[str, object]) -> None:
        self.model_id = model_id
        self.kwargs = kwargs
        self.max_seq_length = 0
        self.encode_calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 768

    def encode(self, texts: list[str], **kwargs: object) -> FakeValues:
        self.encode_calls.append({"texts": texts, **kwargs})
        return FakeValues([[1.0] * 768 for _ in texts])


class FakeCuda:
    def __init__(self) -> None:
        self.allocated = 0
        self.synchronize_calls = 0
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def get_device_name(self, _: int) -> str:
        return "Synthetic GPU"

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def memory_allocated(self, _: int) -> int:
        return self.allocated


class FakeFactory:
    def __init__(self) -> None:
        self.models: list[FakeModel] = []

    def __call__(self, model_id: str, **kwargs: object) -> FakeModel:
        model = FakeModel(model_id, kwargs)
        self.models.append(model)
        return model


def make_runtime(
    monkeypatch: pytest.MonkeyPatch, *, residual_limit: int = 16_777_216
) -> tuple[SentenceTransformerRuntime, FakeCuda, FakeFactory]:
    cuda = FakeCuda()
    factory = FakeFactory()
    torch = SimpleNamespace(cuda=cuda, inference_mode=nullcontext)
    sentence_transformers = SimpleNamespace(SentenceTransformer=factory)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    settings = EmbeddingServerSettings(
        api_key="a-secure-test-key", max_residual_cuda_bytes=residual_limit
    )
    return SentenceTransformerRuntime(settings), cuda, factory


def test_runtime_switches_specs_without_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, cuda, factory = make_runtime(monkeypatch)
    runtime.load(NOMIC)
    nomic = factory.models[-1]
    assert nomic.max_seq_length == 2048
    assert nomic.kwargs["trust_remote_code"] is True
    assert "code_revision" in nomic.kwargs["model_kwargs"]  # type: ignore[operator]
    runtime.encode(NOMIC, ["search_query: synthetic"], 2)
    assert nomic.encode_calls[-1]["normalize_embeddings"] is False
    with pytest.raises(RuntimeError, match="already resident"):
        runtime.load(E5)

    runtime.unload(NOMIC)
    assert runtime.loaded_model is None
    assert runtime.last_unload_allocated_bytes == 0
    assert cuda.synchronize_calls == 1
    assert cuda.empty_cache_calls == 1

    runtime.load(E5)
    e5 = factory.models[-1]
    assert e5.max_seq_length == 512
    assert e5.kwargs["trust_remote_code"] is False
    assert "model_kwargs" not in e5.kwargs
    runtime.encode(E5, ["passage: synthetic"], 2)
    assert e5.encode_calls[-1]["normalize_embeddings"] is True
    assert runtime.max_resident_models_observed == 1


def test_residual_cuda_allocation_blocks_next_load(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, cuda, factory = make_runtime(monkeypatch, residual_limit=10)
    runtime.load(NOMIC)
    cuda.allocated = 11
    runtime.unload(NOMIC)
    assert runtime.loaded_model is None
    with pytest.raises(RuntimeError, match="load safety threshold"):
        runtime.load(E5)
    assert len(factory.models) == 1


def test_runtime_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _, _ = make_runtime(monkeypatch)
    with pytest.raises(ValueError, match="unsupported model key"):
        runtime.load("arbitrary")


def test_failed_load_releases_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, cuda, _ = make_runtime(monkeypatch)

    def fail(_: str, **__: Any) -> None:
        raise RuntimeError("synthetic failure")

    runtime._factory = fail  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.load(E5)
    assert runtime.loaded_model is None
    assert cuda.empty_cache_calls == 1


def test_inference_failure_records_sanitized_category(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _, factory = make_runtime(monkeypatch)
    runtime.load(E5)

    def fail(*_: object, **__: object) -> None:
        raise RuntimeError("CUDA out of memory while processing private input")

    factory.models[-1].encode = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        runtime.encode(E5, ["passage: synthetic"], 1)
    assert runtime.last_inference_failure == "CUDA_OUT_OF_MEMORY"
    assert runtime.inference_failure_count == 1
