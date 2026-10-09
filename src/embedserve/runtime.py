"""CUDA model lifecycle with strict single-residency checks."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Protocol, cast

NOMIC = "nomic"
E5 = "e5"
NATIVE_DIMENSION = 768
NOMIC_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
NOMIC_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
E5_MODEL_ID = "intfloat/multilingual-e5-base"
E5_MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


class RuntimeSettings(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def code_revision(self) -> str: ...

    @property
    def e5_model_id(self) -> str: ...

    @property
    def e5_model_revision(self) -> str: ...

    @property
    def require_cuda(self) -> bool: ...

    @property
    def max_residual_cuda_bytes(self) -> int: ...


class Runtime(Protocol):
    device: str
    gpu_name: str | None
    loaded_model: str | None
    max_resident_models_observed: int
    last_unload_allocated_bytes: int | None
    last_inference_failure: str | None
    inference_failure_count: int

    def load(self, model: str) -> None: ...

    def unload(self, model: str) -> None: ...

    def encode(self, model: str, texts: list[str], batch_size: int) -> list[list[float]]: ...


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    max_seq_length: int
    normalize_embeddings: bool
    trust_remote_code: bool = False
    code_revision: str | None = None


class SentenceTransformerRuntime:
    def __init__(self, settings: RuntimeSettings) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "embedding dependencies are missing; install embedserve[cuda-pascal]"
            ) from error

        cuda = torch.cuda.is_available()
        if settings.require_cuda and not cuda:
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
        self.device = "cuda" if cuda else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if cuda else None
        self.loaded_model: str | None = None
        self.max_resident_models_observed = 0
        self.last_unload_allocated_bytes: int | None = None
        self.last_inference_failure: str | None = None
        self.inference_failure_count = 0
        self._settings = settings
        self._torch = torch
        self._factory = SentenceTransformer
        self._model: Any | None = None
        self._specs = {
            NOMIC: ModelSpec(
                key=NOMIC,
                model_id=settings.model_id,
                revision=settings.model_revision,
                code_revision=settings.code_revision,
                max_seq_length=2048,
                normalize_embeddings=False,
                trust_remote_code=True,
            ),
            E5: ModelSpec(
                key=E5,
                model_id=settings.e5_model_id,
                revision=settings.e5_model_revision,
                max_seq_length=512,
                normalize_embeddings=True,
            ),
        }

    def load(self, model: str) -> None:
        if model not in self._specs:
            raise ValueError(f"unsupported model key: {model}")
        if self._model is not None or self.loaded_model is not None:
            raise RuntimeError("a model is already resident")
        if (
            self.last_unload_allocated_bytes is not None
            and self.last_unload_allocated_bytes > self._settings.max_residual_cuda_bytes
        ):
            raise RuntimeError("CUDA allocation remains above the load safety threshold")
        spec = self._specs[model]
        kwargs: dict[str, object] = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "device": self.device,
        }
        if spec.code_revision is not None:
            kwargs["model_kwargs"] = {"code_revision": spec.code_revision}
            kwargs["config_kwargs"] = {"code_revision": spec.code_revision}
        loaded: Any | None = None
        try:
            loaded = self._factory(spec.model_id, **kwargs)
            loaded.max_seq_length = spec.max_seq_length
            dimension = loaded.get_sentence_embedding_dimension()
            if dimension != NATIVE_DIMENSION:
                raise RuntimeError(
                    f"model native dimension must be {NATIVE_DIMENSION}, got {dimension}"
                )
        except Exception:
            if loaded is not None:
                del loaded
            self._release_cuda()
            self.last_unload_allocated_bytes = self._allocated_bytes()
            raise
        self._model = loaded
        self.loaded_model = model
        self.max_resident_models_observed = max(self.max_resident_models_observed, 1)

    def unload(self, model: str) -> None:
        if self.loaded_model != model or self._model is None:
            raise RuntimeError(f"model {model} is not resident")
        loaded = self._model
        self._model = None
        self.loaded_model = None
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        del loaded
        self._release_cuda()
        allocated = self._allocated_bytes()
        self.last_unload_allocated_bytes = allocated

    def encode(self, model: str, texts: list[str], batch_size: int) -> list[list[float]]:
        if self.loaded_model != model or self._model is None:
            raise RuntimeError(f"model {model} is not resident")
        spec = self._specs[model]
        try:
            with self._torch.inference_mode():
                values = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=spec.normalize_embeddings,
                )
        except Exception as error:
            message = str(error).lower()
            if "out of memory" in message:
                self.last_inference_failure = "CUDA_OUT_OF_MEMORY"
            elif "cuda" in message or "cublas" in message or "cudnn" in message:
                self.last_inference_failure = "CUDA_RUNTIME_ERROR"
            else:
                self.last_inference_failure = type(error).__name__
            self.inference_failure_count += 1
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
            raise
        self.last_inference_failure = None
        rows = cast(list[list[float]], values.tolist())
        return [[float(value) for value in row] for row in rows]

    def _release_cuda(self) -> None:
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
        gc.collect()

    def _allocated_bytes(self) -> int:
        return int(self._torch.cuda.memory_allocated(0)) if self.device == "cuda" else 0
