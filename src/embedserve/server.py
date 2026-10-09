"""Authenticated, single-GPU embedding service with explicit API contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from embedserve.runtime import (
    E5,
    E5_MODEL_ID,
    E5_MODEL_REVISION,
    NATIVE_DIMENSION,
    NOMIC,
    NOMIC_CODE_REVISION,
    NOMIC_MODEL_ID,
    NOMIC_MODEL_REVISION,
    Runtime,
    SentenceTransformerRuntime,
)
from embedserve.scheduler import ModelBusy, ModelCoordinator, ModelLoadError

DEFAULT_MODEL_ALIAS = "nomic-embed-text:v1.5"
DEFAULT_MODEL_DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
log = logging.getLogger("embedserve")


def _validated_key(value: str) -> str:
    if len(value) < 16:
        raise ValueError("embedding server API key must contain at least 16 characters")
    printable = value.isascii() and all(0x21 <= ord(char) <= 0x7E for char in value)
    if not printable:
        raise ValueError("embedding server API key must be a printable ASCII token")
    return value


def _load_api_key() -> str:
    inline = os.getenv("EMBEDSERVE_API_KEY")
    key_file = os.getenv("EMBEDSERVE_API_KEY_FILE")
    if inline and key_file:
        raise ValueError("set only one of EMBEDSERVE_API_KEY or EMBEDSERVE_API_KEY_FILE")
    if key_file:
        path = Path(key_file)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ValueError("EMBEDSERVE_API_KEY_FILE must not be accessible by group or other")
        return _validated_key(path.read_text(encoding="ascii").strip())
    return _validated_key(inline or "")


@dataclass(frozen=True)
class EmbeddingServerSettings:
    api_key: str
    host: str = "127.0.0.1"
    port: int = 11435
    model_id: str = NOMIC_MODEL_ID
    model_alias: str = DEFAULT_MODEL_ALIAS
    model_revision: str = NOMIC_MODEL_REVISION
    code_revision: str = NOMIC_CODE_REVISION
    model_digest: str = DEFAULT_MODEL_DIGEST
    e5_model_id: str = E5_MODEL_ID
    e5_model_revision: str = E5_MODEL_REVISION
    initial_model: str = NOMIC
    batch_size: int = 32
    e5_batch_size: int = 1
    max_items: int = 256
    e5_max_items: int = 32
    max_chars: int = 1_000_000
    max_body_bytes: int = 2_000_000
    require_cuda: bool = True
    max_residual_cuda_bytes: int = 16_777_216
    intent_ttl_seconds: float = 30.0
    reservation_ttl_seconds: float = 10.0
    minimum_residency_seconds: float = 5.0
    retry_after_ms: int = 1500
    e5_recycle_requests: int = 50
    log_level: str = "info"

    def __post_init__(self) -> None:
        _validated_key(self.api_key)
        if self.model_id != NOMIC_MODEL_ID:
            raise ValueError(f"unsupported Nomic model: {self.model_id}")
        if self.model_alias != DEFAULT_MODEL_ALIAS:
            raise ValueError(f"unsupported model alias: {self.model_alias}")
        if self.e5_model_id != E5_MODEL_ID:
            raise ValueError(f"unsupported E5 model: {self.e5_model_id}")
        if self.initial_model not in {NOMIC, E5}:
            raise ValueError("initial_model must be nomic or e5")
        for name, revision in {
            "Nomic model": self.model_revision,
            "Nomic code": self.code_revision,
            "E5 model": self.e5_model_revision,
        }.items():
            if COMMIT_PATTERN.fullmatch(revision) is None:
                raise ValueError(f"{name} revision must be a 40-character commit hash")
        if DIGEST_PATTERN.fullmatch(self.model_digest) is None:
            raise ValueError("model digest must be a 64-character SHA-256 digest")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.batch_size <= 256 or not 1 <= self.e5_batch_size <= 32:
            raise ValueError("model batch size is outside its supported range")
        if not 1 <= self.max_items <= 256 or not 1 <= self.e5_max_items <= 32:
            raise ValueError("request item limit is outside its supported range")
        if self.max_chars < 512:
            raise ValueError("max characters must be at least 512")
        if self.max_body_bytes < self.max_chars:
            raise ValueError("max body bytes must cover max characters")
        if self.max_residual_cuda_bytes < 0:
            raise ValueError("max residual CUDA bytes must not be negative")
        if self.intent_ttl_seconds <= 0 or self.reservation_ttl_seconds <= 0:
            raise ValueError("scheduler TTLs must be positive")
        if self.minimum_residency_seconds < 0:
            raise ValueError("minimum residency must not be negative")
        if self.retry_after_ms <= 0:
            raise ValueError("retry_after_ms must be positive")
        if self.e5_recycle_requests < 1:
            raise ValueError("e5_recycle_requests must be positive")

    @classmethod
    def from_env(cls) -> EmbeddingServerSettings:
        return cls(
            api_key=_load_api_key(),
            host=os.getenv("EMBEDSERVE_HOST", "127.0.0.1"),
            port=int(os.getenv("EMBEDSERVE_PORT", "11435")),
            model_id=os.getenv("EMBEDSERVE_MODEL_ID", NOMIC_MODEL_ID),
            model_alias=os.getenv("EMBEDSERVE_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            model_revision=os.getenv("EMBEDSERVE_MODEL_REVISION", NOMIC_MODEL_REVISION),
            code_revision=os.getenv("EMBEDSERVE_CODE_REVISION", NOMIC_CODE_REVISION),
            model_digest=os.getenv("EMBEDSERVE_MODEL_DIGEST", DEFAULT_MODEL_DIGEST),
            e5_model_id=os.getenv("EMBEDSERVE_E5_MODEL_ID", E5_MODEL_ID),
            e5_model_revision=os.getenv("EMBEDSERVE_E5_MODEL_REVISION", E5_MODEL_REVISION),
            initial_model=os.getenv("EMBEDSERVE_INITIAL_MODEL", NOMIC),
            batch_size=int(os.getenv("EMBEDSERVE_BATCH_SIZE", "32")),
            e5_batch_size=int(os.getenv("EMBEDSERVE_E5_BATCH_SIZE", "1")),
            max_items=int(os.getenv("EMBEDSERVE_MAX_ITEMS", "256")),
            e5_max_items=int(os.getenv("EMBEDSERVE_E5_MAX_ITEMS", "32")),
            max_chars=int(os.getenv("EMBEDSERVE_MAX_CHARS", "1000000")),
            max_body_bytes=int(os.getenv("EMBEDSERVE_MAX_BODY_BYTES", "2000000")),
            require_cuda=os.getenv("EMBEDSERVE_REQUIRE_CUDA", "true").lower()
            not in {"0", "false", "no"},
            max_residual_cuda_bytes=int(
                os.getenv("EMBEDSERVE_MAX_RESIDUAL_CUDA_BYTES", "16777216")
            ),
            intent_ttl_seconds=float(os.getenv("EMBEDSERVE_INTENT_TTL_SECONDS", "30")),
            reservation_ttl_seconds=float(
                os.getenv("EMBEDSERVE_RESERVATION_TTL_SECONDS", "10")
            ),
            minimum_residency_seconds=float(
                os.getenv("EMBEDSERVE_MINIMUM_RESIDENCY_SECONDS", "5")
            ),
            retry_after_ms=int(os.getenv("EMBEDSERVE_RETRY_AFTER_MS", "1500")),
            e5_recycle_requests=int(os.getenv("EMBEDSERVE_E5_RECYCLE_REQUESTS", "50")),
            log_level=os.getenv("EMBEDSERVE_LOG_LEVEL", "info"),
        )


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "invalid or missing API key", "type": "authentication_error"}},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def model_busy_response(error: ModelBusy) -> JSONResponse:
    return JSONResponse(
        error.payload(),
        status_code=503,
        headers={"Retry-After": str(error.retry_after_seconds)},
    )


def model_load_error_response(error: ModelLoadError) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "MODEL_LOAD_FAILED",
                "message": "Requested model could not be loaded.",
                "retryable": error.failure.retryable,
                "requested_model": error.failure.model,
                "restored_model": error.failure.restored_model,
            }
        },
        status_code=503 if error.failure.retryable else 500,
    )


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.api_key = api_key.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        scheme, separator, value = headers.get(b"authorization", b"").partition(b" ")
        authorized = (
            separator == b" "
            and scheme.lower() == b"bearer"
            and secrets.compare_digest(value, self.api_key)
        )
        if not authorized:
            await _unauthorized()(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_embedding_app(
    settings: EmbeddingServerSettings,
    runtime: Runtime,
    *,
    restart_process: Callable[[], None] | None = None,
) -> ASGIApp:
    inference_lock = asyncio.Lock()
    restart_scheduled = False
    restart_process = restart_process or (lambda: os._exit(75))

    def schedule_restart(reason: str) -> None:
        nonlocal restart_scheduled
        if restart_scheduled:
            return
        restart_scheduled = True
        log.warning("scheduled process recycle reason=%s", reason)
        asyncio.get_running_loop().call_later(2.0, restart_process)

    async def load_model(model: str) -> None:
        await asyncio.to_thread(runtime.load, model)

    async def unload_model(model: str) -> None:
        await asyncio.to_thread(runtime.unload, model)

    coordinator = ModelCoordinator(
        loaded_model=runtime.loaded_model,
        load_model=load_model,
        unload_model=unload_model,
        intent_ttl_seconds=settings.intent_ttl_seconds,
        reservation_ttl_seconds=settings.reservation_ttl_seconds,
        minimum_residency_seconds=settings.minimum_residency_seconds,
        retry_after_ms=settings.retry_after_ms,
    )

    async def read_body(request: Request) -> bytes | Response:
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                return JSONResponse({"error": "invalid Content-Length"}, status_code=400)
            if length < 0:
                return JSONResponse({"error": "invalid Content-Length"}, status_code=400)
            if length > settings.max_body_bytes:
                return JSONResponse({"error": "request body too large"}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > settings.max_body_bytes:
                return JSONResponse({"error": "request body too large"}, status_code=413)
        return bytes(body)

    async def read_object(request: Request) -> dict[str, object] | Response:
        body = await read_body(request)
        if isinstance(body, Response):
            return body
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "request body must be an object"}, status_code=400)
        return payload

    def validate_texts(
        source: object, *, allow_string: bool, max_items: int
    ) -> list[str] | Response:
        texts = [source] if allow_string and isinstance(source, str) else source
        if (
            not isinstance(texts, list)
            or not texts
            or not all(isinstance(text, str) for text in texts)
        ):
            expected = "a string or non-empty string list" if allow_string else "a string list"
            return JSONResponse({"error": f"input must be {expected}"}, status_code=400)
        if len(texts) > max_items:
            return JSONResponse(
                {"error": f"batch exceeds maximum of {max_items} items"}, status_code=400
            )
        if sum(len(text) for text in texts) > settings.max_chars:
            return JSONResponse(
                {"error": f"input exceeds maximum of {settings.max_chars} characters"},
                status_code=400,
            )
        return texts

    async def parse_nomic(request: Request) -> tuple[str, list[str]] | Response:
        payload = await read_object(request)
        if isinstance(payload, Response):
            return payload
        model = payload.get("model", settings.model_alias)
        if model != settings.model_alias:
            return JSONResponse({"error": f"unknown model: {model}"}, status_code=404)
        dimensions = payload.get("dimensions")
        if dimensions is not None and dimensions != NATIVE_DIMENSION:
            return JSONResponse(
                {"error": f"dimensions must be {NATIVE_DIMENSION}"}, status_code=400
            )
        if payload.get("encoding_format") not in {None, "float"}:
            return JSONResponse({"error": "only float encoding is supported"}, status_code=400)
        if payload.get("truncate") is False:
            return JSONResponse({"error": "truncate=false is not supported"}, status_code=400)
        texts = validate_texts(
            payload.get("input"), allow_string=True, max_items=settings.max_items
        )
        if isinstance(texts, Response):
            return texts
        return settings.model_alias, texts

    async def parse_e5(request: Request) -> list[str] | Response:
        payload = await read_object(request)
        if isinstance(payload, Response):
            return payload
        if set(payload) != {"inputs"}:
            return JSONResponse(
                {"error": "request body must contain only inputs"}, status_code=400
            )
        return validate_texts(
            payload.get("inputs"), allow_string=False, max_items=settings.e5_max_items
        )

    async def infer(
        model: str, texts: list[str], batch_size: int
    ) -> list[list[float]] | Response:
        if restart_scheduled:
            snapshot = await coordinator.snapshot()
            return model_busy_response(
                ModelBusy(
                    loaded_model=snapshot.loaded_model,
                    requested_model=model,
                    state=snapshot.state,
                    retry_after_ms=8000,
                )
            )
        try:
            admission = await coordinator.acquire(model)
        except ModelBusy as error:
            return model_busy_response(error)
        except ModelLoadError as error:
            return model_load_error_response(error)
        async with admission, inference_lock:
            task = asyncio.create_task(
                asyncio.to_thread(runtime.encode, model, texts, batch_size),
                name=f"embedserve-infer-{model}",
            )
            try:
                vectors = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            except Exception:
                log.error(
                    "inference failed model=%s category=%s failures=%d",
                    model,
                    runtime.last_inference_failure or "UNKNOWN",
                    runtime.inference_failure_count,
                )
                cuda_failure = runtime.last_inference_failure in {
                    "CUDA_OUT_OF_MEMORY",
                    "CUDA_RUNTIME_ERROR",
                }
                if cuda_failure:
                    schedule_restart("cuda_inference_failure")
                return JSONResponse(
                    {
                        "error": {
                            "code": "INFERENCE_FAILED",
                            "message": "Embedding inference failed.",
                            "retryable": cuda_failure,
                        }
                    },
                    status_code=503 if cuda_failure else 500,
                    headers={"Retry-After": "5"} if cuda_failure else None,
                )
        if len(vectors) != len(texts) or any(
            len(vector) != NATIVE_DIMENSION
            or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            return JSONResponse({"error": "encoder returned invalid vector dimensions"}, 500)
        if model == E5 and runtime.successful_requests(E5) >= settings.e5_recycle_requests:
            schedule_restart("e5_request_budget")
        return vectors

    async def health(_: Request) -> Response:
        snapshot = await coordinator.snapshot()
        return JSONResponse(
            {
                "status": "ok",
                "model": settings.model_id,
                "ollama_name": settings.model_alias,
                "model_revision": settings.model_revision,
                "code_revision": settings.code_revision,
                "model_digest": settings.model_digest,
                "device": runtime.device,
                "gpu": runtime.gpu_name,
                "native_dimension": NATIVE_DIMENSION,
                "models": {
                    NOMIC: {
                        "id": settings.model_id,
                        "revision": settings.model_revision,
                        "code_revision": settings.code_revision,
                        "normalized": False,
                        "max_seq_length": 2048,
                    },
                    E5: {
                        "id": settings.e5_model_id,
                        "revision": settings.e5_model_revision,
                        "normalized": True,
                        "max_seq_length": 512,
                    },
                },
                "scheduler": {
                    "state": snapshot.state,
                    "loaded_model": snapshot.loaded_model,
                    "active_requests": snapshot.active_requests,
                    "switch_in_progress": snapshot.switch_in_progress,
                    "single_residency_verified": runtime.max_resident_models_observed <= 1,
                    "last_unload_allocated_bytes": runtime.last_unload_allocated_bytes,
                    "last_inference_failure": runtime.last_inference_failure,
                    "inference_failure_count": runtime.inference_failure_count,
                    "successful_requests": {
                        NOMIC: runtime.successful_requests(NOMIC),
                        E5: runtime.successful_requests(E5),
                    },
                    "restart_scheduled": restart_scheduled,
                },
            }
        )

    async def tags(_: Request) -> Response:
        return JSONResponse(
            {
                "models": [
                    {
                        "name": settings.model_alias,
                        "model": settings.model_alias,
                        "digest": settings.model_digest,
                        "details": {
                            "family": "nomic-bert",
                            "parameter_size": "137M",
                            "context_length": 2048,
                            "embedding_length": NATIVE_DIMENSION,
                            "model_revision": settings.model_revision,
                            "code_revision": settings.code_revision,
                        },
                        "capabilities": ["embedding"],
                    }
                ]
            }
        )

    async def ollama_embed(request: Request) -> Response:
        parsed = await parse_nomic(request)
        if isinstance(parsed, Response):
            return parsed
        model, texts = parsed
        vectors = await infer(NOMIC, texts, settings.batch_size)
        if isinstance(vectors, Response):
            return vectors
        return JSONResponse({"model": model, "embeddings": vectors})

    async def openai_models(_: Request) -> Response:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": settings.model_alias, "object": "model", "owned_by": "embedserve"}
                ],
            }
        )

    async def openai_embed(request: Request) -> Response:
        parsed = await parse_nomic(request)
        if isinstance(parsed, Response):
            return parsed
        model, texts = parsed
        vectors = await infer(NOMIC, texts, settings.batch_size)
        if isinstance(vectors, Response):
            return vectors
        token_estimate = sum(len(text) // 4 for text in texts)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": vector, "index": index}
                    for index, vector in enumerate(vectors)
                ],
                "model": model,
                "usage": {"prompt_tokens": token_estimate, "total_tokens": token_estimate},
            }
        )

    async def tei_embed(request: Request) -> Response:
        texts = await parse_e5(request)
        if isinstance(texts, Response):
            return texts
        vectors = await infer(E5, texts, settings.e5_batch_size)
        if isinstance(vectors, Response):
            return vectors
        return JSONResponse(vectors)

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/api/tags", tags),
            Route("/api/embed", ollama_embed, methods=["POST"]),
            Route("/v1/models", openai_models),
            Route("/v1/embeddings", openai_embed, methods=["POST"]),
            Route("/embed", tei_embed, methods=["POST"]),
        ]
    )
    return BearerAuthMiddleware(app, settings.api_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedserve")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    try:
        settings = EmbeddingServerSettings.from_env()
        runtime = SentenceTransformerRuntime(settings)
        runtime.load(settings.initial_model)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    uvicorn.run(
        create_embedding_app(settings, runtime),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=settings.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
