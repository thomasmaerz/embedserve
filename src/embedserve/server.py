"""Authenticated PyTorch embedding service with Ollama and OpenAI-compatible APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from embedserve.scheduler import ModelBusy

NATIVE_DIMENSION = 768
DEFAULT_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_MODEL_ALIAS = "nomic-embed-text:v1.5"
DEFAULT_MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
DEFAULT_MODEL_DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class Encoder(Protocol):
    model_id: str
    device: str
    gpu_name: str | None

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]: ...


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
    model_id: str = DEFAULT_MODEL_ID
    model_alias: str = DEFAULT_MODEL_ALIAS
    model_revision: str = DEFAULT_MODEL_REVISION
    code_revision: str = DEFAULT_CODE_REVISION
    model_digest: str = DEFAULT_MODEL_DIGEST
    batch_size: int = 32
    max_items: int = 256
    max_chars: int = 1_000_000
    max_body_bytes: int = 2_000_000
    require_cuda: bool = True
    log_level: str = "info"

    def __post_init__(self) -> None:
        _validated_key(self.api_key)
        if self.model_id != DEFAULT_MODEL_ID:
            raise ValueError(f"unsupported model: {self.model_id}")
        if self.model_alias != DEFAULT_MODEL_ALIAS:
            raise ValueError(f"unsupported model alias: {self.model_alias}")
        if COMMIT_PATTERN.fullmatch(self.model_revision) is None:
            raise ValueError("model revision must be a 40-character commit hash")
        if COMMIT_PATTERN.fullmatch(self.code_revision) is None:
            raise ValueError("code revision must be a 40-character commit hash")
        if DIGEST_PATTERN.fullmatch(self.model_digest) is None:
            raise ValueError("model digest must be a 64-character SHA-256 digest")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch size must be between 1 and 256")
        if not 1 <= self.max_items <= 256:
            raise ValueError("max items must be between 1 and 256")
        if self.max_chars < 512:
            raise ValueError("max characters must be at least 512")
        if self.max_body_bytes < self.max_chars:
            raise ValueError("max body bytes must cover max characters")

    @classmethod
    def from_env(cls) -> EmbeddingServerSettings:
        return cls(
            api_key=_load_api_key(),
            host=os.getenv("EMBEDSERVE_HOST", "127.0.0.1"),
            port=int(os.getenv("EMBEDSERVE_PORT", "11435")),
            model_id=os.getenv("EMBEDSERVE_MODEL_ID", DEFAULT_MODEL_ID),
            model_alias=os.getenv("EMBEDSERVE_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
            model_revision=os.getenv("EMBEDSERVE_MODEL_REVISION", DEFAULT_MODEL_REVISION),
            code_revision=os.getenv("EMBEDSERVE_CODE_REVISION", DEFAULT_CODE_REVISION),
            model_digest=os.getenv("EMBEDSERVE_MODEL_DIGEST", DEFAULT_MODEL_DIGEST),
            batch_size=int(os.getenv("EMBEDSERVE_BATCH_SIZE", "32")),
            max_items=int(os.getenv("EMBEDSERVE_MAX_ITEMS", "256")),
            max_chars=int(os.getenv("EMBEDSERVE_MAX_CHARS", "1000000")),
            max_body_bytes=int(os.getenv("EMBEDSERVE_MAX_BODY_BYTES", "2000000")),
            require_cuda=os.getenv("EMBEDSERVE_REQUIRE_CUDA", "true").lower()
            not in {"0", "false", "no"},
            log_level=os.getenv("EMBEDSERVE_LOG_LEVEL", "info"),
        )


class SentenceTransformerEncoder:
    def __init__(self, settings: EmbeddingServerSettings) -> None:
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
        self.model_id = settings.model_id
        self.device = "cuda" if cuda else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if cuda else None
        self._torch = torch
        self._model = SentenceTransformer(
            settings.model_id,
            revision=settings.model_revision,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={"code_revision": settings.code_revision},
            config_kwargs={"code_revision": settings.code_revision},
        )
        self._model.max_seq_length = 2048
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension != NATIVE_DIMENSION:
            raise RuntimeError(
                f"model native dimension must be {NATIVE_DIMENSION}, got {dimension}"
            )

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        with self._torch.inference_mode():
            values = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        rows = cast(list[list[float]], values.tolist())
        return [[float(value) for value in row] for row in rows]


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


def create_embedding_app(settings: EmbeddingServerSettings, encoder: Encoder) -> ASGIApp:
    lock = asyncio.Lock()

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

    async def parse_input(request: Request) -> tuple[str, list[str]] | Response:
        body = await read_body(request)
        if isinstance(body, Response):
            return body
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "request body must be an object"}, status_code=400)
        model = payload.get("model", settings.model_alias)
        if model != settings.model_alias:
            return JSONResponse({"error": f"unknown model: {model}"}, status_code=404)
        dimensions = payload.get("dimensions")
        if dimensions is not None and dimensions != NATIVE_DIMENSION:
            return JSONResponse(
                {"error": f"dimensions must be {NATIVE_DIMENSION}"}, status_code=400
            )
        encoding_format = payload.get("encoding_format")
        if encoding_format not in {None, "float"}:
            return JSONResponse({"error": "only float encoding is supported"}, status_code=400)
        if payload.get("truncate") is False:
            return JSONResponse({"error": "truncate=false is not supported"}, status_code=400)
        source = payload.get("input")
        texts = [source] if isinstance(source, str) else source
        if (
            not isinstance(texts, list)
            or not texts
            or not all(isinstance(text, str) for text in texts)
        ):
            return JSONResponse(
                {"error": "input must be a string or non-empty string list"}, status_code=400
            )
        if len(texts) > settings.max_items:
            return JSONResponse(
                {"error": f"batch exceeds maximum of {settings.max_items} items"}, status_code=400
            )
        if sum(len(text) for text in texts) > settings.max_chars:
            return JSONResponse(
                {"error": f"input exceeds maximum of {settings.max_chars} characters"},
                status_code=400,
            )
        return settings.model_alias, texts

    async def encode(request: Request) -> tuple[str, list[str], list[list[float]]] | Response:
        parsed = await parse_input(request)
        if isinstance(parsed, Response):
            return parsed
        model, texts = parsed
        async with lock:
            vectors = await asyncio.to_thread(encoder.encode, texts, settings.batch_size)
        if len(vectors) != len(texts) or any(
            len(vector) != NATIVE_DIMENSION for vector in vectors
        ):
            return JSONResponse({"error": "encoder returned invalid vector dimensions"}, 500)
        return model, texts, vectors

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "status": "ok",
                "model": encoder.model_id,
                "ollama_name": settings.model_alias,
                "model_revision": settings.model_revision,
                "code_revision": settings.code_revision,
                "model_digest": settings.model_digest,
                "device": encoder.device,
                "gpu": encoder.gpu_name,
                "native_dimension": NATIVE_DIMENSION,
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
        result = await encode(request)
        if isinstance(result, Response):
            return result
        model, _, vectors = result
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
        result = await encode(request)
        if isinstance(result, Response):
            return result
        model, texts, vectors = result
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

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/api/tags", tags),
            Route("/api/embed", ollama_embed, methods=["POST"]),
            Route("/v1/models", openai_models),
            Route("/v1/embeddings", openai_embed, methods=["POST"]),
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
        encoder = SentenceTransformerEncoder(settings)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    uvicorn.run(
        create_embedding_app(settings, encoder),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=settings.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
