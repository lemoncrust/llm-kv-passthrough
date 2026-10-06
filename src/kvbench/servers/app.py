"""
FastAPI Application for KV-Bench Server.

This module provides a FastAPI application that exposes OpenAI-compatible
endpoints for the KV-Bench mock LLM server.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from kvbench.core.config import KVBenchConfig
from kvbench.kv.factory import create_kv_stack
from kvbench.metrics.prometheus import CONTENT_TYPE_LATEST, MetricsExporter
from kvbench.servers.combined import CombinedServer
from kvbench.servers.decode import DecodeServer
from kvbench.servers.factory import create_server
from kvbench.servers.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelList,
)
from kvbench.servers.prefill import PrefillServer
from kvbench.servers.proxy import DisaggregatedProxy
from kvbench.servers.warmup import WarmupController, WarmupRequest
from kvbench.trace import spans
from kvbench.trace.identity import ConversationTracker, RequestIdentity, identify

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class KVBenchApp:
    """KV-Bench FastAPI application wrapper.

    This class manages the lifecycle of the server and provides
    the FastAPI application instance.

    Attributes:
        config: KV-Bench configuration.
        app: FastAPI application instance.
    """

    def __init__(self, config: KVBenchConfig | None = None) -> None:
        """Initialize the KV-Bench application.

        Args:
            config: KV-Bench configuration. Uses defaults if not provided.
        """
        self.config = config or KVBenchConfig()
        self._server: PrefillServer | DecodeServer | CombinedServer | DisaggregatedProxy | None = (
            None
        )
        self._kv = None
        self._warmup: WarmupController | None = None
        self._metrics = MetricsExporter(self)
        self._conversations = ConversationTracker()

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
            """Manage application lifecycle."""
            await self._startup()
            yield
            await self._shutdown()

        self.app = FastAPI(
            title="KV-Bench",
            description="Mock LLM server for KV cache benchmarking",
            version="0.1.0",
            lifespan=lifespan,
        )
        self._setup_routes()

    async def _startup(self) -> None:
        """Initialize server components on startup."""
        logger.info(f"Starting KV-Bench server (type: {self.config.server.server_type})")

        # Start the KV management stack (not needed for proxy); it owns
        # all storage — KV-Bench itself never touches a storage backend
        if self.config.server.server_type != "proxy":
            self._kv = create_kv_stack(self.config)
            await self._kv.start()
            self._warmup = WarmupController(self._kv)

        # Create server
        self._server = create_server(self.config, self._kv)
        await self._server.start()

        logger.info(
            f"KV-Bench server started on {self.config.server.host}:{self.config.server.port}"
        )

    async def _shutdown(self) -> None:
        """Cleanup server components on shutdown."""
        logger.info("Shutting down KV-Bench server...")

        if self._warmup is not None:
            await self._warmup.cancel()

        if self._server:
            await self._server.stop()

        if self._kv is not None:
            await self._kv.close()

        logger.info("KV-Bench server stopped")

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.health_check()

        @self.app.get("/metrics")
        async def get_metrics() -> Response:
            """Prometheus metrics endpoint."""
            return Response(content=self._metrics.render(), media_type=CONTENT_TYPE_LATEST)

        @self.app.get("/stats")
        async def get_stats() -> MetricsResponse:
            """Get server metrics as JSON."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.get_metrics()

        @self.app.get("/kvbench/state")
        async def get_kv_state() -> JSONResponse:
            """KV stack capacity, live tier usage, and operation stats."""
            if self._kv is None:
                raise HTTPException(
                    status_code=404, detail="This server type has no KV stack (proxy?)"
                )
            state: dict = {}
            capacity_fn = getattr(self._kv, "capacity_info", None)
            if capacity_fn is not None:
                state["capacity"] = capacity_fn()
            usage_fn = getattr(self._kv, "usage_info", None)
            if usage_fn is not None:
                state["usage"] = usage_fn()
            stats = self._kv.stats
            state["kv_stats"] = {
                "lookups": stats.lookups,
                "lookup_tokens": stats.lookup_tokens,
                "hit_tokens": stats.hit_tokens,
                "token_hit_rate": stats.token_hit_rate,
                "stores": stats.stores,
                "stored_tokens": stats.stored_tokens,
                "retrieves": stats.retrieves,
                "retrieved_tokens": stats.retrieved_tokens,
                "errors": stats.errors,
            }
            return JSONResponse(content=state)

        @self.app.post("/kvbench/warmup", status_code=202)
        async def start_warmup(request: WarmupRequest) -> JSONResponse:
            """Fill the KV stack to steady state (background task)."""
            if self._warmup is None:
                raise HTTPException(
                    status_code=404, detail="This server type has no KV stack (proxy?)"
                )
            try:
                status = self._warmup.start(request)
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e)) from None
            return JSONResponse(status_code=202, content=status.to_dict())

        @self.app.get("/kvbench/warmup")
        async def warmup_status() -> JSONResponse:
            """Status of the current/last warmup run."""
            if self._warmup is None:
                raise HTTPException(
                    status_code=404, detail="This server type has no KV stack (proxy?)"
                )
            return JSONResponse(content=self._warmup.status.to_dict())

        @self.app.delete("/kvbench/warmup")
        async def cancel_warmup() -> JSONResponse:
            """Cancel a running warmup."""
            if self._warmup is None:
                raise HTTPException(
                    status_code=404, detail="This server type has no KV stack (proxy?)"
                )
            await self._warmup.cancel()
            return JSONResponse(content=self._warmup.status.to_dict())

        @self.app.get("/v1/models")
        async def list_models() -> ModelList:
            """List available models (OpenAI-compatible)."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self._server.list_models()

        @self.app.post("/v1/chat/completions", response_model=None)
        async def chat_completions(
            request: ChatCompletionRequest,
            http_request: Request,
        ) -> ChatCompletionResponse | StreamingResponse:
            """Chat completion endpoint (OpenAI-compatible)."""
            if self._server is None:
                raise HTTPException(status_code=503, detail="Server not initialized")

            # The request's trace is rooted here, in the HTTP layer, so it
            # covers the whole exchange; stages nest underneath. Identity
            # comes from the client's headers when it sends them.
            identity = identify(
                http_request.headers, f"chatcmpl-{uuid.uuid4().hex[:24]}", self._conversations
            )
            root_attrs: dict[str, Any] = {
                "model": request.model,
                "stream": bool(request.stream),
                **identity.span_attrs(),
            }

            # Handle streaming
            if request.stream:
                return StreamingResponse(
                    self._stream_response(request, identity, root_attrs),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            # Non-streaming response
            start = time.perf_counter()
            with spans.root("request", trace_id=identity.request_id, **root_attrs) as req_span:
                if isinstance(self._server, CombinedServer):
                    result = await self._server.chat_completions(
                        request, completion_id=identity.completion_id
                    )
                elif isinstance(self._server, (DisaggregatedProxy, PrefillServer)):
                    result = await self._server.chat_completions(request)
                elif isinstance(self._server, DecodeServer):
                    # Decode-only server: derive context length with the same
                    # simulated tokenizer the KV path uses, so timing and KV
                    # token counts agree.
                    prompt = request.prompt_text or ""
                    context_length = max(
                        1, len(self._server.processor.simulate_tokenize(prompt))
                    )
                    result = await self._server.chat_completions(request, context_length)
                else:
                    raise HTTPException(status_code=500, detail="Unknown server type")
                req_span.set(status="error" if isinstance(result, ErrorResponse) else "ok")
            self._metrics.request_duration.observe(time.perf_counter() - start)

            if isinstance(result, ErrorResponse):
                return JSONResponse(
                    status_code=500,
                    content=result.model_dump(),
                )

            return result

        @self.app.exception_handler(Exception)
        async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
            """Handle uncaught exceptions."""
            logger.exception(f"Unhandled exception: {exc}")
            error = ErrorResponse.create(str(exc), "server_error")
            return JSONResponse(status_code=500, content=error.model_dump())

    async def _stream_response(
        self,
        request: ChatCompletionRequest,
        identity: RequestIdentity,
        root_attrs: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response, observing TTFT and total duration.

        The request's root span wraps the whole stream, so a client that
        disconnects mid-stream still yields a closed span (``cancelled``).
        """
        if self._server is None:
            yield ErrorResponse.create("Server not initialized", "server_error").model_dump_json()
            return

        with spans.root("request", trace_id=identity.request_id, **root_attrs) as req_span:
            if isinstance(self._server, CombinedServer):
                stream = self._server.chat_completions_stream(
                    request, completion_id=identity.completion_id
                )
            elif isinstance(self._server, DisaggregatedProxy):
                stream = self._server.chat_completions_stream(request)
            elif isinstance(self._server, DecodeServer):
                prompt = request.prompt_text or ""
                context_length = max(
                    1, len(self._server.processor.simulate_tokenize(prompt))
                )
                stream = self._server.chat_completions_stream(request, context_length)
            else:
                # PrefillServer doesn't support streaming well
                error = ErrorResponse.create(
                    "Prefill-only server doesn't support streaming",
                    "invalid_request_error",
                )
                req_span.set(status="error")
                yield f"data: {error.model_dump_json()}\n\n"
                return

            start = time.perf_counter()
            chunks = 0
            async for chunk in stream:
                if chunks == 0:
                    ttft = time.perf_counter() - start
                    self._metrics.ttft.observe(ttft)
                    req_span.set(first_chunk_ms=round(ttft * 1000, 3))
                chunks += 1
                yield chunk
            req_span.set(chunks=chunks, status="ok")
            self._metrics.request_duration.observe(time.perf_counter() - start)


def create_app(config: KVBenchConfig | None = None) -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        config: KV-Bench configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application.
    """
    kvbench_app = KVBenchApp(config)
    return kvbench_app.app


def load_config() -> KVBenchConfig:
    """Load the server configuration from the environment.

    Resolution order:
    1. If KVBENCH_CONFIG_FILE is set, load that YAML file (this is how the
       `kvbench serve` CLI passes its fully-resolved configuration through
       to uvicorn worker processes).
    2. Otherwise, build the configuration from KVBENCH_* environment
       variables (the path used by Docker/compose deployments running
       uvicorn directly).

    Returns:
        The resolved KVBenchConfig.
    """
    import os

    config_file = os.environ.get("KVBENCH_CONFIG_FILE")
    if config_file:
        logger.info(f"Loading configuration from {config_file}")
        return KVBenchConfig.from_yaml(config_file)
    return KVBenchConfig.from_env()


# Lazy app instance for uvicorn
# Created on first access to avoid import-time initialization
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the default FastAPI application.

    This function lazily creates the app on first access, avoiding
    import-time initialization issues. The configuration is resolved via
    load_config(), so KVBENCH_CONFIG_FILE and KVBENCH_* environment
    variables are honored.
    """
    global _app
    if _app is None:
        _app = create_app(load_config())
    return _app


# For uvicorn (factory): uvicorn kvbench.servers.app:app --factory
# or programmatically: uvicorn.run("kvbench.servers.app:get_app", factory=True)
app = get_app
