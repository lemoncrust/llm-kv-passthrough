"""
V1 rate-based serving and OpenAI-shaped protocol types.

The older combined/prefill/decode/proxy implementations remain historical
experiments and are not selected by the application or exported here.
"""

from kvbench.servers.factory import create_server
from kvbench.servers.openai_compat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ErrorResponse,
    HealthResponse,
    MessageRole,
    MetricsResponse,
    ModelInfo,
    ModelList,
    ResponseFormat,
    StreamChoice,
    Usage,
)
from kvbench.servers.rate_server import RateServer, ServingStats

__all__ = [
    # Servers
    "RateServer",
    "ServingStats",
    "create_server",
    # OpenAI Types
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatMessage",
    "MessageRole",
    "Choice",
    "ChoiceMessage",
    "StreamChoice",
    "DeltaMessage",
    "Usage",
    "ResponseFormat",
    "ErrorResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelInfo",
    "ModelList",
]
