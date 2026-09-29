"""Synthetic tokenizer/formatter with stable multi-turn cache identity.

One token is a deterministic hash of a byte span. Rational span boundaries
support fractional bytes/token without input-length-dependent prefix changes.
This is a workload model, not a model tokenizer or chat template.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kvbench.servers.openai_compat import ChatMessage


class SyntheticTokenizer:
    def __init__(self, bytes_per_token: float = 4.0, vocab_size: int = 128256):
        self.ratio = Fraction(str(bytes_per_token))
        if self.ratio < 1 or vocab_size < 1:
            raise ValueError("bytes_per_token and vocab_size must be positive")
        self.vocab_size = vocab_size

    def boundary(self, count: int) -> int:
        n, d = self.ratio.numerator, self.ratio.denominator
        return (count * n + d - 1) // d

    def token_id(self, piece: bytes) -> int:
        return int.from_bytes(hashlib.blake2s(piece, digest_size=4).digest(), "little") % self.vocab_size

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        result: list[int] = []
        start = 0
        while start < len(data):
            end = self.boundary(len(result) + 1)
            result.append(self.token_id(data[start:end]))
            start = end
        return result

    def _pad_header(self, prefix: str) -> str:
        length = len(prefix.encode("utf-8"))
        n, d = self.ratio.numerator, self.ratio.denominator
        count = (length * d + n - 1) // n
        return prefix + " " * (self.boundary(count) - length)

    def prompt(self, messages: list[ChatMessage]) -> str:
        text = ""
        for message in messages:
            text += f"{message.role.value}: "
            if message.role.value == "assistant":
                text = self._pad_header(text)
            content = message.content or ""
            if isinstance(content, list):
                content = "".join(part["text"] for part in content)
            text += content + "\n"
        return self._pad_header(text + "assistant: ")

    def sample(self, position: int) -> tuple[str, int]:
        """Return exactly one synthetic token, round-trippable as output text."""
        width = self.boundary(position + 1) - self.boundary(position)
        words = (" the", " cat", " sat", " on ", " a  ", " mat")
        text = words[position % len(words)].ljust(width)[:width]
        return text, self.token_id(text.encode("ascii"))
