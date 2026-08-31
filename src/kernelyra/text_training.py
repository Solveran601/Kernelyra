"""Loss-mask-safe text preparation for a future causal language-model trainer.

This module is deliberately *not* a text or LLM trainer.  It converts UTF-8
text into byte-token causal examples and marks carried context as loss-free, so
a future trainer can consume document chunks without teaching the model to
predict the repeated overlap again.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .native_core import NativeCore, NativeCoreError


@dataclass(frozen=True, slots=True)
class ByteTokenizer:
    """A reversible UTF-8 byte tokenizer with two explicit special tokens.

    Byte tokenisation is a stable preparation baseline, not a claim that it is
    an optimal tokenizer for every language or model architecture.
    """

    bos_id: int = 256
    pad_id: int = 257

    @property
    def vocab_size(self) -> int:
        return self.pad_id + 1

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("ByteTokenizer.encode expects str")
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = bytearray()
        for token in token_ids:
            value = int(token)
            if value in {self.bos_id, self.pad_id}:
                continue
            if not 0 <= value <= 255:
                raise ValueError("Token is outside the byte tokenizer vocabulary")
            raw.append(value)
        return bytes(raw).decode("utf-8")


@dataclass(frozen=True, slots=True)
class MaskedTextExample:
    """One causal sequence with loss disabled for its carried context prefix."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    loss_mask: tuple[float, ...]
    context_prefix_tokens: int
    content_tokens: int


@dataclass(frozen=True, slots=True)
class MaskedTextBatch:
    """Padded NumPy arrays ready for a trainer that honours ``loss_mask``."""

    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray
    attention_mask: np.ndarray


def _floor_utf8_boundary(payload: bytes, index: int) -> int:
    """Move an interior byte position left to the start of a UTF-8 character."""
    index = max(0, min(len(payload), int(index)))
    while 0 < index < len(payload) and payload[index] & 0b11000000 == 0b10000000:
        index -= 1
    return index


def _python_chunk_plan(
    text: str,
    *,
    minimum_bytes: int,
    target_bytes: int | None,
    maximum_bytes: int,
    overlap_bytes: int,
) -> list[dict[str, object]]:
    """Portable boundary-safe fallback used only when the native planner is absent."""
    payload = text.encode("utf-8")
    if not payload:
        return []
    result: list[dict[str, object]] = []
    if target_bytes is None:
        if len(payload) <= maximum_bytes or minimum_bytes == maximum_bytes:
            target = min(len(payload), maximum_bytes)
        else:
            span = maximum_bytes - minimum_bytes
            preferred = minimum_bytes + (span * 2) // 3
            count = max(1, (len(payload) + preferred - 1) // preferred)
            target = min(maximum_bytes, max(minimum_bytes, (len(payload) + count - 1) // count))
    else:
        target = target_bytes
    start = 0
    while start < len(payload):
        ceiling = _floor_utf8_boundary(payload, min(len(payload), start + maximum_bytes))
        preferred = _floor_utf8_boundary(payload, min(ceiling, start + target))
        minimum = _floor_utf8_boundary(payload, min(ceiling, start + minimum_bytes))
        if ceiling <= start:
            raise ValueError("Text chunk byte bounds cannot contain one UTF-8 character")
        # Prefer a sentence or whitespace end only when that still meets the
        # requested minimum.  This changes no bytes and never drops content.
        boundary = preferred if preferred > start else ceiling
        for separator in (b"\n", b" ", b".", b"!", b"?"):
            point = payload.rfind(separator, minimum, boundary)
            if point >= minimum:
                candidate = _floor_utf8_boundary(payload, point + 1)
                if candidate > start:
                    boundary = candidate
                    break
        end = boundary if boundary > start else ceiling
        context_start = _floor_utf8_boundary(payload, max(0, start - overlap_bytes))
        context = payload[context_start:end].decode("utf-8")
        content = payload[start:end].decode("utf-8")
        result.append(
            {
                "context_start_byte": context_start,
                "content_start_byte": start,
                "end_byte": end,
                "context_prefix_bytes": start - context_start,
                "context": context,
                "content": content,
            }
        )
        start = end
    return result


def plan_text_for_training(
    text: str,
    *,
    minimum_bytes: int = 512,
    target_bytes: int | str | None = "auto",
    maximum_bytes: int = 4096,
    overlap_bytes: int = 256,
    core: NativeCore | None = None,
) -> list[dict[str, object]]:
    """Plan contiguous UTF-8 chunks with an explicit carried-context prefix.

    Set ``target_bytes="auto"`` (the default) to let the Rust policy choose a
    repeatable nominal size inside ``minimum_bytes`` and ``maximum_bytes``.
    The planner then moves individual boundaries to newline/sentence ends,
    which deliberately makes chunk sizes variable without losing or repeating
    training content.  Pass a positive integer for a manual nominal size.

    This is opt-in preparation only: normal tabular training does not invoke
    it.  For multi-gigabyte JSONL or Telegram exports, use
    :func:`iter_conversation_chunks` to stream whole messages first, then plan
    each selected text payload if a downstream trainer needs byte spans.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    minimum, maximum, overlap = map(int, (minimum_bytes, maximum_bytes, overlap_bytes))
    automatic = target_bytes is None or target_bytes == "auto"
    if isinstance(target_bytes, str) and not automatic:
        raise ValueError('target_bytes must be a positive integer, None, or "auto"')
    target = 0 if automatic else int(target_bytes)
    if not (1 <= minimum <= maximum <= 2 * 1024 * 1024):
        raise ValueError("Require 1 <= minimum_bytes <= maximum_bytes <= 2 MiB")
    if not automatic and not minimum <= target <= maximum:
        raise ValueError("Manual target_bytes must satisfy minimum_bytes <= target_bytes <= maximum_bytes")
    if not 0 <= overlap < maximum:
        raise ValueError("overlap_bytes must be non-negative and smaller than maximum_bytes")
    selected = core
    if selected is None:
        try:
            selected = NativeCore()
        except NativeCoreError:
            selected = None
    if selected is not None:
        try:
            return selected.plan_text_chunks(
                text,
                minimum_bytes=minimum,
                target_bytes=target,
                maximum_bytes=maximum,
                overlap_bytes=overlap,
            )
        except NativeCoreError:
            # A pre-text-planner binary is compatible with the SDK; falling
            # back preserves the explicit mask contract instead of failing
            # after a caller has already streamed a large source file.
            pass
    return _python_chunk_plan(
        text,
        minimum_bytes=minimum,
        target_bytes=None if automatic else target,
        maximum_bytes=maximum,
        overlap_bytes=overlap,
    )


def prepare_masked_text_examples(
    text: str,
    *,
    tokenizer: ByteTokenizer | None = None,
    minimum_bytes: int = 512,
    target_bytes: int | str | None = "auto",
    maximum_bytes: int = 4096,
    overlap_bytes: int = 256,
    core: NativeCore | None = None,
) -> list[MaskedTextExample]:
    """Create causal input/target pairs where carried overlap receives no loss.

    ``loss_mask[i] == 1`` means target token ``target_ids[i]`` belongs to new
    document content.  No model is trained here; this is a verified input
    contract for an eventual text trainer.
    """
    codec = tokenizer or ByteTokenizer()
    examples: list[MaskedTextExample] = []
    for chunk in plan_text_for_training(
        text,
        minimum_bytes=minimum_bytes,
        target_bytes=target_bytes,
        maximum_bytes=maximum_bytes,
        overlap_bytes=overlap_bytes,
        core=core,
    ):
        context = str(chunk["context"])
        prefix = int(chunk["context_prefix_bytes"])
        token_ids = codec.encode(context)
        if not token_ids:
            continue
        sequence = (codec.bos_id, *token_ids)
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        loss_mask = (0.0,) * prefix + (1.0,) * (len(token_ids) - prefix)
        if len(input_ids) != len(target_ids) or len(target_ids) != len(loss_mask):
            raise RuntimeError("Text mask contract is inconsistent")
        examples.append(
            MaskedTextExample(
                input_ids=input_ids,
                target_ids=target_ids,
                loss_mask=loss_mask,
                context_prefix_tokens=prefix,
                content_tokens=len(token_ids) - prefix,
            )
        )
    return examples


def batch_masked_text_examples(
    examples: Sequence[MaskedTextExample],
    *,
    tokenizer: ByteTokenizer | None = None,
    max_tokens: int | None = None,
) -> MaskedTextBatch:
    """Pad a bounded set of examples without dropping tokens or their mask."""
    if not examples:
        raise ValueError("At least one masked text example is required")
    codec = tokenizer or ByteTokenizer()
    width = max(len(item.input_ids) for item in examples)
    if max_tokens is not None:
        limit = int(max_tokens)
        if not 1 <= limit <= 16 * 1024 * 1024:
            raise ValueError("max_tokens must be between 1 and 16 MiB")
        if width > limit:
            raise ValueError("An example exceeds max_tokens; split it before batching")
        width = limit
    inputs = np.full((len(examples), width), codec.pad_id, dtype=np.int32)
    targets = np.full((len(examples), width), codec.pad_id, dtype=np.int32)
    loss = np.zeros((len(examples), width), dtype=np.float32)
    attention = np.zeros((len(examples), width), dtype=np.float32)
    for row, item in enumerate(examples):
        length = len(item.input_ids)
        inputs[row, :length] = item.input_ids
        targets[row, :length] = item.target_ids
        loss[row, :length] = item.loss_mask
        attention[row, :length] = 1.0
    return MaskedTextBatch(inputs, targets, loss, attention)


def iter_masked_text_batches(
    examples: Sequence[MaskedTextExample],
    batch_size: int,
    *,
    tokenizer: ByteTokenizer | None = None,
) -> Iterator[MaskedTextBatch]:
    """Yield deterministic, non-shuffled batches for caller-controlled training."""
    size = int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(examples), size):
        yield batch_masked_text_examples(examples[start : start + size], tokenizer=tokenizer)
