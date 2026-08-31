"""Streaming, conversation-aware text preparation.

This module is deliberately a *preparation* API.  It neither trains an LLM
nor writes a dataset, a workspace or a checkpoint by itself.  It reads one
message at a time from supported text exports and yields chunks that never
split a source message.  A caller decides whether, where and how to persist
the yielded chunks.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

ConversationFormat = Literal["auto", "jsonl", "telegram_json", "plain_text"]
TopicDetector = Callable[[Sequence["ConversationMessage"], "ConversationMessage"], bool]

_TOKEN_PATTERN = re.compile(r"[^\W_]{2,}", re.UNICODE)
_TELEGRAM_MESSAGES_ARRAY = re.compile(r'"messages"\s*:\s*\[', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """A single unsplit input message.

    ``conversation_id`` is the context boundary.  The chunker never carries
    messages or topic history from one id into another one.
    """

    conversation_id: str
    text: str
    sequence: int
    timestamp: datetime | None = None
    author: str | None = None
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationChunk:
    """A contiguous group of whole messages prepared for a downstream trainer."""

    conversation_id: str
    text: str
    messages: tuple[ConversationMessage, ...]
    boundary: str
    topic_similarity: float | None = None

    @property
    def character_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSONL-safe representation without writing it anywhere."""
        return {
            "conversation_id": self.conversation_id,
            "text": self.text,
            "character_count": self.character_count,
            "message_count": len(self.messages),
            "boundary": self.boundary,
            "topic_similarity": self.topic_similarity,
            "messages": [
                {
                    "sequence": item.sequence,
                    "message_id": item.message_id,
                    "author": item.author,
                    "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                    "text": item.text,
                }
                for item in self.messages
            ],
        }


def _as_text(value: Any) -> str:
    """Normalize common Telegram/JSON message values without guessing schemas."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(part for item in value if (part := _as_text(item)))
    if isinstance(value, dict):
        for key in ("text", "content", "message", "body", "value"):
            if key in value:
                return _as_text(value[key])
    return ""


def _first_value(record: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _message_from_record(
    record: dict[str, Any],
    *,
    sequence: int,
    fallback_conversation_id: str,
    conversation_fields: Sequence[str],
    text_fields: Sequence[str],
    timestamp_fields: Sequence[str],
    author_fields: Sequence[str],
    message_id_fields: Sequence[str],
) -> ConversationMessage | None:
    text = _as_text(_first_value(record, text_fields))
    if not text:
        return None
    conversation_id = _first_value(record, conversation_fields)
    return ConversationMessage(
        conversation_id=str(conversation_id or fallback_conversation_id),
        text=text,
        sequence=sequence,
        timestamp=_parse_timestamp(_first_value(record, timestamp_fields)),
        author=(str(value) if (value := _first_value(record, author_fields)) is not None else None),
        message_id=(str(value) if (value := _first_value(record, message_id_fields)) is not None else None),
    )


def iter_jsonl_messages(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    conversation_fields: Sequence[str] = ("conversation_id", "chat_id", "dialog_id", "peer_id"),
    text_fields: Sequence[str] = ("text", "message", "content", "body"),
    timestamp_fields: Sequence[str] = ("timestamp", "date", "created_at", "time"),
    author_fields: Sequence[str] = ("author", "from", "sender", "from_id"),
    message_id_fields: Sequence[str] = ("message_id", "id"),
) -> Iterator[ConversationMessage]:
    """Read JSONL/NDJSON line-by-line; invalid lines are reported, never ignored."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding=encoding, newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from None
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record at line {line_number} must be an object")
            message = _message_from_record(
                payload,
                sequence=line_number,
                fallback_conversation_id=source.stem,
                conversation_fields=conversation_fields,
                text_fields=text_fields,
                timestamp_fields=timestamp_fields,
                author_fields=author_fields,
                message_id_fields=message_id_fields,
            )
            if message is not None:
                yield message


def _iter_telegram_records(path: Path, *, encoding: str, block_chars: int = 128 * 1024) -> Iterator[dict[str, Any]]:
    """Incrementally decode the ``messages`` array from Telegram ``result.json``.

    ``json.load`` is intentionally avoided: a multi-gigabyte export must not
    be copied into RAM merely to find message boundaries.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    array_started = False
    eof = False
    with path.open("r", encoding=encoding, newline="") as handle:
        while True:
            if not eof and (not array_started or len(buffer) < block_chars):
                fragment = handle.read(block_chars)
                if fragment:
                    buffer += fragment
                else:
                    eof = True
            if not array_started:
                match = _TELEGRAM_MESSAGES_ARRAY.search(buffer)
                if match is None:
                    if eof:
                        raise ValueError("Telegram JSON export does not contain a messages array")
                    if len(buffer) > 4 * block_chars:
                        raise ValueError("Telegram JSON header is unexpectedly large")
                    continue
                buffer = buffer[match.end() :]
                array_started = True
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if not buffer:
                if eof:
                    raise ValueError("Telegram JSON messages array ended unexpectedly")
                continue
            try:
                value, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Telegram JSON contains an incomplete message record") from None
                fragment = handle.read(block_chars)
                if fragment:
                    buffer += fragment
                else:
                    eof = True
                continue
            buffer = buffer[consumed:]
            if not isinstance(value, dict):
                raise ValueError("Telegram JSON messages array must contain objects")
            yield value


def iter_telegram_messages(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    conversation_id: str | None = None,
) -> Iterator[ConversationMessage]:
    """Stream textual messages from a Telegram Desktop JSON export."""
    source = Path(path).expanduser().resolve()
    default_id = conversation_id or source.stem
    for sequence, record in enumerate(_iter_telegram_records(source, encoding=encoding), start=1):
        if record.get("type") not in {None, "message"}:
            continue
        message = _message_from_record(
            record,
            sequence=sequence,
            fallback_conversation_id=default_id,
            conversation_fields=("conversation_id", "chat_id", "peer_id"),
            text_fields=("text", "message", "content"),
            timestamp_fields=("date", "timestamp", "created_at"),
            author_fields=("from", "author", "sender", "from_id"),
            message_id_fields=("id", "message_id"),
        )
        if message is not None:
            yield message


def iter_plain_text_messages(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    conversation_id: str | None = None,
) -> Iterator[ConversationMessage]:
    """Yield non-empty lines from a plain-text corpus without loading the file.

    Plain text has no reliable universal conversation schema.  Therefore every
    non-empty source line is an indivisible message; blank lines are retained
    only as natural separators through the surrounding chunk boundaries.
    """
    source = Path(path).expanduser().resolve()
    identifier = conversation_id or source.stem
    with source.open("r", encoding=encoding, newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                yield ConversationMessage(identifier, text, line_number)


def _detect_format(path: Path, selected: ConversationFormat) -> ConversationFormat:
    if selected != "auto":
        return selected
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "telegram_json"
    if suffix in {".txt", ".md", ".log"}:
        return "plain_text"
    raise ValueError(
        f"No streaming conversation reader is installed for '{suffix or 'this file type'}'. "
        "Supported: .txt, .md, .log, .jsonl, .ndjson and Telegram result.json."
    )


def iter_conversation_messages(
    path: str | Path,
    *,
    format: ConversationFormat = "auto",
    encoding: str = "utf-8",
    conversation_id: str | None = None,
) -> Iterator[ConversationMessage]:
    """Select a real streaming reader; no extension is treated as support by itself."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Conversation source was not found: {source}")
    selected = _detect_format(source, format)
    if selected == "jsonl":
        yield from iter_jsonl_messages(source, encoding=encoding)
        return
    if selected == "telegram_json":
        yield from iter_telegram_messages(source, encoding=encoding, conversation_id=conversation_id)
        return
    if selected == "plain_text":
        yield from iter_plain_text_messages(source, encoding=encoding, conversation_id=conversation_id)
        return
    raise ValueError(f"Unknown conversation format: {selected}")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in _TOKEN_PATTERN.findall(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class ConversationChunker:
    """Build bounded chunks while preserving message and conversation boundaries.

    Topic changes use a deterministic lexical Jaccard heuristic.  It is a
    useful baseline for large exports, not a claim of human-level semantic
    understanding.  Applications that own a semantic classifier can pass it
    as ``topic_detector`` and retain the same no-split contract.
    """

    def __init__(
        self,
        *,
        maximum_characters: int = 8_192,
        minimum_characters: int = 1_024,
        topic_similarity_threshold: float = 0.18,
        topic_window_messages: int = 4,
        topic_gap_seconds: float | None = 1_800,
        topic_detector: TopicDetector | None = None,
    ):
        maximum = int(maximum_characters)
        minimum = int(minimum_characters)
        if not 64 <= minimum <= maximum <= 16 * 1024 * 1024:
            raise ValueError("Require 64 <= minimum_characters <= maximum_characters <= 16 MiB")
        if not 0.0 <= float(topic_similarity_threshold) <= 1.0:
            raise ValueError("topic_similarity_threshold must be between 0 and 1")
        if not 1 <= int(topic_window_messages) <= 128:
            raise ValueError("topic_window_messages must be between 1 and 128")
        if topic_gap_seconds is not None and float(topic_gap_seconds) < 0:
            raise ValueError("topic_gap_seconds must be non-negative or None")
        self.maximum_characters = maximum
        self.minimum_characters = minimum
        self.topic_similarity_threshold = float(topic_similarity_threshold)
        self.topic_window_messages = int(topic_window_messages)
        self.topic_gap_seconds = None if topic_gap_seconds is None else float(topic_gap_seconds)
        self.topic_detector = topic_detector

    @staticmethod
    def _render(messages: Sequence[ConversationMessage]) -> str:
        return "\n".join(message.text for message in messages)

    def iter_chunks(self, messages: Iterable[ConversationMessage]) -> Iterator[ConversationChunk]:
        active: list[ConversationMessage] = []
        active_tokens: deque[frozenset[str]] = deque(maxlen=self.topic_window_messages)
        conversation_id: str | None = None
        previous_timestamp: datetime | None = None
        active_characters = 0

        def emit(boundary: str, similarity: float | None = None) -> ConversationChunk | None:
            if not active or conversation_id is None:
                return None
            return ConversationChunk(
                conversation_id=conversation_id,
                text=self._render(active),
                messages=tuple(active),
                boundary=boundary,
                topic_similarity=similarity,
            )

        for message in messages:
            if not isinstance(message, ConversationMessage):
                raise TypeError("ConversationChunker expects ConversationMessage values")
            if conversation_id is not None and message.conversation_id != conversation_id:
                chunk = emit("conversation_end")
                if chunk is not None:
                    yield chunk
                active.clear()
                active_tokens.clear()
                previous_timestamp = None
                active_characters = 0
            if conversation_id != message.conversation_id:
                conversation_id = message.conversation_id

            incoming_tokens = _tokens(message.text)
            prior_tokens = frozenset().union(*active_tokens) if active_tokens else frozenset()
            similarity = _jaccard(prior_tokens, incoming_tokens)
            rendered_length = active_characters
            time_gap = (
                previous_timestamp is not None
                and message.timestamp is not None
                and self.topic_gap_seconds is not None
                and (message.timestamp - previous_timestamp).total_seconds() > self.topic_gap_seconds
            )
            detected_topic_change = bool(
                active
                and (
                    time_gap
                    or (
                        self.topic_detector is not None
                        and self.topic_detector(tuple(active[-self.topic_window_messages :]), message)
                    )
                    or (
                        self.topic_detector is None
                        and rendered_length >= self.minimum_characters
                        and len(prior_tokens) >= 3
                        and len(incoming_tokens) >= 3
                        and similarity < self.topic_similarity_threshold
                    )
                )
            )
            exceeds_maximum = bool(active and rendered_length + 1 + len(message.text) > self.maximum_characters)
            if detected_topic_change or exceeds_maximum:
                boundary = "topic_shift" if detected_topic_change else "size_limit"
                chunk = emit(boundary, similarity if detected_topic_change else None)
                if chunk is not None:
                    yield chunk
                active.clear()
                active_tokens.clear()
                active_characters = 0
            if active:
                active_characters += 1  # newline inserted by _render
            active.append(message)
            active_characters += len(message.text)
            active_tokens.append(incoming_tokens)
            previous_timestamp = message.timestamp
            if len(message.text) > self.maximum_characters:
                chunk = emit("oversize_message")
                if chunk is not None:
                    yield chunk
                active.clear()
                active_tokens.clear()
                active_characters = 0
        chunk = emit("end_of_input")
        if chunk is not None:
            yield chunk


def iter_conversation_chunks(
    path: str | Path,
    *,
    format: ConversationFormat = "auto",
    encoding: str = "utf-8",
    conversation_id: str | None = None,
    **chunk_options: Any,
) -> Iterator[ConversationChunk]:
    """Create a no-write chunk stream from one supported text source."""
    messages = iter_conversation_messages(
        path,
        format=format,
        encoding=encoding,
        conversation_id=conversation_id,
    )
    yield from ConversationChunker(**chunk_options).iter_chunks(messages)


def write_conversation_chunks(
    chunks: Iterable[ConversationChunk],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write JSONL only to the caller-selected path.

    Existing files are protected unless the caller explicitly passes
    ``overwrite=True``.  This is the only function in this module that writes
    to disk; planning and iteration remain side-effect-free.
    """
    output = Path(destination).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing chunk file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return output
