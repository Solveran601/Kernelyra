"""Bounded, streaming text extraction for the built-in text catalogue."""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import DatasetError
from .formats import FormatDescriptor, format_for_path
from .streaming import MAX_FOLDER_FILES

TEXT_CATEGORIES = {"plain-text", "source-code", "structured-text"}
DEFAULT_CHUNK_BYTES = 64 * 1024
DEFAULT_MAX_CHARS = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TextChunk:
    path: str
    format: str
    handler: str
    encoding: str
    index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def text_format_count() -> int:
    from .formats import BUILTIN_FORMATS

    return sum(item.category in TEXT_CATEGORIES for item in BUILTIN_FORMATS)


def _text_descriptor(path: Path) -> FormatDescriptor:
    descriptor = format_for_path(path)
    if descriptor is None or descriptor.category not in TEXT_CATEGORIES:
        raise DatasetError(f"'{path.suffix or path.name}' is not a built-in text extraction route")
    if descriptor.training not in {"extract", "train"}:
        raise DatasetError(f"Built-in decoder for '{descriptor.id}' is not available")
    return descriptor


def _encoding(sample: bytes) -> str:
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith(codecs.BOM_UTF16_LE) or sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    for candidate in ("utf-8", "cp1251"):
        try:
            sample.decode(candidate)
            return candidate
        except UnicodeDecodeError:
            continue
    return "latin-1"


def extract_text(
    path: str | Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[TextChunk]:
    """Yield decoded chunks without loading the file into memory.

    This extracts textual content; it does not claim language-model training.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise DatasetError("Text extraction requires a regular non-symlink file")
    descriptor = _text_descriptor(source)
    safe_limit = int(max_chars)
    safe_chunk = int(chunk_bytes)
    if not 1 <= safe_limit <= 1024 * 1024 * 1024:
        raise DatasetError("max_chars must be between 1 and 1 GiB")
    if not 1024 <= safe_chunk <= 4 * 1024 * 1024:
        raise DatasetError("chunk_bytes must be between 1 KiB and 4 MiB")

    with source.open("rb") as handle:
        sample = handle.read(min(safe_chunk, 64 * 1024))
        if b"\0" in sample and not sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            raise DatasetError("Text route contains binary NUL bytes")
        encoding = _encoding(sample)
        handle.seek(0)
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        emitted = 0
        index = 0
        while True:
            raw = handle.read(safe_chunk)
            final = not raw
            try:
                decoded = decoder.decode(raw, final=final)
            except UnicodeDecodeError as error:
                raise DatasetError(f"Cannot decode {descriptor.id} as text near byte {handle.tell()}: {error.reason}") from None
            if decoded:
                decoded = decoded.replace("\r\n", "\n").replace("\r", "\n")
                if emitted + len(decoded) > safe_limit:
                    decoded = decoded[: safe_limit - emitted]
                if decoded:
                    yield TextChunk(str(source), descriptor.id, descriptor.handler, encoding, index, decoded)
                    emitted += len(decoded)
                    index += 1
            if final or emitted >= safe_limit:
                return


def extract_folder(
    path: str | Path,
    *,
    max_files: int = MAX_FOLDER_FILES,
    max_total_chars: int = DEFAULT_MAX_CHARS,
) -> Iterator[TextChunk]:
    """Extract a deterministic whole folder while enforcing global limits."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise DatasetError("Folder extraction requires a regular non-symlink directory")
    safe_files = max(1, min(MAX_FOLDER_FILES, int(max_files)))
    remaining = max(1, min(1024 * 1024 * 1024, int(max_total_chars)))
    selected = 0
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix().casefold()):
        if item.is_symlink() or not item.is_file():
            continue
        descriptor = format_for_path(item)
        if descriptor is None or descriptor.category not in TEXT_CATEGORIES:
            continue
        selected += 1
        if selected > safe_files:
            raise DatasetError(f"Text folder exceeds the {safe_files} file limit")
        for chunk in extract_text(item, max_chars=remaining):
            yield chunk
            remaining -= len(chunk.text)
            if remaining <= 0:
                return
