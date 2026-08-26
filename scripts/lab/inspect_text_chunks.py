"""Print native UTF-8 text chunk boundaries and context prefixes as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kernelyra.native_core import NativeCore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=768)
    parser.add_argument("--target-bytes", type=int, default=1536)
    parser.add_argument("--maximum-bytes", type=int, default=2048)
    parser.add_argument("--overlap-bytes", type=int, default=256)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    chunks = NativeCore().plan_text_chunks(
        text,
        minimum_bytes=args.minimum_bytes,
        target_bytes=args.target_bytes,
        maximum_bytes=args.maximum_bytes,
        overlap_bytes=args.overlap_bytes,
    )
    print(json.dumps({"chunks": chunks, "content_reconstructs_source": "".join(item["content"] for item in chunks) == text}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
