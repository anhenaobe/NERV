"""Explicit recovery CLI for a fully validated failed embedding publication."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nerv.embeddings.artifacts import (
    recover_embedding_artifact,
    validate_recoverable_embedding_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the recovery-only command contract."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish a completed embedding temporary pair without "
            "recomputing vectors."
        )
    )
    parser.add_argument("--temporary-embeddings", type=Path, required=True)
    parser.add_argument("--temporary-manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Recover one proven pair and fail closed on any mismatch."""
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            manifest = validate_recoverable_embedding_artifact(
                args.temporary_embeddings,
                args.temporary_manifest,
                args.chunks,
            )
        else:
            manifest = recover_embedding_artifact(
                args.temporary_embeddings,
                args.temporary_manifest,
                args.embeddings,
                args.manifest,
                args.chunks,
            )
    except Exception as error:
        print(
            f"[NERV] embedding recovery: FAIL {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    action = "validation" if args.validate_only else "recovery"
    print(
        f"[NERV] embedding {action}: PASS "
        f"rows={manifest['row_count']} sha256={manifest['embeddings_sha256']}"
    )


if __name__ == "__main__":
    main()
