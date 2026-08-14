"""Backwards-compatible entry point for the packaged NERV ingestion CLI."""

from __future__ import annotations

import sys

from nerv.ingestion import lector_corpus as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
