"""Bounded CUDA and real-encoder smoke used by setup_cuda_env.ps1."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"


def main() -> int:
    """Verify the proven Torch build, GPU identity, tensor math, and encoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-smoke", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expected-device-substring", default="NVIDIA")
    args = parser.parse_args()

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"expected Torch {EXPECTED_TORCH}, found {torch.__version__}."
        )
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"expected Torch CUDA {EXPECTED_TORCH_CUDA}, found {torch.version.cuda}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Torch reports no CUDA devices.")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_device_substring not in device_name:
        raise RuntimeError(
            "expected device containing "
            f"{args.expected_device_substring!r}, found {device_name!r}."
        )
    left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    product = left @ left
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product).all().item()):
        raise RuntimeError("tiny CUDA tensor smoke produced non-finite values.")

    if args.encoder_smoke:
        from nerv.embeddings.embedding_generator import generate_embeddings
        from nerv.embeddings.encoder import load_encoder

        encoder = load_encoder(
            device="cuda",
            local_files_only=args.local_files_only,
        )
        vectors = generate_embeddings(
            ["evidencia orbital estable"],
            encoder,
            batch_size=1,
            kind="passage",
        )
        if vectors.shape != (1, 384) or vectors.dtype != np.float32:
            raise RuntimeError(
                "unexpected encoder output: "
                f"shape={vectors.shape}, dtype={vectors.dtype}."
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("tiny encoder smoke produced non-finite values.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise RuntimeError("tiny encoder smoke output is not L2-normalized.")

    print(
        "[NERV CUDA SETUP] PASS "
        f"torch={torch.__version__} cuda={torch.version.cuda} gpu={device_name}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"[NERV CUDA SETUP] FAIL {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
