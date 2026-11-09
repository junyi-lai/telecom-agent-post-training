#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Callable

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_FILENAME = "model.safetensors-00001-of-00001.safetensors"
MODEL_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_directory(
    destination: Path,
    *,
    expected_sha256: str = MODEL_SHA256,
) -> None:
    config = destination / "config.json"
    weight = destination / MODEL_FILENAME
    if not config.is_file():
        raise FileNotFoundError(f"Missing model config: {config}")
    if not weight.is_file():
        raise FileNotFoundError(f"Missing model weight: {weight}")
    actual_sha256 = _sha256(weight)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Model weight SHA256 mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )


def download_model(
    destination: Path,
    *,
    downloader: Callable[..., str] | None = None,
) -> None:
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface-hub is unavailable; run bash scripts/setup_inference.sh"
            ) from error

        downloader = snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        token=os.getenv("HF_TOKEN") or None,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned Qwen student model."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=root / "models" / "Qwen3.5-2B",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip the network request and verify an existing model directory.",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not args.verify_only:
        download_model(destination)
    verify_model_directory(destination)
    print(f"Student model ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
