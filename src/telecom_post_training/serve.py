from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from telecom_post_training.settings import ROOT, Settings


def build_command(settings: Settings) -> list[str]:
    vllm = str(Path(sys.executable).with_name("vllm"))
    model_path = (
        str(settings.student_model_path)
        if settings.student_model_path.exists()
        else settings.student_model
    )
    command = [
        vllm,
        "serve",
        model_path,
        "--served-model-name",
        settings.vllm_model_alias,
        "--host",
        os.getenv("VLLM_HOST", "127.0.0.1"),
        "--port",
        os.getenv("VLLM_PORT", "8000"),
        "--max-model-len",
        os.getenv("VLLM_MAX_MODEL_LEN", "8192"),
        "--gpu-memory-utilization",
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.72"),
        "--max-num-seqs",
        os.getenv("VLLM_MAX_NUM_SEQS", "4"),
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        os.getenv("VLLM_TOOL_CALL_PARSER", "qwen3_coder"),
        "--reasoning-parser",
        os.getenv("VLLM_REASONING_PARSER", "qwen3"),
        "--language-model-only",
    ]
    if not settings.student_model_path.exists():
        command.extend(["--revision", settings.student_model_revision])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    command = build_command(Settings.from_env())
    if args.dry_run:
        print(" ".join(command))
        return 0
    environment = os.environ.copy()
    # Required on the Blackwell RTX 50 series with CUDA 13.
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
