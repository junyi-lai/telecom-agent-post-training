from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from telecom_post_training.deepseek_client import DeepSeekClientConfig
from telecom_post_training.settings import validate_deepseek_base_url

expected_data = {
    "results/data/processed/sft/sft_uniform.parquet": (
        227,
        "f1eec24f562dbb7f26375b036fb2a136601f4e26f518057973e9a53abfb1d08d",
    ),
    "results/data/processed/sft/sft_failure_aware.parquet": (
        225,
        "b423e4657f0e1f57074751e2e478db0ecd259eb29c10e50b07c4c044107d653e",
    ),
    "results/data/processed/rl/rl_uniform.parquet": (
        164,
        "6a4d0519e9c474881a3bec22be70720df8cc5530b0840bd046ca869d3123e8b4",
    ),
    "results/data/processed/rl/rl_failure_aware.parquet": (
        328,
        "bb4e835b2394e90680310259d1e4b5f5665cf3ce0741121bc28f528b078e5351",
    ),
    "results/data/processed/rl_dev/rl_uniform.parquet": (
        10,
        "c90ed4568d8b73d6521e1e40844bfef32fcd80b2197c4ca17c088fca7b4d6823",
    ),
    "results/data/processed/rl_dev/rl_failure_aware.parquet": (
        20,
        "eb4ac087325251c48cf7cc0e36a363e8a98394c5973c709e3b950fcf5f10a60c",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-credentials", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or root / "tmp/artifacts/preflight.json"
    commit = _command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _command_output(["git", "-C", str(root), "status", "--porcelain"])
    if status:
        raise ValueError("Training requires a clean Git repository")
    git_shallow = _command_output(
        ["git", "-C", str(root), "rev-parse", "--is-shallow-repository"]
    ) == "true"

    manifest = json.loads(
        (root / "data/manifests/telecom_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["counts"] != {
        "smoke": 5,
        "train": 64,
        "dev": 10,
        "test": 40,
        "expanded_train": 100,
        "train_pool": 164,
    }:
        raise ValueError("Task manifest counts changed")

    data_hashes = {}
    for relative, (expected_rows, expected_hash) in expected_data.items():
        path = root / relative
        table = pq.read_table(path)
        if table.num_rows != expected_rows:
            raise ValueError(
                f"{relative}: expected {expected_rows} rows, found {table.num_rows}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{relative}: SHA256 does not match the frozen data")
        data_hashes[relative] = actual_hash

    model_dir = root / "models/Qwen3.5-2B"
    model_weight = model_dir / "model.safetensors-00001-of-00001.safetensors"
    expected_model_hash = (
        "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
    )
    if _sha256(model_weight) != expected_model_hash:
        raise ValueError("Model weight SHA256 does not match")

    profile = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(root / "scripts/run_metadata.py"),
                "validate",
            ],
            text=True,
        )
    )
    if profile["GPU_MEMORY_GB"] != 96:
        raise ValueError("Training profile is not the 96GB profile")

    if not args.skip_credentials:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is missing")
        validate_deepseek_base_url(os.environ.get("DEEPSEEK_BASE_URL", ""))
        deepseek_config = DeepSeekClientConfig.from_env()
        if not deepseek_config.fallback_enabled:
            raise ValueError(
                "DEEPSEEK fallback URL and API key must both be configured"
            )
        if os.getenv("ALLOW_INSECURE_DEEPSEEK_HTTP", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print(
                "WARNING: public plaintext HTTP is explicitly enabled",
                file=sys.stderr,
            )

    if not args.skip_runtime:
        expected_commits = {
            root / "third_party/tau2-bench": (
                "19f13ea3f5fba8b4c4d272d1353590f29dc53062"
            ),
            root / "third_party/verl": (
                "184687bb3b7c601627eed3660283ecabd2bb75ed"
            ),
        }
        for path, expected in expected_commits.items():
            actual = _command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
            if actual != expected:
                raise ValueError(f"Unexpected dependency commit at {path}")
        verl_patches = (
            root / "patches/verl-qwen35-sft-compat.patch",
            root / "patches/verl-agent-worker-round-robin.patch",
            root / "patches/verl-export-checkpoint-at-start.patch",
        )
        for verl_patch in verl_patches:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root / "third_party/verl"),
                    "apply",
                    "--reverse",
                    "--check",
                    str(verl_patch),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

    gpu_memory_mib = None
    if not args.skip_gpu:
        line = _command_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()[0]
        gpu_memory_mib = int(line.strip())
        if gpu_memory_mib < 90 * 1024:
            raise ValueError(
                f"GPU memory is below the 96GB profile class: {gpu_memory_mib} MiB"
            )

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_clean": True,
        "git_shallow": git_shallow,
        "profile": profile,
        "allow_insecure_public_http": (
            os.getenv("ALLOW_INSECURE_DEEPSEEK_HTTP", "").lower()
            in {"1", "true", "yes", "on"}
        ),
        "routing_config": {
            "primary_base_url": os.getenv("DEEPSEEK_BASE_URL", ""),
            "primary_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "primary_request_model": (
                os.getenv("DEEPSEEK_REQUEST_MODEL", "").strip()
                or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            ),
            "fallback_base_url": os.getenv(
                "DEEPSEEK_FALLBACK_BASE_URL",
                "",
            ),
            "fallback_model": os.getenv(
                "DEEPSEEK_FALLBACK_MODEL",
                "deepseek-flash",
            ),
            "fallback_enabled": bool(
                os.getenv("DEEPSEEK_FALLBACK_BASE_URL")
                and os.getenv("DEEPSEEK_FALLBACK_API_KEY")
            ),
            "fallback_api_key_present": bool(
                os.getenv("DEEPSEEK_FALLBACK_API_KEY")
            ),
        },
        "model_weight_sha256": expected_model_hash,
        "data_sha256": data_hashes,
        "gpu_memory_mib": gpu_memory_mib,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
