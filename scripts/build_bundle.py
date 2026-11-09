from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

MODEL_WEIGHT = "model.safetensors-00001-of-00001.safetensors"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_MODEL_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
FORBIDDEN_TOP_LEVEL = {
    ".env",
    ".venv",
    ".tools",
    ".cache",
    "artifacts",
    "checkpoints",
    "third_party",
    "tmp",
    ".pytest_cache",
    ".ruff_cache",
}
FORBIDDEN_ANY_SEGMENT = {
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "checkpoints",
}


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    return subprocess.check_output(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_results(source_root: Path, clone_root: Path) -> None:
    for relative in ("results/data/processed", "results/reports"):
        source = source_root / relative
        destination = clone_root / relative
        if not source.is_dir():
            raise FileNotFoundError(f"Missing final result directory: {source}")
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_model(source_root: Path, clone_root: Path) -> Path:
    source = source_root / "models" / "Qwen3.5-2B"
    destination = clone_root / "models" / "Qwen3.5-2B"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing model directory: {source}")
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".cache"),
    )
    return destination


def _relative_files(root: Path, *, exclude_parts: set[str] | None = None):
    excluded = exclude_parts or set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts)
    }


def _profile_snapshot(root: Path) -> dict:
    command = (
        'source "$1/scripts/lib_profile.sh"; '
        'load_training_profile "$1"; '
        'exec "$1/.venv/bin/python" "$1/scripts/run_metadata.py" validate'
    )
    output = _run(["bash", "-c", command, "_", str(root)])
    return json.loads(output)


def _validate_data(root: Path) -> dict[str, str]:
    expected = {
        "results/data/processed/sft/sft_uniform.parquet": 227,
        "results/data/processed/sft/sft_failure_aware.parquet": 225,
        "results/data/processed/rl/rl_uniform.parquet": 164,
        "results/data/processed/rl/rl_failure_aware.parquet": 328,
        "results/data/processed/rl_dev/rl_uniform.parquet": 10,
        "results/data/processed/rl_dev/rl_failure_aware.parquet": 20,
    }
    hashes = {}
    for relative, expected_rows in expected.items():
        path = root / relative
        table = pq.read_table(path)
        if table.num_rows != expected_rows:
            raise ValueError(
                f"{relative}: expected {expected_rows} rows, found {table.num_rows}"
            )
        hashes[relative] = _sha256(path)
    return hashes


def _assert_clean_clone(clone_root: Path) -> None:
    status = _run(["git", "-C", str(clone_root), "status", "--porcelain"])
    if status:
        raise ValueError(f"Bundle clone is not clean:\n{status}")
    if _run(
        ["git", "-C", str(clone_root), "rev-parse", "--is-shallow-repository"]
    ) != "true":
        raise ValueError("Bundle clone is not shallow")
    if (clone_root / ".git" / "modules").exists():
        raise ValueError("Bundle clone contains .git/modules")


def _forbidden_path(relative: Path) -> bool:
    if relative.parts and relative.parts[0] in FORBIDDEN_TOP_LEVEL:
        return True
    if relative.parts[:2] == (".git", "modules"):
        return True
    if any(part in FORBIDDEN_ANY_SEGMENT for part in relative.parts):
        return True
    name = relative.name
    return name.startswith("id_rsa") or name.endswith((".pem", ".key"))


def _secret_scan(root: Path, stage: Path) -> None:
    env_path = root / ".env"
    api_key = ""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip("\"'")
                break

    text_suffixes = {
        ".env",
        ".example",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    private_markers = tuple(
        f"{'-' * 5}BEGIN {key_type} PRIVATE KEY{'-' * 5}"
        for key_type in ("OPENSSH", "RSA", "")
    )
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(stage)
        if _forbidden_path(relative):
            raise ValueError(f"Forbidden path in training package: {relative}")
        is_known_text = path.suffix.lower() in text_suffixes or path.name in {
            "Makefile",
            ".gitignore",
        }
        is_small_git_text = (
            relative.parts[:1] == (".git",) and path.stat().st_size <= 8 * 1024 * 1024
        )
        if not is_known_text and not is_small_git_text:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if api_key and api_key in text:
            raise ValueError(f"API key found in bundle file: {relative}")
        if any(marker in text for marker in private_markers):
            raise ValueError(f"Private key material found in bundle file: {relative}")


def _manifest_file_list(stage: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            relative = path.relative_to(stage)
            entries.append(
                {
                    "path": str(relative),
                    "size_bytes": path.stat().st_size,
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(root), "status", "--porcelain"])
    if status:
        raise ValueError("Source repository must be committed and clean")

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else root / "tmp" / "bundles"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stage = output_root / "telecom-agent-post-training"
    archive = output_root / f"telecom_training_96gb_{commit}.tar.gz"
    checksum_file = output_root / (
        f"telecom_training_96gb_{commit}.tar.gz.sha256"
    )
    manifest_path = output_root / "bundle_manifest.json"
    for path in (stage, archive, checksum_file, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing bundle artifact: {path}")

    clone_url = f"file://{root}"
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            "--single-branch",
            clone_url,
            str(stage),
        ]
    )
    cloned_commit = _run(["git", "-C", str(stage), "rev-parse", "HEAD"])
    if cloned_commit != commit:
        raise ValueError(
            f"Shallow clone commit mismatch: expected {commit}, got {cloned_commit}"
        )

    _copy_results(root, stage)
    model_dir = _copy_model(root, stage)
    source_model_dir = root / "models" / "Qwen3.5-2B"
    source_model_files = _relative_files(source_model_dir, exclude_parts={".cache"})
    staged_model_files = _relative_files(model_dir)
    if source_model_files != staged_model_files:
        raise ValueError("Model file list does not match the source model")

    _assert_clean_clone(stage)
    data_hashes = _validate_data(stage)
    model_weight = model_dir / MODEL_WEIGHT
    model_hash = _sha256(model_weight)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(
            f"Model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}, got {model_hash}"
        )
    if not all(
        (model_dir / name).exists()
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            MODEL_WEIGHT,
        )
    ):
        raise ValueError("Model directory is incomplete")

    _secret_scan(root, stage)
    profile = _profile_snapshot(root)
    profile_hash = _sha256(root / "configs" / "profiles" / "96gb.env")

    _run(
        [
            "tar",
            "--sort=name",
            "-czf",
            str(archive),
            "-C",
            str(output_root),
            stage.name,
        ]
    )
    archive_hash = _sha256(archive)
    checksum_file.write_text(
        f"{archive_hash}  {archive.name}\n",
        encoding="utf-8",
    )

    listing = _run(["tar", "-tzf", str(archive)]).splitlines()
    for name in listing:
        relative = Path(name).relative_to(stage.name)
        if _forbidden_path(relative):
            raise ValueError(f"Forbidden archive path: {name}")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_short_commit": commit[:12],
        "git_shallow": True,
        "profile_file": "configs/profiles/96gb.env",
        "profile_sha256": profile_hash,
        "effective_profile": profile,
        "effective_config_hash": hashlib.sha256(
            json.dumps(
                profile,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": model_hash,
        "data_sha256": data_hashes,
        "archive": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "files": _manifest_file_list(stage),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Remove the staging checkout after the archive and manifest are finalized.
    shutil.rmtree(stage)

    print(f"ARCHIVE={archive}")
    print(f"ARCHIVE_SIZE_BYTES={archive.stat().st_size}")
    print(f"ARCHIVE_SHA256={archive_hash}")
    print(f"MODEL_WEIGHT_SHA256={model_hash}")
    print(f"CHECKSUM_FILE={checksum_file}")
    print(f"UPLOAD_MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
