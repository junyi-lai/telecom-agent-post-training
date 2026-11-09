import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_model",
    ROOT / "scripts" / "download_model.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_model(directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / module.MODEL_FILENAME).write_bytes(b"model")


def test_verify_model_directory_checks_hash(tmp_path, monkeypatch) -> None:
    _write_model(tmp_path)
    monkeypatch.setattr(module, "_sha256", lambda _: module.MODEL_SHA256)
    module.verify_model_directory(tmp_path)


def test_verify_model_directory_rejects_wrong_hash(tmp_path, monkeypatch) -> None:
    _write_model(tmp_path)
    monkeypatch.setattr(module, "_sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.verify_model_directory(tmp_path)


def test_download_uses_pinned_revision(tmp_path) -> None:
    captured = {}

    def fake_download(**kwargs) -> str:
        captured.update(kwargs)
        _write_model(tmp_path)
        return str(tmp_path)

    module.download_model(tmp_path, downloader=fake_download)
    assert captured["repo_id"] == module.MODEL_ID
    assert captured["revision"] == module.MODEL_REVISION
    assert captured["local_dir"] == tmp_path
