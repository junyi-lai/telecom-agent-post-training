from pathlib import Path

from telecom_post_training.manifest import build_manifest, validate_manifest


def test_manifest_invariants() -> None:
    manifest = build_manifest(Path("third_party/tau2-bench"))
    validate_manifest(manifest)

    assert manifest["counts"] == {
        "smoke": 5,
        "train": 64,
        "dev": 10,
        "test": 40,
        "expanded_train": 100,
        "train_pool": 164,
    }
    assert not set(manifest["splits"]["train_pool"]) & set(manifest["splits"]["test"])
