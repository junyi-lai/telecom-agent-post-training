from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

TASK_ID_RE = re.compile(r"^\[([^\]]+)\](.*)\[PERSONA:([^\]]+)\]$")
ISSUE_ORDER = ("mms_issue", "mobile_data_issue", "service_issue")
EXTRA_QUOTAS = {
    "mms_issue": 42,
    "mobile_data_issue": 40,
    "service_issue": 18,
}


@dataclass(frozen=True)
class TaskIdentity:
    task_id: str
    issue: str
    faults: tuple[str, ...]
    persona: str

    @property
    def cluster(self) -> tuple[str, tuple[str, ...]]:
        return self.issue, tuple(sorted(self.faults))


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_task_id(task_id: str) -> TaskIdentity:
    match = TASK_ID_RE.match(task_id)
    if not match:
        raise ValueError(f"Unsupported telecom task id: {task_id}")
    issue, raw_faults, persona = match.groups()
    faults = tuple(part for part in raw_faults.split("|") if part)
    if not faults:
        raise ValueError(f"Task has no faults: {task_id}")
    return TaskIdentity(task_id=task_id, issue=issue, faults=faults, persona=persona)


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _stratified_pick(
    candidates: Iterable[str],
    quota: int,
    seed: int,
) -> list[str]:
    """Pick across issue/fault-count/persona strata before filling leftovers."""
    groups: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for task_id in candidates:
        identity = parse_task_id(task_id)
        fault_bucket = min(len(identity.faults), 4)
        groups[(identity.issue, fault_bucket, identity.persona)].append(task_id)

    for values in groups.values():
        values.sort(key=lambda item: _stable_rank(item, seed))

    keys = sorted(
        groups,
        key=lambda key: (
            ISSUE_ORDER.index(key[0]) if key[0] in ISSUE_ORDER else len(ISSUE_ORDER),
            key[1],
            key[2],
        ),
    )
    selected: list[str] = []
    offsets = {key: 0 for key in keys}
    while len(selected) < quota:
        progressed = False
        for key in keys:
            values = groups[key]
            offset = offsets[key]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
    return selected


def _pick_dev_tasks(train_ids: list[str], count: int, seed: int) -> list[str]:
    """Pick one task per fault signature so dev cannot share exact templates."""
    by_cluster: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for task_id in train_ids:
        by_cluster[parse_task_id(task_id).cluster].append(task_id)

    clusters = sorted(
        by_cluster,
        key=lambda cluster: _stable_rank(str(cluster), seed),
    )
    if len(clusters) < count:
        raise ValueError(f"Need {count} distinct dev clusters, found {len(clusters)}")
    return [
        sorted(by_cluster[cluster], key=lambda item: _stable_rank(item, seed))[0]
        for cluster in clusters[:count]
    ]


def build_manifest(tau2_repo: Path, seed: int = 20261103) -> dict:
    domain_dir = tau2_repo / "data" / "tau2" / "domains" / "telecom"
    tasks_path = domain_dir / "tasks.json"
    split_path = domain_dir / "split_tasks.json"

    tasks = load_json(tasks_path)
    splits = load_json(split_path)
    if not isinstance(tasks, list) or not isinstance(splits, dict):
        raise TypeError("Unexpected tau2 telecom data format")

    full_ids = [task["id"] for task in tasks]
    full_set = set(full_ids)
    split_ids = {
        name: list(values)
        for name, values in splits.items()
        if isinstance(values, list)
    }

    expected = {
        "small": 20,
        "train": 74,
        "test": 40,
        "base": 114,
        "full": 2285,
    }
    for name, count in expected.items():
        actual = len(split_ids.get(name, []))
        if actual != count:
            raise ValueError(f"{name}: expected {count} tasks, found {actual}")
        unknown = set(split_ids[name]) - full_set
        if unknown:
            raise ValueError(f"{name}: {len(unknown)} ids are absent from tasks.json")

    official_train = split_ids["train"]
    dev_ids = _pick_dev_tasks(official_train, count=10, seed=seed)
    dev_clusters = {parse_task_id(task_id).cluster for task_id in dev_ids}
    train_ids = [
        task_id
        for task_id in official_train
        if parse_task_id(task_id).cluster not in dev_clusters
    ]
    if len(train_ids) != 64:
        raise ValueError(
            f"Expected 64 official train ids after removing 10 dev ids; got {len(train_ids)}"
        )

    test_ids = split_ids["test"]
    base_set = set(split_ids["base"])
    test_clusters = {parse_task_id(task_id).cluster for task_id in test_ids}
    extra_candidates = [
        task_id
        for task_id in full_ids
        if task_id not in base_set
        and task_id not in test_ids
        and parse_task_id(task_id).cluster not in test_clusters
    ]
    candidates_by_issue: dict[str, list[str]] = defaultdict(list)
    for task_id in extra_candidates:
        candidates_by_issue[parse_task_id(task_id).issue].append(task_id)

    expanded_ids: list[str] = []
    for issue in ISSUE_ORDER:
        quota = EXTRA_QUOTAS[issue]
        available = candidates_by_issue[issue]
        picked = _stratified_pick(available, quota=quota, seed=seed + len(expanded_ids))
        if len(picked) != quota:
            raise ValueError(
                f"Expected {quota} extra {issue} tasks, only selected {len(picked)}"
            )
        expanded_ids.extend(picked)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "source": {
            "tasks_path": str(tasks_path.relative_to(tau2_repo)),
            "tasks_sha256": sha256_file(tasks_path),
            "split_path": str(split_path.relative_to(tau2_repo)),
            "split_sha256": sha256_file(split_path),
        },
        "splits": {
            "smoke": split_ids["small"][:5],
            "train": train_ids,
            "dev": dev_ids,
            "test": test_ids,
            "expanded_train": expanded_ids,
            "train_pool": train_ids + expanded_ids,
        },
        "task_metadata": {
            task_id: asdict(parse_task_id(task_id))
            for task_id in (
                split_ids["small"][:5] + train_ids + dev_ids + test_ids + expanded_ids
            )
        },
        "counts": {
            "smoke": 5,
            "train": len(train_ids),
            "dev": len(dev_ids),
            "test": len(test_ids),
            "expanded_train": len(expanded_ids),
            "train_pool": len(train_ids) + len(expanded_ids),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    splits = manifest["splits"]
    train_pool = set(splits["train_pool"])
    dev = set(splits["dev"])
    test = set(splits["test"])
    if len(train_pool) != manifest["counts"]["train_pool"]:
        raise ValueError("train_pool contains duplicate ids")
    if train_pool & dev:
        raise ValueError("train_pool and dev overlap")
    if train_pool & test:
        raise ValueError("train_pool and test overlap")
    if dev & test:
        raise ValueError("dev and test overlap")
    for split in ("smoke", "train", "dev", "test"):
        if not splits[split]:
            raise ValueError(f"{split} is empty")


def write_manifest(manifest: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
