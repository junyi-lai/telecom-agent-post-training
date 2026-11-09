from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

INFRASTRUCTURE_TERMINATIONS = frozenset(
    {
        "context_window_exceeded",
        "infrastructure_error",
        "timeout",
        "unexpected_error",
    }
)


@dataclass(frozen=True)
class EvaluationValidation:
    valid: bool
    simulations: int
    unique_tasks: int
    infrastructure_errors: int
    missing_tasks: list[str]
    unexpected_tasks: list[str]
    duplicate_tasks: list[str]
    termination_counts: dict[str, int]
    errors: list[str]


def validate_evaluation(
    results_path: Path,
    *,
    expected_task_ids: Iterable[str],
) -> EvaluationValidation:
    """Validate one single-seed evaluator result before it is accepted."""

    expected = tuple(str(task_id) for task_id in expected_task_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise ValueError("expected_task_ids must be unique")

    with results_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected result root in {results_path}")

    raw_simulations = payload.get("simulations")
    simulations = raw_simulations if isinstance(raw_simulations, list) else []
    task_ids: list[str] = []
    infrastructure_error_count = 0
    termination_counts: Counter[str] = Counter()
    errors: list[str] = []

    if not isinstance(raw_simulations, list):
        errors.append("results.json does not contain a simulations list")
    if len(simulations) != len(expected):
        errors.append(
            f"expected {len(expected)} simulations, found {len(simulations)}"
        )

    for index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            errors.append(f"simulation {index} is not an object")
            continue
        task_id = simulation.get("task_id")
        if task_id is None:
            errors.append(f"simulation {index} has no task_id")
        else:
            task_ids.append(str(task_id))

        termination = str(simulation.get("termination_reason") or "")
        termination_counts[termination] += 1
        info = simulation.get("info")
        error_type = info.get("error_type") if isinstance(info, dict) else None
        if termination in INFRASTRUCTURE_TERMINATIONS or error_type:
            infrastructure_error_count += 1

    actual_set = set(task_ids)
    task_counts = Counter(task_ids)
    missing_tasks = sorted(expected_set - actual_set)
    unexpected_tasks = sorted(actual_set - expected_set)
    duplicate_tasks = sorted(
        task_id for task_id, count in task_counts.items() if count > 1
    )

    if missing_tasks:
        errors.append(f"missing tasks: {len(missing_tasks)}")
    if unexpected_tasks:
        errors.append(f"unexpected tasks: {len(unexpected_tasks)}")
    if duplicate_tasks:
        errors.append(f"duplicate tasks: {len(duplicate_tasks)}")
    if infrastructure_error_count:
        errors.append(f"infrastructure errors: {infrastructure_error_count}")

    return EvaluationValidation(
        valid=not errors,
        simulations=len(simulations),
        unique_tasks=len(actual_set),
        infrastructure_errors=infrastructure_error_count,
        missing_tasks=missing_tasks,
        unexpected_tasks=unexpected_tasks,
        duplicate_tasks=duplicate_tasks,
        termination_counts=dict(sorted(termination_counts.items())),
        errors=errors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate one seed of a frozen tau2 evaluation run."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/telecom_manifest.json"),
    )
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_task_ids = manifest["splits"][args.split]
    result = validate_evaluation(
        args.input,
        expected_task_ids=expected_task_ids,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
