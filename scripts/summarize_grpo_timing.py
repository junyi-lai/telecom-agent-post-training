from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_METRICS = (
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/testing",
)


def _steps(log_path: Path) -> dict[int, dict[str, float]]:
    text = log_path.read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    text = re.sub(r"\x1b\[[0-9;]*[mK]", "", text)
    starts = list(re.finditer(r"step:(\d+) -", text))
    result: dict[int, dict[str, float]] = {}
    for index, match in enumerate(starts):
        end = (
            starts[index + 1].start()
            if index + 1 < len(starts)
            else (min(len(text), match.start() + 50000))
        )
        chunk = text[match.start() : end]
        values: dict[str, float] = {}
        for metric in _METRICS:
            found = re.search(
                re.escape(metric) + r":(?:np\.float64\()?([-+0-9.eE]+)",
                chunk,
            )
            if found:
                values[metric] = float(found.group(1))
        if values:
            result[int(match.group(1))] = values
    return result


def _telemetry(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    events = Counter(str(row.get("event")) for row in rows)
    channels = Counter(str(row.get("channel")) for row in rows)
    statuses = Counter(str(row.get("status_code")) for row in rows)
    attempts = events.get("success", 0) + events.get("failure", 0)
    fallback_attempts = events.get("fallback", 0)
    return {
        "events": dict(events),
        "channels": dict(channels),
        "status_codes": dict(statuses),
        "timeouts": sum(bool(row.get("timeout")) for row in rows),
        "final_failures": events.get("failure", 0),
        "fallback_attempts": fallback_attempts,
        "fallback_rate": (fallback_attempts / attempts if attempts else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--cpu-summary", type=Path)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_step = {
        step: values
        for step, values in _steps(args.train_log).items()
        if args.start_step < step <= args.end_step
    }
    timing = {}
    for metric in _METRICS:
        values = [step[metric] for step in by_step.values() if metric in step]
        timing[metric] = {
            "mean_seconds": sum(values) / len(values) if values else 0.0,
            "min_seconds": min(values) if values else 0.0,
            "max_seconds": max(values) if values else 0.0,
        }
    cpu_summary = {}
    if args.cpu_summary and args.cpu_summary.exists():
        cpu_summary = json.loads(args.cpu_summary.read_text(encoding="utf-8"))
    result = {
        "start_step": args.start_step,
        "end_step": args.end_step,
        "completed_steps": sorted(by_step),
        "step_count": len(by_step),
        "timing": timing,
        "cpu": cpu_summary,
        "api": _telemetry(args.telemetry) if args.telemetry else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
