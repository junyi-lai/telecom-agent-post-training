from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _cpu_totals(line: str) -> tuple[int, int]:
    fields = [int(value) for value in line.split()[1:]]
    total = sum(fields)
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return total, idle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[float] = []
    started = time.monotonic()
    previous_total, previous_idle = _cpu_totals(
        Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    )
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write("timestamp_epoch,utilization_percent\n")
        while (
            not args.stop_file.exists()
            and time.monotonic() - started < args.max_seconds
        ):
            time.sleep(args.interval)
            total, idle = _cpu_totals(
                Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            )
            delta_total = total - previous_total
            delta_idle = idle - previous_idle
            utilization = (
                100.0 * (1.0 - delta_idle / delta_total) if delta_total > 0 else 0.0
            )
            samples.append(utilization)
            stream.write(f"{time.time():.3f},{utilization:.3f}\n")
            stream.flush()
            previous_total, previous_idle = total, idle

    samples.sort()
    summary = {
        "samples": len(samples),
        "mean_percent": sum(samples) / len(samples) if samples else 0.0,
        "p95_percent": (
            samples[min(len(samples) - 1, int(len(samples) * 0.95))] if samples else 0.0
        ),
        "max_percent": samples[-1] if samples else 0.0,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
