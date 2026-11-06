"""Alternate two benchmark_matmul_focus binaries; preserve raw samples.

Usage: python scripts/compare_matmul_runs.py BASELINE CANDIDATE OUTPUT_DIRECTORY
Append --ops for benchmark_ops_focus binaries.
Both binaries must use the same compiler, flags and benchmark driver.
"""

import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def main():
    if len(sys.argv) not in (4,5) or (len(sys.argv)==5 and sys.argv[4]!="--ops"):
        raise SystemExit(__doc__)
    ops = len(sys.argv)==5
    key_columns = ("Operation", "Shape", "Threads") if ops else ("Rows", "Inner", "Cols", "Threads")
    binaries = dict(zip(("baseline", "candidate"),
                        (Path(arg).resolve(strict=True) for arg in sys.argv[1:3])))
    output = Path(sys.argv[3]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "platform": platform.platform(), "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(), "rounds": 3,
        "order": ["baseline,candidate", "candidate,baseline", "baseline,candidate"],
        "binaries": {key: str(value) for key, value in binaries.items()},
        "OpenMP_environment": {k: v for k, v in os.environ.items()
                               if k.startswith("OMP_")},
        "timing": "Output allocation, initialization, compute and destruction",
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = {name: {} for name in binaries}
    expected_keys = None
    for round_number in range(1, 4):
        order = ("candidate", "baseline") if round_number == 2 else ("baseline", "candidate")
        for name in order:
            print(f"Round {round_number}: {name}", flush=True)
            stem = output / f"round{round_number}-{name}"
            with stem.with_suffix(".csv").open("w") as stdout, \
                    stem.with_suffix(".log").open("w") as stderr:
                subprocess.run([str(binaries[name])], cwd=output,
                               stdout=stdout, stderr=stderr, check=True)
            with stem.with_suffix(".csv").open() as source:
                rows = list(csv.DictReader(source))
            keys = set()
            for row in rows:
                key = tuple(row[k] for k in key_columns)
                if key in keys:
                    raise RuntimeError("Duplicate benchmark configuration")
                keys.add(key)
                samples = [float(row[f"Sample{i}_ms"]) for i in range(10)]
                if row["Correctness"] != "PASS" or row["Warmups"] != "3" or \
                        row["Measurements"] != "10" or \
                        not all(math.isfinite(x) and x > 0 for x in samples):
                    raise RuntimeError("Invalid benchmark data")
                results[name].setdefault(key, []).append(float(row["Median_ms"]))
            if not keys or (not ops and len(keys) != 20):
                raise RuntimeError("Unexpected focused configuration count")
            if expected_keys is not None and keys != expected_keys:
                raise RuntimeError("Baseline/candidate workloads differ")
            expected_keys = keys
    with (output / "summary.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow([*key_columns,
                         "Baseline_median_of_run_medians_ms", "Candidate_median_of_run_medians_ms",
                         "Runtime_reduction_percent", "Baseline_min_run_median_ms",
                         "Baseline_max_run_median_ms", "Candidate_min_run_median_ms",
                         "Candidate_max_run_median_ms"])
        for key in sorted(expected_keys, key=lambda x: x if ops else tuple(map(int, x))):
            base, candidate = (results[name][key] for name in ("baseline", "candidate"))
            a, b = statistics.median(base), statistics.median(candidate)
            writer.writerow([*key, a, b, (1 - b / a) * 100,
                             min(base), max(base), min(candidate), max(candidate)])
    print(f"Saved raw samples and summary to {output}", flush=True)


if __name__ == "__main__":
    main()
