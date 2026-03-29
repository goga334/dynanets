from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dynanets.benchmark import (
    aggregate_benchmark_runs,
    run_benchmark_config,
    write_aggregate_csv,
    write_aggregate_json,
    write_aggregate_markdown,
    write_benchmark_plots,
    write_runs_csv,
    write_runs_json,
)
from dynanets.protocol import BenchmarkProtocol



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark protocol manifest.")
    parser.add_argument("protocol", help="Path to a benchmark protocol YAML manifest")
    parser.add_argument("--output-dir", help="Directory for benchmark artifacts")
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    protocol = BenchmarkProtocol.from_yaml(protocol_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("reports") / protocol.name
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_experiments = protocol.resolve_configs(protocol_path.parent)
    runs = []
    for experiment in resolved_experiments:
        for seed in protocol.seeds:
            run = run_benchmark_config(experiment.config, seed)
            run["protocol_name"] = protocol.name
            run["protocol_track"] = protocol.track
            run["protocol_role"] = experiment.role
            run["protocol_tags"] = list(experiment.tags)
            runs.append(run)

    aggregate = aggregate_benchmark_runs(runs)
    write_runs_json(output_dir / "runs.json", runs)
    write_runs_csv(output_dir / "runs.csv", runs)
    write_aggregate_json(output_dir / "summary.json", aggregate)
    write_aggregate_csv(output_dir / "summary.csv", aggregate)
    write_benchmark_plots(output_dir, aggregate)
    write_aggregate_markdown(output_dir / "summary.md", aggregate, protocol.seeds)
    (output_dir / "protocol.json").write_text(json.dumps(asdict(protocol), indent=2), encoding="utf-8")

    print(output_dir / "summary.md")
    print(f"protocol={protocol.name}")
    print(f"track={protocol.track}")
    print(f"experiments={len(resolved_experiments)}")


if __name__ == "__main__":
    main()
