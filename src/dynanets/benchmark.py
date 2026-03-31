from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.reporting import mermaid_architecture
from dynanets.runtime import set_global_seed


DEFAULT_SEEDS = [7, 11, 23, 42, 99]
PLOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed dynanets benchmarks and aggregate results.")
    parser.add_argument("configs", nargs="+", help="Paths to YAML experiment configs")
    parser.add_argument("--output-dir", default="reports/benchmark_summary", help="Directory for benchmark artifacts")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS, help="Seeds to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for config_path in args.configs:
        for seed in args.seeds:
            runs.append(run_benchmark_config(config_path, seed))

    aggregate = aggregate_benchmark_runs(runs)

    write_runs_json(output_dir / "runs.json", runs)
    write_runs_csv(output_dir / "runs.csv", runs)
    write_aggregate_json(output_dir / "summary.json", aggregate)
    write_aggregate_csv(output_dir / "summary.csv", aggregate)
    write_benchmark_plots(output_dir, aggregate)
    write_aggregate_markdown(output_dir / "summary.md", aggregate, args.seeds)

    print(output_dir / "summary.md")
    for item in aggregate:
        print(
            f"{item['name']}: mean_final_val_accuracy={item['mean_final_val_accuracy']:.4f}, "
            f"std_final_val_accuracy={item['std_final_val_accuracy']:.4f}, runs={item['num_runs']}"
        )


def run_benchmark_config(config_path: str, seed: int) -> dict[str, Any]:
    config = ExperimentConfig.from_yaml(config_path)
    config = with_seed(config, seed)
    set_global_seed(seed)
    registries = default_registries()
    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
        workflows=registries["workflows"],
    )
    experiment = builder.build(config)
    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)
    run = result.to_report_item()
    run["benchmark_seed"] = seed
    return run


def with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": config.name,
            "dataset": {
                "name": config.dataset.name,
                "params": {**config.dataset.params, "seed": seed},
            },
            "model": {
                "name": config.model.name,
                "params": dict(config.model.params),
            },
            "metrics": [{"name": item.name, "params": dict(item.params)} for item in config.metrics],
            "adaptation": (
                {"name": config.adaptation.name, "params": dict(config.adaptation.params)} if config.adaptation is not None else None
            ),
            "search": (
                {"name": config.search.name, "params": dict(config.search.params)} if config.search is not None else None
            ),
            "workflow": (
                {"name": config.workflow.name, "params": dict(config.workflow.params)} if config.workflow is not None else None
            ),
            "trainer": dict(config.trainer),
            "runtime": {**config.runtime, "seed": seed},
        }
    )


def aggregate_benchmark_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["name"]].append(run)

    aggregate: list[dict[str, Any]] = []
    for name, group in grouped.items():
        representative = max(group, key=lambda item: (item["best_val_accuracy"], item["final_val_accuracy"]))
        final_scores = [float(item["final_val_accuracy"]) for item in group]
        best_scores = [float(item["best_val_accuracy"]) for item in group]
        adaptations = [int(item["adaptations_applied"]) for item in group]
        hidden_dims = [item["final_hidden_dim"] for item in group if item["final_hidden_dim"] is not None]
        parameter_counts = [int(item["constraints"]["parameter_count"]) for item in group if item.get("constraints")]
        nonzero_parameter_counts = [
            int(item["constraints"]["nonzero_parameter_count"])
            for item in group
            if item.get("constraints")
        ]
        weight_sparsities = [
            float(item["constraints"]["weight_sparsity"])
            for item in group
            if item.get("constraints")
        ]
        forward_flop_proxies = [
            int(item["constraints"]["forward_flop_proxy"])
            for item in group
            if item.get("constraints")
        ]
        activation_elements = [
            int(item["constraints"]["activation_elements"])
            for item in group
            if item.get("constraints")
        ]
        aggregate.append(
            {
                "name": name,
                "method_type": representative["metadata"].get("method_type", "train"),
                "notes": representative["metadata"].get("notes", ""),
                "runtime_environment": representative["metadata"].get("runtime_environment", {}),
                "num_runs": len(group),
                "seeds": [int(item["benchmark_seed"]) for item in group],
                "mean_final_val_accuracy": mean(final_scores),
                "std_final_val_accuracy": pstdev(final_scores) if len(final_scores) > 1 else 0.0,
                "mean_best_val_accuracy": mean(best_scores),
                "std_best_val_accuracy": pstdev(best_scores) if len(best_scores) > 1 else 0.0,
                "mean_adaptations_applied": mean(adaptations),
                "mean_final_hidden_dim": mean(hidden_dims) if hidden_dims else None,
                "mean_parameter_count": mean(parameter_counts) if parameter_counts else None,
                "mean_nonzero_parameter_count": mean(nonzero_parameter_counts) if nonzero_parameter_counts else None,
                "mean_weight_sparsity": mean(weight_sparsities) if weight_sparsities else None,
                "mean_forward_flop_proxy": mean(forward_flop_proxies) if forward_flop_proxies else None,
                "mean_activation_elements": mean(activation_elements) if activation_elements else None,
                "best_seed": int(representative["benchmark_seed"]),
                "best_seed_final_val_accuracy": float(representative["final_val_accuracy"]),
                "best_seed_best_val_accuracy": float(representative["best_val_accuracy"]),
                "representative_architecture_spec": representative.get("architecture_spec"),
                "representative_architecture_graph": representative.get("architecture_graph"),
                "representative_constraints": representative.get("constraints", {}),
                "representative_stage_history": representative.get("stage_history", []),
                "seed_runs": [
                    {
                        "seed": int(item["benchmark_seed"]),
                        "final_val_accuracy": float(item["final_val_accuracy"]),
                        "best_val_accuracy": float(item["best_val_accuracy"]),
                        "adaptations_applied": int(item["adaptations_applied"]),
                        "parameter_count": item.get("constraints", {}).get("parameter_count"),
                        "nonzero_parameter_count": item.get("constraints", {}).get("nonzero_parameter_count"),
                        "weight_sparsity": item.get("constraints", {}).get("weight_sparsity"),
                    }
                    for item in sorted(group, key=lambda item: int(item["benchmark_seed"]))
                ],
            }
        )

    aggregate.sort(key=lambda item: item["mean_final_val_accuracy"], reverse=True)
    return aggregate


def write_runs_json(path: Path, runs: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def write_runs_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [
        {
            "name": item["name"],
            "seed": item["benchmark_seed"],
            "final_val_accuracy": item["final_val_accuracy"],
            "best_val_accuracy": item["best_val_accuracy"],
            "adaptations_applied": item["adaptations_applied"],
            "final_hidden_dim": item["final_hidden_dim"],
            "parameter_count": item.get("constraints", {}).get("parameter_count"),
            "nonzero_parameter_count": item.get("constraints", {}).get("nonzero_parameter_count"),
            "weight_sparsity": item.get("constraints", {}).get("weight_sparsity"),
            "forward_flop_proxy": item.get("constraints", {}).get("forward_flop_proxy"),
            "activation_elements": item.get("constraints", {}).get("activation_elements"),
            "method_type": item["metadata"].get("method_type", "train"),
            "notes": item["metadata"].get("notes", ""),
        }
        for item in runs
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_json(path: Path, aggregate: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def write_aggregate_csv(path: Path, aggregate: list[dict[str, Any]]) -> None:
    rows = [
        {
            "name": item["name"],
            "method_type": item["method_type"],
            "num_runs": item["num_runs"],
            "mean_final_val_accuracy": item["mean_final_val_accuracy"],
            "std_final_val_accuracy": item["std_final_val_accuracy"],
            "mean_best_val_accuracy": item["mean_best_val_accuracy"],
            "std_best_val_accuracy": item["std_best_val_accuracy"],
            "mean_adaptations_applied": item["mean_adaptations_applied"],
            "mean_final_hidden_dim": item["mean_final_hidden_dim"],
            "mean_parameter_count": item["mean_parameter_count"],
            "mean_nonzero_parameter_count": item["mean_nonzero_parameter_count"],
            "mean_weight_sparsity": item["mean_weight_sparsity"],
            "mean_forward_flop_proxy": item["mean_forward_flop_proxy"],
            "mean_activation_elements": item["mean_activation_elements"],
            "best_seed": item["best_seed"],
            "notes": item["notes"],
        }
        for item in aggregate
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_markdown(path: Path, aggregate: list[dict[str, Any]], seeds: list[int]) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Aggregate Plots",
        "",
        "![Mean final validation accuracy](mean_final_val_accuracy.png)",
        "",
        "![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)",
        "",
        "| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregate:
        lines.append(
            "| {name} | {method_type} | {num_runs} | {mean_final_val_accuracy:.4f} | {std_final_val_accuracy:.4f} | {mean_best_val_accuracy:.4f} | {mean_adaptations_applied:.2f} | {mean_final_hidden_dim} | {best_seed} |".format(
                name=item["name"],
                method_type=item["method_type"],
                num_runs=item["num_runs"],
                mean_final_val_accuracy=item["mean_final_val_accuracy"],
                std_final_val_accuracy=item["std_final_val_accuracy"],
                mean_best_val_accuracy=item["mean_best_val_accuracy"],
                mean_adaptations_applied=item["mean_adaptations_applied"],
                mean_final_hidden_dim=(f"{item['mean_final_hidden_dim']:.1f}" if item["mean_final_hidden_dim"] is not None else "-"),
                best_seed=item["best_seed"],
            )
        )

    constraint_items = [item for item in aggregate if item.get("mean_parameter_count") is not None]
    if constraint_items:
        lines.extend(
            [
                "",
                "## Constraint Summary",
                "",
                "| Experiment | Mean params | Mean nonzero params | Mean weight sparsity | Mean FLOP proxy | Mean activation elems |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in constraint_items:
            lines.append(
                "| {name} | {mean_parameter_count} | {mean_nonzero_parameter_count} | {mean_weight_sparsity:.4f} | {mean_forward_flop_proxy} | {mean_activation_elements} |".format(
                    name=item["name"],
                    mean_parameter_count=_format_number(item["mean_parameter_count"]),
                    mean_nonzero_parameter_count=_format_number(item["mean_nonzero_parameter_count"]),
                    mean_weight_sparsity=float(item["mean_weight_sparsity"] or 0.0),
                    mean_forward_flop_proxy=_format_number(item["mean_forward_flop_proxy"]),
                    mean_activation_elements=_format_number(item["mean_activation_elements"]),
                )
            )

    notes_items = [item for item in aggregate if item.get("notes")]
    if notes_items:
        lines.extend(["", "## Experiment Notes", ""])
        for item in notes_items:
            lines.append(f"- `{item['name']}`: {item['notes']}")

    lines.extend(["", "## Per-Seed Results", ""])
    for item in aggregate:
        lines.append(f"### {item['name']}")
        for run in item["seed_runs"]:
            lines.append(
                f"- seed {run['seed']}: final={run['final_val_accuracy']:.4f}, best={run['best_val_accuracy']:.4f}, adaptations={run['adaptations_applied']}, params={run.get('parameter_count')}, nonzero={run.get('nonzero_parameter_count')}, sparsity={_format_sparsity(run.get('weight_sparsity'))}"
            )
        lines.append("")

    stage_items = [item for item in aggregate if item.get("representative_stage_history")]
    if stage_items:
        lines.extend(["## Representative Stage Histories", ""])
        for item in stage_items:
            lines.append(f"### {item['name']} (best seed {item['best_seed']})")
            for stage in item["representative_stage_history"]:
                lines.append(
                    f"- {stage['name']}: epochs={stage['epochs']}, range={stage['epoch_start']}..{stage['epoch_end']}, adaptation_enabled={stage['adaptation_enabled']}, final_val={stage['final_val_accuracy']}"
                )
            lines.append("")

    graph_items = [item for item in aggregate if item.get("representative_architecture_graph") or item.get("representative_architecture_spec")]
    if graph_items:
        lines.extend(["## Representative Architectures", ""])
        for item in graph_items:
            lines.append(f"### {item['name']} (best seed {item['best_seed']})")
            lines.append(
                mermaid_architecture(
                    item.get("representative_architecture_graph"),
                    item.get("representative_architecture_spec"),
                    item["name"],
                )
            )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_plots(output_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    _write_mean_accuracy_plot(output_dir / "mean_final_val_accuracy.png", aggregate)
    _write_per_seed_plot(output_dir / "per_seed_final_val_accuracy.png", aggregate)


def _write_mean_accuracy_plot(path: Path, aggregate: list[dict[str, Any]]) -> None:
    names = [item["name"] for item in aggregate]
    means = [item["mean_final_val_accuracy"] for item in aggregate]
    stds = [item["std_final_val_accuracy"] for item in aggregate]
    colors = [PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(aggregate))]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = list(range(len(names)))
    ax.bar(positions, means, yerr=stds, color=colors, capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean final validation accuracy")
    ax.set_ylim(0.0, min(1.0, max(means) + max(stds, default=0.0) + 0.08))
    ax.set_title("Benchmark Summary: Mean Final Validation Accuracy")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_per_seed_plot(path: Path, aggregate: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, item in enumerate(aggregate):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        seed_runs = item["seed_runs"]
        seeds = [run["seed"] for run in seed_runs]
        values = [run["final_val_accuracy"] for run in seed_runs]
        ax.plot(seeds, values, marker="o", linewidth=2.0, color=color, label=item["name"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title("Per-Seed Final Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _format_sparsity(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
