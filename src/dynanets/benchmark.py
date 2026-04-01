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
PLOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


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
        mean_metric_history = _mean_metric_history(group)
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
                "mean_metric_history": mean_metric_history,
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
        "![Mean validation accuracy by epoch](mean_validation_accuracy_by_epoch.png)",
        "",
        "![Mean final validation accuracy](mean_final_val_accuracy.png)",
        "",
        "![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)",
        "",
        "![Mean parameter count](mean_parameter_count.png)",
        "",
        "![Mean FLOP proxy](mean_forward_flop_proxy.png)",
        "",
        "![Accuracy vs FLOP proxy](accuracy_vs_flop_proxy.png)",
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

    frontier_items = _pareto_frontier(aggregate, x_key="mean_forward_flop_proxy", y_key="mean_final_val_accuracy")
    if frontier_items:
        lines.extend(["", "## Accuracy-FLOP Pareto Frontier", ""])
        for item in frontier_items:
            lines.append(
                f"- `{item['name']}`: acc={item['mean_final_val_accuracy']:.4f}, flop_proxy={_format_number(item.get('mean_forward_flop_proxy'))}, params={_format_number(item.get('mean_parameter_count'))}"
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
    styles = _style_map(aggregate)
    _write_epoch_accuracy_plot(output_dir / "mean_validation_accuracy_by_epoch.png", aggregate, styles)
    _write_mean_accuracy_plot(output_dir / "mean_final_val_accuracy.png", aggregate, styles)
    _write_per_seed_plot(output_dir / "per_seed_final_val_accuracy.png", aggregate, styles)
    _write_constraint_bar_plot(
        output_dir / "mean_parameter_count.png",
        aggregate,
        metric_key="mean_parameter_count",
        title="Mean Parameter Count",
        ylabel="Parameters",
        styles=styles,
    )
    _write_constraint_bar_plot(
        output_dir / "mean_forward_flop_proxy.png",
        aggregate,
        metric_key="mean_forward_flop_proxy",
        title="Mean FLOP Proxy",
        ylabel="FLOP proxy",
        styles=styles,
    )
    _write_accuracy_vs_constraint_plot(
        output_dir / "accuracy_vs_flop_proxy.png",
        aggregate,
        metric_key="mean_forward_flop_proxy",
        xlabel="Mean FLOP proxy",
        title="Accuracy vs FLOP Proxy",
        styles=styles,
    )


def _method_style(name: str, names: list[str]) -> dict[str, str]:
    ordered = sorted(dict.fromkeys(names))
    index = ordered.index(name)
    return {
        "color": PLOT_COLORS[index % len(PLOT_COLORS)],
        "marker": PLOT_MARKERS[index % len(PLOT_MARKERS)],
    }



def _style_map(aggregate: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    names = [item["name"] for item in aggregate]
    return {name: _method_style(name, names) for name in names}


def _write_epoch_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7), constrained_layout=True)
    plotted = False
    for item in aggregate:
        values = list(item.get("mean_metric_history") or [])
        if not values:
            continue
        epochs = list(range(1, len(values) + 1))
        style = styles[item["name"]]
        markevery = max(1, len(epochs) // 8)
        ax.plot(
            epochs,
            values,
            marker=style["marker"],
            markevery=markevery,
            linewidth=2.0,
            color=style["color"],
            label=item["name"],
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean validation accuracy")
        ax.set_title("Mean Validation Accuracy by Epoch")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "No epoch history available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _write_mean_accuracy_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    names = [item["name"] for item in aggregate]
    means = [item["mean_final_val_accuracy"] for item in aggregate]
    stds = [item["std_final_val_accuracy"] for item in aggregate]
    colors = [styles[name]["color"] for name in names]

    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.42 * len(names) + 2.0)), constrained_layout=True)
    positions = list(range(len(names)))
    ax.barh(positions, means, xerr=stds, color=colors, capsize=5, edgecolor="#222222", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean final validation accuracy")
    ax.set_xlim(0.0, min(1.0, max(means) + max(stds, default=0.0) + 0.08))
    ax.set_title("Benchmark Summary: Mean Final Validation Accuracy")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _mean_metric_history(group: list[dict[str, Any]]) -> list[float]:
    histories = []
    for item in group:
        metric_history = item.get("metric_history") or []
        if not metric_history:
            continue
        curve = []
        for point in metric_history:
            accuracy = point.get("accuracy")
            if accuracy is None:
                curve.append(float("nan"))
            else:
                curve.append(float(accuracy))
        histories.append(curve)
    if not histories:
        return []
    max_len = max(len(history) for history in histories)
    mean_curve = []
    for epoch_index in range(max_len):
        values = [history[epoch_index] for history in histories if epoch_index < len(history)]
        if not values:
            continue
        mean_curve.append(sum(values) / len(values))
    return mean_curve


def _write_per_seed_plot(path: Path, aggregate: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7), constrained_layout=True)
    for item in aggregate:
        style = styles[item["name"]]
        seed_runs = item["seed_runs"]
        seeds = [run["seed"] for run in seed_runs]
        values = [run["final_val_accuracy"] for run in seed_runs]
        ax.plot(seeds, values, marker=style["marker"], linewidth=2.0, color=style["color"], label=item["name"])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title("Per-Seed Final Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _write_constraint_bar_plot(
    path: Path,
    aggregate: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    styles: dict[str, dict[str, str]],
) -> None:
    filtered = [item for item in aggregate if item.get(metric_key) is not None]
    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.42 * len(filtered) + 2.0)), constrained_layout=True)
    if not filtered:
        ax.text(0.5, 0.5, f"No data available for {title.lower()}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        names = [item["name"] for item in filtered]
        values = [float(item[metric_key]) for item in filtered]
        colors = [styles[name]["color"] for name in names]
        positions = list(range(len(names)))
        ax.barh(positions, values, color=colors, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_accuracy_vs_constraint_plot(
    path: Path,
    aggregate: list[dict[str, Any]],
    *,
    metric_key: str,
    xlabel: str,
    title: str,
    styles: dict[str, dict[str, str]],
) -> None:
    filtered = [item for item in aggregate if item.get(metric_key) is not None]
    fig, ax = plt.subplots(figsize=(11.5, 7), constrained_layout=True)
    if not filtered:
        ax.text(0.5, 0.5, f"No data available for {title.lower()}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for item in filtered:
            style = styles[item["name"]]
            x_value = float(item[metric_key])
            y_value = float(item["mean_final_val_accuracy"])
            ax.scatter(
                x_value,
                y_value,
                color=style["color"],
                marker=style["marker"],
                s=80,
                edgecolors="#222222",
                linewidths=0.6,
                label=item["name"],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean final validation accuracy")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _pareto_frontier(aggregate: list[dict[str, Any]], *, x_key: str, y_key: str) -> list[dict[str, Any]]:
    candidates = [item for item in aggregate if item.get(x_key) is not None and item.get(y_key) is not None]
    ordered = sorted(candidates, key=lambda item: (float(item[x_key]), -float(item[y_key])))
    frontier: list[dict[str, Any]] = []
    best_y = float('-inf')
    for item in ordered:
        y_value = float(item[y_key])
        if y_value > best_y:
            frontier.append(item)
            best_y = y_value
    return frontier


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


