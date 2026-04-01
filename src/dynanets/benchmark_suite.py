from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from dynanets.reporting import mermaid_architecture


BASELINE_METHOD_TYPES = {"baseline"}
TRACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
TRACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


@dataclass(slots=True)
class BenchmarkSuiteEntry:
    label: str
    report_dir: str
    description: str = ""
    track: str | None = None


@dataclass(slots=True)
class BenchmarkSuiteManifest:
    name: str
    title: str | None = None
    description: str = ""
    entries: list[BenchmarkSuiteEntry] | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkSuiteManifest":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [BenchmarkSuiteEntry(**item) for item in data.get("entries", [])]
        return cls(
            name=str(data["name"]),
            title=data.get("title"),
            description=data.get("description", ""),
            entries=entries,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dynanets benchmark reports into one suite summary.")
    parser.add_argument("manifest", help="Path to benchmark suite YAML manifest")
    parser.add_argument("--output-dir", help="Directory for suite artifacts")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = BenchmarkSuiteManifest.from_yaml(manifest_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("reports") / manifest.name
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = build_suite_summary(manifest, manifest_path.parent)
    write_suite_json(output_dir / "suite.json", suite)
    write_suite_markdown(output_dir / "summary.md", suite)
    write_suite_latex(output_dir / "summary.tex", suite)
    write_suite_plots(output_dir, suite)

    print(output_dir / "summary.md")
    print(f"suite={suite['name']}")
    print(f"benchmarks={len(suite['benchmarks'])}")
    print(f"methods={len(suite['method_coverage'])}")


def build_suite_summary(manifest: BenchmarkSuiteManifest, manifest_dir: Path) -> dict[str, Any]:
    benchmarks = [load_benchmark_entry(entry, manifest_dir) for entry in (manifest.entries or [])]
    method_coverage = build_method_coverage(benchmarks)
    return {
        "name": manifest.name,
        "title": manifest.title or manifest.name,
        "description": manifest.description,
        "benchmarks": benchmarks,
        "method_coverage": method_coverage,
    }


def load_benchmark_entry(entry: BenchmarkSuiteEntry, manifest_dir: Path) -> dict[str, Any]:
    report_dir = (manifest_dir / entry.report_dir).resolve()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    runs_path = report_dir / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else []
    protocol_path = report_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else None
    acceptance_path = report_dir / "protocol_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else None
    leaderboard_path = report_dir / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8")) if leaderboard_path.exists() else None

    track = entry.track or (protocol.get("track") if protocol else None) or entry.label
    seeds = protocol.get("seeds", []) if protocol else _infer_seeds(summary)
    top_overall = select_top_method(summary, include_baselines=True)
    top_method = select_top_method(summary, include_baselines=False) or top_overall

    return {
        "label": entry.label,
        "track": track,
        "description": entry.description or (protocol.get("description") if protocol else ""),
        "report_dir": str(report_dir),
        "summary_path": str(report_dir / "summary.json"),
        "protocol": protocol,
        "acceptance": acceptance,
        "leaderboard": leaderboard,
        "seeds": seeds,
        "results": summary,
        "runs": runs,
        "epoch_curves": build_epoch_curves(summary, runs),
        "top_overall": top_overall,
        "top_method": top_method,
    }


def build_epoch_curves(results: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        item["name"]: index
        for index, item in enumerate(
            sorted(results, key=lambda item: item.get("mean_final_val_accuracy", float("-inf")), reverse=True)
        )
    }
    grouped: dict[str, list[list[float]]] = {}
    for run in runs:
        name = run.get("name")
        metric_history = run.get("metric_history") or []
        if not name or not metric_history:
            continue
        curve = []
        for point in metric_history:
            accuracy = point.get("accuracy")
            curve.append(float(accuracy) if accuracy is not None else float("nan"))
        grouped.setdefault(name, []).append(curve)

    curves = []
    for name, histories in grouped.items():
        max_len = max((len(history) for history in histories), default=0)
        mean_curve = []
        for epoch_index in range(max_len):
            values = [history[epoch_index] for history in histories if epoch_index < len(history)]
            if not values:
                continue
            mean_curve.append(sum(values) / len(values))
        curves.append(
            {
                "name": name,
                "epochs": list(range(1, len(mean_curve) + 1)),
                "mean_validation_accuracy": mean_curve,
                "sort_order": order.get(name, 10_000),
            }
        )
    curves.sort(key=lambda item: (item["sort_order"], item["name"]))
    return curves


def build_method_coverage(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        for result in benchmark["results"]:
            item = coverage.setdefault(
                result["name"],
                {
                    "name": result["name"],
                    "method_types": set(),
                    "tracks": set(),
                    "benchmarks": [],
                    "appearances": 0,
                    "best_mean_final_val_accuracy": float("-inf"),
                    "best_benchmark": None,
                },
            )
            item["method_types"].add(result.get("method_type", "unknown"))
            item["tracks"].add(benchmark["track"])
            item["benchmarks"].append(
                {
                    "label": benchmark["label"],
                    "track": benchmark["track"],
                    "mean_final_val_accuracy": result["mean_final_val_accuracy"],
                    "std_final_val_accuracy": result.get("std_final_val_accuracy", 0.0),
                }
            )
            item["appearances"] += 1
            if result["mean_final_val_accuracy"] > item["best_mean_final_val_accuracy"]:
                item["best_mean_final_val_accuracy"] = result["mean_final_val_accuracy"]
                item["best_benchmark"] = benchmark["label"]

    rows = []
    for item in coverage.values():
        rows.append(
            {
                "name": item["name"],
                "method_types": sorted(item["method_types"]),
                "tracks": sorted(item["tracks"]),
                "appearances": item["appearances"],
                "best_mean_final_val_accuracy": item["best_mean_final_val_accuracy"],
                "best_benchmark": item["best_benchmark"],
                "benchmarks": sorted(item["benchmarks"], key=lambda benchmark: (benchmark["track"], benchmark["label"])),
            }
        )
    rows.sort(key=lambda item: (-item["best_mean_final_val_accuracy"], item["name"]))
    return rows


def select_top_method(results: list[dict[str, Any]], *, include_baselines: bool) -> dict[str, Any] | None:
    candidates = results if include_baselines else [item for item in results if item.get("method_type") not in BASELINE_METHOD_TYPES]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.get("mean_final_val_accuracy", float("-inf")), item.get("mean_best_val_accuracy", float("-inf"))))


def write_suite_json(path: Path, suite: dict[str, Any]) -> None:
    path.write_text(json.dumps(suite, indent=2), encoding="utf-8")


def write_suite_markdown(path: Path, suite: dict[str, Any]) -> None:
    lines = [
        f"# {suite['title']}",
        "",
        suite.get("description", ""),
        "",
        "## Suite Plots",
        "",
        "![Accuracy by epoch](benchmark_epoch_curves.png)",
        "",
        "![Final accuracy by benchmark](benchmark_final_accuracy.png)",
        "",
        "![Benchmark leaderboards](benchmark_leaderboards.png)",
        "",
        "![Method coverage](method_coverage.png)",
        "",
        "## Benchmark Inventory",
        "",
        "| Benchmark | Track | Tier | Acceptance | Seeds | Methods | Top overall | Top method |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for benchmark in suite["benchmarks"]:
        top_overall = benchmark.get("top_overall")
        top_method = benchmark.get("top_method")
        acceptance = benchmark.get("acceptance") or {}
        protocol = benchmark.get("protocol") or {}
        lines.append(
            "| {label} | {track} | {tier} | {acceptance} | {seeds} | {count} | {top_overall} ({top_overall_score:.4f}) | {top_method} ({top_method_score:.4f}) |".format(
                label=benchmark["label"],
                track=benchmark["track"],
                tier=protocol.get("tier", "preview"),
                acceptance=("PASS" if acceptance.get("passed") else ("FAIL" if acceptance else "-")),
                seeds=len(benchmark.get("seeds", [])),
                count=len(benchmark["results"]),
                top_overall=top_overall["name"] if top_overall else "-",
                top_overall_score=top_overall["mean_final_val_accuracy"] if top_overall else float("nan"),
                top_method=top_method["name"] if top_method else "-",
                top_method_score=top_method["mean_final_val_accuracy"] if top_method else float("nan"),
            )
        )

    lines.extend([
        "",
        "## Method Coverage",
        "",
        "| Method | Type | Appearances | Tracks | Best mean final val acc | Best benchmark |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ])
    for item in suite["method_coverage"]:
        lines.append(
            "| {name} | {method_types} | {appearances} | {tracks} | {best:.4f} | {benchmark} |".format(
                name=item["name"],
                method_types=", ".join(item["method_types"]),
                appearances=item["appearances"],
                tracks=", ".join(item["tracks"]),
                best=item["best_mean_final_val_accuracy"],
                benchmark=item["best_benchmark"],
            )
        )

    for benchmark in suite["benchmarks"]:
        lines.extend(
            [
                "",
                f"## {benchmark['label']}",
                "",
                benchmark.get("description", ""),
                "",
                f"Source: `{benchmark['report_dir']}`",
                f"Tier: `{(benchmark.get('protocol') or {}).get('tier', 'preview')}`",
                f"Acceptance: `{('PASS' if (benchmark.get('acceptance') or {}).get('passed') else ('FAIL' if benchmark.get('acceptance') else '-') )}`",
                "",
                f"![{benchmark['label']} epoch curves]({_benchmark_epoch_plot_name(benchmark)})",
                "",
                f"![{benchmark['label']} final accuracy]({_benchmark_final_plot_name(benchmark)})",
                "",
                "| Method | Type | Mean final val acc | Std | Mean best val acc | Params | Weight sparsity |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in benchmark["results"]:
            lines.append(
                "| {name} | {method_type} | {mean_final:.4f} | {std_final:.4f} | {mean_best:.4f} | {params} | {sparsity:.4f} |".format(
                    name=result["name"],
                    method_type=result.get("method_type", "unknown"),
                    mean_final=result.get("mean_final_val_accuracy", float("nan")),
                    std_final=result.get("std_final_val_accuracy", float("nan")),
                    mean_best=result.get("mean_best_val_accuracy", float("nan")),
                    params=_format_int(result.get("mean_parameter_count")),
                    sparsity=float(result.get("mean_weight_sparsity") or 0.0),
                )
            )

        winner = benchmark.get("top_method") or benchmark.get("top_overall")
        if winner is not None:
            lines.extend(
                [
                    "",
                    f"### Representative Graph: {winner['name']}",
                    "",
                    mermaid_architecture(winner.get("representative_architecture_graph"), winner.get("representative_architecture_spec"), winner["name"]),
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_suite_latex(path: Path, suite: dict[str, Any]) -> None:
    lines = [
        "% Auto-generated by dynanets.benchmark_suite.write_suite_latex",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{" + _latex_escape(f"{suite['title']} benchmark summary") + "}",
        "\\label{" + _latex_label(f"tab:{suite['name']}_summary") + "}",
        "\\begin{tabular}{llrlrlrl}",
        "\\toprule",
        r"Benchmark & Track & Seeds & Top overall & Acc. & Top method & Acc. & Tier \\",
        "\\midrule",
    ]
    for benchmark in suite["benchmarks"]:
        row = " & ".join(
            [
                _latex_escape(benchmark["label"]),
                _latex_escape(benchmark["track"]),
                str(len(benchmark.get("seeds", []))),
                _latex_escape((benchmark.get("top_overall") or {}).get("name", "-")),
                f"{(benchmark.get('top_overall') or {}).get('mean_final_val_accuracy', float('nan')):.4f}",
                _latex_escape((benchmark.get("top_method") or {}).get("name", "-")),
                f"{(benchmark.get('top_method') or {}).get('mean_final_val_accuracy', float('nan')):.4f}",
                _latex_escape((benchmark.get("protocol") or {}).get("tier", "preview")),
            ]
        ) + r" \\\\"
        lines.append(row)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _suite_method_styles(benchmarks: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    names = sorted({item["name"] for benchmark in benchmarks for item in benchmark["results"]} | {item["name"] for item in coverage})
    return {
        name: {
            "color": TRACK_COLORS[index % len(TRACK_COLORS)],
            "marker": TRACK_MARKERS[index % len(TRACK_MARKERS)],
        }
        for index, name in enumerate(names)
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "benchmark"


def _benchmark_epoch_plot_name(benchmark: dict[str, Any]) -> str:
    return f"{_slugify(benchmark['label'])}_epoch_curves.png"


def _benchmark_final_plot_name(benchmark: dict[str, Any]) -> str:
    return f"{_slugify(benchmark['label'])}_final_accuracy.png"


def write_suite_plots(output_dir: Path, suite: dict[str, Any]) -> None:
    styles = _suite_method_styles(suite["benchmarks"], suite["method_coverage"])
    _write_benchmark_epoch_curves(output_dir / "benchmark_epoch_curves.png", suite["benchmarks"], styles)
    _write_benchmark_final_accuracy(output_dir / "benchmark_final_accuracy.png", suite["benchmarks"], styles)
    _write_benchmark_leaderboards(output_dir / "benchmark_leaderboards.png", suite["benchmarks"], styles)
    _write_method_coverage_plot(output_dir / "method_coverage.png", suite["method_coverage"], styles)
    for benchmark in suite["benchmarks"]:
        _write_single_benchmark_epoch_curve(output_dir / _benchmark_epoch_plot_name(benchmark), benchmark, styles)
        _write_single_benchmark_final_accuracy(output_dir / _benchmark_final_plot_name(benchmark), benchmark, styles)


def _write_benchmark_epoch_curves(path: Path, benchmarks: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=(13.5, max(4, 4.2 * len(benchmarks))), constrained_layout=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for axis, benchmark in zip(axes, benchmarks):
        curves = benchmark.get("epoch_curves", [])
        if not curves:
            axis.text(0.5, 0.5, "No epoch curves available", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
            axis.set_axis_off()
            continue
        for curve_index, curve in enumerate(curves[:10]):
            style = styles.get(curve["name"], {"color": TRACK_COLORS[curve_index % len(TRACK_COLORS)], "marker": TRACK_MARKERS[curve_index % len(TRACK_MARKERS)]})
            epochs = curve["epochs"]
            markevery = max(1, len(epochs) // 8)
            axis.plot(
                epochs,
                curve["mean_validation_accuracy"],
                label=curve["name"],
                color=style["color"],
                marker=style["marker"],
                markevery=markevery,
                linewidth=2,
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean val acc")
        axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_benchmark_final_accuracy(path: Path, benchmarks: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=(13.0, max(4, 3.8 * len(benchmarks))), constrained_layout=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for axis, benchmark in zip(axes, benchmarks):
        results = sorted(benchmark["results"], key=lambda item: item["mean_final_val_accuracy"], reverse=True)
        labels = [item["name"] for item in results]
        values = [item["mean_final_val_accuracy"] for item in results]
        errors = [item.get("std_final_val_accuracy", 0.0) for item in results]
        colors = [styles[item["name"]]["color"] for item in results]
        axis.barh(labels, values, xerr=errors, color=colors, alpha=0.85, edgecolor="#222222", linewidth=0.8)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean final validation accuracy")
        axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
        axis.grid(axis="x", linestyle="--", alpha=0.3)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_benchmark_leaderboards(path: Path, benchmarks: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    winners = []
    for benchmark in benchmarks:
        results = sorted(benchmark["results"], key=lambda item: item["mean_final_val_accuracy"], reverse=True)
        if not results:
            continue
        top_results = results[: min(3, len(results))]
        for rank, item in enumerate(top_results, start=1):
            winners.append(
                {
                    "benchmark": benchmark["label"],
                    "track": benchmark["track"],
                    "name": item["name"],
                    "accuracy": item["mean_final_val_accuracy"],
                    "std": item.get("std_final_val_accuracy", 0.0),
                    "rank": rank,
                }
            )

    fig, axis = plt.subplots(figsize=(13, max(5, 0.55 * len(winners))), constrained_layout=True)
    if not winners:
        axis.text(0.5, 0.5, "No leaderboard data available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        labels = [f"#{item['rank']} {item['benchmark']} :: {item['name']}" for item in winners]
        values = [item["accuracy"] for item in winners]
        errors = [item["std"] for item in winners]
        colors = [styles[item["name"]]["color"] for item in winners]
        axis.barh(labels, values, xerr=errors, color=colors, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean final validation accuracy")
        axis.set_title("Benchmark Leaderboards (Top 3 Per Benchmark)")
        axis.grid(axis="x", linestyle="--", alpha=0.3)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_method_coverage_plot(path: Path, coverage: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    ordered = sorted(coverage, key=lambda item: (-item["appearances"], -item["best_mean_final_val_accuracy"], item["name"]))
    labels = [item["name"] for item in ordered]
    values = [item["appearances"] for item in ordered]

    fig, axis = plt.subplots(figsize=(13, max(6, 0.34 * len(labels) + 2.0)), constrained_layout=True)
    positions = list(range(len(labels)))
    axis.barh(
        positions,
        values,
        color=[styles[name]["color"] for name in labels],
        edgecolor="#222222",
        linewidth=0.8,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Benchmark appearances")
    axis.set_title("Implemented Method Coverage Across Suite Benchmarks")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_single_benchmark_epoch_curve(path: Path, benchmark: dict[str, Any], styles: dict[str, dict[str, str]]) -> None:
    fig, axis = plt.subplots(figsize=(12.5, 7), constrained_layout=True)
    curves = benchmark.get("epoch_curves", [])
    if not curves:
        axis.text(0.5, 0.5, "No epoch curves available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        for curve in curves:
            style = styles.get(curve["name"], {"color": TRACK_COLORS[0], "marker": TRACK_MARKERS[0]})
            epochs = curve["epochs"]
            markevery = max(1, len(epochs) // 8)
            axis.plot(
                epochs,
                curve["mean_validation_accuracy"],
                label=curve["name"],
                color=style["color"],
                marker=style["marker"],
                markevery=markevery,
                linewidth=2,
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean val acc")
        axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False, borderaxespad=0.0)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_single_benchmark_final_accuracy(path: Path, benchmark: dict[str, Any], styles: dict[str, dict[str, str]]) -> None:
    fig, axis = plt.subplots(figsize=(12.5, max(5.5, 0.42 * len(benchmark["results"]) + 2.0)), constrained_layout=True)
    results = sorted(benchmark["results"], key=lambda item: item["mean_final_val_accuracy"], reverse=True)
    if not results:
        axis.text(0.5, 0.5, "No results available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        labels = [item["name"] for item in results]
        values = [item["mean_final_val_accuracy"] for item in results]
        errors = [item.get("std_final_val_accuracy", 0.0) for item in results]
        colors = [styles[item["name"]]["color"] for item in results]
        axis.barh(labels, values, xerr=errors, color=colors, alpha=0.85, edgecolor="#222222", linewidth=0.8)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean final validation accuracy")
        axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
        axis.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _infer_seeds(summary: list[dict[str, Any]]) -> list[int]:
    seeds = set()
    for item in summary:
        for run in item.get("seed_runs", []):
            seed = run.get("seed")
            if seed is not None:
                seeds.add(int(seed))
    return sorted(seeds)


def _format_int(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _latex_escape(value: str) -> str:
    replacements = {
        '\\': r'\\textbackslash{}',
        '&': r'\\&',
        '%': r'\\%',
        '$': r'\\$',
        '#': r'\\#',
        '_': r'\\_',
        '{': r'\\{',
        '}': r'\\}',
        '~': r'\\textasciitilde{}',
        '^': r'\\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in value)


def _latex_label(value: str) -> str:
    clean = []
    for ch in value.lower():
        if ch.isalnum() or ch in {':', '_'}:
            clean.append(ch)
        else:
            clean.append('_')
    return ''.join(clean)


if __name__ == "__main__":
    main()
