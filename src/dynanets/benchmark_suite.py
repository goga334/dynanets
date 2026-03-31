from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from dynanets.reporting import mermaid_architecture


BASELINE_METHOD_TYPES = {"baseline"}
TRACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


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
    protocol_path = report_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else None

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
        "seeds": seeds,
        "results": summary,
        "top_overall": top_overall,
        "top_method": top_method,
    }


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
        "![Benchmark leaderboards](benchmark_leaderboards.png)",
        "",
        "![Method coverage](method_coverage.png)",
        "",
        "## Benchmark Inventory",
        "",
        "| Benchmark | Track | Seeds | Methods | Top overall | Top method |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for benchmark in suite["benchmarks"]:
        top_overall = benchmark.get("top_overall")
        top_method = benchmark.get("top_method")
        lines.append(
            "| {label} | {track} | {seeds} | {count} | {top_overall} ({top_overall_score:.4f}) | {top_method} ({top_method_score:.4f}) |".format(
                label=benchmark["label"],
                track=benchmark["track"],
                seeds=len(benchmark.get("seeds", [])),
                count=len(benchmark["results"]),
                top_overall=top_overall["name"] if top_overall else "-",
                top_overall_score=top_overall["mean_final_val_accuracy"] if top_overall else float("nan"),
                top_method=top_method["name"] if top_method else "-",
                top_method_score=top_method["mean_final_val_accuracy"] if top_method else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Method Coverage",
            "",
            "| Method | Type | Appearances | Tracks | Best mean final val acc | Best benchmark |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
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


def write_suite_plots(output_dir: Path, suite: dict[str, Any]) -> None:
    _write_benchmark_leaderboards(output_dir / "benchmark_leaderboards.png", suite["benchmarks"])
    _write_method_coverage_plot(output_dir / "method_coverage.png", suite["method_coverage"])


def _write_benchmark_leaderboards(path: Path, benchmarks: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=(12, max(4, 3.6 * len(benchmarks))), constrained_layout=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for index, (axis, benchmark) in enumerate(zip(axes, benchmarks)):
        results = sorted(benchmark["results"], key=lambda item: item["mean_final_val_accuracy"], reverse=True)
        labels = [item["name"] for item in results]
        values = [item["mean_final_val_accuracy"] for item in results]
        errors = [item.get("std_final_val_accuracy", 0.0) for item in results]
        colors = [TRACK_COLORS[index % len(TRACK_COLORS)] for _ in results]
        axis.barh(labels, values, xerr=errors, color=colors, alpha=0.85)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Mean final validation accuracy")
        axis.set_title(f"{benchmark['label']} ({benchmark['track']})")
        axis.grid(axis="x", linestyle="--", alpha=0.3)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_method_coverage_plot(path: Path, coverage: list[dict[str, Any]]) -> None:
    ordered = sorted(coverage, key=lambda item: (-item["appearances"], -item["best_mean_final_val_accuracy"], item["name"]))
    labels = [item["name"] for item in ordered]
    values = [item["appearances"] for item in ordered]

    fig, axis = plt.subplots(figsize=(max(12, len(labels) * 0.45), 5.5), constrained_layout=True)
    axis.bar(range(len(labels)), values, color="#4c78a8")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_ylabel("Benchmark appearances")
    axis.set_title("Implemented Method Coverage Across Suite Benchmarks")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(path, dpi=160)
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


if __name__ == "__main__":
    main()
