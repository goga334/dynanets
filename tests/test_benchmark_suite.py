from __future__ import annotations

import json
from pathlib import Path

from dynanets.benchmark_suite import (
    BenchmarkSuiteManifest,
    build_suite_summary,
    write_suite_latex,
    write_suite_markdown,
    write_suite_plots,
)


def _write_report(report_dir: Path, *, track: str, results: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    runs = []
    for result in results:
        for seed_index, seed_run in enumerate(result.get("seed_runs", []), start=1):
            final_accuracy = float(result["mean_final_val_accuracy"])
            runs.append(
                {
                    "name": result["name"],
                    "metric_history": [
                        {"accuracy": max(0.0, final_accuracy - 0.1)},
                        {"accuracy": final_accuracy},
                    ],
                    "seed": seed_run.get("seed", seed_index),
                }
            )
    (report_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (report_dir / "protocol.json").write_text(
        json.dumps(
            {
                "name": report_dir.name,
                "track": track,
                "description": f"Protocol for {track}",
                "seeds": [7, 42],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_benchmark_suite_builds_cross_track_summary(tmp_path: Path) -> None:
    synthetic_report = tmp_path / "synthetic_report"
    mnist_report = tmp_path / "mnist_report"

    _write_report(
        synthetic_report,
        track="synthetic",
        results=[
            {
                "name": "fixed-mlp",
                "method_type": "baseline",
                "mean_final_val_accuracy": 0.61,
                "std_final_val_accuracy": 0.02,
                "mean_best_val_accuracy": 0.65,
                "mean_parameter_count": 120,
                "mean_weight_sparsity": 0.0,
                "representative_architecture_spec": {
                    "input_dim": 2,
                    "output_dim": 2,
                    "hidden_dims": [8],
                    "hidden_activation": "relu",
                    "output_activation": None,
                    "bias": True,
                    "metadata": {},
                },
                "representative_architecture_graph": None,
                "seed_runs": [{"seed": 7}, {"seed": 42}],
            },
            {
                "name": "gradmax",
                "method_type": "dynamic",
                "mean_final_val_accuracy": 0.64,
                "std_final_val_accuracy": 0.01,
                "mean_best_val_accuracy": 0.66,
                "mean_parameter_count": 140,
                "mean_weight_sparsity": 0.0,
                "representative_architecture_spec": {
                    "input_dim": 2,
                    "output_dim": 2,
                    "hidden_dims": [10],
                    "hidden_activation": "relu",
                    "output_activation": None,
                    "bias": True,
                    "metadata": {},
                },
                "representative_architecture_graph": None,
                "seed_runs": [{"seed": 7}, {"seed": 42}],
            },
        ],
    )
    _write_report(
        mnist_report,
        track="mnist",
        results=[
            {
                "name": "wide-cnn",
                "method_type": "baseline",
                "mean_final_val_accuracy": 0.83,
                "std_final_val_accuracy": 0.03,
                "mean_best_val_accuracy": 0.85,
                "mean_parameter_count": 16000,
                "mean_weight_sparsity": 0.0,
                "representative_architecture_spec": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "blocks": [{"out_channels": 24, "kernel_size": 3, "pool": "max"}],
                    "classifier_hidden_dims": [64],
                    "activation": "relu",
                    "use_batch_norm": True,
                    "metadata": {},
                },
                "representative_architecture_graph": None,
                "seed_runs": [{"seed": 7}, {"seed": 42}],
            },
            {
                "name": "channel-gating",
                "method_type": "workflow",
                "mean_final_val_accuracy": 0.55,
                "std_final_val_accuracy": 0.02,
                "mean_best_val_accuracy": 0.58,
                "mean_parameter_count": 11000,
                "mean_weight_sparsity": 0.0,
                "representative_architecture_spec": {
                    "input_channels": 1,
                    "input_size": [28, 28],
                    "num_classes": 10,
                    "blocks": [{"out_channels": 24, "kernel_size": 3, "pool": "max"}],
                    "classifier_hidden_dims": [],
                    "activation": "relu",
                    "use_batch_norm": False,
                    "metadata": {"routing_family": "routed_cnn"},
                },
                "representative_architecture_graph": None,
                "seed_runs": [{"seed": 7}, {"seed": 42}],
            },
        ],
    )

    manifest_path = tmp_path / "suite.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "name: suite_test",
                "title: Suite Test",
                "entries:",
                "  - label: synthetic-family",
                f"    report_dir: {synthetic_report.name}",
                "  - label: mnist-family",
                f"    report_dir: {mnist_report.name}",
            ]
        ),
        encoding="utf-8",
    )

    manifest = BenchmarkSuiteManifest.from_yaml(manifest_path)
    suite = build_suite_summary(manifest, tmp_path)

    assert suite["title"] == "Suite Test"
    assert len(suite["benchmarks"]) == 2
    assert suite["benchmarks"][0]["top_method"]["name"] == "gradmax"
    assert suite["method_coverage"][0]["name"] == "wide-cnn"

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_suite_markdown(output_dir / "summary.md", suite)
    write_suite_latex(output_dir / "summary.tex", suite)
    write_suite_plots(output_dir, suite)

    assert (output_dir / "summary.md").exists()
    assert (output_dir / "summary.tex").exists()
    assert (output_dir / "benchmark_epoch_curves.png").exists()
    assert (output_dir / "benchmark_final_accuracy.png").exists()
    assert (output_dir / "benchmark_leaderboards.png").exists()
    assert (output_dir / "method_coverage.png").exists()
    assert (output_dir / "synthetic_family_epoch_curves.png").exists()
    assert (output_dir / "synthetic_family_final_accuracy.png").exists()

    latex = (output_dir / "summary.tex").read_text(encoding="utf-8")
    assert "\\begin{table*}" in latex
    assert "wide-cnn" in latex
