from pathlib import Path

from dynanets.protocol import BenchmarkProtocol
from dynanets.stadium import (
    build_protocol_leaderboard,
    evaluate_protocol_acceptance,
    write_protocol_leaderboard_latex,
)


def _protocol(tmp_path: Path) -> BenchmarkProtocol:
    manifest = tmp_path / "protocol.yaml"
    manifest.write_text(
        """
name: official_test
track: synthetic
tier: official
seeds: [7, 42]
required_metrics: [accuracy]
description: Test protocol
acceptance:
  min_seeds: 2
  required_roles: [baseline, method]
  required_experiments: [baseline_cfg, method_cfg]
  minimum_methods: 1
  minimum_baselines: 1
  require_constraints: true
  require_runtime_environment: true
  require_stage_history: true
experiments:
  - config: baseline_cfg.yaml
    role: baseline
  - config: method_cfg.yaml
    role: method
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "baseline_cfg.yaml").write_text("name: baseline_cfg\n", encoding="utf-8")
    (tmp_path / "method_cfg.yaml").write_text("name: method_cfg\n", encoding="utf-8")
    return BenchmarkProtocol.from_yaml(manifest)



def test_protocol_acceptance_passes_for_complete_report(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    runs = [
        {
            "name": "baseline_cfg",
            "benchmark_seed": 7,
            "final_val_accuracy": 0.5,
            "metric_history": [{"accuracy": 0.5}],
        },
        {
            "name": "method_cfg",
            "benchmark_seed": 42,
            "final_val_accuracy": 0.6,
            "metric_history": [{"accuracy": 0.6}],
        },
    ]
    aggregate = [
        {
            "name": "baseline_cfg",
            "method_type": "baseline",
            "mean_final_val_accuracy": 0.5,
            "mean_best_val_accuracy": 0.5,
            "mean_parameter_count": 100,
            "mean_forward_flop_proxy": 200,
            "runtime_environment": {"requested_device": "auto", "resolved_device": "cpu", "torch_version": "2.x", "cuda_available": False},
            "representative_stage_history": [],
            "best_seed": 7,
        },
        {
            "name": "method_cfg",
            "method_type": "workflow",
            "mean_final_val_accuracy": 0.6,
            "mean_best_val_accuracy": 0.61,
            "mean_parameter_count": 120,
            "mean_forward_flop_proxy": 180,
            "runtime_environment": {"requested_device": "auto", "resolved_device": "cpu", "torch_version": "2.x", "cuda_available": False},
            "representative_stage_history": [{"name": "train", "epochs": 1, "epoch_start": 1, "epoch_end": 1, "adaptation_enabled": False, "final_val_accuracy": 0.6}],
            "best_seed": 42,
        },
    ]

    report = evaluate_protocol_acceptance(protocol, runs, aggregate, manifest_dir=tmp_path)

    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"])



def test_protocol_leaderboard_builds_pareto_rows(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    aggregate = [
        {
            "name": "baseline_cfg",
            "method_type": "baseline",
            "mean_final_val_accuracy": 0.5,
            "mean_best_val_accuracy": 0.52,
            "mean_parameter_count": 100,
            "mean_forward_flop_proxy": 200,
            "best_seed": 7,
        },
        {
            "name": "method_cfg",
            "method_type": "workflow",
            "mean_final_val_accuracy": 0.6,
            "mean_best_val_accuracy": 0.61,
            "mean_parameter_count": 120,
            "mean_forward_flop_proxy": 180,
            "best_seed": 42,
        },
    ]

    leaderboard = build_protocol_leaderboard(protocol, aggregate)

    assert leaderboard["top_overall"]["name"] == "method_cfg"
    assert leaderboard["top_method"]["name"] == "method_cfg"
    assert leaderboard["top_baseline"]["name"] == "baseline_cfg"
    assert leaderboard["accuracy_flop_pareto_frontier"]



def test_protocol_leaderboard_latex_writes_table(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    aggregate = [
        {
            "name": "baseline_cfg",
            "method_type": "baseline",
            "mean_final_val_accuracy": 0.5,
            "mean_best_val_accuracy": 0.52,
            "mean_parameter_count": 100,
            "mean_forward_flop_proxy": 200,
            "best_seed": 7,
        },
        {
            "name": "method_cfg",
            "method_type": "workflow",
            "mean_final_val_accuracy": 0.6,
            "mean_best_val_accuracy": 0.61,
            "mean_parameter_count": 120,
            "mean_forward_flop_proxy": 180,
            "best_seed": 42,
        },
    ]
    leaderboard = build_protocol_leaderboard(protocol, aggregate)
    output = tmp_path / "leaderboard.tex"
    write_protocol_leaderboard_latex(output, leaderboard)
    content = output.read_text(encoding="utf-8")

    assert "\\begin{table}" in content
    assert "method\\_cfg" in content
    assert "baseline\\_cfg" in content
