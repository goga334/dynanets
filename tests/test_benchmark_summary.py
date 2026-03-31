from pathlib import Path

from dynanets.benchmark import aggregate_benchmark_runs, write_aggregate_markdown, write_benchmark_plots



def test_aggregate_benchmark_runs_computes_mean_std_and_representation(tmp_path: Path) -> None:
    runs = [
        {
            "name": "demo-method",
            "benchmark_seed": 7,
            "final_val_accuracy": 0.60,
            "best_val_accuracy": 0.62,
            "adaptations_applied": 2,
            "final_hidden_dim": 10,
            "architecture_spec": {"input_dim": 20, "hidden_dims": [10], "output_dim": 2},
            "architecture_graph": {
                "name": "demo-method",
                "nodes": [
                    {"id": "input", "label": "Input (20)", "op": "input", "params": {}},
                    {"id": "hidden_1", "label": "Hidden 1 (10)", "op": "linear", "params": {}},
                    {"id": "output", "label": "Output (2)", "op": "linear", "params": {}},
                ],
                "edges": [
                    {"source": "input", "target": "hidden_1", "metadata": {}},
                    {"source": "hidden_1", "target": "output", "metadata": {}},
                ],
                "metadata": {"family": "mlp"},
            },
            "constraints": {
                "parameter_count": 232,
                "nonzero_parameter_count": 180,
                "masked_weight_count": 52,
                "weight_sparsity": 0.2241,
                "forward_flop_proxy": 420,
                "activation_elements": 12,
            },
            "stage_history": [{"name": "adapt", "epochs": 2, "epoch_start": 1, "epoch_end": 2, "adaptation_enabled": True, "final_val_accuracy": 0.62}],
            "metadata": {
                "method_type": "dynamic",
                "notes": "demo; device=cpu; requested_device=auto; cuda_available=False",
                "runtime_environment": {"resolved_device": "cpu", "requested_device": "auto", "cuda_available": False},
            },
        },
        {
            "name": "demo-method",
            "benchmark_seed": 11,
            "final_val_accuracy": 0.70,
            "best_val_accuracy": 0.74,
            "adaptations_applied": 3,
            "final_hidden_dim": 12,
            "architecture_spec": {"input_dim": 20, "hidden_dims": [12], "output_dim": 2},
            "architecture_graph": {
                "name": "demo-method",
                "nodes": [
                    {"id": "input", "label": "Input (20)", "op": "input", "params": {}},
                    {"id": "hidden_1", "label": "Hidden 1 (12)", "op": "linear", "params": {}},
                    {"id": "output", "label": "Output (2)", "op": "linear", "params": {}},
                ],
                "edges": [
                    {"source": "input", "target": "hidden_1", "metadata": {}},
                    {"source": "hidden_1", "target": "output", "metadata": {}},
                ],
                "metadata": {"family": "mlp"},
            },
            "constraints": {
                "parameter_count": 274,
                "nonzero_parameter_count": 190,
                "masked_weight_count": 84,
                "weight_sparsity": 0.3066,
                "forward_flop_proxy": 504,
                "activation_elements": 14,
            },
            "stage_history": [{"name": "adapt", "epochs": 2, "epoch_start": 1, "epoch_end": 2, "adaptation_enabled": True, "final_val_accuracy": 0.74}],
            "metadata": {
                "method_type": "dynamic",
                "notes": "demo; device=cpu; requested_device=auto; cuda_available=False",
                "runtime_environment": {"resolved_device": "cpu", "requested_device": "auto", "cuda_available": False},
            },
        },
    ]

    aggregate = aggregate_benchmark_runs(runs)
    assert len(aggregate) == 1
    item = aggregate[0]

    assert abs(item["mean_final_val_accuracy"] - 0.65) < 1e-9
    assert item["best_seed"] == 11
    assert item["representative_architecture_spec"]["hidden_dims"] == [12]
    assert item["representative_architecture_graph"]["nodes"][1]["label"] == "Hidden 1 (12)"
    assert item["representative_stage_history"][0]["name"] == "adapt"
    assert item["runtime_environment"]["resolved_device"] == "cpu"
    assert item["mean_parameter_count"] == 253
    assert item["mean_nonzero_parameter_count"] == 185
    assert abs(item["mean_weight_sparsity"] - 0.26535) < 1e-9

    write_benchmark_plots(tmp_path, aggregate)
    assert (tmp_path / "mean_final_val_accuracy.png").exists()
    assert (tmp_path / "per_seed_final_val_accuracy.png").exists()

    output = tmp_path / "summary.md"
    write_aggregate_markdown(output, aggregate, [7, 11])
    text = output.read_text(encoding="utf-8")

    assert "Seeds: 7, 11" in text
    assert "Mean final validation accuracy" in text
    assert "Constraint Summary" in text
    assert "Experiment Notes" in text
    assert "Representative Stage Histories" in text
    assert "Representative Architectures" in text
    assert "Hidden 1 (12)" in text
    assert "requested_device=auto" in text
    assert "Mean nonzero params" in text
