from pathlib import Path

from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.models.torch_mlp import DynamicMLPClassifier
from dynanets.reporting import summarize_run, write_csv, write_json, write_markdown, write_plots
from dynanets.runners.search import SearchRunner
from dynanets.runners.train import TrainingRunner
from dynanets.runtime import set_global_seed

import json
import torch



def test_net2wider_preserves_outputs() -> None:
    set_global_seed(123)
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01)
    inputs = torch.randn(16, 2)

    before = model.forward(inputs).detach().clone()
    from dynanets.adaptation import AdaptationEvent
    model.apply_adaptation(AdaptationEvent(event_type="net2wider", params={"amount": 3, "seed": 7}))
    after = model.forward(inputs).detach().clone()

    assert model.hidden_dim == 7
    assert torch.allclose(before, after, atol=1e-6)



def test_seeded_training_is_reproducible() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "repro-run",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 11}},
            "model": {
                "name": "dynamic_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "adaptation": {
                "name": "net2wider",
                "params": {"every_n_epochs": 1, "grow_by": 2, "max_hidden_dim": 8, "seed": 5},
            },
            "trainer": {"epochs": 3},
            "runtime": {"seed": 99},
        }
    )

    def run_once() -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, object]]]:
        set_global_seed(config.runtime["seed"])
        registries = default_registries()
        builder = ExperimentBuilder(
            datasets=registries["datasets"],
            models=registries["models"],
            metrics=registries["metrics"],
            adaptations=registries["adaptations"],
            searches=registries["searches"],
        )
        experiment = builder.build(config)
        dataset = experiment.dataset.build()
        summary = TrainingRunner().run(
            model=experiment.model,
            dataset=dataset,
            metrics=experiment.metrics,
            epochs=3,
            adaptation=experiment.adaptation,
        )
        return summary.train_history, summary.metric_history, summary.adaptation_history

    first = run_once()
    second = run_once()

    assert first == second



def test_search_best_score_matches_history_max() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "search-consistency",
            "dataset": {"name": "gaussian_blobs", "params": {"train_size": 64, "validation_size": 32, "seed": 17}},
            "model": {
                "name": "torch_mlp_classifier",
                "params": {"input_dim": 2, "hidden_dim": 4, "output_dim": 2, "lr": 0.01},
            },
            "metrics": [{"name": "accuracy", "params": {}}],
            "search": {
                "name": "regularized_evolution",
                "params": {
                    "hidden_dim_choices": [4, 8],
                    "activation_choices": ["relu", "tanh"],
                    "lr_choices": [0.01, 0.02],
                    "cycles": 4,
                    "population_size": 2,
                    "sample_size": 2,
                    "seed": 3,
                    "metric": "accuracy",
                },
            },
            "trainer": {"epochs": 2},
            "runtime": {"seed": 21},
        }
    )

    set_global_seed(config.runtime["seed"])
    registries = default_registries()
    builder = ExperimentBuilder(
        datasets=registries["datasets"],
        models=registries["models"],
        metrics=registries["metrics"],
        adaptations=registries["adaptations"],
        searches=registries["searches"],
    )
    experiment = builder.build(config)
    result = SearchRunner().run(config=config, experiment=experiment, registries=registries)

    history_scores = [entry["score"] for entry in result.evaluation_history]
    best_history_entry = max(result.evaluation_history, key=lambda item: item["score"])

    assert result.best_score == max(history_scores)
    assert result.best_model_params == best_history_entry["model_params"]



def test_reporting_outputs_are_consistent(tmp_path: Path) -> None:
    report = summarize_run(
        "demo-experiment",
        summary=type(
            "Summary",
            (),
            {
                "train_history": [{"loss": 0.9, "accuracy": 0.5}, {"loss": 0.4, "accuracy": 0.8}],
                "metric_history": [{"accuracy": 0.55}, {"accuracy": 0.82}],
                "adaptation_history": [
                    {"epoch": 0, "event_type": None, "params": {}, "metadata": {}, "applied": False, "reason": "schedule-not-reached"},
                    {"epoch": 1, "event_type": "net2wider", "params": {"amount": 2}, "metadata": {"hidden_dim": 10}, "applied": True, "reason": None},
                ],
            },
        )(),
        final_hidden_dim=12,
        metadata={"method_type": "dynamic", "notes": "demo notes"},
    )
    data = [report]

    write_json(tmp_path / "comparison.json", data)
    write_csv(tmp_path / "comparison.csv", data)
    write_plots(tmp_path, data)
    write_markdown(tmp_path / "comparison.md", data)

    json_data = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "comparison.csv").read_text(encoding="utf-8")
    markdown_text = (tmp_path / "comparison.md").read_text(encoding="utf-8")

    assert json_data[0]["final_val_accuracy"] == 0.82
    assert json_data[0]["adaptation_history"][1]["event_type"] == "net2wider"
    assert "demo-experiment" in csv_text
    assert "0.8200" in markdown_text
    assert (tmp_path / "validation_accuracy.png").exists()
    assert (tmp_path / "training_accuracy.png").exists()
    assert (tmp_path / "training_loss.png").exists()