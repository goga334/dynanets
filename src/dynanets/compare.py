from __future__ import annotations

import argparse
from pathlib import Path

from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.models.base import DynamicNeuralModel
from dynanets.reporting import summarize_run, write_csv, write_json, write_markdown, write_plots
from dynanets.runners.search import SearchRunner
from dynanets.runners.train import TrainingRunner
from dynanets.runtime import set_global_seed



def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple dynanets experiments and compare them.")
    parser.add_argument("configs", nargs="+", help="Paths to YAML experiment configs")
    parser.add_argument("--output-dir", default="reports/initial_baselines", help="Directory for generated report files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for config_path in args.configs:
        config = ExperimentConfig.from_yaml(config_path)
        set_global_seed(config.runtime.get("seed"))
        registries = default_registries()
        builder = ExperimentBuilder(
            datasets=registries["datasets"],
            models=registries["models"],
            metrics=registries["metrics"],
            adaptations=registries["adaptations"],
            searches=registries["searches"],
        )
        experiment = builder.build(config)

        if experiment.search is not None:
            summary = SearchRunner().run(config=config, experiment=experiment, registries=registries)
            notes = (
                f"best model params={summary.best_model_params}; "
                f"evaluations={len(summary.evaluation_history)}; "
                f"search={summary.metadata.get('search_method')}"
            )
            reports.append(
                summarize_run(
                    config.name,
                    summary.best_summary,
                    final_hidden_dim=summary.final_hidden_dim,
                    metadata={"method_type": "search", "notes": notes},
                )
            )
            continue

        dataset = experiment.dataset.build()
        summary = TrainingRunner().run(
            model=experiment.model,
            dataset=dataset,
            metrics=experiment.metrics,
            epochs=int(config.trainer.get("epochs", 1)),
            adaptation=experiment.adaptation,
        )
        final_hidden_dim = None
        if isinstance(experiment.model, DynamicNeuralModel):
            final_hidden_dim = int(experiment.model.architecture_state().metadata.get("hidden_dim", 0))
        method_type = "dynamic" if experiment.adaptation is not None else "baseline"
        notes = None
        if experiment.adaptation is not None:
            notes = f"adaptation={config.adaptation.name}"
        reports.append(
            summarize_run(
                config.name,
                summary,
                final_hidden_dim=final_hidden_dim,
                metadata={"method_type": method_type, "notes": notes},
            )
        )

    write_json(output_dir / "comparison.json", reports)
    write_csv(output_dir / "comparison.csv", reports)
    write_plots(output_dir, reports)
    write_markdown(output_dir / "comparison.md", reports)

    print(output_dir / "comparison.md")
    for item in reports:
        print(
            f"{item['name']}: final_val_accuracy={item['final_val_accuracy']:.4f}, "
            f"best_val_accuracy={item['best_val_accuracy']:.4f}, adaptations={item['adaptations_applied']}"
        )


if __name__ == "__main__":
    main()