from __future__ import annotations

import argparse

from dynanets.config import ExperimentConfig
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.runners.search import SearchRunner
from dynanets.runners.train import TrainingRunner
from dynanets.runtime import set_global_seed



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dynanets experiment config.")
    parser.add_argument("config", help="Path to a YAML experiment config")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
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
        print(f"experiment={config.name}")
        print(f"search_best_score={summary.best_score}")
        print(f"search_best_model_params={summary.best_model_params}")
        print(f"train_history={summary.best_summary.train_history}")
        print(f"metric_history={summary.best_summary.metric_history}")
        print(f"search_history={summary.evaluation_history}")
        return

    dataset = experiment.dataset.build()
    runner = TrainingRunner()
    summary = runner.run(
        model=experiment.model,
        dataset=dataset,
        metrics=experiment.metrics,
        epochs=int(config.trainer.get("epochs", 1)),
        adaptation=experiment.adaptation,
    )
    print(f"experiment={config.name}")
    print(f"train_history={summary.train_history}")
    print(f"metric_history={summary.metric_history}")
    print(f"adaptation_history={summary.adaptation_history}")


if __name__ == "__main__":
    main()