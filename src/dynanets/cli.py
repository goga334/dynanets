from __future__ import annotations

import argparse

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
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
        workflows=registries["workflows"],
    )
    experiment = builder.build(config)
    result = ExperimentExecutor().execute(config=config, experiment=experiment, registries=registries)

    print(f"experiment={config.name}")
    if result.mode == "search":
        print(f"search_best_score={result.best_score}")
        print(f"search_best_model_params={result.best_model_params}")
        print(f"train_history={result.summary.train_history}")
        print(f"metric_history={result.summary.metric_history}")
        print(f"stage_history={result.summary.stage_history}")
        print(f"search_history={result.search_history}")
        return

    print(f"train_history={result.summary.train_history}")
    print(f"metric_history={result.summary.metric_history}")
    print(f"stage_history={result.summary.stage_history}")
    print(f"adaptation_history={result.summary.adaptation_history}")


if __name__ == "__main__":
    main()
