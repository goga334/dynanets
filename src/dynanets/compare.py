from __future__ import annotations

import argparse
from pathlib import Path

from dynanets.config import ExperimentConfig
from dynanets.execution import ExperimentExecutor
from dynanets.experiment import ExperimentBuilder, default_registries
from dynanets.reporting import write_csv, write_json, write_markdown, write_plots
from dynanets.runtime import set_global_seed



def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple dynanets experiments and compare them.")
    parser.add_argument("configs", nargs="+", help="Paths to YAML experiment configs")
    parser.add_argument("--output-dir", default="reports/initial_baselines", help="Directory for generated report files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    executor = ExperimentExecutor()
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
            workflows=registries["workflows"],
        )
        experiment = builder.build(config)
        result = executor.execute(config=config, experiment=experiment, registries=registries)
        reports.append(result.to_report_item())

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
