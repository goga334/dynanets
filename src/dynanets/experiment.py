from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.adaptation.base import AdaptationMethod
from dynanets.adaptation.grow import WidthGrowthAdaptation
from dynanets.adaptation.net2net import Net2WiderAdaptation
from dynanets.config import ExperimentConfig
from dynanets.datasets.base import DatasetFactory
from dynanets.datasets.synthetic import GaussianBlobsDatasetFactory
from dynanets.metrics.base import Metric
from dynanets.metrics.classification import AccuracyMetric
from dynanets.models.base import NeuralModel
from dynanets.models.torch_mlp import DynamicMLPClassifier, TorchMLPClassifier
from dynanets.registry import Registry
from dynanets.search.base import SearchMethod
from dynanets.search.regularized_evolution import RegularizedEvolutionSearch


@dataclass(slots=True)
class Experiment:
    config: ExperimentConfig
    dataset: DatasetFactory
    model: NeuralModel
    metrics: list[Metric]
    adaptation: AdaptationMethod | None
    search: SearchMethod | None


class ExperimentBuilder:
    def __init__(
        self,
        datasets: Registry[DatasetFactory],
        models: Registry[NeuralModel],
        metrics: Registry[Metric],
        adaptations: Registry[AdaptationMethod],
        searches: Registry[SearchMethod],
    ) -> None:
        self.datasets = datasets
        self.models = models
        self.metrics = metrics
        self.adaptations = adaptations
        self.searches = searches

    def build(self, config: ExperimentConfig) -> Experiment:
        dataset = self.datasets.build(config.dataset.name, **config.dataset.params)
        model = self.models.build(config.model.name, **config.model.params)
        metrics = [self.metrics.build(item.name, **item.params) for item in config.metrics]
        adaptation = None
        if config.adaptation is not None:
            adaptation = self.adaptations.build(config.adaptation.name, **config.adaptation.params)
        search = None
        if config.search is not None:
            search = self.searches.build(config.search.name, **config.search.params)
        return Experiment(
            config=config,
            dataset=dataset,
            model=model,
            metrics=metrics,
            adaptation=adaptation,
            search=search,
        )



def default_registries() -> dict[str, Registry[Any]]:
    datasets: Registry[Any] = Registry()
    datasets.register("gaussian_blobs", GaussianBlobsDatasetFactory)

    models: Registry[Any] = Registry()
    models.register("torch_mlp_classifier", TorchMLPClassifier)
    models.register("dynamic_mlp_classifier", DynamicMLPClassifier)

    metrics: Registry[Any] = Registry()
    metrics.register("accuracy", AccuracyMetric)

    adaptations: Registry[Any] = Registry()
    adaptations.register("width_growth", WidthGrowthAdaptation)
    adaptations.register("net2wider", Net2WiderAdaptation)

    searches: Registry[Any] = Registry()
    searches.register("regularized_evolution", RegularizedEvolutionSearch)

    return {
        "datasets": datasets,
        "models": models,
        "metrics": metrics,
        "adaptations": adaptations,
        "searches": searches,
    }