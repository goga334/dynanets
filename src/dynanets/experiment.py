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
from dynanets.models.base import DynamicNeuralModel, NeuralModel
from dynanets.models.torch_mlp import DynamicMLPClassifier, TorchMLPClassifier
from dynanets.registry import Registry
from dynanets.search.base import SearchMethod
from dynanets.search.regularized_evolution import RegularizedEvolutionSearch


class ExperimentAssemblyError(ValueError):
    """Raised when a valid config requests an unsupported component combination."""


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
        config.validate()
        dataset = self.datasets.build(config.dataset.name, **config.dataset.params)
        model = self.models.build(config.model.name, **config.model.params)
        metrics = [self.metrics.build(item.name, **item.params) for item in config.metrics]
        adaptation = None
        if config.adaptation is not None:
            adaptation = self.adaptations.build(config.adaptation.name, **config.adaptation.params)
        search = None
        if config.search is not None:
            search = self.searches.build(config.search.name, **config.search.params)
        self._validate_compatibility(config=config, model=model, adaptation=adaptation, search=search)
        return Experiment(
            config=config,
            dataset=dataset,
            model=model,
            metrics=metrics,
            adaptation=adaptation,
            search=search,
        )

    def _validate_compatibility(
        self,
        *,
        config: ExperimentConfig,
        model: NeuralModel,
        adaptation: AdaptationMethod | None,
        search: SearchMethod | None,
    ) -> None:
        if adaptation is not None and not isinstance(model, DynamicNeuralModel):
            raise ExperimentAssemblyError(
                f"Adaptation '{config.adaptation.name}' requires a DynamicNeuralModel, got '{type(model).__name__}'"
            )
        if search is not None and config.adaptation is not None:
            raise ExperimentAssemblyError(
                "Search experiments and adaptation experiments must currently be run separately"
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