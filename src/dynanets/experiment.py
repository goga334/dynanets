from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.adaptation.base import AdaptationMethod
from dynanets.adaptation.den import DENAdaptation
from dynanets.adaptation.dynamic_nodes import DynamicNodesAdaptation
from dynanets.adaptation.edge_growth import EdgeGrowthAdaptation
from dynanets.adaptation.gradmax import GradMaxAdaptation
from dynanets.adaptation.grow import WidthGrowthAdaptation
from dynanets.adaptation.insert_layer import LayerInsertionAdaptation
from dynanets.adaptation.nest import NeSTAdaptation
from dynanets.adaptation.net2deeper import Net2DeeperAdaptation
from dynanets.adaptation.net2net import Net2WiderAdaptation
from dynanets.adaptation.prune import WidthPruningAdaptation
from dynanets.adaptation.weights_connections import WeightsConnectionsAdaptation
from dynanets.config import ExperimentConfig
from dynanets.datasets.base import DatasetFactory
from dynanets.datasets.synthetic import (
    ConcentricCirclesDatasetFactory,
    GaussianBlobsDatasetFactory,
    TwoSpiralsDatasetFactory,
)
from dynanets.metrics.base import Metric
from dynanets.metrics.classification import AccuracyMetric
from dynanets.models.base import DynamicNeuralModel, NeuralModel
from dynanets.models.torch_mlp import DynamicMLPClassifier, TorchMLPClassifier
from dynanets.registry import Registry
from dynanets.runtime import prepare_factory_kwargs
from dynanets.search.base import SearchMethod
from dynanets.search.random_search import RandomSearch
from dynanets.search.regularized_evolution import RegularizedEvolutionSearch
from dynanets.workflows import (
    AdaNetRoundsWorkflow,
    MethodWorkflow,
    ScheduledWorkflow,
    SingleStageWorkflow,
    WorkflowStageConfig,
)


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
    workflow: MethodWorkflow


class ExperimentBuilder:
    def __init__(
        self,
        datasets: Registry[DatasetFactory],
        models: Registry[NeuralModel],
        metrics: Registry[Metric],
        adaptations: Registry[AdaptationMethod],
        searches: Registry[SearchMethod],
        workflows: Registry[MethodWorkflow] | None = None,
    ) -> None:
        self.datasets = datasets
        self.models = models
        self.metrics = metrics
        self.adaptations = adaptations
        self.searches = searches
        self.workflows = workflows if workflows is not None else _default_workflow_registry()

    def build(self, config: ExperimentConfig) -> Experiment:
        config.validate()
        dataset = self.datasets.build(config.dataset.name, **config.dataset.params)
        model_factory = self.models.get(config.model.name)
        model_params = prepare_factory_kwargs(model_factory, config.model.params, runtime=config.runtime)
        model = model_factory(**model_params)
        metrics = [self.metrics.build(item.name, **item.params) for item in config.metrics]
        adaptation = None
        if config.adaptation is not None:
            adaptation = self.adaptations.build(config.adaptation.name, **config.adaptation.params)
        search = None
        if config.search is not None:
            search = self.searches.build(config.search.name, **self._search_method_params(config.search.params))
        workflow = self._build_workflow(config)
        self._validate_compatibility(config=config, model=model, adaptation=adaptation, search=search)
        return Experiment(
            config=config,
            dataset=dataset,
            model=model,
            metrics=metrics,
            adaptation=adaptation,
            search=search,
            workflow=workflow,
        )

    def _search_method_params(self, params: dict[str, Any]) -> dict[str, Any]:
        excluded = {"hidden_dim_choices", "activation_choices", "lr_choices"}
        return {key: value for key, value in params.items() if key not in excluded}

    def _build_workflow(self, config: ExperimentConfig) -> MethodWorkflow:
        if config.workflow is None:
            return self.workflows.build("single_stage")
        params = dict(config.workflow.params)
        if config.workflow.name == "scheduled":
            stages = []
            for item in params.get("stages", []):
                stages.append(
                    WorkflowStageConfig(
                        name=str(item["name"]),
                        epochs=int(item["epochs"]),
                        adaptation_enabled=bool(item.get("adaptation_enabled", True)),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
            params["stages"] = stages
        return self.workflows.build(config.workflow.name, **params)

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
        if adaptation is not None and isinstance(model, DynamicNeuralModel):
            unsupported = adaptation.supported_event_types() - model.supported_event_types()
            if unsupported:
                unsupported_list = ", ".join(sorted(unsupported))
                raise ExperimentAssemblyError(
                    f"Adaptation '{config.adaptation.name}' requires unsupported event types: {unsupported_list}"
                )
        if search is not None and config.adaptation is not None:
            raise ExperimentAssemblyError(
                "Search experiments and adaptation experiments must currently be run separately"
            )
        if config.workflow is not None and config.workflow.name == "adanet_rounds":
            if adaptation is not None:
                raise ExperimentAssemblyError(
                    "AdaNet workflow manages structure selection internally and cannot be combined with adaptation"
                )
            if search is not None:
                raise ExperimentAssemblyError(
                    "AdaNet workflow cannot be combined with search in the same experiment"
                )
            if not isinstance(model, DynamicMLPClassifier):
                raise ExperimentAssemblyError(
                    "AdaNet workflow currently requires the dynamic_mlp_classifier model"
                )


def _default_workflow_registry() -> Registry[Any]:
    workflows: Registry[Any] = Registry()
    workflows.register("single_stage", SingleStageWorkflow)
    workflows.register("scheduled", ScheduledWorkflow)
    workflows.register("adanet_rounds", AdaNetRoundsWorkflow)
    return workflows


def default_registries() -> dict[str, Registry[Any]]:
    datasets: Registry[Any] = Registry()
    datasets.register("gaussian_blobs", GaussianBlobsDatasetFactory)
    datasets.register("two_spirals", TwoSpiralsDatasetFactory)
    datasets.register("concentric_circles", ConcentricCirclesDatasetFactory)

    models: Registry[Any] = Registry()
    models.register("torch_mlp_classifier", TorchMLPClassifier)
    models.register("dynamic_mlp_classifier", DynamicMLPClassifier)

    metrics: Registry[Any] = Registry()
    metrics.register("accuracy", AccuracyMetric)

    adaptations: Registry[Any] = Registry()
    adaptations.register("width_growth", WidthGrowthAdaptation)
    adaptations.register("net2wider", Net2WiderAdaptation)
    adaptations.register("net2deeper", Net2DeeperAdaptation)
    adaptations.register("insert_layer", LayerInsertionAdaptation)
    adaptations.register("prune_hidden", WidthPruningAdaptation)
    adaptations.register("gradmax", GradMaxAdaptation)
    adaptations.register("den", DENAdaptation)
    adaptations.register("nest", NeSTAdaptation)
    adaptations.register("dynamic_nodes", DynamicNodesAdaptation)
    adaptations.register("edge_growth", EdgeGrowthAdaptation)
    adaptations.register("weights_connections", WeightsConnectionsAdaptation)

    searches: Registry[Any] = Registry()
    searches.register("regularized_evolution", RegularizedEvolutionSearch)
    searches.register("random_search", RandomSearch)

    return {
        "datasets": datasets,
        "models": models,
        "metrics": metrics,
        "adaptations": adaptations,
        "searches": searches,
        "workflows": _default_workflow_registry(),
    }
