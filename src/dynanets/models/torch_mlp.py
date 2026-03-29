from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn

from dynanets.adaptation.events import AdaptationEvent
from dynanets.architecture import (
    MLPArchitectureSpec,
    build_mlp_network,
    grow_hidden_layer,
    insert_hidden_layer,
    mlp_spec_from_params,
    remove_hidden_layer,
    shrink_hidden_layer,
)
from dynanets.models.base import ArchitectureState, DynamicNeuralModel, NeuralModel
from dynanets.runtime import resolve_device


class TorchMLPClassifier(NeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 2,
        activation: str = "relu",
        lr: float = 1e-2,
        hidden_dims: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        self.spec = mlp_spec_from_params(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.device = resolve_device(device)
        self._lr = lr
        self._rebuild_from_spec()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.spec.hidden_dim

    @property
    def hidden_dims(self) -> list[int]:
        return list(self.spec.hidden_dims)

    @property
    def output_dim(self) -> int:
        return self.spec.output_dim

    @property
    def activation_name(self) -> str:
        return self.spec.hidden_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.to(self.device))

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.network.train()
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()

        input_grad_norm = 0.0
        linear_layers = self._linear_layers()
        if linear_layers and linear_layers[0].weight.grad is not None:
            input_grad_norm = float(linear_layers[0].weight.grad.norm().item())

        self.optimizer.step()
        self._after_optimizer_step()

        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {
            "loss": float(loss.item()),
            "accuracy": float(accuracy),
            "grad_norm_input_layer": input_grad_norm,
        }

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self.forward(batch["inputs"])

    def architecture_spec(self) -> MLPArchitectureSpec:
        return self.spec

    def init_params(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "lr": self._lr,
            "device": str(self.device),
        }

    def _rebuild_from_spec(self) -> None:
        self.network = build_mlp_network(self.spec).to(self.device)

    def _linear_layers(self, network: nn.Sequential | None = None) -> list[nn.Linear]:
        active = network if network is not None else self.network
        return [module for module in active if isinstance(module, nn.Linear)]

    def _parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def _after_optimizer_step(self) -> None:
        return None


class DynamicMLPClassifier(TorchMLPClassifier, DynamicNeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 2,
        activation: str = "relu",
        lr: float = 1e-2,
        hidden_dims: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            lr=lr,
            hidden_dims=hidden_dims,
            device=device,
        )
        self._weight_masks: list[torch.Tensor] = []
        self._reset_weight_masks()
        self._state = ArchitectureState(step=0, version=0, metadata={})
        self._refresh_architecture_state_metadata()

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        result = super().training_step(batch)
        self._state.step += 1
        self._refresh_architecture_state_metadata()
        return result

    def architecture_state(self) -> ArchitectureState:
        return self._state

    def clone(self) -> "DynamicMLPClassifier":
        clone = DynamicMLPClassifier(**self.init_params())
        clone.network.load_state_dict(self.network.state_dict())
        clone.optimizer = torch.optim.Adam(clone.network.parameters(), lr=self._lr)
        clone._weight_masks = [mask.clone().to(clone.device) for mask in self._weight_masks]
        clone._apply_active_weight_masks()
        clone._state = ArchitectureState(
            step=self._state.step,
            version=self._state.version,
            metadata=dict(self._state.metadata),
        )
        return clone

    def load_from(self, other: "DynamicMLPClassifier") -> None:
        self.spec = MLPArchitectureSpec.from_dict(other.spec.to_dict())
        self._lr = other._lr
        self._rebuild_from_spec()
        self.network.load_state_dict(other.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._weight_masks = [mask.clone().to(self.device) for mask in other._weight_masks]
        self._apply_active_weight_masks()
        self._state = ArchitectureState(
            step=other._state.step,
            version=other._state.version,
            metadata=dict(other._state.metadata),
        )

    def supported_event_types(self) -> set[str]:
        return {
            "grow_hidden",
            "prune_hidden",
            "net2wider",
            "insert_hidden_layer",
            "remove_hidden_layer",
            "apply_weight_mask",
        }

    def capabilities(self) -> dict[str, Any]:
        masked_weight_count, nonzero_parameter_count, weight_sparsity = self._weight_mask_statistics()
        return {
            "architecture_family": "mlp",
            "supports_dynamic_structure": True,
            "supports_weight_masks": True,
            "supported_event_types": sorted(self.supported_event_types()),
            "weight_transfer_strategies": [
                "copy_prefix_growth",
                "copy_prefix_pruning",
                "net2wider_function_preserving",
                "identity_layer_insertion",
                "matching_layer_copy_removal",
                "global_magnitude_mask_pruning",
            ],
            "current_hidden_dims": self.hidden_dims,
            "current_num_hidden_layers": len(self.hidden_dims),
            "parameter_count": self._parameter_count(),
            "masked_weight_count": masked_weight_count,
            "nonzero_parameter_count": nonzero_parameter_count,
            "weight_sparsity": weight_sparsity,
            "device": str(self.device),
        }

    def global_weight_threshold(self, target_sparsity: float) -> float:
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0.0, 1.0)")
        magnitudes = torch.cat([layer.weight.detach().abs().flatten() for layer in self._linear_layers()])
        total = magnitudes.numel()
        prune_count = int(total * target_sparsity)
        if prune_count <= 0:
            return -1.0
        if prune_count >= total:
            prune_count = total - 1
        values, _ = torch.sort(magnitudes)
        return float(values[prune_count - 1].item())

    def apply_adaptation(self, event: AdaptationEvent) -> None:
        action = event.event_type
        amount = int(event.params.get("amount", 0))

        if action == "grow_hidden":
            if amount <= 0:
                return
            mutated_spec = grow_hidden_layer(self.spec, layer_index=0, amount=amount)
            self._apply_simple_growth(mutated_spec)
            return
        if action == "prune_hidden":
            if amount <= 0:
                return
            mutated_spec = shrink_hidden_layer(
                self.spec,
                layer_index=0,
                amount=amount,
                min_width=int(event.params.get("min_width", 1)),
            )
            self._apply_simple_pruning(mutated_spec)
            return
        if action == "net2wider":
            if amount <= 0:
                return
            mutated_spec = grow_hidden_layer(self.spec, layer_index=0, amount=amount)
            self._apply_net2wider(mutated_spec=mutated_spec, seed=int(event.params.get("seed", 42)))
            return
        if action == "insert_hidden_layer":
            width = int(event.params.get("width", 0))
            if width <= 0:
                return
            layer_index = int(event.params.get("layer_index", len(self.hidden_dims)))
            mutated_spec = insert_hidden_layer(self.spec, layer_index=layer_index, width=width)
            self._apply_layer_insertion(mutated_spec, init_mode=str(event.params.get("init_mode", "random")))
            return
        if action == "remove_hidden_layer":
            if len(self.hidden_dims) <= 1:
                return
            layer_index = int(event.params.get("layer_index", len(self.hidden_dims) - 1))
            mutated_spec = remove_hidden_layer(self.spec, layer_index=layer_index)
            self._apply_layer_removal(mutated_spec)
            return
        if action == "apply_weight_mask":
            threshold = float(event.params.get("threshold", -1.0))
            target_sparsity = float(event.params.get("target_sparsity", 0.0))
            self._apply_weight_mask(threshold=threshold, target_sparsity=target_sparsity)
            return
        raise ValueError(f"Unsupported adaptation action '{action}'")

    def _after_optimizer_step(self) -> None:
        self._apply_active_weight_masks()

    def _reset_weight_masks(self) -> None:
        self._weight_masks = [torch.ones_like(layer.weight, device=self.device) for layer in self._linear_layers()]

    def _apply_active_weight_masks(self) -> None:
        if not self._weight_masks:
            return
        with torch.no_grad():
            for layer, mask in zip(self._linear_layers(), self._weight_masks):
                layer.weight.mul_(mask)

    def _apply_weight_mask(self, *, threshold: float, target_sparsity: float) -> None:
        if not self._weight_masks:
            self._reset_weight_masks()
        with torch.no_grad():
            for layer, mask in zip(self._linear_layers(), self._weight_masks):
                candidate = (layer.weight.detach().abs() > threshold).to(layer.weight.dtype)
                mask.mul_(candidate)
                layer.weight.mul_(mask)
        self._update_architecture_state()
        self._state.metadata["mask_threshold"] = threshold
        self._state.metadata["target_weight_sparsity"] = target_sparsity

    def _weight_mask_statistics(self) -> tuple[int, int, float]:
        linear_layers = self._linear_layers()
        if not linear_layers or not self._weight_masks:
            total_params = self._parameter_count()
            return 0, total_params, 0.0
        total_weight_params = sum(layer.weight.numel() for layer in linear_layers)
        active_weight_params = sum(int(mask.sum().item()) for mask in self._weight_masks)
        masked_weight_count = total_weight_params - active_weight_params
        total_params = self._parameter_count()
        nonzero_parameter_count = total_params - masked_weight_count
        weight_sparsity = (masked_weight_count / total_weight_params) if total_weight_params else 0.0
        return masked_weight_count, nonzero_parameter_count, weight_sparsity

    def _apply_simple_growth(self, mutated_spec: MLPArchitectureSpec) -> None:
        old_hidden_dim = self.hidden_dim
        new_hidden_dim = mutated_spec.hidden_dim
        old_input = self.network[0]
        old_output = self.network[-1]

        new_input = nn.Linear(self.input_dim, new_hidden_dim).to(self.device)
        new_output = nn.Linear(new_hidden_dim, self.output_dim).to(self.device)

        with torch.no_grad():
            new_input.weight[:old_hidden_dim] = old_input.weight
            new_input.bias[:old_hidden_dim] = old_input.bias
            nn.init.kaiming_uniform_(new_input.weight[old_hidden_dim:], a=5**0.5)
            nn.init.zeros_(new_input.bias[old_hidden_dim:])

            new_output.weight[:, :old_hidden_dim] = old_output.weight
            new_output.bias.copy_(old_output.bias)
            nn.init.kaiming_uniform_(new_output.weight[:, old_hidden_dim:], a=5**0.5)

        self.spec = mutated_spec
        self._replace_two_layer_network(new_input, new_output)

    def _apply_simple_pruning(self, mutated_spec: MLPArchitectureSpec) -> None:
        new_hidden_dim = mutated_spec.hidden_dim
        old_input = self.network[0]
        old_output = self.network[-1]

        new_input = nn.Linear(self.input_dim, new_hidden_dim).to(self.device)
        new_output = nn.Linear(new_hidden_dim, self.output_dim).to(self.device)

        with torch.no_grad():
            new_input.weight.copy_(old_input.weight[:new_hidden_dim])
            new_input.bias.copy_(old_input.bias[:new_hidden_dim])
            new_output.weight.copy_(old_output.weight[:, :new_hidden_dim])
            new_output.bias.copy_(old_output.bias)

        self.spec = mutated_spec
        self._replace_two_layer_network(new_input, new_output)

    def _apply_net2wider(self, mutated_spec: MLPArchitectureSpec, seed: int) -> None:
        old_hidden_dim = self.hidden_dim
        new_hidden_dim = mutated_spec.hidden_dim
        old_input = self.network[0]
        old_output = self.network[-1]
        new_input = nn.Linear(self.input_dim, new_hidden_dim).to(self.device)
        new_output = nn.Linear(new_hidden_dim, self.output_dim).to(self.device)
        rng = random.Random(seed)
        replicated_sources = [rng.randrange(old_hidden_dim) for _ in range(new_hidden_dim - old_hidden_dim)]
        replica_counts = {index: 1 for index in range(old_hidden_dim)}
        for source in replicated_sources:
            replica_counts[source] += 1

        with torch.no_grad():
            for index in range(old_hidden_dim):
                new_input.weight[index] = old_input.weight[index]
                new_input.bias[index] = old_input.bias[index]
                new_output.weight[:, index] = old_output.weight[:, index] / replica_counts[index]

            for offset, source in enumerate(replicated_sources):
                new_index = old_hidden_dim + offset
                new_input.weight[new_index] = old_input.weight[source]
                new_input.bias[new_index] = old_input.bias[source]
                new_output.weight[:, new_index] = old_output.weight[:, source] / replica_counts[source]

            new_output.bias.copy_(old_output.bias)

        self.spec = mutated_spec
        self._replace_two_layer_network(new_input, new_output)

    def _apply_layer_insertion(self, mutated_spec: MLPArchitectureSpec, init_mode: str = "random") -> None:
        old_network = self.network
        self.spec = mutated_spec
        new_network = build_mlp_network(self.spec).to(self.device)
        self._copy_matching_linear_layers(old_network, new_network)
        if init_mode == "identity":
            self._initialize_inserted_identity(new_network)
        self.network = new_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._reset_weight_masks()
        self._update_architecture_state()

    def _apply_layer_removal(self, mutated_spec: MLPArchitectureSpec) -> None:
        old_network = self.network
        self.spec = mutated_spec
        new_network = build_mlp_network(self.spec).to(self.device)
        self._copy_matching_linear_layers(old_network, new_network)
        self.network = new_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._reset_weight_masks()
        self._update_architecture_state()

    def _replace_two_layer_network(self, new_input: nn.Linear, new_output: nn.Linear) -> None:
        activation_module = build_mlp_network(self.spec)[1]
        self.network = nn.Sequential(new_input, activation_module, new_output).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._reset_weight_masks()
        self._update_architecture_state()

    def _copy_matching_linear_layers(self, old_network: nn.Sequential, new_network: nn.Sequential) -> None:
        old_linear_layers = self._linear_layers(old_network)
        new_linear_layers = self._linear_layers(new_network)
        used_old_indices: set[int] = set()
        with torch.no_grad():
            for new_layer in new_linear_layers:
                for old_index, old_layer in enumerate(old_linear_layers):
                    if old_index in used_old_indices:
                        continue
                    if (
                        old_layer.in_features == new_layer.in_features
                        and old_layer.out_features == new_layer.out_features
                    ):
                        new_layer.weight.copy_(old_layer.weight)
                        if old_layer.bias is not None and new_layer.bias is not None:
                            new_layer.bias.copy_(old_layer.bias)
                        used_old_indices.add(old_index)
                        break

    def _initialize_inserted_identity(self, new_network: nn.Sequential) -> None:
        linear_layers = self._linear_layers(new_network)
        if len(linear_layers) < 3:
            return
        inserted_layer = linear_layers[-2]
        if inserted_layer.in_features != inserted_layer.out_features:
            return
        with torch.no_grad():
            inserted_layer.weight.zero_()
            inserted_layer.weight += torch.eye(inserted_layer.in_features, device=inserted_layer.weight.device)
            if inserted_layer.bias is not None:
                inserted_layer.bias.zero_()

    def _refresh_architecture_state_metadata(self) -> None:
        masked_weight_count, nonzero_parameter_count, weight_sparsity = self._weight_mask_statistics()
        self._state.metadata["hidden_dim"] = self.hidden_dim
        self._state.metadata["hidden_dims"] = self.hidden_dims
        self._state.metadata["num_hidden_layers"] = len(self.hidden_dims)
        self._state.metadata["parameter_count"] = self._parameter_count()
        self._state.metadata["masked_weight_count"] = masked_weight_count
        self._state.metadata["nonzero_parameter_count"] = nonzero_parameter_count
        self._state.metadata["weight_sparsity"] = weight_sparsity
        self._state.metadata["device"] = str(self.device)
        self._state.metadata["supported_events"] = sorted(self.supported_event_types())

    def _update_architecture_state(self) -> None:
        self._state.version += 1
        self._refresh_architecture_state_metadata()
