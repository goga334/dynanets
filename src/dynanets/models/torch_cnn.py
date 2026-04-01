from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynanets.adaptation.events import AdaptationEvent
from dynanets.architecture import CNNArchitectureSpec, build_cnn_network, cnn_spec_from_params
from dynanets.constraints import ConstraintEvaluator
from dynanets.models.base import ArchitectureState, DynamicNeuralModel
from dynanets.runtime import resolve_device
from dynanets.sparsity import (
    MaskAwareSparsityState,
    channel_importance,
    magnitude_mask,
    resolve_keep_count,
    select_topk_indices,
)


class TorchCNNClassifier(DynamicNeuralModel):
    def __init__(
        self,
        input_channels: int,
        input_size: int | list[int] | tuple[int, int] = (28, 28),
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        classifier_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        lr_schedule: str = "none",
        min_lr_ratio: float = 0.1,
        label_smoothing: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.spec = cnn_spec_from_params(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=conv_channels or [16, 32],
            classifier_hidden_dims=classifier_hidden_dims,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )
        self.device = resolve_device(device)
        self._lr = float(lr)
        self._optimizer_name = str(optimizer_name).lower()
        self._momentum = float(momentum)
        self._weight_decay = float(weight_decay)
        self._lr_schedule = str(lr_schedule).lower()
        self._min_lr_ratio = float(min_lr_ratio)
        self._label_smoothing = float(label_smoothing)
        self._batch_norm_sparsity_strength = 0.0
        self._sparsity_state = MaskAwareSparsityState()
        self._state = ArchitectureState(step=0, version=0, metadata={})
        self.network = build_cnn_network(self.spec).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing)
        self.optimizer = self._build_optimizer()
        self._sync_weight_masks()
        self._refresh_state_metadata()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        return self.network(inputs)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.network.train()
        inputs = batch["inputs"]
        targets = batch["targets"].to(self.device)

        self._configure_optimizer_for_epoch(
            epoch=int(batch.get("epoch", 0)),
            total_epochs=int(batch.get("total_epochs", 1)),
        )
        self.optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        bn_scale_l1 = self._batch_norm_scale_l1()
        regularized_loss = loss + (self._batch_norm_sparsity_strength * bn_scale_l1)
        regularized_loss.backward()
        self.optimizer.step()
        self._apply_active_weight_masks()

        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        self._state.step += 1
        self._refresh_state_metadata()
        return {
            "loss": float(loss.item()),
            "regularized_loss": float(regularized_loss.item()),
            "accuracy": float(accuracy),
            "bn_scale_l1": float(bn_scale_l1.item()),
        }

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self.forward(batch["inputs"])

    def architecture_spec(self) -> CNNArchitectureSpec:
        return self.spec

    def architecture_state(self) -> ArchitectureState:
        return self._state

    def init_params(self) -> dict[str, Any]:
        return {
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
            "conv_channels": self.spec.conv_channels,
            "classifier_hidden_dims": list(self.spec.classifier_hidden_dims),
            "activation": self.spec.activation,
            "use_batch_norm": self.spec.use_batch_norm,
            "lr": self._lr,
            "optimizer_name": self._optimizer_name,
            "momentum": self._momentum,
            "weight_decay": self._weight_decay,
            "lr_schedule": self._lr_schedule,
            "min_lr_ratio": self._min_lr_ratio,
            "label_smoothing": self._label_smoothing,
            "device": str(self.device),
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer_name == "sgd":
            return torch.optim.SGD(
                self.network.parameters(),
                lr=self._lr,
                momentum=self._momentum,
                weight_decay=self._weight_decay,
            )
        if self._optimizer_name == "adam":
            return torch.optim.Adam(
                self.network.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
            )
        raise ValueError("optimizer_name must be one of: adam, sgd")

    def _configure_optimizer_for_epoch(self, *, epoch: int, total_epochs: int) -> None:
        lr = self._scheduled_lr(epoch=epoch, total_epochs=total_epochs)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _scheduled_lr(self, *, epoch: int, total_epochs: int) -> float:
        if self._lr_schedule == "none" or total_epochs <= 1:
            return self._lr
        if self._lr_schedule == "cosine":
            progress = max(0.0, min(1.0, float(epoch) / float(max(1, total_epochs - 1))))
            min_lr = self._lr * self._min_lr_ratio
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return min_lr + ((self._lr - min_lr) * cosine)
        raise ValueError("lr_schedule must be one of: none, cosine")

    def clone(self) -> "TorchCNNClassifier":
        clone = TorchCNNClassifier(**self.init_params())
        clone.network.load_state_dict(self.network.state_dict())
        clone.optimizer = clone._build_optimizer()
        clone._batch_norm_sparsity_strength = self._batch_norm_sparsity_strength
        clone._sparsity_state = self._sparsity_state.clone(device=clone.device)
        clone._sync_weight_masks()
        clone._apply_active_weight_masks()
        clone._state = ArchitectureState(
            step=self._state.step,
            version=self._state.version,
            metadata=dict(self._state.metadata),
        )
        return clone

    def load_from(self, other: "TorchCNNClassifier") -> None:
        self.spec = CNNArchitectureSpec.from_dict(other.spec.to_dict())
        self._lr = other._lr
        self._optimizer_name = other._optimizer_name
        self._momentum = other._momentum
        self._weight_decay = other._weight_decay
        self._lr_schedule = other._lr_schedule
        self._min_lr_ratio = other._min_lr_ratio
        self._label_smoothing = other._label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing)
        self.network = build_cnn_network(self.spec).to(self.device)
        self.network.load_state_dict(other.network.state_dict())
        self.optimizer = self._build_optimizer()
        self._batch_norm_sparsity_strength = other._batch_norm_sparsity_strength
        self._sparsity_state = other._sparsity_state.clone(device=self.device)
        self._sync_weight_masks()
        self._apply_active_weight_masks()
        self._state = ArchitectureState(
            step=other._state.step,
            version=other._state.version,
            metadata=dict(other._state.metadata),
        )

    def set_batch_norm_sparsity(self, strength: float) -> None:
        self._batch_norm_sparsity_strength = max(0.0, float(strength))
        self._refresh_state_metadata()

    def structure_state(self) -> dict[str, Any]:
        return self._state.to_dict()

    def supported_event_types(self) -> set[str]:
        return {"prune_channels", "apply_weight_mask", "merge_hidden_layers"}

    def capabilities(self) -> dict[str, Any]:
        masked_weight_count, nonzero_parameter_count, weight_sparsity = self._weight_mask_statistics()
        supported_event_types = sorted(self.supported_event_types())
        return {
            "architecture_family": "cnn",
            "supports_dynamic_structure": True,
            "supports_batch_norm_sparsity": self.spec.use_batch_norm,
            "supports_weight_masks": True,
            "supports_mask_aware_sparsity_state": True,
            "structured_pruning_primitives": ["prune_channels"],
            "merge_primitives": ["merge_hidden_layers"],
            "supported_event_types": supported_event_types,
            "supported_structural_ops": supported_event_types,
            "weight_transfer_strategies": [
                "structured_channel_pruning",
                "global_magnitude_mask_pruning",
                "layerwise_magnitude_mask_pruning",
                "classifier_layer_merge",
            ],
            "conv_channels": self.spec.conv_channels,
            "classifier_hidden_dims": list(self.spec.classifier_hidden_dims),
            "parameter_count": self._parameter_count(),
            "masked_weight_count": masked_weight_count,
            "nonzero_parameter_count": nonzero_parameter_count,
            "weight_sparsity": weight_sparsity,
            "device": str(self.device),
        }

    def global_weight_threshold(self, target_sparsity: float) -> float:
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0.0, 1.0)")
        named_weights = self._named_prunable_weights()
        magnitudes = torch.cat([tensor.detach().abs().flatten() for tensor in named_weights.values()])
        total = magnitudes.numel()
        prune_count = int(total * target_sparsity)
        if prune_count <= 0:
            return -1.0
        if prune_count >= total:
            prune_count = total - 1
        values, _ = torch.sort(magnitudes)
        return float(values[prune_count - 1].item())

    def layerwise_weight_thresholds(self, prune_fraction: float) -> dict[str, float]:
        if not 0.0 <= prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0.0, 1.0)")
        self._sync_weight_masks()
        thresholds: dict[str, float] = {}
        for name, weight in self._named_prunable_weights().items():
            mask = self._sparsity_state.masks.get(name)
            active = weight.detach().abs()
            if mask is not None:
                active = active[mask > 0]
            else:
                active = active.flatten()
            total = active.numel()
            prune_count = int(total * prune_fraction)
            if prune_count <= 0 or total == 0:
                thresholds[name] = -1.0
                continue
            if prune_count >= total:
                prune_count = total - 1
            values, _ = torch.sort(active.flatten())
            thresholds[name] = float(values[prune_count - 1].item())
        return thresholds

    def apply_adaptation(self, event: AdaptationEvent) -> None:
        if event.event_type == "prune_channels":
            self.prune_channels(
                prune_fraction=float(event.params.get("prune_fraction", 0.0)),
                min_channels_per_block=int(event.params.get("min_channels_per_block", 4)),
            )
            return
        if event.event_type == "apply_weight_mask":
            threshold = float(event.params.get("threshold", -1.0))
            target_sparsity = float(event.params.get("target_sparsity", 0.0))
            thresholds_by_name = (
                {str(name): float(value) for name, value in event.params.get("thresholds_by_name", {}).items()}
                if "thresholds_by_name" in event.params
                else None
            )
            self._apply_weight_mask(
                threshold=threshold,
                target_sparsity=target_sparsity,
                thresholds_by_name=thresholds_by_name,
            )
            return
        if event.event_type == "merge_hidden_layers":
            self.merge_classifier_layers(merge_index=int(event.params.get("merge_index", 0)))
            return
        raise ValueError(f"Unsupported adaptation action '{event.event_type}'")

    def prune_channels(self, *, prune_fraction: float, min_channels_per_block: int = 4) -> dict[str, Any]:
        if not self.spec.use_batch_norm:
            raise ValueError("Channel pruning currently requires batch normalization")
        if not 0.0 <= prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0.0, 1.0)")
        keep_indices: list[torch.Tensor] = []
        kept_channels: list[int] = []
        pruned = False
        for block, bn_layer in zip(self._feature_blocks(self.network, self.spec), self._batch_norm_layers()):
            importance = channel_importance(
                conv_weight=block["conv"].weight.detach(),
                batch_norm_weight=bn_layer.weight.detach(),
            )
            out_channels = importance.numel()
            keep_count = resolve_keep_count(
                out_channels,
                prune_fraction=prune_fraction,
                min_count=min_channels_per_block,
            )
            if keep_count < out_channels:
                pruned = True
            keep = select_topk_indices(importance, keep_count)
            keep_indices.append(keep.to(self.device))
            kept_channels.append(int(keep_count))

        if not pruned:
            return {
                "pruned": False,
                "before_conv_channels": self.spec.conv_channels,
                "after_conv_channels": self.spec.conv_channels,
            }

        mutated = CNNArchitectureSpec.from_dict(
            {
                **self.spec.to_dict(),
                "blocks": [
                    {**dict(block), "out_channels": kept}
                    for block, kept in zip(self.spec.to_dict()["blocks"], kept_channels)
                ],
            }
        )
        self._apply_pruned_channels(mutated, keep_indices)
        return {
            "pruned": True,
            "before_conv_channels": self._state.metadata.get("conv_channels_before_prune", []),
            "after_conv_channels": self.spec.conv_channels,
            "kept_channels": kept_channels,
        }


    def merge_classifier_layers(self, *, merge_index: int = 0) -> dict[str, Any]:
        hidden_dims = list(self.spec.classifier_hidden_dims)
        if len(hidden_dims) < 2:
            raise ValueError("Layer merge currently requires at least two classifier hidden layers")
        if merge_index < 0 or merge_index >= len(hidden_dims) - 1:
            raise ValueError("merge_index must point to an adjacent hidden-layer pair")

        before_hidden_dims = list(hidden_dims)
        merged_hidden_dims = hidden_dims[:merge_index] + [hidden_dims[merge_index + 1]] + hidden_dims[merge_index + 2 :]
        mutated = CNNArchitectureSpec.from_dict(
            {
                **self.spec.to_dict(),
                "classifier_hidden_dims": merged_hidden_dims,
            }
        )
        self._apply_merged_classifier_layers(mutated, merge_index=merge_index, before_hidden_dims=before_hidden_dims)
        return {
            "merged": True,
            "merge_index": merge_index,
            "before_classifier_hidden_dims": before_hidden_dims,
            "after_classifier_hidden_dims": list(self.spec.classifier_hidden_dims),
        }

    def _named_prunable_weights(self, network: nn.Module | None = None, spec: CNNArchitectureSpec | None = None) -> dict[str, torch.Tensor]:
        active_network = network if network is not None else self.network
        active_spec = spec if spec is not None else self.spec
        weights: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self._feature_blocks(active_network, active_spec)):
            weights[f"conv_{index}.weight"] = block["conv"].weight
        for index, layer in enumerate(self._classifier_linear_layers(active_network)):
            weights[f"linear_{index}.weight"] = layer.weight
        return weights

    def _sync_weight_masks(self) -> None:
        self._sparsity_state.sync(self._named_prunable_weights())

    def _apply_active_weight_masks(self) -> None:
        self._sync_weight_masks()
        self._sparsity_state.apply_(self._named_prunable_weights())

    def _apply_weight_mask(
        self,
        *,
        threshold: float,
        target_sparsity: float,
        thresholds_by_name: dict[str, float] | None = None,
    ) -> None:
        self._sync_weight_masks()
        named_weights = self._named_prunable_weights()
        candidate_masks = {
            name: magnitude_mask(weight, thresholds_by_name.get(name, threshold) if thresholds_by_name is not None else threshold)
            for name, weight in named_weights.items()
        }
        self._sparsity_state.multiply_(candidate_masks)
        self._apply_active_weight_masks()
        self._state.version += 1
        self._refresh_state_metadata()
        if thresholds_by_name is None:
            self._state.metadata["mask_threshold"] = threshold
        else:
            self._state.metadata["mask_threshold"] = None
            self._state.metadata["mask_thresholds_by_name"] = {
                name: round(value, 8) for name, value in thresholds_by_name.items()
            }
        self._state.metadata["target_weight_sparsity"] = target_sparsity

    def _weight_mask_statistics(self) -> tuple[int, int, float]:
        named_weights = self._named_prunable_weights()
        if not named_weights:
            total_params = self._parameter_count()
            return 0, total_params, 0.0
        self._sync_weight_masks()
        statistics = self._sparsity_state.statistics(named_weights)
        total_params = self._parameter_count()
        nonzero_parameter_count = total_params - statistics.masked_params
        return statistics.masked_params, nonzero_parameter_count, statistics.weight_sparsity

    def _apply_pruned_channels(self, mutated_spec: CNNArchitectureSpec, keep_indices: list[torch.Tensor]) -> None:
        old_network = self.network
        old_spec = self.spec
        old_conv_channels = list(old_spec.conv_channels)
        old_blocks = self._feature_blocks(old_network, old_spec)
        old_linears = self._classifier_linear_layers(old_network)
        old_masks = self._sparsity_state.clone(device=self.device)

        self.spec = mutated_spec
        new_network = build_cnn_network(self.spec).to(self.device)
        new_blocks = self._feature_blocks(new_network, self.spec)
        new_linears = self._classifier_linear_layers(new_network)

        prev_keep = torch.arange(old_spec.input_channels, device=self.device)
        transferred_masks: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for index, (old_block, new_block, keep) in enumerate(zip(old_blocks, new_blocks, keep_indices)):
                old_conv = old_block["conv"]
                new_conv = new_block["conv"]
                new_conv.weight.copy_(old_conv.weight[keep][:, prev_keep])
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias[keep])

                old_conv_mask = old_masks.masks.get(f"conv_{index}.weight")
                if old_conv_mask is not None:
                    transferred_masks[f"conv_{index}.weight"] = old_conv_mask[keep][:, prev_keep].clone()

                old_bn = old_block.get("bn")
                new_bn = new_block.get("bn")
                if old_bn is not None and new_bn is not None:
                    new_bn.weight.copy_(old_bn.weight[keep])
                    new_bn.bias.copy_(old_bn.bias[keep])
                    new_bn.running_mean.copy_(old_bn.running_mean[keep])
                    new_bn.running_var.copy_(old_bn.running_var[keep])
                prev_keep = keep

            if old_linears and new_linears:
                new_linears[0].weight.copy_(old_linears[0].weight[:, prev_keep])
                if old_linears[0].bias is not None and new_linears[0].bias is not None:
                    new_linears[0].bias.copy_(old_linears[0].bias)
                old_linear_mask = old_masks.masks.get("linear_0.weight")
                if old_linear_mask is not None:
                    transferred_masks["linear_0.weight"] = old_linear_mask[:, prev_keep].clone()
                for index, (old_linear, new_linear) in enumerate(zip(old_linears[1:], new_linears[1:]), start=1):
                    new_linear.weight.copy_(old_linear.weight)
                    if old_linear.bias is not None and new_linear.bias is not None:
                        new_linear.bias.copy_(old_linear.bias)
                    old_mask = old_masks.masks.get(f"linear_{index}.weight")
                    if old_mask is not None:
                        transferred_masks[f"linear_{index}.weight"] = old_mask.clone()

        self.network = new_network
        self.optimizer = self._build_optimizer()
        self._sparsity_state = MaskAwareSparsityState(masks=transferred_masks)
        self._sync_weight_masks()
        self._apply_active_weight_masks()
        self._state.version += 1
        self._refresh_state_metadata()
        self._state.metadata["conv_channels_before_prune"] = old_conv_channels


    def _apply_merged_classifier_layers(
        self,
        mutated_spec: CNNArchitectureSpec,
        *,
        merge_index: int,
        before_hidden_dims: list[int],
    ) -> None:
        old_network = self.network
        old_spec = self.spec
        old_blocks = self._feature_blocks(old_network, old_spec)
        old_linears = self._classifier_linear_layers(old_network)

        self.spec = mutated_spec
        new_network = build_cnn_network(self.spec).to(self.device)
        new_blocks = self._feature_blocks(new_network, self.spec)
        new_linears = self._classifier_linear_layers(new_network)

        with torch.no_grad():
            for old_block, new_block in zip(old_blocks, new_blocks):
                new_block["conv"].weight.copy_(old_block["conv"].weight)
                if old_block["conv"].bias is not None and new_block["conv"].bias is not None:
                    new_block["conv"].bias.copy_(old_block["conv"].bias)
                old_bn = old_block.get("bn")
                new_bn = new_block.get("bn")
                if old_bn is not None and new_bn is not None:
                    new_bn.weight.copy_(old_bn.weight)
                    new_bn.bias.copy_(old_bn.bias)
                    new_bn.running_mean.copy_(old_bn.running_mean)
                    new_bn.running_var.copy_(old_bn.running_var)

            for index in range(merge_index):
                new_linears[index].weight.copy_(old_linears[index].weight)
                if old_linears[index].bias is not None and new_linears[index].bias is not None:
                    new_linears[index].bias.copy_(old_linears[index].bias)

            first = old_linears[merge_index]
            second = old_linears[merge_index + 1]
            merged_weight = second.weight @ first.weight
            first_bias = first.bias if first.bias is not None else torch.zeros(first.out_features, device=self.device)
            second_bias = second.bias if second.bias is not None else torch.zeros(second.out_features, device=self.device)
            merged_bias = (second.weight @ first_bias) + second_bias
            new_linears[merge_index].weight.copy_(merged_weight)
            if new_linears[merge_index].bias is not None:
                new_linears[merge_index].bias.copy_(merged_bias)

            for old_index in range(merge_index + 2, len(old_linears)):
                new_index = old_index - 1
                new_linears[new_index].weight.copy_(old_linears[old_index].weight)
                if old_linears[old_index].bias is not None and new_linears[new_index].bias is not None:
                    new_linears[new_index].bias.copy_(old_linears[old_index].bias)

        self.network = new_network
        self.optimizer = self._build_optimizer()
        self._sparsity_state = MaskAwareSparsityState()
        self._sync_weight_masks()
        self._apply_active_weight_masks()
        self._state.version += 1
        self._refresh_state_metadata()
        self._state.metadata["classifier_hidden_dims_before_merge"] = before_hidden_dims
        self._state.metadata["merge_index"] = merge_index

    def _feature_blocks(self, network: nn.Module, spec: CNNArchitectureSpec) -> list[dict[str, nn.Module]]:
        modules = list(network.features)
        blocks: list[dict[str, nn.Module]] = []
        index = 0
        for block_spec in spec.blocks:
            block: dict[str, nn.Module] = {"conv": modules[index]}
            index += 1
            if spec.use_batch_norm:
                block["bn"] = modules[index]
                index += 1
            block["activation"] = modules[index]
            index += 1
            if block_spec.pool is not None:
                block["pool"] = modules[index]
                index += 1
            blocks.append(block)
        return blocks

    def _classifier_linear_layers(self, network: nn.Module) -> list[nn.Linear]:
        return [module for module in network.classifier if isinstance(module, nn.Linear)]

    def _batch_norm_layers(self) -> list[nn.BatchNorm2d]:
        return [module for module in self.network.features if isinstance(module, nn.BatchNorm2d)]

    def _batch_norm_scale_l1(self) -> torch.Tensor:
        bn_layers = self._batch_norm_layers()
        if not bn_layers:
            return torch.tensor(0.0, device=self.device)
        return sum(layer.weight.abs().sum() for layer in bn_layers)

    def _parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def _refresh_state_metadata(self) -> None:
        masked_weight_count, nonzero_parameter_count, weight_sparsity = self._weight_mask_statistics()
        self._state.metadata["conv_channels"] = self.spec.conv_channels
        self._state.metadata["num_conv_blocks"] = len(self.spec.blocks)
        self._state.metadata["classifier_hidden_dims"] = list(self.spec.classifier_hidden_dims)
        self._state.metadata["nonzero_parameter_count"] = nonzero_parameter_count
        self._state.metadata["masked_weight_count"] = masked_weight_count
        self._state.metadata["weight_sparsity"] = weight_sparsity
        self._state.metadata["mask_state_names"] = self._sparsity_state.named_mask_keys()
        self._state.metadata["device"] = str(self.device)
        self._state.metadata["optimizer_name"] = self._optimizer_name
        self._state.metadata["lr_schedule"] = self._lr_schedule
        self._state.metadata["weight_decay"] = self._weight_decay
        self._state.metadata["label_smoothing"] = self._label_smoothing
        self._state.metadata["use_batch_norm"] = self.spec.use_batch_norm
        self._state.metadata["batch_norm_sparsity_strength"] = self._batch_norm_sparsity_strength
        self._state.metadata["supported_events"] = sorted(self.supported_event_types())
        constraint_summary = ConstraintEvaluator().evaluate(
            architecture_spec=self.spec,
            metadata=self._state.metadata,
        )
        self._state.metadata.update(constraint_summary.to_dict())

