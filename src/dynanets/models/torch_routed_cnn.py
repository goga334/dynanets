from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dynanets.architecture import CNNArchitectureSpec, cnn_spec_from_params
from dynanets.constraints import ConstraintEvaluator
from dynanets.models.base import NeuralModel
from dynanets.routing import (
    GateConfig,
    accept_rate_penalty,
    blend_gate_scores,
    budget_penalty,
    distillation_loss,
    weighted_distillation_loss,
    gate_accept_targets,
    gate_entropy,
    gate_scores,
    initial_gate_bias,
    relaxed_accept_probability,
    select_accept_mask,
    staged_accept_rate,
)
from dynanets.runtime import resolve_device

_DYNAMIC_WIDTH_FAMILIES = {'slimmable_pyramid', 'channel_gate_pyramid', 'instance_sparse_pyramid'}
_EARLY_EXIT_FAMILIES = {'early_exit_cascade', 'skip_cascade', 'iterative_refine'}


class TorchRoutedCNNClassifier(nn.Module, NeuralModel):
    def __init__(
        self,
        input_channels: int,
        input_size: int | list[int] | tuple[int, int] = (28, 28),
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        activation: str = "relu",
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        lr_schedule: str = "none",
        min_lr_ratio: float = 0.1,
        label_smoothing: float = 0.0,
        width_multipliers: list[float] | None = None,
        eval_width_multipliers: list[float] | None = None,
        routing_policy: str = "dynamic_width",
        routing_family: str | None = None,
        gate_mode: str = "metric",
        confidence_threshold: float = 0.85,
        min_confidence_threshold: float | None = None,
        gate_metric: str = "confidence",
        gate_temperature: float = 0.25,
        eval_threshold_offset: float = 0.0,
        eval_min_threshold: float | None = None,
        gate_metric_blend: float = 0.0,
        gate_budget_weight: float = 0.0,
        target_cost_ratio: float = 1.0,
        distillation_weight: float = 0.25,
        gate_supervision_weight: float = 0.0,
        teacher_confidence_floor: float = 0.55,
        gate_entropy_weight: float = 0.0,
        target_accept_rate: float | None = None,
        gate_accept_rate_weight: float = 0.0,
        gate_target_strategy: str = "hybrid",
        student_confidence_floor: float = 0.35,
        min_target_cost_ratio: float | None = None,
        min_target_accept_rate: float | None = None,
        route_trace_limit: int = 8,
        early_exit_loss_weight: float = 0.3,
        early_exit_focus_weight: float = 0.0,
        early_exit_focus_floor: float = 0.0,
        dynamic_width_focus_weight: float = 0.75,
        dynamic_width_focus_floor: float = 0.10,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.device = resolve_device(device)
        self._lr = float(lr)
        self._optimizer_name = str(optimizer_name).lower()
        self._momentum = float(momentum)
        self._weight_decay = float(weight_decay)
        self._lr_schedule = str(lr_schedule).lower()
        self._min_lr_ratio = float(min_lr_ratio)
        self._label_smoothing = float(label_smoothing)
        self.routing_policy = str(routing_policy)
        if self.routing_policy not in {"largest", "dynamic_width", "early_exit"}:
            raise ValueError("routing_policy must be one of: largest, dynamic_width, early_exit")
        self.routing_family = self._resolve_routing_family(self.routing_policy, routing_family)
        resolved_conv_channels = self._resolve_conv_channels(conv_channels, self.routing_family)
        self.spec = cnn_spec_from_params(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=resolved_conv_channels,
            classifier_hidden_dims=[],
            activation=activation,
            use_batch_norm=False,
            metadata={"routing_family": self.routing_family, "routing_policy": self.routing_policy},
        )
        self.gate_config = GateConfig(
            mode=str(gate_mode),
            metric=str(gate_metric),
            threshold=float(confidence_threshold),
            min_threshold=(float(min_confidence_threshold) if min_confidence_threshold is not None else None),
            temperature=float(gate_temperature),
            eval_threshold_offset=float(eval_threshold_offset),
            eval_min_threshold=(float(eval_min_threshold) if eval_min_threshold is not None else None),
            metric_blend=float(gate_metric_blend),
            budget_weight=float(gate_budget_weight),
            target_cost_ratio=float(target_cost_ratio),
            target_cost_ratio_min=(float(min_target_cost_ratio) if min_target_cost_ratio is not None else None),
            distillation_weight=float(distillation_weight),
            supervision_weight=float(gate_supervision_weight),
            teacher_confidence_floor=float(teacher_confidence_floor),
            student_confidence_floor=float(student_confidence_floor),
            target_strategy=str(gate_target_strategy),
            entropy_weight=float(gate_entropy_weight),
            target_accept_rate=(float(target_accept_rate) if target_accept_rate is not None else None),
            target_accept_rate_min=(float(min_target_accept_rate) if min_target_accept_rate is not None else None),
            accept_rate_weight=float(gate_accept_rate_weight),
            trace_limit=int(route_trace_limit),
        )
        if self.gate_config.mode not in {"metric", "learned"}:
            raise ValueError("gate_mode must be one of: metric, learned")
        self.confidence_threshold = float(confidence_threshold)
        self.min_confidence_threshold = float(min_confidence_threshold) if min_confidence_threshold is not None else None
        self.early_exit_loss_weight = float(early_exit_loss_weight)
        self.early_exit_focus_weight = float(early_exit_focus_weight)
        self.early_exit_focus_floor = float(early_exit_focus_floor)
        self.dynamic_width_focus_weight = float(dynamic_width_focus_weight)
        self.dynamic_width_focus_floor = float(dynamic_width_focus_floor)
        training_widths = width_multipliers or [1.0, 0.75, 0.5]
        self.width_multipliers = self._normalize_widths(training_widths)
        eval_widths = eval_width_multipliers or self.width_multipliers
        self.eval_width_multipliers = sorted(self._normalize_widths(eval_widths))

        self.block_channels = list(self.spec.conv_channels)
        self.exit_stage_index = self._resolve_exit_stage_index()
        self.gate_stage_index = self._resolve_gate_stage_index()
        self.conv_layers = nn.ModuleList()
        in_channels_value = self.spec.input_channels
        for out_channels in self.block_channels:
            self.conv_layers.append(nn.Conv2d(in_channels_value, out_channels, kernel_size=3, padding=1))
            in_channels_value = out_channels
        exit_channels = self.block_channels[self.exit_stage_index]
        gate_channels = self.block_channels[self.gate_stage_index]
        final_channels = self.block_channels[-1]
        self.early_head = nn.Linear(exit_channels, self.spec.num_classes)
        self.final_head = nn.Linear(final_channels, self.spec.num_classes)
        self.early_gate_head = nn.Linear(exit_channels, 1)
        self.width_gate_head = nn.Linear(gate_channels, 1)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing)
        self.gate_loss_fn = nn.BCELoss()
        self._initialize_gate_heads()
        self.to(self.device)
        self.optimizer = self._build_optimizer()
        self._last_route_summary: dict[str, Any] = {}
        self._route_trace_history: list[dict[str, Any]] = []
        self._latest_epoch: int = 0
        self._latest_total_epochs: int = 8


    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer_name == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self._lr,
                momentum=self._momentum,
                weight_decay=self._weight_decay,
            )
        if self._optimizer_name == "adam":
            return torch.optim.Adam(
                self.parameters(),
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, logits = self._forward_width(inputs, 1.0)
        return logits

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.train()
        inputs = self._prepare_inputs(batch["inputs"])
        targets = batch["targets"].to(self.device)
        self._latest_epoch = int(batch.get("epoch", self._latest_epoch))
        self._latest_total_epochs = int(batch.get("total_epochs", self._latest_total_epochs))
        progress = self._training_progress(self._latest_epoch, self._latest_total_epochs)
        threshold = self.gate_config.threshold_for_progress(progress)
        target_cost_ratio = self.gate_config.target_cost_ratio_for_progress(progress)
        target_accept_rate = self.gate_config.target_accept_rate_for_progress(progress)

        self._configure_optimizer_for_epoch(
            epoch=self._latest_epoch,
            total_epochs=self._latest_total_epochs,
        )
        self.optimizer.zero_grad()
        ordered_widths = sorted(self.width_multipliers, reverse=True)
        width_outputs: list[dict[str, Any]] = []
        max_logits: torch.Tensor | None = None
        for multiplier in ordered_widths:
            early_logits, final_logits, early_pooled, gate_pooled = self._forward_width_features(inputs, multiplier)
            width_outputs.append(
                {
                    "multiplier": multiplier,
                    "early_logits": early_logits,
                    "final_logits": final_logits,
                    "early_pooled": early_pooled,
                    "gate_pooled": gate_pooled,
                }
            )
            if abs(multiplier - max(ordered_widths)) < 1e-8:
                max_logits = final_logits

        assert max_logits is not None
        teacher_logits = max_logits.detach()
        losses: list[torch.Tensor] = [self.loss_fn(max_logits, targets)]
        route_mean_cost = torch.ones((), device=self.device)
        route_mean_width = torch.ones((), device=self.device)
        gate_loss_total = torch.zeros((), device=self.device)

        if self.routing_policy == "dynamic_width":
            smaller = sorted(width_outputs, key=lambda item: item["multiplier"])
            unresolved_prob = torch.ones(inputs.shape[0], device=self.device)
            expected_cost = torch.zeros(inputs.shape[0], device=self.device)
            expected_width = torch.zeros(inputs.shape[0], device=self.device)
            route_probs: dict[str, float] = {}
            stage_targets: dict[str, float | None] = {}
            gate_probabilities: list[torch.Tensor] = []
            gate_entropies: list[float] = []
            for index, item in enumerate(smaller):
                multiplier = float(item["multiplier"])
                final_logits = item["final_logits"]
                gate_pooled = item["gate_pooled"]
                if multiplier < 1.0:
                    width_gate_targets = gate_accept_targets(
                        final_logits,
                        teacher_logits,
                        targets=targets,
                        teacher_confidence_floor=self.gate_config.teacher_confidence_floor,
                        student_confidence_floor=self.gate_config.student_confidence_floor,
                        strategy=self.gate_config.target_strategy,
                    )
                    width_focus = 1.0 + (self.dynamic_width_focus_weight * width_gate_targets.clamp_min(self.dynamic_width_focus_floor))
                    per_sample_loss = F.cross_entropy(final_logits, targets, reduction="none")
                    supervised = (per_sample_loss * width_focus).mean()
                    if self.gate_config.distillation_weight > 0.0:
                        supervised = supervised + (
                            self.gate_config.distillation_weight
                            * weighted_distillation_loss(final_logits, teacher_logits, weights=width_focus)
                        )
                else:
                    supervised = self.loss_fn(final_logits, targets)
                losses.append(supervised)

                if index == len(smaller) - 1:
                    accept_prob = unresolved_prob
                    stage_target = None
                else:
                    score = self._dynamic_gate_score(gate_pooled, final_logits)
                    gate_prob = relaxed_accept_probability(
                        score,
                        threshold=threshold,
                        temperature=self.gate_config.temperature,
                    )
                    gate_probabilities.append(gate_prob)
                    remaining_gated_stages = len(smaller) - index - 1
                    stage_target = staged_accept_rate(target_accept_rate, remaining_gated_stages=remaining_gated_stages)
                    accept_prob = unresolved_prob * gate_prob
                    gate_loss = self._gate_supervision_loss(score, final_logits, teacher_logits, targets)
                    gate_loss_total = gate_loss_total + gate_loss
                    if self.gate_config.supervision_weight > 0.0:
                        losses.append(self.gate_config.supervision_weight * gate_loss)
                    losses.extend(self._gate_regularization_losses(gate_prob, target_accept_rate=stage_target))
                    gate_entropies.append(float(gate_entropy(gate_prob).detach().item()))
                expected_cost = expected_cost + (accept_prob * self._full_path_cost_ratio(multiplier))
                expected_width = expected_width + (accept_prob * multiplier)
                unresolved_prob = unresolved_prob - accept_prob
                route_probs[str(multiplier)] = float(accept_prob.mean().detach().item())
                stage_targets[str(multiplier)] = (round(float(stage_target), 4) if stage_target is not None else None)

            if gate_probabilities:
                route_entropy_value = float(sum(gate_entropies) / len(gate_entropies)) if gate_entropies else 0.0
                accept_rate_value = float(sum(prob.mean().detach().item() for prob in gate_probabilities) / len(gate_probabilities))
            else:
                route_entropy_value = 0.0
                accept_rate_value = 1.0

            route_mean_cost = expected_cost.mean()
            route_mean_width = expected_width.mean()
            losses.append(
                budget_penalty(
                    expected_cost_ratio=route_mean_cost,
                    target_cost_ratio=target_cost_ratio,
                    weight=self.gate_config.budget_weight,
                )
            )
            self._last_route_summary = {
                "policy": self.routing_policy,
                "routing_family": self.routing_family,
                "mode": "train",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "threshold": round(float(threshold), 4),
                "target_cost_ratio": round(float(target_cost_ratio), 4),
                "target_accept_rate": (round(float(target_accept_rate), 4) if target_accept_rate is not None else None),
                "trained_widths": [round(value, 4) for value in sorted(self.width_multipliers)],
                "route_probabilities": route_probs,
                "stage_target_accept_rates": stage_targets,
                "mean_width": round(float(route_mean_width.detach().item()), 4),
                "mean_cost_ratio": round(float(route_mean_cost.detach().item()), 4),
                "route_entropy": round(route_entropy_value, 6),
                "accept_rate": round(accept_rate_value, 4),
                "gate_loss": round(float(gate_loss_total.detach().item()), 6),
            }
        elif self.routing_policy == "early_exit":
            item = width_outputs[0]
            early_logits = item["early_logits"]
            final_logits = item["final_logits"]
            early_pooled = item["early_pooled"]
            gate_targets = gate_accept_targets(
                early_logits,
                teacher_logits,
                targets=targets,
                teacher_confidence_floor=self.gate_config.teacher_confidence_floor,
                student_confidence_floor=self.gate_config.student_confidence_floor,
                strategy=self.gate_config.target_strategy,
            )
            focus_weights = 1.0 + (self.early_exit_focus_weight * gate_targets.clamp_min(self.early_exit_focus_floor))
            per_sample_early_loss = F.cross_entropy(early_logits, targets, reduction="none")
            early_loss = (per_sample_early_loss * focus_weights).mean()
            distill = (
                weighted_distillation_loss(early_logits, teacher_logits, weights=focus_weights)
                if self.gate_config.distillation_weight > 0.0
                else torch.zeros((), device=self.device)
            )
            losses.append((self.early_exit_loss_weight * early_loss) + (self.gate_config.distillation_weight * distill))
            score = self._early_gate_score(early_pooled, early_logits)
            gate_loss_total = self._gate_supervision_loss(score, early_logits, teacher_logits, targets)
            if self.gate_config.supervision_weight > 0.0:
                losses.append(self.gate_config.supervision_weight * gate_loss_total)
            early_prob = relaxed_accept_probability(score, threshold=threshold, temperature=self.gate_config.temperature)
            losses.extend(self._gate_regularization_losses(early_prob, target_accept_rate=target_accept_rate))
            route_mean_cost = ((early_prob * self._early_exit_cost_ratio()) + (1.0 - early_prob)).mean()
            route_mean_width = torch.ones((), device=self.device)
            losses.append(
                budget_penalty(
                    expected_cost_ratio=route_mean_cost,
                    target_cost_ratio=target_cost_ratio,
                    weight=self.gate_config.budget_weight,
                )
            )
            self._last_route_summary = {
                "policy": self.routing_policy,
                "routing_family": self.routing_family,
                "mode": "train",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "threshold": round(float(threshold), 4),
                "target_cost_ratio": round(float(target_cost_ratio), 4),
                "target_accept_rate": (round(float(target_accept_rate), 4) if target_accept_rate is not None else None),
                "expected_early_exit_fraction": round(float(early_prob.mean().detach().item()), 4),
                "focus_weight_mean": round(float(focus_weights.mean().detach().item()), 4),
                "route_entropy": round(float(gate_entropy(early_prob).detach().item()), 6),
                "mean_width": 1.0,
                "mean_cost_ratio": round(float(route_mean_cost.detach().item()), 4),
                "gate_loss": round(float(gate_loss_total.detach().item()), 6),
            }
        else:
            self._last_route_summary = {
                "policy": self.routing_policy,
                "routing_family": self.routing_family,
                "mode": "train",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "threshold": round(float(threshold), 4),
                "mean_width": 1.0,
                "mean_cost_ratio": 1.0,
                "gate_loss": 0.0,
            }

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        self.optimizer.step()

        accuracy = (max_logits.argmax(dim=1) == targets).float().mean().item()
        return {
            "loss": float(total_loss.item()),
            "accuracy": float(accuracy),
            "route_mean_width": float(route_mean_width.detach().item()),
            "route_mean_cost_ratio": float(route_mean_cost.detach().item()),
            "gate_threshold": float(threshold),
            "gate_loss": float(gate_loss_total.detach().item()),
        }

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        inputs = self._prepare_inputs(batch["inputs"])
        with torch.no_grad():
            if self.routing_policy == "dynamic_width":
                return self._evaluate_dynamic_width(inputs)
            if self.routing_policy == "early_exit":
                return self._evaluate_early_exit(inputs)
            _, logits = self._forward_width(inputs, 1.0)
            self._last_route_summary = {
                "policy": "largest",
                "routing_family": self.routing_family,
                "mode": "eval",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "used_width": 1.0,
                "mean_width": 1.0,
                "mean_cost_ratio": 1.0,
            }
            return logits

    def architecture_spec(self) -> CNNArchitectureSpec:
        return self.spec

    def structure_state(self) -> dict[str, Any]:
        metadata = self._structure_metadata()
        return {"metadata": metadata}

    def route_summary(self) -> dict[str, Any]:
        return dict(self._last_route_summary)

    def route_trace(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._route_trace_history]

    def init_params(self) -> dict[str, Any]:
        return {
            "input_channels": self.spec.input_channels,
            "input_size": list(self.spec.input_size),
            "num_classes": self.spec.num_classes,
            "conv_channels": list(self.spec.conv_channels),
            "activation": self.spec.activation,
            "lr": self._lr,
            "optimizer_name": self._optimizer_name,
            "momentum": self._momentum,
            "weight_decay": self._weight_decay,
            "lr_schedule": self._lr_schedule,
            "min_lr_ratio": self._min_lr_ratio,
            "label_smoothing": self._label_smoothing,
            "width_multipliers": list(self.width_multipliers),
            "eval_width_multipliers": list(self.eval_width_multipliers),
            "routing_policy": self.routing_policy,
            "routing_family": self.routing_family,
            "gate_mode": self.gate_config.mode,
            "confidence_threshold": self.confidence_threshold,
            "min_confidence_threshold": self.min_confidence_threshold,
            "gate_metric": self.gate_config.metric,
            "gate_temperature": self.gate_config.temperature,
            "eval_threshold_offset": self.gate_config.eval_threshold_offset,
            "eval_min_threshold": self.gate_config.eval_min_threshold,
            "gate_metric_blend": self.gate_config.metric_blend,
            "gate_budget_weight": self.gate_config.budget_weight,
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "distillation_weight": self.gate_config.distillation_weight,
            "gate_supervision_weight": self.gate_config.supervision_weight,
            "teacher_confidence_floor": self.gate_config.teacher_confidence_floor,
            "gate_entropy_weight": self.gate_config.entropy_weight,
            "target_accept_rate": self.gate_config.target_accept_rate,
            "gate_accept_rate_weight": self.gate_config.accept_rate_weight,
            "gate_target_strategy": self.gate_config.target_strategy,
            "student_confidence_floor": self.gate_config.student_confidence_floor,
            "min_target_cost_ratio": self.gate_config.target_cost_ratio_min,
            "min_target_accept_rate": self.gate_config.target_accept_rate_min,
            "route_trace_limit": self.gate_config.trace_limit,
            "early_exit_loss_weight": self.early_exit_loss_weight,
            "early_exit_focus_weight": self.early_exit_focus_weight,
            "early_exit_focus_floor": self.early_exit_focus_floor,
            "dynamic_width_focus_weight": self.dynamic_width_focus_weight,
            "dynamic_width_focus_floor": self.dynamic_width_focus_floor,
            "device": str(self.device),
        }

    def _initialize_gate_heads(self) -> None:
        default_accept = self.gate_config.target_accept_rate
        width_default = 0.55 if self.routing_policy == "dynamic_width" else 0.35
        if self.routing_family == "channel_gate_pyramid":
            width_default = 0.50
        elif self.routing_family == "instance_sparse_pyramid":
            width_default = 0.46
        elif self.routing_family == "skip_cascade":
            width_default = 0.30
        elif self.routing_family == "iterative_refine":
            width_default = 0.26
        width_bias = initial_gate_bias(default_accept, default=width_default)
        early_bias = initial_gate_bias(default_accept, default=width_default)
        nn.init.zeros_(self.early_gate_head.weight)
        nn.init.zeros_(self.width_gate_head.weight)
        nn.init.constant_(self.early_gate_head.bias, early_bias)
        nn.init.constant_(self.width_gate_head.bias, width_bias)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        resolved = inputs.to(self.device)
        if resolved.ndim == 3:
            resolved = resolved.unsqueeze(1)
        return resolved

    def _forward_width(self, inputs: torch.Tensor, multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
        early_logits, final_logits, _, _ = self._forward_width_features(inputs, multiplier)
        return early_logits, final_logits

    def _forward_width_features(
        self,
        inputs: torch.Tensor,
        multiplier: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        channels = self._width_channels(multiplier)
        pooled_features: list[torch.Tensor] = []
        current = inputs
        in_channels_value = self.spec.input_channels
        for index, conv in enumerate(self.conv_layers):
            out_channels = channels[index]
            current = F.conv2d(current, conv.weight[:out_channels, :in_channels_value], conv.bias[:out_channels], padding=1)
            current = self._apply_activation(current)
            current = F.max_pool2d(current, kernel_size=2, stride=2)
            pooled_features.append(F.adaptive_avg_pool2d(current, (1, 1)).flatten(1))
            in_channels_value = out_channels

        early_pooled = pooled_features[self.exit_stage_index]
        gate_pooled = pooled_features[self.gate_stage_index]
        final_pooled = pooled_features[-1]
        early_channels = channels[self.exit_stage_index]
        final_channels = channels[-1]
        early_logits = F.linear(early_pooled, self.early_head.weight[:, :early_channels], self.early_head.bias)
        final_logits = F.linear(final_pooled, self.final_head.weight[:, :final_channels], self.final_head.bias)
        return early_logits, final_logits, early_pooled, gate_pooled

    def _evaluate_dynamic_width(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        outputs = torch.empty(batch_size, self.spec.num_classes, device=self.device)
        unresolved = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        selected_widths = torch.full((batch_size,), float(self.eval_width_multipliers[-1]), device=self.device)
        counts: dict[str, int] = {str(mult): 0 for mult in self.eval_width_multipliers}
        stage_targets: dict[str, float | None] = {}
        progress = self._training_progress(self._latest_epoch, self._latest_total_epochs)
        threshold = self.gate_config.eval_threshold_for_progress(progress)
        target_accept_rate = self.gate_config.target_accept_rate_for_progress(progress)

        for index, multiplier in enumerate(self.eval_width_multipliers):
            _, logits, _, pooled2 = self._forward_width_features(inputs, multiplier)
            score = self._dynamic_gate_score(pooled2, logits)
            if index == len(self.eval_width_multipliers) - 1:
                selected = unresolved.clone()
                stage_target = None
            else:
                remaining_indices = unresolved.nonzero(as_tuple=False).flatten()
                selected = torch.zeros_like(unresolved)
                remaining_gated_stages = len(self.eval_width_multipliers) - index - 1
                stage_target = staged_accept_rate(target_accept_rate, remaining_gated_stages=remaining_gated_stages)
                if remaining_indices.numel() > 0:
                    confidence_scores = gate_scores(logits[remaining_indices], metric="confidence")
                    eligible_mask = confidence_scores >= float(self.gate_config.student_confidence_floor)
                    local_selected = select_accept_mask(
                        score[remaining_indices],
                        threshold=threshold,
                        target_accept_rate=stage_target,
                        eligible_mask=eligible_mask,
                    )
                    selected_indices = remaining_indices[local_selected]
                    if selected_indices.numel() > 0:
                        selected[selected_indices] = True
            outputs[selected] = logits[selected]
            selected_widths[selected] = float(multiplier)
            counts[str(multiplier)] += int(selected.sum().item())
            stage_targets[str(multiplier)] = (round(float(stage_target), 4) if stage_target is not None else None)
            unresolved = unresolved & (~selected)
            if not unresolved.any():
                break

        total = max(1, batch_size)
        mean_width = sum(float(multiplier) * counts[str(multiplier)] for multiplier in self.eval_width_multipliers) / total
        mean_cost_ratio = sum(self._full_path_cost_ratio(multiplier) * counts[str(multiplier)] for multiplier in self.eval_width_multipliers) / total
        trace_samples = [
            {"sample": index, "width": round(float(selected_widths[index].item()), 4)}
            for index in range(min(batch_size, max(1, int(self.gate_config.trace_limit))))
        ]
        self._record_route_trace(
            {
                "policy": "dynamic_width",
                "routing_family": self.routing_family,
                "mode": "eval",
                "threshold": round(float(threshold), 4),
                "target_accept_rate": (round(float(target_accept_rate), 4) if target_accept_rate is not None else None),
                "trace_samples": trace_samples,
            }
        )
        self._last_route_summary = {
            "policy": "dynamic_width",
            "routing_family": self.routing_family,
            "mode": "eval",
            "gate_mode": self.gate_config.mode,
            "gate_metric": self.gate_config.metric,
            "confidence_threshold": round(float(threshold), 4),
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "target_accept_rate": self.gate_config.target_accept_rate,
            "stage_target_accept_rates": stage_targets,
            "route_counts": counts,
            "trace_samples": trace_samples,
            "mean_width": round(float(mean_width), 4),
            "mean_cost_ratio": round(float(mean_cost_ratio), 4),
        }
        return outputs

    def _evaluate_early_exit(self, inputs: torch.Tensor) -> torch.Tensor:
        early_logits, final_logits, pooled1, _ = self._forward_width_features(inputs, 1.0)
        progress = self._training_progress(self._latest_epoch, self._latest_total_epochs)
        threshold = self.gate_config.eval_threshold_for_progress(progress)
        target_accept_rate = self.gate_config.target_accept_rate_for_progress(progress)
        score = self._early_gate_score(pooled1, early_logits)
        confidence_scores = gate_scores(early_logits, metric="confidence")
        eligible_mask = confidence_scores >= float(self.gate_config.student_confidence_floor)
        use_early = select_accept_mask(
            score,
            threshold=threshold,
            target_accept_rate=target_accept_rate,
            eligible_mask=eligible_mask,
        )
        outputs = final_logits.clone()
        outputs[use_early] = early_logits[use_early]
        total = max(1, inputs.shape[0])
        early_count = int(use_early.sum().item())
        full_count = total - early_count
        mean_cost_ratio = ((early_count * self._early_exit_cost_ratio()) + full_count) / total
        trace_samples = [
            {"sample": index, "path": ("early" if bool(use_early[index].item()) else "full")}
            for index in range(min(inputs.shape[0], max(1, int(self.gate_config.trace_limit))))
        ]
        self._record_route_trace(
            {
                "policy": "early_exit",
                "routing_family": self.routing_family,
                "mode": "eval",
                "threshold": round(float(threshold), 4),
                "target_accept_rate": (round(float(target_accept_rate), 4) if target_accept_rate is not None else None),
                "trace_samples": trace_samples,
            }
        )
        self._last_route_summary = {
            "policy": "early_exit",
            "routing_family": self.routing_family,
            "mode": "eval",
            "gate_mode": self.gate_config.mode,
            "gate_metric": self.gate_config.metric,
            "confidence_threshold": round(float(threshold), 4),
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "target_accept_rate": (round(float(target_accept_rate), 4) if target_accept_rate is not None else self.gate_config.target_accept_rate),
            "early_exit_fraction": round(float(early_count / total), 4),
            "eligible_fraction": round(float(eligible_mask.float().mean().item()), 4),
            "mean_gate_score": round(float(score.mean().item()), 4),
            "max_gate_score": round(float(score.max().item()), 4),
            "mean_exit_confidence": round(float(confidence_scores[use_early].mean().item()), 4) if early_count > 0 else 0.0,
            "full_path_fraction": round(float(full_count / total), 4),
            "trace_samples": trace_samples,
            "mean_width": 1.0,
            "mean_cost_ratio": round(float(mean_cost_ratio), 4),
        }
        return outputs

    def _dynamic_gate_score(self, pooled_features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        metric_scores = gate_scores(logits, metric=self.gate_config.metric)
        if self.gate_config.mode == "learned":
            learned_scores = torch.sigmoid(self._sliced_gate_logits(self.width_gate_head, pooled_features)).squeeze(1)
            base_scores = blend_gate_scores(learned_scores, metric_scores, metric_blend=self.gate_config.metric_blend)
        else:
            base_scores = metric_scores
        family_scores = self._family_gate_scores(pooled_features)
        if family_scores is None:
            return base_scores
        blend = 0.15 if self.routing_family == "slimmable_pyramid" else (0.32 if self.routing_family == "channel_gate_pyramid" else 0.42)
        return blend_gate_scores(base_scores, family_scores, metric_blend=blend)

    def _early_gate_score(self, pooled_features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        metric_scores = gate_scores(logits, metric=self.gate_config.metric)
        if self.gate_config.mode == "learned":
            learned_scores = torch.sigmoid(self._sliced_gate_logits(self.early_gate_head, pooled_features)).squeeze(1)
            base_scores = blend_gate_scores(learned_scores, metric_scores, metric_blend=self.gate_config.metric_blend)
        else:
            base_scores = metric_scores
        family_scores = self._family_gate_scores(pooled_features)
        if family_scores is None:
            return base_scores
        blend = 0.22 if self.routing_family == "early_exit_cascade" else (0.36 if self.routing_family == "skip_cascade" else 0.44)
        return blend_gate_scores(base_scores, family_scores, metric_blend=blend)

    def _sliced_gate_logits(self, head: nn.Linear, pooled_features: torch.Tensor) -> torch.Tensor:
        in_features = pooled_features.shape[1]
        return F.linear(pooled_features, head.weight[:, :in_features], head.bias)

    def _family_gate_scores(self, pooled_features: torch.Tensor) -> torch.Tensor | None:
        magnitude = pooled_features.abs()
        if self.routing_family == "channel_gate_pyramid":
            concentration = magnitude.max(dim=1).values / magnitude.mean(dim=1).clamp_min(1e-6)
            return torch.sigmoid(concentration - 1.3)
        if self.routing_family == "instance_sparse_pyramid":
            threshold = magnitude.mean(dim=1, keepdim=True)
            return (magnitude < threshold).float().mean(dim=1)
        if self.routing_family == "skip_cascade":
            energy = pooled_features.pow(2).mean(dim=1)
            normalized = (energy - energy.mean()) / energy.std().clamp_min(1e-6)
            return torch.sigmoid(normalized)
        if self.routing_family == "iterative_refine":
            stability = magnitude.mean(dim=1) / magnitude.max(dim=1).values.clamp_min(1e-6)
            return torch.sigmoid((stability - 0.45) * 6.0)
        return None

    def _gate_supervision_loss(
        self,
        gate_scores_tensor: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate_config.supervision_weight <= 0.0:
            return torch.zeros((), device=self.device)
        gate_targets = gate_accept_targets(
            student_logits,
            teacher_logits,
            targets=targets,
            teacher_confidence_floor=self.gate_config.teacher_confidence_floor,
            student_confidence_floor=self.gate_config.student_confidence_floor,
            strategy=self.gate_config.target_strategy,
        )
        return self.gate_loss_fn(gate_scores_tensor.clamp(1e-6, 1.0 - 1e-6), gate_targets)

    def _gate_regularization_losses(
        self,
        probabilities: torch.Tensor,
        *,
        target_accept_rate: float | None,
    ) -> list[torch.Tensor]:
        losses: list[torch.Tensor] = []
        if self.gate_config.entropy_weight > 0.0:
            losses.append(self.gate_config.entropy_weight * gate_entropy(probabilities))
        if target_accept_rate is not None and self.gate_config.accept_rate_weight > 0.0:
            losses.append(
                accept_rate_penalty(
                    probabilities,
                    target_accept_rate=target_accept_rate,
                    weight=self.gate_config.accept_rate_weight,
                )
            )
        return losses

    def _record_route_trace(self, trace: dict[str, Any]) -> None:
        self._route_trace_history.append(dict(trace))
        self._route_trace_history = self._route_trace_history[-4:]

    def _width_channels(self, multiplier: float) -> list[int]:
        return [self._scaled_channels(base_channels, multiplier) for base_channels in self.spec.conv_channels]

    def _scaled_channels(self, base_channels: int, multiplier: float) -> int:
        return max(4, min(base_channels, int(round(base_channels * multiplier))))

    def _apply_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.spec.activation == "tanh":
            return torch.tanh(tensor)
        if self.spec.activation == "gelu":
            return F.gelu(tensor)
        return F.relu(tensor)

    def _structure_metadata(self) -> dict[str, Any]:
        summary = ConstraintEvaluator().evaluate(architecture_spec=self.spec).to_dict()
        summary.update(
            {
                "architecture_family": "routed_cnn",
                "routing_policy": self.routing_policy,
                "routing_family": self.routing_family,
                "width_multipliers": [round(value, 4) for value in self.width_multipliers],
                "eval_width_multipliers": [round(value, 4) for value in self.eval_width_multipliers],
                "gate_config": self.gate_config.to_dict(),
                "device": str(self.device),
                "optimizer_name": self._optimizer_name,
                "lr_schedule": self._lr_schedule,
                "weight_decay": self._weight_decay,
                "label_smoothing": self._label_smoothing,
                "route_summary": dict(self._last_route_summary),
                "route_trace_tail": self.route_trace(),
            }
        )
        return summary

    def _full_path_cost_ratio(self, multiplier: float) -> float:
        scaled_channels = self._width_channels(multiplier)
        numerator = self._conv_cost(scaled_channels)
        denominator = self._conv_cost(list(self.spec.conv_channels))
        return float(numerator / max(1.0, denominator))

    def _early_exit_cost_ratio(self) -> float:
        numerator = self._conv_cost(list(self.spec.conv_channels[: self.exit_stage_index + 1]))
        denominator = self._conv_cost(list(self.spec.conv_channels))
        return float(numerator / max(1.0, denominator))

    def _conv_cost(self, channels: list[int]) -> float:
        cost = 0.0
        in_channels_value = float(self.spec.input_channels)
        for out_channels in channels:
            cost += in_channels_value * float(out_channels)
            in_channels_value = float(out_channels)
        return cost

    def _training_progress(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return 1.0
        return max(0.0, min(1.0, float(epoch) / float(total_epochs - 1)))

    def _normalize_widths(self, values: list[float]) -> list[float]:
        normalized = sorted({round(float(value), 4) for value in values}, reverse=True)
        if not normalized:
            raise ValueError("width_multipliers must contain at least one value")
        if normalized[0] > 1.0 or normalized[-1] <= 0.0:
            raise ValueError("width_multipliers must be in (0.0, 1.0]")
        if 1.0 not in normalized:
            normalized.insert(0, 1.0)
        return normalized

    def _resolve_exit_stage_index(self) -> int:
        if self.routing_family in {"skip_cascade", "iterative_refine"} and len(self.spec.conv_channels) > 1:
            return len(self.spec.conv_channels) - 2
        return 0

    def _resolve_gate_stage_index(self) -> int:
        if self.routing_policy == "early_exit":
            return self._resolve_exit_stage_index()
        if self.routing_family in {"channel_gate_pyramid", "instance_sparse_pyramid"} and len(self.spec.conv_channels) > 1:
            return len(self.spec.conv_channels) - 2
        return len(self.spec.conv_channels) - 1

    @staticmethod
    def _resolve_routing_family(routing_policy: str, routing_family: str | None) -> str:
        if routing_family is None:
            if routing_policy == "dynamic_width":
                return "slimmable_pyramid"
            if routing_policy == "early_exit":
                return "early_exit_cascade"
            return "slimmable_pyramid"
        resolved = str(routing_family)
        if resolved not in (_DYNAMIC_WIDTH_FAMILIES | _EARLY_EXIT_FAMILIES):
            raise ValueError(f"Unsupported routing_family '{resolved}'")
        if routing_policy == "largest":
            return resolved
        if resolved in _DYNAMIC_WIDTH_FAMILIES and routing_policy != "dynamic_width":
            raise ValueError(f"routing_family '{resolved}' requires routing_policy='dynamic_width'")
        if resolved in _EARLY_EXIT_FAMILIES and routing_policy != "early_exit":
            raise ValueError(f"routing_family '{resolved}' requires routing_policy='early_exit'")
        return resolved

    @staticmethod
    def _resolve_conv_channels(conv_channels: list[int] | None, routing_family: str) -> list[int]:
        values = [int(value) for value in (conv_channels or [24, 48])]
        if conv_channels is None:
            if routing_family == "channel_gate_pyramid":
                return [28, 72]
            if routing_family == "instance_sparse_pyramid":
                return [24, 64]
            if routing_family == "skip_cascade":
                return [32, 72]
            if routing_family == "iterative_refine":
                return [36, 80]
        return values




