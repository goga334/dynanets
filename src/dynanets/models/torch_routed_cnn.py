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
    budget_penalty,
    distillation_loss,
    gate_accept_targets,
    gate_entropy,
    gate_scores,
    relaxed_accept_probability,
)
from dynanets.runtime import resolve_device


class TorchRoutedCNNClassifier(nn.Module, NeuralModel):
    def __init__(
        self,
        input_channels: int,
        input_size: int | list[int] | tuple[int, int] = (28, 28),
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        activation: str = "relu",
        lr: float = 1e-3,
        width_multipliers: list[float] | None = None,
        eval_width_multipliers: list[float] | None = None,
        routing_policy: str = "dynamic_width",
        gate_mode: str = "metric",
        confidence_threshold: float = 0.85,
        min_confidence_threshold: float | None = None,
        gate_metric: str = "confidence",
        gate_temperature: float = 0.25,
        gate_budget_weight: float = 0.0,
        target_cost_ratio: float = 1.0,
        distillation_weight: float = 0.25,
        gate_supervision_weight: float = 0.0,
        teacher_confidence_floor: float = 0.55,
        gate_entropy_weight: float = 0.0,
        target_accept_rate: float | None = None,
        route_trace_limit: int = 8,
        early_exit_loss_weight: float = 0.3,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.spec = cnn_spec_from_params(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=conv_channels or [24, 48],
            classifier_hidden_dims=[],
            activation=activation,
            use_batch_norm=False,
            metadata={"routing_family": "routed_cnn"},
        )
        self.device = resolve_device(device)
        self._lr = float(lr)
        self.routing_policy = str(routing_policy)
        if self.routing_policy not in {"largest", "dynamic_width", "early_exit"}:
            raise ValueError("routing_policy must be one of: largest, dynamic_width, early_exit")
        self.gate_config = GateConfig(
            mode=str(gate_mode),
            metric=str(gate_metric),
            threshold=float(confidence_threshold),
            min_threshold=(float(min_confidence_threshold) if min_confidence_threshold is not None else None),
            temperature=float(gate_temperature),
            budget_weight=float(gate_budget_weight),
            target_cost_ratio=float(target_cost_ratio),
            distillation_weight=float(distillation_weight),
            supervision_weight=float(gate_supervision_weight),
            teacher_confidence_floor=float(teacher_confidence_floor),
            entropy_weight=float(gate_entropy_weight),
            target_accept_rate=(float(target_accept_rate) if target_accept_rate is not None else None),
            trace_limit=int(route_trace_limit),
        )
        if self.gate_config.mode not in {"metric", "learned"}:
            raise ValueError("gate_mode must be one of: metric, learned")
        self.confidence_threshold = float(confidence_threshold)
        self.min_confidence_threshold = float(min_confidence_threshold) if min_confidence_threshold is not None else None
        self.early_exit_loss_weight = float(early_exit_loss_weight)
        training_widths = width_multipliers or [1.0, 0.75, 0.5]
        self.width_multipliers = self._normalize_widths(training_widths)
        eval_widths = eval_width_multipliers or self.width_multipliers
        self.eval_width_multipliers = sorted(self._normalize_widths(eval_widths))

        max_c1, max_c2 = self.spec.conv_channels
        self.conv1 = nn.Conv2d(self.spec.input_channels, max_c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(max_c1, max_c2, kernel_size=3, padding=1)
        self.early_head = nn.Linear(max_c1, self.spec.num_classes)
        self.final_head = nn.Linear(max_c2, self.spec.num_classes)
        self.early_gate_head = nn.Linear(max_c1, 1)
        self.width_gate_head = nn.Linear(max_c2, 1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.gate_loss_fn = nn.BCELoss()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._last_route_summary: dict[str, Any] = {}
        self._route_trace_history: list[dict[str, Any]] = []
        self._latest_epoch: int = 0
        self._latest_total_epochs: int = 8

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

        self.optimizer.zero_grad()
        ordered_widths = sorted(self.width_multipliers, reverse=True)
        width_outputs: list[dict[str, Any]] = []
        max_logits: torch.Tensor | None = None
        for multiplier in ordered_widths:
            early_logits, final_logits, pooled1, pooled2 = self._forward_width_features(inputs, multiplier)
            width_outputs.append(
                {
                    "multiplier": multiplier,
                    "early_logits": early_logits,
                    "final_logits": final_logits,
                    "pooled1": pooled1,
                    "pooled2": pooled2,
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
            gate_probabilities: list[torch.Tensor] = []
            for index, item in enumerate(smaller):
                multiplier = float(item["multiplier"])
                final_logits = item["final_logits"]
                pooled2 = item["pooled2"]
                supervised = self.loss_fn(final_logits, targets)
                if multiplier < 1.0 and self.gate_config.distillation_weight > 0.0:
                    supervised = supervised + (self.gate_config.distillation_weight * distillation_loss(final_logits, teacher_logits))
                losses.append(supervised)

                if index == len(smaller) - 1:
                    accept_prob = unresolved_prob
                else:
                    score = self._dynamic_gate_score(pooled2, final_logits)
                    gate_prob = relaxed_accept_probability(
                        score,
                        threshold=threshold,
                        temperature=self.gate_config.temperature,
                    )
                    gate_probabilities.append(gate_prob)
                    accept_prob = unresolved_prob * gate_prob
                    gate_loss = self._gate_supervision_loss(score, final_logits, teacher_logits)
                    gate_loss_total = gate_loss_total + gate_loss
                    if self.gate_config.supervision_weight > 0.0:
                        losses.append(self.gate_config.supervision_weight * gate_loss)
                expected_cost = expected_cost + (accept_prob * self._full_path_cost_ratio(multiplier))
                expected_width = expected_width + (accept_prob * multiplier)
                unresolved_prob = unresolved_prob - accept_prob
                route_probs[str(multiplier)] = float(accept_prob.mean().detach().item())

            if gate_probabilities:
                combined_probs = torch.cat([item.reshape(-1) for item in gate_probabilities], dim=0)
                losses.extend(self._gate_regularization_losses(combined_probs))
                route_entropy_value = float(gate_entropy(combined_probs).detach().item())
                accept_rate_value = float(combined_probs.mean().detach().item())
            else:
                route_entropy_value = 0.0
                accept_rate_value = 1.0

            route_mean_cost = expected_cost.mean()
            route_mean_width = expected_width.mean()
            losses.append(
                budget_penalty(
                    expected_cost_ratio=route_mean_cost,
                    target_cost_ratio=self.gate_config.target_cost_ratio,
                    weight=self.gate_config.budget_weight,
                )
            )
            self._last_route_summary = {
                "policy": self.routing_policy,
                "mode": "train",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "threshold": round(float(threshold), 4),
                "target_cost_ratio": self.gate_config.target_cost_ratio,
                "target_accept_rate": self.gate_config.target_accept_rate,
                "trained_widths": [round(value, 4) for value in sorted(self.width_multipliers)],
                "route_probabilities": route_probs,
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
            pooled1 = item["pooled1"]
            early_loss = self.loss_fn(early_logits, targets)
            distill = distillation_loss(early_logits, teacher_logits) if self.gate_config.distillation_weight > 0.0 else torch.zeros((), device=self.device)
            losses.append((self.early_exit_loss_weight * early_loss) + (self.gate_config.distillation_weight * distill))
            score = self._early_gate_score(pooled1, early_logits)
            gate_loss_total = self._gate_supervision_loss(score, early_logits, teacher_logits)
            if self.gate_config.supervision_weight > 0.0:
                losses.append(self.gate_config.supervision_weight * gate_loss_total)
            early_prob = relaxed_accept_probability(score, threshold=threshold, temperature=self.gate_config.temperature)
            losses.extend(self._gate_regularization_losses(early_prob))
            route_mean_cost = ((early_prob * self._early_exit_cost_ratio()) + (1.0 - early_prob)).mean()
            route_mean_width = torch.ones((), device=self.device)
            losses.append(
                budget_penalty(
                    expected_cost_ratio=route_mean_cost,
                    target_cost_ratio=self.gate_config.target_cost_ratio,
                    weight=self.gate_config.budget_weight,
                )
            )
            self._last_route_summary = {
                "policy": self.routing_policy,
                "mode": "train",
                "gate_mode": self.gate_config.mode,
                "gate_metric": self.gate_config.metric,
                "threshold": round(float(threshold), 4),
                "target_cost_ratio": self.gate_config.target_cost_ratio,
                "target_accept_rate": self.gate_config.target_accept_rate,
                "expected_early_exit_fraction": round(float(early_prob.mean().detach().item()), 4),
                "route_entropy": round(float(gate_entropy(early_prob).detach().item()), 6),
                "mean_width": 1.0,
                "mean_cost_ratio": round(float(route_mean_cost.detach().item()), 4),
                "gate_loss": round(float(gate_loss_total.detach().item()), 6),
            }
        else:
            self._last_route_summary = {
                "policy": self.routing_policy,
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
            "width_multipliers": list(self.width_multipliers),
            "eval_width_multipliers": list(self.eval_width_multipliers),
            "routing_policy": self.routing_policy,
            "gate_mode": self.gate_config.mode,
            "confidence_threshold": self.confidence_threshold,
            "min_confidence_threshold": self.min_confidence_threshold,
            "gate_metric": self.gate_config.metric,
            "gate_temperature": self.gate_config.temperature,
            "gate_budget_weight": self.gate_config.budget_weight,
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "distillation_weight": self.gate_config.distillation_weight,
            "gate_supervision_weight": self.gate_config.supervision_weight,
            "teacher_confidence_floor": self.gate_config.teacher_confidence_floor,
            "gate_entropy_weight": self.gate_config.entropy_weight,
            "target_accept_rate": self.gate_config.target_accept_rate,
            "route_trace_limit": self.gate_config.trace_limit,
            "early_exit_loss_weight": self.early_exit_loss_weight,
            "device": str(self.device),
        }

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
        c1, c2 = self._width_channels(multiplier)
        x1 = F.conv2d(inputs, self.conv1.weight[:c1], self.conv1.bias[:c1], padding=1)
        x1 = self._apply_activation(x1)
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        pooled1 = F.adaptive_avg_pool2d(x1, (1, 1)).flatten(1)
        early_logits = F.linear(pooled1, self.early_head.weight[:, :c1], self.early_head.bias)

        x2 = F.conv2d(x1, self.conv2.weight[:c2, :c1], self.conv2.bias[:c2], padding=1)
        x2 = self._apply_activation(x2)
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        pooled2 = F.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)
        final_logits = F.linear(pooled2, self.final_head.weight[:, :c2], self.final_head.bias)
        return early_logits, final_logits, pooled1, pooled2

    def _evaluate_dynamic_width(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        outputs = torch.empty(batch_size, self.spec.num_classes, device=self.device)
        unresolved = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        selected_widths = torch.full((batch_size,), float(self.eval_width_multipliers[-1]), device=self.device)
        counts: dict[str, int] = {str(mult): 0 for mult in self.eval_width_multipliers}
        threshold = self.gate_config.threshold_for_progress(self._training_progress(self._latest_epoch, self._latest_total_epochs))

        for index, multiplier in enumerate(self.eval_width_multipliers):
            _, logits, _, pooled2 = self._forward_width_features(inputs, multiplier)
            score = self._dynamic_gate_score(pooled2, logits)
            if index == len(self.eval_width_multipliers) - 1:
                selected = unresolved.clone()
            else:
                selected = unresolved & (score >= threshold)
            outputs[selected] = logits[selected]
            selected_widths[selected] = float(multiplier)
            counts[str(multiplier)] += int(selected.sum().item())
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
                "mode": "eval",
                "threshold": round(float(threshold), 4),
                "trace_samples": trace_samples,
            }
        )
        self._last_route_summary = {
            "policy": "dynamic_width",
            "mode": "eval",
            "gate_mode": self.gate_config.mode,
            "gate_metric": self.gate_config.metric,
            "confidence_threshold": round(float(threshold), 4),
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "target_accept_rate": self.gate_config.target_accept_rate,
            "route_counts": counts,
            "trace_samples": trace_samples,
            "mean_width": round(float(mean_width), 4),
            "mean_cost_ratio": round(float(mean_cost_ratio), 4),
        }
        return outputs

    def _evaluate_early_exit(self, inputs: torch.Tensor) -> torch.Tensor:
        early_logits, final_logits, pooled1, _ = self._forward_width_features(inputs, 1.0)
        threshold = self.gate_config.threshold_for_progress(self._training_progress(self._latest_epoch, self._latest_total_epochs))
        score = self._early_gate_score(pooled1, early_logits)
        use_early = score >= threshold
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
                "mode": "eval",
                "threshold": round(float(threshold), 4),
                "trace_samples": trace_samples,
            }
        )
        self._last_route_summary = {
            "policy": "early_exit",
            "mode": "eval",
            "gate_mode": self.gate_config.mode,
            "gate_metric": self.gate_config.metric,
            "confidence_threshold": round(float(threshold), 4),
            "target_cost_ratio": self.gate_config.target_cost_ratio,
            "target_accept_rate": self.gate_config.target_accept_rate,
            "early_exit_fraction": round(float(early_count / total), 4),
            "full_path_fraction": round(float(full_count / total), 4),
            "trace_samples": trace_samples,
            "mean_width": 1.0,
            "mean_cost_ratio": round(float(mean_cost_ratio), 4),
        }
        return outputs

    def _dynamic_gate_score(self, pooled_features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_config.mode == "learned":
            return torch.sigmoid(self._sliced_gate_logits(self.width_gate_head, pooled_features)).squeeze(1)
        return gate_scores(logits, metric=self.gate_config.metric)

    def _early_gate_score(self, pooled_features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_config.mode == "learned":
            return torch.sigmoid(self._sliced_gate_logits(self.early_gate_head, pooled_features)).squeeze(1)
        return gate_scores(logits, metric=self.gate_config.metric)

    def _sliced_gate_logits(self, head: nn.Linear, pooled_features: torch.Tensor) -> torch.Tensor:
        in_features = pooled_features.shape[1]
        return F.linear(pooled_features, head.weight[:, :in_features], head.bias)

    def _gate_supervision_loss(
        self,
        gate_scores_tensor: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate_config.supervision_weight <= 0.0:
            return torch.zeros((), device=self.device)
        targets = gate_accept_targets(
            student_logits,
            teacher_logits,
            teacher_confidence_floor=self.gate_config.teacher_confidence_floor,
        )
        return self.gate_loss_fn(gate_scores_tensor.clamp(1e-6, 1.0 - 1e-6), targets)

    def _gate_regularization_losses(self, probabilities: torch.Tensor) -> list[torch.Tensor]:
        losses: list[torch.Tensor] = []
        if self.gate_config.entropy_weight > 0.0:
            losses.append(self.gate_config.entropy_weight * gate_entropy(probabilities))
        if self.gate_config.target_accept_rate is not None and self.gate_config.budget_weight > 0.0:
            losses.append(
                accept_rate_penalty(
                    probabilities,
                    target_accept_rate=self.gate_config.target_accept_rate,
                    weight=self.gate_config.budget_weight,
                )
            )
        return losses

    def _record_route_trace(self, trace: dict[str, Any]) -> None:
        self._route_trace_history.append(dict(trace))
        self._route_trace_history = self._route_trace_history[-4:]

    def _width_channels(self, multiplier: float) -> tuple[int, int]:
        base1, base2 = self.spec.conv_channels
        c1 = self._scaled_channels(base1, multiplier)
        c2 = self._scaled_channels(base2, multiplier)
        return c1, c2

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
                "width_multipliers": [round(value, 4) for value in self.width_multipliers],
                "eval_width_multipliers": [round(value, 4) for value in self.eval_width_multipliers],
                "gate_config": self.gate_config.to_dict(),
                "device": str(self.device),
                "route_summary": dict(self._last_route_summary),
                "route_trace_tail": self.route_trace(),
            }
        )
        return summary

    def _full_path_cost_ratio(self, multiplier: float) -> float:
        c1, c2 = self._width_channels(multiplier)
        max_c1, max_c2 = self.spec.conv_channels
        numerator = (self.spec.input_channels * c1) + (c1 * c2)
        denominator = (self.spec.input_channels * max_c1) + (max_c1 * max_c2)
        return float(numerator / max(1, denominator))

    def _early_exit_cost_ratio(self) -> float:
        max_c1, max_c2 = self.spec.conv_channels
        numerator = self.spec.input_channels * max_c1
        denominator = (self.spec.input_channels * max_c1) + (max_c1 * max_c2)
        return float(numerator / max(1, denominator))

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



