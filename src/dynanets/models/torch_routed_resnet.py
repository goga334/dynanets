from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dynanets.architecture import CNNArchitectureSpec, cnn_spec_from_params
from dynanets.constraints import ConstraintEvaluator
from dynanets.models.torch_routed_cnn import TorchRoutedCNNClassifier


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None


class TorchRoutedResNetClassifier(TorchRoutedCNNClassifier):
    def __init__(
        self,
        input_channels: int,
        input_size: int | list[int] | tuple[int, int] = (32, 32),
        num_classes: int = 10,
        stage_channels: list[int] | None = None,
        blocks_per_stage: list[int] | None = None,
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
        resolved_stage_channels = [int(value) for value in (stage_channels or [32, 64, 96])]
        resolved_blocks = [int(value) for value in (blocks_per_stage or [1, 1, 1])]
        if len(resolved_blocks) != len(resolved_stage_channels):
            if len(resolved_blocks) == 1:
                resolved_blocks = resolved_blocks * len(resolved_stage_channels)
            else:
                raise ValueError("blocks_per_stage must match stage_channels length")
        self.blocks_per_stage = resolved_blocks
        super().__init__(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=resolved_stage_channels,
            activation=activation,
            lr=lr,
            optimizer_name=optimizer_name,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            min_lr_ratio=min_lr_ratio,
            label_smoothing=label_smoothing,
            width_multipliers=width_multipliers,
            eval_width_multipliers=eval_width_multipliers,
            routing_policy=routing_policy,
            routing_family=routing_family,
            gate_mode=gate_mode,
            confidence_threshold=confidence_threshold,
            min_confidence_threshold=min_confidence_threshold,
            gate_metric=gate_metric,
            gate_temperature=gate_temperature,
            eval_threshold_offset=eval_threshold_offset,
            eval_min_threshold=eval_min_threshold,
            gate_metric_blend=gate_metric_blend,
            gate_budget_weight=gate_budget_weight,
            target_cost_ratio=target_cost_ratio,
            distillation_weight=distillation_weight,
            gate_supervision_weight=gate_supervision_weight,
            teacher_confidence_floor=teacher_confidence_floor,
            gate_entropy_weight=gate_entropy_weight,
            target_accept_rate=target_accept_rate,
            gate_accept_rate_weight=gate_accept_rate_weight,
            gate_target_strategy=gate_target_strategy,
            student_confidence_floor=student_confidence_floor,
            min_target_cost_ratio=min_target_cost_ratio,
            min_target_accept_rate=min_target_accept_rate,
            route_trace_limit=route_trace_limit,
            early_exit_loss_weight=early_exit_loss_weight,
            early_exit_focus_weight=early_exit_focus_weight,
            early_exit_focus_floor=early_exit_focus_floor,
            dynamic_width_focus_weight=dynamic_width_focus_weight,
            dynamic_width_focus_floor=dynamic_width_focus_floor,
            device=device,
        )
        self.spec = cnn_spec_from_params(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=resolved_stage_channels,
            classifier_hidden_dims=[],
            activation=activation,
            use_batch_norm=False,
            metadata={
                "routing_family": self.routing_family,
                "routing_policy": self.routing_policy,
                "backbone_family": "routed_resnet",
                "blocks_per_stage": list(self.blocks_per_stage),
            },
        )
        self.block_channels = list(self.spec.conv_channels)
        self.exit_stage_index = self._resolve_exit_stage_index()
        self.gate_stage_index = self._resolve_gate_stage_index()

        self.stem = nn.Conv2d(self.spec.input_channels, self.block_channels[0], kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        in_channels_value = self.block_channels[0]
        for stage_channels_value, block_count in zip(self.block_channels, self.blocks_per_stage):
            stage = nn.ModuleList()
            for block_index in range(block_count):
                unit_in = in_channels_value if block_index == 0 else stage_channels_value
                stage.append(_ResidualUnit(unit_in, stage_channels_value))
                in_channels_value = stage_channels_value
            self.stages.append(stage)

        exit_channels = self.block_channels[self.exit_stage_index]
        gate_channels = self.block_channels[self.gate_stage_index]
        final_channels = self.block_channels[-1]
        self.early_head = nn.Linear(exit_channels, self.spec.num_classes)
        self.final_head = nn.Linear(final_channels, self.spec.num_classes)
        self.early_gate_head = nn.Linear(exit_channels, 1)
        self.width_gate_head = nn.Linear(gate_channels, 1)
        self._initialize_gate_heads()
        self.to(self.device)
        self.optimizer = self._build_optimizer()

    def init_params(self) -> dict[str, Any]:
        params = super().init_params()
        params.pop("conv_channels", None)
        params["stage_channels"] = list(self.spec.conv_channels)
        params["blocks_per_stage"] = list(self.blocks_per_stage)
        return params

    def _forward_width_features(
        self,
        inputs: torch.Tensor,
        multiplier: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_channels = self._width_channels(multiplier)
        current = F.conv2d(
            inputs,
            self.stem.weight[: stage_channels[0], : self.spec.input_channels],
            self.stem.bias[: stage_channels[0]],
            padding=1,
        )
        current = self._apply_activation(current)
        current = F.max_pool2d(current, kernel_size=2, stride=2)

        pooled_features: list[torch.Tensor] = []
        for stage_index, (stage, out_channels) in enumerate(zip(self.stages, stage_channels)):
            current = self._forward_stage(stage, current, out_channels)
            pooled_features.append(F.adaptive_avg_pool2d(current, (1, 1)).flatten(1))
            if stage_index < len(self.stages) - 1:
                current = F.max_pool2d(current, kernel_size=2, stride=2)

        early_pooled = pooled_features[self.exit_stage_index]
        gate_pooled = pooled_features[self.gate_stage_index]
        final_pooled = pooled_features[-1]
        early_logits = F.linear(
            early_pooled,
            self.early_head.weight[:, : stage_channels[self.exit_stage_index]],
            self.early_head.bias,
        )
        final_logits = F.linear(
            final_pooled,
            self.final_head.weight[:, : stage_channels[-1]],
            self.final_head.bias,
        )
        return early_logits, final_logits, early_pooled, gate_pooled

    def _forward_stage(self, stage: nn.ModuleList, inputs: torch.Tensor, out_channels: int) -> torch.Tensor:
        current = inputs
        in_channels_value = inputs.shape[1]
        for unit in stage:
            residual = current
            if unit.proj is not None or in_channels_value != out_channels:
                residual = F.conv2d(
                    current,
                    unit.proj.weight[:out_channels, :in_channels_value],
                    unit.proj.bias[:out_channels] if unit.proj.bias is not None else None,
                )
            else:
                residual = residual[:, :out_channels]
            out = F.conv2d(
                current,
                unit.conv1.weight[:out_channels, :in_channels_value],
                unit.conv1.bias[:out_channels],
                padding=1,
            )
            out = self._apply_activation(out)
            out = F.conv2d(
                out,
                unit.conv2.weight[:out_channels, :out_channels],
                unit.conv2.bias[:out_channels],
                padding=1,
            )
            current = self._apply_activation(out + residual)
            in_channels_value = out_channels
        return current

    def _conv_cost(self, channels: list[int]) -> float:
        cost = float(self.spec.input_channels * channels[0])
        previous = channels[0]
        for stage_index, out_channels in enumerate(channels):
            blocks = self.blocks_per_stage[stage_index]
            for block_index in range(blocks):
                if block_index == 0:
                    cost += float(previous * out_channels)
                else:
                    cost += float(out_channels * out_channels)
                cost += float(out_channels * out_channels)
                if block_index == 0 and previous != out_channels:
                    cost += float(previous * out_channels)
                previous = out_channels
        return cost

    def _structure_metadata(self) -> dict[str, Any]:
        summary = ConstraintEvaluator().evaluate(architecture_spec=self.spec).to_dict()
        summary.update(
            {
                "architecture_family": "routed_resnet",
                "routing_policy": self.routing_policy,
                "routing_family": self.routing_family,
                "blocks_per_stage": list(self.blocks_per_stage),
                "width_multipliers": [round(value, 4) for value in self.width_multipliers],
                "eval_width_multipliers": [round(value, 4) for value in self.eval_width_multipliers],
                "gate_config": self.gate_config.to_dict(),
                "device": str(self.device),
                "route_summary": dict(self._last_route_summary),
                "route_trace_tail": self.route_trace(),
            }
        )
        return summary
