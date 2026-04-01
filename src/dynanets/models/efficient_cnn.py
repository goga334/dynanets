from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynanets.architecture.efficient import EfficientBlockSpec, EfficientCNNArchitectureSpec
from dynanets.models.base import NeuralModel
from dynanets.runtime import resolve_device


class FireModule(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int) -> None:
        super().__init__()
        half = max(1, expand_channels // 2)
        remainder = max(1, expand_channels - half)
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.expand1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, half, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.expand3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, remainder, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = self.squeeze(inputs)
        return torch.cat([self.expand1(squeezed), self.expand3(squeezed)], dim=1)


class GroupedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        effective_groups = max(1, min(groups, in_channels, out_channels))
        while in_channels % effective_groups != 0 or out_channels % effective_groups != 0:
            effective_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=effective_groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=effective_groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class _BaseEfficientCNNClassifier(NeuralModel):
    def __init__(self, *, lr: float, device: str | None = None) -> None:
        self.device = resolve_device(device)
        self._lr = float(lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        self._structure_metadata = self._build_structure_metadata()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.train()
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"loss": float(loss.item()), "accuracy": float(accuracy)}

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(batch["inputs"].to(self.device))

    def structure_state(self) -> dict[str, Any]:
        return {"metadata": dict(self._structure_metadata)}

    def architecture_spec(self) -> EfficientCNNArchitectureSpec:
        return self.spec

    def init_params(self) -> dict[str, Any]:
        return dict(self._init_params)

    def _parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _build_structure_metadata(self) -> dict[str, Any]:
        parameter_count = self._parameter_count()
        forward_flop_proxy, activation_elements = self._estimate_costs()
        return {
            "architecture_family": self.spec.family,
            "parameter_count": parameter_count,
            "nonzero_parameter_count": parameter_count,
            "masked_weight_count": 0,
            "weight_sparsity": 0.0,
            "forward_flop_proxy": forward_flop_proxy,
            "activation_elements": activation_elements,
            "device": str(self.device),
        }

    def _estimate_costs(self) -> tuple[int, int]:
        raise NotImplementedError


class TorchSqueezeStyleCNNClassifier(_BaseEfficientCNNClassifier, nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int | list[int] | tuple[int, int] = (32, 32),
        num_classes: int = 10,
        stem_channels: int = 32,
        fire_channels: list[int] | None = None,
        squeeze_ratio: float = 0.5,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        size = input_size if not isinstance(input_size, int) else (input_size, input_size)
        fire_channels = fire_channels or [48, 64, 96]
        squeeze_channels = [max(8, int(round(channel * squeeze_ratio))) for channel in fire_channels]
        self.spec = EfficientCNNArchitectureSpec(
            family="squeezenet_style",
            input_channels=int(input_channels),
            input_size=(int(size[0]), int(size[1])),
            num_classes=int(num_classes),
            blocks=[
                EfficientBlockSpec(kind="conv", out_channels=int(stem_channels), kernel_size=3, pool="max"),
                *[
                    EfficientBlockSpec(kind="fire", out_channels=int(out), squeeze_channels=int(sq), pool=("max" if idx == 1 else None))
                    for idx, (out, sq) in enumerate(zip(fire_channels, squeeze_channels), start=1)
                ],
            ],
            metadata={"paper_family": "squeezenet"},
        )
        self.stem = nn.Sequential(
            nn.Conv2d(self.spec.input_channels, int(stem_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        current_channels = int(stem_channels)
        fire_modules: list[nn.Module] = []
        for index, (out_channels, squeeze) in enumerate(zip(fire_channels, squeeze_channels), start=1):
            fire_modules.append(FireModule(current_channels, squeeze, out_channels))
            current_channels = int(out_channels)
            if index == 2:
                fire_modules.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*fire_modules)
        self.classifier = nn.Conv2d(current_channels, self.spec.num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_params = {
            "input_channels": int(input_channels),
            "input_size": list(self.spec.input_size),
            "num_classes": int(num_classes),
            "stem_channels": int(stem_channels),
            "fire_channels": [int(value) for value in fire_channels],
            "squeeze_ratio": float(squeeze_ratio),
            "lr": float(lr),
            "device": str(resolve_device(device)),
        }
        super().__init__(lr=lr, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.features(x)
        x = self.classifier(x)
        x = self.pool(x).flatten(1)
        return x

    def _estimate_costs(self) -> tuple[int, int]:
        height, width = self.spec.input_size
        activation_elements = 0
        flop_proxy = 0
        in_channels = self.spec.input_channels
        current_h, current_w = height, width
        for block in self.spec.blocks:
            if block.kind == "conv":
                weights = block.out_channels * in_channels * block.kernel_size * block.kernel_size
                flop_proxy += 2 * current_h * current_w * weights
                activation_elements += current_h * current_w * block.out_channels
                in_channels = block.out_channels
            elif block.kind == "fire":
                squeeze = int(block.squeeze_channels or max(8, block.out_channels // 2))
                flop_proxy += 2 * current_h * current_w * in_channels * squeeze
                flop_proxy += 2 * current_h * current_w * squeeze * block.out_channels
                activation_elements += current_h * current_w * (squeeze + block.out_channels)
                in_channels = block.out_channels
            if block.pool is not None:
                current_h = max(1, current_h // 2)
                current_w = max(1, current_w // 2)
        flop_proxy += 2 * current_h * current_w * in_channels * self.spec.num_classes
        activation_elements += self.spec.num_classes
        return int(flop_proxy), int(activation_elements)


class TorchCondenseStyleCNNClassifier(_BaseEfficientCNNClassifier, nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int | list[int] | tuple[int, int] = (32, 32),
        num_classes: int = 10,
        stem_channels: int = 32,
        grouped_channels: list[int] | None = None,
        groups: list[int] | None = None,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        size = input_size if not isinstance(input_size, int) else (input_size, input_size)
        grouped_channels = grouped_channels or [48, 64, 96]
        groups = groups or [4, 4, 8]
        self.spec = EfficientCNNArchitectureSpec(
            family="condensenet_style",
            input_channels=int(input_channels),
            input_size=(int(size[0]), int(size[1])),
            num_classes=int(num_classes),
            blocks=[
                EfficientBlockSpec(kind="conv", out_channels=int(stem_channels), kernel_size=3, pool="max"),
                *[
                    EfficientBlockSpec(kind="grouped", out_channels=int(out), groups=int(group), pool=("max" if idx == 2 else None))
                    for idx, (out, group) in enumerate(zip(grouped_channels, groups), start=1)
                ],
            ],
            metadata={"paper_family": "condensenet"},
        )
        self.stem = nn.Sequential(
            nn.Conv2d(self.spec.input_channels, int(stem_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(stem_channels)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        current_channels = int(stem_channels)
        blocks: list[nn.Module] = []
        for index, (out_channels, group) in enumerate(zip(grouped_channels, groups), start=1):
            blocks.append(GroupedConvBlock(current_channels, int(out_channels), int(group)))
            current_channels = int(out_channels)
            if index == 2:
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, self.spec.num_classes)
        self._init_params = {
            "input_channels": int(input_channels),
            "input_size": list(self.spec.input_size),
            "num_classes": int(num_classes),
            "stem_channels": int(stem_channels),
            "grouped_channels": [int(value) for value in grouped_channels],
            "groups": [int(value) for value in groups],
            "lr": float(lr),
            "device": str(resolve_device(device)),
        }
        super().__init__(lr=lr, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def _estimate_costs(self) -> tuple[int, int]:
        height, width = self.spec.input_size
        activation_elements = 0
        flop_proxy = 0
        in_channels = self.spec.input_channels
        current_h, current_w = height, width
        for block in self.spec.blocks:
            if block.kind == "conv":
                weights = block.out_channels * in_channels * block.kernel_size * block.kernel_size
                flop_proxy += 2 * current_h * current_w * weights
                activation_elements += current_h * current_w * block.out_channels
                in_channels = block.out_channels
            elif block.kind == "grouped":
                effective_groups = max(1, min(block.groups, in_channels, block.out_channels))
                while in_channels % effective_groups != 0 or block.out_channels % effective_groups != 0:
                    effective_groups -= 1
                weights_1x1 = (in_channels * block.out_channels) // effective_groups
                weights_3x3 = (block.out_channels * block.out_channels * 9) // effective_groups
                flop_proxy += 2 * current_h * current_w * (weights_1x1 + weights_3x3)
                activation_elements += current_h * current_w * block.out_channels
                in_channels = block.out_channels
            if block.pool is not None:
                current_h = max(1, current_h // 2)
                current_w = max(1, current_w // 2)
        flop_proxy += 2 * in_channels * self.spec.num_classes
        activation_elements += self.spec.num_classes
        return int(flop_proxy), int(activation_elements)
