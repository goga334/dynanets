from __future__ import annotations

from dataclasses import dataclass

import torch

from dynanets.datasets.base import DataSplit, DatasetBundle, DatasetFactory


@dataclass(slots=True)
class GaussianBlobsDatasetFactory(DatasetFactory):
    train_size: int = 512
    validation_size: int = 128
    test_size: int = 128
    input_dim: int = 2
    num_classes: int = 2
    cluster_std: float = 0.8
    seed: int = 42

    def build(self) -> DatasetBundle:
        generator = torch.Generator().manual_seed(self.seed)
        centers = self._make_centers()

        train = self._make_split(self.train_size, centers, generator)
        validation = self._make_split(self.validation_size, centers, generator)
        test = self._make_split(self.test_size, centers, generator)

        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            metadata={
                "task": "classification",
                "dataset_name": "gaussian_blobs",
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
            },
        )

    def _make_centers(self) -> torch.Tensor:
        if self.num_classes == 2 and self.input_dim == 2:
            return torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float32)

        base = torch.linspace(-3.0, 3.0, steps=self.num_classes, dtype=torch.float32)
        centers = torch.zeros((self.num_classes, self.input_dim), dtype=torch.float32)
        centers[:, 0] = base
        if self.input_dim > 1:
            centers[:, 1] = torch.flip(base, dims=[0])
        return centers

    def _make_split(
        self,
        size: int,
        centers: torch.Tensor,
        generator: torch.Generator,
    ) -> DataSplit:
        labels = torch.randint(0, self.num_classes, (size,), generator=generator)
        noise = torch.randn(size, self.input_dim, generator=generator) * self.cluster_std
        inputs = centers[labels] + noise
        return DataSplit(inputs=inputs.float(), targets=labels.long())


@dataclass(slots=True)
class TwoSpiralsDatasetFactory(DatasetFactory):
    train_size: int = 1024
    validation_size: int = 256
    test_size: int = 256
    input_dim: int = 2
    turns: float = 2.0
    noise: float = 0.12
    feature_noise: float = 0.08
    seed: int = 42

    def build(self) -> DatasetBundle:
        generator = torch.Generator().manual_seed(self.seed)
        projection = self._make_projection(generator)
        train = self._make_split(self.train_size, generator, projection)
        validation = self._make_split(self.validation_size, generator, projection)
        test = self._make_split(self.test_size, generator, projection)
        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            metadata={
                "task": "classification",
                "dataset_name": "two_spirals",
                "input_dim": self.input_dim,
                "num_classes": 2,
            },
        )

    def _make_projection(self, generator: torch.Generator) -> torch.Tensor:
        base_dim = 9
        if self.input_dim == base_dim:
            return torch.eye(base_dim)
        weights = torch.randn(base_dim, self.input_dim, generator=generator)
        return weights / (base_dim**0.5)

    def _make_split(self, size: int, generator: torch.Generator, projection: torch.Tensor) -> DataSplit:
        labels = torch.arange(size, dtype=torch.long) % 2
        t = torch.rand(size, generator=generator)
        radius = 0.5 + 3.5 * t
        angle = self.turns * 2.0 * torch.pi * t + labels.float() * torch.pi
        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        core = torch.stack([x, y], dim=1)
        core = core + torch.randn(size, 2, generator=generator) * self.noise
        feature_bank = self._feature_bank(core)
        if self.input_dim == feature_bank.shape[1]:
            inputs = feature_bank
        else:
            inputs = feature_bank @ projection
            inputs = inputs + torch.randn(size, self.input_dim, generator=generator) * self.feature_noise
        permutation = torch.randperm(size, generator=generator)
        return DataSplit(inputs=inputs[permutation].float(), targets=labels[permutation].long())

    def _feature_bank(self, core: torch.Tensor) -> torch.Tensor:
        x = core[:, 0]
        y = core[:, 1]
        return torch.stack(
            [
                x,
                y,
                x * y,
                x.square(),
                y.square(),
                torch.sin(x),
                torch.sin(y),
                torch.cos(x),
                torch.cos(y),
            ],
            dim=1,
        )


@dataclass(slots=True)
class ConcentricCirclesDatasetFactory(DatasetFactory):
    train_size: int = 1024
    validation_size: int = 256
    test_size: int = 256
    inner_radius: float = 1.0
    outer_radius: float = 2.5
    radial_noise: float = 0.15
    seed: int = 42

    def build(self) -> DatasetBundle:
        generator = torch.Generator().manual_seed(self.seed)
        train = self._make_split(self.train_size, generator)
        validation = self._make_split(self.validation_size, generator)
        test = self._make_split(self.test_size, generator)
        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            metadata={
                "task": "classification",
                "dataset_name": "concentric_circles",
                "input_dim": 2,
                "num_classes": 2,
            },
        )

    def _make_split(self, size: int, generator: torch.Generator) -> DataSplit:
        labels = torch.arange(size, dtype=torch.long) % 2
        theta = torch.rand(size, generator=generator) * 2.0 * torch.pi
        base_radius = torch.where(labels == 0, torch.full((size,), self.inner_radius), torch.full((size,), self.outer_radius))
        radius = base_radius + torch.randn(size, generator=generator) * self.radial_noise
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        inputs = torch.stack([x, y], dim=1)
        permutation = torch.randperm(size, generator=generator)
        return DataSplit(inputs=inputs[permutation].float(), targets=labels[permutation].long())
