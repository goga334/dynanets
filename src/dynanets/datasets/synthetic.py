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
