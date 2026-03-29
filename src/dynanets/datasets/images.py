from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from dynanets.datasets.base import DataSplit, DatasetBundle, DatasetFactory


@dataclass(slots=True)
class SyntheticImagePatternsDatasetFactory(DatasetFactory):
    train_size: int = 2048
    validation_size: int = 512
    test_size: int = 512
    image_size: int = 28
    num_classes: int = 10
    noise: float = 0.12
    seed: int = 42

    def build(self) -> DatasetBundle:
        generator = torch.Generator().manual_seed(self.seed)
        templates = self._build_templates()
        train = self._make_split(self.train_size, templates, generator)
        validation = self._make_split(self.validation_size, templates, generator)
        test = self._make_split(self.test_size, templates, generator)
        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            metadata={
                "task": "classification",
                "dataset_name": "synthetic_image_patterns",
                "input_shape": [1, self.image_size, self.image_size],
                "num_classes": self.num_classes,
            },
        )

    def _make_split(self, size: int, templates: torch.Tensor, generator: torch.Generator) -> DataSplit:
        labels = torch.randint(0, self.num_classes, (size,), generator=generator)
        inputs = templates[labels].clone()
        noise = torch.randn(inputs.shape, generator=generator) * self.noise
        inputs = (inputs + noise).clamp(0.0, 1.0)
        permutation = torch.randperm(size, generator=generator)
        return DataSplit(inputs=inputs[permutation].float(), targets=labels[permutation].long())

    def _build_templates(self) -> torch.Tensor:
        size = self.image_size
        margin = max(2, size // 6)
        thickness = max(2, size // 10)
        center = size // 2
        templates = torch.zeros(self.num_classes, 1, size, size, dtype=torch.float32)

        templates[0, 0, center - thickness : center + thickness, center - thickness : center + thickness] = 1.0
        templates[1, 0, margin : margin + thickness, margin : size - margin] = 1.0
        templates[2, 0, size - margin - thickness : size - margin, margin : size - margin] = 1.0
        templates[3, 0, margin : size - margin, margin : margin + thickness] = 1.0
        templates[4, 0, margin : size - margin, size - margin - thickness : size - margin] = 1.0
        templates[5, 0, center - thickness : center + thickness, margin : size - margin] = 1.0
        templates[5, 0, margin : size - margin, center - thickness : center + thickness] = 1.0
        templates[6, 0, margin : size - margin, margin : margin + thickness] = 1.0
        templates[6, 0, size - margin - thickness : size - margin, margin : size - margin] = 1.0
        templates[7, 0, margin : margin + thickness, margin : size - margin] = 1.0
        templates[7, 0, size - margin - thickness : size - margin, margin : size - margin] = 1.0
        templates[7, 0, margin : size - margin, margin : margin + thickness] = 1.0
        templates[7, 0, margin : size - margin, size - margin - thickness : size - margin] = 1.0
        templates[8, 0, center - thickness : center + thickness, margin : size - margin] = 1.0
        templates[8, 0, margin : size - margin, center - thickness : center + thickness] = 1.0
        for offset in range(-thickness, thickness + 1):
            row = torch.arange(size)
            col_main = torch.clamp(row + offset, 0, size - 1)
            col_anti = torch.clamp((size - 1 - row) + offset, 0, size - 1)
            templates[9, 0, row, col_main] = 1.0
            templates[9, 0, row, col_anti] = 1.0
        return templates


@dataclass(slots=True)
class MNISTDatasetFactory(DatasetFactory):
    data_dir: str = "data"
    validation_size: int = 5000
    train_size: int | None = None
    test_size: int | None = None
    download: bool = True
    seed: int = 42

    def build(self) -> DatasetBundle:
        torchvision = self._load_torchvision()
        datasets = torchvision.datasets
        transforms = torchvision.transforms
        transform = transforms.ToTensor()

        train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=self.download, transform=transform)
        test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=self.download, transform=transform)

        generator = torch.Generator().manual_seed(self.seed)
        train_indices = torch.randperm(len(train_dataset), generator=generator)
        validation_indices = train_indices[: self.validation_size]
        fit_indices = train_indices[self.validation_size :]
        if self.train_size is not None:
            fit_indices = fit_indices[: self.train_size]

        test_indices = torch.arange(len(test_dataset))
        if self.test_size is not None:
            test_indices = test_indices[: self.test_size]

        return DatasetBundle(
            train=self._dataset_to_split(train_dataset, fit_indices.tolist()),
            validation=self._dataset_to_split(train_dataset, validation_indices.tolist()),
            test=self._dataset_to_split(test_dataset, test_indices.tolist()),
            metadata={
                "task": "classification",
                "dataset_name": "mnist",
                "input_shape": [1, 28, 28],
                "num_classes": 10,
            },
        )

    def _load_torchvision(self):
        try:
            return importlib.import_module("torchvision")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MNISTDatasetFactory requires torchvision. Install the optional vision dependency first."
            ) from exc

    def _dataset_to_split(self, dataset, indices: list[int]) -> DataSplit:
        images = []
        labels = []
        for index in indices:
            image, label = dataset[index]
            images.append(image)
            labels.append(label)
        return DataSplit(
            inputs=torch.stack(images).float(),
            targets=torch.tensor(labels, dtype=torch.long),
        )
