from __future__ import annotations

import random

import torch
from torch import nn

from dynanets.architecture import MLPArchitectureSpec, build_mlp_network, grow_hidden_layer, mlp_spec_from_params
from dynanets.models.base import ArchitectureState, DynamicNeuralModel, NeuralModel


class TorchMLPClassifier(NeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 2,
        activation: str = "relu",
        lr: float = 1e-2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        self.spec = mlp_spec_from_params(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.device = torch.device("cpu")
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
        self.optimizer.step()

        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        return {"loss": float(loss.item()), "accuracy": float(accuracy)}

    def evaluate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self.forward(batch["inputs"])

    def architecture_spec(self) -> MLPArchitectureSpec:
        return self.spec

    def _rebuild_from_spec(self) -> None:
        self.network = build_mlp_network(self.spec).to(self.device)


class DynamicMLPClassifier(TorchMLPClassifier, DynamicNeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 2,
        activation: str = "relu",
        lr: float = 1e-2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            lr=lr,
            hidden_dims=hidden_dims,
        )
        self._state = ArchitectureState(
            step=0,
            version=0,
            metadata={"hidden_dim": self.hidden_dim, "hidden_dims": self.hidden_dims, "output_dim": output_dim},
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        result = super().training_step(batch)
        self._state.step += 1
        return result

    def architecture_state(self) -> ArchitectureState:
        return self._state

    def apply_adaptation(self, adaptation: dict[str, int]) -> None:
        action = adaptation.get("action")
        amount = int(adaptation.get("amount", 0))
        if amount <= 0:
            return

        if action == "grow_hidden":
            mutated_spec = grow_hidden_layer(self.spec, layer_index=0, amount=amount)
            self._apply_simple_growth(mutated_spec)
            return
        if action == "net2wider":
            mutated_spec = grow_hidden_layer(self.spec, layer_index=0, amount=amount)
            self._apply_net2wider(mutated_spec=mutated_spec, seed=int(adaptation.get("seed", 42)))
            return
        raise ValueError(f"Unsupported adaptation action '{action}'")

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
        self._replace_layers(new_input, new_output)

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
        self._replace_layers(new_input, new_output)

    def _replace_layers(self, new_input: nn.Linear, new_output: nn.Linear) -> None:
        activation_module = build_mlp_network(self.spec)[1]
        self.network = nn.Sequential(new_input, activation_module, new_output).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._state.version += 1
        self._state.metadata["hidden_dim"] = self.hidden_dim
        self._state.metadata["hidden_dims"] = self.hidden_dims