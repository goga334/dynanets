from __future__ import annotations

import random

import torch
from torch import nn

from dynanets.models.base import ArchitectureState, DynamicNeuralModel, NeuralModel



def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        available = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation '{name}'. Available: {available}") from exc


class TorchMLPClassifier(NeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        lr: float = 1e-2,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.device = torch.device("cpu")
        self._lr = lr

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, output_dim),
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

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


class DynamicMLPClassifier(TorchMLPClassifier, DynamicNeuralModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        lr: float = 1e-2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            lr=lr,
        )
        self._state = ArchitectureState(
            step=0,
            version=0,
            metadata={"hidden_dim": hidden_dim, "output_dim": output_dim},
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
            self._apply_simple_growth(amount)
            return
        if action == "net2wider":
            self._apply_net2wider(amount=amount, seed=int(adaptation.get("seed", 42)))
            return
        raise ValueError(f"Unsupported adaptation action '{action}'")

    def _apply_simple_growth(self, amount: int) -> None:
        old_input = self.network[0]
        old_output = self.network[2]
        new_hidden_dim = self.hidden_dim + amount

        new_input = nn.Linear(self.input_dim, new_hidden_dim).to(self.device)
        new_output = nn.Linear(new_hidden_dim, self.output_dim).to(self.device)

        with torch.no_grad():
            new_input.weight[: self.hidden_dim] = old_input.weight
            new_input.bias[: self.hidden_dim] = old_input.bias
            nn.init.kaiming_uniform_(new_input.weight[self.hidden_dim :], a=5**0.5)
            nn.init.zeros_(new_input.bias[self.hidden_dim :])

            new_output.weight[:, : self.hidden_dim] = old_output.weight
            new_output.bias.copy_(old_output.bias)
            nn.init.kaiming_uniform_(new_output.weight[:, self.hidden_dim :], a=5**0.5)

        self._replace_layers(new_input, new_output, new_hidden_dim)

    def _apply_net2wider(self, amount: int, seed: int) -> None:
        old_input = self.network[0]
        old_output = self.network[2]
        new_hidden_dim = self.hidden_dim + amount
        new_input = nn.Linear(self.input_dim, new_hidden_dim).to(self.device)
        new_output = nn.Linear(new_hidden_dim, self.output_dim).to(self.device)
        rng = random.Random(seed)
        replicated_sources = [rng.randrange(self.hidden_dim) for _ in range(amount)]
        replica_counts = {index: 1 for index in range(self.hidden_dim)}
        for source in replicated_sources:
            replica_counts[source] += 1

        with torch.no_grad():
            for index in range(self.hidden_dim):
                new_input.weight[index] = old_input.weight[index]
                new_input.bias[index] = old_input.bias[index]
                new_output.weight[:, index] = old_output.weight[:, index] / replica_counts[index]

            for offset, source in enumerate(replicated_sources):
                new_index = self.hidden_dim + offset
                new_input.weight[new_index] = old_input.weight[source]
                new_input.bias[new_index] = old_input.bias[source]
                new_output.weight[:, new_index] = old_output.weight[:, source] / replica_counts[source]

            new_output.bias.copy_(old_output.bias)

        self._replace_layers(new_input, new_output, new_hidden_dim)

    def _replace_layers(self, new_input: nn.Linear, new_output: nn.Linear, new_hidden_dim: int) -> None:
        self.hidden_dim = new_hidden_dim
        self.network = nn.Sequential(new_input, _activation(self.activation_name), new_output).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self._lr)
        self._state.version += 1
        self._state.metadata["hidden_dim"] = new_hidden_dim