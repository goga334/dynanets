from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dynanets.architecture import MLPArchitectureSpec, mlp_spec_from_params
from dynanets.search.base import SearchProposal, SearchSpace


@dataclass(slots=True)
class MLPSearchSpace(SearchSpace):
    input_dim: int
    output_dim: int
    hidden_dim_choices: list[int]
    activation_choices: list[str]
    lr_choices: list[float]

    def sample(self, rng: Any) -> SearchProposal:
        hidden_dim = rng.choice(self.hidden_dim_choices)
        activation = rng.choice(self.activation_choices)
        lr = rng.choice(self.lr_choices)
        spec = mlp_spec_from_params(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        return SearchProposal(
            model_overrides={"hidden_dims": list(spec.hidden_dims), "activation": spec.hidden_activation, "lr": lr},
            metadata={"architecture_spec": spec.to_dict(), "architecture_graph": spec.to_graph(name="mlp_search_candidate").to_dict(), "lr": lr},
        )

    def mutate(self, proposal: SearchProposal, rng: Any) -> SearchProposal:
        current_spec = MLPArchitectureSpec.from_dict(proposal.metadata["architecture_spec"])
        current_lr = float(proposal.metadata["lr"])
        field = rng.choice(["hidden_dim", "activation", "lr"])

        hidden_dims = list(current_spec.hidden_dims)
        activation = current_spec.hidden_activation
        lr = current_lr

        if field == "hidden_dim":
            alternatives = [value for value in self.hidden_dim_choices if value != hidden_dims[0]]
            if alternatives:
                hidden_dims[0] = rng.choice(alternatives)
        elif field == "activation":
            alternatives = [value for value in self.activation_choices if value != activation]
            if alternatives:
                activation = rng.choice(alternatives)
        else:
            alternatives = [value for value in self.lr_choices if value != lr]
            if alternatives:
                lr = rng.choice(alternatives)

        spec = mlp_spec_from_params(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        return SearchProposal(
            model_overrides={"hidden_dims": list(spec.hidden_dims), "activation": spec.hidden_activation, "lr": lr},
            metadata={"architecture_spec": spec.to_dict(), "architecture_graph": spec.to_graph(name="mlp_search_candidate").to_dict(), "lr": lr},
        )
