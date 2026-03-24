from dynanets.architecture import (
    MLPArchitectureSpec,
    build_mlp_network,
    grow_hidden_layer,
    insert_hidden_layer,
    mlp_spec_from_params,
    remove_hidden_layer,
    replace_hidden_activation,
)
from dynanets.models.torch_mlp import DynamicMLPClassifier, TorchMLPClassifier



def test_mlp_spec_roundtrip_and_layers() -> None:
    spec = MLPArchitectureSpec(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 6],
        hidden_activation="tanh",
        metadata={"tag": "demo"},
    )
    spec.validate()

    rebuilt = MLPArchitectureSpec.from_dict(spec.to_dict())
    layers = rebuilt.dense_layers()

    assert rebuilt.hidden_dims == [8, 6]
    assert layers[0].in_features == 4
    assert layers[0].out_features == 8
    assert layers[0].activation == "tanh"
    assert layers[-1].out_features == 2
    assert layers[-1].activation is None



def test_build_mlp_network_from_spec() -> None:
    spec = mlp_spec_from_params(input_dim=3, output_dim=2, hidden_dims=[5, 4], activation="relu")
    network = build_mlp_network(spec)

    linear_layers = [module for module in network if module.__class__.__name__ == "Linear"]
    assert len(linear_layers) == 3
    assert linear_layers[0].in_features == 3
    assert linear_layers[0].out_features == 5
    assert linear_layers[1].out_features == 4
    assert linear_layers[2].out_features == 2



def test_spec_mutation_primitives() -> None:
    spec = mlp_spec_from_params(input_dim=2, output_dim=2, hidden_dims=[4, 3], activation="relu")

    grown = grow_hidden_layer(spec, layer_index=1, amount=2)
    inserted = insert_hidden_layer(spec, layer_index=1, width=5)
    removed = remove_hidden_layer(spec, layer_index=1)
    changed_activation = replace_hidden_activation(spec, activation="tanh")

    assert grown.hidden_dims == [4, 5]
    assert inserted.hidden_dims == [4, 5, 3]
    assert removed.hidden_dims == [4]
    assert changed_activation.hidden_activation == "tanh"
    assert spec.hidden_dims == [4, 3]



def test_torch_mlp_uses_architecture_spec() -> None:
    model = TorchMLPClassifier(input_dim=2, hidden_dims=[7], output_dim=2, activation="relu", lr=0.01)

    spec = model.architecture_spec()

    assert spec.input_dim == 2
    assert spec.hidden_dims == [7]
    assert spec.output_dim == 2



def test_dynamic_mlp_updates_architecture_spec_on_growth() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01)
    model.apply_adaptation({"action": "grow_hidden", "amount": 2})

    assert model.architecture_spec().hidden_dims == [6]
    assert model.architecture_state().metadata["hidden_dims"] == [6]