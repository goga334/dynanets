from dynanets.architecture import (
    MLPArchitectureSpec,
    build_mlp_network,
    extract_architecture_artifacts,
    grow_hidden_layer,
    insert_hidden_layer,
    mlp_spec_from_params,
    remove_hidden_layer,
    replace_hidden_activation,
    shrink_hidden_layer,
)
from dynanets.models.torch_mlp import DynamicMLPClassifier, TorchMLPClassifier
from dynanets.adaptation import AdaptationEvent

import torch



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



def test_mlp_spec_to_graph_and_artifact_extraction() -> None:
    spec = mlp_spec_from_params(input_dim=3, output_dim=2, hidden_dims=[5, 4], activation="relu")
    graph = spec.to_graph(name="demo-graph")
    model = TorchMLPClassifier(input_dim=3, hidden_dims=[5, 4], output_dim=2, activation="relu", lr=0.01)
    spec_dict, graph_dict = extract_architecture_artifacts(model, name="demo-model")

    assert graph.name == "demo-graph"
    assert [node.id for node in graph.nodes] == ["input", "hidden_1", "hidden_2", "output"]
    assert [(edge.source, edge.target) for edge in graph.edges] == [("input", "hidden_1"), ("hidden_1", "hidden_2"), ("hidden_2", "output")]
    assert spec_dict is not None
    assert graph_dict is not None
    assert graph_dict["metadata"]["family"] == "mlp"
    assert graph_dict["nodes"][1]["label"] == "Hidden 1 (5)"



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
    shrunk = shrink_hidden_layer(spec, layer_index=0, amount=1, min_width=2)
    inserted = insert_hidden_layer(spec, layer_index=1, width=5)
    removed = remove_hidden_layer(spec, layer_index=1)
    changed_activation = replace_hidden_activation(spec, activation="tanh")

    assert grown.hidden_dims == [4, 5]
    assert shrunk.hidden_dims == [3, 3]
    assert inserted.hidden_dims == [4, 5, 3]
    assert removed.hidden_dims == [4]
    assert changed_activation.hidden_activation == "tanh"
    assert spec.hidden_dims == [4, 3]



def test_torch_mlp_uses_architecture_spec() -> None:
    model = TorchMLPClassifier(input_dim=2, hidden_dims=[7], output_dim=2, activation="relu", lr=0.01, device="cpu")

    spec = model.architecture_spec()

    assert spec.input_dim == 2
    assert spec.hidden_dims == [7]
    assert spec.output_dim == 2
    assert str(model.device) == "cpu"



def test_dynamic_mlp_updates_architecture_spec_on_growth() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01, device="cpu")
    model.apply_adaptation(AdaptationEvent(event_type="grow_hidden", params={"amount": 2}))

    assert model.architecture_spec().hidden_dims == [6]
    assert model.architecture_state().metadata["hidden_dims"] == [6]
    assert model.architecture_state().metadata["parameter_count"] > 0



def test_dynamic_mlp_updates_architecture_spec_on_pruning() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=6, output_dim=2, activation="relu", lr=0.01, device="cpu")
    model.apply_adaptation(AdaptationEvent(event_type="prune_hidden", params={"amount": 2, "min_width": 4}))

    assert model.architecture_spec().hidden_dims == [4]
    assert model.architecture_state().metadata["hidden_dims"] == [4]



def test_dynamic_mlp_updates_architecture_spec_on_layer_insertion() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01, device="cpu")
    model.apply_adaptation(AdaptationEvent(event_type="insert_hidden_layer", params={"layer_index": 1, "width": 4}))

    assert model.architecture_spec().hidden_dims == [4, 4]
    assert model.architecture_state().metadata["num_hidden_layers"] == 2



def test_dynamic_mlp_updates_architecture_spec_on_layer_removal() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dims=[4, 4], output_dim=2, activation="relu", lr=0.01, device="cpu")
    model.apply_adaptation(AdaptationEvent(event_type="remove_hidden_layer", params={"layer_index": 1}))

    assert model.architecture_spec().hidden_dims == [4]
    assert "remove_hidden_layer" in model.supported_event_types()



def test_dynamic_mlp_applies_weight_masks() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=8, output_dim=2, activation="relu", lr=0.01, device="cpu")
    before_nonzero = model.architecture_state().metadata["nonzero_parameter_count"]
    threshold = model.global_weight_threshold(0.30)
    model.apply_adaptation(
        AdaptationEvent(event_type="apply_weight_mask", params={"threshold": threshold, "target_sparsity": 0.30})
    )
    after_state = model.architecture_state().metadata

    assert after_state["weight_sparsity"] > 0.0
    assert after_state["nonzero_parameter_count"] < before_nonzero
    assert "apply_weight_mask" in model.supported_event_types()



def test_net2deeper_identity_insertion_preserves_outputs() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01, device="cpu")
    inputs = torch.randn(16, 2)
    before = model.forward(inputs).detach().clone()

    model.apply_adaptation(
        AdaptationEvent(
            event_type="insert_hidden_layer",
            params={"layer_index": 1, "width": 4, "init_mode": "identity"},
        )
    )
    after = model.forward(inputs).detach().clone()

    assert model.architecture_spec().hidden_dims == [4, 4]
    assert torch.allclose(before, after, atol=1e-6)
