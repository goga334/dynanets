import pytest
import torch

from dynanets.adaptation import AdaptationEvent
from dynanets.models.torch_mlp import DynamicMLPClassifier
from dynanets.sparsity import MaskAwareSparsityState, linear_neuron_importance, magnitude_mask, resolve_keep_count, select_topk_indices



def test_mask_aware_sparsity_state_applies_and_tracks_masks() -> None:
    state = MaskAwareSparsityState()
    weights = {
        "linear_0.weight": torch.tensor([[1.0, -0.5], [0.1, 0.0]]),
        "linear_1.weight": torch.tensor([[0.2, -0.3]]),
    }

    state.sync(weights)
    state.multiply_({"linear_0.weight": magnitude_mask(weights["linear_0.weight"], threshold=0.4)})
    state.apply_(weights)
    stats = state.statistics(weights)

    assert torch.equal(weights["linear_0.weight"], torch.tensor([[1.0, -0.5], [0.0, 0.0]]))
    assert stats.total_params == 6
    assert stats.active_params == 4
    assert stats.masked_params == 2
    assert abs(stats.weight_sparsity - (2 / 6)) < 1e-9



def test_pruning_helpers_choose_high_importance_units() -> None:
    importance = linear_neuron_importance(
        incoming_weight=torch.tensor([[1.0, 0.0], [5.0, 0.0], [2.0, 0.0], [0.0, 4.0]]),
        outgoing_weight=torch.tensor([[0.1, 0.1, 0.1, 3.0], [0.1, 2.0, 0.1, 0.1]]),
    )
    keep_count = resolve_keep_count(4, prune_fraction=0.5, min_count=1)
    keep_indices = select_topk_indices(importance, keep_count)

    assert keep_count == 2
    assert keep_indices.tolist() == [1, 3]



def test_dynamic_mlp_supports_structured_neuron_pruning() -> None:
    model = DynamicMLPClassifier(input_dim=2, hidden_dim=4, output_dim=2, activation="relu", lr=0.01, device="cpu")
    input_layer, output_layer = model._linear_layers()  # noqa: SLF001

    with torch.no_grad():
        input_layer.weight.copy_(torch.tensor([[1.0, 0.0], [5.0, 0.0], [2.0, 0.0], [0.0, 4.0]]))
        input_layer.bias.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3]))
        output_layer.weight.copy_(torch.tensor([[0.1, 0.1, 0.1, 3.0], [0.1, 2.0, 0.1, 0.1]]))
        output_layer.bias.copy_(torch.tensor([0.5, -0.5]))

    model.apply_adaptation(
        AdaptationEvent(
            event_type="prune_neurons",
            params={"layer_index": 0, "keep_count": 2, "min_width": 2},
        )
    )

    pruned_input, pruned_output = model._linear_layers()  # noqa: SLF001
    state = model.architecture_state().metadata

    assert model.architecture_spec().hidden_dims == [2]
    assert pruned_input.weight.tolist() == [[5.0, 0.0], [0.0, 4.0]]
    assert pruned_input.bias.tolist() == pytest.approx([0.1, 0.3])
    assert pruned_output.weight[0].tolist() == pytest.approx([0.1, 3.0])
    assert pruned_output.weight[1].tolist() == pytest.approx([2.0, 0.1])
    assert pruned_output.bias.tolist() == pytest.approx([0.5, -0.5])
    assert state["hidden_dims"] == [2]
    assert state["mask_state_names"] == ["linear_0.weight", "linear_1.weight"]
    assert "prune_neurons" in model.supported_event_types()




