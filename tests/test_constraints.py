from dynanets.architecture import CNNArchitectureSpec, ConvBlockSpec, MLPArchitectureSpec
from dynanets.constraints import ConstraintEvaluator



def test_constraint_evaluator_computes_mlp_costs() -> None:
    spec = MLPArchitectureSpec(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 6],
        hidden_activation="relu",
    )

    summary = ConstraintEvaluator().evaluate(architecture_spec=spec)

    assert summary.architecture_family == "mlp"
    assert summary.parameter_count == 108
    assert summary.nonzero_parameter_count == 108
    assert summary.masked_weight_count == 0
    assert summary.weight_sparsity == 0.0
    assert summary.forward_flop_proxy == 200
    assert summary.activation_elements == 16



def test_constraint_evaluator_uses_metadata_overrides_for_sparse_models() -> None:
    spec = MLPArchitectureSpec(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8],
        hidden_activation="relu",
    )

    summary = ConstraintEvaluator().evaluate(
        architecture_spec=spec,
        metadata={
            "nonzero_parameter_count": 30,
            "masked_weight_count": 20,
            "weight_sparsity": 0.4,
        },
    )

    assert summary.parameter_count == 58
    assert summary.nonzero_parameter_count == 30
    assert summary.masked_weight_count == 20
    assert summary.weight_sparsity == 0.4



def test_constraint_evaluator_computes_cnn_costs() -> None:
    spec = CNNArchitectureSpec(
        input_channels=1,
        input_size=(28, 28),
        num_classes=10,
        blocks=[
            ConvBlockSpec(out_channels=8, kernel_size=3, pool="max"),
            ConvBlockSpec(out_channels=16, kernel_size=3, pool="max"),
        ],
        classifier_hidden_dims=[32],
        activation="relu",
        use_batch_norm=True,
    )

    summary = ConstraintEvaluator().evaluate(architecture_spec=spec)

    assert summary.architecture_family == "cnn"
    assert summary.parameter_count == 2170
    assert summary.nonzero_parameter_count == 2170
    assert summary.masked_weight_count == 0
    assert summary.weight_sparsity == 0.0
    assert summary.forward_flop_proxy > 0
    assert summary.activation_elements == 2410

