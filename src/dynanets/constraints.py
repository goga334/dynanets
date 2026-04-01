from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dynanets.architecture import CNNArchitectureSpec, MLPArchitectureSpec


@dataclass(slots=True)
class ConstraintSummary:
    architecture_family: str
    parameter_count: int
    nonzero_parameter_count: int
    masked_weight_count: int
    weight_sparsity: float
    forward_flop_proxy: int
    activation_elements: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConstraintEvaluator:
    """Compute lightweight model-cost proxies from architecture specs and state metadata."""

    def evaluate(
        self,
        *,
        architecture_spec: MLPArchitectureSpec | CNNArchitectureSpec | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConstraintSummary:
        resolved_metadata = dict(metadata or {})
        normalized_spec = _normalize_spec(architecture_spec)

        if isinstance(normalized_spec, MLPArchitectureSpec):
            summary = self._evaluate_mlp(normalized_spec)
        elif isinstance(normalized_spec, CNNArchitectureSpec):
            summary = self._evaluate_cnn(normalized_spec)
        else:
            summary = ConstraintSummary(
                architecture_family=str(resolved_metadata.get("architecture_family", "unknown")),
                parameter_count=int(resolved_metadata.get("parameter_count", 0)),
                nonzero_parameter_count=int(
                    resolved_metadata.get(
                        "nonzero_parameter_count",
                        resolved_metadata.get("parameter_count", 0),
                    )
                ),
                masked_weight_count=int(resolved_metadata.get("masked_weight_count", 0)),
                weight_sparsity=float(resolved_metadata.get("weight_sparsity", 0.0)),
                forward_flop_proxy=int(resolved_metadata.get("forward_flop_proxy", 0)),
                activation_elements=int(resolved_metadata.get("activation_elements", 0)),
            )

        parameter_count = (
            summary.parameter_count
            if normalized_spec is not None
            else int(resolved_metadata.get("parameter_count", summary.parameter_count))
        )
        nonzero_parameter_count = int(
            resolved_metadata.get("nonzero_parameter_count", summary.nonzero_parameter_count)
        )
        masked_weight_count = int(
            resolved_metadata.get(
                "masked_weight_count",
                max(0, parameter_count - nonzero_parameter_count),
            )
        )
        if "weight_sparsity" in resolved_metadata:
            weight_sparsity = float(resolved_metadata["weight_sparsity"])
        else:
            weight_sparsity = (
                masked_weight_count / parameter_count
                if parameter_count > 0
                else 0.0
            )

        forward_flop_proxy = (
            summary.forward_flop_proxy
            if normalized_spec is not None
            else int(resolved_metadata.get("forward_flop_proxy", summary.forward_flop_proxy))
        )
        activation_elements = (
            summary.activation_elements
            if normalized_spec is not None
            else int(resolved_metadata.get("activation_elements", summary.activation_elements))
        )

        return ConstraintSummary(
            architecture_family=summary.architecture_family,
            parameter_count=parameter_count,
            nonzero_parameter_count=nonzero_parameter_count,
            masked_weight_count=masked_weight_count,
            weight_sparsity=weight_sparsity,
            forward_flop_proxy=forward_flop_proxy,
            activation_elements=activation_elements,
        )

    def _evaluate_mlp(self, spec: MLPArchitectureSpec) -> ConstraintSummary:
        parameter_count = 0
        forward_flop_proxy = 0
        activation_elements = 0

        for layer in spec.dense_layers():
            weights = layer.in_features * layer.out_features
            bias = layer.out_features if layer.bias else 0
            parameter_count += weights + bias
            forward_flop_proxy += (2 * weights) + bias
            activation_elements += layer.out_features

        return ConstraintSummary(
            architecture_family="mlp",
            parameter_count=parameter_count,
            nonzero_parameter_count=parameter_count,
            masked_weight_count=0,
            weight_sparsity=0.0,
            forward_flop_proxy=forward_flop_proxy,
            activation_elements=activation_elements,
        )

    def _evaluate_cnn(self, spec: CNNArchitectureSpec) -> ConstraintSummary:
        parameter_count = 0
        forward_flop_proxy = 0
        activation_elements = 0

        height, width = spec.input_size
        in_channels = spec.input_channels

        for block in spec.blocks:
            kernel_area = block.kernel_size * block.kernel_size
            conv_weights = block.out_channels * in_channels * kernel_area
            conv_bias = block.out_channels
            parameter_count += conv_weights + conv_bias
            forward_flop_proxy += (2 * height * width * conv_weights) + (height * width * conv_bias)

            if spec.use_batch_norm:
                bn_params = 2 * block.out_channels
                parameter_count += bn_params
                forward_flop_proxy += 2 * height * width * block.out_channels

            pooled_height, pooled_width = height, width
            if block.pool is not None:
                pooled_height = max(1, pooled_height // 2)
                pooled_width = max(1, pooled_width // 2)
                forward_flop_proxy += pooled_height * pooled_width * block.out_channels

            activation_elements += pooled_height * pooled_width * block.out_channels
            height, width = pooled_height, pooled_width
            in_channels = block.out_channels

        activation_elements += spec.final_feature_channels

        classifier_dims = [spec.final_feature_channels, *spec.classifier_hidden_dims, spec.num_classes]
        for in_features, out_features in zip(classifier_dims, classifier_dims[1:]):
            weights = in_features * out_features
            bias = out_features
            parameter_count += weights + bias
            forward_flop_proxy += (2 * weights) + bias
            activation_elements += out_features

        return ConstraintSummary(
            architecture_family="cnn",
            parameter_count=parameter_count,
            nonzero_parameter_count=parameter_count,
            masked_weight_count=0,
            weight_sparsity=0.0,
            forward_flop_proxy=forward_flop_proxy,
            activation_elements=activation_elements,
        )


def _normalize_spec(
    architecture_spec: MLPArchitectureSpec | CNNArchitectureSpec | dict[str, Any] | None,
) -> MLPArchitectureSpec | CNNArchitectureSpec | None:
    if architecture_spec is None:
        return None
    if isinstance(architecture_spec, (MLPArchitectureSpec, CNNArchitectureSpec)):
        return architecture_spec
    if not isinstance(architecture_spec, dict):
        return None
    if {"input_dim", "output_dim", "hidden_dims"}.issubset(architecture_spec):
        return MLPArchitectureSpec.from_dict(architecture_spec)
    if {"input_channels", "input_size", "num_classes", "blocks"}.issubset(architecture_spec):
        blocks = architecture_spec.get("blocks", [])
        if blocks and isinstance(blocks[0], dict) and "kind" in blocks[0]:
            return None
        return CNNArchitectureSpec.from_dict(architecture_spec)
    return None


__all__ = ["ConstraintEvaluator", "ConstraintSummary"]

