from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class MaskStatistics:
    total_params: int
    active_params: int
    masked_params: int
    weight_sparsity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MaskAwareSparsityState:
    masks: dict[str, torch.Tensor] = field(default_factory=dict)

    def clone(self, *, device: torch.device | str | None = None) -> "MaskAwareSparsityState":
        target_device = torch.device(device) if device is not None else None
        cloned = {
            name: mask.clone().to(target_device or mask.device)
            for name, mask in self.masks.items()
        }
        return MaskAwareSparsityState(masks=cloned)

    def sync(self, named_tensors: dict[str, torch.Tensor]) -> None:
        refreshed: dict[str, torch.Tensor] = {}
        for name, tensor in named_tensors.items():
            existing = self.masks.get(name)
            if existing is not None and tuple(existing.shape) == tuple(tensor.shape):
                refreshed[name] = existing.to(device=tensor.device, dtype=tensor.dtype)
            else:
                refreshed[name] = torch.ones_like(tensor, device=tensor.device)
        self.masks = refreshed

    def apply_(self, named_tensors: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, tensor in named_tensors.items():
                mask = self.masks.get(name)
                if mask is None:
                    continue
                tensor.mul_(mask)

    def multiply_(self, candidate_masks: dict[str, torch.Tensor]) -> None:
        for name, candidate in candidate_masks.items():
            current = self.masks.get(name)
            if current is None:
                self.masks[name] = candidate.clone()
                continue
            self.masks[name] = current * candidate.to(device=current.device, dtype=current.dtype)

    def statistics(self, named_tensors: dict[str, torch.Tensor]) -> MaskStatistics:
        if not named_tensors:
            return MaskStatistics(total_params=0, active_params=0, masked_params=0, weight_sparsity=0.0)
        total_params = sum(tensor.numel() for tensor in named_tensors.values())
        active_params = 0
        for name, tensor in named_tensors.items():
            mask = self.masks.get(name)
            if mask is None:
                active_params += tensor.numel()
            else:
                active_params += int(mask.sum().item())
        masked_params = total_params - active_params
        weight_sparsity = (masked_params / total_params) if total_params else 0.0
        return MaskStatistics(
            total_params=total_params,
            active_params=active_params,
            masked_params=masked_params,
            weight_sparsity=weight_sparsity,
        )

    def named_mask_keys(self) -> list[str]:
        return sorted(self.masks)


def magnitude_mask(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    return (tensor.detach().abs() > threshold).to(tensor.dtype)


def resolve_keep_count(
    total_count: int,
    *,
    amount: int | None = None,
    prune_fraction: float | None = None,
    keep_count: int | None = None,
    min_count: int = 1,
) -> int:
    if keep_count is not None:
        resolved = int(keep_count)
    elif prune_fraction is not None:
        if not 0.0 <= prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0.0, 1.0)")
        resolved = total_count - int(total_count * prune_fraction)
    elif amount is not None:
        resolved = total_count - int(amount)
    else:
        raise ValueError("One of amount, prune_fraction, or keep_count must be provided")
    return max(int(min_count), min(int(total_count), int(resolved)))


def select_topk_indices(importance: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep = torch.topk(importance, k=int(keep_count), largest=True).indices
    keep, _ = torch.sort(keep)
    return keep


def linear_neuron_importance(
    incoming_weight: torch.Tensor,
    outgoing_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = incoming_weight.detach().abs().sum(dim=1)
    if outgoing_weight is not None:
        scores = scores + outgoing_weight.detach().abs().sum(dim=0)
    return scores


def channel_importance(
    *,
    conv_weight: torch.Tensor | None = None,
    batch_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch_norm_weight is not None:
        return batch_norm_weight.detach().abs()
    if conv_weight is None:
        raise ValueError("channel_importance requires conv_weight or batch_norm_weight")
    return conv_weight.detach().abs().sum(dim=(1, 2, 3))


__all__ = [
    "MaskAwareSparsityState",
    "MaskStatistics",
    "channel_importance",
    "linear_neuron_importance",
    "magnitude_mask",
    "resolve_keep_count",
    "select_topk_indices",
]
